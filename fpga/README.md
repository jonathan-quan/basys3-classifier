# Basys 3 classifier hardware

**Status, October 29, 2026: built, programmed and verified on the board.** The
bitstream meets 100 MHz timing, was loaded into the connected Basys 3 by volatile
JTAG, and reproduced the integer software reference bit-for-bit on all 490 original
test photos. The live laptop-webcam demo was exercised by the owner.

This is a **classification-only** FPGA implementation of the preserved
6,084-parameter `best_model.pth` (INT8 export, model ID `d70fd5dfdfad0b31`). It
accepts a 96×96 RGB image over USB UART and executes the quantized CNN, pooling,
four-class linear head, argmax, and fixed-point softmax score on the Artix-7.
There is no external DDR or soft CPU and no laptop-side neural inference in the
hardware webcam application. The original floating-point checkpoints are unchanged.
The separate box detector (see `detection/`) is **not** on the FPGA.

## Verified results

| Stage | Result | Evidence |
|---|---|---|
| Core RTL simulation (XSim) | Every byte of all 9 layers, 48 pooled values, 4 logits and confidence match the integer reference on 3 real validation photos; 3,674,813 cycles/image | `simulation_core.log`, `build/simulation_pass.json` |
| UART integration simulation (XSim) | HELLO, bad-CRC rejection, bad configuration rejection, full 27,652-byte image packet and exact 34-byte result packet | `simulation_uart.log`, `build/uart_simulation_pass.json` |
| Routed timing, 100 MHz | Setup WNS **+0.343 ns**, hold WHS **+0.019 ns**, 0 failing of 4,163 endpoints, no unconstrained internal endpoints | `build/timing.rpt` |
| DRC / routing | 0 errors, 0 critical warnings (3 DSP-pipelining advisories); 3,617/3,617 nets routed | `build/drc.rpt`, `build/route_status.rpt` |
| Routed utilization (XC7A35T) | 2,636 LUTs (12.7%), 1,446 FFs (3.5%), 17.5/50 BRAM tiles (16 RAMB36 activation banks + 3 RAMB18 ROMs), 7 DSPs | `build/utilization.rpt` |
| Bitstream | `build/basys3_classifier.bit`, 615,078 bytes, SHA-256 `0da6fe2e964e2039696370dd2c37065866fe3ae2f11a43a33adb68b54942f154` | `build/build_status.json` |
| Programming | Volatile JTAG, target `localhost:3121/xilinx_tcf/Digilent/210183BF7BDEA`, device `xc7a35t_0`, DONE=1; SPI flash untouched | `program.log`, `build/program_status.json` |
| Board self-test | 3 real photos: logits, class and confidence exactly equal to the integer reference; bad CRC rejected; truncated packet recovered | `hardware_test.json`, `selftest.log` |
| **Board test set (490 photos)** | **75.92% (372/490)**, every logit and score bit-exact with the integer reference. Per class: cube 79.4%, pen 64.9%, book 65.9%, mouse 92.9% | `hardware_benchmark.json`, `benchmark.log` |
| Board latency | **36.75 ms** FPGA compute (3,674,813 cycles @ 100 MHz); median **323 ms** UART round trip at 1 Mbaud | `hardware_benchmark.json` |
| Live webcam demo | ~3.1 FPGA updates/s; owner-tested on real objects | (manual) |

For reference: FP32 test accuracy 76.53% (375/490); integer software reference
75.92% (372/490). The 0.61-point gap is quantization, not hardware. Public-photo
accuracy does not establish accuracy on arbitrary physical objects: thin or small
objects must fill much of the centre square, and colour-driven mistakes occur
(e.g. a white stylus on the red surface was reported as a Rubik's Cube).

## Use the board

Connect the powered Basys 3 through its **PROG/J4 Micro-USB** port. The JTAG
target is serial `210183BF7BDEA`; its UART was COM6 (the client auto-detects the
FTDI 0403:6010 port).

1. [ProgramFPGA.cmd](C:/Users/ndaqu/Documents/Basys3Classifier/ProgramFPGA.cmd) loads the bitstream (~30 s). Needed after every power loss: configuration is volatile.
2. [FPGARecognition.cmd](C:/Users/ndaqu/Documents/Basys3Classifier/FPGARecognition.cmd) starts the webcam preview with real FPGA classification.

Close other webcam applications first. The preview shows class name/score, FPGA
compute time, UART round-trip time, preview FPS and hardware update FPS. **F**
switches central 75% square versus full frame; **Q/Escape** quits. There are no
bounding boxes. Scores are uncalibrated; the model always picks one of four classes.

## Design changes made to reach a working build

The first implementation failed placement (DRC UTLZ-1): both 32K×8 activation
banks were mapped to LUTRAM (18,444 memory LUTs, 192% of the device). Causes and fixes:

1. **Memory inference.** Bank A had three addresses in one process (image-load write,
   inference write, read) and both banks fed one shared read register, which no
   7-series BRAM can implement. Each bank is now a simple-dual-port template: one
   write port (load/inference writes multiplexed with the original priority), one
   registered read, bank select registered alongside the read. Result: 8 RAMB36 per bank.
2. **Timing (WNS was −4.61 ns, 225 endpoints).** Without changing arithmetic or the
   cycle count: requantization split into bias-add and window/saturate stages
   (bit-exact form `(p + (p<0 ? h−1 : h)) >>> s`, checked exhaustively incl. halfway
   cases); layer dimensions loaded into registers once per layer; tap and output
   address arithmetic split over two pipeline stages; MAC multiply and accumulate
   separated; GAP `/36` replaced by an exhaustively verified reciprocal multiply,
   pipelined. The RAM write is one cycle later; `layer_done` (testbench-only) follows it.
3. **Latent X bug.** The address `always @*` blocks were not evaluated at time 0 in
   XSim, so the first tap of an inference saw X; the original code masked this by
   chance. Both blocks are now `always_comb`.

Previous RTL/scripts are archived in `archive/pre_bram_fix/`.

## Arithmetic and architecture

INT8 weights are symmetric per output channel; features are signed INT8 with
per-layer scale and ReLU clamp 0…127. Input is `pixel−128`, scale 1/128.
Bias/accumulation is signed INT32. Requantization uses a positive ≤23-bit integer
multiplier, signed 64-bit product, nearest rounding with halfway cases away from
zero, and saturation. Padding is integer zero. GAP is `(sum+18)//36`. FC logits
are signed INT16 at scale 1/256. FPGA softmax uses a fixed exponential table and a
serial integer divider, returning a UINT16 score (`score/65535`).

The design uses one shared scalar MAC (≈4.2 cycles per MAC including control), two
ping-pong CHW activation banks, and weight/parameter/exp ROMs in BRAM. UART at
1 Mbaud dominates the update interval (~277 ms per image versus 36.75 ms compute).

## Verification and rebuilding

Launchers in `fpga/scripts/` log to files and write a status marker in `fpga/logs/`:

| Launcher | Does | Success marker |
|---|---|---|
| `run_simulations.cmd` | both XSim testbenches | explicit success text in each log (XSim can exit 0 after a fatal) |
| `run_build.cmd` | synthesis → BRAM-mapping gate → place/route → setup+hold+DRC gates → bitstream | `BITSTREAM_READY` + `build/build_status.json` |
| `run_build_resume.cmd` | same, reusing `build/post_synth.dcp` only if newer than every source | same |
| `run_program.cmd` | volatile JTAG programming of serial `210183BF7BDEA` | `PROGRAMMING_COMPLETE` + `build/program_status.json` |
| `run_selftest.cmd` / `run_benchmark.cmd` | board self-test / 490-image board benchmark | exit code 0 + JSON report |

The build refuses stale simulations, LUTRAM-mapped activation banks, negative
setup or hold slack, unrouted nets and DRC errors. The project-root
`StartJobRunner.cmd` runs these (and detector jobs) from a fixed allow-list.

UART protocol: magic `B3 5D A7 C1`, version 1, frame IDs, bounded lengths,
CRC-32/ISO-HDLC. HELLO checks the compiled model identity before image transfer.
One request outstanding; truncated packets time out after 0.5 s; invalid checksums
cannot start inference.

LED0–3 last class (cube/pen/book/mouse), LED4 busy, LED5 latched communication
error (the self-test deliberately sets it with a bad CRC; centre button clears),
LED6 packet reception, LED7 heartbeat.

## Not done / limitations

- The bitstream is volatile; persistent SPI-flash programming was intentionally not done.
- No background/unknown rejection in the FPGA classifier.
- Accuracy on the owner's physical objects was checked only informally (live demo), not measured.
- The improved box detector (`detection/runs/webcam_ft_v1`) runs only in laptop software; an FPGA version would need a new INT8 export, RTL for its wider 128×128 network and soft-argmax box head, a new protocol, and a new build.
