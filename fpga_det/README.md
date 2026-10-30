# Basys 3 FPGA box detector

A single-object detector (class + bounding box) running entirely on a Digilent Basys 3
(Artix-7 XC7A35T) at 100 MHz. The laptop captures webcam frames, letterboxes them to
128x128 and sends them over the USB UART; the FPGA runs the whole network and returns
the class, presence and box. No laptop neural inference, no HLS, no soft CPU.

Model: `detection/runs/webcam_ft_v1` (31,977 parameters, 4.78M MACs, classes Rubik's cube /
pen / book / computer mouse), quantized to INT8 without retraining.

## Verified results

| | |
|---|---|
| INT8 vs FP32, webcam validation | mAP50 0.741 vs 0.740, class accuracy 98.8% vs 98.8%, mean IoU 0.657 vs 0.656 |
| INT8 vs FP32, all validation (incl. public photos) | class accuracy 71.9% vs 73.1%, mean IoU 0.491 vs 0.490 |
| RTL simulation | 5 webcam frames bit-exact vs the integer reference: every byte of all 11 layers, pooled features, heat map, attention, all outputs |
| UART simulation | HELLO, bad-CRC rejection, bad image config, full 128x128 image -> exact 24-byte result |
| Implementation | 100 MHz, WNS +0.213 ns, WHS +0.046 ns, 0 DRC errors; 33.5/50 BRAM tiles, 7,034 LUTs, 2,167 FFs, 7 DSPs |
| On the board | 5/5 frames bit-exact (`hardware_test.json`), bad-CRC rejected, partial-packet recovery |
| Latency | 20,118,456 cycles = 201 ms on the FPGA; ~706 ms round trip including the 49 KB transfer at 1 Mbaud |

Known limitations: four known classes only and one object per frame; pen boxes are poorly sized
(webcam validation AP50 0.05); people are not in the negative training data, so a person filling
the frame can be reported as an object. The webcam validation clips come from one setup (red table).
The test split has not been evaluated for this run.

## Design

- **Backbone**: 11 convolutions (3x3 stem, depthwise-separable stages, 16/32/64/96 channels) on a
  shared scalar MAC engine driven by a generated layer table (`export/layers.svh`). Per-channel INT8
  weights, INT32 accumulation, 23-bit multiplier + shift requantisation with round-half-away-from-zero.
- **Memory**: two 64 KiB ping-pong activation banks in block RAM. Per-layer offsets keep the
  stride-8 tap map (64x16x16) intact for the head while later layers run.
- **Head**: global average pool -> class (4) and presence logits; heat map = 1x1 conv on the tap +
  nearest-upsampled 1x1 conv on the final map; softmax via a 513-entry exp table; soft-argmax box
  centre and attended tap features via a serial divider; width/height from pooled + attended features.
- The host applies only the final softmax/sigmoid to the 9 returned integers.

## Files

- `quantize.py` - INT8 export, bit-exact NumPy reference (`IntegerDetector`), validation comparison
- `rtl/detector_core.sv`, `rtl/basys3_detector.sv` - core and UART top (UART and pins shared with `fpga/`)
- `sim/` - core and UART testbenches; `scripts/simulate_*.ps1`, `scripts/build.tcl`, `scripts/program.tcl`
- `client.py` - protocol client and hardware self-test; `webcam.py` - live demo
- `build/basys3_detector.bit` - the verified bitstream

## Run

```powershell
.\ProgramDetector.cmd      # volatile JTAG load (~10 s); SPI flash untouched
.\FPGADetector.cmd         # live webcam demo; Q quits, T shows low-score guesses
.\.venv\Scripts\python.exe -B -m fpga_det.client --self-test
```

Rebuild: `fpga_det\scripts\simulate_core.ps1`, `simulate_uart.ps1`, then
`vivado -mode batch -source fpga_det/scripts/build.tcl`. The build refuses to run on RTL that has not
passed both simulations, and refuses a bitstream that misses timing or has DRC errors.
On some shells Vivado's launcher needs `PROCESSOR_ARCHITECTURE=AMD64` set.
