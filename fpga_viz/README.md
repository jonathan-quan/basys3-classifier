# Recording demo: Basys 3 detector / classifier + live chip map (`fpga_viz/`)

Separate copies of the two verified FPGA designs (`fpga/` and `fpga_det/` are untouched),
each with an **observation-only telemetry block**, plus a 1920×1080 recording view: the
webcam, the FPGA's answer (box or class), the Artix-7's real placement lighting up, and the
activation maps the FPGA computed for that exact frame.

## Record a video (box detector)

1. Plug the Basys 3 into **PROG/J4** and switch it on.
2. **`ProgramDetectorDemo.cmd`** (again after every power-off; configuration is volatile, flash untouched).
3. **`RecordDemo.cmd`**. Close other webcam apps first.

The 4-class classifier version: `ProgramVizFPGA.cmd`, then `RecordClassifierDemo.cmd`.

| Key | Action |
|---|---|
| **R** | start / stop recording → `recordings\fpga_demo_<time>.mp4` (1080p30, H.264) |
| **SPACE** | freeze on the current frame and replay its inference slowly, layer by layer (press again to resume) |
| **A** | detector: attention overlay (where the FPGA's box head looked) on / off |
| **F** | classifier: centre 75 % square / full frame |
| **H** | help overlay (not recorded) |
| **Q / Esc** | quit (a recording in progress is finalised) |

The window can be resized or maximised; the MP4 is always the full 1920×1080 view.
`--fullscreen` starts full-screen. Without a board, `--simulate` runs the laptop integer
reference instead and says so on screen.

## How it stays synchronised

Detector: ~0.78 s per frame (0.49 s to upload 49,152 pixels at 1 Mbaud, 201 ms of compute,
~70 ms of telemetry), so the view runs **1.1 s behind real time**. Classifier: ~0.36 s per
frame, view **0.55 s** behind. Either way every processed frame is shown with its own result:

1. The frame the FPGA classified appears; the chip lights the UART and bank A (image in).
2. The layers sweep across the chip in their **measured** proportions (layer 1 alone is
   37 % of the detector's time, 56 % of the classifier's), replayed 2.5× / 6× slower (labelled).
3. The answer and box are drawn starting on the very frame they were computed from: the view
   is delayed by more than the round trip, so the result is already known when that frame is
   shown. Between FPGA updates (~0.78 s for the detector) the box stays in place, so a moving
   object can lead its box by up to one update. Hold objects still for a clean shot.

The footer still shows the real latency (upload + compute) and the FPGA update rate.
On the detector, the attention glow is drawn on the camera only while an object is detected
(the input thumbnail always shows it).

Each telemetry block carries the frame ID of the inference it describes, and the laptop
re-runs the integer reference (in a separate process) on the exact bytes sent and checks outputs **and**
telemetry bit for bit; the running count is in the footer (“bit-exact vs reference n/n”).

## What is real on screen

| Element | Source |
|---|---|
| Chip map | Vivado's routed placement of *this* bitstream (`build/placement/*.csv`, exported by the build) |
| Glow per group | which hardware is busy in each phase (RTL behaviour); bank brightness ∝ that layer's mean activation from telemetry |
| Activation maps + channel strips | read back from the FPGA every frame (telemetry) |
| Attention glow (detector) | the 16×16 softmax weights the FPGA's box head used for the box centre (telemetry) |
| Time per layer | cycle counts latched by the FPGA |
| Class name, FPGA score, logits | FPGA result packet (unchanged protocol) |
| Bar percentages | softmax of the FPGA's 4 logits, computed on the laptop for display |
| LED8–15 on the board | bar graph of the FPGA score (8 LEDs = 100 %) |

## Hardware changes

Detector (`fpga_viz/det_rtl/`): `detector_core.sv` gains the same observation taps plus the
softmax weight of each heat-map cell; `detector_telemetry.sv` keeps 11 layer stamps, 96 pooled
values, 704 channel sums, 1,664 heat bins and 256 attention weights (6,796 bytes, `det_telemetry.py`);
`basys3_detector_viz.sv` adds request `0x20` → `0xA0` (6,800-byte payload). Verified by
`scripts/simulate_det.ps1` (original core testbench on 5 frames + UART/telemetry testbench) and
`scripts/build_det.tcl` (same gates as `fpga_det`).

Classifier (`fpga_viz/rtl/`):

* `classifier_core.sv` — identical arithmetic and cycle count; adds tap outputs that mirror
  each activation write (layer, channel, pixel) and each pooled value.
* `inference_telemetry.sv` — new: per-layer finish cycle, 256 channel sums, 864 heat-map
  bins (12×12 / 6×6 per layer, read-modify-write in block RAM), 48 pooled values.
* `basys3_classifier_viz.sv` — original protocol byte-identical; new request `0x20` →
  response `0xA0` (2,840-byte payload); LED8–15 score bar.

Telemetry layout: `telemetry.py`. Verification: `scripts/run_simulations.cmd` (original
core testbench + `sim/tb_viz_uart.sv`, which checks every original packet byte-for-byte
and the telemetry against the integer reference), `scripts/run_build.cmd` (same gates as
the original build), `python -m fpga_viz.selftest` on the board.
