# Basys 3 neural object detector — feasibility report

Assessed: 2026-10-28. Scope: Stage 1 only. No detector training, quantization, RTL, synthesis, camera capture, or hardware testing was performed.

## 1. Recommendation and limits

**A small, single-object neural detector is analytically feasible on the Basys 3. Start by extending the existing trained CNN, keeping 96×96 RGB.** Branch before global average pooling for localization. Retain the existing classification branch, add an objectness branch, and use a small spatial compression plus fully connected box branch. This is Approach A below: **7,681 inference parameters, 886,704 MACs/image, 45 KiB peak simultaneous INT8 feature storage, and a concrete 19-of-50 BRAM36 allocation**. Allow additional implementation margin; neither fit nor clock speed has been demonstrated by synthesis.

All **6,084 existing trained parameters can initialize this design directly**, including the four-class head. The 1,597 new parameters require box-annotated training and negative examples. Initializing from the classifier does not make it a detector, and its 76.53% classification accuracy does not predict detection accuracy.

The likely throughput constraint is UART rather than arithmetic. At 1 Mbaud, a 96×96 RGB input packet takes **276.70 ms** on the wire. Allowing the proposed inference schedule and response, the calculated rate is about **3.49–3.55 detection updates/s before host/USB overhead**. At 115,200 baud it is about 0.41 updates/s. At 3 Mbaud, roughly 9.8–10.3 updates/s could be possible if the link and accelerator work as assumed. These are estimates, not measurements. A smooth laptop preview can run independently, but the neural bounding box only updates when a new FPGA result arrives.

Proceed first with annotated data and a software detector. Defer multiple-object detection. Do not build a large YOLO network, implement an accelerator, or assume an accuracy figure before the software localization experiment succeeds.

This architecture uses the **laptop webcam and the board's USB UART**. It needs no external camera module or VGA connection. A data-capable Micro-USB cable and the Basys 3 are sufficient for the planned laptop/FPGA link. The existing `.pth` remains a laptop checkpoint; eventual board execution requires an integer export and an FPGA bitstream.

## 2. What was inspected and verified

Project: `C:\Users\ndaqu\Documents\Basys3Classifier`.

Reviewed `README.md`, `RESULTS.md`, `model_profiles.json`, `model_selection.json`, the selected checkpoint, selected test metrics, `classifier/model.py`, training/evaluation/data preparation/download code, and `classifier/webcam.py`. Also reviewed the earlier `C:\Users\ndaqu\Documents\Basys3-feasibility-report.md`, dataset manifests, attribution and source metadata, and the cached SAS archive/README.

Read-only checkpoint loading with `torch.load(..., weights_only=True)` and strict loading into `TinyClassifier("context")` succeeded. CPU dummy forward passes at 64, 96, and 128 pixels verified the existing layer shapes and MAC counts. Candidate-head figures below were calculated from their specified dimensions; candidate detector models were not implemented or trained.

| Existing artifact | Verified state |
|---|---|
| `best_model.pth` | FP32, architecture `context`, 6,084 inference parameters; 31,605-byte serialized file |
| Training state | Epoch 46 selected; training BN already folded into convolution weights/biases |
| Validation result | 73.48% on 460 images |
| Test result | 76.53%, 375/490; classification only |
| Original-size reference | `baseline_3252.pth`, 3,252 parameters, 67.96% classification test accuracy |
| INT8 detector/classifier results | None |
| Existing hardware accelerator | None found in this project: no Verilog/SystemVerilog, weight-memory export, or `.bit` |
| Webcam application | Laptop classification inside a fixed central ROI; its rectangle is not a detected object box |
| Physical-object benchmark | Not established; previous still-image application check even produced one confidently wrong book prediction |

The earlier eight-lane accelerator is a **proposal**, not reusable completed RTL. Its architectural ideas can carry over; implementation work remains. Vivado was not found on PATH or in the common installation directories checked; that limited check does not prove it is absent elsewhere.

Preservation identifiers:

- Checkpoint SHA-256: `98c66c324f34d66a1c4d876ec3fcf05f6ed1f57f0df1a7c6733870fbb3e4cf61`.
- Active classifier manifest SHA-256: `571c2377de21eace6d06aec7ee99fecf023587e3cf0b30a5020c795d48042cce`.
- A content/path digest was taken over 5,831 existing project files, excluding `.venv`, `__pycache__`, `.git`, and this new report. Digest: `b3218708979a3258bddcacb84e004c8c2e39c9253863d58cf86bd9a9d4ebe145`. Rechecking it after report creation can detect changes to that original set.

## 3. Exact existing network and transfer points

Shapes exclude batch dimension. Convolutions have bias and are followed by ReLU. DW means depthwise convolution; PW means pointwise convolution. RF is the theoretical receptive field of one spatial feature, not a measurement of what the trained network uses effectively.

| Checkpoint layer | Operation | Output at 96 RGB | Weights | Biases | MACs | RF / input stride |
|---|---|---|---:|---:|---:|---|
| `features.0` | Conv 3→8, 3×3, stride 2, pad 1 | 8×48×48 | 216 | 8 | 497,664 | 3 / 2 |
| `features.2` | DW 8, 3×3, stride 2, pad 1 | 8×24×24 | 72 | 8 | 41,472 | 7 / 4 |
| `features.4` | PW 8→16 | 16×24×24 | 128 | 16 | 73,728 | 7 / 4 |
| `features.6` | DW 16, 3×3, stride 2, pad 1 | 16×12×12 | 144 | 16 | 20,736 | 15 / 8 |
| `features.8` | PW 16→32 | 32×12×12 | 512 | 32 | 73,728 | 15 / 8 |
| `features.10` | DW 32, 3×3, stride 2, pad 1 | 32×6×6 | 288 | 32 | 10,368 | 31 / 16 |
| `features.12` | PW 32→48 | 48×6×6 | 1,536 | 48 | 55,296 | 31 / 16 |
| `features.14` | DW 48, 3×3, stride 1, pad 1 | 48×6×6 | 432 | 48 | 15,552 | 63 / 16 |
| `features.16` | PW 48→48 | 48×6×6 | 2,304 | 48 | 82,944 | 63 / 16 |
| `pool`, `fc` | Global average → Linear 48→4 | 4 | 192 | 4 | 192 | Global aggregation |
| **Total** | | | **5,824** | **260** | **871,680** | |

The convolution backbone is 5,888 parameters: 5,632 weights plus 256 biases. Its nine weight tensors and nine bias tensors transfer directly. The existing `fc.weight` (4×48) and `fc.bias` (4) also transfer directly **when applied to the same pooled 48-channel representation and class order**. The checkpoint contains 20 such tensors in total. BN statistics do not need exporting because BN is already folded.

For Approach A, branch from the 48-channel feature map **after `features.17` ReLU, before `pool`**. Keep the full feature map for box prediction. GAP deliberately removes most explicit position information; a four-coordinate linear head on GAP alone is a poor first localization design. Padding can leak some position information, but that is not a reliable localization strategy.

The current RF is 63×63 and its final stride is 16. A feature cannot individually see all of a large 96-pixel-wide object. However, a flattened spatial head combines all 6×6 positions, whose theoretical union spans 143 pixels, covering the input. This makes single-object regression plausible without adding another context block. It does **not** recover detail lost during stride-2 downsampling. Thin pens, tiny objects, and precise boundaries remain risks. At 64/128 pixels, the spatial unions are 111/175 pixels respectively, still covering those inputs.

If localization is poor, investigate a stride-8 feature tap or retaining more spatial resolution before enlarging the whole network. Those changes need a fresh memory/compute profile; do not silently modify strides or input resolution. The classifier weights still have useful shapes, but changed feature statistics require fine-tuning and evaluation.

## 4. Three fully specified detector candidates

These are proposed inference graphs, with ordinary bias-bearing convolution/linear layers and no inference BN. At each input size, choose a separate trained/exported model. In particular, the flattened box FC in A/B changes size with resolution; its new weights cannot be transferred unchanged between resolutions.

### A — Reuse the trained CNN; one global object and one box (recommended)

```text
RGB → existing trained features → F: 48×S×S, S=N/16
                                 ├─ GAP → existing Linear(48,4) → class logits
                                 ├─ GAP → new Linear(48,1) → presence logit
                                 └─ new Conv1×1(48,8) → ReLU
                                      → flatten(8S²) → new Linear(8S²,4) → box logits
```

The two GAP branches share one pooled vector. Apply sigmoid to the four box logits to obtain normalized center X/Y and width/height. Convert to clipped corners on FPGA. A joint detection score is presence score × maximum four-class softmax score. Both sigmoid/softmax and box decoding must eventually run with fixed-point arithmetic on FPGA; Python only scales the returned coordinates for display.

At 96 pixels, new layers are:

| New component | Weight count | Bias count | Parameters | MACs/image |
|---|---:|---:|---:|---:|
| Objectness Linear 48→1 | 48 | 1 | 49 | 48 |
| Spatial compression Conv 48→8 | 384 | 8 | 392 | 13,824 |
| Box FC 288→4 | 1,152 | 4 | 1,156 | 1,152 |
| **Additional** | **1,584** | **13** | **1,597** | **15,024** |

Total: **7,681 parameters**, 7,408 INT8 weight bytes plus 1,092 INT32 bias bytes = **8,500 bytes** before quantization metadata and physical weight padding. FP32 parameter payload would be 30,724 bytes; a future `.pth` file would also contain serialization/metadata. This is only **1.72% more MACs** than the existing classifier. One global box avoids anchor assignment, box competition, and NMS. Empty frames are learned through objectness, not forced into one of four classes.

Initial head warm-up can freeze the transferred layers, followed by low-learning-rate fine-tuning of the shared backbone and class head. This is a proposed training sequence, not a performed experiment. The folded model can be fine-tuned directly; reintroducing training BN would require deliberate layer mapping and verified folding rather than assuming identical state keys.

### B — Purpose-built stride-8 tiny single-object detector

For a concrete comparison, define a fresh, smaller-channel backbone:

```text
Conv3×3 s2 3→8
DW3×3 s2 8  → PW 8→16
DW3×3 s2 16 → PW 16→24
DW3×3 s1 24 → PW 24→32
DW3×3 s1 32 → PW 32→32
F: 32×(N/8)×(N/8)
  ├─ GAP → Linear(32,4)
  ├─ GAP → Linear(32,1)
  └─ Conv1×1(32,8) → ReLU → flatten → Linear(8(N/8)²,4)
```

All DW layers use pad 1; each convolution has ReLU. At 96, this graph has **8,473 parameters** and **1,061,152 MACs**. Its feature map is 12×12, with RF 47 and stride 8. Spatial sampling is finer than A, but features see less context per cell and its flattened box head grows faster. There is no evidence it will be more accurate. Its backbone has 3,240 weights and 192 biases; all heads together add 5,024 weights and 17 biases.

This is the explicit purpose-built/from-scratch alternative, **not authorized training work for this stage**. Some early layers happen to match existing shapes, but initializing those from the checkpoint would constitute a separate transfer experiment rather than a pure scratch comparator. Train B only if A's measured localization warrants the extra comparison.

### C — Reuse the backbone; simplified grid detector

```text
Existing features → F: 48×S×S
  → new DW3×3 s1 pad1, 48 channels → ReLU
  → new Conv1×1 48→9, no final ReLU
  → S×S predictions: objectness, four class logits, four box logits
```

Remove the global FC. Backbone 5,888 parameters + new DW 480 + PW 441 = **6,809 parameters**. At 96, work is **902,592 MACs**, with a 6×6 output grid, RF 95, and stride 16. Initialize compatible backbone tensors directly; the changed local detection representation does not justify treating the old global classifier as an already trained detection head.

Use one responsible cell per target center. Decode `cx=(column+sigmoid(tx))/S`, `cy=(row+sigmoid(ty))/S`, with normalized `w=sigmoid(tw)`, `h=sigmoid(th)`. For version 1, FPGA selects the highest joint score and returns at most one box, so NMS is unnecessary. A full multiple-box version would need duplicate suppression, additional packet output, and a defined matching/training policy. The first YOLO paper supports the general grid-regression concept; **this custom graph is not YOLO, pretrained YOLO, or evidence of YOLO-level performance**. [Original YOLO paper](https://arxiv.org/abs/1506.02640).

C has fewer weights than A and global context is improved by its extra DW layer, but cell assignment, negative-cell imbalance, boundary transitions, and local confidence ranking introduce training/verification complexity. Prefer A for the one-prominent-object requirement unless validation shows C offers a meaningful benefit.

## 5. Resolution and architecture comparison

All values below are **calculated estimates**, except the previously measured classifier results in section 2. None are Vivado utilization or measured FPGA latency. A MAC means one multiply-accumulate; counts exclude pooling additions, bias additions, requantization, nonlinear LUTs, and output selection, whose timing is considered separately.

`W8` is unpadded INT8 weight storage, excluding INT32 biases/metadata. `Peak` is simultaneous source+destination feature payload, excluding weights and tiny register scratch. BRAM values are the concrete byte/weight-word packing plan described in section 6; the range adds up to six blocks as an implementation contingency, not a statistical confidence interval. All rows share a provisional **10–12/90 DSP** allocation. LUT requirements cannot be derived from these graphs alone; see section 7.

| Design | RGB input | Parameters | W8 bytes | Peak INT8 features | BRAM36 plan / contingency | MACs/image | Estimated inference ms | Input UART ms at 1 Mbaud |
|---|---|---:|---:|---:|---|---:|---:|---:|
| A reuse | 64×64 | 7,041 | 6,768 | 20 KiB | 11 / 11–17 | 394,224 | 2.10–4.19 | 123.10 |
| **A reuse** | **96×96** | **7,681** | **7,408** | **45 KiB** | **19 / 19–25** | **886,704** | **4.71–9.43** | **276.70** |
| A reuse | 128×128 | 8,577 | 8,304 | 80 KiB | 27 / 27–33 | 1,576,176 | 8.38–16.76 | 491.74 |
| B purpose-built | 64×64 | 5,913 | 5,704 | 20 KiB | 11 / 11–17 | 471,712 | 2.78–5.55 | 123.10 |
| B purpose-built | 96×96 | 8,473 | 8,264 | 45 KiB | 20 / 20–26 | 1,061,152 | 6.25–12.49 | 276.70 |
| B purpose-built | 128×128 | 12,057 | 11,848 | 80 KiB | 29 / 29–35 | 1,886,368 | 11.10–22.20 | 491.74 |
| C grid | 64×64 | 6,809 | 6,496 | 20 KiB | 11 / 11–17 | 401,152 | 2.25–4.49 | 123.10 |
| C grid | 96×96 | 6,809 | 6,496 | 45 KiB | 18 / 18–24 | 902,592 | 5.05–10.10 | 276.70 |
| C grid | 128×128 | 6,809 | 6,496 | 80 KiB | 26 / 26–32 | 1,604,608 | 8.98–17.96 | 491.74 |

| Design | Detection functionality | Implementation difficulty | Main technical risk | Provisional LUT gate, not utilization prediction |
|---|---|---|---|---|
| A | One box, four classes, presence, joint score | Lowest of these three; reuse all trained parameters | Coarse features and regression quality, especially pens | ≤10,000 LUTs |
| B | Same one-object outputs, finer stride-8 features | Similar arithmetic, more training/selection work | No proven accuracy benefit; reduced per-cell context | ≤10,000 LUTs |
| C | Grid predictions, top-one box initially; possible later multi-object extension | Higher assignment/decoding/selection complexity | Cell duplicates, missing context, score imbalance | ≤12,000 LUTs |

Resolution choice is **provisionally 96**, preserving the existing feature extractor's training distribution. No detection quality has been measured at any size. 64 may lose pen width and small-object detail; 128 increases communication by 78% over 96, with no demonstrated accuracy gain. Evaluate 64/128 on the same validation cohorts only after the 96 detector exists. Choose using detection metrics and measured latency, not unused BRAM or classification accuracy alone.

Calculation formulas, for reproducibility:

- Conv weights: `Cout × (Cin/groups) × k²`; biases: `Cout`.
- Conv MACs: `Hout × Wout × Cout × (Cin/groups) × k²`.
- Linear weights/MACs: `input_features × output_features`; add one bias per output.
- A parameter total: `6529 + 32(N/16)²`; A MACs: `871488(N/96)² + 384(N/16)² + 32(N/16)² + 240`.
- B parameter total: `3865 + 32(N/8)²`; its MACs at these sizes scale as `(1060992)(N/96)² + 160`.
- C parameters: `6809`; C MACs: `902592(N/96)²`.
- A/B bias counts remain 273/209 respectively across resolutions; C has 313 biases. Their INT32 payloads are 1,092/836/1,252 bytes.
- For all candidates at these even dimensions, first-layer input and output dominate feature payload: `3N² + 8(N/2)² = 5N²` bytes.

## 6. BRAM, buffering, and bandwidth

The board has 50 BRAM36 equivalents, 90 DSP48E1s, and 20,800 LUTs at the stated Artix-7 size; 100 MHz is the board oscillator frequency, not proof this design meets timing. [Digilent Basys 3 manual](https://digilent.com/reference/_media/basys3:basys3_rm.pdf).

For the proposed layout, use 4,096 useful byte addresses per BRAM36 for byte memories, and 512 words of 64 useful weight bits in a 72-bit-wide configuration. Parity is unused. Nominal 1,800 Kbits therefore should not be treated as 225 KiB of freely interchangeable byte RAM; this mapping offers about 200 KiB of useful bytes. Port widths and inference into the intended RAM primitives must be checked. [AMD UG473](https://docs.amd.com/api/khub/documents/9gZGbqBxtlKXxBfkBt~lAg/content).

### Concrete A96 allocation

| Region | Required payload or capacity | BRAM36 blocks |
|---|---|---:|
| Activation A | Up to 27,648 B input; allocate 28 KiB | 7 |
| Activation B | Up to 18,432 B first-conv output; allocate 20 KiB | 5 |
| Weight ROM, padded for eight-lane access | 9,088 B; allocate 12 KiB | 3 |
| Biases, requantization multipliers/shifts, descriptors, nonlinear tables | Reserve 8 KiB | 2 |
| RX/TX staging and UART FIFO | Reserve 4 KiB | 1 |
| Scratch / implementation reserve | Reserve 4 KiB | 1 |
| **Total** | **76 KiB allocated useful capacity** | **19/50 = 38%** |

This already includes ordinary block rounding. The additional contingency up to 25 blocks is for implementation choices such as bank duplication or debug storage. It is not a reason to allocate everything. An ILA can consume further BRAM; budget it explicitly.

Weight padding matters: all backbone channel groups pack efficiently, but four-output FCs use only four of eight lanes. A96 packs 5,632 backbone bytes + 384 class-FC bytes + 384 presence-FC bytes + 384 compression-conv bytes + 2,304 box-FC bytes = **9,088 bytes**, larger than the mathematical 7,408 weights. The equivalent packed totals are A64/A128: 7,808/10,880; B64/B96/B128: 8,104/13,224/20,392; C at every resolution: 6,832 bytes. The proposed weight ROM uses three blocks for A96 rather than assuming its raw payload fits in two.

Schedule at 96:

1. Receive and validate one image into A. INT8 input conversion can occur during reception.
2. Conv reads A, writes B; peak live payload is 27,648 + 18,432 = **46,080 B**.
3. Alternate A/B through DW/PW stages. The largest later pair is much smaller; never overwrite a live source within an operator.
4. Final shared feature F is B, 1,728 B. GAP reads it into 48 INT32 sums and a small pooled vector. Class and presence FCs run first, without modifying F.
5. Box compression reads F and writes 288 B into A. F then becomes dead. FC reads A, producing four logits; decode the result into registers.

Eight INT32 accumulators need only 32 B of register state. GAP sums need 192 B. Avoid a full INT32 activation/partial-sum image: accumulate one output group at a time in registers, then requantize and write bytes. Avoid materialized im2col. Full-image buffers replace the need for a separate full convolution line-buffer implementation in the initial design.

For ordinary/PW convolutions, broadcast **one activation byte/cycle** to eight output-channel MACs and read one 64-bit weight word/cycle. Serialize up to eight output writes and matching per-channel requantization. Depthwise uses one active lane initially, avoiding eight unrelated activation reads. This sacrifices arithmetic utilization but preserves a simple memory-port schedule. At 100 MHz, the local peak read rates are 100 MB/s of activation bytes and 800 MB/s of weight bytes; these are internal BRAM access rates, not UART rates. Synchronous BRAM latency must be pipelined.

Keep source/destination in separate banks and specify read-during-write behavior. Reset valid flags/controllers, not every RAM element. Use explicit flatten order compatible with PyTorch, such as CHW, and a documented RGB-wire-to-memory mapping; eight-channel write serialization must preserve that order.

The initial protocol is stop-and-wait, so the input storage is reused for computation; no second complete input is required. Overlapping reception of the next frame would add at least `ceil(3N²/4096)` blocks (3/7/12 at 64/96/128), plus scheduling complexity. It barely helps while transmission already dominates, so defer it.

## 7. Arithmetic resources, latency model, and LUT uncertainty

Use eight shared DSP MAC lanes, plus provisionally 2–4 DSPs for a shared wide requantization multiplier: **10–12 DSPs, 11.1–13.3% of 90**. Reuse arithmetic for classification, objectness, boxes, and score products. Adding a box head does not require dedicated parallel box DSPs. Wider products or a different quantizer may change this count. One DSP48E1 has a 25×18 signed multiplier; its adder SIMD modes do not automatically yield several independent INT8 multipliers. [AMD UG479](https://docs.amd.com/api/khub/documents/gu4oRPFEh_Pm2uaAlfY6Kg/content).

For the specified schedule:

- Ordinary/PW issue cycles: `Hout×Wout×ceil(Cout/8)×(Cin/groups)×k²`.
- DW issue cycles: all DW MACs, because only one lane is active.
- FC issue cycles: `input_features×ceil(output_features/8)`; a four-output FC cannot use all eight lanes.
- Add one cycle per output byte for serialized writes and one cycle per final feature byte for GAP. LUT/decoder fill, control, CRC, bank switching, and stalls add more time.

For A96, an unrealistic eight-lanes-always-full bound is 886,704/(8×100 MHz) = **1.108 ms**. The actual proposed lane schedule has **188,160 MAC-issue cycles**. Adding output writes and GAP reads gives **235,689 cycles = 2.357 ms**, still excluding pipeline/control penalties. The comparison table uses **2–4 times that scheduled figure**, 4.71–9.43 ms, as a provisional allowance for an initial implementation. This allowance is an engineering planning assumption, not a cycle-accurate simulation, upper bound, or timing guarantee. At 50 MHz, compute latency approximately doubles; UART time is unchanged.

For comparison, B96/C96 scheduled figures are 3.123/2.526 ms before that allowance. The head therefore does not threaten the UART-scale budget unless implementation introduces severe stalls. At A96's planning latency, compute-only throughput would be approximately 106–212 frames/s, but this is **not** end-to-end performance and has not been measured.

**An honest LUT utilization estimate is not available without RTL and memory inference decisions.** A graph's parameter count does not specify controller, address generators, multiplexers, CRC, divider, or requantization implementation. A useful accounting formula is `LUT_total = 8×LUT_MAC_glue + LUT_address/control + LUT_quantize/decode + LUT_UART/CRC + LUT_memory_mux/debug`. Set a first synthesis acceptance budget of ≤10,000 LUTs for A/B (48.1% of available) or ≤12,000 for C (57.7%); these are design gates, not calculated usage. Each resolution needs its own synthesis/timing report. Do not describe those gates as measured utilization or promise that 100 MHz closes.

Highest hardware risks, in priority order, are bit-exact signed rounding/requantization; correct DW/PW weight addressing and channel order; BRAM port scheduling and synchronous read alignment; final-logit scale consistency and score/coordinate decode; UART framing/CRC and reset recovery; and routing/timing of wide arithmetic and control. A small first-layer proof and integer layer-by-layer comparisons should precede integration of the full detector.

## 8. Dataset: what exists and what must be added

### Verified annotation availability

The active classifier collection has **3,884 images**, split 2,934 train / 460 validation / 490 test, in 1,251 groups. It contains no usable bounding-box columns in its manifest or cached Google labels and no local detection annotation files found during inspection.

- **Google crowdsourced extension:** the local label CSV contains `ImageID,Source,LabelName,Confidence`, with no coordinates. Its official documentation explicitly describes image-level labels and states that bounding-box labels are unavailable for these images. Core Open Images boxes are a different annotation collection; class-name agreement does not make a box valid for an extension image. [Official crowdsourced extension](https://storage.googleapis.com/openimages/web/extended.html).
- **SAS cubes:** the cached zip contains 3,325 JPEGs in `good`, `bad`, and `empty`, plus directory entries; no coordinate annotation files. Its README describes a classification task requiring the cube to be positioned in frame. The project used 848 correlated cube photos from it; do not label all of them with a full-frame box or split neighboring views across sets. [SAS source](https://github.com/sascommunities/iot-image-classification-rubiks-cubes).
- **Core Open Images alternative:** its official boxable-class CSV, checked during this analysis, contains the exact target IDs for Pen (`/m/0k1tl`), Book (`/m/0bt_c3`), and Computer mouse (`/m/020lf`), but not Rubik's Cube (`/m/0h0b5`). It also contains an animal Mouse class: never substitute it for Computer mouse. The CSV has 601 records including its entity/root entry; the official documentation describes 600 boxable classes. [Class list](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv), [official box download/format documentation](https://storage.googleapis.com/openimages/web/download_v7.html).

The availability of core classes does not establish how many suitable images remain after licensing, size, source grouping, and single-object filtering. Full core annotations were not downloaded or joined against local image IDs in this stage, so the exact overlap/count is **unverified**. No new training images or models were downloaded. Only a small official class-list file was read into memory for research.

Existing image attribution is preserved in `data/attributions.csv`; current Google photos were selected with explicit CC BY 4.0 metadata, and SAS provenance/README/LICENSE are cached. For any new source, record per-image license and author, annotation license/version, and transformations before ingestion. Do not assume every image in another Open Images subset has the current collection's license.

### Practical data plan

Use a separate proposed `detection/` area in a later stage; leave classifier images, manifests, reports, and checkpoints unchanged. Store COCO annotations in original, EXIF-corrected pixel coordinates, or YOLO text `class cx cy w h` normalized to that image. Use fixed IDs 0 cube, 1 pen, 2 book, 3 computer mouse. In YOLO format an explicitly reviewed negative has an empty label file; a missing/unreviewed annotation must not silently mean background. Keep source, author, capture session, physical instance, review state, and split in a companion manifest.

1. **Pilot:** manually verify boxes on roughly 200 diverse images/class plus 200 reviewed negatives, about 1,000 total. Select whole source/session groups first. This tests whether the reused features localize at all; it is not a sufficient basis for category-wide accuracy claims.
2. **Reuse existing positives:** annotate suitable current images, retaining inherited classifier split membership. Some will be unsuitable due to small targets, ambiguity, multiple supported objects, or depictions. Existing class-level review is not box-quality review.
3. **Add real webcam scenes:** use varied object positions, scale, backgrounds, hands, orientations, motion blur, and lighting. Avoid a dataset of only centered objects. Start with prominent objects occupying roughly 20–80% of frame extent and record actual resized pixel dimensions; include a separate small/thin-object challenge set. Partial occlusion should be explicitly represented, not assumed solved by augmentation.
4. **Broader development collection:** roughly 3,000–5,000 verified positive frames total plus 1,000–2,000 genuine negatives is a reasonable collection budget, not a statistical requirement or accuracy guarantee. Existing annotations can contribute; more correlated frames alone add little. Collect additional independent groups when validation saturates.
5. **Category generalization:** use multiple physical instances per class, for example 5–10 development instances and separate unseen test instances. If only the user's four particular objects are available, report session generalization on those four objects, not generalization to arbitrary pens/books/mice/cubes.

Use a manual rectangle annotation tool such as CVAT; automated proposals are acceptable offline only if **every training proposal is reviewed and corrected**, and holdout ground truth is independently checked. Annotation assistance does not authorize another model to localize live hardware-demo frames. [CVAT rectangle workflow](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-rectangles/).

Annotate a tight, axis-aligned box around the visible object extent, excluding the hand/background; use a consistent visible-extent convention for occlusion and truncation. For the first version, exclude ambiguous frames with multiple supported objects from the main single-object benchmark, or label them separately as a documented challenge set. Never silently call a frame negative because its extra target was not annotated. Include empty scenes, hands alone, unrelated objects, and plausible confusers among negatives. SAS's `empty` folder can contain invalid/partial cube presentations, so it is not automatically a verified negative set.

Estimated annotation effort, conditional on 20–60 seconds per straightforward image, is **5.6–16.7 hours per 1,000 images**, before capture, imports, difficult examples, and independent QA. Existing 3,884 images would therefore take about 22–65 hours if all were annotated this way; pilot first rather than committing to every image. Actual tool-assisted speed must be measured.

### Leakage prevention and the transferred checkpoint

The checkpoint has already learned from original classifier training images and was selected using original validation images. Therefore those images cannot become an independent final detector test merely by assigning new box labels. The original classifier test has also already been inspected/evaluated in the project and should not become the repeatedly tuned final detector benchmark.

Preserve the original split files and assign their reused images the same roles in detection development. Retain old test labels for an explicitly named secondary legacy benchmark if desired; do not move them into training or claim they are a fresh personal-object holdout. Reserve **new webcam sessions** for final detection testing, with unseen physical instances as well if claiming category generalization. Do not use these images for calibration, automatic-proposal tuning, threshold selection, or QAT.

Group all frames from a capture session/video; group known same physical instances for an unseen-instance benchmark; group correlated source sequences and common authors conservatively. Reconcile those groups across old and new sources before splitting, with exact decoded-pixel hashes and perceptual duplicate review. Keep all SAS images in their single existing source group. If session/instance relationships connect everything, collect separate cohorts rather than random-splitting frames. Public author metadata cannot prove independence of every physical object.

Detection augmentation must transform boxes and pixels together: crop/translate/scale/rotate/flip plus brightness/contrast/color/blur/occlusion variations. Clip and validate boxes after transforms; do not leave original coordinates on rotated images. Keep evaluation deterministic. Replace the webcam's fixed 75% central crop with a documented full-frame aspect-preserving letterbox for detection, so moving objects remain observable. Save exact rounded resized dimensions and pad offsets for inverse mapping. Transfer weights still work structurally, but this framing change needs fine-tuning.

## 9. Training comparisons, evaluation, and realistic targets

No detector metric currently exists. The inherited 76.53% is closed-set image classification on public data; there are no box IoUs, detection precision/recall, background false-positive rates, or physical webcam results to report.

Train A first from the preserved checkpoint, initializing new layers separately. Use classification cross-entropy on positive examples, binary objectness loss on positives and negatives, and positive-only normalized box regression loss, initially Smooth L1 plus an IoU loss. Tune loss weights and score thresholds on validation only. Separate frozen-head warm-up from backbone fine-tuning and log both. If useful, a same-architecture scratch control can measure the value of transfer; B is an additional architectural comparison only if A underperforms. No scratch run is needed merely to satisfy this feasibility table.

Use identical grouped data, augmentation, training budget, and validation selection criteria for comparisons. The earlier classification runs favor the context model but do not rank detector architectures. Do not choose detector checkpoints by class accuracy alone: use validation detection quality with a specified score threshold or AP, and preserve negative-scene performance as a gate. A zero-area or full-frame prediction is not an acceptable detector shortcut.

Required reports:

| Metric | Explicit definition/reporting rule |
|---|---|
| Classification accuracy | On all positive ground-truth frames, argmax class accuracy; state whether abstentions are counted as errors and additionally report detection coverage |
| Per-class detection precision/recall | TP requires correct class, score ≥ validation-selected threshold, and IoU ≥0.50; wrong-class or low-IoU accepted output is FP and leaves the true object FN |
| IoU quality | Mean/median and distribution on positive frames irrespective of correct class; additionally show matched-only IoU so selection does not hide failures |
| Presence false positives | Fraction of reviewed negative frames that produce an accepted detection, plus presence recall on positive frames |
| AP/mAP | AP50 per class and mean; if sample size permits, AP averaged over IoU 0.50:0.05:0.95, explicitly limited to this single-object benchmark |
| Webcam robustness | Per-session and per-instance results, lighting/background/occlusion/thin-object breakdowns; separate physical-object and public-source results |
| Runtime | Model-only cycles, round-trip latency distribution, actual baud/error rate, detection update FPS, preview FPS, and age of displayed results |

Use class-aware one-to-one matching, even for one prediction. For AP, sweep the genuine joint scores rather than computing it from a single threshold; use low-threshold candidates for evaluation. Report AP settings, including the one-detection cap, rather than calling a custom score standard COCO AP without qualification. [COCO evaluator source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

**Proposed development acceptance goals, not predictions:** positive-frame class accuracy ≥85%; precision ≥85% and recall ≥80% at IoU 0.50 on held-out physical scenes; median positive-frame IoU ≥0.65; negative-frame false-positive rate ≤5%. Report each class separately and enough independent sessions to reveal uncertainty. These goals may not be achievable with this input/backbone, especially thin pens; failure should trigger data/feature-resolution investigation rather than invented claims. Dataset counts alone cannot justify expected percentages.

Earlier classifier training took about 4 minutes for the selected run, but that does not predict detection training or annotation time. Measure each detector run's actual duration. After quantization, compare FP32 and integer metrics on identical validation/test cohorts; a provisional gate is ≤2 percentage points lost AP50 and ≤0.02 lost mean IoU, with per-class review. Those are project tolerances to review, not measured regressions.

## 10. Proposed fixed-point arithmetic contract

This is a recommended contract for Stage 4, **not an existing quantized export**. No activation/weight scales, calibration ranges, or quantized accuracy have yet been measured. Exported numerical scales, LUT contents, thresholds, and multiplier/shift pairs must be frozen with each trained detector and used by both RTL and the integer reference.

| Item | Proposed rule |
|---|---|
| Input | RGB bytes; `q = pixel−128`, signed INT8, `s=1/128`, `zero_point=0`; matches inherited normalization |
| Intermediate features | Signed INT8, per-tensor positive scale, zero-point 0; ReLU outputs 0…127 |
| Weights | Signed INT8 symmetric −127…127, per-output-channel scale; zero-point 0, including each DW channel |
| Initial PTQ weight scale | `max(abs(channel_weights))/127`; use a positive fallback for an all-zero channel |
| Bias | INT32 `R(bias_float/(s_input*s_weight[channel]))`; fail export on overflow |
| Accumulation | Signed INT32 mathematical sum including bias; prove bounds/export checks, no silent wrapping |
| Requantization | `q_out = saturate(R(acc * M[channel] / 2^r[channel]))`, integer M/r approximating `s_input*s_weight/s_output`; explicit ReLU clamp as appropriate |
| Rounding | Rounds nearest with exact halfway cases away from zero; for positive shift r: `sign(t)*floor((abs(t)+2^(r−1))/2^r)` |
| Intermediate product | Use sufficiently wide signed product (at least 64-bit for INT32 × positive 31-bit M); handle r=0 and any left shifts explicitly |
| Padding | Input gray 128 corresponds to q=0; all convolution padding is integer 0 under the symmetric zero-point rule |
| GAP | Sum each channel's INT8 features in INT32, then `R(sum/(H×W))` at the same scale, saturating to INT8; share that pooled vector between class/presence |
| Output logits | Requantize to signed INT16 with common class-logit scale; box/presence can have their own fixed scales |
| Coordinates/scores | Unsigned 16-bit values with normalized value `u/65535`; do not imply an exactly represented calibrated probability |

Calibrate feature ranges on representative **training-only** positives and negatives with current preprocessing; compare max-range and percentile clipping using validation. Try PTQ first, then QAT from the fine-tuned detector if localization, objectness, or class quality degrades. QAT fake quantization must reproduce this hardware rounding/saturation contract; default CPU backend quantization kernels are not automatically a golden reference. Avoid mixing asymmetric library kernels with a symmetric RTL implementation. PyTorch provides quantization tooling, but exported custom arithmetic still needs independent verification. [PyTorch quantization documentation](https://docs.pytorch.org/docs/2.14/quantization.html).

The largest A96 dot product is box FC with K=288. Before bias its worst-case magnitude is bounded by `288×128×127 = 4,681,728`. Across the compared models/resolutions, B128 box FC has K=2,048, giving 33,292,288 before bias. These fit INT32, but **folded biases must also be checked**. Pooling sums and integer score products need separate bounds. DSPs may physically use 48-bit accumulation while exposing the same proven INT32 result; the reference must agree exactly.

### Confidence and coordinate decode

One concrete nonlinear plan is a 513-entry sigmoid LUT over −8…8 at step 1/32, and a 513-entry exponential LUT over −16…0 at step 1/32, both with UINT16 outputs. Map scaled logits to nearest grid points using the same rounding rule, and clamp outside the LUT range. For class softmax subtract the maximum common-scale class logit, evaluate exponential values, sum in sufficient width, and divide `exp_class×65535` by that sum with defined nearest rounding. Integer division can be serialized; only four classes are involved. Sigmoid entries represent `R(65535*sigmoid(x))`; exponential entries represent `R(65535*exp(x))`. Freeze actual table contents in the export rather than recomputing them differently in RTL.

At an initial output-logit scale of 2^-11, signed INT16 covers about −16…16; LUT lookup still clamps to its stated domain. This scale is a **candidate**, to validate against trained logit ranges and QAT. Export a changed scale explicitly if needed. Larger clipping or LUT approximation error can harm localization; do not assume it is negligible.

FPGA computes the class argmax, presence sigmoid, and `joint_score = R(presence_u16 * class_softmax_u16 / 65535)`, and compares it with a validation-selected UINT16 threshold. Class ties use the lowest class ID. This is a neural score, not an assertion of calibrated confidence. The host displays the returned score and class without redoing classification.

For A/B, apply sigmoid to center X/Y and width/height, then form corners using widened integer arithmetic: `xmin=clip(R((2*cx−w)/2),0,65535)`, `xmax=clip(R((2*cx+w)/2),0,65535)`, and similarly Y. Reject degenerate boxes. For C, first include the selected grid-cell offset and division by S using defined rounding, with score ties resolved by row-major cell index. FPGA emits clipped normalized corners; any host inverse-letterbox clipping is geometric scaling only.

Build an explicit Python integer reference with exported tensors/tables and these exact rules. Validate scalar arithmetic boundary vectors, each layer, pooling, all heads, and final packets before comparing end-to-end labels. A PyTorch FP32 forward pass or generic quantized model is not bit-accurate hardware evidence. The proposed A graph is easiest to verify because it adds only ordinary PW/FC operations and small fixed decoding to the shared convolution engine.

## 11. UART protocol and throughput

The Basys 3 J4 Micro-USB connection exposes independent programming and UART functions through FT2232HQ. The UART path is not a USB video device. [Digilent interface description](https://digilent.com/reference/_media/basys3:basys3_rm.pdf). FTDI documents higher UART baud rates, but that does not establish reliability of this FPGA UART, host driver, or cable; benchmark them. [FT2232H datasheet](https://ftdichip.com/wp-content/uploads/2024/03/DS_FT2232H.pdf).

Recommended protocol, version 1, 8N1, fixed little-endian integers:

| Field | Bytes | Meaning |
|---|---:|---|
| Magic | 4 | Fixed synchronization sequence, e.g. `B3 5D A7 C1` |
| Version | 1 | Protocol version, initially 1 |
| Type | 1 | HELLO=0x01, IMAGE=0x10, RESULT=0x11, ERROR=0x7F |
| Frame ID | 4 | Monotonic host-generated ID, echoed unchanged |
| Payload length | 4 | Bounded length before accepting data |
| Payload | variable | Defined below |
| CRC32 | 4 | CRC-32/ISO-HDLC over version through end of payload, excluding magic |

CRC parameters: polynomial 0x04C11DB7 (reflected implementation 0xEDB88320), init 0xFFFFFFFF, reflected input/output, xor-out 0xFFFFFFFF; serialize CRC little-endian. Include known test vectors in both software and RTL. Firmware runs inference only after full packet/CRC validation, rejects unsupported resolution/format, and returns an explicit error for a malformed request. Reset and timeout must clear partial-frame state and resynchronize safely; a magic-like byte sequence inside a length-validated image is ordinary payload.

IMAGE payload: uint16 side length N, uint8 pixel format (RGB888), uint8 reserved, then `3N²` interleaved RGB bytes. Total input size: **3N²+22 bytes**. Send unsigned pixels; FPGA converts to signed input quantization during storage. A HELLO response should identify model/version/hash, input size, class order, and score/coordinate encoding so the host cannot silently use the wrong export.

RESULT payload: uint8 detection-valid flag, uint8 class ID (0…3, 255 if invalid), uint16 joint score, four uint16 normalized corners, uint32 inference cycles. Payload 16 B; total result **34 B**. Define cycle count from accepted inference start through final decoded result, excluding UART. On no-object frames, send detection-valid=0 and no drawable box. ERROR uses its own documented payload.

Start stop-and-wait at 115,200 baud for protocol/CRC/loopback tests. Then validate 1 Mbaud, and only then 3 Mbaud. Use an RX synchronizer, a fractional baud generator or a quantitatively checked divider, and bounded RX buffering. At 3 Mbaud the 100 MHz clock gives only ~33 cycles/bit; a naive integer 16× oversampling divider can be unsuitable. Measure actual error rates and sustained transfer latency rather than assuming a driver setting proves throughput.

At baud B, `t_input = 10*(3N²+22)/B`. With stop-and-wait, `t_roundtrip >= 10*(3N²+56)/B + t_inference + t_host/USB`. Tables omit unknown host/USB overhead and retransmissions.

| Input | Input bytes incl. framing | 115,200 baud input ms | 1 Mbaud input ms | 3 Mbaud input ms |
|---|---:|---:|---:|---:|
| 64×64 RGB | 12,310 | 1,068.58 | 123.10 | 41.03 |
| 96×96 RGB | 27,670 | 2,401.91 | 276.70 | 92.23 |
| 128×128 RGB | 49,174 | 4,268.58 | 491.74 | 163.91 |

For A, using its tabled inference allowance and the 34-byte response:

| Input | End-to-end FPS at 115,200 | At 1 Mbaud | At 3 Mbaud |
|---|---:|---:|---:|
| 64×64 | ~0.93 | ~7.84–7.97 | ~22.1–23.1 |
| 96×96 | ~0.414–0.415 | ~3.49–3.55 | ~9.80–10.28 |
| 128×128 | ~0.233–0.234 | ~1.97–2.00 | ~5.53–5.80 |

These rates are optimistic system estimates before host overhead, not promised FPS. Preview rate is independently determined by laptop capture/display. UART transfer FPS, model-only FPGA FPS, and displayed detection updates must be measured and labeled separately. UART at 115,200 does not meet a few-FPS goal at 96 RGB. If 1 Mbaud is reliable, a few updates/s is plausible; 30 FPS detection at 96 RGB is not supported by this 1/3-Mbaud plan. Faster links or transport changes would require a separate design investigation.

## 12. Webcam behavior and coordinate ownership

The future hardware-mode application should capture/display continuously and run UART communication in a worker, keeping at most one pending latest frame. Drop old frames rather than accumulating a queue. Preserve frame ID, capture timestamp, exact resize/padding transform, and sent image dimensions for every in-flight image. Show link status, preview FPS, completed FPGA detection FPS, FPGA cycle latency, and capture-to-result latency/age.

Use full-frame letterboxing rather than classifying a fixed box. If original dimensions are W×H, calculate resized integer dimensions and their actual X/Y scales, pad into N×N, and save left/top padding. Map returned normalized corners to N-space, subtract padding, then divide by those saved scales and clip to frame boundaries. Python may perform these geometric operations; **class choice, confidence computation, objectness thresholding, and predicted box location originate on FPGA**.

Do not display a stale box as a fresh inference. A detection belongs to its frame ID; either show a paired analyzed-frame view or overlay the latest result with visible age and expire stale boxes. Without a new FPGA result or a separately authorized tracker, there is no legitimate continuously updated location between detections. At ~3.5 updates/s, rapidly moving objects will visibly lag. Do not hide this with laptop optical flow, contours, a second network, or invented boxes/scores.

If hardware disconnects, show hardware inference unavailable and remove/expire its boxes. Do not silently fall back to laptop classification. Provide a separately selected, clearly labeled software reference mode for FP32/integer debugging. Existing `Webcam.cmd` remains the original laptop classifier and must not be relabeled as FPGA detection.

## 13. Development sequence and review gates

| Stage | Work after this report is reviewed | Evidence required before advancing |
|---|---|---|
| 1 — This report | Inspect, calculate, document; stop here | Existing model/data preserved; assumptions and resource limits visible |
| 2 — Annotated data | Separate detection manifests/COCO or YOLO labels, negatives, new webcam cohorts | Reviewed boxes, explicit no-object labels, group/instance/duplicate audit, frozen fresh holdout |
| 3 — Software A detector | Transfer all compatible tensors; train new heads; fine-tune | Actual localization and presence metrics, visual box review, independent webcam examples |
| 3b — Conditional comparison | B/C or stride/resolution changes only if A underperforms | Fair validation comparison, recomputed memory/MACs, architecture choice frozen before final testing |
| 4 — Integer reference | PTQ then QAT if needed; export tensors/scales/LUTs | Measured FP32→integer regression, exact arithmetic contract, per-layer golden vectors |
| 5 — Early hardware proof | UART/CRC loopback, BRAM access, first conv, requantizer | Software/RTL exact agreement including edge/padding cases; preliminary utilization/timing |
| 5b — Accelerator | Shared PW/ordinary/DW engine, GAP, class/presence/box heads, fixed decode | Layer-by-layer equality and full-detector simulation against integer reference |
| 6 — Implementation | Vivado synthesis/place-and-route for chosen input/model | Actual LUT/FF/DSP/BRAM use, RAM inference, timing at target clock, constraints reviewed |
| 7 — Board tests | Program `.bit` with initialized exported memories; send labeled test images | Correct packet IDs/CRC, integer-identical returned results/cycles; sustained baud/error benchmark |
| 8 — Webcam demo | Nonblocking preview + real board inference | Genuine box/score updates, latency/FPS distributions, independently annotated physical tests |

The earliest hardware proof should be small; it need not wait until an entire accelerator is written. However, a successful first convolution or bitstream build is not evidence that detection quality is acceptable. Do not invest in the full RTL graph until A has demonstrated useful localization in software.

Single-object detection is the recommended deliverable. A cannot emit two genuine boxes by changing the display code. C has a plausible multi-object path, but same-cell collisions, duplicate handling, contextual coverage, scoring, and a corresponding multi-object benchmark need work; defer that scope even though raw BRAM alone leaves room.

## 14. Direct answers to the feasibility questions

| Question | Answer |
|---|---|
| 1. Can Basys 3 classify and localize? | Analytically yes for the specified small INT8 single-object graph; synthesis, integer accuracy, and board tests remain prerequisites. |
| 2–3. Can the 6,084-parameter model be reused; which layers? | All nine convolution weights/biases plus the original GAP/class FC in A; 20 trained tensors, all 6,084 parameters. |
| 4. Where should branching occur? | After final 48-channel context ReLU, before GAP; GAP remains only on class/presence paths. |
| 5. Is RF sufficient? | RF63 per cell is limited; A's flattened head sees the whole spatial map. Adequacy for fine localization is unmeasured. |
| 6–7. Additional parameters/model size? | A96: box branch 1,548, presence 49; total 7,681. Raw INT8 weights+INT32 biases 8,500 B, plus metadata/padding. |
| 8–9. BRAM and simultaneous activations? | A96 concrete 19/50 blocks, allow up to 25 provisionally; 46,080 B feature payload with sequential heads. |
| 10–11. Work/eight MAC lanes? | A96 886,704 MACs; eight lanes with scalar DW are sufficient for the estimated UART rate. |
| 12. Latencies? | A96 planning 4.71–9.43 ms compute at 100 MHz; 276.70 ms input at 1 Mbaud; neither measured on hardware. |
| 13. Existing boxes? | No usable boxes locally. Crowdsourced extension/SAS are classification sources; core OI offers three target box classes, not cube. |
| 14. Additional preparation/training? | Verified box annotations, negatives, and fresh session holdouts; train new heads and fine-tune; unknown final data/time requirements. |
| 15–16. One or multiple objects? | One prominent object is the initial feasible target; multiple objects require a different output/training/decode scope and are deferred. |
| 17. Accuracy/localization expectations? | No measured detector accuracy. Section 9 states aspirational acceptance goals, not forecasts or reported results. |
| 18. Highest hardware risks? | Quantization agreement, memory/weight addressing, BRAM ports, nonlinear decode, UART robustness, timing. |
| 19. Easiest golden-reference design? | A, with sequential PW/FC heads and precisely specified integer arithmetic. |
| 20. Recommended architecture/sequence? | A96 → annotated software validation → PTQ/QAT and integer reference → small hardware proof → full RTL → implementation → physical tests. |

**Stage 1 is complete with this document. Stop here for review.** Original trained classifier artifacts and reported results remain authoritative for classification only. Nothing in this feasibility calculation establishes trained detection quality, INT8 accuracy, successful FPGA fit, or a working live detector.
