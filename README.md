# Basys 3 object recognition — CNNs running in FPGA hardware

Small INT8 neural networks running entirely in the fabric of a Digilent Basys 3
(Artix-7 XC7A35T, 100 MHz): no soft CPU, no HLS, no laptop neural inference. The laptop
webcam captures a frame, sends it over USB UART, and the FPGA returns the answer.
Classes: Rubik's cube, pen, book, computer mouse.

| Design | What runs on the FPGA | Verified results |
|---|---|---|
| [`fpga/`](fpga/README.md) classifier | 9-layer CNN on 96×96 RGB, 6,084 parameters | 75.92% on 490 test photos, every output bit-exact with the integer reference; 36.75 ms per image; 2,636 LUTs, 7 DSPs |
| [`fpga_det/`](fpga_det/README.md) box detector | 11-layer CNN on 128×128 RGB + soft-argmax box head, 31,977 parameters, 4.78M MACs | class + box per frame; webcam validation mAP50 0.741, class accuracy 98.8%; bit-exact on the board; 201 ms per image |
| [`fpga_viz/`](fpga_viz/README.md) recording demo | either design plus an observation-only telemetry block | live chip map from Vivado's real placement, per-layer activation maps and timing read back from the FPGA every frame, all re-checked bit-exact on the laptop |

Each RTL design is checked in simulation against a bit-exact NumPy integer reference
before it is built, and the builds refuse to produce a bitstream that misses 100 MHz
timing. Quick start with a board: `ProgramDetectorDemo.cmd`, then `RecordDemo.cmd`
(see [`fpga_viz/README.md`](fpga_viz/README.md)).

## Training the classifier (`classifier/`)

The sections below document how the classifier was trained from scratch on 96×96 RGB
images. Class order is permanently `rubiks_cube`, `pen`, `book`, `mouse` (computer
mouse, not the animal). The training code itself uses no FPGA tools or pretrained network.

**Completed results:** see [RESULTS.md](RESULTS.md). The selected `best_model.pth`
has 6,084 inference parameters and measured 73.48% validation / 76.53% test accuracy.
The original-size reference is preserved as `baseline_3252.pth` (62.83% validation /
67.96% test). The larger model was chosen using validation before test evaluation.
These public-data results do not establish accuracy on your own physical objects.

## Environment and quick commands

Use PowerShell from this folder:

```powershell
Set-Location 'C:\Users\ndaqu\Documents\Basys3Classifier'
.\.venv\Scripts\python.exe -m classifier.check_environment
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m classifier.model
```

`setup.ps1` can recreate the environment using the pinned dependencies. It uses the
official PyTorch CUDA 13.0 index. CPU-only laptops can install a CPU torch/torchvision
pair instead. Device `auto` checks CUDA availability and executes a real GPU tensor
operation before selecting it. `--device cpu` and `--device cuda` override selection.
The GPU driver is not changed by setup. No separate CUDA toolkit is required for
these prebuilt-wheel training commands.

## Public data: approximately 1,000 images per class

```powershell
.\.venv\Scripts\python.exe -m classifier.fetch_data --target 1000
```

Sources, selected transparently:

- [Google Open Images crowdsourced extension](https://storage.googleapis.com/openimages/web/extended.html):
  positive human-verified labels only. Exact label IDs are `/m/0h0b5` Rubik's cube,
  `/m/0k1tl` Pen, `/m/0bt_c3` Book, `/m/020lf` Computer mouse. Photos positive for
  more than one of our targets are excluded. The importer requires an explicit
  CC BY 4.0 image-license record and a named author. Machine-only labels are not used.
- [SAS cube collection](https://github.com/sascommunities/iot-image-classification-rubiks-cubes):
  its README expressly permits use of the accompanying photographs, and the project
  is Apache-2.0. Only `good` (solved cube) and `bad` (scrambled cube) photographs
  are mapped to `rubiks_cube`. The `empty` directory is excluded. The source
  README and LICENSE are preserved in `data/sources/sas`.

The Google extension has fewer cube images than the requested count. The importer
uses eligible Google images and fills the cube shortfall with SAS photos. These
SAS frames are correlated, not hundreds of independent scenes. All SAS photos stay
in ONE group. All Google photos credited to the same author stay in ONE group,
even across classes. True sessions and physical instances are not fully documented
in these sources; this grouping reduces leakage but cannot prove its absence.

`data/acquisition.json` records actual counts/failures. `data/attributions.csv`
preserves per-image authors, URLs, image IDs, licenses, and hashes. Retain these
records and the licenses if sharing derived data. Do not commit all images blindly.
Downloads use only documented public data/metadata URLs, not a paid API or a model.
Dead URLs and duplicates can reduce actual counts below the target; do not count
augmented variants as new independent photographs.

## Prepared dataset in this workspace

The downloaded collection contained 1,000 images/class. After pretraining visual
curation, the active `data/manifest.csv` contains 3,884 images: 1,000 cubes,
948 pens, 936 books, and 1,000 computer mice. All 2,000 pen/book thumbnails were
reviewed, plus deterministic 80-image samples for cubes/mice. The 116 exclusions
are recorded in `data/exclusions.csv`; originals and attributions remain intact.
This is a thumbnail review, not certification of every source label.

The cube source mix before curation is 848 SAS photos and 152 Google photos.
The SAS photographs predominantly show a 2×2 cube; Google includes 3×3 puzzles.
Consequently these data do not establish performance on your own 3×3 cube.
Books include bound notebooks as well as printed books. Source/author groups
and near-duplicate components remain intact. Exact 70/15/15 class proportions
are impossible for the dominant SAS group:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Rubik's Cube | 923 | 43 | 34 |
| Pen | 664 | 136 | 148 |
| Book | 666 | 132 | 138 |
| Mouse | 681 | 149 | 170 |
| Total | 2,934 | 460 | 490 |

The active manifest is the curated split. `manifest_initial.csv` preserves the
pre-curation split for auditing; it was never used to train a real model.
`manifest_curated.csv` is an identical copy of the active manifest. To reproduce
the curated preparation in a NEW filename, run:

```powershell
.\.venv\Scripts\python.exe -m classifier.prepare --raw data/raw --exclude data/exclusions.csv --output data/manifest_reproduced.csv
```

## Review and freeze the splits

```powershell
.\.venv\Scripts\python.exe -m classifier.prepare --raw data/raw --output data/manifest.csv
```

The preparation tool decodes every image, hashes exact file and decoded pixels,
and merges near-duplicate candidates (same-class dHash distance ≤3) into connected
groups BEFORE splitting. It seeks approximately 70/15/15 class proportions while
preserving whole groups. Large source groups can prevent those exact proportions.
Every class must have at least three independent groups and appear in all splits.
Grouping is never chosen using model accuracy. The seed defaults to 42.

Inspect `data/review/*.jpg` and the source descriptions. Reject any visibly wrong,
ambiguous, tiny, nonphotographic, or low-quality images. Remove rejected generated
downloads only, then prepare a NEW manifest filename. Do not edit or reshuffle a
manifest after viewing test performance. Source labels are human verified, not
infallible; sample visual review does not certify every image.

Preparation saves the fixed manifest plus `manifest.audit.json`, split counts,
group counts, and cross-split near-duplicate warnings. Training rechecks group and
exact-duplicate separation and verifies that image files have not changed. dHash
is a heuristic; it does not discover every crop, augmentation, or related frame.

For local photos the expected layout is:

```text
data/personal_raw/
  rubiks_cube/my_cube/session_01/image.jpg
  pen/my_pen/session_01/image.jpg
  book/my_book/session_01/image.jpg
  mouse/my_mouse/session_01/image.jpg
```

Each session folder needs `session.json` with `source` and `license`/ownership.
`classifier.collect` and `classifier.import_images` create it. Session names should
be shared across classes captured in the same setting; changing class is not a
new capture session. For your four specific objects, use session grouping. If
generalizing to different physical instances, use `--group-by instance-session`;
both sessions and instances must then be disjoint, which requires collection in
separate cohorts. If everything connects into one group, collect independent data
rather than forcing a random frame split.

## Train and select using validation only

```powershell
.\.venv\Scripts\python.exe -m classifier.train --manifest data/manifest.csv --output runs/baseline_new --epochs 120 --workers 0
```

Default optimizer: AdamW, learning rate 0.002, weight decay 0.0001; plateau LR
reduction, gradient clipping, batch 64, maximum 120 epochs, early stopping after
20 epochs without a better checkpoint. Checkpoint selection uses validation
accuracy, then validation loss as tie-breaker. The test loader is not constructed
during training unless `--evaluate-test` is explicitly requested after selection.
Use a new output folder for every experiment; existing results are not overwritten.
Images are cached at 96×96 in RAM by default; `--no-cache` disables this.

For the public collection's source imbalance, the compared training recipe uses
`--balance-sources --batch-norm`. Source balancing gives each class equal expected
sampling mass, then each source within a class equal mass. It samples training
images with replacement; augmentation creates variations, not independent photos.
Validation and test remain unchanged and are never oversampled.

Optional batch normalization is used only for optimization during training. It is
folded into each convolution's weights and bias before checkpointing, with no BN
operators or extra inference parameters in `best_model.pth`. Baseline training has
3,412 learnable parameters in this mode, but its exported inference model still
has exactly 3,252. Context training has 6,340; inference has 6,084. Tests compare
folded and unfused evaluation outputs with explicit floating-point tolerances.
This is not INT8 conversion or a claim of quantized accuracy.

```powershell
.\.venv\Scripts\python.exe -m classifier.train --output runs/baseline_balanced --epochs 80 --patience 15 --balance-sources --batch-norm
.\.venv\Scripts\python.exe -m classifier.train --architecture context --output runs/context_balanced --epochs 80 --patience 15 --balance-sources --batch-norm
```

The plain baseline and this recipe differ in both sampling and normalization;
their comparison cannot isolate the effect of either change. The two balanced
models use the same recipe and frozen splits for the architecture comparison.

Augmentation applies only to training: moderate rotation, translation, scaling,
horizontal flip, brightness, contrast, saturation, and hue variation. Validation,
test, still-image, and webcam preprocessing share the same deterministic code:
RGB, EXIF orientation, aspect-preserving bilinear resize/pad with gray 128, followed
by `(pixel - 128)/128`. Padding therefore represents zero after normalization.
Holdout images are not augmented. No pretrained weights are downloaded.

Saved outputs include:

- `best_model.pth`: model tensors plus architecture, class order, preprocessing,
  seed, selected epoch, and manifest digest; safe tensor-oriented loading.
- `config.json`, `manifest_snapshot.csv`, `history.csv`, `learning_curves.png`.
- `summary.json`: measured training seconds, device, epochs, train/validation
  accuracy, model payload sizes, actual checkpoint file size, optional test accuracy.
- `train_*.json/csv/png`, `val_*.json/csv/png`: metrics, per-class precision/recall/F1,
  confusion matrices, and per-image predictions. Per-class accuracy means recall:
  correct predictions divided by that true class's example count.

Training time covers epochs, validation, and checkpoint writes; it excludes dataset
download/preparation, cache loading, and final plotting/evaluation. Checkpoint file
size includes metadata/serialization overhead; it is not FPGA weight memory.

After deciding the architecture and training recipe using validation:

```powershell
.\.venv\Scripts\python.exe -m classifier.evaluate --checkpoint runs/baseline/best_model.pth --manifest data/manifest.csv --split test --output runs/baseline
```

This generates test confusion matrices (counts and row-normalized), per-class
metrics, per-image predictions, and a 95% Wilson interval with an explicit
independence caveat. Repeated test-driven architecture tuning invalidates the
holdout's role. A public-data test is not a test on your own physical objects.

## Model count and future INT8 compatibility

| Model | Weights | Biases | Total parameters | MACs/image | INT8 live activations |
|---|---:|---:|---:|---:|---:|
| Baseline (default) | 3,088 | 164 | **3,252** | **773,184** | 45 KiB |
| Extra context block | 5,824 | 260 | 6,084 | 871,680 | 45 KiB |
| Two context blocks | 8,560 | 356 | 8,916 | 970,176 | 45 KiB |
| Standard convolutions | 19,992 | 108 | 20,100 | 2,322,624 | 45 KiB |

The baseline is conv3×3/s2/8, DW3×3/s2 + PW1×1/16, DW3×3/s2 + PW1×1/32,
DW3×3/s2 + PW1×1/48, global average pooling, FC4. Every convolution has bias and
ReLU. There is no batch norm, expansion, residual path, or inference softmax in
the model itself. FP32 parameter payload is 13,008 bytes; future INT8 weights plus
INT32 biases are 3,744 bytes, excluding scales/alignment.

If the baseline underperforms, first inspect labels, framing, and train/validation
gap. The context model adds 87% parameters but only 12.7% MACs, grows receptive
field from 31 to 63 pixels, and keeps peak INT8 feature memory unchanged. Standard
convolutions cost about 3× the baseline MACs and 6.47× weight bytes but simplify
the operator set. These are comparisons, not automatic architecture changes.

`--architecture context2` adds a second identical stride-1 context block: 8,916
inference parameters, 9,268 during optional BN training, a 95-pixel receptive field,
and about 25.5% more MACs than the baseline. The fixed 96×96 input and peak INT8
feature memory remain unchanged. This is the last small architecture trial in
the initial experiment set; the final choice is frozen before test evaluation.

```powershell
.\.venv\Scripts\python.exe -m classifier.train --architecture context --output runs/context
```

Keep the same fixed manifest. Compare validation results before selecting any new
model. Do not report analytical INT8 storage as measured INT8 accuracy. Today the
checkpoints are FP32; future calibration/QAT and bit-exact integer arithmetic are
separate work. Input centering, Conv/ReLU/pooling/linear operators, lack of runtime
normalization, and small parameter counts keep that work practical. For INT8 later,
define per-channel weight scales, per-tensor activation scales, bias scales,
rounding/saturation, GAP division by 36, and common-scale FC comparisons.

## Webcam and your own physical-object evaluation

Double-click `Webcam.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe -m classifier.webcam --checkpoint best_model.pth
```

Place one object inside the square. Q/Escape quits. Camera 0 is the default;
try `--camera 1` if needed. Check Windows camera permissions and close competing
camera apps if access fails. This runs inference on the LAPTOP. Displayed softmax
scores are not calibrated probabilities or reliable unknown-object rejection.

For a camera-free prediction:

```powershell
.\.venv\Scripts\python.exe -m classifier.webcam --image 'C:\path\photo.jpg'
```

For personal photos, use new backgrounds/lighting/days, varied distance and
orientation, and keep the full object visible. Capture at least 10 independent
sessions with 50–100 varied images/class/session for training/validation work.
Reserve additional sessions exclusively for a final local test. Do not add local
test images to training after inspecting their predictions.

For a separate personal benchmark containing only new photos, use
`classifier.prepare --raw data/personal_raw --output data/personal_test.csv --test-only`,
then `classifier.evaluate --checkpoint runs/CHOSEN_RUN/best_model.pth --manifest
data/personal_test.csv --external --split test --output runs/personal_test`.
This does not require inventing training/validation splits for an evaluation-only
collection. All four classes must be present. Use the chosen run's checkpoint
path so its neighboring manifest snapshot can also be checked for exact image
overlap. Multiple independent capture sessions are preferable to one burst.

```powershell
.\.venv\Scripts\python.exe -m classifier.collect --label pen --instance my_pen --session desk_day1
.\.venv\Scripts\python.exe -m classifier.prepare --raw data/personal_raw --output data/personal_manifest.csv
```

Space saves a photo; A toggles timed capture; Q exits. The capture tool saves the
same center crop that the webcam predictor uses. Use `classifier.import_images
--help` to import manually labeled local folders with explicit provenance.

## Verification and limits

The unittest suite checks exact architecture counts/MACs and gradients, RGB/scaling/
padding conventions, confusion-matrix arithmetic, rejection of group/duplicate
leakage, conservative grouping, loss reduction on a fixed batch, and a full
synthetic fixture prepare/train/checkpoint/evaluate cycle. Synthetic fixtures are
flagged and never presented as object-classification accuracy. The webcam refuses
to use a synthetic smoke-test checkpoint.

Physical-object accuracy and live camera behavior require actual camera trials.
Do not infer those results from passing software tests. All claims should retain
the dataset provenance, split limitations, test sample counts, and FP32/INT8 status.
