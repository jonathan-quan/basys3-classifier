# Basys 3 project — software detector training

This directory is separate from the original classifier. It implements the report's
96×96 RGB, **7,681-parameter** single-object network. All 6,084 compatible classifier
parameters initialize the backbone and class head. New learned branches predict
object presence and center/size; the box branch consumes the spatial map before GAP.
There are no color/contour location rules, pretrained YOLO models, FPGA RTL, or INT8
inference in this stage. PyTorch floating-point training does not prove FPGA fit.

The source classifier, checkpoints, original manifests, results, and webcam tool
are preserved. Their hashes are recorded in
[original_files_sha256.json](C:/Users/ndaqu/Documents/Basys3Classifier/detection/original_files_sha256.json).
Detector training outputs live under this directory's `runs/`, never in the
original classifier run directories.

## Environment and checks

Use the existing isolated Python/CUDA environment; no new large ML framework is
required. From PowerShell:

```powershell
Set-Location 'C:\Users\ndaqu\Documents\Basys3Classifier'
.\.venv\Scripts\python.exe -B -m unittest discover -s detection/tests -v
.\.venv\Scripts\python.exe -B -m detection.model
```

Training automatically attempts CUDA execution and falls back to CPU. Runs record
the actual selected device, GPU, package versions, timing, source-checkpoint hash,
and dataset hash. The test suite uses explicitly synthetic fixtures to check
geometry, leakage guards, gradients, BN folding, and the train/export/evaluate
workflow. Synthetic test results are never real object accuracy evidence.

## Data and licenses

The retained dataset has **2,629 images**: 2,539 positive frames with boxes and 90
reviewed negative frames. See the active manifest and audit:

- [training_manifest.jsonl](C:/Users/ndaqu/Documents/Basys3Classifier/detection/data/training_manifest.jsonl)
- [training_manifest.audit.json](C:/Users/ndaqu/Documents/Basys3Classifier/detection/data/training_manifest.audit.json)
- [active_dataset.json](C:/Users/ndaqu/Documents/Basys3Classifier/detection/data/active_dataset.json)

The initial existing annotations are 407 single-cube images from
[phonght32's cube webcam collection](https://github.com/phonght32/yoloV2-detectRubik),
whose author describes self-captured images and provides an MIT project license,
and 2,093 pen/book/computer-mouse images from
[official Open Images boxes](https://storage.googleapis.com/openimages/web/download_v7.html).
Train annotations use the v6 release, validation/test source annotations v5.
Open Images image selection requires an explicit allowed CC BY 2.0/3.0/4.0 or
CC0 per-image license and known author. Source annotations are CC BY 4.0.
Metadata, source box CSVs, source README/LICENSE, URLs, and per-image provenance
are preserved under `data/sources/` and in the manifest. The large official source
CSV caches use several GB on disk; this is dataset tooling, not model storage.

The owner reviewed 130 additional candidates in the local UI. All 39 retained
positive cube boxes differ from the automatic proposals; 90 background images
were explicitly confirmed. One original cube image incorrectly marked as
background was excluded after visual QA, without automatic relabeling. The raw
review and exclusion records remain in `data/reviews.jsonl` and
`data/exclusions.json`. No unreviewed candidate enters training or evaluation.
Official source boxes are existing ground truth, not a claim that every source
image was manually rechecked locally. Some are small, occluded, cropped, or
cluttered; source label noise and imperfect single-object coverage remain risks.

The cube webcam collection stays **entirely in training**, because its photos are
correlated and original sessions/instances are undocumented. All known common
authors, sessions, exact duplicates, and detected same-class near duplicates are
grouped before splitting. New images are also checked against the classifier's
original training/validation/test identities. Original classifier split roles
remain fixed when its images are reused. The 39 manually boxed Google cube
images inherit original validation/test roles; they are a **legacy public cube
benchmark**, not a fresh personal-webcam or guaranteed unseen-instance test.
Public metadata cannot establish every unknown physical-object relationship.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Rubik's Cube | 407 | 19 | 20 |
| Pen | 465 | 93 | 106 |
| Book | 700 | 152 | 148 |
| Computer mouse | 298 | 64 | 67 |
| Reviewed background | 64 | 13 | 13 |
| Total | 1,934 | 341 | 354 |

The original 2,934 classifier **training-only** images also supply auxiliary class
rehearsal/distillation. This reuses images lacking boxes without inventing
localization targets. Original classifier validation/test images are not used
for these updates.

The detector pilot and draft manifests are retained for provenance. The selected
training dataset is the active manifest above. Every retained image keeps its
pilot split role; adding reviewed data did not reshuffle the test set.

## Acquisition and manual review

For a fresh acquisition, the source tool downloads only the documented sources:

```powershell
.\.venv\Scripts\python.exe -B -m detection.fetch_data --limit 1000 --workers 8
```

Do not rerun acquisition/preparation to change a frozen experiment silently. Use
a new manifest filename for additions and inherit the previous image roles.
Images are referenced from the original dataset where possible; originals are
not modified. Missing original boxes are listed separately as candidates.

The existing manual-review queue can be opened by running
[ReviewAnnotations.cmd](C:/Users/ndaqu/Documents/Basys3Classifier/detection/ReviewAnnotations.cmd),
then visiting http://127.0.0.1:8765. If the review server is already running, use
that page directly. Positive verification requires a valid drawn/corrected box;
negative verification explicitly means none of the four supported objects is
visible. Skipping does not approve a candidate. Reviews record reviewer identity
and optional lighting/background/orientation tags. Editing reviews after training
does not change frozen manifests or completed results; prepare a new dataset
version and retrain if labels change.

```powershell
.\.venv\Scripts\python.exe -B -m detection.prepare --output detection/data/new_manifest.jsonl --inherit-manifest detection/data/training_manifest.jsonl
```

Preparation refuses incomplete four-class/negative splits by default, never
overwrites an existing manifest, exports COCO JSON and YOLO labels, and creates
box contact sheets. `--allow-incomplete` is for explicitly limited pilots only;
its checkpoints are marked accordingly. For documented unseen-instance tests,
use `--instance-disjoint` and independently captured cohorts. Do not force a
random split when those constraints connect everything.

## Reproduce detector training

The selected architecture remains fixed. Training-only normalization in the new
box compression head uses 7,689 learnable training parameters, and folds into
**7,681 inference parameters**. Exported checkpoints contain no BN operators.
Only the spatial head uses this new normalization; classification-only rehearsal
does not update its statistics.

```powershell
.\.venv\Scripts\python.exe -B -m detection.train --manifest detection/data/training_manifest.jsonl --output detection/runs/reproduction --epochs 120 --warmup 5 --patience 120 --backbone-lr 0.0002 --head-lr 0.001 --head-batch-norm --schedule cosine
```

Losses include positive-only class cross-entropy and box Smooth L1/IoU loss,
presence binary cross-entropy on positives/negatives, and original-training-only
classification rehearsal. Augmentation transforms pixels **and boxes** together:
rotation, translation, scale, mirroring, brightness/contrast/color, and light
blur. Evaluation is deterministic letterboxing, gray128 padding, and
`(pixel−128)/128` normalization. No full-frame box is inferred merely from a
class label; source-provided full-frame boxes are possible for tightly cropped
photos and are identified by their annotation provenance.

Validation AP50 across the four classes selects checkpoints, with all-positive
mean IoU as tie-breaker. Validation alone chooses the displayed joint-score and
presence thresholds. No test images are evaluated during training. Score is
presence sigmoid × maximum four-class softmax, an **uncalibrated neural score**.

```powershell
.\.venv\Scripts\python.exe -B -m detection.evaluate --checkpoint detection/runs/reproduction/best_detector.pth --manifest detection/data/training_manifest.jsonl --split test --output detection/runs/reproduction/evaluation --compare-original
```

This reports class accuracy on all positive frames; object-presence precision,
recall, and negative-frame false positives; class-aware detection precision/
recall at IoU≥0.50; all-positive and matched-only IoU; per-class AP50 and custom
single-object AP50:95. A wrong-class or low-IoU accepted detection is an FP and
leaves the true object as an FN. The original classifier comparison uses the
**same old 490 test images and same preprocessing** for both class heads. Directly
comparing old 76.53% with accuracy on a different detector test is invalid.

Test results are final reporting evidence. Do not tune repeatedly on test
failures while calling it an untouched benchmark; obtain new independent data
for further claims.

## Software webcam demonstration

Use the trained detector checkpoint with:

```powershell
.\.venv\Scripts\python.exe -B -m detection.webcam --checkpoint detection/best_detector.pth
```

The preview says **SOFTWARE reference / Laptop inference / FPGA not used**. Its
box, class, and score all come from the trained network. It does not use a
central fixed ROI, contours, color segmentation, or another detector. A worker
keeps only the newest pending frame; preview FPS, detection FPS, model latency,
and result age are shown. Stale boxes expire. Q/Escape quits; `--camera 1` selects
another camera. This app does not implement a hardware mode or silently claim
the FPGA is connected.

For a camera-free image check:

```powershell
.\.venv\Scripts\python.exe -B -m detection.webcam --checkpoint detection/best_detector.pth --image C:/path/to/photo.jpg --output detection/prediction.jpg
```

`WebcamPilot.cmd` is explicitly an older incomplete foreground-only experiment,
not the selected detector. It is retained for debugging; it cannot validate
object-presence quality. Synthetic fixture checkpoints are rejected by the app.

## Collect a genuine webcam benchmark

No personal-webcam data was supplied. The implementation does not claim measured
performance on your four physical objects. Capture complete frames and use
different sessions for training, validation, and final testing:

```powershell
.\.venv\Scripts\python.exe -B -m detection.capture --label rubiks_cube --instance my_cube --session bright_plain_day1 --lighting bright --background plain --orientation tilted
```

Space saves an unboxed positive candidate; **N** explicitly confirms that none
of the four objects is visible and saves a reviewed negative. Do not press N
when a supported object is present. Repeat with pens/books/mice, varied distance,
position, orientation and lighting. Keep a session name shared across classes
captured in the same setting; changing the object does not create an independent
scene. Do not capture hundreds of adjacent near-identical frames and then split
them randomly.

```powershell
.\.venv\Scripts\python.exe -B -m detection.review_queue --personal-only --output detection/data/personal_review_queue.jsonl
.\.venv\Scripts\python.exe -B -m detection.review --queue detection/data/personal_review_queue.jsonl --port 8766
```

Prepare a new frozen manifest after box review, retaining existing split roles.
Controlled brightness/mirror/rotation evaluation is provided by
`detection.robustness`; it measures sensitivity of the same photographs, not new
physical lighting, unseen backgrounds, or 3D orientations. Manual condition tags
remain unknown until actually supplied. An independent personal-object benchmark
and integer/FPGA testing remain later work.
