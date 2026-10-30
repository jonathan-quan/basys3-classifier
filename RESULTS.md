# Training and evaluation results — October 28, 2026

**The Python pipeline is implemented and tested. The selected model achieved 76.53% on the reserved public-image test set. This is not yet a high-accuracy classifier, and accuracy on your own physical objects has not been measured.**

## Environment

NVIDIA RTX 5060 Laptop GPU (about 8 GB VRAM), Python 3.11.9, PyTorch 2.14.1+cu130, CUDA 13.0. Actual GPU forward/backward execution and all training runs succeeded. Dependencies are isolated in `.venv`; see `environment.json` and `requirements-lock.txt`.

## Data and split

Acquired 1,000 images per class before curation. The active dataset contains 3,884 images after 116 pretraining exclusions. All 2,000 pen/book thumbnails and deterministic 80-image samples for cubes/mice were reviewed. This review does not certify every image or label. Originals and per-image attribution records are preserved.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| rubiks_cube | 923 | 43 | 34 | 1000 |
| pen | 664 | 136 | 148 | 948 |
| book | 666 | 132 | 138 | 936 |
| mouse | 681 | 149 | 170 | 1000 |
| **Total** | **2,934** | **460** | **490** | **3,884** |

Sources: [Google Open Images crowdsourced extension](https://storage.googleapis.com/openimages/web/extended.html), using positive human-verified labels and explicit per-image CC BY 4.0 metadata; and the [SAS cube collection](https://github.com/sascommunities/iot-image-classification-rubiks-cubes), whose README permits use of its photos and whose project is Apache-2.0. Exact attribution is in `data/attributions.csv`; the SAS LICENSE and README are preserved in `data/sources/sas`.

All photos credited to one Google author remain in one split, and all SAS photos form one group. Exact decoded duplicates and same-class near-duplicate hash candidates are merged before splitting. There are 1,251 final groups and no cross-split exact or detected near duplicates. Unknown capture sessions and physical instances cannot be fully audited from public metadata.

The cube collection consists of 848 correlated SAS photos and 152 Google photos. SAS predominantly shows a 2×2 cube; Google includes 3×3 puzzles. Keeping the SAS collection together leaves only 43 validation and 34 test cube photos. Thus 1,000 cube images do not mean 1,000 independent scenes. Books include bound notebooks.

## Measured experiment comparison

All runs used the same frozen manifest. The plain baseline used ordinary shuffled sampling without batch normalization. All other runs used identical settings: source/class balancing, training-only BN folded into saved convolutions, AdamW at 0.002, batch 64, early-stop patience 15, and maximum 80 epochs. The plain baseline had a 120-epoch limit and patience 20. Therefore only comparisons among the balanced runs hold the training recipe fixed.

| Run | Inference parameters | Epochs / selected epoch | Training seconds | Validation accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|
| baseline | 3,252 | 110 / 90 | 425.5 | 62.17% | Reserved / not evaluated |
| baseline_balanced | 3,252 | 64 / 49 | 203.5 | 62.83% | 67.96% |
| context_balanced | 6,084 | 61 / 46 | 248.7 | 73.48% | 76.53% |
| context2_balanced | 8,916 | 40 / 25 | 201.2 | 72.83% | Reserved / not evaluated |
| standard_balanced | 20,100 | 80 / 75 | 343.6 | 70.87% | Reserved / not evaluated |

**Selection was frozen using validation accuracy before any real test evaluation**, with fewer parameters as the tie-breaker. See `model_selection.json`. The selected context model and the strongest original-size baseline were then each evaluated once on the same test set. No model changes were made from test results. The other candidates remain unevaluated on test.

Selected-model training took **248.7 seconds (4 minutes 9 seconds)**. Total training-loop time across all five runs was **1422.5 seconds (about 23 minutes 42 seconds)**. These measurements include epochs, validation, and checkpoint writes, but exclude installation, downloads, curation, cache preparation, and final plotting/evaluation.

## Selected model and size

The requested 3,252-parameter baseline underperformed and is preserved as `baseline_3252.pth`. The selected `best_model.pth` is a **6,084-parameter depthwise-separable CNN with one additional stride-1 context block**. The change was compared and its tradeoffs explained before selection.

- INT8 weight payload estimate: **5,824 bytes**.
- INT32 bias payload estimate: **1,040 bytes**.
- FP32 inference-parameter payload: **24,336 bytes**.
- Actual saved checkpoint file: **31,605 bytes** (includes metadata/serialization).
- Weighted convolution/FC work: **871,680 MACs/image**, about 12.7% above the baseline.
- Estimated peak live INT8 activations: **46,080 bytes (45 KiB)**, unchanged from the baseline.
- Receptive field: **63×63**, compared with 31×31 for the baseline.
- Training used 6,340 learnable parameters with BN. Folding removes BN operators and leaves exactly 6,084 inference parameters.

The two-context-block model used 8,916 inference parameters / 970,176 MACs and reached 72.83% validation accuracy. The ordinary-convolution model used 20,100 parameters / 2,322,624 MACs and reached 70.87%. Their extra size did not improve validation performance over the selected 6,084-parameter model on this split.

**This checkpoint is FP32. INT8 accuracy, FPGA resource use, FPGA timing, and physical-object webcam accuracy have not been measured. No Verilog or FPGA synthesis was performed.**

## Held-out results for the selected model

Overall test accuracy: **76.53% (375/490)**. Balanced accuracy: **76.33%**. Macro F1: **0.747**.
The approximate 95% Wilson accuracy interval is 72.6%–80.1%; it assumes independent examples and may be optimistic for correlated public photos.

| Class | Correct / examples | Accuracy within class (recall) | Precision | F1 |
|---|---:|---:|---:|---:|
| rubiks_cube | 27 / 34 | 79.41% | 64.29% | 0.711 |
| pen | 99 / 148 | 66.89% | 78.57% | 0.723 |
| book | 92 / 138 | 66.67% | 69.70% | 0.681 |
| mouse | 157 / 170 | 92.35% | 82.63% | 0.872 |

![Selected-model confusion matrix](C:/Users/ndaqu/Documents/Basys3Classifier/runs/context_balanced/test_confusion_matrix.png)

![Selected-model learning curves](C:/Users/ndaqu/Documents/Basys3Classifier/runs/context_balanced/learning_curves.png)

Per-image predictions, raw/normalized confusion matrices, per-class metrics, training history, and saved configs are in the corresponding `runs/` directories. Confusion-matrix rows are actual classes and columns are predicted classes. Per-class accuracy is recall, not one-vs-rest accuracy.

## Webcam application

Double-click `Webcam.cmd`, or run from this project folder:

```powershell
.\.venv\Scripts\python.exe -m classifier.webcam --checkpoint best_model.pth
```

Put one object inside the square; Q/Escape exits. Predictions run on the laptop. The camera itself has not been opened or validated during this task. The still-image application loaded the selected checkpoint and produced scores successfully; its one validation-book check was misclassified as a cube (84.35% softmax score). This is a functional application check, not evidence of correct classification, and illustrates why high scores are not guarantees. Softmax scores do not establish unknown-object rejection.

## Assessment and next step

The training/evaluation software works, but the high-accuracy goal is not met. Pens and books are the main measured weaknesses. More parameters alone were not enough. The next priority is a dataset of your actual four objects, with varied framing, lighting, backgrounds, orientation, and distance, separated by capture session. Reserve entirely new sessions for the final personal-object test.

Use `classifier.collect` for guided capture or `classifier.import_images` for explicitly labeled photos. For an evaluation-only personal collection, use `classifier.prepare --test-only` and `classifier.evaluate --external`; see README. Public category-recognition accuracy should not be treated as a measurement on your own four physical objects.

Nine automated tests passed: model counts/MACs and gradients, pixel normalization/padding, confusion metrics, group/duplicate leakage rejection, related-image grouping, optimizer learning, source balancing, BN-fold equivalence, and an explicitly synthetic end-to-end fixture test including rejection of an overlapping external benchmark. Synthetic fixture metrics are not used in any reported object accuracy.

## Reproduce the selected recipe

```powershell
.\.venv\Scripts\python.exe -m classifier.train --manifest data/manifest.csv --architecture context --output runs/context_reproduction --epochs 80 --patience 15 --workers 0 --balance-sources --batch-norm
.\.venv\Scripts\python.exe -m classifier.evaluate --checkpoint runs/context_reproduction/best_model.pth --manifest data/manifest.csv --split test --output runs/context_reproduction
```

Use validation to make choices. The current test set has now been evaluated; do not use its errors repeatedly to tune models while calling it an untouched final holdout. For further development, create a new independent personal-object holdout.
