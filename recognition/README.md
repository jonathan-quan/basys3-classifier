# Webcam recognition comparison

Double-click [CompareRecognition.cmd](C:/Users/ndaqu/Documents/Basys3Classifier/CompareRecognition.cmd).
Focus the video window to use its keyboard controls:

- **1** — original classifier (`best_model.pth`). Starts in this mode.
- **2** — available detector's classification head only.
- **Tab / Space** — switch models.
- **F** — toggle native framing versus the same full webcam frame for both.
- **Q / Escape** — quit.

The display shows the active model, predicted object, and its maximum four-class
softmax score. These are uncalibrated class scores, not measured live accuracy
or probabilities that an unknown object belongs to the supported set. Both
models still choose among cube, pen, book, and computer mouse on every frame.
There are **no bounding boxes**. The detector's box and presence branches do not
execute; objectness and joint detection scores are not used.

Native framing preserves the original webcam's central 75% square crop followed
by its existing `image_transform(False)`. No rectangle is drawn around that crop;
hold the object near the center. The detector uses its exact full-frame RGB
letterbox and `(pixel−128)/128` preprocessing followed by `model.classify()`.
Press **F** for a comparison using the full frame for both models; each still
uses its own exact tensor preprocessing. Input framing alone can affect scores.

The launcher uses the project's existing Python environment and automatically
uses CUDA if available. A background inference worker keeps the video responsive
and discards results from the previous model when switching. It does not capture
training data, update weights, write checkpoints, or launch training.

The original classifier loads independently. A missing, corrupt, synthetic, or
incompatible detector is reported and does not prevent original-model testing.
A compatible but unfinished/pilot detector checkpoint can be inspected, with an
explicit status label. Checkpoint paths and status are printed in the console.
The app first checks a published detector checkpoint, then available run
checkpoints, preferring completed runs and the newest file. This is a convenient
experimental default, not a new model-quality selection claim.

At creation, the latest available detector run is `reviewed_cosine`. Its training
run finished, but its held-out detector evaluation and final detector selection
are unfinished. Its selected checkpoint is from epoch 2. The app checks whether
the shared classification parameters are identical to the original classifier
and displays that fact when true. Do not interpret a newer checkpoint filename
as an improvement in recognition accuracy. No model was retrained for this app.

If camera 0 is unavailable, close other camera programs, check Windows camera
permissions, or use camera 1 from PowerShell:

```powershell
Set-Location 'C:\Users\ndaqu\Documents\Basys3Classifier'
.\CompareRecognition.cmd --camera 1
```

To choose an explicit detector, use `--detector C:/path/to/best_detector.pth`.
`--image` provides a camera-free functional check. Such checks do not establish
accuracy on physical objects or prove the live webcam works on this laptop.
