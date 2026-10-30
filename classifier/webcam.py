"""Interactive laptop webcam inference; the FPGA is not used."""
import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image

from . import DISPLAY_NAMES
from .common import choose_device, seed_everything
from .data import image_transform
from .evaluate import load_checkpoint


def open_camera(index):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. Check Windows camera permissions, close other camera apps, or try --camera 1.")
    return camera


def crop_roi(frame, fraction=0.75):
    height, width = frame.shape[:2]
    size = max(1, int(min(height, width) * fraction))
    left, top = (width - size) // 2, (height - size) // 2
    return frame[top:top + size, left:left + size].copy(), (left, top, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--image", help="Optional still image for headless inference / camera-free testing")
    args = parser.parse_args()
    seed_everything(42)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("smoke_test"):
        raise ValueError("This checkpoint is a synthetic smoke test, not a usable object classifier")
    transform = image_transform(False)

    def predict(image):
        with torch.inference_mode():
            scores = model(transform(image).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().numpy()
        return scores

    if args.image:
        with Image.open(args.image) as image:
            scores = predict(image)
        print({name: float(score) for name, score in zip(DISPLAY_NAMES, scores)})
        print("Prediction:", DISPLAY_NAMES[int(scores.argmax())])
        return
    print("Put ONE object inside the square. Q/Escape quits. Softmax scores are not calibrated probabilities or unknown rejection.")
    camera = open_camera(args.camera)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            roi, (left, top, size) = crop_roi(frame)
            start = time.perf_counter()
            scores = predict(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
            elapsed = time.perf_counter() - start
            prediction = int(scores.argmax())
            cv2.rectangle(frame, (left, top), (left + size, top + size), (0, 220, 0), 2)
            lines = [f"{DISPLAY_NAMES[prediction]}   {scores[prediction]:.1%} score",
                     f"Laptop inference: {elapsed * 1000:.1f} ms ({device})",
                     "One object in box | Q quits | closed-set classifier"]
            lines += [f"{name}: {value:.1%}" for name, value in zip(DISPLAY_NAMES, scores)]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 26 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (12, 26 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Basys3 project - laptop classifier preview", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Basys3 project - laptop classifier preview", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

