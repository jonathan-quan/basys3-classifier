"""Live webcam view of the improved single-object DETECTOR (laptop software inference).

This is NOT FPGA inference: the box detector runs on the laptop with PyTorch.
The FPGA classifier demo is FPGARecognition.cmd.

Keys: Q / Esc quit, T toggle showing low-score guesses.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from classifier.common import choose_device
from . import ROOT
from .improve import letterbox, load_improved, tensorize

NAMES = ("Rubik's Cube", "Pen", "Book", "Computer Mouse")
COLORS = ((0, 200, 255), (255, 160, 0), (80, 220, 80), (230, 90, 230))


def outline(img, text, org, scale=.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(ROOT / "runs" / "webcam_ft_v1"), help="Run folder with best_detector.pth and summary.json")
    p.add_argument("--camera", type=int, default=0); p.add_argument("--video", help="Play a video file instead of the webcam")
    p.add_argument("--save", help="Write annotated frames to this video file instead of showing a window")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    run = Path(a.run); device = choose_device(a.device)
    model = load_improved(run / "best_detector.pth", device)
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    threshold = summary.get("score_threshold_selected_on_validation", .5)
    size = model.cfg["size"]
    cap = cv2.VideoCapture(a.video) if a.video else cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not a.video and not cap.isOpened():
        cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam/video; close other camera apps and check Windows camera permissions")
    writer = None; show_low = False; times = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lb, _ = letterbox(image, size)
        start = time.perf_counter()
        with torch.inference_mode():
            out = model(tensorize(lb)[None].to(device))
        times.append(time.perf_counter() - start); times = times[-30:]
        prob = out["class_logits"].softmax(1)[0]; cls = int(prob.argmax())
        presence = float(out["presence_logits"].sigmoid()[0]); score = presence * float(prob[cls])
        cx, cy, bw, bh = out["box_cxcywh"][0].tolist()
        s = min(size / w, size / h); nw, nh = round(w * s), round(h * s); left, top = (size - nw) // 2, (size - nh) // 2
        x0 = int(np.clip(((cx - bw / 2) * size - left) / s, 0, w)); x1 = int(np.clip(((cx + bw / 2) * size - left) / s, 0, w))
        y0 = int(np.clip(((cy - bh / 2) * size - top) / s, 0, h)); y1 = int(np.clip(((cy + bh / 2) * size - top) / s, 0, h))
        scale = w / 960
        if score >= threshold:
            cv2.rectangle(frame, (x0, y0), (x1, y1), COLORS[cls], max(2, int(4 * scale)))
            ly = y0 - 10 if y0 > 140 * scale else y0 + int(34 * scale)  # keep the label clear of the header text
            outline(frame, f"{NAMES[cls]} {score:.0%}", (x0 + 6, ly), .9 * scale, COLORS[cls])
            status = f"Detected: {NAMES[cls]}  score {score:.0%}"
        else:
            if show_low:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 1)
            status = f"No object (best guess {NAMES[cls]} {score:.0%} < threshold {threshold:.0%})"
        outline(frame, "LAPTOP software detector (not FPGA)", (12, int(36 * scale)), .8 * scale, (80, 230, 255))
        outline(frame, status, (12, int(76 * scale)), .8 * scale)
        outline(frame, f"model {1000 * np.mean(times):.1f} ms | Q quits, T shows low-score guesses", (12, int(112 * scale)), .6 * scale)
        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            writer.write(frame); continue
        cv2.imshow("Single-object detector (laptop)", frame); key = cv2.waitKey(1) & 0xff
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("t"), ord("T")):
            show_low = not show_low
    cap.release()
    if writer:
        writer.release()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
