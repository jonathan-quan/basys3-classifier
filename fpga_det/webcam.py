"""Laptop webcam preview; the box detector runs ONLY on the Basys 3 FPGA (no laptop neural inference).

Keys: Q / Esc quit, T toggle showing low-score guesses.
"""
import argparse
import json
import threading
import time

import cv2
import numpy as np
from PIL import Image

from detection.improve import letterbox
from fpga.webcam import text
from . import PROJECT
from .client import SIDE, DetectorClient

COLORS = ((0, 200, 255), (255, 160, 0), (80, 220, 80), (230, 90, 230))
SUMMARY = PROJECT / "detection" / "runs" / "webcam_ft_v1" / "summary.json"
WINDOW = "Basys 3 - REAL FPGA box detector"


class Worker:
    """Sends the most recent frame to the FPGA; frames that arrive while it is busy are dropped."""
    def __init__(self, port):
        self.port = port; self.client = None; self.condition = threading.Condition(); self.pending = None; self.result = None
        self.error = "Connecting to FPGA..."; self.closed = False; self.count = 0; self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()

    def submit(self, frame, captured_at):
        with self.condition:
            self.pending = (frame.copy(), captured_at); self.condition.notify()

    def run(self):
        while not self.closed:
            try:
                if self.client is None:
                    self.client = DetectorClient(self.port); self.error = None
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closed)
                    if self.closed:
                        break
                    frame, timestamp = self.pending; self.pending = None
                h, w = frame.shape[:2]
                lb, _ = letterbox(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), SIDE)
                result = self.client.detect_letterboxed(lb)
                result.update(captured_at=timestamp, frame_size=(w, h)); self.result = result; self.count += 1
            except Exception as error:
                self.error = str(error); self.result = None
                if self.client:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                    self.client = None
                with self.condition:
                    if not self.closed:
                        self.condition.wait(timeout=2)
        if self.client:
            self.client.close()

    def close(self):
        with self.condition:
            self.closed = True; self.condition.notify()
        self.thread.join(timeout=5)


def frame_box(result):
    """Letterbox (cx, cy, w, h) in [0,1] -> pixel corners in the original frame (same as detection.webcam_live)."""
    w, h = result["frame_size"]; cx, cy, bw, bh = result["box_cxcywh_letterbox"]
    s = min(SIDE / w, SIDE / h); nw, nh = round(w * s), round(h * s); left, top = (SIDE - nw) // 2, (SIDE - nh) // 2
    x0 = int(np.clip(((cx - bw / 2) * SIDE - left) / s, 0, w)); x1 = int(np.clip(((cx + bw / 2) * SIDE - left) / s, 0, w))
    y0 = int(np.clip(((cy - bh / 2) * SIDE - top) / s, 0, h)); y1 = int(np.clip(((cy + bh / 2) * SIDE - top) / s, 0, h))
    return x0, y0, x1, y1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port"); p.add_argument("--camera", type=int, default=0)
    a = p.parse_args()
    threshold = json.loads(SUMMARY.read_text())["score_threshold_selected_on_validation"]
    camera = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release(); camera = cv2.VideoCapture(a.camera)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam; close other camera apps and check Windows camera permissions")
    worker = Worker(a.port); frames = 0; started = time.perf_counter(); show_low = False
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped delivering frames")
            now = time.perf_counter(); frames += 1; worker.submit(frame, now); result = worker.result
            scale = frame.shape[1] / 960
            if result and now - result["captured_at"] < 3:
                x0, y0, x1, y1 = frame_box(result); cls = result["class_id"]
                if result["score"] >= threshold:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), COLORS[cls], max(2, int(4 * scale)))
                    cv2.putText(frame, f"{result['name']} {result['score']:.0%}", (x0 + 6, max(y0 + int(30 * scale), int(200 * scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, .9 * scale, COLORS[cls], 2, cv2.LINE_AA)
                    status, color = f"Detected: {result['name']}  score {result['score']:.0%}", (50, 240, 100)
                else:
                    if show_low:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 1)
                    status, color = f"No object (best guess {result['name']} {result['score']:.0%} < {threshold:.0%})", (255, 255, 255)
                text(frame, status, 62, color, .7)
                text(frame, f"FPGA {result['inference_ms']:.0f} ms | UART round trip {result['roundtrip_ms']:.0f} ms", 91)
            else:
                text(frame, "Hardware detection unavailable" if worker.error else "Waiting for FPGA detection...", 62, (20, 160, 255), .6)
            text(frame, "BASYS 3 FPGA BOX DETECTOR - no laptop neural inference", 25, (50, 240, 100), .53)
            text(frame, f"Preview {frames / max(.01, now - started):.1f} FPS | FPGA updates {worker.count / max(.01, now - worker.started):.1f} FPS", 117)
            text(frame, "Q/Escape quits | T shows low-score guesses | box is from the frame sent to the FPGA", 143, scale=.45)
            if worker.error:
                text(frame, worker.error[:105], 169, (20, 160, 255), .43)
            cv2.imshow(WINDOW, frame); key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("t"), ord("T")):
                show_low = not show_low
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        worker.close(); camera.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
