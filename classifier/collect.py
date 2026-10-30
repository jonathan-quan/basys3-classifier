"""Capture your physical objects in explicitly named independent sessions."""
import argparse
import json
import time
from pathlib import Path

import cv2

from . import CLASSES
from .common import write_json
from .webcam import crop_roi, open_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=CLASSES)
    parser.add_argument("--instance", required=True, help="Physical object ID, e.g. my_pen")
    parser.add_argument("--session", required=True, help="Shared session ID across classes captured in the same setting")
    parser.add_argument("--raw", default="data/personal_raw")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    for value in (args.instance, args.session):
        if not value or any(ch in value for ch in '/\\:') or value in (".", ".."):
            raise ValueError("Instance/session must be simple folder names")
    if args.interval < 0.5:
        raise ValueError("Use at least 0.5 seconds between automatically captured images")
    folder = Path(args.raw) / args.label / args.instance / args.session
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"source": "self-captured by project owner", "license": "user-owned; not licensed for redistribution",
                "instance": args.label + ":" + args.instance, "session": args.session}
    meta_path = folder / "session.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Existing session metadata differs; choose a new session")
    write_json(meta_path, metadata)
    camera = open_camera(args.camera)
    automatic, last_capture, count = False, 0.0, 0
    print("Space: save | A: toggle timed capture | Q: quit. Change orientation/distance; do not treat adjacent frames as independent sessions.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            roi, (left, top, size) = crop_roi(frame)
            view = frame.copy()
            cv2.rectangle(view, (left, top), (left + size, top + size), (0, 220, 0), 2)
            text = f"{args.label} | saved {count} | auto {'ON' if automatic else 'OFF'} | Space/A/Q"
            cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Collect personal object photos", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                automatic = not automatic
            now = time.monotonic()
            if key == 32 or (automatic and now - last_capture >= args.interval):
                destination = folder / f"{time.time_ns()}.jpg"
                if not cv2.imwrite(str(destination), roi, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Cannot save {destination}")
                last_capture, count = now, count + 1
            if cv2.getWindowProperty("Collect personal object photos", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {count} images to {folder.resolve()}")


if __name__ == "__main__":
    main()

