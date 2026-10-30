"""Extract frames from the owner's webcam clips and propose DRAFT boxes.

Draft boxes come from an OpenCV CSRT tracker (started from one manually placed box
per clip) refined by a colour mask against the red demo surface that excludes skin.
They are proposals only: nothing here is ground truth until the owner approves it
in the review grid (detection/webcam_review.py).

Usage (outputs to --out): python webcam_drafts.py --videos DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("rubiks_cube", "pen", "book", "mouse")
W, H = 640, 360
STEP = 10           # keep every 10th frame (~3 per second)
VAL_FRACTION = .2   # last 20% of each clip (by time) is validation

# One clip per object, session 1 (training session). Initial boxes are x0,y0,x1,y1
# in 640x360 pixels on the first frame, placed by eye on a gridded preview.
CLIPS = {
    "pen":   dict(file="WIN_20261029_14_40_45_Pro.mp4", label=1, init=(150, 120, 503, 178)),
    "cube":  dict(file="WIN_20261029_14_55_43_Pro.mp4", label=0, init=(208, 12, 328, 138)),
    "mouse": dict(file="WIN_20261029_15_01_32_Pro.mp4", label=3, init=(158, 52, 455, 235)),
    "book":  dict(file="WIN_20261029_19_03_18_Pro.mp4", label=2, init=(182, 55, 495, 295)),
    "empty": dict(file="WIN_20261029_15_03_14_Pro.mp4", label=-1, init=None),
}


def masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    reddish = (h <= 15) | (h >= 168)
    surface = reddish & (s >= 120) & (v >= 40)
    skin = (h <= 25) & (s >= 35) & (s < 150) & (v >= 110) & ~surface
    foreground = ~surface & ~skin
    return foreground.astype(np.uint8), skin.astype(np.uint8)


def refine(frame, box):
    """Tight box of non-surface, non-skin pixels around the tracked region."""
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    ex0, ey0 = max(0, int(x0 - .3 * bw)), max(0, int(y0 - .3 * bh))
    ex1, ey1 = min(W, int(x1 + .3 * bw)), min(H, int(y1 + .3 * bh))
    fg, skin = masks(frame[ey0:ey1, ex0:ex1])
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = []
    tx0, ty0, tx1, ty1 = x0 - ex0, y0 - ey0, x1 - ex0, y1 - ey0
    big = max([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)] or [0])
    for i in range(1, n):
        x, y, w, hh, a = stats[i]
        overlaps = not (x > tx1 or x + w < tx0 or y > ty1 or y + hh < ty0)
        if overlaps and a >= max(40, .08 * big):
            keep.append((x, y, x + w, y + hh))
    hand = float(skin.mean())
    if not keep:
        return None, hand
    k = np.array(keep)
    return (int(k[:, 0].min() + ex0), int(k[:, 1].min() + ey0), int(k[:, 2].max() + ex0), int(k[:, 3].max() + ey0)), hand


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def run(videos, out, session="s1"):
    out.mkdir(parents=True, exist_ok=True); (out / "frames").mkdir(exist_ok=True)
    rows = []
    for name, clip in CLIPS.items():
        cap = cv2.VideoCapture(str(videos / clip["file"]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = None; box = clip["init"]; index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            if box is not None:
                if tracker is None:
                    tracker = cv2.TrackerCSRT_create(); tracker.init(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
                    tracked = box
                else:
                    ok_t, (x, y, w, h) = tracker.update(frame)
                    tracked = (int(x), int(y), int(x + w), int(y + h)) if ok_t else None
            if index % STEP == 0:
                fid = f"webcam_{session}_{name}_{index:05d}"
                path = out / "frames" / f"{fid}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                row = {"id": fid, "file": f"frames/{fid}.jpg", "clip": clip["file"], "session": session, "frame_index": index,
                       "label": clip["label"], "present": int(clip["label"] >= 0), "bbox": None, "flags": []}
                if clip["label"] >= 0:
                    flags = []
                    if tracked is None:
                        flags.append("tracker_lost"); final = None; hand = 0.
                    else:
                        seg, hand = refine(frame, tracked)
                        if seg is None:
                            final = tracked; flags.append("mask_failed")
                        elif not (area(tracked) / 4 <= area(seg) <= area(tracked) * 4):
                            final = tracked; flags.append("mask_disagrees")
                        else:
                            final = seg
                            tracker = cv2.TrackerCSRT_create()  # re-anchor to the refined box (handles rotation/scale)
                            tracker.init(frame, (seg[0], seg[1], seg[2] - seg[0], seg[3] - seg[1]))
                    if hand > .04:
                        flags.append("hand_in_box_area")
                    if final is not None:
                        x0, y0, x1, y1 = [min(max(v, 0), m) for v, m in zip(final, (W, H, W, H))]
                        if x1 - x0 >= 4 and y1 - y0 >= 4:
                            row["bbox"] = [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)]
                    row["flags"] = flags
                row["split"] = "val" if index >= (1 - VAL_FRACTION) * total else "train"
                rows.append(row)
            index += 1
        cap.release()
        print(name, sum(r["clip"] == clip["file"] for r in rows), "frames", flush=True)
    with open(out / "drafts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); run(Path(a.videos), Path(a.out))
