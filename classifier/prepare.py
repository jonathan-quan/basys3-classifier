"""Build a reproducible group-disjoint manifest from raw labeled images."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import CLASSES
from .common import file_hash, write_json
from .data import audit_rows, near_duplicate_pairs

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_fingerprints(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if min(image.size) < 32:
            raise ValueError(f"Image smaller than 32 pixels: {path}")
        payload = f"{image.width},{image.height}:".encode() + image.tobytes()
        tiny = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = (tiny[:, 1:] > tiny[:, :-1]).flatten()
        value = sum(int(bit) << i for i, bit in enumerate(bits))
        return hashlib.sha256(payload).hexdigest(), f"{value:016x}"


def scan_raw(root):
    root = Path(root).resolve()
    records = []
    for label in CLASSES:
        for path in sorted((root / label).rglob("*")):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(root / label)
            if len(relative.parts) < 3:
                raise ValueError(f"Expected raw/{label}/INSTANCE/SESSION/image.jpg: {path}")
            metadata_path = path.parent / "session.json"
            if not metadata_path.is_file():
                raise ValueError(f"Missing provenance: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("source") or not metadata.get("license"):
                raise ValueError(f"Missing source/license in {metadata_path}")
            pixel_hash, dhash = image_fingerprints(path)
            records.append({"path": str(path.resolve()), "label": label,
                            "instance": metadata.get("instance", f"{label}:{relative.parts[0]}"),
                            "session": metadata.get("session", relative.parts[1]),
                            "source": metadata["source"], "license": metadata["license"],
                            "sha256": file_hash(path), "pixel_sha256": pixel_hash, "dhash": dhash,
                            "synthetic": str(bool(metadata.get("synthetic", False))).lower()})
    if not records:
        raise ValueError(f"No labeled images found under {root}")
    return records


def assign_groups(records, mode="session", seed=42):
    # Connected components keep session/author, exact duplicates, and optionally
    # physical instances together. This can correctly reveal an unsplittable dataset.
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(j)] = find(i)

    tokens, seen_pixels = {}, {}
    for i, row in enumerate(records):
        if row["pixel_sha256"] in seen_pixels:
            previous = seen_pixels[row["pixel_sha256"]]
            if records[previous]["label"] != row["label"]:
                raise ValueError("Same image has conflicting class labels")
        seen_pixels[row["pixel_sha256"]] = i
        keys = [("session", row["session"]), ("pixels", row["pixel_sha256"])]
        if mode == "instance-session":
            keys.append(("instance", row["instance"]))
        for key in keys:
            if key in tokens:
                union(i, tokens[key])
            tokens[key] = i
    # Conservatively merge visually similar same-class images before splitting.
    # A hash collision only combines groups; it never moves copies across sets.
    by_class = defaultdict(list)
    for i, row in enumerate(records):
        by_class[row["label"]].append((i, int(row["dhash"], 16)))
    for entries in by_class.values():
        for position, (i, left) in enumerate(entries):
            for j, right in entries[position + 1:]:
                if (left ^ right).bit_count() <= 3:
                    union(i, j)
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    for indices in components.values():
        identity = hashlib.sha256("|".join(sorted(records[i]["sha256"] for i in indices)).encode()).hexdigest()[:16]
        for i in indices:
            records[i]["group"] = identity
    return records


def choose_split(records, seed=42):
    groups = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    keys = sorted(groups)
    for label in CLASSES:
        count = sum(any(row["label"] == label for row in group) for group in groups.values())
        if count < 3:
            raise ValueError(f"{label} has only {count} independent groups; at least 3 are required")
    target = np.array([0.70, 0.15, 0.15])
    counts = np.array([[sum(row["label"] == label for row in groups[key]) for label in CLASSES] for key in keys])
    totals = counts.sum(axis=0)
    best_score, best = float("inf"), None
    rng = np.random.default_rng(seed)
    # Fixed seed and all four class distributions, without using model performance.
    for _ in range(4000):
        assignment = rng.choice(3, size=len(keys), p=target)
        distribution = np.array([counts[assignment == split].sum(axis=0) for split in range(3)])
        if np.any(distribution == 0):
            continue
        score = np.square(distribution / totals - target[:, None]).sum()
        if score < best_score:
            best_score, best = score, assignment.copy()
    if best is None:
        raise ValueError("Cannot form three class-complete group-disjoint splits; add independent sessions")
    mapping = {key: ("train", "val", "test")[int(split)] for key, split in zip(keys, best)}
    for row in records:
        row["split"] = mapping[row["group"]]
    return records


def contact_sheet(rows, path, title="Dataset review", count=80):
    chosen = list(rows)
    random.Random(42).shuffle(chosen)
    chosen = chosen[:count]
    columns, width, height = 8, 128, 150
    canvas = Image.new("RGB", (columns * width, 40 + ((len(chosen) + columns - 1) // columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for i, row in enumerate(chosen):
        x, y = (i % columns) * width, 40 + (i // columns) * height
        with Image.open(row["path"]) as image:
            thumb = ImageOps.pad(ImageOps.exif_transpose(image).convert("RGB"), (120, 120), color=(128, 128, 128))
        canvas.paste(thumb, (x + 4, y))
        draw.text((x + 4, y + 121), f"{i}: {Path(row['path']).stem[:15]}", fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def prepare(raw, output, group_by="session", seed=42, exclude=None, test_only=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Do not overwrite a frozen split: {output}")
    rows = scan_raw(raw)
    excluded = []
    if exclude:
        with Path(exclude).open(newline="", encoding="utf-8-sig") as handle:
            exclusions = list(csv.DictReader(handle))
        blocked = {str(Path(item["path"]).resolve()): item.get("reason", "manual review") for item in exclusions}
        excluded = [{"path": row["path"], "reason": blocked[row["path"]]} for row in rows if row["path"] in blocked]
        rows = [row for row in rows if row["path"] not in blocked]
    rows = assign_groups(rows, group_by, seed)
    if test_only:
        for row in rows:
            row["split"] = "test"
    else:
        rows = choose_split(rows, seed)
    counts = audit_rows(rows, require_splits=not test_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    near = near_duplicate_pairs(rows)
    report = {"seed": seed, "group_by": group_by, "counts": counts,
              "groups": len(set(row["group"] for row in rows)),
              "grouping": "Connected components of session/author, exact decoded pixels, and same-class dHash distance <=3; optional instance grouping",
              "excluded_after_visual_review": excluded,
              "test_only": test_only,
              "near_duplicate_candidates": near, "near_duplicate_list_capped_at": 200,
              "note": "Hash checks cannot establish absence of all visually related images. Review provenance and sessions."}
    write_json(output.with_suffix(".audit.json"), report)
    for label in CLASSES:
        contact_sheet([row for row in rows if row["label"] == label],
                      output.parent / "review" / f"{label}.jpg", label)
    print(json.dumps(report, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="data/manifest.csv")
    parser.add_argument("--group-by", choices=("session", "instance-session"), default="session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", help="CSV with path,reason columns from visual curation")
    parser.add_argument("--test-only", action="store_true", help="Prepare a separate all-test benchmark; never use it for training")
    args = parser.parse_args()
    prepare(args.raw, args.output, args.group_by, args.seed, args.exclude, args.test_only)


if __name__ == "__main__":
    main()
