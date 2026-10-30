"""Fixed splits and matching training/webcam preprocessing."""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import CLASSES, INPUT_SIZE
from .common import file_hash


class Letterbox:
    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.pad(image, (INPUT_SIZE, INPUT_SIZE), method=Image.Resampling.BILINEAR,
                            color=(128, 128, 128), centering=(0.5, 0.5))


class NormalizePixels:
    def __call__(self, image):
        # Deliberately match future q_input = pixel - 128, scale = 1/128.
        array = np.array(image, dtype=np.float32, copy=True)
        return torch.from_numpy(array.transpose(2, 0, 1).copy()).sub_(128).div_(128)


def image_transform(training=False):
    operations = [Letterbox()]
    if training:
        operations += [
            transforms.RandomAffine(degrees=25, translate=(0.10, 0.10), scale=(0.85, 1.10),
                                    interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        ]
    return transforms.Compose(operations + [NormalizePixels()])


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Dataset manifest is empty")
    required = {"path", "label", "split", "group", "sha256", "pixel_sha256", "dhash", "source", "license"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest needs columns: {sorted(required)}")
    for row in rows:
        p = Path(row["path"])
        row["path"] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return rows


def audit_rows(rows, verify_files=True, require_splits=True):
    counts = {split: Counter() for split in ("train", "val", "test")}
    groups, images, pixels, paths = {}, {}, {}, set()
    for row in rows:
        label, split, group = row["label"], row["split"], row["group"]
        if label not in CLASSES or split not in counts or not group:
            raise ValueError(f"Invalid label/split/group: {row}")
        if not row["source"] or not row["license"]:
            raise ValueError("Source and license/provenance must be recorded")
        if row["path"] in paths:
            raise ValueError(f"Repeated file path: {row['path']}")
        paths.add(row["path"])
        if group in groups and groups[group] != split:
            raise ValueError(f"Leakage: group {group} appears in multiple splits")
        groups[group] = split
        for index, key in ((images, "sha256"), (pixels, "pixel_sha256")):
            identity = row[key]
            if not identity:
                raise ValueError(f"Missing {key}")
            if identity in index:
                previous = index[identity]
                if previous != (split, label):
                    raise ValueError("Exact duplicate crosses splits or has conflicting labels")
            index[identity] = (split, label)
        if verify_files and file_hash(row["path"]) != row["sha256"]:
            raise ValueError(f"Image changed since preparation: {row['path']}")
        counts[split][label] += 1
    if require_splits:
        for split in counts:
            missing = set(CLASSES) - set(counts[split])
            if missing:
                raise ValueError(f"{split} lacks classes: {sorted(missing)}")
    return {key: dict(value) for key, value in counts.items()}


def near_duplicate_pairs(rows, distance=3, limit=200):
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    pairs = []
    for entries in by_class.values():
        for i, left in enumerate(entries):
            for right in entries[i + 1:]:
                if left["split"] == right["split"]:
                    continue
                bits = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
                if bits <= distance:
                    pairs.append({"left": left["path"], "right": right["path"], "distance": bits,
                                  "left_split": left["split"], "right_split": right["split"]})
                    if len(pairs) >= limit:
                        return pairs
    return pairs


class ManifestDataset(Dataset):
    def __init__(self, rows, split, training=False, cache=False):
        self.rows = [row for row in rows if row["split"] == split]
        self.transform = image_transform(training)
        self.cached = None
        if cache:
            self.cached = []
            for row in self.rows:
                with Image.open(row["path"]) as image:
                    self.cached.append(Letterbox()(image))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.cached is not None:
            tensor = self.transform(self.cached[index])
        else:
            with Image.open(row["path"]) as image:
                tensor = self.transform(image)
        return tensor, CLASSES.index(row["label"]), index


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def source_balanced_weights(rows):
    """Equal class mass, then equal source mass within each class."""
    frequencies = Counter((row["label"], row["source"]) for row in rows)
    sources = {label: {row["source"] for row in rows if row["label"] == label} for label in CLASSES}
    if any(not value for value in sources.values()):
        raise ValueError("Source balancing requires all four classes")
    return [1.0 / (len(CLASSES) * len(sources[row["label"]]) * frequencies[row["label"], row["source"]])
            for row in rows]

