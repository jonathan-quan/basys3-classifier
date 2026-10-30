"""Acquire a documented public-image subset; never use machine-only labels.

Sources: Open Images crowdsourced extension (per-image CC BY 4.0 metadata),
and SAS's explicitly reusable cube photographs (Apache-2.0 project).
This script does not fetch or execute third-party code or pretrained models.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

LABEL_IDS = {"/m/0h0b5": "rubiks_cube", "/m/0k1tl": "pen", "/m/0bt_c3": "book", "/m/020lf": "mouse"}
OI_BASE = "https://storage.googleapis.com/open-images-extended/"
OI_SOURCE = "https://storage.googleapis.com/openimages/web/extended.html"
SAS_SOURCE = "https://github.com/sascommunities/iot-image-classification-rubiks-cubes"
SAS_COMMIT = "273f965009533aa1877bb283788ee86d079ad1a5"
SAS_BASE = f"https://raw.githubusercontent.com/sascommunities/iot-image-classification-rubiks-cubes/{SAS_COMMIT}/"


def json_file(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Basys3Classifier-EducationalDataset/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            temporary.replace(destination)
            return destination
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fingerprint(path_or_bytes):
    stream = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(stream) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if min(image.size) < 32:
            raise ValueError("Image too small")
        return hashlib.sha256(f"{image.width},{image.height}:".encode() + image.tobytes()).hexdigest()


def acquire(root, target=1000, seed=42, workers=8):
    root = Path(root).resolve()
    sources, raw = root / "sources", root / "raw"
    labels_path = download(OI_BASE + "extended-crowdsourced-image-labels.csv", sources / "oi_labels.csv")
    positives = defaultdict(set)
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["LabelName"] in LABEL_IDS and row["Confidence"] == "1":
                positives[row["ImageID"]].add(LABEL_IDS[row["LabelName"]])
    candidates = {image_id: next(iter(labels)) for image_id, labels in positives.items() if len(labels) == 1}
    print("Unambiguous human-verified candidates:", dict(Counter(candidates.values())), flush=True)
    metadata_path = download(OI_BASE + "extended-crowdsourced-image-ids.csv", sources / "oi_image_metadata.csv")
    metadata = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ImageID"] not in candidates:
                continue
            # This subset accepts only the explicit license recorded on each image.
            if row["License"].rstrip("/") != "https://creativecommons.org/licenses/by/4.0":
                continue
            author = row["AuthorProfileURL"] if row["AuthorProfileURL"] not in ("", "n/a") else row["Author"]
            if author.strip().lower() in ("", "n/a", "unknown"):
                continue
            row["label"] = candidates[row["ImageID"]]
            row["group"] = "oi-author-" + hashlib.sha256(author.strip().casefold().encode()).hexdigest()[:16]
            metadata.append(row)
    random.Random(seed).shuffle(metadata)
    selected, author_counts, counts = [], Counter(), Counter()
    # Try extra candidates to compensate for dead URLs and exact duplicates.
    for row in metadata:
        key = (row["label"], row["group"])
        if counts[row["label"]] >= target + 80 or author_counts[key] >= 20:
            continue
        selected.append(row)
        counts[row["label"]] += 1
        author_counts[key] += 1
    print("Downloading licensed images:", dict(counts), flush=True)

    def fetch_one(row):
        folder = raw / row["label"] / row["group"] / row["group"]
        path = folder / (row["ImageID"] + ".jpg")
        url = row["Thumbnail300KURL"] or row["OriginalURL"]
        download(url, path)
        pixel_hash = fingerprint(path)
        return row, path, pixel_hash, url

    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(fetch_one, row): row for row in selected}
        for i, future in enumerate(as_completed(jobs), 1):
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append({"id": jobs[future]["ImageID"], "error": str(error)})
            if i % 100 == 0 or i == len(jobs):
                print(f"Images checked: {i}/{len(jobs)}, successful={len(successes)}, failures={len(failures)}", flush=True)
    # Stable order independent of thread completion order.
    order = {row["ImageID"]: i for i, row in enumerate(selected)}
    successes.sort(key=lambda entry: order[entry[0]["ImageID"]])
    accepted, seen, kept = [], set(), Counter()
    for row, path, digest, url in successes:
        if digest in seen or kept[row["label"]] >= target:
            # This is a generated download, not a user's original image.
            path.unlink()
            continue
        seen.add(digest)
        kept[row["label"]] += 1
        json_file(path.parent / "session.json", {"source": OI_SOURCE, "license": "CC-BY-4.0",
                  "instance": row["group"], "session": row["group"],
                  "grouping_note": "All images credited to one author stay in one split. True capture sessions are unavailable."})
        accepted.append({"path": str(path), "label": row["label"], "image_id": row["ImageID"],
                         "author": row["Author"], "author_profile": row["AuthorProfileURL"],
                         "source": OI_SOURCE, "original_url": row["OriginalURL"], "download_url": url,
                         "landing_url": row["OriginalLandingURL"], "license": row["License"],
                         "title": row["Title"], "group": row["group"], "pixel_sha256": digest})
    sas = sources / "sas"
    for name in ("LICENSE", "README.md", "training/TrainingImages.zip"):
        download(SAS_BASE + name, sas / Path(name).name)
    needed = max(0, target - kept["rubiks_cube"])
    with zipfile.ZipFile(sas / "TrainingImages.zip") as archive:
        names = [name for name in archive.namelist()
                 if name.lower().endswith(".jpg") and name.split("/")[0] in ("good", "bad")]
        random.Random(seed).shuffle(names)
        for name in names:
            if needed <= 0:
                break
            data = archive.read(name)
            digest = fingerprint(data)
            if digest in seen:
                continue
            seen.add(digest)
            # One source group: its original capture sessions are undocumented.
            group = "sas-cube-source"
            folder = raw / "rubiks_cube" / group / group
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / ("sas_" + name.replace("/", "_"))
            path.write_bytes(data)
            json_file(folder / "session.json", {"source": SAS_SOURCE, "license": "Apache-2.0",
                      "instance": group, "session": group,
                      "grouping_note": "All SAS images grouped together; do not split adjacent frames across sets."})
            accepted.append({"path": str(path), "label": "rubiks_cube", "image_id": name,
                             "author": "SAS Communities; see preserved source README/LICENSE",
                             "author_profile": SAS_SOURCE, "source": SAS_SOURCE,
                             "original_url": SAS_BASE + "training/TrainingImages.zip",
                             "download_url": SAS_BASE + "training/TrainingImages.zip",
                             "landing_url": SAS_SOURCE, "license": "Apache-2.0", "title": name,
                             "group": group, "pixel_sha256": digest})
            kept["rubiks_cube"] += 1
            needed -= 1
    if accepted:
        with (root / "attributions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(accepted[0]))
            writer.writeheader()
            writer.writerows(accepted)
    summary = {"target_per_class": target, "counts": dict(kept), "download_failures": failures,
               "sources": [OI_SOURCE, SAS_SOURCE], "seed": seed,
               "sas_commit": SAS_COMMIT,
               "annotation_license_open_images": "CC-BY-4.0",
               "review_status": "Human-verified source labels; local visual review still required.",
               "limitations": ["Public data does not test the user's own physical objects.",
                               "Capture sessions and physical instances are not fully documented.",
                               "Author/source grouping and duplicate checks reduce, but cannot prove absence of, leakage.",
                               "SAS cube photos are a correlated source; count is not independent scene count."]}
    json_file(root / "acquisition.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    acquire(args.data, args.target, args.seed, args.workers)
