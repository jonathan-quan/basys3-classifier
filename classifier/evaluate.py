"""Evaluate a fixed checkpoint; test data is never used for checkpoint selection."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CLASSES, DISPLAY_NAMES
from .common import choose_device, file_hash, seed_everything, write_json
from .data import ManifestDataset, audit_rows, read_manifest
from .model import TinyClassifier


def metrics_from_predictions(truth, predictions):
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)), 1)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot evaluate an empty split")
    per_class = {}
    for i, name in enumerate(CLASSES):
        true_positive, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall,
                           "accuracy_within_class": recall, "f1": f1, "support": support,
                           "correct": true_positive}
    accuracy = float(np.trace(matrix) / total)
    # Wilson binomial interval; correlated photos can make this optimistic.
    z = 1.96
    denominator = 1 + z * z / total
    center = (accuracy + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(accuracy * (1 - accuracy) / total + z * z / (4 * total * total)) / denominator
    return {"accuracy": accuracy, "examples": total, "confusion_matrix": matrix.tolist(),
            "per_class": per_class,
            "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])),
            "balanced_accuracy": float(np.mean([value["recall"] for value in per_class.values()])),
            "accuracy_95pct_wilson_interval": [float(center - radius), float(center + radius)],
            "interval_note": "Assumes independent examples; image correlation can understate uncertainty.",
            "class_order": list(CLASSES)}


def save_metrics(metrics, output, prefix):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{prefix}_metrics.json", metrics)
    with (output / f"{prefix}_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["class", "precision", "recall", "accuracy_within_class", "f1", "support", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in metrics["per_class"].items():
            writer.writerow({"class": name, **row})
    matrix = np.asarray(metrics["confusion_matrix"])
    with (output / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual / predicted", *CLASSES])
        for name, row in zip(CLASSES, matrix):
            writer.writerow([name, *row])
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for axis, values, title in zip(axes, (matrix, normalized), ("Counts", "Recall by true class")):
        axis.imshow(values, cmap="Blues", vmin=0)
        axis.set(xticks=range(4), yticks=range(4), xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                 xlabel="Predicted class", ylabel="Actual class", title=title)
        plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
        for i in range(4):
            for j in range(4):
                label = str(matrix[i, j]) if title == "Counts" else f"{values[i, j]:.1%}"
                axis.text(j, i, label, ha="center", va="center",
                          color="white" if values[i, j] > values.max() / 2 else "black")
    fig.suptitle(f"{prefix}: accuracy {metrics['accuracy']:.2%}, n={metrics['examples']}")
    fig.savefig(output / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != list(CLASSES) or checkpoint.get("input_size") != 96:
        raise ValueError("Checkpoint class order/input size is incompatible")
    model = TinyClassifier(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def evaluate_model(model, dataset, device, batch_size=64, workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    truth, predictions, records = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, labels, indices in loader:
            logits = model(images.to(device, non_blocking=True))
            scores = logits.softmax(dim=1).cpu()
            guess = scores.argmax(dim=1)
            truth.extend(labels.tolist())
            predictions.extend(guess.tolist())
            for index, actual, predicted, probabilities in zip(indices.tolist(), labels.tolist(), guess.tolist(), scores.tolist()):
                records.append({"path": dataset.rows[index]["path"], "actual": CLASSES[actual],
                                "predicted": CLASSES[predicted],
                                **{f"softmax_{name}": value for name, value in zip(CLASSES, probabilities)}})
    return metrics_from_predictions(truth, predictions), records


def evaluate_checkpoint(checkpoint_path, manifest, split, output, device, workers=0, external=False):
    rows = read_manifest(manifest)
    audit_rows(rows, require_splits=not external)
    selected_classes = {row["label"] for row in rows if row["split"] == split}
    if selected_classes != set(CLASSES):
        raise ValueError("The evaluated split must contain all four classes")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if not external and checkpoint["manifest_sha256"] != file_hash(manifest):
        raise ValueError("Manifest differs from the checkpoint's frozen dataset; use --external for a separate benchmark")
    if external:
        snapshot = Path(checkpoint_path).parent / "manifest_snapshot.csv"
        if snapshot.is_file():
            used_pixels = {row["pixel_sha256"] for row in read_manifest(snapshot) if row["split"] in ("train", "val")}
            if any(row["pixel_sha256"] in used_pixels for row in rows if row["split"] == split):
                raise ValueError("External benchmark overlaps training/validation images")
    if any(row.get("synthetic") == "true" for row in rows) and not checkpoint.get("smoke_test"):
        raise ValueError("Synthetic examples are not a real-world evaluation")
    metrics, predictions = evaluate_model(model, ManifestDataset(rows, split), device, workers=workers)
    metrics.update({"split": split, "checkpoint": str(Path(checkpoint_path).resolve()),
                    "checkpoint_sha256": file_hash(checkpoint_path), "manifest_sha256": file_hash(manifest),
                    "evaluation_scope": "external" if external else "frozen_public_or_local_split",
                    "smoke_test": checkpoint.get("smoke_test", False)})
    prefix = "external_" + split if external else split
    save_metrics(metrics, output, prefix)
    with (Path(output) / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary_path = Path(output) / "summary.json"
    if split == "test" and not external and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if Path(summary["checkpoint_path"]).resolve() == Path(checkpoint_path).resolve():
            summary["test_accuracy"] = metrics["accuracy"]
            summary["test_status"] = "evaluated after validation selection"
            write_json(summary_path, summary)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", default="runs/evaluation")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--external", action="store_true", help="Evaluate a genuinely separate benchmark manifest")
    args = parser.parse_args()
    seed_everything(42)
    evaluate_checkpoint(args.checkpoint, args.manifest, args.split, args.output,
                        choose_device(args.device), args.workers, args.external)


if __name__ == "__main__":
    main()

