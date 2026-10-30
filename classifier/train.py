"""Train the exact 3,252-parameter baseline from scratch on a fixed manifest."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import CLASSES, INPUT_SIZE
from .common import choose_device, environment, file_hash, seed_everything, write_json
from .data import ManifestDataset, audit_rows, near_duplicate_pairs, read_manifest, seed_worker, source_balanced_weights
from .evaluate import evaluate_checkpoint
from .model import training_model, describe_model, export_inference_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop and inspect training")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return {"loss": total_loss / count, "accuracy": correct / count}


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    for axis, key in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, [row[f"train_{key}"] for row in history], label="Train (augmented)")
        axis.plot(epochs, [row[f"val_{key}"] for row in history], label="Validation")
        axis.set(xlabel="Epoch", ylabel=key.capitalize())
        axis.legend()
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.workers < 0:
        raise ValueError("Epochs, batch size, and patience must be positive; workers must be nonnegative")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty; choose a new --output: {output}")
    rows = read_manifest(args.manifest)
    counts = audit_rows(rows)
    synthetic = any(row.get("synthetic") == "true" for row in rows)
    if synthetic and not args.smoke_test:
        raise ValueError("Synthetic fixtures require --smoke-test and are not real accuracy evidence")
    if args.smoke_test and "smoke" not in str(output).lower():
        raise ValueError("Smoke-test output directory must contain 'smoke'")
    near = near_duplicate_pairs(rows)
    if near and not args.acknowledge_near_duplicates:
        raise ValueError("Near-duplicate candidates cross splits. Review data/manifest.audit.json, then correct "
                         "the groups or explicitly use --acknowledge-near-duplicates if they are false positives.")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(json.dumps(environment(device), indent=2), flush=True)
    description = describe_model(args.architecture)
    if args.architecture == "baseline":
        assert description["parameters"] == 3252
    model = training_model(args.architecture, getattr(args, "batch_norm", False)).to(device)
    training_parameters = sum(parameter.numel() for parameter in model.parameters())
    generator = torch.Generator().manual_seed(args.seed)
    datasets = {split: ManifestDataset(rows, split, training=split == "train", cache=not args.no_cache)
                for split in ("train", "val")}
    sampler = None
    if getattr(args, "balance_sources", False):
        sampler = WeightedRandomSampler(source_balanced_weights(datasets["train"].rows),
                                        len(datasets["train"]), replacement=True, generator=generator)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train" and sampler is None,
                                 sampler=sampler if split == "train" else None,
                                 num_workers=args.workers, pin_memory=device.type == "cuda",
                                 worker_init_fn=seed_worker, generator=generator if split == "train" else None,
                                 persistent_workers=args.workers > 0)
               for split, dataset in datasets.items()}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    manifest_hash = file_hash(args.manifest)
    config = {**vars(args), "environment": environment(device), "model": description,
              "training_parameters": training_parameters,
              "counts": counts, "manifest_sha256": manifest_hash,
              "near_duplicate_warning_count_capped": len(near), "synthetic": synthetic,
              "preprocessing": "RGB; EXIF transpose; aspect-preserving bilinear pad to 96x96, fill128; (pixel-128)/128",
              "selection": "highest validation accuracy, then lowest validation loss; no test selection"}
    write_json(output / "config.json", config)
    # Snapshot rows as well as a file digest; never reshuffle splits during training.
    with (output / "manifest_snapshot.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_accuracy, best_loss, stale, history, best_epoch = -1.0, float("inf"), 0, [], 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        training = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(validation["loss"])
        record = {"epoch": epoch, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
                  "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, best_epoch, stale = validation["accuracy"], validation["loss"], epoch, 0
            inference_model = export_inference_model(model)
            assert sum(parameter.numel() for parameter in inference_model.parameters()) == description["parameters"]
            checkpoint = {"format_version": 1, "architecture": args.architecture,
                          "classes": list(CLASSES), "input_size": INPUT_SIZE,
                          "model_state_dict": {key: value.detach().cpu().clone() for key, value in inference_model.state_dict().items()},
                          "epoch": epoch, "validation_accuracy": best_accuracy,
                          "validation_loss": best_loss, "manifest_sha256": manifest_hash,
                          "seed": args.seed, "smoke_test": args.smoke_test,
                          "training_batch_norm_folded": getattr(args, "batch_norm", False),
                          "training_parameters": training_parameters,
                          "source_balanced_sampling": getattr(args, "balance_sources", False),
                          "preprocessing": config["preprocessing"], "torch_version": str(torch.__version__)}
            temporary = output / "best_model.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best_model.pth")
        else:
            stale += 1
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:03d} train={training['accuracy']:.2%} val={validation['accuracy']:.2%} "
              f"val_loss={validation['loss']:.4f} time={record['epoch_seconds']:.2f}s" +
              (" [best]" if improved else ""), flush=True)
        if stale >= args.patience:
            print(f"Early stopping after {stale} epochs without a better validation checkpoint", flush=True)
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    plot_history(history, output / "learning_curves.png")
    # All following metrics use the restored best checkpoint, with no augmentation.
    best = output / "best_model.pth"
    val_metrics = evaluate_checkpoint(best, args.manifest, "val", output, device, args.workers)
    train_metrics = evaluate_checkpoint(best, args.manifest, "train", output, device, args.workers)
    test_metrics = evaluate_checkpoint(best, args.manifest, "test", output, device, args.workers) if args.evaluate_test else None
    summary = {"status": "synthetic_pipeline_smoke_test" if args.smoke_test else "trained",
               "training_parameters": training_parameters,
               "training_batch_norm_folded": getattr(args, "batch_norm", False),
               "source_balanced_sampling": getattr(args, "balance_sources", False),
               "architecture": args.architecture, "device": str(device), "epochs_completed": len(history),
               "best_epoch": best_epoch, "training_seconds": training_seconds,
               "training_accuracy_unaugmented": train_metrics["accuracy"],
               "validation_accuracy": val_metrics["accuracy"],
               "test_accuracy": test_metrics["accuracy"] if test_metrics else None,
               "test_status": "evaluated after validation selection" if test_metrics else "not evaluated; reserved",
               "checkpoint_path": str(best), "checkpoint_file_bytes": best.stat().st_size,
               "model": description, "counts": counts,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
               "limitations": ["FP32 model; INT8 accuracy has not been measured.",
                               "Public dataset accuracy does not establish accuracy on your physical objects.",
                               "Source/author grouping cannot prove unknown capture-session independence."]}
    write_json(output / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--output", default="runs/baseline")
    parser.add_argument("--architecture", choices=("baseline", "context", "context2", "standard"), default="baseline")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0, help="0 is most robust on Windows; try 2 for throughput")
    parser.add_argument("--no-cache", action="store_true", help="Disable the small 96x96 decoded-image RAM cache")
    parser.add_argument("--evaluate-test", action="store_true", help="Use once after choosing architecture on validation")
    parser.add_argument("--acknowledge-near-duplicates", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Fixture validation only, never real accuracy evidence")
    parser.add_argument("--balance-sources", action="store_true", help="Balance classes and sources within each class during training")
    parser.add_argument("--batch-norm", action="store_true", help="Training-only BN; automatically fold into the saved Conv weights")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
