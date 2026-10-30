import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from classifier import CLASSES
from classifier.common import seed_everything, write_json
from classifier.data import audit_rows, image_transform, near_duplicate_pairs, source_balanced_weights
from classifier.evaluate import load_checkpoint, metrics_from_predictions, evaluate_checkpoint
from classifier.model import TinyClassifier, describe_model, training_model, export_inference_model
from classifier.prepare import assign_groups, prepare
from classifier.train import train


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(7)

    def test_exact_model_contract(self):
        expected = {"baseline": (3252, 773184), "context": (6084, 871680), "context2": (8916, 970176), "standard": (20100, 2322624)}
        for name, (parameters, macs) in expected.items():
            result = describe_model(name)
            self.assertEqual((result["parameters"], result["macs"]), (parameters, macs))
            model = TinyClassifier(name)
            output = model(torch.randn(2, 3, 96, 96))
            self.assertEqual(tuple(output.shape), (2, 4))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_preprocessing_matches_input_integer_convention(self):
        image = Image.new("RGB", (96, 96), (0, 128, 255))
        tensor = image_transform()(image)
        self.assertEqual(tuple(tensor.shape), (3, 96, 96))
        self.assertEqual(tensor[:, 30, 30].tolist(), [-1.0, 0.0, 127 / 128])
        padded = image_transform()(Image.new("RGB", (96, 48), (255, 0, 0)))
        self.assertEqual(padded[:, 0, 0].tolist(), [0.0, 0.0, 0.0])

    def test_confusion_matrix_and_class_recall(self):
        result = metrics_from_predictions([0, 0, 1, 2, 3], [0, 1, 1, 2, 2])
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["per_class"]["rubiks_cube"]["recall"], 0.5)
        self.assertAlmostEqual(result["per_class"]["mouse"]["recall"], 0)
        self.assertEqual(result["confusion_matrix"][3][2], 1)

    def sample_rows(self):
        return [{"path": "left.jpg", "label": "book", "split": "train", "group": "left",
                 "sha256": "file1", "pixel_sha256": "pixel1", "dhash": "123456789abcdef0",
                 "source": "fixture", "license": "fixture"},
                {"path": "right.jpg", "label": "book", "split": "test", "group": "right",
                 "sha256": "file2", "pixel_sha256": "pixel2", "dhash": "fedcba9876543210",
                 "source": "fixture", "license": "fixture"}]

    def test_audit_rejects_group_and_pixel_leakage(self):
        rows = self.sample_rows()
        rows[1]["group"] = rows[0]["group"]
        with self.assertRaisesRegex(ValueError, "Leakage"):
            audit_rows(rows, verify_files=False, require_splits=False)
        rows = self.sample_rows()
        rows[1]["pixel_sha256"] = rows[0]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_rows(rows, verify_files=False, require_splits=False)

    def test_related_images_join_before_splitting(self):
        rows = self.sample_rows()
        for i, row in enumerate(rows):
            row.update(session=f"session{i}", instance="same_physical_object")
        rows[1]["dhash"] = rows[0]["dhash"]
        assign_groups(rows)
        self.assertEqual(rows[0]["group"], rows[1]["group"])
        self.assertEqual(len(near_duplicate_pairs(rows)), 1)

    def test_optimizer_can_reduce_a_fixed_batch_loss(self):
        model = TinyClassifier()
        images = torch.randn(4, 3, 96, 96)
        labels = torch.tensor([0, 1, 2, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()
        initial = criterion(model(images), labels).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        self.assertLess(criterion(model(images), labels).item(), initial)

    def test_bn_folding_preserves_outputs_and_inference_parameter_count(self):
        for architecture, inference_parameters, train_parameters in [("baseline", 3252, 3412), ("context", 6084, 6340), ("context2", 8916, 9268), ("standard", 20100, 20204)]:
            model = training_model(architecture, batch_norm=True)
            self.assertEqual(sum(p.numel() for p in model.parameters()), train_parameters)
            for _ in range(3):
                model(torch.randn(4, 3, 96, 96))
            model.eval()
            x = torch.randn(2, 3, 96, 96)
            exported = export_inference_model(model)
            self.assertEqual(sum(p.numel() for p in exported.parameters()), inference_parameters)
            self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in exported.modules()))
            torch.testing.assert_close(model(x), exported(x), rtol=1e-4, atol=1e-5)
            TinyClassifier(architecture).load_state_dict(exported.state_dict(), strict=True)

    def test_source_sampler_has_equal_class_and_source_mass(self):
        rows = [{"label": label, "source": "main"} for label in CLASSES for _ in range(3)]
        rows.append({"label": "rubiks_cube", "source": "rare"})
        weights = source_balanced_weights(rows)
        for label in CLASSES:
            self.assertAlmostEqual(sum(w for w, r in zip(weights, rows) if r["label"] == label), 0.25)
        self.assertAlmostEqual(weights[-1], 0.125)

    def test_end_to_end_smoke_is_explicitly_synthetic(self):
        with tempfile.TemporaryDirectory(prefix="basys3-smoke-") as directory:
            root = Path(directory)
            rng = np.random.default_rng(123)
            for label in CLASSES:
                for session in range(6):
                    folder = root / "raw" / label / "fixture" / f"s{session}"
                    folder.mkdir(parents=True)
                    write_json(folder / "session.json", {"source": "synthetic test fixture",
                               "license": "test-only", "synthetic": True, "session": f"s{session}"})
                    for index in range(2):
                        Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(folder / f"{index}.png")
            manifest = root / "manifest.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                rows = prepare(root / "raw", manifest)
                self.assertFalse(near_duplicate_pairs(rows))
                args = Namespace(manifest=str(manifest), output=str(root / "smoke_run"), architecture="baseline",
                                 epochs=2, patience=2, batch_size=8, learning_rate=0.002, weight_decay=0.0001,
                                 seed=42, device="cpu", workers=0, no_cache=False, evaluate_test=True,
                                 acknowledge_near_duplicates=False, smoke_test=True)
                summary = train(args)
            self.assertEqual(summary["status"], "synthetic_pipeline_smoke_test")
            output = root / "smoke_run"
            for name in ("best_model.pth", "test_confusion_matrix.png", "test_per_class.csv", "summary.json"):
                self.assertTrue((output / name).is_file(), name)
            model, checkpoint = load_checkpoint(output / "best_model.pth", torch.device("cpu"))
            self.assertTrue(checkpoint["smoke_test"])
            self.assertEqual(sum(p.numel() for p in model.parameters()), 3252)
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark = root / "personal_test.csv"
                test_rows = prepare(root / "raw", benchmark, test_only=True)
            self.assertTrue(all(row["split"] == "test" for row in test_rows))
            with self.assertRaisesRegex(ValueError, "overlaps"):
                evaluate_checkpoint(output / "best_model.pth", benchmark, "test", root / "external",
                                    torch.device("cpu"), external=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

