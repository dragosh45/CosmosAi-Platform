"""The learning diagnostic must not touch test pixels or serving artifacts."""

import csv
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


def test_learning_check_is_bounded_and_preserves_checkpoints(tmp_path):
    labels = ["elliptical", "spiral", "lenticular", "irregular"]
    rows = []
    expected_ids = set()
    for index, label in enumerate(labels):
        path = f"{label}.png"
        Image.new("RGB", (12, 12), (40 * index, 120, 180)).save(tmp_path / path)
        for sample in range(2):
            image_id = f"train-{label}-{sample}"
            expected_ids.add(image_id)
            rows.append([image_id, path, label, "train", "synthetic_test"])
        rows.append([f"val-{label}", path, label, "val", "synthetic_test"])
        rows.append([f"test-{label}", "must-not-be-loaded.png", label, "test", "synthetic_test"])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image_id", "image_path", "label", "split", "source"])
        writer.writerows(rows)
    serving = Path("data/demo/gz2_pilot_80.pt")
    serving_before = serving.read_bytes()
    destination = tmp_path / "diagnostic"
    command = [sys.executable, "scripts/check_galaxy_learning.py", str(manifest),
               "--data-root", str(tmp_path), "--output-dir", str(destination),
               "--epochs", "1", "--sanity-steps", "1", "--image-size", "8", "--threads", "1"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / "evaluation.json").read_text())
    assert report["test_evaluated"] is False
    assert report["test_records_excluded"] == 4
    assert report["serving_checkpoint_changed"] is False
    assert report["sanity_check"]["validation_evaluated"] is False
    assert set(report["sanity_check"]["training_image_ids"]) == expected_ids
    assert set(report["evaluation"]) == {"train", "val"}
    assert report["class_counts"]["train"] == dict.fromkeys(labels, 2)
    assert serving.read_bytes() == serving_before

    from cosmosai.galaxy.model import load_torch_checkpoint_bundle
    bundle = load_torch_checkpoint_bundle(destination / "candidate.pt")
    assert bundle.preprocessing.target_size == (8, 8)
    saved = (destination / "candidate.pt").read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "preserve earlier checkpoints" in repeated.stderr
    assert (destination / "candidate.pt").read_bytes() == saved

    # Bounds fail before loading the manifest or allocating image tensors.
    invalid = command.copy()
    invalid[invalid.index("--output-dir") + 1] = str(tmp_path / "unused")
    invalid[invalid.index("--threads") + 1] = "100"
    rejected = subprocess.run(invalid, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert "Bounds:" in rejected.stderr
    assert not (tmp_path / "unused").exists()
