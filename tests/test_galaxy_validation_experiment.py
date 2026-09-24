"""Candidate training must keep test images out and preserve earlier artifacts."""

import csv
import json
import subprocess
import sys

from PIL import Image


def test_validation_candidate_excludes_test_pixels_and_prevents_overwrite(tmp_path):
    Image.new("RGB", (12, 12), (80, 120, 160)).save(tmp_path / "image.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image_id", "image_path", "label", "split", "source"])
        writer.writerow(["train", "image.png", "spiral", "train", "synthetic_test"])
        writer.writerow(["val", "image.png", "spiral", "val", "synthetic_test"])
        writer.writerow(["test", "must-not-be-loaded.png", "spiral", "test", "synthetic_test"])
    destination = tmp_path / "candidate"
    command = [sys.executable, "scripts/run_galaxy_validation_experiment.py", str(manifest),
               "--data-root", str(tmp_path), "--output-dir", str(destination),
               "--epochs", "1", "--image-size", "8"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / "evaluation.json").read_text())
    assert report["test_evaluated"] is False
    assert report["test_records_excluded"] == 1
    assert set(report["evaluation"]) == {"train", "val"}
    assert report["training"]["test_sample_count"] == 0
    saved = (destination / "candidate.pt").read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "preserve earlier checkpoints" in repeated.stderr
    assert (destination / "candidate.pt").read_bytes() == saved
