"""M49: a checkpoint's input recipe must survive training -> CLI/API inference."""

import csv
import json
from pathlib import Path
import subprocess
import sys

from fastapi import HTTPException
from PIL import Image
import pytest
import torch

from cosmosai.galaxy import checkpoint_inference
from cosmosai.galaxy.model import (
    create_tiny_galaxy_cnn,
    load_torch_checkpoint,
    load_torch_checkpoint_bundle,
    run_torch_forward_pass,
    save_torch_checkpoint,
)
from cosmosai.galaxy.preprocessing import CheckpointContractError, GalaxyPreprocessingPolicy


@pytest.fixture
def trained_png(tmp_path):
    # A pattern (not one flat color) exposes pixel/resize differences. Both source
    # and target are non-square, so swapped width and height cannot pass unnoticed.
    image = Image.new("RGBA", (13, 9))
    image.putdata([(x * 19, y * 28, (x * 13 + y * 7) % 256, 180)
                   for y in range(9) for x in range(13)])
    image.save(tmp_path / "galaxy.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image_id", "image_path", "label", "split", "source"])
        writer.writerow(["png-proof", "galaxy.png", "spiral", "train", "synthetic_test"])
    checkpoint = tmp_path / "model.pt"
    result = subprocess.run(
        [sys.executable, "scripts/train_galaxy_cnn_baseline.py", str(manifest),
         "--galaxy-data-root", str(tmp_path), "--epochs", "2", "--batch-size", "2",
         "--image-width", "5", "--image-height", "3", "--checkpoint-path", str(checkpoint)],
        text=True, capture_output=True, check=True,
    )
    assert "logits_match: True" in result.stdout
    return manifest, checkpoint


def test_training_cli_saves_recipe_and_cli_api_use_identical_pixels(
    trained_png, tmp_path, monkeypatch, load_service_module
):
    manifest, checkpoint = trained_png
    saved = load_torch_checkpoint_bundle(checkpoint)
    assert saved.preprocessing.to_metadata() == {
        "version": 1, "color_mode": "RGB", "target_size": [5, 3],
        "resize_interpolation": "bilinear", "normalization": "divide_by_255",
    }
    sample = checkpoint_inference.load_sample_for_prediction(manifest, tmp_path, "png-proof", (5, 3))
    with Image.open(tmp_path / "galaxy.png") as image:
        expected_bytes = image.convert("RGB").resize((5, 3), Image.Resampling.BILINEAR).tobytes()
    assert sample.tensor.shape == (3, 5, 3)
    assert sample.tensor.values == [value / 255 for value in expected_bytes]
    expected = run_torch_forward_pass(sample, saved.model)

    # Spy on both entry paths: compare full prepared values, not only winning class.
    seen = []
    forward = checkpoint_inference.run_torch_forward_pass
    def capture_forward(sample, model):
        seen.append(sample.tensor)
        return forward(sample, model)
    monkeypatch.setattr(checkpoint_inference, "run_torch_forward_pass", capture_forward)
    reads = []
    load = checkpoint_inference.load_torch_checkpoint_bundle
    def capture_load(path):
        reads.append(path)
        return load(path)
    monkeypatch.setattr(checkpoint_inference, "load_torch_checkpoint_bundle", capture_load)

    prediction = checkpoint_inference.predict_from_checkpoint(manifest, tmp_path, "png-proof", checkpoint)
    assert prediction.input_shape == (1, 3, 3, 5)
    assert prediction.logits == expected.logits
    assert prediction.probabilities == expected.probabilities
    monkeypatch.setenv("COSMOSAI_GALAXY_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("COSMOSAI_GALAXY_MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("COSMOSAI_GALAXY_DATA_ROOT", str(tmp_path))
    service = load_service_module("m49_service", "apps/galaxy-classifier-service/main.py")
    response = service.classify(service.ClassifyRequest(image_id="png-proof"))
    assert response.status == "checkpoint_inference"
    assert response.label == prediction.predicted_label
    assert response.confidence == prediction.probabilities[prediction.predicted_label_id]
    assert seen == [sample.tensor, sample.tensor]
    assert reads == [checkpoint, checkpoint]  # Exactly one file read per request.

    command = [sys.executable, "scripts/predict_galaxy_checkpoint.py", str(manifest),
               "--galaxy-data-root", str(tmp_path), "--image-id", "png-proof",
               "--checkpoint-path", str(checkpoint)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "input_shape: (1, 3, 3, 5)" in result.stdout
    assert f"logits: {[round(value, 4) for value in prediction.logits]}" in result.stdout
    matching = subprocess.run(command + ["--image-width", "5", "--image-height", "3"],
                              text=True, capture_output=True, check=True)
    assert matching.stdout == result.stdout
    conflict = subprocess.run(command + ["--image-width", "3", "--image-height", "5"],
                              text=True, capture_output=True)
    assert conflict.returncode == 1
    assert "conflicts with checkpoint" in conflict.stdout


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", True), ("color_mode", "L"),
    ("normalization", "none"), ("resize_interpolation", "nearest"),
    ("target_size", [0, 3]), ("target_size", [-1, 3]),
    ("target_size", [True, 3]), ("target_size", [3.0, 3]),
    ("target_size", [3]), ("target_size", "3x3"), ("target_size", (3, 3)),
])
def test_rejects_unsupported_recipe_on_load(tmp_path, field, value):
    path = save_torch_checkpoint(create_tiny_galaxy_cnn(), tmp_path / "bad.pt")
    payload = torch.load(path, weights_only=True)
    payload["preprocessing"][field] = value
    torch.save(payload, path)
    with pytest.raises(CheckpointContractError, match="[Pp]reprocessing"):
        load_torch_checkpoint(path)


@pytest.mark.parametrize("metadata", [{}, [], {"target_size": [3, 3]}, None])
def test_rejects_incomplete_recipe(metadata):
    with pytest.raises(CheckpointContractError):
        GalaxyPreprocessingPolicy.from_metadata(metadata)


def test_legacy_checkpoint_fails_clearly_in_cli_and_api(tmp_path, monkeypatch, load_service_module):
    path = save_torch_checkpoint(create_tiny_galaxy_cnn(), tmp_path / "legacy.pt")
    payload = torch.load(path, weights_only=True)
    del payload["preprocessing"]  # This is exactly the old M48 format.
    torch.save(payload, path)
    with pytest.raises(CheckpointContractError, match="Regenerate"):
        load_torch_checkpoint(path)
    result = subprocess.run(
        [sys.executable, "scripts/predict_galaxy_checkpoint.py", "--checkpoint-path", str(path)],
        text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "Legacy settings cannot be inferred" in result.stdout
    monkeypatch.setenv("COSMOSAI_GALAXY_CHECKPOINT_PATH", str(path))
    monkeypatch.setenv("COSMOSAI_GALAXY_MANIFEST_PATH", "data/samples/galaxy_manifest_sample.csv")
    monkeypatch.setenv("COSMOSAI_GALAXY_DATA_ROOT", "data/samples/images")
    service = load_service_module("m49_legacy_service", "apps/galaxy-classifier-service/main.py")
    with pytest.raises(HTTPException) as failure:
        service.classify(service.ClassifyRequest(image_id="gz2-000001"))
    assert failure.value.status_code == 503
    assert "Regenerate" in failure.value.detail


def test_explicit_no_resize_policy_and_prepared_sample_validation(tmp_path):
    path = save_torch_checkpoint(create_tiny_galaxy_cnn(), tmp_path / "native.pt")
    policy = load_torch_checkpoint_bundle(path).preprocessing
    assert policy.target_size is None
    assert policy.resolve_target_size(None) is None
    with pytest.raises(CheckpointContractError, match="conflicts"):
        policy.resolve_target_size((3, 3))
    fixed_path = save_torch_checkpoint(create_tiny_galaxy_cnn(), tmp_path / "fixed.pt",
                                      GalaxyPreprocessingPolicy((5, 3)))
    sample = checkpoint_inference.load_sample_for_prediction(
        Path("data/samples/galaxy_manifest_sample.csv"), Path("data/samples/images"), "gz2-000001"
    )
    with pytest.raises(CheckpointContractError, match="Prepared tensor shape"):
        checkpoint_inference.predict_sample_from_checkpoint(fixed_path, sample)


def test_stub_does_not_import_ml_helpers():
    # A fresh interpreter ensures an earlier test has not cached torch imports.
    code = '''
import importlib.util, json, os, sys
for key in list(os.environ):
    if key.startswith("COSMOSAI_GALAXY_"):
        del os.environ[key]
spec = importlib.util.spec_from_file_location("stub", "apps/galaxy-classifier-service/main.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)
print(json.dumps(service.classify(service.ClassifyRequest(image_id="demo")).model_dump()))
assert "torch" not in sys.modules
assert "cosmosai.galaxy.checkpoint_inference" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["status"] == "stub"
