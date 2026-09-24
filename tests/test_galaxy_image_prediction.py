"""New-image inference must match the saved training recipe without training."""

import asyncio
import csv
import json
import subprocess
import sys

from PIL import Image
import pytest
import torch

from cosmosai.galaxy import checkpoint_inference
from cosmosai.galaxy.labels import ID_TO_LABEL
from cosmosai.galaxy.model import create_tiny_galaxy_cnn, save_torch_checkpoint
from cosmosai.galaxy.preprocessing import CheckpointContractError, GalaxyPreprocessingPolicy


@pytest.fixture
def image_case(tmp_path):
    image = Image.new("RGBA", (13, 9))
    image.putdata([(x * 19, y * 28, (x * 13 + y * 7) % 256, 180)
                   for y in range(9) for x in range(13)])
    image_path = tmp_path / "unknown-galaxy.png"
    image.save(image_path)
    manifest = tmp_path / "samples.csv"
    with manifest.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["image_id", "image_path", "label", "split", "source"])
        writer.writerow(["known-id", image_path.name, "spiral", "test", "synthetic_test"])
    checkpoint = save_torch_checkpoint(
        create_tiny_galaxy_cnn(), tmp_path / "model.pt", GalaxyPreprocessingPolicy((5, 3)),
    )
    return image_path, manifest, checkpoint


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_file_prediction_matches_manifest_and_does_not_update_weights(
    image_case, monkeypatch, image_format,
):
    image_path, manifest, checkpoint = image_case
    if image_format == "JPEG":
        with Image.open(image_path) as source:
            source.convert("RGB").save(image_path, format="JPEG")
    expected = checkpoint_inference.predict_from_checkpoint(
        manifest, image_path.parent, "known-id", checkpoint,
    )
    saved_bytes = checkpoint.read_bytes()
    load = checkpoint_inference.load_torch_checkpoint_bundle
    loaded = []

    def capture_load(path):
        bundle = load(path)
        weights = {name: value.detach().clone() for name, value in bundle.model.state_dict().items()}
        loaded.append((bundle, weights))
        return bundle

    monkeypatch.setattr(checkpoint_inference, "load_torch_checkpoint_bundle", capture_load)
    manifest.unlink()  # Prediction must not need a manifest or fabricated label.
    result = checkpoint_inference.predict_image_from_checkpoint(image_path, checkpoint)
    assert result.manifest_label is None
    assert result.input_shape == (1, 3, 3, 5)
    assert result.logits == expected.logits
    assert result.probabilities == expected.probabilities
    assert result.preprocessing == expected.preprocessing
    assert checkpoint.read_bytes() == saved_bytes
    assert len(loaded) == 1
    bundle, original_weights = loaded[0]
    assert not bundle.model.training
    assert all(parameter.grad is None for parameter in bundle.model.parameters())
    assert all(torch.equal(value, original_weights[name])
               for name, value in bundle.model.state_dict().items())


def test_unlabelled_prediction_cli_and_resize_contract(image_case):
    image_path, manifest, checkpoint = image_case
    manifest.unlink()
    command = [sys.executable, "scripts/predict_galaxy_checkpoint.py",
               "--image-path", str(image_path), "--checkpoint-path", str(checkpoint)]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "known_label: not provided (prediction only)" in completed.stdout
    assert "input_shape: (1, 3, 3, 5)" in completed.stdout
    assert "no training" in completed.stdout
    with pytest.raises(CheckpointContractError, match="conflicts"):
        checkpoint_inference.predict_image_from_checkpoint(image_path, checkpoint, (3, 5))
    with pytest.raises(ValueError, match="Unreadable image file"):
        checkpoint_inference.predict_image_from_checkpoint(image_path.with_name("missing.png"), checkpoint)


def test_file_prediction_preserves_native_size_when_checkpoint_has_no_resize(image_case):
    image_path, _, checkpoint = image_case
    save_torch_checkpoint(create_tiny_galaxy_cnn(), checkpoint)
    result = checkpoint_inference.predict_image_from_checkpoint(image_path, checkpoint)
    assert result.input_shape == (1, 3, 9, 13)


@pytest.fixture
def upload_service(image_case, monkeypatch, load_service_module):
    monkeypatch.setenv("COSMOSAI_GALAXY_CHECKPOINT_PATH", str(image_case[2]))
    monkeypatch.delenv("COSMOSAI_GALAXY_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("COSMOSAI_GALAXY_DATA_ROOT", raising=False)
    return load_service_module("m53_upload_service", "apps/galaxy-classifier-service/main.py")


def post_image(service, content, content_type="image/png", chunk_size=1024):
    """Exercise the actual ASGI route without requiring a new HTTP test library."""
    messages = []
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)] or [b""]

    async def receive():
        return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/classify/image",
             "raw_path": b"/classify/image", "query_string": b"", "root_path": "",
             "headers": [(b"content-type", content_type.encode())],
             "client": ("127.0.0.1", 12000), "server": ("127.0.0.1", 8002)}
    asyncio.run(service.app(scope, receive, send))
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages)
    return status, json.loads(body)


@pytest.mark.parametrize("image_format,content_type", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_upload_matches_local_prediction_and_cleans_temporary_file(
    image_case, upload_service, monkeypatch, image_format, content_type,
):
    image_path, _, checkpoint = image_case
    if image_format == "JPEG":
        with Image.open(image_path) as source:
            source.convert("RGB").save(image_path, format="JPEG")
    expected = checkpoint_inference.predict_image_from_checkpoint(image_path, checkpoint)
    predict = checkpoint_inference.predict_image_from_checkpoint
    paths = []

    def capture_prediction(path, checkpoint_path):
        paths.append(path)
        return predict(path, checkpoint_path)

    monkeypatch.setattr(checkpoint_inference, "predict_image_from_checkpoint", capture_prediction)
    status, result = post_image(upload_service, image_path.read_bytes(), content_type)
    assert status == 200
    assert result["status"] == "checkpoint_inference"
    assert result["label"] == expected.predicted_label
    assert result["input_shape"] == [1, 3, 3, 5]
    assert result["checkpoint_name"] == checkpoint.name
    assert result["preprocessing"] == expected.preprocessing.to_metadata()
    assert result["probabilities"] == {ID_TO_LABEL[i]: p for i, p in enumerate(expected.probabilities)}
    assert result["logits"] == {ID_TO_LABEL[i]: z for i, z in enumerate(expected.logits)}
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert len(paths) == 1 and not paths[0].exists()
    assert not upload_service.UPLOAD_INFERENCE_LOCK.locked()


@pytest.mark.parametrize("payload,content_type,expected", [
    (b"", "image/png", 422),
    (b"not an image", "image/png", 422),
    (b"not an image", "text/plain", 415),
])
def test_upload_rejects_bad_inputs(upload_service, payload, content_type, expected):
    status, response = post_image(upload_service, payload, content_type)
    assert status == expected
    assert "detail" in response
    assert not upload_service.UPLOAD_INFERENCE_LOCK.locked()


def test_upload_rejects_mime_mismatch_and_pixel_limit(image_case, upload_service, monkeypatch):
    content = image_case[0].read_bytes()
    assert post_image(upload_service, content, "image/jpeg")[0] == 415
    monkeypatch.setattr(upload_service, "MAX_UPLOAD_PIXELS", 100)
    assert post_image(upload_service, content)[0] == 413


def test_upload_rejects_stream_over_byte_limit_without_content_length(upload_service, monkeypatch):
    monkeypatch.setattr(upload_service, "MAX_UPLOAD_BYTES", 12)
    status, _ = post_image(upload_service, b"x" * 13, chunk_size=4)
    assert status == 413
    assert not upload_service.UPLOAD_INFERENCE_LOCK.locked()


@pytest.mark.parametrize("configuration", ["absent", "missing", "corrupt", "legacy"])
def test_upload_unavailable_checkpoint_is_never_a_stub_prediction(
    image_case, upload_service, monkeypatch, configuration,
):
    image_path, _, checkpoint = image_case
    if configuration == "absent":
        monkeypatch.delenv("COSMOSAI_GALAXY_CHECKPOINT_PATH")
    elif configuration == "missing":
        checkpoint.unlink()
    elif configuration == "corrupt":
        checkpoint.write_bytes(b"not a checkpoint")
    else:
        saved = torch.load(checkpoint, weights_only=True)
        del saved["preprocessing"]
        torch.save(saved, checkpoint)
    status, response = post_image(upload_service, image_path.read_bytes())
    assert status == 503
    assert "label" not in response
    assert not upload_service.UPLOAD_INFERENCE_LOCK.locked()


def test_upload_returns_busy_instead_of_overlapping_cpu_work(image_case, upload_service):
    with upload_service.UPLOAD_INFERENCE_LOCK:
        status, response = post_image(upload_service, image_case[0].read_bytes())
        assert status == 503
        assert "retry" in response["detail"]



def test_prepared_lazy_sample_keeps_its_existing_label(image_case):
    from cosmosai.galaxy.training_sample import GalaxyLazyTrainingSample

    image_path, manifest, checkpoint = image_case
    prepared = checkpoint_inference.load_sample_for_prediction(
        manifest, image_path.parent, "known-id", (5, 3),
    )
    lazy = GalaxyLazyTrainingSample(
        image_id=prepared.image_id, label=prepared.label, label_id=prepared.label_id,
        split=prepared.split, tensor=prepared.tensor,
    )
    prediction = checkpoint_inference.predict_sample_from_checkpoint(checkpoint, lazy)
    assert prediction.manifest_label == "spiral"
