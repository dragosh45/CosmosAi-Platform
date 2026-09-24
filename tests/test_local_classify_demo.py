"""Verify the actual three-service upload path and its model/evaluation contract."""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from fastapi import HTTPException
from PIL import Image
import pytest
import requests

from cosmosai import upload_proxy
from cosmosai.galaxy.checkpoint_inference import predict_image_from_checkpoint
from cosmosai.galaxy.model import create_tiny_galaxy_cnn, save_torch_checkpoint
from cosmosai.galaxy.preprocessing import GalaxyPreprocessingPolicy


@pytest.fixture(scope="module")
def live_demo(tmp_path_factory):
    directory = tmp_path_factory.mktemp("local-demo")
    checkpoint = save_torch_checkpoint(create_tiny_galaxy_cnn(), directory / "demo.pt",
                                      GalaxyPreprocessingPolicy((5, 3)))
    image = directory / "image.png"
    Image.new("RGBA", (17, 11), (73, 127, 211, 120)).save(image)
    card = directory / "card.json"
    card.write_text(json.dumps({"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                               "training_images": 0, "epochs": 0,
                               "validation": {"sample_count": 0, "accuracy": None},
                               "test": {"sample_count": 0, "accuracy": None}, "notes": ["Test fixture"]}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    session = requests.Session()
    session.trust_env = False
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen([
            sys.executable, "scripts/run_local_demo.py", "--checkpoint", str(checkpoint),
            "--model-card", str(card), "--port", str(port),
        ], stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        try:
            for _ in range(100):
                assert process.poll() is None, (directory / "server.log").read_text()
                try:
                    if session.get(base + "/model", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail((directory / "server.log").read_text())
            yield session, base, image, checkpoint, card
        finally:
            process.terminate()
            process.wait(timeout=30)
            session.close()


def test_browser_assets_and_full_upload_path_match_direct_inference(live_demo):
    session, base, image, checkpoint, _ = live_demo
    page = session.get(base + "/", timeout=5)
    assert page.status_code == 200
    assert "Galaxy classifier" in page.text
    for asset in ("/assets/classify.css", "/assets/classify.js", "/assets/vendor/lucide.min.js",
                  "/samples/gz2-587725590382051490.jpg"):
        assert session.get(base + asset, timeout=5).status_code == 200
    expected = predict_image_from_checkpoint(image, checkpoint)
    before = checkpoint.read_bytes()
    response = session.post(base + "/classify/image", data=image.read_bytes(),
                            headers={"Content-Type": "image/png"}, timeout=20)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_service"] == "galaxy-classifier-service"
    assert body["status"] == body["classification"]["status"] == "checkpoint_inference"
    result = body["classification"]
    assert result["label"] == expected.predicted_label
    assert list(result["probabilities"].values()) == expected.probabilities
    assert list(result["logits"].values()) == expected.logits
    assert result["input_shape"] == [1, 3, 3, 5]
    assert result["checkpoint_sha256"] == hashlib.sha256(before).hexdigest()
    assert checkpoint.read_bytes() == before


def test_evaluation_is_bound_to_active_checkpoint(live_demo):
    session, base, _, checkpoint, card = live_demo
    info = session.get(base + "/model", timeout=20).json()
    assert info["checkpoint_name"] == checkpoint.name
    assert info["evaluation"]["checkpoint_sha256"] == info["checkpoint_sha256"]
    saved = card.read_text()
    try:
        card.write_text(json.dumps({"checkpoint_sha256": "other-model", "validation": {"accuracy": 0.99}}))
        assert session.get(base + "/model", timeout=20).json()["evaluation"] is None
    finally:
        card.write_text(saved)


def test_learning_replay_and_assets_are_served_by_gateway(live_demo):
    session, base, _, _, _ = live_demo
    home = session.get(base + "/", timeout=5)
    assert 'href="/learn/"' in home.text
    assert "CNN Learning Replay" in home.text
    assert 'href="/" aria-current="page">Galaxy Classifier</a>' in home.text
    assert 'href="#model"' not in home.text
    assert 'id="model"' in home.text
    redirect = session.get(base + "/learn", allow_redirects=False, timeout=5)
    assert redirect.status_code == 307
    assert redirect.headers["Location"].endswith("/learn/")
    replay = session.get(base + "/learn/", timeout=5)
    assert replay.status_code == 200
    assert replay.headers["Cache-Control"] == "no-store"
    assert "separate from uploaded-image predictions" in replay.text
    assert 'href="/">Galaxy Classifier</a>' in replay.text
    assert 'href="/#model"' not in replay.text
    for asset in ("viewer.css", "viewer.js", "sample_trace.js"):
        response = session.get(base + "/learn/" + asset, timeout=5)
        assert response.status_code == 200
        assert response.content == (Path("apps/learning-replay") / asset).read_bytes()
    assert session.get(base + "/learn/%2e%2e/api-gateway/main.py", timeout=5).status_code == 404


def test_upload_errors_are_preserved_through_both_proxies(live_demo):
    session, base, _, _, _ = live_demo
    endpoint = base + "/classify/image"
    for content, content_type, status in [(b"", "image/png", 422),
                                           (b"invalid", "image/png", 422),
                                           (b"invalid", "text/plain", 415)]:
        response = session.post(endpoint, data=content, headers={"Content-Type": content_type}, timeout=5)
        assert response.status_code == status
        assert isinstance(response.json()["detail"], str)
    # No Content-Length: the gateway must still cap streamed input.
    response = session.post(endpoint, data=iter([b"x" * 1024 * 1024] * 6),
                            headers={"Content-Type": "image/png"}, timeout=5)
    assert response.status_code == 413


def test_concepts_and_diagrams_are_allowlisted_static_content(live_demo):
    session, base, _, _, _ = live_demo
    for page in ("/learn/concepts/", "/learn/diagrams/"):
        response = session.get(base + page, timeout=5)
        assert response.status_code == 200
        assert 'href="/learn/concepts/"' in response.text
        assert 'href="/learn/diagrams/"' in response.text
    for asset in ("/assets/learning.css", "/assets/learning.js", "/learn/diagrams/catalog.json",
                  "/learn/diagrams/svg/neural_network_concepts.svg",
                  "/learn/diagrams/scenes/neural_network_concepts.excalidraw"):
        assert session.get(base + asset, timeout=5).status_code == 200
    for path in ("/docs/concepts_explanations.md", "/learn/concepts/%2e%2e/content.json",
                 "/learn/diagrams/%2e%2e/%2e%2e/api-gateway/main.py", "/.env", "/.git/config"):
        assert session.get(base + path, timeout=5).status_code == 404


@pytest.mark.parametrize("status", ["stub", "checkpoint_inference"])
def test_legacy_route_propagates_classifier_status(load_service_module, monkeypatch, status):
    router = load_service_module("m53b_router", "apps/inference-router/main.py")
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({"status": status, "label": "spiral"}).encode()
    monkeypatch.setattr(router.requests, "post", lambda *args, **kwargs: response)
    result = router.route(router.RouteRequest(input_type="galaxy_image", image_id="example"))
    assert result.status == result.classification["status"] == status


@pytest.mark.parametrize("exception,status", [(requests.Timeout(), 504), (requests.ConnectionError(), 502)])
def test_unreachable_upstream_returns_actionable_error(monkeypatch, exception, status):
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(upload_proxy.requests, "request", fail)
    with pytest.raises(HTTPException) as error:
        upload_proxy.upstream_request("GET", "http://unused.test/model")
    assert error.value.status_code == status


@pytest.mark.parametrize("payload,status", [(b"<html>failed</html>", 500), (b"[]", 200)])
def test_invalid_upstream_json_is_a_gateway_error(payload, status):
    response = requests.Response()
    response.status_code = status
    response._content = payload
    with pytest.raises(HTTPException) as error:
        upload_proxy.response_json(response)
    assert error.value.status_code == 502



@pytest.mark.parametrize("service", ["api-gateway", "inference-router"])
def test_service_can_still_run_from_its_own_directory(service):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", "import main; print(main.health()['status'])"],
                            cwd=Path("apps") / service, env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
