#!/usr/bin/env python3
"""Smoke-test a running portable release: pages, trusted model, upload and limits.

No training or deployment occurs here. The default target is the separate local
Docker rehearsal on port 8090; --base-url can later verify an approved HTTPS URL.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def check_learning(session, base):
    """Compare the served learning release with this checkout, including assets.

    A 200 status alone can hide a stale revision or an HTML fallback for missing
    JavaScript. Hash checks prove the selected pages and calculations match.
    """
    site = ROOT / "apps/learning-content/site"
    files = {
        "/learn/": ROOT / "apps/learning-replay/index.html",
        "/learn/viewer.js": ROOT / "apps/learning-replay/viewer.js",
        "/assets/learning.js": ROOT / "apps/api-gateway/static/learning.js",
        "/assets/learning.css": ROOT / "apps/api-gateway/static/learning.css",
    }
    for relative in ("concepts/index.html", "concepts/build.json",
                     "courses/index.html", "diagrams/index.html", "diagrams/catalog.json"):
        files["/learn/" + relative.removesuffix("index.html")] = site / relative
    for directory in ("courses", "examples"):
        for path in sorted((site / directory).rglob("*")):
            if path.is_file():
                relative = path.relative_to(site).as_posix()
                files["/learn/" + relative.removesuffix("index.html")] = path
    for url, path in files.items():
        response = session.get(base + url, timeout=90)
        response.raise_for_status()
        assert hashlib.sha256(response.content).digest() == hashlib.sha256(path.read_bytes()).digest(), (
            f"Stale or incorrect learning asset: {url}; check the deployed revision")
    return len(files)


def check(base_url, identity_token=None):
    """Exercise only public gateway endpoints, using a bundled galaxy JPEG."""
    base = base_url.rstrip("/")
    expected = hashlib.sha256((ROOT / "data/demo/gz2_pilot_80.pt").read_bytes()).hexdigest()
    session = requests.Session()
    if identity_token:
        session.headers["Authorization"] = f"Bearer {identity_token}"
    started = time.monotonic()
    try:
        for page in ("/", "/learn/", "/learn/concepts/", "/learn/diagrams/",
                     "/assets/learning.js", "/learn/diagrams/svg/neural_network_concepts.svg"):
            response = session.get(base + page, timeout=90)
            response.raise_for_status()
        learning_assets = check_learning(session, base)
        model = session.get(base + "/model", timeout=30)
        model.raise_for_status()
        info = model.json()
        assert info["checkpoint_sha256"] == expected, "Wrong release checkpoint"
        assert info["evaluation"]["checkpoint_sha256"] == expected, "Model card mismatch"
        image = ROOT / "data/samples/images/gz2_pilot_preview/gz2-587725590382051490.jpg"
        before = time.monotonic()
        response = session.post(base + "/classify/image", data=image.read_bytes(),
                                headers={"Content-Type": "image/jpeg"}, timeout=60)
        response.raise_for_status()
        latency = time.monotonic() - before
        body = response.json()
        result = body["classification"]
        assert body["status"] == result["status"] == "checkpoint_inference"
        assert result["checkpoint_sha256"] == expected
        assert len(result["probabilities"]) == 4
        assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
        for content, content_type, status in [(b"not an image", "image/png", 422),
                                              (b"not an image", "text/plain", 415),
                                              (b"x" * (5 * 1024 * 1024 + 1), "image/jpeg", 413)]:
            error = session.post(base + "/classify/image", data=content,
                                 headers={"Content-Type": content_type}, timeout=15)
            assert error.status_code == status, (error.status_code, error.text)
        for private in ("/.env", "/.git/config", "/docs/concepts_explanations.md"):
            assert session.get(base + private, timeout=5).status_code == 404
        return {"base_url": base, "status": "passed", "checkpoint_sha256": expected,
                "predicted_label": result["label"], "prediction_seconds": round(latency, 3),
                "checks_seconds": round(time.monotonic() - started, 3),
                "learning_assets": learning_assets, "upload_error_codes": [422, 415, 413]}
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--identity-token",
        help="Optional Google identity token when checking a private Cloud Run service",
    )
    arguments = parser.parse_args()
    print(json.dumps(check(arguments.base_url, arguments.identity_token), indent=2))
