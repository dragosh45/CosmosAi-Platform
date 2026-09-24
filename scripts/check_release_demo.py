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


def check(base_url):
    """Exercise only public gateway endpoints, using a bundled galaxy JPEG."""
    base = base_url.rstrip("/")
    expected = hashlib.sha256((ROOT / "data/demo/gz2_pilot_80.pt").read_bytes()).hexdigest()
    session = requests.Session()
    started = time.monotonic()
    try:
        for page in ("/", "/learn/", "/learn/concepts/", "/learn/diagrams/",
                     "/assets/learning.js", "/learn/diagrams/svg/neural_network_concepts.svg"):
            response = session.get(base + page, timeout=15)
            response.raise_for_status()
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
                "pages": 4, "upload_error_codes": [422, 415, 413]}
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    print(json.dumps(check(parser.parse_args().base_url), indent=2))
