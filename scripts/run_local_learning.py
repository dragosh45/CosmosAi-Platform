#!/usr/bin/env python3
"""Serve generated learning pages on localhost without loading model services."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-root", type=Path, default=ROOT / "apps/learning-content/site")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not (args.content_root / "concepts/index.html").is_file():
        parser.error("Use a valid port and a generated content root containing concepts/index.html")
    env = dict(os.environ, COSMOSAI_LEARNING_CONTENT_ROOT=str(args.content_root.resolve()))
    print(f"Learning: http://127.0.0.1:{args.port}/learn/courses/\n"
          "Local learning pages and browser examples; model services are not started.", flush=True)
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "apps/api-gateway"),
        "--host", "127.0.0.1", "--port", str(args.port),
    ], cwd=ROOT, env=env))
