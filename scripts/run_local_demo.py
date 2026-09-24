#!/usr/bin/env python3
"""Run the local Classify page, gateway, router and saved-model service.

This starts inference only. It does not download data or train a model.
All three services listen on localhost and stop when this command is stopped.
Docs: docs/local_classifier_demo.md.
"""

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def available_port(preferred: int = 0) -> int:
    """Check a requested port or ask the OS for a free internal port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", preferred))
        return listener.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-card", type=Path,
                        default=ROOT / "data/samples/gz2_pilot_model_card.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("Use a port between 1 and 65535 and at least one CPU thread")
    try:
        ports = [available_port(args.port)]
        while len(ports) < 3:
            port = available_port()
            if port not in ports:
                ports.append(port)
    except OSError:
        parser.error(f"Port {args.port} is occupied. Choose another --port")
    gateway_port, router_port, classifier_port = ports
    env = dict(os.environ)
    env.update({
        "COSMOSAI_GALAXY_CHECKPOINT_PATH": str(args.checkpoint.resolve()),
        "COSMOSAI_GALAXY_MODEL_CARD": str(args.model_card.resolve()),
        "COSMOSAI_ROUTER_BASE_URL": f"http://127.0.0.1:{router_port}",
        "COSMOSAI_GALAXY_BASE_URL": f"http://127.0.0.1:{classifier_port}",
        "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
    })
    # Launches share the repo Python environment; no extra model or UI runtime.
    processes = []
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for service, port in [("galaxy-classifier-service", classifier_port),
                              ("inference-router", router_port), ("api-gateway", gateway_port)]:
            processes.append(subprocess.Popen([
                sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "apps" / service),
                "--host", "127.0.0.1", "--port", str(port),
            ], cwd=ROOT, env=env))
        print(f"\nCosmosAI Classify: http://127.0.0.1:{gateway_port}\n"
              "Saved-checkpoint inference. Ctrl+C stops all demo services.\n", flush=True)
        while not stopping:
            if any(process.poll() is not None for process in processes):
                print("A demo service exited; stopping the remaining services.", file=sys.stderr)
                return 1
            time.sleep(0.5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
