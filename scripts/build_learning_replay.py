#!/usr/bin/env python3
"""Build a movable static HTML replay from a version-2 observer trace."""

import argparse
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SOURCE = REPO_ROOT / "apps" / "learning-replay"
MAX_TRACE_BYTES = 2_000_000


def load_trace(trace_path: Path) -> dict:
    if trace_path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("Trace is larger than the 2 MB teaching-replay limit")
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Trace is not valid JSON: {error}") from error
    if trace.get("format_version") != 2:
        raise ValueError("Learning replay requires observer trace format_version 2")
    if not isinstance(trace.get("updates"), list) or not trace["updates"]:
        raise ValueError("Trace must contain at least one update")
    if trace.get("parameter_count") != 132:
        raise ValueError("Learning replay expects the 132-parameter TinyGalaxyCNN")
    return trace


def build_replay(trace_path: Path, output_dir: Path) -> Path:
    trace = load_trace(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("index.html", "viewer.js", "viewer.css"):
        shutil.copyfile(REPLAY_SOURCE / filename, output_dir / filename)

    # Keep the report self-contained so file:// opening does not depend on fetch()
    # or a web server. ensure_ascii also avoids JavaScript line-separator surprises.
    trace_source = "window.COSMOSAI_SAMPLE_TRACE = " + json.dumps(
        trace, indent=2, ensure_ascii=True, allow_nan=False
    ) + ";\n"
    (output_dir / "sample_trace.js").write_text(trace_source, encoding="utf-8")
    return output_dir / "index.html"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML replay from a CosmosAI CNN trace."
    )
    parser.add_argument("--trace", type=Path, required=True, help="Version-2 observer JSON")
    parser.add_argument("--output", type=Path, required=True, help="Report directory")
    args = parser.parse_args()
    try:
        report_path = build_replay(args.trace, args.output)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Learning replay: {report_path}")
    print("Open the index.html file directly; no backend service is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
