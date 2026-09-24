#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 and docs/excalidraw/manifest_tooling_code_flow.excalidraw explain this loader.

# Import argparse so this loader can also be inspected from the command line.
import argparse

# Import Path to represent manifest and image paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return useful exit codes.
import sys

# Add the repo root so this CLI wrapper imports the same package as training.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-export the shared manifest API for existing tests and learning links.
from cosmosai.galaxy.manifest import (  # noqa: E402
    DEFAULT_GALAXY_DATA_ROOT,
    GalaxyManifestRecord,
    load_manifest,
    missing_image_paths,
    records_by_split,
    resolve_image_path,
    resolved_image_paths_by_id,
)


# Parse command-line arguments for quickly inspecting a manifest.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Load and summarize a CosmosAI galaxy manifest CSV file."
    )

    # Accept a manifest path, defaulting to the tiny sample manifest in this repo.
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/samples/galaxy_manifest_sample.csv",
        help="Path to the galaxy manifest CSV file to load.",
    )

    # Accept a configurable galaxy data root without requiring real image files yet.
    parser.add_argument(
        "--galaxy-data-root",
        default=str(DEFAULT_GALAXY_DATA_ROOT),
        help="External galaxy data root used to resolve manifest image_path values.",
    )

    # Optionally check image file existence when real data has been downloaded.
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Report how many resolved image files are missing.",
    )

    # Return the parsed arguments to main().
    return parser.parse_args()


# Run the loader as a command-line program.
def main() -> int:
    # Read the manifest path argument from the command line.
    args = parse_args()

    # Convert the incoming path string into a Path object.
    manifest_path = Path(args.manifest_path)

    try:
        # Load the manifest into structured records.
        records = load_manifest(manifest_path)
    except ValueError as error:
        # Print validation errors and return a non-zero exit code.
        print(error)
        return 1

    # Group records by split for an easy human-readable summary.
    grouped = records_by_split(records)

    # Resolve the first image path as a quick sanity check for future training code.
    resolved_paths = resolved_image_paths_by_id(
        records,
        Path(args.galaxy_data_root),
    )
    first_record = records[0]

    # Print a compact summary showing that the manifest can be loaded.
    print(f"Loaded {len(records)} galaxy manifest records from {manifest_path}")
    print(f"train: {len(grouped['train'])}")
    print(f"val: {len(grouped['val'])}")
    print(f"test: {len(grouped['test'])}")
    print(
        "first resolved image path: "
        f"{resolved_paths[first_record.image_id]}"
    )

    # Only check file existence when requested because sample metadata has no real images.
    if args.check_images:
        missing_paths = missing_image_paths(records, Path(args.galaxy_data_root))
        print(f"missing image files: {len(missing_paths)}")

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
