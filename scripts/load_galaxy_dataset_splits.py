#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 explains the current galaxy data path.
# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.

# Import argparse so the split loader can run from the command line.
import argparse

# Import Path to handle manifest and image root paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys

# Add the repo root so this CLI wrapper can import the shared package by path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-export shared split helpers so old tests and CLI notes keep working.
from cosmosai.galaxy.dataset_splits import (  # noqa: E402
    GalaxyDatasetSplits,
    create_dataset_splits,
    empty_split_buckets,
)

# Import shared manifest loading so this wrapper starts from the package data contract.
from cosmosai.galaxy.manifest import load_manifest  # noqa: E402

# Import the shared training sample object for type hints in print helpers.
from cosmosai.galaxy.training_sample import (  # noqa: E402
    GalaxyTrainingSample,
)


# Convert optional image width/height CLI values into a Pillow resize target.
def target_size_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    # If neither value is passed, do not resize images.
    if args.image_width is None and args.image_height is None:
        return None

    # Require both dimensions so resizing cannot silently distort intent.
    if args.image_width is None or args.image_height is None:
        raise ValueError("--image-width and --image-height must be used together")

    # Both dimensions must be positive pixel counts.
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("--image-width and --image-height must be at least 1")

    # Pillow expects target size as width, height.
    return (args.image_width, args.image_height)


# Parse command-line arguments for the dataset split proof.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Create tiny CosmosAI galaxy training samples grouped by split."
    )

    # Accept a manifest path, defaulting to the sample manifest.
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/samples/galaxy_manifest_sample.csv",
        help="Path to the galaxy manifest CSV file.",
    )

    # Accept a galaxy data root, defaulting to the sample image folder.
    parser.add_argument(
        "--galaxy-data-root",
        default="data/samples/images",
        help="Galaxy data root used to resolve manifest image_path values.",
    )

    # Allow strict mode when the caller wants missing images to invalidate the run.
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Reject missing images. Corrupt, unsupported, and invalid images are "
            "always rejected."
        ),
    )

    # Optionally resize normal PNG/JPG inputs while creating split samples.
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Optional target image width for Pillow-loaded PNG/JPG files.",
    )

    # Keep height separate so commands say exactly which image shape they want.
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional target image height for Pillow-loaded PNG/JPG files.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Print one compact sample summary for a split if a sample exists.
def print_first_sample(split_name: str, samples: list[GalaxyTrainingSample]) -> None:
    # Do not print a sample block for empty splits.
    if not samples:
        return

    # Keep the output short so it is useful from the terminal.
    first_sample = samples[0]
    print(f"first {split_name} sample:")
    print(f"  image_id: {first_sample.image_id}")
    print(f"  label: {first_sample.label}")
    print(f"  label_id: {first_sample.label_id}")
    print(f"  shape: {first_sample.tensor.shape}")


# Run the dataset split proof from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    try:
        # Convert optional resize arguments before creating split samples.
        target_size = target_size_from_args(args)

        # Load validated manifest records.
        records = load_manifest(Path(args.manifest_path))

        # Convert manifest records into grouped tiny training samples.
        dataset_splits = create_dataset_splits(
            records,
            Path(args.galaxy_data_root),
            skip_missing=not args.strict,
            target_size=target_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        # Print strict-mode failures and return a non-zero exit code.
        print(error)
        return 1

    # Print explicit accepted/skipped/rejected counts for each split.
    for split_name in ("train", "val", "test"):
        counts = dataset_splits.quality_counts_by_split()[split_name]
        print(
            f"{split_name}: accepted={counts['accepted']}, "
            f"skipped_missing={counts['skipped_missing']}, "
            f"rejected={counts['rejected']}"
        )

    # Keep every intentional exclusion visible, not just the aggregate count.
    for split_name in ("train", "val", "test"):
        for issue in dataset_splits.skipped_by_split[split_name]:
            print(f"{split_name} skipped_missing: {issue}")

    # Show one example training sample so the user can inspect the data shape.
    print_first_sample("train", dataset_splits.samples_by_split["train"])

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
