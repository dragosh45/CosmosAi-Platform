#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 explains the current galaxy data path.

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

    # Allow strict mode when the caller wants missing images to fail the script.
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing or unreadable images instead of skipping them.",
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
        # Load validated manifest records.
        records = load_manifest(Path(args.manifest_path))

        # Convert manifest records into grouped tiny training samples.
        dataset_splits = create_dataset_splits(
            records,
            Path(args.galaxy_data_root),
            skip_missing=not args.strict,
        )
    except (FileNotFoundError, ValueError) as error:
        # Print strict-mode failures and return a non-zero exit code.
        print(error)
        return 1

    # Print how many usable samples exist in each split.
    for split_name in ("train", "val", "test"):
        samples = dataset_splits.samples_by_split[split_name]
        print(f"{split_name} samples: {len(samples)}")

    # Print how many rows were skipped in sample mode.
    for split_name in ("train", "val", "test"):
        skipped = dataset_splits.skipped_by_split[split_name]
        print(f"skipped {split_name}: {len(skipped)}")

    # Show one example training sample so the user can inspect the data shape.
    print_first_sample("train", dataset_splits.samples_by_split["train"])

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
