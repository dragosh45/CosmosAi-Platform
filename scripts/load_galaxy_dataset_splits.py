#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 explains the current galaxy data path.

# Import argparse so the split loader can run from the command line.
import argparse

# Import dataclass to return grouped samples as a structured object.
from dataclasses import dataclass

# Import Path to handle manifest and image root paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys

# Import sample creation helpers so each manifest row becomes a training sample.
from create_galaxy_training_sample import (
    GalaxyTrainingSample,
    create_training_sample_for_record,
)

# Import manifest helpers so the split loader starts from validated metadata.
from load_galaxy_manifest import GalaxyManifestRecord, load_manifest, records_by_split


# Store the samples and skipped rows for each dataset split.
@dataclass(frozen=True)
class GalaxyDatasetSplits:
    # Samples that were successfully loaded and preprocessed by split.
    samples_by_split: dict[str, list[GalaxyTrainingSample]]

    # Manifest image IDs skipped because the local tiny image file is not ready yet.
    skipped_by_split: dict[str, list[str]]


# Create empty train, val, and test buckets.
def empty_split_buckets() -> dict[str, list]:
    # Return predictable split keys even if a manifest has no rows for one split.
    return {
        "train": [],
        "val": [],
        "test": [],
    }


# Convert manifest records into model-ready samples grouped by split.
def create_dataset_splits(
    records: list[GalaxyManifestRecord],
    galaxy_data_root: Path,
    skip_missing: bool = True,
) -> GalaxyDatasetSplits:
    # Prepare grouped output samples.
    samples_by_split: dict[str, list[GalaxyTrainingSample]] = empty_split_buckets()

    # Prepare grouped skip notes for rows that cannot load an image yet.
    skipped_by_split: dict[str, list[str]] = empty_split_buckets()

    # Group records by train, val, and test before creating samples.
    grouped_records = records_by_split(records)

    # Walk each split so the output shape matches a future training loop.
    for split_name, split_records in grouped_records.items():
        # Convert each manifest record in this split into a training sample.
        for record in split_records:
            try:
                # Load image pixels, normalize them, and attach the numeric label.
                sample = create_training_sample_for_record(record, galaxy_data_root)
            except (FileNotFoundError, ValueError) as error:
                # In sample mode, skip rows whose real image files are not present yet.
                if skip_missing:
                    skipped_by_split[split_name].append(
                        f"{record.image_id}: {error}"
                    )
                    continue

                # In strict mode, fail immediately so real data problems are visible.
                raise

            # Store successfully created samples in the matching split bucket.
            samples_by_split[split_name].append(sample)

    # Return both successful samples and skipped rows for inspection.
    return GalaxyDatasetSplits(
        samples_by_split=samples_by_split,
        skipped_by_split=skipped_by_split,
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
