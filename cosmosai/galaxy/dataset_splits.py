"""Galaxy dataset split helpers shared by training entrypoints."""

# Docs: docs/architecture.md Step 12 explains why split helpers live in this package.
# Docs: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw maps this object flow.

# Import dataclass to return grouped samples as a structured object.
from dataclasses import dataclass

# Import Path to handle local and mounted image roots.
from pathlib import Path

# Import manifest grouping and training-sample creation from the shared package.
from cosmosai.galaxy.manifest import GalaxyManifestRecord, records_by_split
from cosmosai.galaxy.training_sample import (
    GalaxyTrainingSample,
    create_training_sample_for_record,
)


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
