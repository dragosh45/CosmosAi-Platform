"""Galaxy dataset split helpers shared by training entrypoints."""

# Docs: docs/architecture.md Step 12 explains why split helpers live in this package.
# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.
# Docs: docs/architecture.md Step 20 explains accepted/skipped/rejected rows.
# Docs: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw maps this object flow.

# Import dataclass to return grouped samples as a structured object.
from dataclasses import dataclass

# Import Path to handle local and mounted image roots.
from pathlib import Path

# Import manifest grouping and training-sample creation from the shared package.
from cosmosai.galaxy.manifest import GalaxyManifestRecord, records_by_split
from cosmosai.galaxy.image_loader import validate_image_for_record
from cosmosai.galaxy.training_sample import (
    GalaxyLazyTrainingSample,
    GalaxyTrainingSample,
    create_lazy_training_sample_for_record,
    create_training_sample_for_record,
)


GalaxyDatasetSample = GalaxyTrainingSample | GalaxyLazyTrainingSample


# Store the samples and skipped rows for each dataset split.
@dataclass(frozen=True)
class GalaxyDatasetSplits:
    # Samples that were successfully loaded and preprocessed by split.
    samples_by_split: dict[str, list[GalaxyDatasetSample]]

    # Missing image rows intentionally excluded in permissive mode.
    skipped_by_split: dict[str, list[str]]

    # Invalid image/preprocessing rows. Successful results always keep this empty.
    rejected_by_split: dict[str, list[str]]

    # Return accepted, skipped, and rejected counts for each split.
    def quality_counts_by_split(self) -> dict[str, dict[str, int]]:
        return {
            split_name: {
                "accepted": len(self.samples_by_split[split_name]),
                "skipped_missing": len(self.skipped_by_split[split_name]),
                "rejected": len(self.rejected_by_split[split_name]),
            }
            for split_name in ("train", "val", "test")
        }

    # Return whole-dataset counts for a compact CLI summary.
    def quality_totals(self) -> dict[str, int]:
        counts = self.quality_counts_by_split().values()
        return {
            key: sum(split_counts[key] for split_counts in counts)
            for key in ("accepted", "skipped_missing", "rejected")
        }


class GalaxyDatasetQualityError(ValueError):
    """One or more dataset rows were invalid and must not be silently skipped."""

    def __init__(self, dataset_splits: GalaxyDatasetSplits) -> None:
        self.dataset_splits = dataset_splits
        totals = dataset_splits.quality_totals()
        lines = [
            "Galaxy dataset quality check failed: "
            f"accepted={totals['accepted']}, "
            f"skipped_missing={totals['skipped_missing']}, "
            f"rejected={totals['rejected']}"
        ]
        for split_name in ("train", "val", "test"):
            for issue in dataset_splits.skipped_by_split[split_name]:
                lines.append(f"- {split_name} skipped_missing: {issue}")
        for split_name in ("train", "val", "test"):
            for issue in dataset_splits.rejected_by_split[split_name]:
                lines.append(f"- {split_name} rejected: {issue}")
        super().__init__("\n".join(lines))


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
    target_size: tuple[int, int] | None = None,
    lazy: bool = True,
) -> GalaxyDatasetSplits:
    # Prepare grouped metadata-backed samples. Lazy samples do not retain pixels.
    samples_by_split: dict[str, list[GalaxyDatasetSample]] = empty_split_buckets()

    # Prepare grouped skip notes for rows that cannot load an image yet.
    skipped_by_split: dict[str, list[str]] = empty_split_buckets()

    # Invalid files/preprocessing are rejected even when missing files may be skipped.
    rejected_by_split: dict[str, list[str]] = empty_split_buckets()

    # Group records by train, val, and test before creating samples.
    grouped_records = records_by_split(records)

    # Walk each split so the output shape matches a future training loop.
    for split_name, split_records in grouped_records.items():
        # Convert each manifest record in this split into a training sample.
        for record in split_records:
            try:
                # Validate file integrity now so M50's quality policy remains
                # visible, but defer model-ready decoding until Dataset access.
                validate_image_for_record(
                    record,
                    galaxy_data_root,
                    target_size=target_size,
                )
                if lazy:
                    sample = create_lazy_training_sample_for_record(
                        record,
                        galaxy_data_root,
                        target_size=target_size,
                    )
                else:
                    sample = create_training_sample_for_record(
                        record,
                        galaxy_data_root,
                        target_size=target_size,
                    )
            except FileNotFoundError as error:
                issue = _format_dataset_issue(record, error)

                # Permissive mode intentionally excludes only absent files.
                if skip_missing:
                    skipped_by_split[split_name].append(issue)
                    continue

                # Strict mode treats an absent file as a rejected dataset row.
                rejected_by_split[split_name].append(issue)
                continue
            except (OSError, ValueError) as error:
                # Corrupt, unsupported, or invalid data must never look like a skip.
                rejected_by_split[split_name].append(
                    _format_dataset_issue(record, error)
                )
                continue

            # Store successfully created samples in the matching split bucket.
            samples_by_split[split_name].append(sample)

    # Build one inspectable result even when rejected rows will make the run fail.
    dataset_splits = GalaxyDatasetSplits(
        samples_by_split=samples_by_split,
        skipped_by_split=skipped_by_split,
        rejected_by_split=rejected_by_split,
    )

    # Rejected rows invalidate the run; the exception retains the full count report.
    if any(rejected_by_split.values()):
        raise GalaxyDatasetQualityError(dataset_splits)

    return dataset_splits


# Format one issue with the manifest row and stable image ID when available.
def _format_dataset_issue(
    record: GalaxyManifestRecord,
    error: Exception,
) -> str:
    row_prefix = (
        f"row {record.manifest_row}, "
        if record.manifest_row is not None
        else ""
    )
    return f"{row_prefix}image_id '{record.image_id}': {error}"
