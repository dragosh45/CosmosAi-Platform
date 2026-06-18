#!/usr/bin/env python3

# Import argparse so this loader can also be inspected from the command line.
import argparse

# Import csv so the loader can read manifest rows without extra dependencies.
import csv

# Import dataclass to define a small structured record for each manifest row.
from dataclasses import dataclass

# Import Path to represent manifest and image paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return useful exit codes.
import sys

# Import the validator so loading only happens after the manifest contract passes.
from validate_galaxy_manifest import validate_manifest


# Default external galaxy data root from docs/galaxy_data_contract.md.
DEFAULT_GALAXY_DATA_ROOT = Path("/media/h1dr0/KINGSTON/cosmosai-data/galaxy")


# Represent one galaxy image row from the manifest.
@dataclass(frozen=True)
class GalaxyManifestRecord:
    # Stable image identifier used by logs, APIs, and future training code.
    image_id: str

    # Path to the image relative to the galaxy data root.
    image_path: Path

    # Galaxy morphology label used as the correct training answer.
    label: str

    # Dataset split: train, val, or test.
    split: str

    # Dataset source name, such as galaxy_zoo_2.
    source: str


# Resolve a manifest image path against the external galaxy data root.
def resolve_image_path(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path = DEFAULT_GALAXY_DATA_ROOT,
) -> Path:
    # Absolute paths are returned unchanged so advanced callers can opt out of root joining.
    if record.image_path.is_absolute():
        return record.image_path

    # Manifest image_path values are normally relative to the galaxy data root.
    return galaxy_data_root / record.image_path


# Resolve every manifest record into a mapping from image_id to full image path.
def resolved_image_paths_by_id(
    records: list[GalaxyManifestRecord],
    galaxy_data_root: Path = DEFAULT_GALAXY_DATA_ROOT,
) -> dict[str, Path]:
    # Build a lookup that future training code can use to find image files by ID.
    return {
        record.image_id: resolve_image_path(record, galaxy_data_root)
        for record in records
    }


# Return resolved image paths that do not exist on disk.
def missing_image_paths(
    records: list[GalaxyManifestRecord],
    galaxy_data_root: Path = DEFAULT_GALAXY_DATA_ROOT,
) -> dict[str, Path]:
    # Resolve every manifest path against the configured data root.
    paths_by_id = resolved_image_paths_by_id(records, galaxy_data_root)

    # Keep only paths that are currently missing.
    return {
        image_id: image_path
        for image_id, image_path in paths_by_id.items()
        if not image_path.exists()
    }


# Load a validated galaxy manifest CSV into structured records.
def load_manifest(manifest_path: Path) -> list[GalaxyManifestRecord]:
    # Validate the manifest first so downstream code can trust the required fields.
    errors = validate_manifest(manifest_path)

    # Refuse to load invalid data because future training would be misleading.
    if errors:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Galaxy manifest is invalid:\n{joined_errors}")

    # Store each parsed manifest row as a typed record.
    records: list[GalaxyManifestRecord] = []

    # Read the CSV rows after validation succeeds.
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        # DictReader returns one dictionary per manifest row.
        reader = csv.DictReader(manifest_file)

        # Convert each row into the small dataclass used by future training code.
        for row in reader:
            records.append(
                GalaxyManifestRecord(
                    image_id=row["image_id"].strip(),
                    image_path=Path(row["image_path"].strip()),
                    label=row["label"].strip(),
                    split=row["split"].strip(),
                    source=row["source"].strip(),
                )
            )

    # Return the loaded records to the caller.
    return records


# Group manifest records by split so future training code can request train/val/test rows.
def records_by_split(
    records: list[GalaxyManifestRecord],
) -> dict[str, list[GalaxyManifestRecord]]:
    # Start with all known splits so missing splits still have predictable keys.
    grouped: dict[str, list[GalaxyManifestRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Put each record into its split bucket.
    for record in records:
        grouped[record.split].append(record)

    # Return the grouped records.
    return grouped


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
