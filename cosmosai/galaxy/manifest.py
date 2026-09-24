"""Manifest loading helpers for galaxy training and inference paths."""

# Docs: docs/architecture.md Step 20 explains M50 data-quality boundaries.

# Import csv so the package can read manifest rows without extra dependencies.
import csv

# Import Counter to detect duplicate CSV column names before row parsing.
from collections import Counter

# Import dataclass helpers so records can retain their source row for diagnostics.
from dataclasses import dataclass, field

# Import Path to represent local and mounted filesystem paths.
from pathlib import Path

# Import shared contract values so validation matches the model label mapping.
from cosmosai.galaxy.labels import ALLOWED_LABELS, ALLOWED_SPLITS


# Columns that every galaxy manifest must contain.
REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "label",
    "split",
    "source",
}

# Default external galaxy data root from the local project docs.
DEFAULT_GALAXY_DATA_ROOT = Path("/media/h1dr0/KINGSTON/cosmosai-data/galaxy")


# Represent one galaxy image row from the manifest CSV.
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

    # Original CSV row, used only for data-quality messages. Programmatic records
    # can omit it, and it does not change record equality in existing callers.
    manifest_row: int | None = field(default=None, compare=False)


# Validate one galaxy manifest CSV file and return human-readable errors.
def validate_manifest(manifest_path: Path) -> list[str]:
    # Collect all validation problems so they can be fixed together.
    errors: list[str] = []

    # Fail clearly when the requested manifest file does not exist.
    if not manifest_path.exists():
        return [f"Manifest file does not exist: {manifest_path}"]

    try:
        # Open the CSV with newline="" so csv handles platform line endings.
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            # Strict mode reports malformed quoting instead of guessing a row shape.
            reader = csv.DictReader(manifest_file, strict=True)

            # Normalize fieldnames to an empty list when the CSV has no header row.
            fieldnames = reader.fieldnames or []

            # Duplicate/blank headers make DictReader mappings ambiguous.
            duplicate_columns = sorted(
                name for name, count in Counter(fieldnames).items() if count > 1
            )
            if duplicate_columns:
                errors.append(
                    "Duplicate column names: " + ", ".join(duplicate_columns)
                )
            if any(not name.strip() for name in fieldnames):
                errors.append("Manifest header contains an empty column name")

            # Check whether all required contract columns are present.
            missing_columns = sorted(REQUIRED_COLUMNS - set(fieldnames))
            if missing_columns:
                errors.append(
                    "Missing required columns: " + ", ".join(missing_columns)
                )

            # Row dictionaries are unreliable until the header contract is sound.
            if duplicate_columns or missing_columns or any(
                not name.strip() for name in fieldnames
            ):
                return errors

            # Track image IDs so duplicate IDs can be reported.
            seen_image_ids: set[str] = set()

            # Count rows so an empty manifest can be reported.
            row_count = 0

            # Validate every manifest row against the current data contract.
            for row_number, row in enumerate(reader, start=2):
                row_count += 1

                # DictReader stores extra values under None and missing values as None.
                extra_values = row.get(None) or []
                if extra_values:
                    errors.append(
                        f"Row {row_number}: has extra value(s) beyond the "
                        f"{len(fieldnames)} header columns: {extra_values}"
                    )

                missing_row_fields = {
                    name for name in fieldnames if row.get(name) is None
                }
                for name in sorted(missing_row_fields):
                    errors.append(
                        f"Row {row_number}, field '{name}': value is missing "
                        "because the row has fewer columns than the header"
                    )

                # Convert present values to stripped text without calling .strip() on None.
                values = {
                    name: (row.get(name) or "").strip()
                    for name in REQUIRED_COLUMNS
                }
                image_id = values["image_id"]
                image_path = values["image_path"]
                label = values["label"]
                split = values["split"]
                source = values["source"]

                # image_id is the stable identifier used by code, logs, and APIs.
                if not image_id and "image_id" not in missing_row_fields:
                    errors.append(f"Row {row_number}, field 'image_id': value is empty")
                elif image_id in seen_image_ids:
                    errors.append(
                        f"Row {row_number}, field 'image_id': duplicate value "
                        f"'{image_id}'"
                    )
                elif image_id:
                    seen_image_ids.add(image_id)

                # image_path should point to the image relative to the data root.
                if not image_path and "image_path" not in missing_row_fields:
                    errors.append(f"Row {row_number}, field 'image_path': value is empty")

                # label must be one of the agreed galaxy morphology classes.
                if not label and "label" not in missing_row_fields:
                    errors.append(f"Row {row_number}, field 'label': value is empty")
                elif label and label not in ALLOWED_LABELS:
                    errors.append(
                        f"Row {row_number}, field 'label': invalid label '{label}' "
                        f"(allowed: {', '.join(sorted(ALLOWED_LABELS))})"
                    )

                # split tells training code how the row should be used.
                if not split and "split" not in missing_row_fields:
                    errors.append(f"Row {row_number}, field 'split': value is empty")
                elif split and split not in ALLOWED_SPLITS:
                    errors.append(
                        f"Row {row_number}, field 'split': invalid split '{split}' "
                        f"(allowed: {', '.join(sorted(ALLOWED_SPLITS))})"
                    )

                # source records where this row came from, such as galaxy_zoo_2.
                if not source and "source" not in missing_row_fields:
                    errors.append(f"Row {row_number}, field 'source': value is empty")

            # The manifest should contain at least one data row.
            if row_count == 0:
                errors.append("Manifest has no data rows")
    except (OSError, UnicodeError, csv.Error) as error:
        # Convert filesystem/encoding/CSV parser failures into normal validation output.
        errors.append(f"Could not read manifest {manifest_path}: {error}")

    # Return every problem found. An empty list means the manifest is valid.
    return errors


# Resolve a manifest image path against the galaxy data root.
def resolve_image_path(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path = DEFAULT_GALAXY_DATA_ROOT,
) -> Path:
    # Absolute paths are returned unchanged so callers can opt out of root joining.
    if record.image_path.is_absolute():
        return record.image_path

    # Manifest image_path values are normally relative to the galaxy data root.
    return galaxy_data_root / record.image_path


# Resolve every manifest record into a mapping from image_id to full image path.
def resolved_image_paths_by_id(
    records: list[GalaxyManifestRecord],
    galaxy_data_root: Path = DEFAULT_GALAXY_DATA_ROOT,
) -> dict[str, Path]:
    # Build a lookup that training/inference code can use to find image files by ID.
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
    # Validate the manifest first so downstream code can trust required fields.
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
        reader = csv.DictReader(manifest_file, strict=True)

        # Convert each row into the small dataclass used by later code.
        for row_number, row in enumerate(reader, start=2):
            records.append(
                GalaxyManifestRecord(
                    image_id=row["image_id"].strip(),
                    image_path=Path(row["image_path"].strip()),
                    label=row["label"].strip(),
                    split=row["split"].strip(),
                    source=row["source"].strip(),
                    manifest_row=row_number,
                )
            )

    # Return the loaded records to the caller.
    return records


# Group manifest records by split so training code can request train/val/test rows.
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
