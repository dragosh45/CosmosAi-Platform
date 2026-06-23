#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 and docs/excalidraw/manifest_tooling_code_flow.excalidraw explain this validator.

# Import argparse to support validating any manifest path from the command line.
import argparse

# Import csv so the validator can read manifest files without extra dependencies.
import csv

# Import sys so the command can return a success or failure exit code.
import sys

# Import Path to handle filesystem paths clearly.
from pathlib import Path


# Columns that every galaxy manifest must contain.
REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "label",
    "split",
    "source",
}

# Labels currently allowed by docs/galaxy_data_contract.md.
ALLOWED_LABELS = {
    "elliptical",
    "spiral",
    "lenticular",
    "irregular",
}

# Dataset split values currently allowed by docs/galaxy_data_contract.md.
ALLOWED_SPLITS = {
    "train",
    "val",
    "test",
}


# Validate one galaxy manifest CSV file and return a list of human-readable errors.
def validate_manifest(manifest_path: Path) -> list[str]:
    # Collect all validation problems so the user can fix them in one pass.
    errors: list[str] = []

    # Fail clearly when the requested manifest file does not exist.
    if not manifest_path.exists():
        return [f"Manifest file does not exist: {manifest_path}"]

    # Open the CSV with newline="" so Python's csv module handles line endings.
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        # DictReader reads each row as a dictionary keyed by column name.
        reader = csv.DictReader(manifest_file)

        # Normalize fieldnames to an empty list when the CSV has no header row.
        fieldnames = reader.fieldnames or []

        # Check whether all required contract columns are present.
        missing_columns = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing_columns:
            errors.append(
                "Missing required columns: " + ", ".join(missing_columns)
            )

        # Stop early when required columns are missing because row checks may be noisy.
        if missing_columns:
            return errors

        # Track image IDs so duplicate IDs can be reported.
        seen_image_ids: set[str] = set()

        # Count rows so an empty manifest can be reported.
        row_count = 0

        # Validate every manifest row against the current data contract.
        for row_number, row in enumerate(reader, start=2):
            row_count += 1

            # Strip whitespace from the core fields before validating them.
            image_id = row["image_id"].strip()
            image_path = row["image_path"].strip()
            label = row["label"].strip()
            split = row["split"].strip()
            source = row["source"].strip()

            # image_id is the stable identifier used by code, logs, and APIs.
            if not image_id:
                errors.append(f"Row {row_number}: image_id is empty")
            elif image_id in seen_image_ids:
                errors.append(f"Row {row_number}: duplicate image_id '{image_id}'")
            else:
                seen_image_ids.add(image_id)

            # image_path should point to the image relative to the galaxy data root.
            if not image_path:
                errors.append(f"Row {row_number}: image_path is empty")

            # label must be one of the agreed galaxy morphology classes.
            if label not in ALLOWED_LABELS:
                errors.append(
                    f"Row {row_number}: invalid label '{label}' "
                    f"(allowed: {', '.join(sorted(ALLOWED_LABELS))})"
                )

            # split must tell training code how the row should be used.
            if split not in ALLOWED_SPLITS:
                errors.append(
                    f"Row {row_number}: invalid split '{split}' "
                    f"(allowed: {', '.join(sorted(ALLOWED_SPLITS))})"
                )

            # source records where this row came from, such as galaxy_zoo_2.
            if not source:
                errors.append(f"Row {row_number}: source is empty")

        # The manifest should contain at least one data row.
        if row_count == 0:
            errors.append("Manifest has no data rows")

    # Return every problem found. An empty list means the manifest is valid.
    return errors


# Parse command-line arguments for the validator script.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Validate a CosmosAI galaxy manifest CSV file."
    )

    # Accept a manifest path, defaulting to the tiny sample manifest in this repo.
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/samples/galaxy_manifest_sample.csv",
        help="Path to the galaxy manifest CSV file to validate.",
    )

    # Return the parsed arguments to the caller.
    return parser.parse_args()


# Run the validator as a command-line program.
def main() -> int:
    # Read the manifest path argument from the command line.
    args = parse_args()

    # Validate the manifest and collect all errors.
    errors = validate_manifest(Path(args.manifest_path))

    # Print errors and return a non-zero exit code when validation fails.
    if errors:
        print("Galaxy manifest validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    # Print a short success message when the manifest matches the contract.
    print(f"Galaxy manifest is valid: {args.manifest_path}")
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
