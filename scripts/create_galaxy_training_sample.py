#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 explains the current galaxy data path.

# Import argparse so the training sample proof can run from the command line.
import argparse

# Import dataclass to return one training sample as a structured object.
from dataclasses import dataclass

# Import Path to handle manifest and image root paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys

# Import preprocessing helpers so the sample includes normalized image values.
from preprocess_galaxy_image import (
    GalaxyImageTensor,
    load_and_preprocess_image_for_record,
)

# Import manifest helpers so the sample starts from the same validated data contract.
from load_galaxy_manifest import GalaxyManifestRecord, load_manifest


# Stable class ID mapping for the current galaxy labels.
LABEL_TO_ID = {
    "elliptical": 0,
    "spiral": 1,
    "lenticular": 2,
    "irregular": 3,
}


# Store one model-training-style example.
@dataclass(frozen=True)
class GalaxyTrainingSample:
    # Stable image identifier from the manifest.
    image_id: str

    # Human-readable galaxy label from the manifest.
    label: str

    # Numeric class ID used by future model training code.
    label_id: int

    # Dataset split from the manifest: train, val, or test.
    split: str

    # Normalized tensor-like image data.
    tensor: GalaxyImageTensor


# Convert a human-readable label into the stable numeric class ID.
def label_to_id(label: str) -> int:
    # Fail clearly when code receives a label outside the current contract.
    if label not in LABEL_TO_ID:
        allowed_labels = ", ".join(sorted(LABEL_TO_ID))
        raise ValueError(f"Unknown label '{label}' (allowed: {allowed_labels})")

    # Return the numeric ID used by future training code.
    return LABEL_TO_ID[label]


# Create one training sample from a manifest record.
def create_training_sample_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
) -> GalaxyTrainingSample:
    # Load and normalize the image referenced by this manifest row.
    tensor = load_and_preprocess_image_for_record(record, galaxy_data_root)

    # Convert the human label into a stable numeric target.
    numeric_label = label_to_id(record.label)

    # Return one future-training-style input/target pair.
    return GalaxyTrainingSample(
        image_id=record.image_id,
        label=record.label,
        label_id=numeric_label,
        split=record.split,
        tensor=tensor,
    )


# Parse command-line arguments for the one-sample proof.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Create one tiny CosmosAI galaxy training sample."
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

    # Accept an image ID so a specific manifest record can become a sample.
    parser.add_argument(
        "--image-id",
        default="gz2-000001",
        help="Manifest image_id to convert into a training sample.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Run the one-sample proof from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    # Load manifest records so the sample starts from validated metadata.
    records = load_manifest(Path(args.manifest_path))

    # Find the requested manifest record by image_id.
    record_by_id = {record.image_id: record for record in records}
    if args.image_id not in record_by_id:
        print(f"Image ID not found in manifest: {args.image_id}")
        return 1

    try:
        # Convert the selected manifest row into one training sample.
        sample = create_training_sample_for_record(
            record_by_id[args.image_id],
            Path(args.galaxy_data_root),
        )
    except (FileNotFoundError, ValueError) as error:
        # Print sample creation failures and return a non-zero exit code.
        print(error)
        return 1

    # Print a compact summary for human inspection.
    print(f"image_id: {sample.image_id}")
    print(f"label: {sample.label}")
    print(f"label_id: {sample.label_id}")
    print(f"split: {sample.split}")
    print(f"shape: {sample.tensor.shape}")
    print(f"normalized_values: {len(sample.tensor.values)}")
    print(f"first_values: {[round(value, 4) for value in sample.tensor.values[:6]]}")

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
