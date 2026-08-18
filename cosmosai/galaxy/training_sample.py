"""Galaxy training-sample helpers shared by CLI and service inference."""

# Import dataclass to return one training sample as a structured object.
from dataclasses import dataclass

# Import Path to handle local and mounted image roots.
from pathlib import Path

# Import shared label mapping and preprocessing helpers.
from cosmosai.galaxy.labels import LABEL_TO_ID
from cosmosai.galaxy.manifest import GalaxyManifestRecord
from cosmosai.galaxy.preprocessing import (
    GalaxyImageTensor,
    load_and_preprocess_image_for_record,
)


# Store one model-training-style example.
@dataclass(frozen=True)
class GalaxyTrainingSample:
    # Stable image identifier from the manifest.
    image_id: str

    # Human-readable galaxy label from the manifest.
    label: str

    # Numeric class ID used by model training code.
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

    # Return the numeric ID used by training and checkpoint metadata.
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

    # Return one input/target pair.
    return GalaxyTrainingSample(
        image_id=record.image_id,
        label=record.label,
        label_id=numeric_label,
        split=record.split,
        tensor=tensor,
    )

