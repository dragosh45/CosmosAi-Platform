"""Checkpoint inference helpers shared by the CLI and FastAPI service."""

# Import dataclass so prediction output can be passed around as a clear object.
from dataclasses import dataclass

# Import Path to handle manifest, image root, and checkpoint paths.
from pathlib import Path

# Import shared labels, manifest loading, model helpers, and sample creation.
from cosmosai.galaxy.labels import ID_TO_LABEL
from cosmosai.galaxy.manifest import load_manifest
from cosmosai.galaxy.model import load_torch_checkpoint, run_torch_forward_pass
from cosmosai.galaxy.training_sample import (
    GalaxyTrainingSample,
    create_training_sample_for_record,
)


# Store one checkpoint-based prediction result.
@dataclass(frozen=True)
class GalaxyCheckpointPrediction:
    # Filesystem path to the checkpoint that supplied the model weights.
    checkpoint_path: Path

    # Manifest image ID used for this tiny inference proof.
    image_id: str

    # The manifest label is printed for learning, but inference does not use it.
    manifest_label: str

    # PyTorch input shape: batch, channels, height, width.
    input_shape: tuple[int, int, int, int]

    # Predicted numeric class ID from the loaded checkpoint model.
    predicted_label_id: int

    # Human-readable predicted class label.
    predicted_label: str

    # Raw class scores produced by the loaded model.
    logits: list[float]

    # Softmax probabilities produced from the logits.
    probabilities: list[float]


# Find one manifest row and convert it into the model-ready sample object.
def load_sample_for_prediction(
    manifest_path: Path,
    galaxy_data_root: Path,
    image_id: str,
) -> GalaxyTrainingSample:
    # Load validated manifest records through the shared manifest pipeline.
    records = load_manifest(manifest_path)

    # Build a lookup so the requested image_id can be found clearly.
    record_by_id = {
        record.image_id: record
        for record in records
    }
    if image_id not in record_by_id:
        raise ValueError(f"Image ID not found in manifest: {image_id}")

    # This creates normalized tensor values. The label is attached for inspection only.
    return create_training_sample_for_record(
        record_by_id[image_id],
        galaxy_data_root,
    )


# Run inference from a saved checkpoint on one prepared sample.
def predict_sample_from_checkpoint(
    checkpoint_path: Path,
    sample: GalaxyTrainingSample,
) -> GalaxyCheckpointPrediction:
    # Load a fresh TinyGalaxyCNN and fill it with saved checkpoint weights.
    model = load_torch_checkpoint(checkpoint_path)

    # Forward pass only: checkpoint inference predicts without retraining.
    forward_result = run_torch_forward_pass(sample, model=model)

    # Translate the numeric prediction into the label name used by the manifest.
    predicted_label = ID_TO_LABEL[forward_result.predicted_label_id]

    # Return a readable prediction object for command output and tests.
    return GalaxyCheckpointPrediction(
        checkpoint_path=checkpoint_path,
        image_id=sample.image_id,
        manifest_label=sample.label,
        input_shape=forward_result.input_shape,
        predicted_label_id=forward_result.predicted_label_id,
        predicted_label=predicted_label,
        logits=forward_result.logits,
        probabilities=forward_result.probabilities,
    )


# Load data and checkpoint, then predict one image_id.
def predict_from_checkpoint(
    manifest_path: Path,
    galaxy_data_root: Path,
    image_id: str,
    checkpoint_path: Path,
) -> GalaxyCheckpointPrediction:
    # Prepare the same normalized input tensor shape used during training.
    sample = load_sample_for_prediction(
        manifest_path,
        galaxy_data_root,
        image_id,
    )

    # Run checkpoint inference on that prepared sample.
    return predict_sample_from_checkpoint(checkpoint_path, sample)

