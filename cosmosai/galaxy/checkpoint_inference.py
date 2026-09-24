"""Checkpoint inference helpers shared by the CLI and FastAPI service."""

# Docs: docs/architecture.md Steps 12 and 19 explain shared inference and the saved input recipe.
# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.
# Docs: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw maps the inference flow.
# Docs: docs/local_classifier_demo.md adds label-free file/upload prediction.

# Import dataclass so prediction output can be passed around as a clear object.
from dataclasses import dataclass

# Import Path to handle manifest, image root, and checkpoint paths.
from pathlib import Path

# Import shared labels, manifest loading, model helpers, and sample creation.
from cosmosai.galaxy.labels import ID_TO_LABEL
from cosmosai.galaxy.image_loader import load_image_file
from cosmosai.galaxy.manifest import load_manifest
from cosmosai.galaxy.preprocessing import GalaxyPreprocessingPolicy, preprocess_image
from cosmosai.galaxy.model import (
    LoadedGalaxyCheckpoint,
    load_torch_checkpoint_bundle,
    run_torch_forward_pass,
)
from cosmosai.galaxy.training_sample import (
    GalaxyInferenceSample,
    GalaxyTrainingSample,
    create_training_sample_for_record,
)


# Store one checkpoint-based prediction result.
@dataclass(frozen=True)
class GalaxyCheckpointPrediction:
    # Filesystem path to the checkpoint that supplied the model weights.
    checkpoint_path: Path

    # Manifest ID or local filename identifying this prediction.
    image_id: str

    # The manifest label is printed for learning, but inference does not use it.
    manifest_label: str | None

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

    # The CLI prints this so automatic resizing is visible when flags are omitted.
    preprocessing: GalaxyPreprocessingPolicy


# Find one manifest row and convert it into the model-ready sample object.
def load_sample_for_prediction(
    manifest_path: Path,
    galaxy_data_root: Path,
    image_id: str,
    target_size: tuple[int, int] | None = None,
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
        target_size=target_size,
    )


# Run inference from a saved checkpoint on one prepared sample.
def predict_sample_from_checkpoint(
    checkpoint_path: Path,
    sample: GalaxyTrainingSample,
) -> GalaxyCheckpointPrediction:
    # This lower-level helper expects an already RGB/normalized sample.
    checkpoint = load_torch_checkpoint_bundle(checkpoint_path)
    return _predict_loaded_checkpoint(checkpoint_path, checkpoint, sample)


def _predict_loaded_checkpoint(
    checkpoint_path: Path,
    checkpoint: LoadedGalaxyCheckpoint,
    sample: GalaxyTrainingSample | GalaxyInferenceSample,
) -> GalaxyCheckpointPrediction:
    checkpoint.preprocessing.validate_tensor_shape(sample.tensor.shape)

    # Forward pass only: checkpoint inference predicts without retraining.
    forward_result = run_torch_forward_pass(sample, model=checkpoint.model)

    # Translate the numeric prediction into the label name used by the manifest.
    predicted_label = ID_TO_LABEL[forward_result.predicted_label_id]

    # Return a readable prediction object for command output and tests.
    return GalaxyCheckpointPrediction(
        checkpoint_path=checkpoint_path,
        image_id=sample.image_id,
        manifest_label=None if isinstance(sample, GalaxyInferenceSample) else sample.label,
        input_shape=forward_result.input_shape,
        predicted_label_id=forward_result.predicted_label_id,
        predicted_label=predicted_label,
        logits=forward_result.logits,
        probabilities=forward_result.probabilities,
        preprocessing=checkpoint.preprocessing,
    )


# Load data and checkpoint, then predict one image_id.
def predict_from_checkpoint(
    manifest_path: Path,
    galaxy_data_root: Path,
    image_id: str,
    checkpoint_path: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyCheckpointPrediction:
    # Read the recipe BEFORE pixels: e.g. a 13x9 PNG must become 5x3 even when
    # the API supplies no resize flags. Load the checkpoint only once per prediction.
    checkpoint = load_torch_checkpoint_bundle(checkpoint_path)
    target_size = checkpoint.preprocessing.resolve_target_size(target_size)
    sample = load_sample_for_prediction(
        manifest_path,
        galaxy_data_root,
        image_id,
        target_size=target_size,
    )

    # Run checkpoint inference on that prepared sample.
    return _predict_loaded_checkpoint(checkpoint_path, checkpoint, sample)


def predict_image_from_checkpoint(
    image_path: Path,
    checkpoint_path: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyCheckpointPrediction:
    """Predict a local image without a manifest, known label, or training step.

    Read the saved input recipe before pixels, exactly as in manifest inference.
    Both entry paths share normalization and the same read-only forward pass.
    """
    checkpoint = load_torch_checkpoint_bundle(checkpoint_path)
    target_size = checkpoint.preprocessing.resolve_target_size(target_size)
    image = load_image_file(image_path, target_size=target_size)
    sample = GalaxyInferenceSample(
        image_id=image_path.name,
        tensor=preprocess_image(image),
    )
    return _predict_loaded_checkpoint(checkpoint_path, checkpoint, sample)
