#!/usr/bin/env python3

# Docs: docs/architecture.md Step 9 explains this checkpoint inference command.

# Import argparse so the prediction proof can run from the command line.
import argparse

# Import dataclass so prediction output can be passed around as a clear object.
from dataclasses import dataclass

# Import Path to handle manifest, image root, and checkpoint paths.
from pathlib import Path

# Import sys so main() can return a process exit code.
import sys

# Import sample creation so inference reuses the same preprocessing pipeline.
from create_galaxy_training_sample import GalaxyTrainingSample, LABEL_TO_ID

# Import manifest loading so the command starts from the validated CSV contract.
from load_galaxy_manifest import load_manifest

# Import checkpoint and forward-pass helpers from the current CNN baseline script.
from train_galaxy_cnn_baseline import (
    load_torch_checkpoint,
    run_torch_forward_pass,
)


# Reverse the training label map so predictions can print human-readable labels.
ID_TO_LABEL = {
    label_id: label
    for label, label_id in LABEL_TO_ID.items()
}


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


# Find one manifest row and convert it into the existing model-ready sample object.
def load_sample_for_prediction(
    manifest_path: Path,
    galaxy_data_root: Path,
    image_id: str,
) -> GalaxyTrainingSample:
    # Load validated manifest records through the existing manifest pipeline.
    records = load_manifest(manifest_path)

    # Build a lookup so the requested image_id can be found clearly.
    record_by_id = {
        record.image_id: record
        for record in records
    }
    if image_id not in record_by_id:
        raise ValueError(f"Image ID not found in manifest: {image_id}")

    # Import here to keep the top-level import list focused on checkpoint inference.
    from create_galaxy_training_sample import create_training_sample_for_record

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


# Parse command-line arguments for the local prediction proof.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Predict one tiny galaxy sample from a saved PyTorch checkpoint."
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

    # Accept an image ID so the user can choose which manifest row to predict.
    parser.add_argument(
        "--image-id",
        default="gz2-000001",
        help="Manifest image_id to load and predict.",
    )

    # Accept a checkpoint path, defaulting to the local ignored model artifact.
    parser.add_argument(
        "--checkpoint-path",
        default="models/tiny_galaxy_cnn_baseline.pt",
        help="Path to the saved TinyGalaxyCNN checkpoint.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Print a compact prediction result for human inspection.
def print_prediction(prediction: GalaxyCheckpointPrediction) -> None:
    # Print the important file and sample identifiers.
    print("Galaxy checkpoint inference")
    print(f"  checkpoint_path: {prediction.checkpoint_path}")
    print(f"  image_id: {prediction.image_id}")
    print(f"  manifest_label_for_learning: {prediction.manifest_label}")
    print(f"  input_shape: {prediction.input_shape}")

    # Print the prediction from the loaded checkpoint.
    print(f"  predicted_label_id: {prediction.predicted_label_id}")
    print(f"  predicted_label: {prediction.predicted_label}")

    # Round model outputs so terminal output is readable.
    rounded_logits = [
        round(logit, 4)
        for logit in prediction.logits
    ]
    rounded_probabilities = [
        round(probability, 4)
        for probability in prediction.probabilities
    ]
    print(f"  logits: {rounded_logits}")
    print(f"  probabilities: {rounded_probabilities}")
    print("  status: loaded checkpoint inference completed - no training")


# Run the checkpoint prediction proof from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    try:
        # Load the checkpoint and one sample, then run inference.
        prediction = predict_from_checkpoint(
            Path(args.manifest_path),
            Path(args.galaxy_data_root),
            args.image_id,
            Path(args.checkpoint_path),
        )
    except (FileNotFoundError, ValueError) as error:
        # Print user-fixable path or data errors and return failure.
        print(error)
        return 1

    # Print the local inference result.
    print_prediction(prediction)

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
