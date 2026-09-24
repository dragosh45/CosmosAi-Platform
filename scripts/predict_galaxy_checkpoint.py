#!/usr/bin/env python3

# Docs: docs/architecture.md Step 9 explains this checkpoint inference command.
# Docs: docs/local_classifier_demo.md explains prediction with --image-path.
# Docs: docs/architecture.md Step 12 explains why this is now a thin shared-package wrapper.
# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.
# Docs: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw maps this CLI to cosmosai.galaxy.

# Import argparse so the prediction proof can run from the command line.
import argparse

# Import Path to handle manifest, image root, and checkpoint paths.
from pathlib import Path

# Import sys so main() can return a process exit code.
import sys

# Store the repo root so this script can import the shared package when run by path.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Add the repo root for direct CLI runs like scripts/predict_galaxy_checkpoint.py.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the shared prediction result object for type hints and output formatting.
from cosmosai.galaxy.checkpoint_inference import (  # noqa: E402
    GalaxyCheckpointPrediction,
    load_sample_for_prediction,
    predict_from_checkpoint,
    predict_image_from_checkpoint,
    predict_sample_from_checkpoint,
)

# Re-export the shared label map so existing tests and notes keep working.
from cosmosai.galaxy.labels import LABEL_TO_ID  # noqa: E402


# Convert optional image width/height CLI values into a Pillow resize target.
def target_size_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    # If neither value is passed, use the checkpoint's saved resize policy.
    if args.image_width is None and args.image_height is None:
        return None

    # Require both dimensions so resizing cannot silently distort intent.
    if args.image_width is None or args.image_height is None:
        raise ValueError("--image-width and --image-height must be used together")

    # Both dimensions must be positive pixel counts.
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("--image-width and --image-height must be at least 1")

    # Pillow expects target size as width, height.
    return (args.image_width, args.image_height)


# Parse command-line arguments for the local prediction proof.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Predict a manifest sample or local image from a saved galaxy checkpoint."
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

    parser.add_argument(
        "--image-path",
        type=Path,
        help="Predict this local JPG/PNG instead of looking up a manifest image ID.",
    )

    # Accept a checkpoint path, defaulting to the local ignored model artifact.
    parser.add_argument(
        "--checkpoint-path",
        default="models/tiny_galaxy_cnn_baseline.pt",
        help="Path to the saved TinyGalaxyCNN checkpoint.",
    )

    # Optionally resize normal PNG/JPG inputs to the same size used in training.
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Optional width assertion; must match the checkpoint. Default: saved policy.",
    )

    # Keep height separate so commands say exactly which image shape they want.
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional height assertion; must match the checkpoint. Default: saved policy.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Print a compact prediction result for human inspection.
def print_prediction(prediction: GalaxyCheckpointPrediction) -> None:
    # Print the important file and sample identifiers.
    print("Galaxy checkpoint inference")
    print(f"  checkpoint_path: {prediction.checkpoint_path}")
    print(f"  image_id: {prediction.image_id}")
    if prediction.manifest_label is None:
        print("  known_label: not provided (prediction only)")
    else:
        print(f"  manifest_label_for_learning: {prediction.manifest_label}")
    print(f"  input_shape: {prediction.input_shape}")
    print(f"  preprocessing: {prediction.preprocessing.to_metadata()}")

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
        # Convert optional resize arguments before loading image data.
        target_size = target_size_from_args(args)

        if args.image_path is not None:
            prediction = predict_image_from_checkpoint(
                args.image_path,
                Path(args.checkpoint_path),
                target_size=target_size,
            )
        else:
            prediction = predict_from_checkpoint(
                Path(args.manifest_path),
                Path(args.galaxy_data_root),
                args.image_id,
                Path(args.checkpoint_path),
                target_size=target_size,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
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
