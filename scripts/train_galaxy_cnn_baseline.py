#!/usr/bin/env python3

# Docs: docs/architecture.md Step 2 explains the current model-ready dataset path.

# Import argparse so the skeleton can run from the command line.
import argparse

# Import dataclass to return training-loop results as structured objects.
from dataclasses import dataclass

# Import math so the placeholder loop can compute softmax and loss values.
import math

# Import Path to handle manifest and image root paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys

# Import label metadata and sample objects from the existing sample builder.
from create_galaxy_training_sample import GalaxyTrainingSample, LABEL_TO_ID

# Import the split loader so this skeleton reuses the existing data pipeline.
from load_galaxy_dataset_splits import GalaxyDatasetSplits, create_dataset_splits

# Import the manifest loader so the CLI starts from the validated CSV contract.
from load_galaxy_manifest import load_manifest


# Store the output of one placeholder training step.
@dataclass(frozen=True)
class CnnSkeletonTrainingStep:
    # Image ID used for tracing the step back to the manifest row.
    image_id: str

    # Human-readable label from the manifest.
    label: str

    # Numeric target class ID from the manifest label.
    label_id: int

    # Placeholder predicted class ID from fake logits.
    predicted_label_id: int

    # Placeholder probability values for the current label set.
    probabilities: list[float]

    # Cross-entropy-style loss computed from placeholder probabilities.
    loss: float


# Store a compact summary of the skeleton training run.
@dataclass(frozen=True)
class CnnSkeletonTrainingSummary:
    # Number of epochs requested by the command or caller.
    epochs: int

    # Number of usable train samples loaded from the split loader.
    train_sample_count: int

    # Number of usable validation samples loaded from the split loader.
    val_sample_count: int

    # Number of usable test samples loaded from the split loader.
    test_sample_count: int

    # Number of manifest rows skipped because sample files are missing or unreadable.
    skipped_row_count: int

    # Number of placeholder training steps executed.
    training_steps: int

    # Average placeholder loss across all executed steps.
    average_loss: float

    # First step is kept for easy terminal inspection and focused tests.
    first_step: CnnSkeletonTrainingStep | None


# Load manifest rows and convert them into train, val, and test sample buckets.
def load_dataset_splits_from_manifest(
    manifest_path: Path,
    galaxy_data_root: Path,
    skip_missing: bool = True,
) -> GalaxyDatasetSplits:
    # Load validated manifest rows from the CSV file.
    records = load_manifest(manifest_path)

    # Reuse the existing split loader instead of duplicating data pipeline logic.
    return create_dataset_splits(
        records,
        galaxy_data_root,
        skip_missing=skip_missing,
    )


# Compute tiny placeholder logits from a sample tensor.
def placeholder_cnn_logits(sample: GalaxyTrainingSample) -> list[float]:
    # This is a fake feature, standing in for future convolutional feature extraction.
    mean_value = sum(sample.tensor.values) / len(sample.tensor.values)

    # This shape feature proves the loop can see tensor dimensions.
    height, width, channels = sample.tensor.shape
    shape_value = (height + width + channels) / 100.0

    # Return one score per label; these are not learned weights.
    return [
        0.10 + mean_value * 0.20,
        0.15 + mean_value * 0.15,
        0.05 + shape_value,
        0.08 + (1.0 - mean_value) * 0.10,
    ]


# Convert logits into probabilities that sum to about 1.0.
def softmax(logits: list[float]) -> list[float]:
    # Subtract the max value to keep exponentials numerically stable.
    max_logit = max(logits)

    # Convert each logit into a positive exponential score.
    exp_values = [math.exp(logit - max_logit) for logit in logits]

    # Sum the exponential scores so each score can be normalized.
    total = sum(exp_values)

    # Return normalized probabilities.
    return [value / total for value in exp_values]


# Compute simple cross-entropy loss for one correct class.
def cross_entropy_loss(probabilities: list[float], label_id: int) -> float:
    # Guard against impossible class IDs before indexing the probability list.
    if label_id < 0 or label_id >= len(probabilities):
        raise ValueError(f"Label ID is outside probability range: {label_id}")

    # Clamp the probability to avoid log(0) in future edge cases.
    correct_probability = max(probabilities[label_id], 1e-12)

    # Cross entropy gets smaller when the correct class probability gets larger.
    return -math.log(correct_probability)


# Run one placeholder training step for one sample.
def run_placeholder_training_step(
    sample: GalaxyTrainingSample,
) -> CnnSkeletonTrainingStep:
    # Compute fake CNN-like output scores from tensor values.
    logits = placeholder_cnn_logits(sample)

    # Keep the placeholder output aligned with the current label mapping.
    if len(logits) != len(LABEL_TO_ID):
        raise ValueError("Placeholder logits must match the label count")

    # Convert the output scores into probabilities.
    probabilities = softmax(logits)

    # Pick the class with the highest placeholder probability.
    predicted_label_id = max(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
    )

    # Compute a loss value against the known manifest label.
    loss = cross_entropy_loss(probabilities, sample.label_id)

    # Return the step details for summary printing and tests.
    return CnnSkeletonTrainingStep(
        image_id=sample.image_id,
        label=sample.label,
        label_id=sample.label_id,
        predicted_label_id=predicted_label_id,
        probabilities=probabilities,
        loss=loss,
    )


# Run the tiny CNN-shaped training loop over available train samples.
def run_training_loop(
    dataset_splits: GalaxyDatasetSplits,
    epochs: int = 1,
) -> CnnSkeletonTrainingSummary:
    # Fail clearly because zero or negative epochs do not make sense.
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    # Read split buckets from the existing data object.
    train_samples = dataset_splits.samples_by_split["train"]
    val_samples = dataset_splits.samples_by_split["val"]
    test_samples = dataset_splits.samples_by_split["test"]

    # Store each placeholder step result.
    steps: list[CnnSkeletonTrainingStep] = []

    # Repeat over epochs like a real training loop will do later.
    for _epoch in range(epochs):
        # Iterate over train samples only; validation/test are not trained on.
        for sample in train_samples:
            steps.append(run_placeholder_training_step(sample))

    # Calculate average loss, using 0.0 when there are no train samples.
    average_loss = (
        sum(step.loss for step in steps) / len(steps)
        if steps
        else 0.0
    )

    # Count skipped manifest rows across all splits.
    skipped_row_count = sum(
        len(skipped_rows)
        for skipped_rows in dataset_splits.skipped_by_split.values()
    )

    # Return a compact summary of the skeleton run.
    return CnnSkeletonTrainingSummary(
        epochs=epochs,
        train_sample_count=len(train_samples),
        val_sample_count=len(val_samples),
        test_sample_count=len(test_samples),
        skipped_row_count=skipped_row_count,
        training_steps=len(steps),
        average_loss=average_loss,
        first_step=steps[0] if steps else None,
    )


# Parse command-line arguments for the training-loop skeleton.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Run a tiny CNN-shaped CosmosAI training loop skeleton."
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

    # Accept epoch count so the loop shape can be inspected.
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of skeleton epochs to run.",
    )

    # Allow strict mode when the caller wants missing images to fail the script.
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing or unreadable images instead of skipping them.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Print a compact summary for human inspection.
def print_training_summary(summary: CnnSkeletonTrainingSummary) -> None:
    # Print the high-level loop shape.
    print("CNN baseline skeleton")
    print(f"epochs: {summary.epochs}")
    print(f"train samples: {summary.train_sample_count}")
    print(f"val samples: {summary.val_sample_count}")
    print(f"test samples: {summary.test_sample_count}")
    print(f"skipped rows: {summary.skipped_row_count}")
    print(f"training steps: {summary.training_steps}")
    print(f"average placeholder loss: {summary.average_loss:.4f}")

    # Print one step so the data can be inspected from the terminal.
    if summary.first_step is not None:
        print("first training step:")
        print(f"  image_id: {summary.first_step.image_id}")
        print(f"  true_label: {summary.first_step.label}")
        print(f"  true_label_id: {summary.first_step.label_id}")
        print(f"  predicted_label_id: {summary.first_step.predicted_label_id}")
        rounded_probabilities = [
            round(probability, 4)
            for probability in summary.first_step.probabilities
        ]
        print(f"  placeholder_probabilities: {rounded_probabilities}")

    # Say clearly that this script does not train weights yet.
    print("status: skeleton only - no weights updated")


# Run the training-loop skeleton from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    try:
        # Load data through the existing manifest and split pipeline.
        dataset_splits = load_dataset_splits_from_manifest(
            Path(args.manifest_path),
            Path(args.galaxy_data_root),
            skip_missing=not args.strict,
        )

        # Run the placeholder training-loop shape.
        summary = run_training_loop(dataset_splits, epochs=args.epochs)
    except (FileNotFoundError, ValueError) as error:
        # Print loading or configuration failures and return a non-zero exit code.
        print(error)
        return 1

    # Print the skeleton training summary.
    print_training_summary(summary)

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
