#!/usr/bin/env python3

# Docs: docs/architecture.md Step 3 explains the placeholder skeleton path.
# Docs: docs/architecture.md Step 4 explains the PyTorch forward-pass proof.
# Docs: docs/architecture.md Step 5 explains the first real PyTorch weight update.
# Docs: docs/architecture.md Step 6 explains the tiny real PyTorch training loop.
# Docs: docs/architecture.md Step 7 explains the read-only evaluation proof.
# Docs: docs/architecture.md Step 8 explains the tiny checkpoint save/load proof.
# Docs: docs/architecture.md Step 12 explains why shared CNN helpers live in cosmosai.galaxy.
# Docs: docs/architecture.md Step 15 explains the tiny PyTorch batch training proof.
# Docs: docs/architecture.md Step 17 explains the PyTorch Dataset/DataLoader proof.
# Docs: docs/architecture.md Step 18 explains DataLoader-driven real training.
# Docs: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw maps this script to the shared package.
# Docs: docs/excalidraw/galaxy_pytorch_batch_training_flow.excalidraw maps the batch path.

# Import argparse so the skeleton can run from the command line.
import argparse
import json
from collections import Counter

# Import dataclass helpers to return and serialize training/evaluation results.
from dataclasses import asdict, dataclass

# Import math so the placeholder loop can compute softmax and loss values.
import math

# Import Path to handle manifest and image root paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys
from typing import Callable

# Add the repo root to imports so this script can run directly from any folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import torch so training-step code can build losses, targets, and optimizers.
import torch

# Import shared label metadata used by training, checkpoint inference, and service code.
from cosmosai.galaxy.labels import ID_TO_LABEL, LABEL_TO_ID

# Import shared split helpers so training uses package data objects, not script-owned ones.
from cosmosai.galaxy.dataset_splits import GalaxyDatasetSplits, create_dataset_splits

# Import shared manifest loading so training starts from the package data contract.
from cosmosai.galaxy.manifest import load_manifest
from cosmosai.galaxy.preprocessing import GalaxyPreprocessingPolicy

# Import shared model/checkpoint helpers so the script uses the same CNN code as serving.
from cosmosai.galaxy.model import (
    TorchGalaxyBatch,
    TinyGalaxyCNN,
    TorchCheckpointRoundTripResult,
    TorchForwardPassResult,
    create_tiny_galaxy_cnn,
    galaxy_samples_to_torch_batch,
    galaxy_tensor_to_torch_image,
    load_torch_checkpoint,
    run_torch_checkpoint_round_trip,
    run_torch_forward_pass,
    save_torch_checkpoint,
)

# Import the same DataLoader factory for the preview and real training loop.
from cosmosai.galaxy.torch_dataset import (
    TorchDataLoaderBatchResult,
    create_galaxy_dataloader,
    preview_first_dataloader_batch,
)

# Import the shared training sample object produced by the data pipeline.
from cosmosai.galaxy.training_sample import GalaxyTrainingSample


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

    # Number of manifest rows intentionally skipped because image paths are missing.
    skipped_row_count: int

    # Number of placeholder training steps executed.
    training_steps: int

    # Average placeholder loss across all executed steps.
    average_loss: float

    # First step is kept for easy terminal inspection and focused tests.
    first_step: CnnSkeletonTrainingStep | None


# Store the output of one real PyTorch training step.
@dataclass(frozen=True)
class TorchTrainingStepResult:
    # Image ID used for tracing the training step back to the manifest row.
    image_id: str

    # Human-readable correct label from the manifest, such as "spiral".
    label: str

    # Numeric correct class ID, such as 1 for "spiral".
    label_id: int

    # Tensor shape sent into the PyTorch model: batch, channels, height, width.
    input_shape: tuple[int, int, int, int]

    # Loss before the optimizer changed the model weights.
    loss_before: float

    # Loss after one tiny optimizer update on the same sample.
    loss_after: float

    # Class probabilities before the weight update.
    probabilities_before: list[float]

    # Class probabilities after the weight update.
    probabilities_after: list[float]

    # Predicted class ID before the weight update.
    predicted_label_id_before: int

    # Predicted class ID after the weight update.
    predicted_label_id_after: int

    # True when at least one tracked model weight changed.
    weight_changed: bool


# Store the output of a tiny PyTorch batch-shape proof.
@dataclass(frozen=True)
class TorchBatchShapeResult:
    # Batch size requested by the command or caller.
    requested_batch_size: int

    # Number of samples actually included in this batch.
    actual_batch_size: int

    # Image IDs included in this batch, preserving manifest order.
    image_ids: list[str]

    # Human-readable labels included in this batch.
    labels: list[str]

    # Numeric class IDs included in this batch.
    label_ids: list[int]

    # Image tensor shape sent into the model: batch, channels, height, width.
    image_tensor_shape: tuple[int, int, int, int]

    # Label tensor shape sent into the loss function.
    label_tensor_shape: tuple[int]


# Store the summary for one epoch in the real PyTorch training loop.
@dataclass(frozen=True)
class TorchTrainingEpochResult:
    # Epoch number shown to humans, starting at 1 instead of 0.
    epoch: int

    # Number of train samples processed in this epoch.
    training_steps: int

    # Average real PyTorch loss for this epoch.
    average_loss: float


# Store the output of the tiny real PyTorch training loop.
@dataclass(frozen=True)
class TorchTrainingLoopSummary:
    # Number of epochs requested by the command or caller.
    epochs: int

    # Number of usable train samples loaded from the split loader.
    train_sample_count: int

    # Number of usable validation samples loaded from the split loader.
    val_sample_count: int

    # Number of usable test samples loaded from the split loader.
    test_sample_count: int

    # Number of manifest rows intentionally skipped because image paths are missing.
    skipped_row_count: int

    # Number of real PyTorch optimizer steps executed.
    training_steps: int

    # Number of samples grouped into each training batch.
    batch_size: int

    # Number of train batches processed in each epoch.
    train_batch_count: int

    # Average real PyTorch loss across all train steps.
    average_loss: float

    # Loss on the first train sample before the loop starts updating weights.
    first_loss: float | None

    # Loss on the first train sample after all loop updates finish.
    final_loss: float | None

    # Correct-label probability on the first train sample before training.
    first_correct_probability: float | None

    # Correct-label probability on the first train sample after training.
    final_correct_probability: float | None

    # Final predicted class ID for the first train sample after training.
    final_predicted_label_id: int | None

    # True when at least one tracked model weight changed during the loop.
    weight_changed: bool

    # One compact loss summary per epoch.
    epoch_results: list[TorchTrainingEpochResult]


# Store one class row in a read-only evaluation report.
@dataclass(frozen=True)
class TorchClassMetrics:
    # Numeric class ID and human-readable label.
    label_id: int
    label: str

    # Number of true examples and predictions for this class.
    support: int
    predicted_count: int

    # Correct predictions for this class.
    true_positives: int

    # Precision, recall, and F1; recall/F1 are None when the class is absent.
    precision: float
    recall: float | None
    f1: float | None


# Store one concrete error example for later review.
@dataclass(frozen=True)
class TorchMisclassifiedExample:
    # Stable manifest ID and relative image path.
    image_id: str
    image_path: str

    # Human-readable and numeric true/predicted labels.
    true_label_id: int
    true_label: str
    predicted_label_id: int
    predicted_label: str

    # Model confidence for the true and predicted classes, when available.
    true_probability: float | None
    predicted_probability: float | None


# Store metrics shared by a model evaluation and the majority baseline.
@dataclass(frozen=True)
class TorchClassificationMetrics:
    # Rows are true classes; columns are predicted classes.
    confusion_matrix: list[list[int]]

    # One precision/recall/F1 row per supported class.
    class_metrics: list[TorchClassMetrics]

    # Total errors and a bounded list of examples for human review.
    misclassified_count: int
    misclassified_examples: list[TorchMisclassifiedExample]


# Store read-only evaluation metrics for one dataset split.
@dataclass(frozen=True)
class TorchEvaluationSplitResult:
    # Split being evaluated: train, val, or test.
    split_name: str

    # Number of usable samples in this split.
    sample_count: int

    # Number of predictions that matched the known label_id.
    correct_predictions: int

    # Accuracy means correct_predictions / sample_count; None means no samples.
    accuracy: float | None

    # Average loss on this split; lower means the correct class got more confidence.
    average_loss: float

    # First sample true class ID, kept so terminal output has a concrete example.
    first_true_label_id: int | None

    # First sample predicted class ID, kept for concrete inspection.
    first_predicted_label_id: int | None

    # First sample probability assigned to its correct class.
    first_correct_probability: float | None

    # Structured class metrics, confusion matrix, and error examples.
    confusion_matrix: list[list[int]]
    class_metrics: list[TorchClassMetrics]
    misclassified_count: int
    misclassified_examples: list[TorchMisclassifiedExample]


# Store the simple majority-class comparison for one split.
@dataclass(frozen=True)
class TorchMajorityBaselineResult:
    # The majority class is calculated from the training split only.
    split_name: str
    majority_label_id: int
    majority_label: str

    # Accuracy of always predicting that class on this split.
    sample_count: int
    correct_predictions: int
    accuracy: float | None

    # Keep the same detailed metrics shape as the CNN report.
    confusion_matrix: list[list[int]]
    class_metrics: list[TorchClassMetrics]


# Store the temporary fake model behind a model-like interface.
@dataclass(frozen=True)
class PlaceholderGalaxyModel:
    # Number of output classes this placeholder should produce logits for.
    label_count: int = len(LABEL_TO_ID)

    # Compute fake logits with the same outside shape as a future model forward pass.
    def forward(self, sample: GalaxyTrainingSample) -> list[float]:
        # This is a fake feature, standing in for future convolutional feature extraction.
        mean_value = sum(sample.tensor.values) / len(sample.tensor.values)

        # This shape feature proves the model interface can see tensor dimensions.
        height, width, channels = sample.tensor.shape
        shape_value = (height + width + channels) / 100.0

        # Return one score per label; these are not learned weights.
        logits = [
            0.10 + mean_value * 0.20,
            0.15 + mean_value * 0.15,
            0.05 + shape_value,
            0.08 + (1.0 - mean_value) * 0.10,
        ]

        # Keep the placeholder output aligned with the current label mapping.
        if len(logits) != self.label_count:
            raise ValueError("Placeholder logits must match the label count")

        # Return fake model output scores.
        return logits


# Load manifest rows and convert them into train, val, and test sample buckets.
def load_dataset_splits_from_manifest(
    manifest_path: Path,
    galaxy_data_root: Path,
    skip_missing: bool = True,
    target_size: tuple[int, int] | None = None,
) -> GalaxyDatasetSplits:
    # Load validated manifest rows from the CSV file.
    records = load_manifest(manifest_path)

    # Reuse the existing split loader instead of duplicating data pipeline logic.
    return create_dataset_splits(
        records,
        galaxy_data_root,
        skip_missing=skip_missing,
        target_size=target_size,
    )


# Compute tiny placeholder logits from a sample tensor.
def placeholder_cnn_logits(sample: GalaxyTrainingSample) -> list[float]:
    # Keep this helper for simple concept links while the model interface matures.
    return PlaceholderGalaxyModel().forward(sample)


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


# Convert optional image width/height CLI values into a Pillow resize target.
def target_size_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    # If neither value is passed, do not resize images.
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


# Compute simple cross-entropy loss for one correct class.
def cross_entropy_loss(probabilities: list[float], label_id: int) -> float:
    # Guard against impossible class IDs before indexing the probability list.
    if label_id < 0 or label_id >= len(probabilities):
        raise ValueError(f"Label ID is outside probability range: {label_id}")

    # Clamp the probability to avoid log(0) in future edge cases.
    correct_probability = max(probabilities[label_id], 1e-12)

    # Cross entropy gets smaller when the correct class probability gets larger.
    return -math.log(correct_probability)


# Run one isolated PyTorch training step and prove at least one weight changed.
# Concept docs: docs/concepts_explanations.md -> "PyTorch Training Step Proof".
def run_torch_training_step(
    sample: GalaxyTrainingSample,
    model: TinyGalaxyCNN | None = None,
    learning_rate: float = 0.1,
) -> TorchTrainingStepResult:
    # Use the provided model or create a deterministic tiny CNN for repeatable output.
    active_model = model if model is not None else create_tiny_galaxy_cnn()

    # Training mode tells PyTorch this model is being updated, not only inspected.
    active_model.train()

    # Convert our project sample into the NCHW tensor expected by Conv2d.
    torch_image = galaxy_tensor_to_torch_image(sample)

    # CrossEntropyLoss compares raw logits with the correct class ID.
    # Important: PyTorch CrossEntropyLoss expects logits, not softmax probabilities.
    loss_function = torch.nn.CrossEntropyLoss()

    # SGD is a simple optimizer: it nudges weights opposite the loss gradient.
    optimizer = torch.optim.SGD(active_model.parameters(), lr=learning_rate)

    # The target tensor contains the correct class ID for the batch.
    # Example for one spiral image: tensor([1]).
    target_label = torch.tensor([sample.label_id], dtype=torch.long)

    # Clone one weight matrix before training so we can prove learning changed it.
    classifier_weight_before = active_model.classifier.weight.detach().clone()

    # Clear old gradients from any previous calculation on this optimizer.
    # Gradient concept: each gradient says how one weight affected the loss.
    optimizer.zero_grad()

    # Forward pass: image numbers move through CNN layers and become raw class scores.
    # Forward concept: image tensor + current weights -> logits; weights are only used.
    logits_before = active_model(torch_image)

    # Loss compares the raw scores with the known label_id from the manifest.
    # Loss concept: lower loss means the model scored the correct label better.
    loss_before_tensor = loss_function(logits_before, target_label)

    # Backward pass: calculate gradients for each trainable weight.
    # Backward concept: PyTorch traces from loss back to the weights.
    loss_before_tensor.backward()

    # Optimizer step: update weights using the gradients and learning rate.
    # This is the moment the model's internal learned numbers actually change.
    optimizer.step()

    # After the update, inspect the same sample without building a second gradient graph.
    with torch.no_grad():
        logits_after = active_model(torch_image)
        loss_after_tensor = loss_function(logits_after, target_label)
        probabilities_before_tensor = torch.softmax(logits_before, dim=1)
        probabilities_after_tensor = torch.softmax(logits_after, dim=1)

    # Compare the saved classifier weights with the current classifier weights.
    weight_changed = not torch.equal(
        classifier_weight_before,
        active_model.classifier.weight.detach(),
    )

    # Return plain Python values so tests and command output are easy to read.
    return TorchTrainingStepResult(
        image_id=sample.image_id,
        label=sample.label,
        label_id=sample.label_id,
        input_shape=tuple(torch_image.shape),
        loss_before=float(loss_before_tensor.item()),
        loss_after=float(loss_after_tensor.item()),
        probabilities_before=probabilities_before_tensor.squeeze(0).tolist(),
        probabilities_after=probabilities_after_tensor.squeeze(0).tolist(),
        predicted_label_id_before=int(torch.argmax(logits_before, dim=1).item()),
        predicted_label_id_after=int(torch.argmax(logits_after, dim=1).item()),
        weight_changed=weight_changed,
    )


# Split samples into small ordered batches.
def create_sample_batches(
    samples: list[GalaxyTrainingSample],
    batch_size: int,
) -> list[list[GalaxyTrainingSample]]:
    # Batch size must be positive because zero-size batches cannot train a model.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    # Slice the sample list into ordered chunks.
    # Example with 2 samples and batch_size=2: one batch with both samples.
    return [
        samples[start_index : start_index + batch_size]
        for start_index in range(0, len(samples), batch_size)
    ]


# Build one batch from the start of the train split so the shape can be inspected.
def run_torch_batch_shape_proof(
    samples: list[GalaxyTrainingSample],
    batch_size: int = 1,
) -> TorchBatchShapeResult | None:
    # If no train samples exist, there is no batch to inspect.
    if not samples:
        return None

    # Reuse the batching helper so preview output and training use the same shape.
    first_batch_samples = create_sample_batches(samples, batch_size)[0]
    first_batch: TorchGalaxyBatch = galaxy_samples_to_torch_batch(first_batch_samples)

    # Return plain Python values so command output and tests are easy to read.
    return TorchBatchShapeResult(
        requested_batch_size=batch_size,
        actual_batch_size=len(first_batch_samples),
        image_ids=first_batch.image_ids,
        labels=first_batch.labels,
        label_ids=first_batch.label_tensor.tolist(),
        image_tensor_shape=tuple(first_batch.image_tensor.shape),
        label_tensor_shape=tuple(first_batch.label_tensor.shape),
    )


# Run a tiny real PyTorch training loop over the available train split.
# Concept docs: docs/concepts_explanations.md -> "PyTorch Training Loop Proof".
def run_torch_training_loop(
    dataset_splits: GalaxyDatasetSplits,
    epochs: int = 1,
    model: TinyGalaxyCNN | None = None,
    learning_rate: float = 0.1,
    batch_size: int = 1,
    on_training_batch: Callable[[int, int, dict[str, object]], None] | None = None,
) -> TorchTrainingLoopSummary:
    # Fail clearly because zero or negative epochs do not make sense.
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    # Fail clearly because zero or negative batch size cannot build tensors.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    # Read split buckets. Only train samples are used for weight updates.
    train_samples = dataset_splits.samples_by_split["train"]
    val_samples = dataset_splits.samples_by_split["val"]
    test_samples = dataset_splits.samples_by_split["test"]

    # DataLoader asks Dataset for CHW images, then stacks them into NCHW batches.
    # Keep manifest order for comparison with the earlier manual-batching proof.
    # This Dataset still wraps images already in RAM; it is not lazy file loading.
    train_loader = create_galaxy_dataloader(
        train_samples, batch_size=batch_size, shuffle=False
    )

    # Count skipped manifest rows across all splits for the printed summary.
    skipped_row_count = sum(
        len(skipped_rows)
        for skipped_rows in dataset_splits.skipped_by_split.values()
    )

    # Use the provided model or create one deterministic tiny CNN for the whole loop.
    active_model = model if model is not None else create_tiny_galaxy_cnn()

    # Training mode tells PyTorch this model is being updated.
    active_model.train()

    # CrossEntropyLoss takes raw logits and the correct class IDs.
    loss_function = torch.nn.CrossEntropyLoss()

    # One optimizer stays attached to the same model across all epochs.
    # This matters because each optimizer.step() continues from the latest weights,
    # instead of restarting learning from a fresh random model.
    optimizer = torch.optim.SGD(active_model.parameters(), lr=learning_rate)

    # Clone one trainable weight matrix before the loop so we can prove it changed.
    classifier_weight_before = active_model.classifier.weight.detach().clone()

    # Capture starting loss/probability on the first train sample before updates.
    first_loss: float | None = None
    first_correct_probability: float | None = None
    if train_samples:
        first_sample = train_samples[0]
        first_image = galaxy_tensor_to_torch_image(first_sample)
        first_target = torch.tensor([first_sample.label_id], dtype=torch.long)
        with torch.no_grad():
            first_logits = active_model(first_image)
            first_loss_tensor = loss_function(first_logits, first_target)
            first_probabilities = torch.softmax(first_logits, dim=1)
        first_loss = float(first_loss_tensor.item())
        first_correct_probability = float(
            first_probabilities[0, first_sample.label_id].item()
        )

    # Count optimizer updates separately from images seen: one batch = one update.
    training_steps = 0
    total_loss_sum = 0.0
    total_sample_count = 0

    # Store one compact summary per epoch for learning/inspection.
    epoch_results: list[TorchTrainingEpochResult] = []

    # Loop over epochs: one epoch means one pass over the available train samples.
    for epoch_index in range(epochs):
        epoch_steps = 0
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        # Loop over train batches only; validation and test stay read-only.
        # Starting a new iteration of the loader gives a fresh pass each epoch.
        # The final batch is kept even when it has fewer than batch_size images.
        for torch_batch in train_loader:
            # Default collation stacks images without blending their pixel values.
            # Example: two RGB 3x3 images -> (2, 3, 3, 3); labels -> (2,).
            image_tensor = torch_batch["image_tensor"]
            label_tensor = torch_batch["label_tensor"]

            # Optional read-only observers receive the actual DataLoader batch,
            # including IDs and labels, before the model sees its image tensor.
            # Normal training passes no callback and follows the same path.
            if on_training_batch is not None:
                on_training_batch(epoch_index + 1, epoch_steps + 1, torch_batch)

            # Clear old gradients before this batch's calculation.
            optimizer.zero_grad()

            # Forward pass: current weights produce logits for every image in the batch.
            # Same concept as the one-step proof, but repeated inside the loop.
            logits = active_model(image_tensor)

            # Loss: compare each logit row with its matching label ID.
            # CrossEntropyLoss averages the batch into one scalar loss.
            loss_tensor = loss_function(logits, label_tensor)

            # Backward pass: calculate gradients for the current weights.
            # Gradients tell the optimizer which way to move each weight.
            loss_tensor.backward()

            # Optimizer step: update weights using those gradients.
            # Repeating this is what makes the loop different from one forward pass.
            optimizer.step()

            # Loss is already a batch mean. Weight it by the actual image count
            # for reporting, so a final one-image batch is not counted like two.
            # This accounting does not change the gradients or optimizer update.
            loss_value = float(loss_tensor.item())
            actual_batch_size = int(label_tensor.shape[0])
            epoch_steps += 1
            epoch_loss_sum += loss_value * actual_batch_size
            epoch_sample_count += actual_batch_size

        # Average this epoch's losses, using 0.0 if there are no train samples.
        epoch_average_loss = (
            epoch_loss_sum / epoch_sample_count
            if epoch_sample_count
            else 0.0
        )
        training_steps += epoch_steps
        total_loss_sum += epoch_loss_sum
        total_sample_count += epoch_sample_count

        # Store the epoch summary with a human-facing epoch number.
        epoch_results.append(
            TorchTrainingEpochResult(
                epoch=epoch_index + 1,
                training_steps=epoch_steps,
                average_loss=epoch_average_loss,
            )
        )

    # Capture final loss/probability on the first train sample after all updates.
    final_loss: float | None = None
    final_correct_probability: float | None = None
    final_predicted_label_id: int | None = None
    if train_samples:
        first_sample = train_samples[0]
        first_image = galaxy_tensor_to_torch_image(first_sample)
        first_target = torch.tensor([first_sample.label_id], dtype=torch.long)
        with torch.no_grad():
            final_logits = active_model(first_image)
            final_loss_tensor = loss_function(final_logits, first_target)
            final_probabilities = torch.softmax(final_logits, dim=1)
        final_loss = float(final_loss_tensor.item())
        final_correct_probability = float(
            final_probabilities[0, first_sample.label_id].item()
        )
        final_predicted_label_id = int(torch.argmax(final_logits, dim=1).item())

    # Compare the saved classifier weights with the final classifier weights.
    weight_changed = not torch.equal(
        classifier_weight_before,
        active_model.classifier.weight.detach(),
    )

    # Average over all image visits across epochs, not over unequal-sized batches.
    average_loss = (
        total_loss_sum / total_sample_count
        if total_sample_count
        else 0.0
    )

    # Return the loop summary for command output and tests.
    return TorchTrainingLoopSummary(
        epochs=epochs,
        train_sample_count=len(train_samples),
        val_sample_count=len(val_samples),
        test_sample_count=len(test_samples),
        skipped_row_count=skipped_row_count,
        training_steps=training_steps,
        batch_size=batch_size,
        train_batch_count=len(train_loader),
        average_loss=average_loss,
        first_loss=first_loss,
        final_loss=final_loss,
        first_correct_probability=first_correct_probability,
        final_correct_probability=final_correct_probability,
        final_predicted_label_id=final_predicted_label_id,
        weight_changed=weight_changed,
        epoch_results=epoch_results,
    )


def calculate_classification_metrics(
    true_label_ids: list[int],
    predicted_label_ids: list[int],
    image_ids: list[str] | None = None,
    image_paths: list[str] | None = None,
    probabilities: list[list[float]] | None = None,
    max_error_examples: int = 20,
) -> TorchClassificationMetrics:
    """Calculate confusion, class metrics, and bounded error examples."""
    class_count = len(ID_TO_LABEL)
    if len(true_label_ids) != len(predicted_label_ids):
        raise ValueError("True and predicted label lists must have equal length")
    if image_ids is not None and len(image_ids) != len(true_label_ids):
        raise ValueError("Image IDs must match the number of predictions")
    if image_paths is not None and len(image_paths) != len(true_label_ids):
        raise ValueError("Image paths must match the number of predictions")
    if probabilities is not None and len(probabilities) != len(true_label_ids):
        raise ValueError("Probabilities must match the number of predictions")
    if max_error_examples < 0:
        raise ValueError("max_error_examples must be non-negative")

    confusion_matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    supports = [0 for _ in range(class_count)]
    predicted_counts = [0 for _ in range(class_count)]
    true_positives = [0 for _ in range(class_count)]
    misclassified_examples: list[TorchMisclassifiedExample] = []
    misclassified_count = 0

    for index, (true_id, predicted_id) in enumerate(
        zip(true_label_ids, predicted_label_ids)
    ):
        if true_id not in ID_TO_LABEL or predicted_id not in ID_TO_LABEL:
            raise ValueError(
                f"Labels must be between 0 and {class_count - 1}; "
                f"got true={true_id}, predicted={predicted_id}"
            )
        confusion_matrix[true_id][predicted_id] += 1
        supports[true_id] += 1
        predicted_counts[predicted_id] += 1
        if true_id == predicted_id:
            true_positives[true_id] += 1
            continue

        misclassified_count += 1
        if len(misclassified_examples) >= max_error_examples:
            continue
        prediction = probabilities[index] if probabilities is not None else None
        true_probability = (
            float(prediction[true_id])
            if prediction is not None and true_id < len(prediction)
            else None
        )
        predicted_probability = (
            float(prediction[predicted_id])
            if prediction is not None and predicted_id < len(prediction)
            else None
        )
        misclassified_examples.append(
            TorchMisclassifiedExample(
                image_id=image_ids[index] if image_ids is not None else "",
                image_path=image_paths[index] if image_paths is not None else "",
                true_label_id=true_id,
                true_label=ID_TO_LABEL[true_id],
                predicted_label_id=predicted_id,
                predicted_label=ID_TO_LABEL[predicted_id],
                true_probability=true_probability,
                predicted_probability=predicted_probability,
            )
        )

    class_metrics: list[TorchClassMetrics] = []
    for label_id in range(class_count):
        precision = (
            true_positives[label_id] / predicted_counts[label_id]
            if predicted_counts[label_id]
            else 0.0
        )
        recall = (
            true_positives[label_id] / supports[label_id]
            if supports[label_id]
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else None if recall is None else 0.0
        )
        class_metrics.append(
            TorchClassMetrics(
                label_id=label_id,
                label=ID_TO_LABEL[label_id],
                support=supports[label_id],
                predicted_count=predicted_counts[label_id],
                true_positives=true_positives[label_id],
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return TorchClassificationMetrics(
        confusion_matrix=confusion_matrix,
        class_metrics=class_metrics,
        misclassified_count=misclassified_count,
        misclassified_examples=misclassified_examples,
    )


def _sample_image_path(sample: GalaxyTrainingSample) -> str:
    record = getattr(sample, "record", None)
    if record is not None:
        return str(record.image_path)
    tensor = getattr(sample, "tensor", None)
    return str(tensor.path) if tensor is not None else ""


# Evaluate one split with the current model weights, without changing weights.
def evaluate_torch_split(
    split_name: str,
    samples: list[GalaxyTrainingSample],
    model: TinyGalaxyCNN,
    max_error_examples: int = 20,
) -> TorchEvaluationSplitResult:
    # Evaluation mode means "use the model for inspection/prediction only."
    model.eval()

    # CrossEntropyLoss gives a real loss number for each read-only prediction.
    loss_function = torch.nn.CrossEntropyLoss()

    # Store metric pieces so the caller can see accuracy and detailed errors.
    losses: list[float] = []
    true_label_ids: list[int] = []
    predicted_label_ids: list[int] = []
    image_ids: list[str] = []
    image_paths: list[str] = []
    probabilities_by_sample: list[list[float]] = []
    correct_predictions = 0
    first_true_label_id: int | None = None
    first_predicted_label_id: int | None = None
    first_correct_probability: float | None = None

    # no_grad keeps evaluation read-only: no gradient memory, no weight updates.
    with torch.no_grad():
        # Loop over this split exactly once, like a future validation/test pass.
        for sample_index, sample in enumerate(samples):
            # Convert project data into the PyTorch image tensor used by the CNN.
            torch_image = galaxy_tensor_to_torch_image(sample)

            # The target tensor stores the known correct class ID for this image.
            target_label = torch.tensor([sample.label_id], dtype=torch.long)

            # Forward pass only: current weights turn the image into logits.
            logits = model(torch_image)

            # Loss measures how wrong the logits are for the known label_id.
            loss_tensor = loss_function(logits, target_label)
            losses.append(float(loss_tensor.item()))

            # Softmax turns logits into probabilities for readable metrics.
            probabilities = torch.softmax(logits, dim=1)

            # The predicted label is the class with the highest logit.
            predicted_label_id = int(torch.argmax(logits, dim=1).item())
            true_label_ids.append(sample.label_id)
            predicted_label_ids.append(predicted_label_id)
            image_ids.append(sample.image_id)
            image_paths.append(_sample_image_path(sample))
            probabilities_by_sample.append(probabilities[0].tolist())
            if predicted_label_id == sample.label_id:
                correct_predictions += 1

            # Keep one concrete example so the terminal output is understandable.
            if sample_index == 0:
                first_true_label_id = sample.label_id
                first_predicted_label_id = predicted_label_id
                first_correct_probability = float(
                    probabilities[0, sample.label_id].item()
                )

    # Accuracy only exists when there is at least one sample to evaluate.
    accuracy = (
        correct_predictions / len(samples)
        if samples
        else None
    )

    # Average loss is 0.0 for empty splits so command output stays numeric.
    average_loss = (
        sum(losses) / len(losses)
        if losses
        else 0.0
    )

    classification = calculate_classification_metrics(
        true_label_ids,
        predicted_label_ids,
        image_ids=image_ids,
        image_paths=image_paths,
        probabilities=probabilities_by_sample,
        max_error_examples=max_error_examples,
    )

    # Return compact read-only metrics for this split.
    return TorchEvaluationSplitResult(
        split_name=split_name,
        sample_count=len(samples),
        correct_predictions=correct_predictions,
        accuracy=accuracy,
        average_loss=average_loss,
        first_true_label_id=first_true_label_id,
        first_predicted_label_id=first_predicted_label_id,
        first_correct_probability=first_correct_probability,
        confusion_matrix=classification.confusion_matrix,
        class_metrics=classification.class_metrics,
        misclassified_count=classification.misclassified_count,
        misclassified_examples=classification.misclassified_examples,
    )


# Evaluate train, validation, and test splits with the same trained model.
def evaluate_torch_model(
    dataset_splits: GalaxyDatasetSplits,
    model: TinyGalaxyCNN,
    max_error_examples: int = 20,
) -> dict[str, TorchEvaluationSplitResult]:
    # Evaluate every split read-only; only training code is allowed to update weights.
    return {
        split_name: evaluate_torch_split(
            split_name,
            dataset_splits.samples_by_split[split_name],
            model,
            max_error_examples=max_error_examples,
        )
        for split_name in ("train", "val", "test")
    }


def evaluate_majority_baseline(
    dataset_splits: GalaxyDatasetSplits,
    max_error_examples: int = 20,
) -> dict[str, TorchMajorityBaselineResult] | None:
    """Evaluate always predicting the most common training label."""
    train_samples = dataset_splits.samples_by_split["train"]
    if not train_samples:
        return None

    counts = Counter(sample.label_id for sample in train_samples)
    majority_label_id = min(
        label_id for label_id, count in counts.items()
        if count == max(counts.values())
    )
    results: dict[str, TorchMajorityBaselineResult] = {}
    for split_name in ("train", "val", "test"):
        samples = dataset_splits.samples_by_split[split_name]
        true_ids = [sample.label_id for sample in samples]
        predicted_ids = [majority_label_id for _ in samples]
        classification = calculate_classification_metrics(
            true_ids,
            predicted_ids,
            image_ids=[sample.image_id for sample in samples],
            image_paths=[_sample_image_path(sample) for sample in samples],
            max_error_examples=max_error_examples,
        )
        correct_predictions = sum(
            true_id == majority_label_id for true_id in true_ids
        )
        results[split_name] = TorchMajorityBaselineResult(
            split_name=split_name,
            majority_label_id=majority_label_id,
            majority_label=ID_TO_LABEL[majority_label_id],
            sample_count=len(samples),
            correct_predictions=correct_predictions,
            accuracy=(
                correct_predictions / len(samples)
                if samples
                else None
            ),
            confusion_matrix=classification.confusion_matrix,
            class_metrics=classification.class_metrics,
        )
    return results


def build_evaluation_report(
    manifest_path: Path,
    galaxy_data_root: Path,
    args: argparse.Namespace,
    dataset_splits: GalaxyDatasetSplits,
    training_summary: TorchTrainingLoopSummary | None,
    evaluation_results: dict[str, TorchEvaluationSplitResult] | None,
    majority_baseline: dict[str, TorchMajorityBaselineResult] | None,
    checkpoint_result: TorchCheckpointRoundTripResult | None,
) -> dict[str, object]:
    """Build a JSON-safe record of the reproducible baseline evaluation."""
    split_counts = {
        split_name: len(dataset_splits.samples_by_split[split_name])
        for split_name in ("train", "val", "test")
    }
    class_counts = {
        split_name: dict(
            sorted(
                Counter(
                    sample.label
                    for sample in dataset_splits.samples_by_split[split_name]
                ).items()
            )
        )
        for split_name in ("train", "val", "test")
    }
    checkpoint = None
    if checkpoint_result is not None:
        checkpoint = {
            "path": str(checkpoint_result.checkpoint_path),
            "logits_match": checkpoint_result.logits_match,
            "probabilities_match": checkpoint_result.probabilities_match,
        }

    return {
        "report_version": 1,
        "classes": [
            {"label_id": label_id, "label": ID_TO_LABEL[label_id]}
            for label_id in sorted(ID_TO_LABEL)
        ],
        "dataset": {
            "manifest_path": str(manifest_path),
            "galaxy_data_root": str(galaxy_data_root),
            "split_counts": split_counts,
            "class_counts_by_split": class_counts,
            "quality_counts": dataset_splits.quality_counts_by_split(),
        },
        "configuration": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "strict": args.strict,
            "target_size": (
                [args.image_width, args.image_height]
                if args.image_width is not None and args.image_height is not None
                else None
            ),
            "max_error_examples": args.max_error_examples,
            "model": "TinyGalaxyCNN",
            "optimizer": "SGD",
            "loss": "CrossEntropyLoss",
        },
        "training": asdict(training_summary) if training_summary is not None else None,
        "evaluation": (
            {
                split_name: asdict(result)
                for split_name, result in evaluation_results.items()
            }
            if evaluation_results is not None
            else None
        ),
        "majority_baseline": (
            {
                split_name: asdict(result)
                for split_name, result in majority_baseline.items()
            }
            if majority_baseline is not None
            else None
        ),
        "checkpoint": checkpoint,
        "interpretation": (
            "Bounded pilot report only; these metrics do not establish useful "
            "galaxy-classification accuracy."
        ),
    }


def write_evaluation_report(path: Path, report: dict[str, object]) -> None:
    """Write one indented, deterministic JSON evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# Run one placeholder training step for one sample.
def run_placeholder_training_step(
    sample: GalaxyTrainingSample,
    model: PlaceholderGalaxyModel | None = None,
) -> CnnSkeletonTrainingStep:
    # Use the provided model or create the default placeholder model.
    active_model = model if model is not None else PlaceholderGalaxyModel()

    # Compute fake CNN-like output scores through a model-like forward method.
    logits = active_model.forward(sample)

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
    model: PlaceholderGalaxyModel | None = None,
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

    # Use one model object for the whole training run, like real training will.
    active_model = model if model is not None else PlaceholderGalaxyModel()

    # Repeat over epochs like a real training loop will do later.
    for _epoch in range(epochs):
        # Iterate over train samples only; validation/test are not trained on.
        for sample in train_samples:
            steps.append(run_placeholder_training_step(sample, active_model))

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

    # Accept PyTorch batch size so Milestone 45 can group samples together.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of train samples per tiny PyTorch batch.",
    )

    # Optionally resize normal PNG/JPG inputs before they become tensors.
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Optional target image width for Pillow-loaded PNG/JPG files.",
    )

    # Keep height separate so commands say exactly which image shape they want.
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional target image height for Pillow-loaded PNG/JPG files.",
    )

    # Allow strict mode when the caller wants missing images to fail the script.
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing or unreadable images instead of skipping them.",
    )

    # Accept a local checkpoint path for the tiny save/load proof.
    parser.add_argument(
        "--checkpoint-path",
        default="models/tiny_galaxy_cnn_baseline.pt",
        help="Path where the tiny trained PyTorch checkpoint should be saved.",
    )

    # Keep the optimizer configuration explicit and reproducible.
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="SGD learning rate used by the real PyTorch loop.",
    )

    # Seed model initialization so repeated baseline reports are comparable.
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="PyTorch model initialization seed.",
    )

    # Save structured evaluation metrics when a report path is supplied.
    parser.add_argument(
        "--evaluation-report-path",
        default=None,
        help="Optional JSON path for class metrics, errors, and baseline comparison.",
    )

    # Bound the number of error rows retained in the report.
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=20,
        help="Maximum misclassified image examples per split in the report.",
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


# Print the PyTorch forward-pass proof separately from the placeholder summary.
def print_torch_forward_result(result: TorchForwardPassResult | None) -> None:
    # Explain why there may be no PyTorch proof when no train samples are available.
    if result is None:
        print("PyTorch forward proof: skipped because no train sample was loaded")
        return

    # Print the real PyTorch forward-pass result.
    print("PyTorch forward proof")
    print(f"  image_id: {result.image_id}")
    print(f"  input_shape: {result.input_shape}")
    rounded_logits = [round(logit, 4) for logit in result.logits]
    rounded_probabilities = [
        round(probability, 4)
        for probability in result.probabilities
    ]
    print(f"  logits: {rounded_logits}")
    print(f"  probabilities: {rounded_probabilities}")
    print(f"  predicted_label_id: {result.predicted_label_id}")
    print("  status: real PyTorch forward pass only - no weights updated")


# Print the PyTorch batch-shape proof separately from one-image forward output.
def print_torch_batch_shape_result(result: TorchBatchShapeResult | None) -> None:
    # Explain why there may be no batch proof when no train samples are available.
    if result is None:
        print("PyTorch batch proof: skipped because no train samples were loaded")
        return

    # Print enough details to see how multiple samples become one tensor.
    print("PyTorch batch proof")
    print(f"  requested_batch_size: {result.requested_batch_size}")
    print(f"  actual_batch_size: {result.actual_batch_size}")
    print(f"  image_ids: {result.image_ids}")
    print(f"  labels: {result.labels}")
    print(f"  label_ids: {result.label_ids}")
    print(f"  image_tensor_shape: {result.image_tensor_shape}")
    print(f"  label_tensor_shape: {result.label_tensor_shape}")
    print("  status: batch tensors ready for a PyTorch loss calculation")


# Print the DataLoader batch proof next to the manual batch proof.
def print_torch_dataloader_batch_result(
    result: TorchDataLoaderBatchResult | None,
) -> None:
    # Explain why there may be no DataLoader proof when no train samples are available.
    if result is None:
        print("PyTorch DataLoader proof: skipped because no train samples were loaded")
        return

    # Print enough details to compare this with the manual batch proof above.
    print("PyTorch DataLoader proof")
    print(f"  requested_batch_size: {result.requested_batch_size}")
    print(f"  actual_batch_size: {result.actual_batch_size}")
    print(f"  image_ids: {result.image_ids}")
    print(f"  labels: {result.labels}")
    print(f"  label_ids: {result.label_ids}")
    print(f"  image_tensor_shape: {result.image_tensor_shape}")
    print(f"  label_tensor_shape: {result.label_tensor_shape}")
    print(
        "  status: DataLoader called Dataset.__getitem__ and stacked samples "
        "into one batch"
    )


# Print the PyTorch training-step proof separately from the forward-pass proof.
def print_torch_training_step_result(result: TorchTrainingStepResult | None) -> None:
    # Explain why there may be no training proof when no train samples are available.
    if result is None:
        print("PyTorch training step proof: skipped because no train sample was loaded")
        return

    # Print the real PyTorch training-step result.
    print("PyTorch training step proof")
    print(f"  image_id: {result.image_id}")
    print(f"  true_label: {result.label}")
    print(f"  true_label_id: {result.label_id}")
    print(f"  input_shape: {result.input_shape}")
    print(f"  loss_before: {result.loss_before:.4f}")
    print(f"  loss_after: {result.loss_after:.4f}")
    rounded_before = [
        round(probability, 4)
        for probability in result.probabilities_before
    ]
    rounded_after = [
        round(probability, 4)
        for probability in result.probabilities_after
    ]
    print(f"  probabilities_before: {rounded_before}")
    print(f"  probabilities_after: {rounded_after}")
    print(f"  predicted_label_id_before: {result.predicted_label_id_before}")
    print(f"  predicted_label_id_after: {result.predicted_label_id_after}")
    print(f"  weight_changed: {result.weight_changed}")
    print("  status: one real PyTorch optimizer step completed")


# Print the tiny real PyTorch training-loop summary.
def print_torch_training_loop_summary(
    summary: TorchTrainingLoopSummary | None,
) -> None:
    # Explain why there may be no loop proof when no train samples are available.
    if summary is None:
        print("PyTorch training loop proof: skipped because no train samples were loaded")
        return

    # Print the real PyTorch loop shape and split counts.
    print("PyTorch training loop proof")
    print("  batch source: PyTorch DataLoader (ordered; final partial batch kept)")
    print(f"  epochs: {summary.epochs}")
    print(f"  train samples: {summary.train_sample_count}")
    print(f"  val samples: {summary.val_sample_count}")
    print(f"  test samples: {summary.test_sample_count}")
    print(f"  skipped rows: {summary.skipped_row_count}")
    print(f"  training steps: {summary.training_steps}")
    print(f"  batch size: {summary.batch_size}")
    print(f"  train batches per epoch: {summary.train_batch_count}")
    # This mean counts images, not batches; training losses were measured before
    # each update, so it is not a fresh evaluation of the final model.
    print(f"  average real loss: {summary.average_loss:.4f}")

    # Print one line per epoch so loss movement can be watched over time.
    for epoch_result in summary.epoch_results:
        print(
            f"  epoch {epoch_result.epoch}: "
            f"steps={epoch_result.training_steps}, "
            f"average_loss={epoch_result.average_loss:.4f}"
        )

    # Print before/after values for the first train sample.
    if summary.first_loss is not None and summary.final_loss is not None:
        print(f"  first_sample_loss_before_loop: {summary.first_loss:.4f}")
        print(f"  first_sample_loss_after_loop: {summary.final_loss:.4f}")

    if (
        summary.first_correct_probability is not None
        and summary.final_correct_probability is not None
    ):
        print(
            "  first_sample_correct_probability_before_loop: "
            f"{summary.first_correct_probability:.4f}"
        )
        print(
            "  first_sample_correct_probability_after_loop: "
            f"{summary.final_correct_probability:.4f}"
        )

    print(f"  final_predicted_label_id: {summary.final_predicted_label_id}")
    print(f"  weight_changed: {summary.weight_changed}")
    print("  status: tiny real PyTorch training loop completed")


# Print read-only evaluation metrics for train, validation, and test.
def print_torch_evaluation_results(
    results: dict[str, TorchEvaluationSplitResult] | None,
    majority_baseline: dict[str, TorchMajorityBaselineResult] | None = None,
) -> None:
    # Explain why evaluation may be skipped when no model was trained.
    if results is None:
        print("PyTorch evaluation proof: skipped because no trained model exists")
        return

    # Print the key idea: evaluation observes predictions but does not update weights.
    print("PyTorch evaluation proof")

    # Keep split order stable so repeated terminal runs are easy to compare.
    for split_name in ("train", "val", "test"):
        result = results[split_name]
        accuracy_text = (
            f"{result.accuracy:.4f}"
            if result.accuracy is not None
            else "N/A"
        )
        print(
            f"  {split_name}: "
            f"samples={result.sample_count}, "
            f"correct={result.correct_predictions}, "
            f"accuracy={accuracy_text}, "
            f"average_loss={result.average_loss:.4f}"
        )

        # Print one example prediction when this split has a usable sample.
        if result.first_true_label_id is not None:
            print(
                f"    first_true_label_id={result.first_true_label_id}, "
                f"first_predicted_label_id={result.first_predicted_label_id}, "
                "first_correct_probability="
                f"{result.first_correct_probability:.4f}"
            )

        print("    per-class metrics:")
        for metric in result.class_metrics:
            recall_text = (
                f"{metric.recall:.4f}"
                if metric.recall is not None
                else "N/A"
            )
            f1_text = f"{metric.f1:.4f}" if metric.f1 is not None else "N/A"
            print(
                f"      {metric.label}: support={metric.support}, "
                f"precision={metric.precision:.4f}, "
                f"recall={recall_text}, f1={f1_text}"
            )

        print("    confusion matrix: rows=true, columns=predicted")
        print(f"      class order: {[ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)]}")
        for row in result.confusion_matrix:
            print(f"      {row}")

        print(
            f"    misclassified: {result.misclassified_count}; "
            f"examples retained: {len(result.misclassified_examples)}"
        )
        for example in result.misclassified_examples[:5]:
            print(
                f"      {example.image_id}: "
                f"{example.true_label} -> {example.predicted_label}"
            )

    if majority_baseline is not None:
        print("  majority-class baseline (class chosen from train labels only)")
        for split_name in ("train", "val", "test"):
            baseline = majority_baseline[split_name]
            accuracy_text = (
                f"{baseline.accuracy:.4f}"
                if baseline.accuracy is not None
                else "N/A"
            )
            print(
                f"    {split_name}: always {baseline.majority_label}, "
                f"accuracy={accuracy_text}, "
                f"correct={baseline.correct_predictions}/{baseline.sample_count}"
            )

    print("  status: read-only evaluation completed - no weights updated")


# Print the checkpoint save/load proof.
def print_torch_checkpoint_result(
    result: TorchCheckpointRoundTripResult | None,
) -> None:
    # Explain why checkpointing may be skipped when no model was trained.
    if result is None:
        print("PyTorch checkpoint proof: skipped because no trained model exists")
        return

    # Print the checkpoint path and the comparison between saved and loaded models.
    print("PyTorch checkpoint proof")
    print(f"  checkpoint_path: {result.checkpoint_path}")
    print(f"  preprocessing: {result.preprocessing.to_metadata()}")
    print(f"  image_id: {result.image_id}")
    print(
        "  saved_model_predicted_label_id: "
        f"{result.saved_model_predicted_label_id}"
    )
    print(
        "  loaded_model_predicted_label_id: "
        f"{result.loaded_model_predicted_label_id}"
    )
    print(f"  logits_match: {result.logits_match}")
    print(f"  probabilities_match: {result.probabilities_match}")
    print("  status: checkpoint saved and loaded into a fresh model")


# Run the training-loop skeleton from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    try:
        # Convert optional resize arguments before loading any image data.
        target_size = target_size_from_args(args)

        # Load data through the existing manifest and split pipeline.
        dataset_splits = load_dataset_splits_from_manifest(
            Path(args.manifest_path),
            Path(args.galaxy_data_root),
            skip_missing=not args.strict,
            target_size=target_size,
        )

        # Run the placeholder training-loop shape.
        summary = run_training_loop(dataset_splits, epochs=args.epochs)

        # Run one real PyTorch forward pass on the first available train sample.
        train_samples = dataset_splits.samples_by_split["train"]
        torch_result = (
            run_torch_forward_pass(
                train_samples[0],
                model=create_tiny_galaxy_cnn(seed=args.seed),
            )
            if train_samples
            else None
        )

        # Run one isolated training-step proof on the first available train sample.
        # This is separate from the loop below so the output can teach one update alone.
        torch_training_result = (
            run_torch_training_step(
                train_samples[0],
                model=create_tiny_galaxy_cnn(seed=args.seed),
                learning_rate=args.learning_rate,
            )
            if train_samples
            else None
        )

        # Build one batch preview so the command shows the NCHW batch shape.
        torch_batch_result = run_torch_batch_shape_proof(
            train_samples,
            batch_size=args.batch_size,
        )

        # Build the same kind of batch with PyTorch Dataset/DataLoader.
        # Dataset returns one image/label item; DataLoader stacks items into a batch.
        torch_dataloader_result = preview_first_dataloader_batch(
            train_samples,
            batch_size=args.batch_size,
        )

        # Create one model for the loop.
        # This same object is updated repeatedly, then evaluated and checkpointed.
        loop_model = create_tiny_galaxy_cnn(seed=args.seed)

        # Run repeated training: forward -> loss -> backward -> optimizer.step.
        torch_loop_summary = (
            run_torch_training_loop(
                dataset_splits,
                epochs=args.epochs,
                model=loop_model,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
            )
            if train_samples
            else None
        )

        # Evaluate the trained loop_model on every split without changing weights.
        torch_evaluation_results = (
            evaluate_torch_model(
                dataset_splits,
                loop_model,
                max_error_examples=args.max_error_examples,
            )
            if train_samples
            else None
        )

        # Save and reload the trained loop_model weights to prove checkpoint round-tripping.
        torch_checkpoint_result = (
            run_torch_checkpoint_round_trip(
                train_samples[0],
                loop_model,
                Path(args.checkpoint_path),
                preprocessing=GalaxyPreprocessingPolicy(target_size=target_size),
            )
            if train_samples
            else None
        )
        majority_baseline = (
            evaluate_majority_baseline(
                dataset_splits,
                max_error_examples=args.max_error_examples,
            )
            if train_samples
            else None
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        # Print loading or configuration failures and return a non-zero exit code.
        print(error)
        return 1

    # Print the skeleton training summary.
    print_training_summary(summary)

    # Print the forward-pass proof: current weights produce logits/probabilities.
    print_torch_forward_result(torch_result)

    # Print the isolated one-step proof: one optimizer step can change weights.
    print_torch_training_step_result(torch_training_result)

    # Print the batch proof: multiple samples can become one NCHW tensor.
    print_torch_batch_shape_result(torch_batch_result)

    # Print the DataLoader proof: PyTorch can build the same batch for us.
    print_torch_dataloader_batch_result(torch_dataloader_result)

    # Print the repeated-loop proof: the same model is updated across epochs.
    print_torch_training_loop_summary(torch_loop_summary)

    # Print read-only evaluation metrics after the tiny training loop.
    print_torch_evaluation_results(
        torch_evaluation_results,
        majority_baseline,
    )

    # Write the structured report only when the caller requests it.
    if args.evaluation_report_path:
        report_path = Path(args.evaluation_report_path)
        report = build_evaluation_report(
            Path(args.manifest_path),
            Path(args.galaxy_data_root),
            args,
            dataset_splits,
            torch_loop_summary,
            torch_evaluation_results,
            majority_baseline,
            torch_checkpoint_result,
        )
        write_evaluation_report(report_path, report)
        print(f"evaluation report: {report_path}")

    # Print checkpoint proof: saved loop_model weights load into a fresh model.
    print_torch_checkpoint_result(torch_checkpoint_result)

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
