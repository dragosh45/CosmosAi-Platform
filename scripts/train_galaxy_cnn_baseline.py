#!/usr/bin/env python3

# Docs: docs/architecture.md Step 3 explains the placeholder skeleton path.
# Docs: docs/architecture.md Step 4 explains the PyTorch forward-pass proof.
# Docs: docs/architecture.md Step 5 explains the first real PyTorch weight update.
# Docs: docs/architecture.md Step 6 explains the tiny real PyTorch training loop.
# Docs: docs/architecture.md Step 7 explains the read-only evaluation proof.
# Docs: docs/architecture.md Step 8 explains the tiny checkpoint save/load proof.

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

# Import torch so Milestone 33 can run one real CNN forward pass.
import torch

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


# Store the output of one real PyTorch forward pass.
@dataclass(frozen=True)
class TorchForwardPassResult:
    # Image ID used for tracing the forward pass back to the manifest row.
    image_id: str

    # Tensor shape sent into the PyTorch model: batch, channels, height, width.
    input_shape: tuple[int, int, int, int]

    # Raw class scores produced by the tiny PyTorch CNN.
    logits: list[float]

    # Probability-like values produced from the logits for inspection.
    probabilities: list[float]

    # Highest-probability class ID from the PyTorch forward pass.
    predicted_label_id: int


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

    # Number of manifest rows skipped because sample files are missing or unreadable.
    skipped_row_count: int

    # Number of real PyTorch optimizer steps executed.
    training_steps: int

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


# Store the result of saving and loading one tiny PyTorch checkpoint.
@dataclass(frozen=True)
class TorchCheckpointRoundTripResult:
    # Filesystem path where the model weights were saved.
    checkpoint_path: Path

    # Image ID used to compare the trained model and loaded model.
    image_id: str

    # Predicted class ID before saving the checkpoint.
    saved_model_predicted_label_id: int

    # Predicted class ID after loading the checkpoint into a fresh model.
    loaded_model_predicted_label_id: int

    # True when loaded logits match the trained model logits.
    logits_match: bool

    # True when loaded probabilities match the trained model probabilities.
    probabilities_match: bool


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


# Define the first tiny real CNN model used only for a forward-pass proof.
class TinyGalaxyCNN(torch.nn.Module):
    # Create a minimal CNN that accepts RGB image tensors.
    def __init__(self, label_count: int = len(LABEL_TO_ID)) -> None:
        # Initialize the parent torch.nn.Module class.
        super().__init__()

        # Conv2d scans small 3x3 windows of RGB pixels and creates four feature maps.
        # Example shape change here: (1, 3, 3, 3) -> (1, 4, 3, 3).
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            # ReLU keeps positive pattern signals and turns negative signals into 0.
            torch.nn.ReLU(),
            # Pooling compresses each feature map to one value: (1, 4, 3, 3) -> (1, 4, 1, 1).
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten removes unused grid dimensions: (1, 4, 1, 1) -> (1, 4).
            torch.nn.Flatten(),
        )

        # Convert the four pooled feature values into one logit per galaxy label.
        # Example output shape: (1, 4), one raw score for each known label.
        self.classifier = torch.nn.Linear(4, label_count)

    # Run a forward pass from image tensor to raw class scores.
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Convert the image tensor into compact feature values.
        features = self.features(image_tensor)

        # Return raw class scores; softmax/loss are intentionally outside the model.
        return self.classifier(features)


# Create the tiny CNN with deterministic initial weights for stable local tests.
def create_tiny_galaxy_cnn(label_count: int = len(LABEL_TO_ID)) -> TinyGalaxyCNN:
    # Seed PyTorch so the proof output is repeatable across runs on this machine.
    torch.manual_seed(7)

    # Return the tiny CNN model.
    return TinyGalaxyCNN(label_count=label_count)


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


# Convert the current GalaxyImageTensor object into PyTorch's NCHW tensor shape.
def galaxy_tensor_to_torch_image(sample: GalaxyTrainingSample) -> torch.Tensor:
    # Read the current height, width, channels shape from the sample tensor.
    height, width, channels = sample.tensor.shape

    # The first tiny CNN expects RGB input.
    if channels != 3:
        raise ValueError(f"Expected 3 RGB channels, got {channels}")

    # Confirm the flat value list matches height * width * channels.
    expected_value_count = height * width * channels
    if len(sample.tensor.values) != expected_value_count:
        raise ValueError(
            f"Expected {expected_value_count} tensor values, "
            f"got {len(sample.tensor.values)}"
        )

    # Build a PyTorch tensor from the normalized flat Python list.
    # Example data changes from [0.0, 0.0, 0.0, 0.251, ...] into tensor([...]).
    image_tensor = torch.tensor(sample.tensor.values, dtype=torch.float32)

    # Reshape from flat HWC values into height, width, channels.
    # Example shape: 27 values -> (3, 3, 3).
    image_tensor = image_tensor.reshape(height, width, channels)

    # Rearrange HWC into CHW because PyTorch Conv2d expects channels first.
    # Example shape: (3, 3, 3) -> (3, 3, 3), same numbers but channel dimension first.
    image_tensor = image_tensor.permute(2, 0, 1)

    # Add a batch dimension so the final shape is NCHW: batch, channels, height, width.
    # Example shape: (3, 3, 3) -> (1, 3, 3, 3), meaning one RGB image.
    return image_tensor.unsqueeze(0)


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


# Compute simple cross-entropy loss for one correct class.
def cross_entropy_loss(probabilities: list[float], label_id: int) -> float:
    # Guard against impossible class IDs before indexing the probability list.
    if label_id < 0 or label_id >= len(probabilities):
        raise ValueError(f"Label ID is outside probability range: {label_id}")

    # Clamp the probability to avoid log(0) in future edge cases.
    correct_probability = max(probabilities[label_id], 1e-12)

    # Cross entropy gets smaller when the correct class probability gets larger.
    return -math.log(correct_probability)


# Run one real PyTorch forward pass without training or updating weights.
def run_torch_forward_pass(
    sample: GalaxyTrainingSample,
    model: TinyGalaxyCNN | None = None,
) -> TorchForwardPassResult:
    # Use the provided PyTorch model or create the deterministic tiny model.
    active_model = model if model is not None else create_tiny_galaxy_cnn()

    # Put the model in evaluation mode because this milestone is forward-only.
    active_model.eval()

    # Convert the project sample tensor into PyTorch's expected NCHW image shape.
    torch_image = galaxy_tensor_to_torch_image(sample)

    # Disable gradient tracking because this is inspection only.
    # Gradient tracking means PyTorch remembers operations so it can later calculate
    # how each weight affected the loss. That memory is only needed during training.
    with torch.no_grad():
        # Run the real PyTorch CNN forward pass to produce logits.
        logits_tensor = active_model(torch_image)

        # Convert logits into probabilities for human inspection.
        probabilities_tensor = torch.softmax(logits_tensor, dim=1)

    # Convert PyTorch tensors back into plain Python lists for summaries and tests.
    logits = logits_tensor.squeeze(0).tolist()
    probabilities = probabilities_tensor.squeeze(0).tolist()

    # Pick the highest-probability class ID.
    predicted_label_id = int(torch.argmax(probabilities_tensor, dim=1).item())

    # Return a compact result object.
    return TorchForwardPassResult(
        image_id=sample.image_id,
        input_shape=tuple(torch_image.shape),
        logits=logits,
        probabilities=probabilities,
        predicted_label_id=predicted_label_id,
    )


# Run one real PyTorch training step and prove at least one weight changed.
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

    # Clear old gradients. Gradients are "which direction should each weight move?"
    optimizer.zero_grad()

    # Forward pass: image numbers move through CNN layers and become raw class scores.
    logits_before = active_model(torch_image)

    # Loss compares the raw scores with the known label_id from the manifest.
    loss_before_tensor = loss_function(logits_before, target_label)

    # Backward pass: calculate gradients for each trainable weight.
    loss_before_tensor.backward()

    # Optimizer step: update weights using the gradients and learning rate.
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


# Run a tiny real PyTorch training loop over the available train split.
def run_torch_training_loop(
    dataset_splits: GalaxyDatasetSplits,
    epochs: int = 1,
    model: TinyGalaxyCNN | None = None,
    learning_rate: float = 0.1,
) -> TorchTrainingLoopSummary:
    # Fail clearly because zero or negative epochs do not make sense.
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    # Read split buckets. Only train samples are used for weight updates.
    train_samples = dataset_splits.samples_by_split["train"]
    val_samples = dataset_splits.samples_by_split["val"]
    test_samples = dataset_splits.samples_by_split["test"]

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
    # This matters because each optimizer.step() continues from the latest weights.
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

    # Store every step loss so we can report average real training loss.
    step_losses: list[float] = []

    # Store one compact summary per epoch for learning/inspection.
    epoch_results: list[TorchTrainingEpochResult] = []

    # Loop over epochs: one epoch means one pass over the available train samples.
    for epoch_index in range(epochs):
        # Store losses for this epoch only.
        epoch_losses: list[float] = []

        # Loop over train samples only; validation and test stay read-only.
        for sample in train_samples:
            # Convert project sample data into PyTorch image and target tensors.
            torch_image = galaxy_tensor_to_torch_image(sample)
            target_label = torch.tensor([sample.label_id], dtype=torch.long)

            # Clear old gradients before this sample's calculation.
            optimizer.zero_grad()

            # Forward pass: current weights produce logits for this image.
            logits = active_model(torch_image)

            # Loss: compare logits with the known correct label_id.
            loss_tensor = loss_function(logits, target_label)

            # Backward pass: calculate gradients for the current weights.
            loss_tensor.backward()

            # Optimizer step: update weights using those gradients.
            optimizer.step()

            # Save the loss as a plain number for summaries.
            loss_value = float(loss_tensor.item())
            step_losses.append(loss_value)
            epoch_losses.append(loss_value)

        # Average this epoch's losses, using 0.0 if there are no train samples.
        epoch_average_loss = (
            sum(epoch_losses) / len(epoch_losses)
            if epoch_losses
            else 0.0
        )

        # Store the epoch summary with a human-facing epoch number.
        epoch_results.append(
            TorchTrainingEpochResult(
                epoch=epoch_index + 1,
                training_steps=len(epoch_losses),
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

    # Average all real PyTorch training-step losses.
    average_loss = (
        sum(step_losses) / len(step_losses)
        if step_losses
        else 0.0
    )

    # Return the loop summary for command output and tests.
    return TorchTrainingLoopSummary(
        epochs=epochs,
        train_sample_count=len(train_samples),
        val_sample_count=len(val_samples),
        test_sample_count=len(test_samples),
        skipped_row_count=skipped_row_count,
        training_steps=len(step_losses),
        average_loss=average_loss,
        first_loss=first_loss,
        final_loss=final_loss,
        first_correct_probability=first_correct_probability,
        final_correct_probability=final_correct_probability,
        final_predicted_label_id=final_predicted_label_id,
        weight_changed=weight_changed,
        epoch_results=epoch_results,
    )


# Evaluate one split with the current model weights, without changing weights.
def evaluate_torch_split(
    split_name: str,
    samples: list[GalaxyTrainingSample],
    model: TinyGalaxyCNN,
) -> TorchEvaluationSplitResult:
    # Evaluation mode means "use the model for inspection/prediction only."
    model.eval()

    # CrossEntropyLoss gives a real loss number for each read-only prediction.
    loss_function = torch.nn.CrossEntropyLoss()

    # Store metric pieces so the caller can see accuracy and average loss.
    losses: list[float] = []
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
    )


# Evaluate train, validation, and test splits with the same trained model.
def evaluate_torch_model(
    dataset_splits: GalaxyDatasetSplits,
    model: TinyGalaxyCNN,
) -> dict[str, TorchEvaluationSplitResult]:
    # Evaluate every split read-only; only training code is allowed to update weights.
    return {
        split_name: evaluate_torch_split(
            split_name,
            dataset_splits.samples_by_split[split_name],
            model,
        )
        for split_name in ("train", "val", "test")
    }


# Save the current TinyGalaxyCNN weights to disk.
def save_torch_checkpoint(
    model: TinyGalaxyCNN,
    checkpoint_path: Path,
) -> Path:
    # Create the parent folder, for example models/, if it does not exist yet.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # state_dict is PyTorch's dictionary of learned tensors/weights.
    # The architecture stays in code; the checkpoint stores the learned numbers.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_to_id": LABEL_TO_ID,
        "label_count": len(LABEL_TO_ID),
    }

    # Write the checkpoint file. This is what serving code will load later.
    torch.save(checkpoint, checkpoint_path)

    # Return the path so callers can print or test it.
    return checkpoint_path


# Load a TinyGalaxyCNN checkpoint into a fresh model object.
def load_torch_checkpoint(checkpoint_path: Path) -> TinyGalaxyCNN:
    # Load the checkpoint from CPU so this works on machines without a GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Fail clearly if the checkpoint was created for a different label mapping.
    if checkpoint.get("label_to_id") != LABEL_TO_ID:
        raise ValueError("Checkpoint label mapping does not match current code")

    # Create a fresh model architecture, then fill it with saved learned weights.
    model = create_tiny_galaxy_cnn(label_count=checkpoint["label_count"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation mode is the normal mode for a loaded model used for prediction.
    model.eval()

    # Return the loaded model so caller code can run predictions.
    return model


# Prove a saved checkpoint can load into a fresh model with the same prediction.
def run_torch_checkpoint_round_trip(
    sample: GalaxyTrainingSample,
    trained_model: TinyGalaxyCNN,
    checkpoint_path: Path,
) -> TorchCheckpointRoundTripResult:
    # Save the trained model's current learned weights to disk.
    saved_path = save_torch_checkpoint(trained_model, checkpoint_path)

    # Load those weights into a fresh TinyGalaxyCNN instance.
    loaded_model = load_torch_checkpoint(saved_path)

    # Compare both models on the same sample without tracking gradients.
    trained_result = run_torch_forward_pass(sample, model=trained_model)
    loaded_result = run_torch_forward_pass(sample, model=loaded_model)

    # allclose allows tiny float differences while still proving practical equality.
    trained_logits = torch.tensor(trained_result.logits)
    loaded_logits = torch.tensor(loaded_result.logits)
    trained_probabilities = torch.tensor(trained_result.probabilities)
    loaded_probabilities = torch.tensor(loaded_result.probabilities)

    # Return a compact proof for tests and terminal output.
    return TorchCheckpointRoundTripResult(
        checkpoint_path=saved_path,
        image_id=sample.image_id,
        saved_model_predicted_label_id=trained_result.predicted_label_id,
        loaded_model_predicted_label_id=loaded_result.predicted_label_id,
        logits_match=bool(torch.allclose(trained_logits, loaded_logits)),
        probabilities_match=bool(
            torch.allclose(trained_probabilities, loaded_probabilities)
        ),
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
    print(f"  epochs: {summary.epochs}")
    print(f"  train samples: {summary.train_sample_count}")
    print(f"  val samples: {summary.val_sample_count}")
    print(f"  test samples: {summary.test_sample_count}")
    print(f"  skipped rows: {summary.skipped_row_count}")
    print(f"  training steps: {summary.training_steps}")
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
        # Load data through the existing manifest and split pipeline.
        dataset_splits = load_dataset_splits_from_manifest(
            Path(args.manifest_path),
            Path(args.galaxy_data_root),
            skip_missing=not args.strict,
        )

        # Run the placeholder training-loop shape.
        summary = run_training_loop(dataset_splits, epochs=args.epochs)

        # Run one real PyTorch forward pass on the first available train sample.
        train_samples = dataset_splits.samples_by_split["train"]
        torch_result = (
            run_torch_forward_pass(train_samples[0])
            if train_samples
            else None
        )

        # Run one real PyTorch training step on the first available train sample.
        torch_training_result = (
            run_torch_training_step(train_samples[0])
            if train_samples
            else None
        )

        # Create one model for the loop so we can evaluate the same trained weights.
        loop_model = create_tiny_galaxy_cnn()

        # Run a tiny real PyTorch loop over the available train samples.
        torch_loop_summary = (
            run_torch_training_loop(
                dataset_splits,
                epochs=args.epochs,
                model=loop_model,
            )
            if train_samples
            else None
        )

        # Evaluate the trained model on every split without changing weights.
        torch_evaluation_results = (
            evaluate_torch_model(dataset_splits, loop_model)
            if train_samples
            else None
        )

        # Save and reload the trained weights to prove checkpoint round-tripping.
        torch_checkpoint_result = (
            run_torch_checkpoint_round_trip(
                train_samples[0],
                loop_model,
                Path(args.checkpoint_path),
            )
            if train_samples
            else None
        )
    except (FileNotFoundError, ValueError) as error:
        # Print loading or configuration failures and return a non-zero exit code.
        print(error)
        return 1

    # Print the skeleton training summary.
    print_training_summary(summary)

    # Print the real PyTorch forward-pass proof.
    print_torch_forward_result(torch_result)

    # Print the real PyTorch training-step proof.
    print_torch_training_step_result(torch_training_result)

    # Print the tiny real PyTorch training-loop proof.
    print_torch_training_loop_summary(torch_loop_summary)

    # Print read-only evaluation metrics after the tiny training loop.
    print_torch_evaluation_results(torch_evaluation_results)

    # Print the tiny checkpoint save/load proof.
    print_torch_checkpoint_result(torch_checkpoint_result)

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
