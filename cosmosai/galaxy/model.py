"""Tiny PyTorch galaxy CNN helpers shared by training and inference."""

# Import dataclass so model outputs can be passed around clearly.
from dataclasses import dataclass

# Import Path to read and write checkpoint files.
from pathlib import Path

# Import torch for the tiny CNN, tensors, losses, and checkpoint IO.
import torch

# Import the shared label mapping and sample object used by the model.
from cosmosai.galaxy.labels import LABEL_TO_ID
from cosmosai.galaxy.training_sample import GalaxyTrainingSample


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


# Define the first tiny real CNN model used for local proofs.
class TinyGalaxyCNN(torch.nn.Module):
    # Create a minimal CNN that accepts RGB image tensors.
    def __init__(self, label_count: int = len(LABEL_TO_ID)) -> None:
        # Initialize the parent torch.nn.Module class.
        super().__init__()

        # Conv2d scans 3x3 windows of RGB pixels and creates four feature maps.
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            # ReLU keeps positive pattern signals and turns negative signals into 0.
            torch.nn.ReLU(),
            # Pooling compresses each feature map to one value.
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten removes unused grid dimensions before the classifier layer.
            torch.nn.Flatten(),
        )

        # Convert four pooled feature values into one logit per galaxy label.
        self.classifier = torch.nn.Linear(4, label_count)

    # Run a forward pass from image tensor to raw class scores.
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Convert the image tensor into compact feature values.
        features = self.features(image_tensor)

        # Return raw class scores; softmax/loss stay outside the model.
        return self.classifier(features)


# Create the tiny CNN with deterministic initial weights for stable local tests.
def create_tiny_galaxy_cnn(label_count: int = len(LABEL_TO_ID)) -> TinyGalaxyCNN:
    # Seed PyTorch so proof output is repeatable across runs on this machine.
    torch.manual_seed(7)

    # Return the tiny CNN model.
    return TinyGalaxyCNN(label_count=label_count)


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
    image_tensor = torch.tensor(sample.tensor.values, dtype=torch.float32)

    # Reshape from flat HWC values into height, width, channels.
    image_tensor = image_tensor.reshape(height, width, channels)

    # Rearrange HWC into CHW because PyTorch Conv2d expects channels first.
    image_tensor = image_tensor.permute(2, 0, 1)

    # Add a batch dimension so the final shape is NCHW.
    return image_tensor.unsqueeze(0)


# Run one real PyTorch forward pass without training or updating weights.
def run_torch_forward_pass(
    sample: GalaxyTrainingSample,
    model: TinyGalaxyCNN | None = None,
) -> TorchForwardPassResult:
    # Use the provided PyTorch model or create the deterministic tiny model.
    active_model = model if model is not None else create_tiny_galaxy_cnn()

    # Put the model in evaluation mode because this helper is forward-only.
    active_model.eval()

    # Convert the project sample tensor into PyTorch's expected NCHW image shape.
    torch_image = galaxy_tensor_to_torch_image(sample)

    # Disable gradient tracking because this is inspection/inference only.
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


# Save the current TinyGalaxyCNN weights to disk.
def save_torch_checkpoint(
    model: TinyGalaxyCNN,
    checkpoint_path: Path,
) -> Path:
    # Create the parent folder, for example models/, if it does not exist yet.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # state_dict stores learned tensors; architecture stays in code.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_to_id": LABEL_TO_ID,
        "label_count": len(LABEL_TO_ID),
    }

    # Write the checkpoint file. This is what serving code loads later.
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

