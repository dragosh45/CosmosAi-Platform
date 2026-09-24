"""PyTorch Dataset/DataLoader helpers for galaxy training samples."""

# Docs: docs/architecture.md Step 17 explains the tiny Dataset/DataLoader proof.
# Docs: docs/concepts_explanations.md explains why PyTorch uses Dataset/DataLoader.
# Docs: docs/excalidraw/galaxy_pytorch_dataloader_flow.excalidraw maps this flow.

# Import dataclass so DataLoader preview output is easy to print and test.
from dataclasses import dataclass

# Import Any because PyTorch batches strings and tensors into a mixed dictionary.
from typing import Any

# Import torch for tensors and the Dataset/DataLoader base classes.
import torch

# Import the existing model conversion helper so Dataset uses the same tensor shape
# as manual batching: image becomes CHW here, then DataLoader adds the N batch axis.
from cosmosai.galaxy.model import galaxy_tensor_to_torch_image

# Import the shared training sample object created by the data pipeline.
from cosmosai.galaxy.dataset_splits import GalaxyDatasetSample


# Store a readable preview of the first DataLoader batch.
@dataclass(frozen=True)
class TorchDataLoaderBatchResult:
    # Batch size requested by the caller.
    requested_batch_size: int

    # Number of samples actually returned in the first batch.
    actual_batch_size: int

    # Image IDs in the first DataLoader batch.
    image_ids: list[str]

    # Human-readable labels in the first DataLoader batch.
    labels: list[str]

    # Numeric class IDs in the first DataLoader batch.
    label_ids: list[int]

    # Image tensor shape from DataLoader: batch, channels, height, width.
    image_tensor_shape: tuple[int, int, int, int]

    # Label tensor shape from DataLoader: one label ID per image.
    label_tensor_shape: tuple[int]


# Wrap GalaxyTrainingSample objects in the standard PyTorch Dataset interface.
class GalaxyTorchDataset(torch.utils.data.Dataset):
    # Keep samples in manifest order so tiny local proofs are deterministic.
    def __init__(self, samples: list[GalaxyDatasetSample]) -> None:
        # Copy the list so outside code cannot mutate dataset order by accident.
        self._samples = list(samples)

    # Tell PyTorch how many examples are available.
    def __len__(self) -> int:
        # DataLoader calls this to know when it has reached the end.
        return len(self._samples)

    # Return one training item by index.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Start from the same GalaxyTrainingSample object used by earlier milestones.
        sample = self._samples[index]

        # Convert from project tensor HWC values to PyTorch CHW image tensor.
        # Example: one RGB 3x3 sample becomes shape (3, 3, 3).
        image_tensor = galaxy_tensor_to_torch_image(sample).squeeze(0)

        # Convert the known label_id into a torch.long scalar used by CrossEntropyLoss.
        # Example: "spiral" stays readable, while label_tensor is tensor(1).
        label_tensor = torch.tensor(sample.label_id, dtype=torch.long)

        # Return tensors plus small metadata so terminal output can stay explainable.
        return {
            "image_id": sample.image_id,
            "label": sample.label,
            "image_tensor": image_tensor,
            "label_tensor": label_tensor,
        }


# Create a DataLoader from eager or manifest-backed lazy project samples.
def create_galaxy_dataloader(
    samples: list[GalaxyDatasetSample],
    batch_size: int = 1,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    # Fail early because PyTorch cannot build meaningful zero-size batches.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    # Wrap our samples in the Dataset interface PyTorch expects.
    dataset = GalaxyTorchDataset(samples)

    # DataLoader calls __getitem__ repeatedly and stacks tensors into batches.
    # shuffle=False keeps this proof stable; real training can turn shuffling on later.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# Build and inspect the first DataLoader batch.
def preview_first_dataloader_batch(
    samples: list[GalaxyDatasetSample],
    batch_size: int = 1,
) -> TorchDataLoaderBatchResult | None:
    # If there are no samples, the caller can print a clear "skipped" message.
    if not samples:
        return None

    # Create the PyTorch DataLoader for this tiny proof.
    dataloader = create_galaxy_dataloader(samples, batch_size=batch_size)

    # Pull one batch from the iterator; this is where DataLoader calls __getitem__.
    first_batch: dict[str, Any] = next(iter(dataloader))

    # PyTorch stacks image tensors into NCHW and label tensors into shape (N,).
    image_tensor = first_batch["image_tensor"]
    label_tensor = first_batch["label_tensor"]

    # Return plain Python values so CLI output and tests are easy to read.
    return TorchDataLoaderBatchResult(
        requested_batch_size=batch_size,
        actual_batch_size=int(image_tensor.shape[0]),
        image_ids=list(first_batch["image_id"]),
        labels=list(first_batch["label"]),
        label_ids=label_tensor.tolist(),
        image_tensor_shape=tuple(image_tensor.shape),
        label_tensor_shape=tuple(label_tensor.shape),
    )
