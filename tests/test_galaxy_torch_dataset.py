# Import Path to point tests at the tiny sample manifest and image root.
from pathlib import Path

# Import torch so tests can check PyTorch tensor dtypes.
import torch

# Import the existing split pipeline so Dataset tests start from real project samples.
from cosmosai.galaxy.dataset_splits import create_dataset_splits
from cosmosai.galaxy.manifest import load_manifest

# Import the new Dataset/DataLoader helpers under test.
from cosmosai.galaxy.torch_dataset import (
    GalaxyTorchDataset,
    create_galaxy_dataloader,
    preview_first_dataloader_batch,
)


# Load the current tiny train samples through the same data path used by the CLI.
def _load_train_samples():
    # Read validated manifest rows from the sample CSV.
    records = load_manifest(Path("data/samples/galaxy_manifest_sample.csv"))

    # Create train/val/test buckets with model-ready samples.
    dataset_splits = create_dataset_splits(records, Path("data/samples/images"))

    # Return only the train split because DataLoader training batches start there.
    return dataset_splits.samples_by_split["train"]


# Test that one Dataset item returns one image tensor and one label tensor.
def test_galaxy_torch_dataset_returns_one_item_as_tensors():
    # Load the two tiny training samples from the current sample data.
    train_samples = _load_train_samples()

    # Wrap project samples in the PyTorch Dataset interface.
    dataset = GalaxyTorchDataset(train_samples)

    # Ask the Dataset for the first item, like DataLoader will do internally.
    item = dataset[0]

    # The Dataset keeps the sample count visible to PyTorch.
    assert len(dataset) == 2

    # Metadata stays readable for debugging and learning output.
    assert item["image_id"] == "gz2-000001"
    assert item["label"] == "spiral"

    # One item is CHW: channels, height, width. DataLoader adds batch later.
    assert tuple(item["image_tensor"].shape) == (3, 3, 3)
    assert item["image_tensor"].dtype == torch.float32

    # The label is a torch.long scalar because CrossEntropyLoss expects class IDs.
    assert item["label_tensor"].item() == 1
    assert item["label_tensor"].dtype == torch.long


def test_lazy_split_sample_loads_pixels_only_when_dataset_requests_item(monkeypatch):
    # Patch the model-ready preprocessing function so the ownership boundary is visible.
    import cosmosai.galaxy.training_sample as training_sample_module

    calls: list[str] = []
    original_loader = training_sample_module.load_and_preprocess_image_for_record

    def counted_loader(*args, **kwargs):
        calls.append(args[0].image_id)
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        training_sample_module,
        "load_and_preprocess_image_for_record",
        counted_loader,
    )

    records = load_manifest(Path("data/samples/galaxy_manifest_sample.csv"))
    dataset_splits = create_dataset_splits(records, Path("data/samples/images"))

    # Split creation keeps manifest-backed objects and does not preprocess tensors.
    assert calls == []

    # Dataset access is the moment that decodes, normalizes, and returns one tensor.
    item = GalaxyTorchDataset(dataset_splits.samples_by_split["train"])[0]
    assert tuple(item["image_tensor"].shape) == (3, 3, 3)
    assert calls == ["gz2-000001"]


# Test that DataLoader stacks Dataset items into one training batch.
def test_galaxy_dataloader_stacks_items_into_batch():
    # Load the tiny train split.
    train_samples = _load_train_samples()

    # Build a deterministic DataLoader batch with both train samples.
    dataloader = create_galaxy_dataloader(train_samples, batch_size=2)
    batch = next(iter(dataloader))

    # DataLoader stacks two CHW images into one NCHW batch tensor.
    assert tuple(batch["image_tensor"].shape) == (2, 3, 3, 3)

    # DataLoader also stacks the scalar labels into one label vector.
    assert tuple(batch["label_tensor"].shape) == (2,)
    assert batch["label_tensor"].tolist() == [1, 2]

    # String metadata is collated into lists in the same sample order.
    assert batch["image_id"] == ["gz2-000001", "gz2-000004"]
    assert batch["label"] == ["spiral", "lenticular"]


# Test that the preview helper returns simple values for terminal output.
def test_preview_first_dataloader_batch_reports_batch_details():
    # Load the tiny train split.
    train_samples = _load_train_samples()

    # Build and summarize the first DataLoader batch.
    result = preview_first_dataloader_batch(train_samples, batch_size=2)

    # Verify the preview mirrors the actual DataLoader output.
    assert result is not None
    assert result.requested_batch_size == 2
    assert result.actual_batch_size == 2
    assert result.image_ids == ["gz2-000001", "gz2-000004"]
    assert result.labels == ["spiral", "lenticular"]
    assert result.label_ids == [1, 2]
    assert result.image_tensor_shape == (2, 3, 3, 3)
    assert result.label_tensor_shape == (2,)


# Test that invalid batch sizes fail before PyTorch builds a DataLoader.
def test_galaxy_dataloader_rejects_invalid_batch_size():
    # Batch size zero would hide a caller bug, so the helper should reject it.
    try:
        create_galaxy_dataloader(_load_train_samples(), batch_size=0)
    except ValueError as error:
        assert "batch_size must be at least 1" in str(error)
    else:
        raise AssertionError("Expected invalid DataLoader batch size to fail")


# Test that an empty sample list can be previewed safely.
def test_preview_first_dataloader_batch_skips_empty_samples():
    # Empty sample lists are allowed for preview output because real data may be missing.
    assert preview_first_dataloader_batch([], batch_size=2) is None
