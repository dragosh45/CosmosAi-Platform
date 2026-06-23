# Import Path to pass local sample paths into the dataset split loader.
from pathlib import Path


# Test that available tiny images become samples and missing images are skipped.
def test_create_dataset_splits_skips_missing_sample_images(load_service_module):
    # Load the manifest helper so the test uses the real sample manifest.
    manifest_loader = load_service_module(
        "galaxy_manifest_loader_for_dataset_splits",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the dataset split helper under test.
    split_loader = load_service_module(
        "galaxy_dataset_split_loader",
        "scripts/load_galaxy_dataset_splits.py",
    )

    # Load the sample manifest records.
    records = manifest_loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # Create train, val, and test sample buckets from the manifest.
    dataset_splits = split_loader.create_dataset_splits(
        records,
        Path("data/samples/images"),
    )

    # Only the first training row has a real tiny local image today.
    assert len(dataset_splits.samples_by_split["train"]) == 1
    assert len(dataset_splits.samples_by_split["val"]) == 0
    assert len(dataset_splits.samples_by_split["test"]) == 0

    # The other sample manifest rows are intentionally skipped until images exist.
    assert len(dataset_splits.skipped_by_split["train"]) == 1
    assert len(dataset_splits.skipped_by_split["val"]) == 1
    assert len(dataset_splits.skipped_by_split["test"]) == 1

    # Verify the loaded sample keeps the label and tensor data together.
    train_sample = dataset_splits.samples_by_split["train"][0]
    assert train_sample.image_id == "gz2-000001"
    assert train_sample.label == "spiral"
    assert train_sample.label_id == 1
    assert train_sample.tensor.shape == (3, 3, 3)


# Test that strict mode fails instead of skipping missing images.
def test_create_dataset_splits_strict_mode_fails_on_missing_image(load_service_module):
    # Load the manifest helper so the test uses the real sample manifest.
    manifest_loader = load_service_module(
        "galaxy_manifest_loader_for_dataset_splits_strict",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the dataset split helper under test.
    split_loader = load_service_module(
        "galaxy_dataset_split_loader_strict",
        "scripts/load_galaxy_dataset_splits.py",
    )

    # Load the sample manifest records.
    records = manifest_loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # Strict mode should expose missing sample image files immediately.
    try:
        split_loader.create_dataset_splits(
            records,
            Path("data/samples/images"),
            skip_missing=False,
        )
    except FileNotFoundError as error:
        assert "Image file does not exist" in str(error)
    else:
        raise AssertionError("Expected strict split loading to fail on missing image")
