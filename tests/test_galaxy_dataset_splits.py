# Import Path to pass local sample paths into the dataset split loader.
from pathlib import Path

# Import Pillow in tests so we can create a normal PNG fixture at runtime.
from PIL import Image

# Import shared split and manifest objects for package-level resize tests.
from cosmosai.galaxy.dataset_splits import (
    GalaxyDatasetQualityError,
    create_dataset_splits,
)
from cosmosai.galaxy.manifest import GalaxyManifestRecord


# Create a tiny PNG file for split-pipeline tests.
def _write_tiny_png(image_path: Path) -> None:
    # Create a tiny RGB image with two concrete pixels.
    image = Image.new("RGB", (2, 1))

    # Use visible channel values so the output is real image data, not empty.
    image.putdata([(10, 20, 30), (200, 210, 220)])

    # Save as PNG so the split pipeline must use Pillow.
    image.save(image_path)


# Test that the sample manifest now loads usable train, val, and test samples.
def test_create_dataset_splits_loads_each_sample_split(load_service_module):
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

    # Milestone 44 gives each split at least one real tiny local image.
    assert len(dataset_splits.samples_by_split["train"]) == 2
    assert len(dataset_splits.samples_by_split["val"]) == 1
    assert len(dataset_splits.samples_by_split["test"]) == 1

    # No sample rows should be skipped now that every sample image exists.
    assert dataset_splits.skipped_by_split["train"] == []
    assert dataset_splits.skipped_by_split["val"] == []
    assert dataset_splits.skipped_by_split["test"] == []
    assert dataset_splits.quality_totals() == {
        "accepted": 4,
        "skipped_missing": 0,
        "rejected": 0,
    }

    # Verify the loaded sample keeps the label and tensor data together.
    train_sample = dataset_splits.samples_by_split["train"][0]
    assert train_sample.image_id == "gz2-000001"
    assert train_sample.label == "spiral"
    assert train_sample.label_id == 1
    assert train_sample.tensor.shape == (3, 3, 3)


# Test that strict mode still fails instead of skipping a truly missing image.
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

    # Add one fake row so strict mode still proves missing images fail clearly.
    missing_record = manifest_loader.GalaxyManifestRecord(
        image_id="missing-999",
        image_path=Path("processed/images_224/missing-999.ppm"),
        label="spiral",
        split="train",
        source="test",
    )

    # Strict mode rejects missing files and keeps inspectable quality counts.
    try:
        split_loader.create_dataset_splits(
            records + [missing_record],
            Path("data/samples/images"),
            skip_missing=False,
        )
    except GalaxyDatasetQualityError as error:
        assert "Image file does not exist" in str(error)
        assert error.dataset_splits.quality_totals() == {
            "accepted": 4,
            "skipped_missing": 0,
            "rejected": 1,
        }
    else:
        raise AssertionError("Expected strict split loading to fail on missing image")


# Test that dataset splits can create resized samples from normal PNG images.
def test_create_dataset_splits_resizes_png_samples(tmp_path):
    # Create a fake image root and one PNG file under a manifest-style path.
    image_root = tmp_path / "images"
    image_folder = image_root / "processed" / "images_224"
    image_folder.mkdir(parents=True)
    image_path = image_folder / "tiny-galaxy.png"
    _write_tiny_png(image_path)

    # Build one train record pointing to the PNG image.
    records = [
        GalaxyManifestRecord(
            image_id="png-001",
            image_path=Path("processed/images_224/tiny-galaxy.png"),
            label="spiral",
            split="train",
            source="test",
        )
    ]

    # Create split samples with a fixed width=3, height=3 target size.
    dataset_splits = create_dataset_splits(
        records,
        image_root,
        target_size=(3, 3),
    )

    # Verify the train sample now has resized, normalized model-ready data.
    train_sample = dataset_splits.samples_by_split["train"][0]
    assert train_sample.image_id == "png-001"
    assert train_sample.tensor.shape == (3, 3, 3)
    assert len(train_sample.tensor.values) == 27
    assert dataset_splits.skipped_by_split["train"] == []


# Test that permissive mode explicitly counts a missing file as skipped.
def test_create_dataset_splits_counts_intentionally_skipped_missing_file(tmp_path):
    record = GalaxyManifestRecord(
        image_id="missing-001",
        image_path=Path("missing-001.ppm"),
        label="spiral",
        split="train",
        source="test",
        manifest_row=8,
    )

    dataset_splits = create_dataset_splits([record], tmp_path, skip_missing=True)

    assert dataset_splits.quality_totals() == {
        "accepted": 0,
        "skipped_missing": 1,
        "rejected": 0,
    }
    assert "row 8, image_id 'missing-001'" in dataset_splits.skipped_by_split["train"][0]


# Test that a corrupt existing file is rejected even when missing files may be skipped.
def test_create_dataset_splits_rejects_corrupt_image_in_permissive_mode(tmp_path):
    image_path = tmp_path / "corrupt.ppm"
    image_path.write_text("P3\n3 x\n255\n0 0 0\n", encoding="utf-8")
    record = GalaxyManifestRecord(
        image_id="corrupt-001",
        image_path=Path("corrupt.ppm"),
        label="spiral",
        split="val",
        source="test",
        manifest_row=12,
    )

    try:
        create_dataset_splits([record], tmp_path, skip_missing=True)
    except GalaxyDatasetQualityError as error:
        assert error.dataset_splits.quality_totals() == {
            "accepted": 0,
            "skipped_missing": 0,
            "rejected": 1,
        }
        assert "row 12, image_id 'corrupt-001'" in str(error)
        assert "width, height, and max value must be integers" in str(error)
    else:
        raise AssertionError("Expected corrupt image data to be rejected")
