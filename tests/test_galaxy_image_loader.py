# Import Path to compare loaded image paths.
from pathlib import Path


# Test that the tiny sample image can be loaded through a manifest record.
def test_load_sample_image_from_manifest_record(load_service_module):
    # Load the manifest helper so the test follows the manifest-to-image flow.
    manifest_loader = load_service_module(
        "galaxy_manifest_loader_for_image",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the image helper under test.
    image_loader = load_service_module(
        "galaxy_image_loader",
        "scripts/load_galaxy_image.py",
    )

    # Load the sample manifest and select the row that points to the sample image.
    records = manifest_loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )
    record = records[0]

    # Load the tiny image through the manifest record.
    image = image_loader.load_image_for_record(
        record,
        Path("data/samples/images"),
    )

    # Verify the tiny image metadata.
    assert image.path == Path(
        "data/samples/images/processed/images_224/gz2-000001.ppm"
    )
    assert image.width == 3
    assert image.height == 3
    assert image.channels == 3
    assert len(image.pixels) == 27


# Test that missing image files fail clearly.
def test_load_missing_image_fails_clearly(load_service_module):
    # Load the image helper under test.
    image_loader = load_service_module(
        "galaxy_image_loader_missing",
        "scripts/load_galaxy_image.py",
    )

    # Build a record pointing to an image that does not exist.
    record = image_loader.GalaxyManifestRecord(
        image_id="missing-001",
        image_path=Path("processed/images_224/missing.ppm"),
        label="spiral",
        split="train",
        source="test",
    )

    # Loading should fail before parsing because the file is missing.
    try:
        image_loader.load_image_for_record(record, Path("data/samples/images"))
    except FileNotFoundError as error:
        assert "Image file does not exist" in str(error)
    else:
        raise AssertionError("Expected missing image to raise FileNotFoundError")
