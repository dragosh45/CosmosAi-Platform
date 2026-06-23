# Import Path to pass the sample image root into preprocessing helpers.
from pathlib import Path


# Test that the tiny sample image can be normalized into tensor-like values.
def test_preprocess_sample_image_from_manifest_record(load_service_module):
    # Load the manifest helper so the test follows the real manifest path.
    manifest_loader = load_service_module(
        "galaxy_manifest_loader_for_preprocessing",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the preprocessing helper under test.
    preprocessor = load_service_module(
        "galaxy_image_preprocessor",
        "scripts/preprocess_galaxy_image.py",
    )

    # Load the sample manifest and select the row that points to the sample image.
    records = manifest_loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )
    record = records[0]

    # Load and preprocess the tiny image through the manifest record.
    tensor = preprocessor.load_and_preprocess_image_for_record(
        record,
        Path("data/samples/images"),
    )

    # Verify the tensor-like shape uses height, width, channels order.
    assert tensor.shape == (3, 3, 3)
    assert len(tensor.values) == 27

    # Verify normalization converts 0-255 integers into 0.0-1.0 floats.
    assert tensor.values[0] == 0.0
    assert tensor.values[3] == 64 / 255.0
    assert max(tensor.values) == 1.0


# Test that malformed pixel counts fail before future training code can use them.
def test_preprocess_rejects_wrong_pixel_count(load_service_module):
    # Load the preprocessing helper under test.
    preprocessor = load_service_module(
        "galaxy_image_preprocessor_bad_count",
        "scripts/preprocess_galaxy_image.py",
    )

    # Build an image whose pixel list is too short for a 2x2 RGB image.
    image = preprocessor.GalaxyImage(
        path=Path("bad.ppm"),
        width=2,
        height=2,
        channels=3,
        pixels=[0, 1, 2],
    )

    # Preprocessing should reject the inconsistent shape.
    try:
        preprocessor.preprocess_image(image)
    except ValueError as error:
        assert "Expected 12 pixel values" in str(error)
    else:
        raise AssertionError("Expected wrong pixel count to raise ValueError")


# Test that invalid pixel values fail before normalization.
def test_preprocess_rejects_out_of_range_pixels(load_service_module):
    # Load the preprocessing helper under test.
    preprocessor = load_service_module(
        "galaxy_image_preprocessor_bad_range",
        "scripts/preprocess_galaxy_image.py",
    )

    # Build a 1x1 RGB image with an impossible channel value.
    image = preprocessor.GalaxyImage(
        path=Path("bad.ppm"),
        width=1,
        height=1,
        channels=3,
        pixels=[0, 128, 300],
    )

    # Preprocessing should reject values outside the image channel range.
    try:
        preprocessor.preprocess_image(image)
    except ValueError as error:
        assert "Pixel value out of 0-255 range" in str(error)
    else:
        raise AssertionError("Expected out-of-range pixel to raise ValueError")
