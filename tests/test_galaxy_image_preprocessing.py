# Import Path to pass the sample image root into preprocessing helpers.
from pathlib import Path

# Import Pillow in tests so we can create a normal PNG fixture at runtime.
from PIL import Image

# Import shared manifest and preprocessing helpers for the real image path.
from cosmosai.galaxy.manifest import GalaxyManifestRecord
from cosmosai.galaxy.preprocessing import load_and_preprocess_image_for_record


# Create a tiny PNG file for package preprocessing tests.
def _write_tiny_png(image_path: Path) -> None:
    # Create a tiny RGB image with two visible pixel values.
    image = Image.new("RGB", (2, 1))

    # Store black and white pixels so normalized min/max are easy to reason about.
    image.putdata([(0, 0, 0), (255, 255, 255)])

    # Save as PNG so the loader must go through Pillow.
    image.save(image_path)


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


# Test that a normal PNG can be resized and normalized into model-ready numbers.
def test_preprocess_png_image_with_resize(tmp_path):
    # Create a fake image root and one PNG file under a manifest-style path.
    image_root = tmp_path / "images"
    image_folder = image_root / "processed" / "images_224"
    image_folder.mkdir(parents=True)
    image_path = image_folder / "tiny-galaxy.png"
    _write_tiny_png(image_path)

    # Build a manifest-style record that points to the PNG.
    record = GalaxyManifestRecord(
        image_id="png-001",
        image_path=Path("processed/images_224/tiny-galaxy.png"),
        label="spiral",
        split="train",
        source="test",
    )

    # Resize to width=3, height=3 and normalize pixels into floats.
    tensor = load_and_preprocess_image_for_record(
        record,
        image_root,
        target_size=(3, 3),
    )

    # Tensor shape uses height, width, channels order after Pillow resizing.
    assert tensor.path == image_path
    assert tensor.shape == (3, 3, 3)
    assert len(tensor.values) == 27
    assert min(tensor.values) >= 0.0
    assert max(tensor.values) <= 1.0
