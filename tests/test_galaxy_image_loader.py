# Import Path to compare loaded image paths.
from pathlib import Path

# Import Pillow in tests so we can create a tiny normal PNG fixture at runtime.
from PIL import Image

# Import shared package loaders to test the real training/inference image path.
from cosmosai.galaxy.image_loader import load_image_file, load_image_for_record

# Import the shared manifest record used by the package loader.
from cosmosai.galaxy.manifest import GalaxyManifestRecord


# Create a tiny PNG file for real-format image loader tests.
def _write_tiny_png(image_path: Path) -> None:
    # Create a tiny RGB image with two concrete pixels.
    image = Image.new("RGB", (2, 1))

    # Store known channel values so exact non-resized loading can be asserted.
    image.putdata([(10, 20, 30), (200, 210, 220)])

    # Save as PNG so the loader must use Pillow instead of the PPM parser.
    image.save(image_path)


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


# Test that the shared package can load a normal PNG image with Pillow.
def test_load_png_image_with_pillow(tmp_path):
    # Write a tiny PNG image into pytest's temporary folder.
    image_path = tmp_path / "tiny-galaxy.png"
    _write_tiny_png(image_path)

    # Load the PNG through the new real-format image loader.
    image = load_image_file(image_path)

    # Verify the PNG became the same GalaxyImage shape used by PPM fixtures.
    assert image.path == image_path
    assert image.width == 2
    assert image.height == 1
    assert image.channels == 3
    assert image.pixels == [10, 20, 30, 200, 210, 220]


# Test that Pillow resizing gives future CNN batches a consistent shape.
def test_load_png_image_with_pillow_resize(tmp_path):
    # Write a tiny PNG image into pytest's temporary folder.
    image_path = tmp_path / "tiny-galaxy.png"
    _write_tiny_png(image_path)

    # Resize to width=4, height=3 before returning pixels.
    image = load_image_file(image_path, target_size=(4, 3))

    # Verify the resized image has model-ready RGB dimensions.
    assert image.width == 4
    assert image.height == 3
    assert image.channels == 3
    assert len(image.pixels) == 4 * 3 * 3
    assert all(0 <= pixel <= 255 for pixel in image.pixels)


# Test that a manifest row pointing to a PNG works through the package loader.
def test_load_manifest_png_record_with_resize(tmp_path):
    # Create a tiny PNG under a fake data root, matching manifest-relative paths.
    image_root = tmp_path / "images"
    image_folder = image_root / "processed" / "images_224"
    image_folder.mkdir(parents=True)
    image_path = image_folder / "tiny-galaxy.png"
    _write_tiny_png(image_path)

    # Build a manifest-style record that points to the PNG file.
    record = GalaxyManifestRecord(
        image_id="png-001",
        image_path=Path("processed/images_224/tiny-galaxy.png"),
        label="spiral",
        split="train",
        source="test",
    )

    # Load and resize the PNG through the package manifest path.
    image = load_image_for_record(record, image_root, target_size=(3, 3))

    # Verify resize shape matches future fixed-size training needs.
    assert image.path == image_path
    assert image.width == 3
    assert image.height == 3
    assert image.channels == 3
    assert len(image.pixels) == 27
