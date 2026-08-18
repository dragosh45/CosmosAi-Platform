"""Galaxy image preprocessing helpers for model-ready tensor values."""

# Import dataclass to return normalized image data in a structured object.
from dataclasses import dataclass

# Import Path to keep source image paths in tensor metadata.
from pathlib import Path

# Import image and manifest objects used by the preprocessing path.
from cosmosai.galaxy.image_loader import GalaxyImage, load_image_for_record
from cosmosai.galaxy.manifest import GalaxyManifestRecord


# Store normalized image data in a simple tensor-like structure.
@dataclass(frozen=True)
class GalaxyImageTensor:
    # Full path to the source image that was preprocessed.
    path: Path

    # Tensor shape in height, width, channels order.
    shape: tuple[int, int, int]

    # Flat normalized pixel values between 0.0 and 1.0.
    values: list[float]


# Convert raw 0-255 pixel values into normalized 0.0-1.0 values.
def normalize_pixel_values(pixels: list[int]) -> list[float]:
    # Keep this strict so bad images fail before future training code uses them.
    for pixel in pixels:
        if pixel < 0 or pixel > 255:
            raise ValueError(f"Pixel value out of 0-255 range: {pixel}")

    # Divide by 255 so model code receives small floating-point inputs.
    return [pixel / 255.0 for pixel in pixels]


# Convert a loaded GalaxyImage into a normalized tensor-like object.
def preprocess_image(image: GalaxyImage) -> GalaxyImageTensor:
    # The current tiny proof only supports RGB images.
    if image.channels != 3:
        raise ValueError(f"Expected 3 image channels, got {image.channels}")

    # Confirm that the flat pixel list matches height * width * channels.
    expected_value_count = image.height * image.width * image.channels
    if len(image.pixels) != expected_value_count:
        raise ValueError(
            f"Expected {expected_value_count} pixel values, got {len(image.pixels)}"
        )

    # Normalize integer pixels to floating-point values.
    normalized_values = normalize_pixel_values(image.pixels)

    # Return the simple tensor-like object for future CNN preprocessing code.
    return GalaxyImageTensor(
        path=image.path,
        shape=(image.height, image.width, image.channels),
        values=normalized_values,
    )


# Load and preprocess the image referenced by one manifest record.
def load_and_preprocess_image_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
) -> GalaxyImageTensor:
    # Reuse the loader so preprocessing does not duplicate image parsing logic.
    image = load_image_for_record(record, galaxy_data_root)

    # Convert the loaded image to normalized tensor-like values.
    return preprocess_image(image)

