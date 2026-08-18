"""Tiny image loading helpers used by local galaxy proofs."""

# Import dataclass to return image metadata in a structured object.
from dataclasses import dataclass

# Import Path to work with filesystem paths clearly.
from pathlib import Path

# Import manifest helpers so image loading can start from a manifest row.
from cosmosai.galaxy.manifest import GalaxyManifestRecord, resolve_image_path


# Store basic information loaded from a tiny image file.
@dataclass(frozen=True)
class GalaxyImage:
    # Full path to the image file that was loaded.
    path: Path

    # Image width in pixels.
    width: int

    # Image height in pixels.
    height: int

    # Number of color channels. PPM P3 stores RGB, so this is 3.
    channels: int

    # Flat list of RGB pixel values, kept simple for the tiny proof step.
    pixels: list[int]


# Read tokens from a plain-text PPM P3 image while ignoring comments.
def read_ppm_tokens(image_path: Path) -> list[str]:
    # Store every meaningful whitespace-separated token from the image file.
    tokens: list[str] = []

    # PPM P3 is text, so it can be read without image libraries.
    with image_path.open(encoding="utf-8") as image_file:
        # Process one line at a time so comments can be removed safely.
        for line in image_file:
            # Ignore comments after # because PPM files allow comment lines.
            content = line.split("#", 1)[0]

            # Add all non-comment tokens from this line.
            tokens.extend(content.split())

    # Return the parsed tokens to the caller.
    return tokens


# Load a tiny PPM P3 image from disk.
def load_ppm_image(image_path: Path) -> GalaxyImage:
    # Read the PPM tokens from the file.
    tokens = read_ppm_tokens(image_path)

    # A valid tiny PPM needs at least magic, width, height, max value, and pixels.
    if len(tokens) < 4:
        raise ValueError(f"Image file is too short: {image_path}")

    # The sample proof uses P3 because it is easy to inspect as text.
    if tokens[0] != "P3":
        raise ValueError(f"Unsupported image format for tiny loader: {tokens[0]}")

    # Parse image dimensions and max channel value.
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])

    # Keep the tiny loader strict so tests catch malformed sample images.
    if max_value != 255:
        raise ValueError(f"Unsupported PPM max value: {max_value}")

    # Convert the remaining tokens into integer RGB channel values.
    pixels = [int(token) for token in tokens[4:]]

    # RGB images should have width * height * 3 values.
    expected_value_count = width * height * 3
    if len(pixels) != expected_value_count:
        raise ValueError(
            f"Expected {expected_value_count} pixel values, got {len(pixels)}"
        )

    # Return structured image information for future preprocessing steps.
    return GalaxyImage(
        path=image_path,
        width=width,
        height=height,
        channels=3,
        pixels=pixels,
    )


# Load the image referenced by one manifest record.
def load_image_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
) -> GalaxyImage:
    # Resolve the manifest-relative image path against the provided data root.
    image_path = resolve_image_path(record, galaxy_data_root)

    # Fail clearly when the manifest points to a missing image file.
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    # Load and return the tiny sample image.
    return load_ppm_image(image_path)

