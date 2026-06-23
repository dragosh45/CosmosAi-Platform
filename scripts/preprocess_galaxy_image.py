#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 explains the current image data path.

# Import argparse so preprocessing can be inspected from the command line.
import argparse

# Import dataclass to return normalized image data in a structured object.
from dataclasses import dataclass

# Import Path to handle manifest and image root paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys

# Import the tiny image loader so preprocessing starts from already-loaded pixels.
from load_galaxy_image import GalaxyImage, load_image_for_record

# Import the manifest loader so the CLI can select an image by manifest image_id.
from load_galaxy_manifest import GalaxyManifestRecord, load_manifest


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
    # Keep this strict so bad sample images fail before future training code uses them.
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


# Parse command-line arguments for the tiny preprocessing proof.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Preprocess one tiny CosmosAI galaxy image from a manifest row."
    )

    # Accept a manifest path, defaulting to the sample manifest.
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/samples/galaxy_manifest_sample.csv",
        help="Path to the galaxy manifest CSV file.",
    )

    # Accept a galaxy data root, defaulting to the sample image folder.
    parser.add_argument(
        "--galaxy-data-root",
        default="data/samples/images",
        help="Galaxy data root used to resolve manifest image_path values.",
    )

    # Accept an image ID so a specific manifest record can be preprocessed.
    parser.add_argument(
        "--image-id",
        default="gz2-000001",
        help="Manifest image_id to preprocess.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Run the tiny preprocessing proof from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    # Load manifest records so preprocessing starts from the same data contract.
    records = load_manifest(Path(args.manifest_path))

    # Find the requested image record by image_id.
    record_by_id = {record.image_id: record for record in records}
    if args.image_id not in record_by_id:
        print(f"Image ID not found in manifest: {args.image_id}")
        return 1

    try:
        # Load and preprocess the selected image.
        tensor = load_and_preprocess_image_for_record(
            record_by_id[args.image_id],
            Path(args.galaxy_data_root),
        )
    except (FileNotFoundError, ValueError) as error:
        # Print preprocessing failures and return a non-zero exit code.
        print(error)
        return 1

    # Print a compact summary for human inspection.
    print(f"Preprocessed image: {tensor.path}")
    print(f"shape: {tensor.shape}")
    print(f"normalized_values: {len(tensor.values)}")
    print(f"min_value: {min(tensor.values):.4f}")
    print(f"max_value: {max(tensor.values):.4f}")

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
