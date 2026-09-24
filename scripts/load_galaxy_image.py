#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 and docs/excalidraw/manifest_tooling_code_flow.excalidraw explain this image loader.
# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.

# Import argparse so the image loader can be used from the command line.
import argparse

# Import dataclass to return image metadata in a structured way.
from dataclasses import dataclass

# Import Path to work with filesystem paths clearly.
from pathlib import Path

# Import sys so the command-line entrypoint can return success or failure.
import sys

# Import manifest helpers so image loading can start from a manifest row.
from load_galaxy_manifest import (
    GalaxyManifestRecord,
    load_manifest,
    resolve_image_path,
)


# Normal image formats that Pillow can read for future real datasets.
PILLOW_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


# Load a normal PNG/JPG/JPEG image from disk with Pillow.
def load_pillow_image(
    image_path: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyImage:
    # Import Pillow only for real image formats so the tiny PPM path stays simple.
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to load PNG/JPG galaxy images. "
            "Install project dependencies with: "
            ".venv/bin/python -m pip install -r requirements.txt"
        ) from error

    # Open the image through Pillow, which understands normal binary image formats.
    with Image.open(image_path) as opened_image:
        # Convert every image to RGB so downstream code always receives 3 channels.
        rgb_image = opened_image.convert("RGB")

        # Resize when requested so future CNN batches can have one consistent shape.
        if target_size is not None:
            rgb_image = rgb_image.resize(
                target_size,
                resample=Image.Resampling.BILINEAR,
            )

        # Pillow reports size as width, height.
        width, height = rgb_image.size

        # Use Pillow's newest pixel reader when available to avoid deprecation noise.
        pixel_rows = (
            rgb_image.get_flattened_data()
            if hasattr(rgb_image, "get_flattened_data")
            else rgb_image.getdata()
        )

        # Flatten each RGB tuple into the same [R, G, B, R, G, B, ...] layout as PPM.
        pixels = [
            channel_value
            for pixel in pixel_rows
            for channel_value in pixel
        ]

    # Return the same object type as the PPM loader so preprocessing is unchanged.
    return GalaxyImage(
        path=image_path,
        width=width,
        height=height,
        channels=3,
        pixels=pixels,
    )


# Load one supported image file, dispatching by file extension.
def load_image_file(
    image_path: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyImage:
    # Normalize the extension so .JPG and .jpg are treated the same.
    extension = image_path.suffix.lower()

    # Keep the old tiny PPM parser for text-based learning fixtures.
    if extension == ".ppm":
        image = load_ppm_image(image_path)

        # PPM fixtures are intentionally tiny and hand-readable; do not resize them.
        if target_size is not None and (image.width, image.height) != target_size:
            raise ValueError("PPM resizing is not supported; use PNG/JPG for resizing")

        return image

    # Use Pillow for normal image formats that real datasets usually contain.
    if extension in PILLOW_IMAGE_EXTENSIONS:
        return load_pillow_image(image_path, target_size=target_size)

    # Fail clearly when a manifest points at an unsupported file type.
    allowed_extensions = [".ppm", *sorted(PILLOW_IMAGE_EXTENSIONS)]
    raise ValueError(
        "Unsupported image extension "
        f"'{extension}' for {image_path}; allowed: {', '.join(allowed_extensions)}"
    )


# Load the image referenced by one manifest record.
def load_image_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyImage:
    # Resolve the manifest-relative image path against the provided data root.
    image_path = resolve_image_path(record, galaxy_data_root)

    # Fail clearly when the manifest points to a missing image file.
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    # Load and return the image using the right parser for its file extension.
    return load_image_file(image_path, target_size=target_size)


# Convert optional image width/height CLI values into a Pillow resize target.
def target_size_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    # If neither value is passed, do not resize images.
    if args.image_width is None and args.image_height is None:
        return None

    # Require both dimensions so resizing cannot silently distort intent.
    if args.image_width is None or args.image_height is None:
        raise ValueError("--image-width and --image-height must be used together")

    # Both dimensions must be positive pixel counts.
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("--image-width and --image-height must be at least 1")

    # Pillow expects target size as width, height.
    return (args.image_width, args.image_height)


# Parse command-line arguments for the tiny image-loading proof.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Load one tiny CosmosAI galaxy image from a manifest row."
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

    # Accept an image ID so a specific manifest record can be loaded.
    parser.add_argument(
        "--image-id",
        default="gz2-000001",
        help="Manifest image_id to load.",
    )

    # Optionally resize normal PNG/JPG inputs before printing the loaded shape.
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Optional target image width for Pillow-loaded PNG/JPG files.",
    )

    # Keep height separate so commands say exactly which image shape they want.
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Optional target image height for Pillow-loaded PNG/JPG files.",
    )

    # Return the parsed arguments.
    return parser.parse_args()


# Run the tiny image loader from the command line.
def main() -> int:
    # Read command-line arguments.
    args = parse_args()

    # Load manifest records first so this proof follows the same future data path.
    records = load_manifest(Path(args.manifest_path))

    # Find the requested image record by image_id.
    record_by_id = {record.image_id: record for record in records}
    if args.image_id not in record_by_id:
        print(f"Image ID not found in manifest: {args.image_id}")
        return 1

    try:
        # Convert optional resize arguments before loading the selected image.
        target_size = target_size_from_args(args)

        # Load the image referenced by the selected manifest record.
        image = load_image_for_record(
            record_by_id[args.image_id],
            Path(args.galaxy_data_root),
            target_size=target_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        # Print image loading failures and return a non-zero exit code.
        print(error)
        return 1

    # Print a compact summary for human inspection.
    print(f"Loaded image: {image.path}")
    print(f"width: {image.width}")
    print(f"height: {image.height}")
    print(f"channels: {image.channels}")
    print(f"pixel_values: {len(image.pixels)}")

    # Return success.
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
