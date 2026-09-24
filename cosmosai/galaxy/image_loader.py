"""Tiny image loading helpers used by local galaxy proofs."""

# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.
# Docs: docs/architecture.md Step 20 explains corrupt-image rejection.

# Import dataclass to return image metadata in a structured object.
from dataclasses import dataclass

# Import Path to work with filesystem paths clearly.
from pathlib import Path

# Import manifest helpers so image loading can start from a manifest row.
from cosmosai.galaxy.manifest import GalaxyManifestRecord, resolve_image_path


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
    try:
        tokens = read_ppm_tokens(image_path)
    except UnicodeError as error:
        raise ValueError(f"PPM file is not valid UTF-8 text: {image_path}") from error

    # A valid tiny PPM needs at least magic, width, height, max value, and pixels.
    if len(tokens) < 4:
        raise ValueError(f"Image file is too short: {image_path}")

    # The sample proof uses P3 because it is easy to inspect as text.
    if tokens[0] != "P3":
        raise ValueError(f"Unsupported image format for tiny loader: {tokens[0]}")

    # Parse image dimensions and max channel value with file-specific diagnostics.
    try:
        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
    except ValueError as error:
        raise ValueError(
            f"PPM width, height, and max value must be integers: {image_path}"
        ) from error

    if width < 1 or height < 1:
        raise ValueError(
            f"PPM width and height must be positive, got {width}x{height}: "
            f"{image_path}"
        )

    # Keep the tiny loader strict so tests catch malformed sample images.
    if max_value != 255:
        raise ValueError(f"Unsupported PPM max value: {max_value}")

    # Convert the remaining tokens into integer RGB channel values.
    try:
        pixels = [int(token) for token in tokens[4:]]
    except ValueError as error:
        raise ValueError(f"PPM pixel values must be integers: {image_path}") from error

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

    try:
        # Open the image through Pillow, which understands normal binary formats.
        with Image.open(image_path) as opened_image:
            # Convert every image to RGB so downstream code always receives 3 channels.
            rgb_image = opened_image.convert("RGB")

            # Resize when requested so future CNN batches have one consistent shape.
            if target_size is not None:
                rgb_image = rgb_image.resize(
                    target_size,
                    resample=Image.Resampling.BILINEAR,
                )

            # Pillow reports size as width, height.
            width, height = rgb_image.size

            # Use Pillow's newest reader when available to avoid deprecation noise.
            pixel_rows = (
                rgb_image.get_flattened_data()
                if hasattr(rgb_image, "get_flattened_data")
                else rgb_image.getdata()
            )

            # Flatten RGB tuples into the same [R, G, B, ...] layout as PPM.
            pixels = [
                channel_value
                for pixel in pixel_rows
                for channel_value in pixel
            ]
    except OSError as error:
        raise ValueError(f"Unreadable image file {image_path}: {error}") from error

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


def validate_image_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
    target_size: tuple[int, int] | None = None,
) -> None:
    """Validate one source file without retaining a model-ready tensor."""
    image_path = resolve_image_path(record, galaxy_data_root)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    extension = image_path.suffix.lower()
    if extension == ".ppm":
        image = load_ppm_image(image_path)
        if target_size is not None and (image.width, image.height) != target_size:
            raise ValueError("PPM resizing is not supported; use PNG/JPG for resizing")
        if any(pixel < 0 or pixel > 255 for pixel in image.pixels):
            raise ValueError(f"Pixel value out of 0-255 range: {image_path}")
        return

    if extension not in PILLOW_IMAGE_EXTENSIONS:
        allowed_extensions = [".ppm", *sorted(PILLOW_IMAGE_EXTENSIONS)]
        raise ValueError(
            "Unsupported image extension "
            f"'{extension}' for {image_path}; allowed: {', '.join(allowed_extensions)}"
        )

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to validate PNG/JPG galaxy images. "
            "Install project dependencies with: "
            ".venv/bin/python -m pip install -r requirements.txt"
        ) from error

    try:
        with Image.open(image_path) as opened_image:
            opened_image.verify()
    except OSError as error:
        raise ValueError(f"Unreadable image file {image_path}: {error}") from error
