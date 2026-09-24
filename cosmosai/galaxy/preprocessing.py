"""Galaxy image preprocessing helpers for model-ready tensor values."""

# Docs: docs/architecture.md Steps 16 and 19 explain image preparation and its saved recipe.

# Import dataclass to return normalized image data in a structured object.
from dataclasses import dataclass

# Import Path to keep source image paths in tensor metadata.
from pathlib import Path

# Import image and manifest objects used by the preprocessing path.
from cosmosai.galaxy.image_loader import GalaxyImage, load_image_for_record
from cosmosai.galaxy.manifest import GalaxyManifestRecord


class CheckpointContractError(ValueError):
    """The saved input recipe is missing or incompatible with this pipeline."""


# Docs: docs/concepts_explanations.md "Checkpoint Preprocessing Contract".
@dataclass(frozen=True)
class GalaxyPreprocessingPolicy:
    # Pillow uses (width, height), NOT the tensor's (height, width, channels).
    # None explicitly means keep the source size. It does not mean missing metadata.
    target_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.target_size is not None and (
            not isinstance(self.target_size, tuple)
            or len(self.target_size) != 2
            or any(type(value) is not int or value < 1 for value in self.target_size)
        ):
            raise CheckpointContractError("Preprocessing target_size must be two positive integers")

    def to_metadata(self) -> dict[str, object]:
        # These fixed values describe our actual loader and normalize_pixel_values().
        # Plain dictionaries/lists can be loaded with torch's weights_only reader.
        return {
            "version": 1,
            "color_mode": "RGB",
            "target_size": list(self.target_size) if self.target_size is not None else None,
            "resize_interpolation": "bilinear",
            "normalization": "divide_by_255",
        }

    @classmethod
    def from_metadata(cls, metadata: object) -> "GalaxyPreprocessingPolicy":
        if metadata is None:
            raise CheckpointContractError(
                "Checkpoint has no preprocessing metadata. Regenerate it with the current "
                "training CLI and the original --image-width/--image-height settings "
                "(omit both only if training did not resize). Legacy settings cannot be inferred."
            )
        expected = cls().to_metadata()
        if not isinstance(metadata, dict) or metadata.keys() != expected.keys():
            raise CheckpointContractError("Invalid checkpoint preprocessing fields")
        for key, value in expected.items():
            if key != "target_size" and (
                type(metadata[key]) is not type(value) or metadata[key] != value
            ):
                raise CheckpointContractError(f"Unsupported checkpoint preprocessing {key}")
        size = metadata["target_size"]
        if size is not None and not isinstance(size, list):
            raise CheckpointContractError("Preprocessing target_size must be a list or null")
        return cls(target_size=tuple(size) if size is not None else None)

    def resolve_target_size(
        self, requested_size: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        # No CLI override means use the saved recipe, including an explicit no-resize.
        if requested_size is not None:
            GalaxyPreprocessingPolicy(target_size=requested_size)
        if requested_size is not None and requested_size != self.target_size:
            raise CheckpointContractError(
                f"Requested target_size {requested_size} conflicts with checkpoint "
                f"target_size {self.target_size}; omit resize flags to use the saved policy"
            )
        return self.target_size

    def validate_tensor_shape(self, shape: tuple[int, int, int]) -> None:
        # Prepared-sample callers still own RGB conversion/normalization. Shape alone
        # cannot prove those operations happened, but catches a wrong fixed resize.
        if self.target_size is not None:
            width, height = self.target_size
            if shape != (height, width, 3):
                raise CheckpointContractError(
                    f"Prepared tensor shape {shape} does not match checkpoint "
                    f"shape {(height, width, 3)}"
                )


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
    target_size: tuple[int, int] | None = None,
) -> GalaxyImageTensor:
    # Reuse the loader so preprocessing does not duplicate image parsing or resizing.
    image = load_image_for_record(
        record,
        galaxy_data_root,
        target_size=target_size,
    )

    # Convert the loaded image to normalized tensor-like values.
    return preprocess_image(image)
