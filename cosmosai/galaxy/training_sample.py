"""Galaxy training-sample helpers shared by CLI and service inference."""

# Docs: docs/architecture.md Step 16 explains the Pillow real-image loading proof.

# Import dataclass to return one training sample as a structured object.
from dataclasses import dataclass

# Import Path to handle local and mounted image roots.
from pathlib import Path

# Import shared label mapping and preprocessing helpers.
from cosmosai.galaxy.labels import LABEL_TO_ID
from cosmosai.galaxy.manifest import GalaxyManifestRecord
from cosmosai.galaxy.preprocessing import (
    GalaxyImageTensor,
    load_and_preprocess_image_for_record,
)


@dataclass(frozen=True)
class GalaxyInferenceSample:
    """Prepared pixels for prediction; no invented label or training split."""

    image_id: str
    tensor: GalaxyImageTensor


# Store one model-training-style example.
@dataclass(frozen=True)
class GalaxyTrainingSample:
    # Stable image identifier from the manifest.
    image_id: str

    # Human-readable galaxy label from the manifest.
    label: str

    # Numeric class ID used by model training code.
    label_id: int

    # Dataset split from the manifest: train, val, or test.
    split: str

    # Normalized tensor-like image data.
    tensor: GalaxyImageTensor


@dataclass(frozen=True, init=False)
class GalaxyLazyTrainingSample:
    """Manifest-backed sample that loads one tensor only when accessed."""

    image_id: str
    label: str
    label_id: int
    split: str
    record: GalaxyManifestRecord | None
    galaxy_data_root: Path | None
    target_size: tuple[int, int] | None
    _tensor_override: GalaxyImageTensor | None

    def __init__(
        self,
        image_id: str,
        label: str,
        label_id: int,
        split: str,
        record: GalaxyManifestRecord | None = None,
        galaxy_data_root: Path | None = None,
        target_size: tuple[int, int] | None = None,
        tensor: GalaxyImageTensor | None = None,
    ) -> None:
        # Accept tensor= for compatibility with older small shape tests and
        # callers that need a manually supplied teaching fixture.
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "label_id", label_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "galaxy_data_root", galaxy_data_root)
        object.__setattr__(self, "target_size", target_size)
        object.__setattr__(self, "_tensor_override", tensor)

    @property
    def tensor(self) -> GalaxyImageTensor:
        return self.load_tensor()

    def load_tensor(self) -> GalaxyImageTensor:
        if self._tensor_override is not None:
            return self._tensor_override
        if self.record is None or self.galaxy_data_root is None:
            raise ValueError("Lazy sample has no manifest record or data root")
        # Do not cache this result: DataLoader owns the current batch, so the
        # dataset does not accumulate decoded tensors as the dataset grows.
        return load_and_preprocess_image_for_record(
            self.record,
            self.galaxy_data_root,
            target_size=self.target_size,
        )


# Convert a human-readable label into the stable numeric class ID.
def label_to_id(label: str) -> int:
    # Fail clearly when code receives a label outside the current contract.
    if label not in LABEL_TO_ID:
        allowed_labels = ", ".join(sorted(LABEL_TO_ID))
        raise ValueError(f"Unknown label '{label}' (allowed: {allowed_labels})")

    # Return the numeric ID used by training and checkpoint metadata.
    return LABEL_TO_ID[label]


# Create one training sample from a manifest record.
def create_training_sample_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyTrainingSample:
    # Load, optionally resize, and normalize the image referenced by this row.
    tensor = load_and_preprocess_image_for_record(
        record,
        galaxy_data_root,
        target_size=target_size,
    )

    # Convert the human label into a stable numeric target.
    numeric_label = label_to_id(record.label)

    # Return one input/target pair.
    return GalaxyTrainingSample(
        image_id=record.image_id,
        label=record.label,
        label_id=numeric_label,
        split=record.split,
        tensor=tensor,
    )


def create_lazy_training_sample_for_record(
    record: GalaxyManifestRecord,
    galaxy_data_root: Path,
    target_size: tuple[int, int] | None = None,
) -> GalaxyLazyTrainingSample:
    """Create metadata for a sample; image decoding remains a Dataset concern."""
    return GalaxyLazyTrainingSample(
        image_id=record.image_id,
        label=record.label,
        label_id=label_to_id(record.label),
        split=record.split,
        record=record,
        galaxy_data_root=galaxy_data_root,
        target_size=target_size,
    )
