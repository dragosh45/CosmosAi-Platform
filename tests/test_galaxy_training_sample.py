# Import Path to pass the sample image root into training sample helpers.
from pathlib import Path


# Test that one manifest row can become a training-style sample.
def test_create_training_sample_from_manifest_record(load_service_module):
    # Load the manifest helper so the test follows the real manifest path.
    manifest_loader = load_service_module(
        "galaxy_manifest_loader_for_training_sample",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the training sample helper under test.
    sample_creator = load_service_module(
        "galaxy_training_sample_creator",
        "scripts/create_galaxy_training_sample.py",
    )

    # Load the sample manifest and select the row that points to the sample image.
    records = manifest_loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )
    record = records[0]

    # Create one model-training-style sample from the manifest row.
    sample = sample_creator.create_training_sample_for_record(
        record,
        Path("data/samples/images"),
    )

    # Verify metadata and numeric target.
    assert sample.image_id == "gz2-000001"
    assert sample.label == "spiral"
    assert sample.label_id == 1
    assert sample.split == "train"

    # Verify the normalized input side of the sample.
    assert sample.tensor.shape == (3, 3, 3)
    assert len(sample.tensor.values) == 27
    assert sample.tensor.values[3] == 64 / 255.0


# Test the current stable label mapping.
def test_label_to_id_mapping(load_service_module):
    # Load the training sample helper under test.
    sample_creator = load_service_module(
        "galaxy_training_sample_label_mapping",
        "scripts/create_galaxy_training_sample.py",
    )

    # Verify the current label IDs are stable and explicit.
    assert sample_creator.label_to_id("elliptical") == 0
    assert sample_creator.label_to_id("spiral") == 1
    assert sample_creator.label_to_id("lenticular") == 2
    assert sample_creator.label_to_id("irregular") == 3


# Test that unknown labels fail before future training code can use them.
def test_label_to_id_rejects_unknown_label(load_service_module):
    # Load the training sample helper under test.
    sample_creator = load_service_module(
        "galaxy_training_sample_bad_label",
        "scripts/create_galaxy_training_sample.py",
    )

    # Unknown labels should not silently become a numeric class.
    try:
        sample_creator.label_to_id("unknown")
    except ValueError as error:
        assert "Unknown label" in str(error)
    else:
        raise AssertionError("Expected unknown label to raise ValueError")
