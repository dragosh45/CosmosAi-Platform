# Import Path to pass local sample paths into the training skeleton.
from pathlib import Path


# Build dataset splits through the same helper the command-line script uses.
def _load_sample_splits(training_script):
    # Load tiny local split data from the sample manifest.
    return training_script.load_dataset_splits_from_manifest(
        Path("data/samples/galaxy_manifest_sample.csv"),
        Path("data/samples/images"),
    )


# Test that the skeleton can run one training-shaped step from current sample data.
def test_training_loop_skeleton_runs_one_step(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_skeleton",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current sample splits using the production helper.
    dataset_splits = _load_sample_splits(training_script)

    # Run one skeleton epoch.
    summary = training_script.run_training_loop(dataset_splits, epochs=1)

    # Verify the current tiny sample shape.
    assert summary.epochs == 1
    assert summary.train_sample_count == 1
    assert summary.val_sample_count == 0
    assert summary.test_sample_count == 0
    assert summary.skipped_row_count == 3
    assert summary.training_steps == 1
    assert summary.average_loss > 0.0

    # Verify the first step keeps prediction-like output separate from the true label.
    assert summary.first_step is not None
    assert summary.first_step.image_id == "gz2-000001"
    assert summary.first_step.label == "spiral"
    assert summary.first_step.label_id == 1
    assert len(summary.first_step.probabilities) == 4
    assert abs(sum(summary.first_step.probabilities) - 1.0) < 0.000001


# Test that epochs repeat the training-shaped loop over train samples.
def test_training_loop_skeleton_repeats_for_epochs(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_skeleton_epochs",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current sample splits using the production helper.
    dataset_splits = _load_sample_splits(training_script)

    # Run two skeleton epochs.
    summary = training_script.run_training_loop(dataset_splits, epochs=2)

    # One train sample repeated for two epochs should produce two steps.
    assert summary.epochs == 2
    assert summary.training_steps == 2
    assert summary.average_loss > 0.0


# Test that invalid epoch counts fail clearly.
def test_training_loop_rejects_invalid_epochs(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_skeleton_bad_epochs",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current sample splits using the production helper.
    dataset_splits = _load_sample_splits(training_script)

    # Zero epochs should not be accepted.
    try:
        training_script.run_training_loop(dataset_splits, epochs=0)
    except ValueError as error:
        assert "epochs must be at least 1" in str(error)
    else:
        raise AssertionError("Expected invalid epochs to raise ValueError")
