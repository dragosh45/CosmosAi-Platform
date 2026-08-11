# Import Path to pass local sample and checkpoint paths into the prediction script.
from pathlib import Path


# Build a temporary checkpoint through the current tiny training helper.
def _create_tiny_checkpoint(load_service_module, tmp_path):
    # Load the training script so the test can create a checkpoint artifact.
    training_script = load_service_module(
        "galaxy_checkpoint_training_helper",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current tiny sample splits.
    dataset_splits = training_script.load_dataset_splits_from_manifest(
        Path("data/samples/galaxy_manifest_sample.csv"),
        Path("data/samples/images"),
    )

    # Train one tiny model for two epochs, matching the documented command.
    model = training_script.create_tiny_galaxy_cnn()
    training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
    )

    # Save the checkpoint into pytest's temporary folder.
    checkpoint_path = tmp_path / "tiny_galaxy_cnn.pt"
    training_script.save_torch_checkpoint(model, checkpoint_path)
    return checkpoint_path


# Test that checkpoint inference loads saved weights and predicts one sample.
def test_predict_from_checkpoint_returns_loaded_model_prediction(
    load_service_module,
    tmp_path,
):
    # Create a temporary checkpoint using the training code from the prior milestone.
    checkpoint_path = _create_tiny_checkpoint(load_service_module, tmp_path)

    # Load the prediction script under test.
    prediction_script = load_service_module(
        "galaxy_checkpoint_prediction",
        "scripts/predict_galaxy_checkpoint.py",
    )

    # Predict one sample using only the saved checkpoint and existing data pipeline.
    prediction = prediction_script.predict_from_checkpoint(
        Path("data/samples/galaxy_manifest_sample.csv"),
        Path("data/samples/images"),
        "gz2-000001",
        checkpoint_path,
    )

    # Verify the script loaded the same tiny image and checkpoint.
    assert prediction.checkpoint_path == checkpoint_path
    assert prediction.image_id == "gz2-000001"
    assert prediction.manifest_label == "spiral"
    assert prediction.input_shape == (1, 3, 3, 3)

    # Verify the checkpoint prediction is readable and deterministic.
    assert prediction.predicted_label_id == 1
    assert prediction.predicted_label == "spiral"
    assert len(prediction.logits) == len(prediction_script.LABEL_TO_ID)
    assert len(prediction.probabilities) == len(prediction_script.LABEL_TO_ID)
    assert abs(sum(prediction.probabilities) - 1.0) < 0.000001


# Test that unknown manifest IDs fail clearly.
def test_predict_from_checkpoint_rejects_unknown_image_id(
    load_service_module,
    tmp_path,
):
    # Create a temporary checkpoint so the only expected error is image lookup.
    checkpoint_path = _create_tiny_checkpoint(load_service_module, tmp_path)

    # Load the prediction script under test.
    prediction_script = load_service_module(
        "galaxy_checkpoint_prediction_bad_id",
        "scripts/predict_galaxy_checkpoint.py",
    )

    # Unknown image IDs should raise a readable ValueError.
    try:
        prediction_script.predict_from_checkpoint(
            Path("data/samples/galaxy_manifest_sample.csv"),
            Path("data/samples/images"),
            "missing-image",
            checkpoint_path,
        )
    except ValueError as error:
        assert "Image ID not found in manifest: missing-image" in str(error)
    else:
        raise AssertionError("Expected missing image_id to raise ValueError")
