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


# Test that the placeholder model exposes the future model-style forward interface.
def test_placeholder_model_forward_returns_one_logit_per_label(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_placeholder_model",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current sample splits using the production helper.
    dataset_splits = _load_sample_splits(training_script)
    train_sample = dataset_splits.samples_by_split["train"][0]

    # Create the placeholder model and run a forward pass.
    model = training_script.PlaceholderGalaxyModel()
    logits = model.forward(train_sample)

    # Verify the placeholder model produces one raw score per known label.
    assert len(logits) == len(training_script.LABEL_TO_ID)
    assert logits == training_script.placeholder_cnn_logits(train_sample)


# Test that the current project tensor can become a PyTorch NCHW image tensor.
def test_galaxy_tensor_to_torch_image_shape(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_tensor",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current sample splits using the production helper.
    dataset_splits = _load_sample_splits(training_script)
    train_sample = dataset_splits.samples_by_split["train"][0]

    # Convert the project tensor object into a PyTorch image tensor.
    torch_image = training_script.galaxy_tensor_to_torch_image(train_sample)

    # PyTorch Conv2d expects NCHW: batch, channels, height, width.
    assert tuple(torch_image.shape) == (1, 3, 3, 3)
    assert torch_image.dtype == training_script.torch.float32
    assert abs(torch_image[0, 0, 0, 1].item() - 64 / 255.0) < 0.000001


# Test that the tiny real PyTorch CNN can produce logits without training.
def test_tiny_galaxy_cnn_forward_pass_returns_logits(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_forward",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load current sample splits using the production helper.
    dataset_splits = _load_sample_splits(training_script)
    train_sample = dataset_splits.samples_by_split["train"][0]

    # Run one real PyTorch forward pass.
    result = training_script.run_torch_forward_pass(train_sample)

    # Verify the model receives the current tiny image in PyTorch's expected shape.
    assert result.image_id == "gz2-000001"
    assert result.input_shape == (1, 3, 3, 3)

    # Verify the model returns one logit and probability per known label.
    assert len(result.logits) == len(training_script.LABEL_TO_ID)
    assert len(result.probabilities) == len(training_script.LABEL_TO_ID)
    assert abs(sum(result.probabilities) - 1.0) < 0.000001
    assert 0 <= result.predicted_label_id < len(training_script.LABEL_TO_ID)


# Test that one real PyTorch training step updates model weights.
def test_tiny_galaxy_cnn_training_step_changes_weight(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_training_step",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the one current usable train sample.
    dataset_splits = _load_sample_splits(training_script)
    train_sample = dataset_splits.samples_by_split["train"][0]

    # Create the tiny CNN and keep a copy of one trainable weight matrix.
    model = training_script.create_tiny_galaxy_cnn()
    weight_before = model.classifier.weight.detach().clone()

    # Run one real training step using the known label_id from the manifest.
    result = training_script.run_torch_training_step(train_sample, model=model)

    # Verify the step used the known sample and target label.
    assert result.image_id == "gz2-000001"
    assert result.label == "spiral"
    assert result.label_id == 1
    assert result.input_shape == (1, 3, 3, 3)

    # Verify a real optimizer update happened.
    assert result.weight_changed is True
    assert not training_script.torch.equal(
        weight_before,
        model.classifier.weight.detach(),
    )

    # For this deterministic tiny sample, one update improves the correct class.
    assert result.loss_after < result.loss_before
    assert (
        result.probabilities_after[result.label_id]
        > result.probabilities_before[result.label_id]
    )


# Test that the tiny real PyTorch training loop runs across epochs.
def test_tiny_galaxy_cnn_training_loop_runs_for_epochs(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_training_loop",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny dataset splits.
    dataset_splits = _load_sample_splits(training_script)

    # Create the tiny CNN and keep a copy of one trainable weight matrix.
    model = training_script.create_tiny_galaxy_cnn()
    weight_before = model.classifier.weight.detach().clone()

    # Run two real PyTorch epochs over the one available train sample.
    summary = training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
    )

    # One usable train sample repeated for two epochs should produce two updates.
    assert summary.epochs == 2
    assert summary.train_sample_count == 1
    assert summary.val_sample_count == 0
    assert summary.test_sample_count == 0
    assert summary.skipped_row_count == 3
    assert summary.training_steps == 2
    assert len(summary.epoch_results) == 2
    assert summary.epoch_results[0].training_steps == 1
    assert summary.epoch_results[1].training_steps == 1

    # Verify real learning mechanics happened during the loop.
    assert summary.weight_changed is True
    assert not training_script.torch.equal(
        weight_before,
        model.classifier.weight.detach(),
    )

    # For this deterministic tiny sample, the loop improves the correct class.
    assert summary.first_loss is not None
    assert summary.final_loss is not None
    assert summary.final_loss < summary.first_loss
    assert summary.first_correct_probability is not None
    assert summary.final_correct_probability is not None
    assert summary.final_correct_probability > summary.first_correct_probability
    assert summary.final_predicted_label_id == 1


# Test that the real PyTorch training loop rejects invalid epoch counts.
def test_tiny_galaxy_cnn_training_loop_rejects_invalid_epochs(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_training_loop_bad_epochs",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny dataset splits.
    dataset_splits = _load_sample_splits(training_script)

    # Zero epochs should not be accepted.
    try:
        training_script.run_torch_training_loop(dataset_splits, epochs=0)
    except ValueError as error:
        assert "epochs must be at least 1" in str(error)
    else:
        raise AssertionError("Expected invalid epochs to raise ValueError")


# Test that evaluation reads predictions without changing model weights.
def test_tiny_galaxy_cnn_evaluation_is_read_only(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_evaluation",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny dataset splits.
    dataset_splits = _load_sample_splits(training_script)

    # Train one tiny model so evaluation has updated weights to inspect.
    model = training_script.create_tiny_galaxy_cnn()
    training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
    )

    # Keep a copy of one trainable weight matrix before read-only evaluation.
    weight_before_evaluation = model.classifier.weight.detach().clone()

    # Evaluate every split with the already-trained model.
    results = training_script.evaluate_torch_model(dataset_splits, model)

    # The current tiny dataset has one usable train sample and empty val/test splits.
    train_result = results["train"]
    assert train_result.sample_count == 1
    assert train_result.correct_predictions == 1
    assert train_result.accuracy == 1.0
    assert train_result.average_loss > 0.0
    assert train_result.first_true_label_id == 1
    assert train_result.first_predicted_label_id == 1
    assert train_result.first_correct_probability is not None

    # Empty evaluation splits should be explicit instead of pretending accuracy exists.
    assert results["val"].sample_count == 0
    assert results["val"].accuracy is None
    assert results["test"].sample_count == 0
    assert results["test"].accuracy is None

    # Evaluation must not update weights.
    assert training_script.torch.equal(
        weight_before_evaluation,
        model.classifier.weight.detach(),
    )


# Test that a trained tiny model can be saved and loaded into a fresh model.
def test_tiny_galaxy_cnn_checkpoint_round_trip(load_service_module, tmp_path):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_checkpoint",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny dataset splits.
    dataset_splits = _load_sample_splits(training_script)
    train_sample = dataset_splits.samples_by_split["train"][0]

    # Train one tiny model so the checkpoint contains updated weights.
    model = training_script.create_tiny_galaxy_cnn()
    training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
    )

    # Save into pytest's temporary folder instead of the repo artifact folder.
    checkpoint_path = tmp_path / "tiny_galaxy_cnn.pt"
    result = training_script.run_torch_checkpoint_round_trip(
        train_sample,
        model,
        checkpoint_path,
    )

    # Verify a real checkpoint file was written.
    assert result.checkpoint_path == checkpoint_path
    assert checkpoint_path.exists()

    # Verify the loaded model reproduces the trained model's prediction.
    assert result.image_id == "gz2-000001"
    assert result.saved_model_predicted_label_id == 1
    assert result.loaded_model_predicted_label_id == 1
    assert result.logits_match is True
    assert result.probabilities_match is True

    # Verify the lower-level loader can return a usable fresh model too.
    loaded_model = training_script.load_torch_checkpoint(checkpoint_path)
    loaded_result = training_script.run_torch_forward_pass(
        train_sample,
        model=loaded_model,
    )
    assert loaded_result.predicted_label_id == 1


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
