# Import Path to pass local sample paths into the training skeleton.
from pathlib import Path

# Import SimpleNamespace to test argparse-like values without parsing sys.argv.
from types import SimpleNamespace

import pytest


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

    # Verify the current tiny dataset shape.
    assert summary.epochs == 1
    assert summary.train_sample_count == 2
    assert summary.val_sample_count == 1
    assert summary.test_sample_count == 1
    assert summary.skipped_row_count == 0
    assert summary.training_steps == 2
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


# Test that multiple samples can be stacked into one PyTorch image batch.
def test_galaxy_samples_to_torch_batch_shape_and_labels(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_batch",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the two current train samples.
    dataset_splits = _load_sample_splits(training_script)
    train_samples = dataset_splits.samples_by_split["train"]

    # Stack both train samples into one batch.
    batch = training_script.galaxy_samples_to_torch_batch(train_samples)

    # Two tiny RGB 3x3 samples should become one NCHW tensor.
    assert tuple(batch.image_tensor.shape) == (2, 3, 3, 3)

    # The label tensor has one correct class ID per image in the same order.
    assert tuple(batch.label_tensor.shape) == (2,)
    assert batch.image_ids == ["gz2-000001", "gz2-000004"]
    assert batch.labels == ["spiral", "lenticular"]
    assert batch.label_tensor.tolist() == [1, 2]


# Test that batches fail clearly when image shapes do not match.
def test_galaxy_samples_to_torch_batch_rejects_mismatched_shapes(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_batch_mismatched_shapes",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load one real train sample.
    dataset_splits = _load_sample_splits(training_script)
    train_sample = dataset_splits.samples_by_split["train"][0]

    # Build a second sample with a different image shape to simulate unresized data.
    smaller_tensor = train_sample.tensor.__class__(
        path=train_sample.tensor.path,
        shape=(1, 1, 3),
        values=[0.0, 0.0, 0.0],
    )
    mismatched_sample = train_sample.__class__(
        image_id="different-shape",
        label=train_sample.label,
        label_id=train_sample.label_id,
        split=train_sample.split,
        tensor=smaller_tensor,
    )

    # The helper should explain the fix instead of failing inside torch.cat.
    try:
        training_script.galaxy_samples_to_torch_batch(
            [train_sample, mismatched_sample]
        )
    except ValueError as error:
        assert "use target_size" in str(error)
    else:
        raise AssertionError("Expected mismatched batch shapes to raise ValueError")


# Test that the batch proof exposes tensor and label shapes for terminal output.
def test_torch_batch_shape_proof_reports_batch_details(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_batch_shape_proof",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the two current train samples.
    dataset_splits = _load_sample_splits(training_script)
    train_samples = dataset_splits.samples_by_split["train"]

    # Build a proof result using both train samples in one batch.
    result = training_script.run_torch_batch_shape_proof(
        train_samples,
        batch_size=2,
    )

    # Verify the proof result is readable and matches the PyTorch tensors.
    assert result is not None
    assert result.requested_batch_size == 2
    assert result.actual_batch_size == 2
    assert result.image_ids == ["gz2-000001", "gz2-000004"]
    assert result.labels == ["spiral", "lenticular"]
    assert result.label_ids == [1, 2]
    assert result.image_tensor_shape == (2, 3, 3, 3)
    assert result.label_tensor_shape == (2,)


# Test that resize CLI values are accepted only when width and height are both set.
def test_target_size_from_args_requires_width_and_height(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_target_size_args",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Missing height should fail before any image loading starts.
    args = SimpleNamespace(image_width=224, image_height=None)

    try:
        training_script.target_size_from_args(args)
    except ValueError as error:
        assert "--image-width and --image-height must be used together" in str(error)
    else:
        raise AssertionError("Expected partial target size to raise ValueError")

    # When both dimensions are present, Pillow target size is width, height.
    args = SimpleNamespace(image_width=224, image_height=224)
    assert training_script.target_size_from_args(args) == (224, 224)


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

    # Load the first usable train sample for one isolated weight update.
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

    # Run two real PyTorch epochs over the available train samples.
    summary = training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
    )

    # Two train samples repeated for two epochs should produce four updates.
    assert summary.epochs == 2
    assert summary.train_sample_count == 2
    assert summary.val_sample_count == 1
    assert summary.test_sample_count == 1
    assert summary.skipped_row_count == 0
    assert summary.training_steps == 4
    assert len(summary.epoch_results) == 2
    assert summary.epoch_results[0].training_steps == 2
    assert summary.epoch_results[1].training_steps == 2

    # Verify real learning mechanics happened during the loop.
    assert summary.weight_changed is True
    assert not training_script.torch.equal(
        weight_before,
        model.classifier.weight.detach(),
    )

    # The loop records before/after metrics for the first train sample.
    assert summary.first_loss is not None
    assert summary.final_loss is not None
    assert summary.first_correct_probability is not None
    assert summary.final_correct_probability is not None
    assert 0 <= summary.final_predicted_label_id < len(training_script.LABEL_TO_ID)


# Test that batch size controls how many optimizer steps happen in the loop.
def test_tiny_galaxy_cnn_training_loop_uses_batch_size(load_service_module):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_training_loop_batches",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny dataset splits.
    dataset_splits = _load_sample_splits(training_script)

    # Train with two train samples per batch.
    model = training_script.create_tiny_galaxy_cnn()
    summary = training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
        batch_size=2,
    )

    # Two train samples become one batch per epoch, so two epochs make two updates.
    assert summary.batch_size == 2
    assert summary.train_batch_count == 1
    assert summary.training_steps == 2
    assert summary.epoch_results[0].training_steps == 1
    assert summary.epoch_results[1].training_steps == 1
    assert summary.weight_changed is True


def test_training_reads_only_train_items_again_each_epoch(
    load_service_module, monkeypatch
):
    from cosmosai.galaxy.torch_dataset import GalaxyTorchDataset

    script = load_service_module(
        "loader_training_visits", "scripts/train_galaxy_cnn_baseline.py"
    )
    splits = _load_sample_splits(script)
    visits = []
    original_getitem = GalaxyTorchDataset.__getitem__

    def record_visit(dataset, index):
        item = original_getitem(dataset, index)
        visits.append(item["image_id"])
        return item

    def reject_manual_batching(*args, **kwargs):
        raise AssertionError("Real training must use DataLoader, not manual batches")

    monkeypatch.setattr(GalaxyTorchDataset, "__getitem__", record_visit)
    monkeypatch.setattr(script, "create_sample_batches", reject_manual_batching)
    monkeypatch.setattr(script, "galaxy_samples_to_torch_batch", reject_manual_batching)
    summary = script.run_torch_training_loop(splits, epochs=2, batch_size=2)

    assert visits == ["gz2-000001", "gz2-000004"] * 2
    assert summary.training_steps == 2
    assert summary.weight_changed is True


def test_dataloader_training_matches_manual_sgd_updates(load_service_module):
    script = load_service_module(
        "loader_training_parity", "scripts/train_galaxy_cnn_baseline.py"
    )
    splits = _load_sample_splits(script)
    # An odd sample count exercises the smaller final batch as well as full batches.
    splits.samples_by_split["train"].append(splits.samples_by_split["train"][0])
    loader_model = script.create_tiny_galaxy_cnn()
    manual_model = script.create_tiny_galaxy_cnn()
    optimizer = script.torch.optim.SGD(manual_model.parameters(), lr=0.1)
    loss_function = script.torch.nn.CrossEntropyLoss()

    for _ in range(2):
        for samples in script.create_sample_batches(splits.samples_by_split["train"], 2):
            batch = script.galaxy_samples_to_torch_batch(samples)
            optimizer.zero_grad()
            loss = loss_function(manual_model(batch.image_tensor), batch.label_tensor)
            loss.backward()
            optimizer.step()

    summary = script.run_torch_training_loop(
        splits, epochs=2, batch_size=2, model=loader_model
    )
    assert summary.training_steps == 4
    for name, weight in manual_model.state_dict().items():
        script.torch.testing.assert_close(loader_model.state_dict()[name], weight)


def test_training_loss_weights_uneven_batches_by_image_count(load_service_module):
    script = load_service_module(
        "loader_training_loss", "scripts/train_galaxy_cnn_baseline.py"
    )
    splits = _load_sample_splits(script)
    samples = splits.samples_by_split["train"]
    samples.append(samples[0])
    model = script.create_tiny_galaxy_cnn()
    expected = script.evaluate_torch_split("train", samples, model).average_loss

    # lr=0 freezes weights: differences in averages can only come from accounting.
    summary = script.run_torch_training_loop(
        splits, epochs=2, batch_size=2, learning_rate=0.0, model=model
    )
    assert summary.train_sample_count == 3
    assert summary.train_batch_count == 2
    assert summary.training_steps == 4
    assert summary.average_loss == pytest.approx(expected, abs=1e-6)
    assert summary.weight_changed is False
    for epoch in summary.epoch_results:
        assert epoch.training_steps == 2
        assert epoch.average_loss == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("train_count, expected_steps", [(0, 0), (2, 2)])
def test_training_handles_empty_or_smaller_than_requested_batch(
    load_service_module, train_count, expected_steps
):
    script = load_service_module(
        "loader_training_small_split", "scripts/train_galaxy_cnn_baseline.py"
    )
    splits = _load_sample_splits(script)
    splits.samples_by_split["train"] = splits.samples_by_split["train"][:train_count]
    summary = script.run_torch_training_loop(splits, epochs=2, batch_size=8)
    assert summary.training_steps == expected_steps
    assert summary.train_batch_count == (1 if train_count else 0)
    if not train_count:
        assert summary.average_loss == 0.0
        assert summary.first_loss is None
        assert summary.final_loss is None
        assert summary.weight_changed is False


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


# Test that invalid batch sizes fail clearly.
def test_tiny_galaxy_cnn_training_loop_rejects_invalid_batch_size(
    load_service_module,
):
    # Load the training skeleton under test.
    training_script = load_service_module(
        "galaxy_cnn_baseline_torch_training_loop_bad_batch",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny dataset splits.
    dataset_splits = _load_sample_splits(training_script)

    # Zero batch size should not be accepted.
    try:
        training_script.run_torch_training_loop(
            dataset_splits,
            epochs=1,
            batch_size=0,
        )
    except ValueError as error:
        assert "batch_size must be at least 1" in str(error)
    else:
        raise AssertionError("Expected invalid batch size to raise ValueError")


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

    # The current tiny dataset has usable train, val, and test samples.
    train_result = results["train"]
    val_result = results["val"]
    test_result = results["test"]

    # Evaluation should now exercise every split instead of empty val/test buckets.
    assert train_result.sample_count == 2
    assert val_result.sample_count == 1
    assert test_result.sample_count == 1

    # Accuracy exists for non-empty splits and must stay inside probability bounds.
    for split_result in (train_result, val_result, test_result):
        assert split_result.average_loss > 0.0
        assert split_result.accuracy is not None
        assert 0.0 <= split_result.accuracy <= 1.0
        assert split_result.first_true_label_id is not None
        assert split_result.first_predicted_label_id is not None
        assert split_result.first_correct_probability is not None

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

    # Two train samples repeated for two epochs should produce four steps.
    assert summary.epochs == 2
    assert summary.train_sample_count == 2
    assert summary.val_sample_count == 1
    assert summary.test_sample_count == 1
    assert summary.skipped_row_count == 0
    assert summary.training_steps == 4
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
