import csv
import json
from pathlib import Path

import pytest
import torch
from torch.optim import optimizer as optimizer_module

from scripts import observe_galaxy_weight_updates as observation
from scripts.train_galaxy_cnn_baseline import load_dataset_splits_from_manifest, run_torch_training_loop
from cosmosai.galaxy.model import create_tiny_galaxy_cnn


def sample_splits():
    return load_dataset_splits_from_manifest(Path("data/samples/galaxy_manifest_sample.csv"),
                                           Path("data/samples/images"))


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_observer_does_not_change_training_and_records_all_parameters(batch_size):
    splits = sample_splits()
    report = observation.observe_training_updates(splits, epochs=2, batch_size=batch_size)
    plain_model = create_tiny_galaxy_cnn()
    summary = run_torch_training_loop(splits, epochs=2, batch_size=batch_size, model=plain_model)
    assert report["parameter_count"] == 132
    assert len(report["updates"]) == summary.training_steps
    assert report["summary"]["average_loss"] == summary.average_loss
    for name, parameter in plain_model.named_parameters():
        assert torch.equal(torch.tensor(report["updates"][-1]["parameters"][name]["after"]), parameter)
    for index, step in enumerate(report["updates"]):
        assert step["weights_unchanged_after_backward"]
        assert len(list(observation.parameter_entries(step))) == 132
        for name, values in step["parameters"].items():
            assert values["sgd_equation_matches"]
            assert torch.allclose(torch.tensor(values["delta"]),
                                  -0.1 * torch.tensor(values["gradient"]), atol=1e-7)
            if index:
                assert values["before"] == report["updates"][index - 1]["parameters"][name]["after"]


def test_observed_classifier_matrix_and_closed_form_gradients():
    report = observation.observe_training_updates(sample_splits())
    for step in report["updates"]:
        features = torch.tensor(step["features_before"])
        weights = torch.tensor(step["parameters"]["classifier.weight"]["before"])
        biases = torch.tensor(step["parameters"]["classifier.bias"]["before"])
        logits = torch.tensor(step["logits_before"])
        assert torch.allclose(features @ weights.T + biases, logits, atol=1e-7)
        labels = torch.tensor(step["labels"])
        # For mean cross entropy, each class error is (probability - target)/N.
        error = (torch.softmax(logits, dim=1) - torch.nn.functional.one_hot(labels, 4)) / len(labels)
        assert torch.allclose(error.T @ features,
                              torch.tensor(step["parameters"]["classifier.weight"]["gradient"]), atol=1e-7)
        assert torch.allclose(error.sum(dim=0),
                              torch.tensor(step["parameters"]["classifier.bias"]["gradient"]), atol=1e-7)
        after_features = torch.tensor(step["features_after"])
        after_weights = torch.tensor(step["parameters"]["classifier.weight"]["after"])
        after_biases = torch.tensor(step["parameters"]["classifier.bias"]["after"])
        assert torch.allclose(after_features @ after_weights.T + after_biases,
                              torch.tensor(step["logits_after"]), atol=1e-7)


def test_observation_includes_cnn_maps_and_expanded_shared_gradient():
    report = observation.observe_training_updates(sample_splits())
    first = report["updates"][0]
    assert report["format_version"] == 2
    assert first["batch_in_epoch"] == 1
    assert torch.tensor(first["pre_relu_maps_before"]).shape == (2, 4, 3, 3)
    assert torch.tensor(first["post_relu_maps_before"]).shape == (2, 4, 3, 3)
    assert torch.tensor(first["dL_d_logits"]).shape == (2, 4)
    assert torch.tensor(first["dL_d_pooled_features"]).shape == (2, 4)
    assert torch.tensor(first["dL_d_conv_output"]).shape == (2, 4, 3, 3)

    trace = first["selected_convolution_gradient"]
    assert trace["index"] == [0, 0, 1, 1]
    assert len(trace["contributions"]) == 18  # two images x nine map positions
    assert trace["matches_autograd"]
    assert trace["sum_of_contributions"] == pytest.approx(0.014876489, abs=1e-7)
    assert trace["autograd_gradient"] == pytest.approx(0.014876489, abs=1e-7)
    assert first["pre_relu_maps_before"][0][0][1][1] == pytest.approx(0.037116095, abs=1e-7)
    assert first["features_before"][0][0] == pytest.approx(0.020664100, abs=1e-7)


def test_observation_uses_actual_batch_metadata_for_partial_batches():
    report = observation.observe_training_updates(sample_splits(), epochs=1, batch_size=1)
    assert [step["image_ids"] for step in report["updates"]] == [
        ["gz2-000001"],
        ["gz2-000004"],
    ]
    assert [step["batch_in_epoch"] for step in report["updates"]] == [1, 2]


def test_trace_exports_full_values_not_only_changed_rows(tmp_path):
    report = observation.observe_training_updates(sample_splits())
    path = tmp_path / "trace.json"
    observation.write_observations(report, path)
    assert json.loads(path.read_text()) == report
    with path.with_suffix(".csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 264  # 132 parameters at EACH of two actual optimizer steps.
    assert any(float(row["gradient"]) == 0 and float(row["delta"]) == 0 for row in rows)
    with pytest.raises(ValueError, match=".json"):
        observation.write_observations(report, tmp_path / "wrong.csv")


def test_observer_removes_global_hooks_after_failure(monkeypatch):
    before_pre = dict(optimizer_module._global_optimizer_pre_hooks)
    before_post = dict(optimizer_module._global_optimizer_post_hooks)
    def fail(*args, **kwargs):
        raise RuntimeError("test interruption")
    monkeypatch.setattr(observation, "run_torch_training_loop", fail)
    with pytest.raises(RuntimeError, match="test interruption"):
        observation.observe_training_updates(sample_splits())
    assert dict(optimizer_module._global_optimizer_pre_hooks) == before_pre
    assert dict(optimizer_module._global_optimizer_post_hooks) == before_post


def test_observer_rejects_empty_or_large_teaching_runs():
    splits = sample_splits()
    with pytest.raises(ValueError, match="20 updates"):
        observation.observe_training_updates(splits, epochs=21)
    splits.samples_by_split["train"].clear()
    with pytest.raises(ValueError, match="requires train samples"):
        observation.observe_training_updates(splits)
