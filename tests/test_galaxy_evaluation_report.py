import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def load_training_script(load_service_module):
    return load_service_module(
        "galaxy_evaluation_report",
        "scripts/train_galaxy_cnn_baseline.py",
    )


def load_sample_splits(training_script):
    return training_script.load_dataset_splits_from_manifest(
        Path("data/samples/galaxy_manifest_sample.csv"),
        Path("data/samples/images"),
    )


def test_classification_metrics_include_confusion_and_bounded_errors(
    load_service_module,
):
    training_script = load_training_script(load_service_module)

    metrics = training_script.calculate_classification_metrics(
        true_label_ids=[0, 1, 2, 3],
        predicted_label_ids=[0, 0, 3, 3],
        image_ids=["a", "b", "c", "d"],
        image_paths=["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        probabilities=[
            [0.9, 0.05, 0.03, 0.02],
            [0.6, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.2, 0.6],
            [0.1, 0.1, 0.1, 0.7],
        ],
        max_error_examples=1,
    )

    assert metrics.confusion_matrix == [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    assert metrics.misclassified_count == 2
    assert len(metrics.misclassified_examples) == 1
    assert metrics.misclassified_examples[0].image_id == "b"
    assert metrics.misclassified_examples[0].true_label == "spiral"
    assert metrics.misclassified_examples[0].predicted_label == "elliptical"

    elliptical = metrics.class_metrics[0]
    assert elliptical.support == 1
    assert elliptical.predicted_count == 2
    assert elliptical.true_positives == 1
    assert elliptical.precision == 0.5
    assert elliptical.recall == 1.0
    assert elliptical.f1 == pytest.approx(2 / 3)


def test_majority_baseline_is_calculated_from_train_only(load_service_module):
    training_script = load_training_script(load_service_module)
    dataset_splits = load_sample_splits(training_script)

    results = training_script.evaluate_majority_baseline(dataset_splits)

    train_labels = [
        sample.label_id for sample in dataset_splits.samples_by_split["train"]
    ]
    counts = Counter(train_labels)
    expected_id = min(
        label_id for label_id, count in counts.items()
        if count == max(counts.values())
    )

    assert results is not None
    assert results["train"].majority_label_id == expected_id
    assert results["val"].majority_label_id == expected_id
    assert results["test"].majority_label_id == expected_id
    assert results["train"].sample_count == len(
        dataset_splits.samples_by_split["train"]
    )


def test_report_contains_configuration_dataset_and_metrics(
    load_service_module,
    tmp_path,
):
    training_script = load_training_script(load_service_module)
    dataset_splits = load_sample_splits(training_script)
    model = training_script.create_tiny_galaxy_cnn(seed=19)
    training_summary = training_script.run_torch_training_loop(
        dataset_splits,
        epochs=1,
        model=model,
        learning_rate=0.05,
        batch_size=2,
    )
    evaluation = training_script.evaluate_torch_model(
        dataset_splits,
        model,
        max_error_examples=3,
    )
    baseline = training_script.evaluate_majority_baseline(
        dataset_splits,
        max_error_examples=3,
    )
    args = SimpleNamespace(
        seed=19,
        epochs=1,
        batch_size=2,
        learning_rate=0.05,
        strict=True,
        image_width=None,
        image_height=None,
        max_error_examples=3,
    )

    report = training_script.build_evaluation_report(
        Path("data/samples/galaxy_manifest_sample.csv"),
        Path("data/samples/images"),
        args,
        dataset_splits,
        training_summary,
        evaluation,
        baseline,
        None,
    )
    output = tmp_path / "evaluation.json"
    training_script.write_evaluation_report(output, report)

    loaded = json.loads(output.read_text())
    assert loaded["configuration"]["seed"] == 19
    assert loaded["configuration"]["learning_rate"] == 0.05
    assert loaded["dataset"]["split_counts"] == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    assert loaded["evaluation"]["train"]["confusion_matrix"]
    assert loaded["evaluation"]["train"]["class_metrics"]
    assert loaded["majority_baseline"]["train"]["majority_label"]
    assert loaded["interpretation"].startswith("Bounded pilot report")


import pytest
