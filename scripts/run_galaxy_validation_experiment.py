#!/usr/bin/env python3
"""Run one bounded training candidate and evaluate train/validation only.

Reuse the existing trainer and checkpoint contract. Test records are excluded
before image loading; candidate files never overwrite the serving checkpoint.
Docs: docs/local_classifier_demo.md.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmosai.galaxy.dataset_splits import create_dataset_splits
from cosmosai.galaxy.manifest import load_manifest
from cosmosai.galaxy.model import create_tiny_galaxy_cnn, save_torch_checkpoint
from cosmosai.galaxy.preprocessing import GalaxyPreprocessingPolicy
from train_galaxy_cnn_baseline import evaluate_torch_split, run_torch_training_loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve earlier checkpoints")
    if args.epochs < 1 or args.image_size < 1 or not 0 < args.learning_rate <= 1:
        parser.error("Use positive epochs/image-size and a learning rate in (0, 1]")
    records = load_manifest(args.manifest)
    experiment_records = [record for record in records if record.split != "test"]
    policy = GalaxyPreprocessingPolicy((args.image_size, args.image_size))
    splits = create_dataset_splits(experiment_records, args.data_root,
                                  skip_missing=False, target_size=policy.target_size)
    if not splits.samples_by_split["train"] or not splits.samples_by_split["val"]:
        parser.error("Both training and validation records are required")
    model = create_tiny_galaxy_cnn(seed=7)

    def progress(epoch, batch, _data):
        if batch == 1:
            print(f"Epoch {epoch}/{args.epochs}", flush=True)

    training = run_torch_training_loop(
        splits, epochs=args.epochs, model=model, learning_rate=args.learning_rate,
        batch_size=8, on_training_batch=progress,
    )
    evaluation = {split: evaluate_torch_split(split, splits.samples_by_split[split], model)
                  for split in ("train", "val")}
    checkpoint = save_torch_checkpoint(model, args.output_dir / "candidate.pt", policy)
    counts = Counter(sample.label_id for sample in splits.samples_by_split["train"])
    majority = min(counts, key=lambda label: (-counts[label], label))
    val_samples = splits.samples_by_split["val"]
    baseline_accuracy = sum(sample.label_id == majority for sample in val_samples) / len(val_samples)
    report = {
        "configuration": {"epochs": args.epochs, "image_size": args.image_size,
                          "learning_rate": args.learning_rate, "seed": 7, "batch_size": 8},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "training": asdict(training),
        "evaluation": {split: asdict(metrics) for split, metrics in evaluation.items()},
        "validation_majority_accuracy": baseline_accuracy,
        "test_evaluated": False,
        "test_records_excluded": len(records) - len(experiment_records),
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    card = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "title": "GZ2 pilot validation candidate", "training_images": len(splits.samples_by_split["train"]),
        "epochs": args.epochs,
        "validation": {"sample_count": len(val_samples), "accuracy": evaluation["val"].accuracy},
        "test": {"sample_count": len(records) - len(experiment_records), "accuracy": None},
        "notes": ["Only 12 validation images; this result does not establish reliable accuracy.",
                  "Lenticular is a derived GZ2 label proxy.", "Test images were not evaluated in this experiment."],
    }
    (args.output_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint), "train_accuracy": evaluation["train"].accuracy,
                      "validation_accuracy": evaluation["val"].accuracy,
                      "majority_accuracy": baseline_accuracy, "test_evaluated": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
