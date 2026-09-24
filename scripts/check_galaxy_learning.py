#!/usr/bin/env python3
"""Diagnose learning on the existing pilot without changing the serving model.

First test whether a small spatial CNN can memorize two TRAIN images per class.
Then try Adam with the existing checkpoint-compatible TinyGalaxyCNN on the full
training split. Only that second trial uses validation, once at the end. Test
pixels are never loaded. All outputs go to a new, separate experiment directory.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmosai.galaxy.dataset_splits import create_dataset_splits
from cosmosai.galaxy.labels import LABEL_TO_ID
from cosmosai.galaxy.manifest import load_manifest
from cosmosai.galaxy.model import create_tiny_galaxy_cnn, save_torch_checkpoint
from cosmosai.galaxy.preprocessing import GalaxyPreprocessingPolicy
from cosmosai.galaxy.torch_dataset import create_galaxy_dataloader
from train_galaxy_cnn_baseline import evaluate_torch_split


def select_training_subset(records, per_class=2):
    """Choose a deterministic balanced subset, never using validation/test rows."""
    counts = Counter()
    selected = []
    for record in sorted(records, key=lambda row: row.image_id):
        if record.split == "train" and counts[record.label] < per_class:
            selected.append(record)
            counts[record.label] += 1
    if any(counts[label] != per_class for label in LABEL_TO_ID):
        raise ValueError(f"Need at least {per_class} training images per class")
    return selected


def create_sanity_model():
    """Keep spatial features for a memorization check, not a deployable checkpoint."""
    torch.manual_seed(7)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((4, 4)), torch.nn.Flatten(),
        torch.nn.Linear(16 * 4 * 4, len(LABEL_TO_ID)),
    )


def train_adam(model, samples, epochs, batch_size, *, shuffle):
    """Run bounded Adam updates through the shared image Dataset/DataLoader."""
    torch.manual_seed(7)
    loader = create_galaxy_dataloader(samples, batch_size=batch_size, shuffle=shuffle)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch["image_tensor"]), batch["label_tensor"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; stopping this candidate")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["label_tensor"])
        losses.append(total_loss / len(samples))
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1}/{epochs}: loss {losses[-1]:.6f}", flush=True)
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sanity-steps", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve earlier checkpoints")
    if not (1 <= args.epochs <= 60 and 1 <= args.sanity_steps <= 500
            and 8 <= args.image_size <= 128 and 1 <= args.threads <= 4):
        parser.error("Bounds: epochs 1..60, sanity-steps 1..500, image-size 8..128, threads 1..4")
    torch.set_num_threads(args.threads)
    records = load_manifest(args.manifest)
    try:
        subset_records = select_training_subset(records)
    except ValueError as error:
        parser.error(str(error))
    policy = GalaxyPreprocessingPolicy((args.image_size, args.image_size))
    usable = [row for row in records if row.split != "test"]
    # Eager loading is bounded here to this small diagnostic; normal training
    # remains lazy. Avoid repeated JPEG decoding during the memorization check.
    if len(usable) > 100:
        parser.error("This check is limited to 100 train/validation images")
    splits = create_dataset_splits(usable, args.data_root, skip_missing=False,
                                  target_size=policy.target_size, lazy=False)
    train = splits.samples_by_split["train"]
    val = splits.samples_by_split["val"]
    if not val:
        parser.error("Validation records are required")
    subset_ids = {row.image_id for row in subset_records}
    subset = [sample for sample in train if sample.image_id in subset_ids]

    print("Training-only memorization check: spatial CNN, 8 images", flush=True)
    sanity_model = create_sanity_model()
    sanity_losses = train_adam(sanity_model, subset, args.sanity_steps, len(subset), shuffle=False)
    sanity_result = evaluate_torch_split("train_subset", subset, sanity_model)
    print(f"  subset accuracy: {sanity_result.accuracy:.4f}", flush=True)

    print("Compatible candidate: original tiny CNN, Adam, shuffled training batches", flush=True)
    model = create_tiny_galaxy_cnn(seed=7)
    losses = train_adam(model, train, args.epochs, 8, shuffle=True)
    evaluation = {name: evaluate_torch_split(name, samples, model)
                  for name, samples in (("train", train), ("val", val))}
    counts = Counter(sample.label_id for sample in train)
    majority = min(counts, key=lambda label: (-counts[label], label))
    baseline = sum(sample.label_id == majority for sample in val) / len(val)

    # No artifacts from the spatial diagnostic enter the production checkpoint
    # loader, which currently understands only TinyGalaxyCNN.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = save_torch_checkpoint(model, args.output_dir / "candidate.pt", policy)
    report = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "configuration": {"seed": 7, "image_size": args.image_size, "optimizer": "Adam",
                          "learning_rate": 0.01, "epochs": args.epochs, "batch_size": 8,
                          "shuffle": True, "threads": args.threads},
        "class_counts": {name: dict(Counter(s.label for s in samples))
                         for name, samples in (("train", train), ("val", val))},
        "sanity_check": {"training_image_ids": [s.image_id for s in subset],
                         "parameter_count": sum(p.numel() for p in sanity_model.parameters()),
                         "steps": args.sanity_steps, "epoch_losses": sanity_losses,
                         "evaluation": asdict(sanity_result), "validation_evaluated": False,
                         "meaning": "Training memorization only, not generalization."},
        "epoch_losses": losses,
        "evaluation": {name: asdict(result) for name, result in evaluation.items()},
        "validation_majority_accuracy": baseline,
        "test_records_excluded": sum(row.split == "test" for row in records),
        "test_evaluated": False,
        "serving_checkpoint_changed": False,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    card = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "title": "GZ2 pilot Adam diagnostic candidate", "training_images": len(train),
        "epochs": args.epochs,
        "validation": {"sample_count": len(val), "accuracy": evaluation["val"].accuracy},
        "test": {"sample_count": report["test_records_excluded"], "accuracy": None},
        "notes": ["Small validation trial, not established accuracy; no automatic promotion.",
                  "Optimizer, learning rate and batch order differ from the SGD candidate.",
                  "Lenticular is a derived GZ2 label proxy. Test images were not evaluated."],
    }
    (args.output_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps({"sanity_train_accuracy": sanity_result.accuracy,
                      "tiny_train_accuracy": evaluation["train"].accuracy,
                      "tiny_validation_accuracy": evaluation["val"].accuracy,
                      "validation_majority_accuracy": baseline,
                      "serving_checkpoint_changed": False, "test_evaluated": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
