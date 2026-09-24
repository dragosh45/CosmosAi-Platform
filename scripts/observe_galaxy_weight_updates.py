#!/usr/bin/env python3
"""Read-only probes around the real tiny training loop, for learning (not serving)."""

# Docs: run_me_observe_results.md -> "25. Observe Exact CNN Weight Updates".
# Concepts: concepts_explanations.md -> "Our CNN Matrices And Exact Weight Updates".
import argparse
import csv
from dataclasses import asdict
from itertools import product
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as functional
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from cosmosai.galaxy.labels import LABEL_TO_ID
from cosmosai.galaxy.model import create_tiny_galaxy_cnn, galaxy_tensor_to_torch_image
from scripts.train_galaxy_cnn_baseline import load_dataset_splits_from_manifest, run_torch_training_loop


SELECTED_CONVOLUTION_WEIGHT = (0, 0, 1, 1)


def _selected_convolution_gradient_trace(
    images: torch.Tensor,
    convolution_gradient: torch.Tensor,
    selected_index: tuple[int, int, int, int],
    padding: int,
) -> dict:
    """Expand one shared kernel gradient into its image/position contributions."""
    output_filter, input_channel, kernel_row, kernel_column = selected_index
    padded_images = functional.pad(images, (padding, padding, padding, padding))
    contributions = []
    image_subtotals = []
    for image_index in range(images.shape[0]):
        image_total = 0.0
        for output_row in range(convolution_gradient.shape[2]):
            for output_column in range(convolution_gradient.shape[3]):
                aligned_input = padded_images[
                    image_index,
                    input_channel,
                    output_row + kernel_row,
                    output_column + kernel_column,
                ]
                d_loss_d_z = convolution_gradient[
                    image_index, output_filter, output_row, output_column
                ]
                product = d_loss_d_z * aligned_input
                image_total += float(product)
                contributions.append(
                    {
                        "image_index": image_index,
                        "output_row": output_row,
                        "output_column": output_column,
                        "aligned_input": float(aligned_input),
                        "dL_dz": float(d_loss_d_z),
                        "product": float(product),
                    }
                )
        image_subtotals.append(image_total)
    total = sum(image_subtotals)
    return {
        "parameter": "features.0.weight",
        "index": list(selected_index),
        "padding": padding,
        "equation": "dL/dw = sum(dL/dz * aligned_input)",
        "contributions": contributions,
        "image_subtotals": image_subtotals,
        "sum_of_contributions": total,
    }


def observe_training_updates(dataset_splits, epochs=2, batch_size=2, learning_rate=0.1) -> dict:
    """Trace the existing ordered CPU training loop without replacing its updates."""
    samples = dataset_splits.samples_by_split["train"]
    if not samples or epochs < 1 or batch_size < 1:
        raise ValueError("Observation requires train samples and positive epochs/batch size")
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and non-negative")
    batches_per_epoch = math.ceil(len(samples) / batch_size)
    # Full parameter copies are educational output, not a scalable training logger.
    if epochs * batches_per_epoch > 20:
        raise ValueError("This teaching trace is limited to 20 updates")

    model = create_tiny_galaxy_cnn()
    parameters = dict(model.named_parameters())
    owned_ids = {id(parameter) for parameter in parameters.values()}
    updates = []
    pending = {}

    def snapshot():
        # detach stops autograd bookkeeping; clone freezes the values at THIS moment.
        # Without clone, a later optimizer update could change our saved observation.
        return {name: parameter.detach().clone() for name, parameter in parameters.items()}

    def owns_optimizer(optimizer):
        return {id(p) for group in optimizer.param_groups for p in group["params"]} == owned_ids

    def observe_batch(epoch, batch_in_epoch, torch_batch):
        """Keep the actual DataLoader metadata for the next model forward."""
        pending.clear()
        pending.update(
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            images=torch_batch["image_tensor"].detach().clone(),
            labels=torch_batch["label_tensor"].detach().clone(),
            image_ids=list(torch_batch["image_id"]),
        )

    def before_forward(module, args):
        if not torch.is_grad_enabled():
            return  # The trainer also makes read-only starting/final predictions.
        if "images" not in pending:
            raise ValueError("Observer did not receive the actual DataLoader batch")
        # Fail rather than attaching the wrong metadata if the trainer changes.
        if not torch.equal(args[0], pending["images"]):
            raise ValueError("Observer batch metadata does not match model input")
        pending["before"] = snapshot()

    def capture_classifier(module, args, output):
        if torch.is_grad_enabled():
            # args[0] is A: four pooled features per image, NOT four raw pixels.
            pending["features"] = args[0].detach().clone()
            pending["logits"] = output.detach().clone()

    def capture_convolution(module, args, output):
        if torch.is_grad_enabled():
            pending["pre_relu_maps"] = output.detach().clone()
            # Register on the actual tensor so the callback receives dL/dz
            # without a module backward-hook warning for non-grad inputs.
            output.register_hook(
                lambda gradient: pending.__setitem__(
                    "dL_d_conv_output", gradient.detach().clone()
                )
            )

    def capture_relu(module, args, output):
        if torch.is_grad_enabled():
            pending["post_relu_maps"] = output.detach().clone()

    def before_step(optimizer, args, kwargs):
        if not owns_optimizer(optimizer):
            return
        if not isinstance(optimizer, torch.optim.SGD) or any(
            group["momentum"] != 0 or group["weight_decay"] != 0 or group["maximize"]
            for group in optimizer.param_groups
        ):
            raise ValueError("Observer's update equation requires plain SGD")
        # backward() has finished, but step() has not run: values must still match
        # the snapshot before the forward pass. The new information lives in .grad.
        pending["unchanged_after_backward"] = all(
            torch.equal(parameters[name].detach(), value)
            for name, value in pending["before"].items()
        )
        pending["gradients"] = {
            name: parameter.grad.detach().clone() for name, parameter in parameters.items()
        }

    def after_step(optimizer, args, kwargs):
        if not owns_optimizer(optimizer):
            return
        after = snapshot()
        # Revisit the SAME batch with updated weights. This extra forward pass is
        # read-only: it cannot produce a second update or accumulate gradients.
        with torch.no_grad():
            conv_after = model.features[0](pending["images"])
            relu_after = model.features[1](conv_after)
            features_after = model.features[2:](relu_after)
            logits_after = model.classifier(features_after)
            loss_before = torch.nn.functional.cross_entropy(pending["logits"], pending["labels"])
            loss_after = torch.nn.functional.cross_entropy(logits_after, pending["labels"])
        probabilities_before = torch.softmax(pending["logits"], dim=1)
        one_hot_labels = torch.nn.functional.one_hot(
            pending["labels"], num_classes=logits_after.shape[1]
        ).to(dtype=logits_after.dtype)
        dL_d_logits = (probabilities_before - one_hot_labels) / pending["labels"].shape[0]
        classifier_weight_before = pending["before"]["classifier.weight"]
        dL_d_features = dL_d_logits @ classifier_weight_before
        selected_gradient = pending["gradients"]["features.0.weight"][
            SELECTED_CONVOLUTION_WEIGHT
        ]
        selected_trace = _selected_convolution_gradient_trace(
            pending["images"],
            pending["dL_d_conv_output"],
            SELECTED_CONVOLUTION_WEIGHT,
            padding=int(model.features[0].padding[0]),
        )
        selected_trace["autograd_gradient"] = float(selected_gradient)
        selected_trace["matches_autograd"] = bool(
            torch.allclose(
                torch.tensor(selected_trace["sum_of_contributions"]),
                selected_gradient,
                atol=1e-7,
                rtol=1e-6,
            )
        )
        parameter_rows = {}
        for name, before in pending["before"].items():
            gradient = pending["gradients"][name]
            # Tiny float32 rounding can differ between fused SGD arithmetic and
            # separately subtracting lr*grad, so compare with a small tolerance.
            predicted_after = before - learning_rate * gradient
            parameter_rows[name] = {
                "shape": list(before.shape), "before": before.tolist(),
                "gradient": gradient.tolist(), "after": after[name].tolist(),
                "delta": (after[name] - before).tolist(),
                "changed_count": int(torch.count_nonzero(after[name] != before)),
                "sgd_equation_matches": bool(torch.allclose(after[name], predicted_after, atol=1e-7, rtol=1e-6)),
            }
        updates.append({
            "update": len(updates) + 1, "epoch": pending["epoch"],
            "batch_in_epoch": pending["batch_in_epoch"],
            "image_ids": pending["image_ids"], "labels": pending["labels"].tolist(),
            "input_shape": list(pending["images"].shape), "images": pending["images"].tolist(),
            "pre_relu_maps_before": pending["pre_relu_maps"].tolist(),
            "post_relu_maps_before": pending["post_relu_maps"].tolist(),
            "dL_d_logits": dL_d_logits.tolist(),
            "dL_d_pooled_features": dL_d_features.tolist(),
            "dL_d_conv_output": pending["dL_d_conv_output"].tolist(),
            "selected_convolution_gradient": selected_trace,
            "features_before": pending["features"].tolist(), "features_after": features_after.tolist(),
            "pre_relu_maps_after": conv_after.tolist(),
            "post_relu_maps_after": relu_after.tolist(),
            "logits_before": pending["logits"].tolist(), "logits_after": logits_after.tolist(),
            "probabilities_before": torch.softmax(pending["logits"], dim=1).tolist(),
            "probabilities_after": torch.softmax(logits_after, dim=1).tolist(),
            "loss_before": float(loss_before), "loss_after_same_batch": float(loss_after),
            "weights_unchanged_after_backward": pending["unchanged_after_backward"],
            "parameters": parameter_rows,
        })

    # Optimizer hooks are temporarily process-wide because the existing trainer
    # creates its optimizer internally. Filter to THIS model, and always remove
    # every hook, including after an error. This is a single-threaded teaching CLI.
    handles = [model.register_forward_pre_hook(before_forward),
               model.features[0].register_forward_hook(capture_convolution),
               model.features[1].register_forward_hook(capture_relu),
               model.classifier.register_forward_hook(capture_classifier),
               register_optimizer_step_pre_hook(before_step),
               register_optimizer_step_post_hook(after_step)]
    try:
        summary = run_torch_training_loop(
            dataset_splits,
            epochs=epochs,
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            on_training_batch=observe_batch,
        )
    finally:
        for handle in handles:
            handle.remove()

    return {
        "format_version": 2, "torch_version": str(torch.__version__), "seed": 7,
        "device": "cpu", "dtype": "float32", "learning_rate": learning_rate,
        "batch_size": batch_size, "epochs": epochs, "shuffle": False,
        "label_to_id": LABEL_TO_ID, "parameter_count": sum(p.numel() for p in parameters.values()),
        "summary": asdict(summary), "updates": updates,
    }


def parameter_entries(update):
    """One row per scalar, including zero gradients and unchanged weights/biases."""
    for name, values in update["parameters"].items():
        for index in product(*(range(size) for size in values["shape"])):
            numbers = {}
            for key in ("before", "gradient", "after", "delta"):
                value = values[key]
                for position in index:
                    value = value[position]
                numbers[key] = value
            yield {"update": update["update"], "epoch": update["epoch"], "parameter": name,
                   "index": ",".join(map(str, index)), **numbers}


def write_observations(report: dict, output_path: Path) -> None:
    if output_path.suffix.lower() != ".json":
        raise ValueError("Use a .json output path; a separate .csv file is also written")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["update", "epoch", "parameter", "index",
                                                    "before", "gradient", "after", "delta"])
        writer.writeheader()
        for update in report["updates"]:
            writer.writerows(parameter_entries(update))


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe exact tiny CNN parameter updates (no checkpoint overwrite).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/cosmosai-weight-observation/trace.json"))
    args = parser.parse_args()
    try:
        splits = load_dataset_splits_from_manifest(REPO_ROOT / "data/samples/galaxy_manifest_sample.csv",
                                                   REPO_ROOT / "data/samples/images")
        report = observe_training_updates(splits, epochs=args.epochs, batch_size=args.batch_size)
        write_observations(report, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        print(error)
        return 1
    print(f"Observed {report['parameter_count']} trainable scalars in the real training loop")
    for update in report["updates"]:
        changed = sum(p["changed_count"] for p in update["parameters"].values())
        print(f"Update {update['update']} (epoch {update['epoch']}): "
              f"loss {update['loss_before']:.6f} -> {update['loss_after_same_batch']:.6f}; "
              f"changed {changed}/{report['parameter_count']} parameters")
        print(f"  weights unchanged after backward: {update['weights_unchanged_after_backward']}")
    print(f"Full values/matrices: {args.output}")
    print(f"One scalar per row: {args.output.with_suffix('.csv')}")
    print("Synthetic fixtures only; no checkpoint saved or overwritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
