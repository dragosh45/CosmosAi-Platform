#!/usr/bin/env python3
"""Reproduce the default one-neuron example without ML or external services."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def calculate(inputs, weights, bias):
    if len(inputs) != len(weights) or not inputs:
        raise ValueError("Use the same nonzero number of inputs and weights")
    products = [x * w for x, w in zip(inputs, weights)]
    z = sum(products) + bias
    return {"inputs": inputs, "weights": weights, "bias": bias,
            "products": products, "weighted_sum": z, "relu": max(0.0, z)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "apps/learning-examples/weighted-sum/trace.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calculate([2.0, -1.0, 0.5], [0.4, 0.2, -0.6], 0.1), indent=2) + "\n")
    print(args.output)
