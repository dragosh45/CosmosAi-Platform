#!/usr/bin/env python3
"""Small arithmetic examples; not a replacement for the MATLAB lab run."""
import json
import math
from pathlib import Path
from statistics import mean, stdev


def interpolate(left, right, missing):
    if not isinstance(missing, int) or not 1 <= missing <= 20:
        raise ValueError('Use between 1 and 20 missing readings')
    if not all(math.isfinite(x) for x in [left, right]):
        raise ValueError('Endpoints must be finite')
    return [left + (right-left)*k/(missing+1) for k in range(1, missing+1)]


def normalize(training, readings):
    if len(training) < 2 or not all(math.isfinite(x) for x in training + readings):
        raise ValueError('Use at least two finite training readings and finite input readings')
    mu, sigma = mean(training), stdev(training)
    if sigma == 0:
        raise ValueError('Training standard deviation must be nonzero')
    return {'training': training, 'readings': readings, 'mean': mu,
            'std': sigma, 'normalized': [(x-mu)/sigma for x in readings]}


def calculate():
    return {'interpolation': {'left': 2, 'right': 5, 'missing': 2,
                             'filled': interpolate(2, 5, 2)},
            'normalization': normalize([1, 2, 3], [1, 2, 3, 5]),
            'lab_display_example': {'current': 3.9725, 'mean': 1.7913, 'std': 1.4243,
                                    'normalized': (3.9725-1.7913)/1.4243},
            'note': 'Tiny teaching examples; lab display calculation uses rounded student values.'}

if __name__ == '__main__':
    output = Path(__file__).resolve().parents[2] / 'apps/learning-examples/lab1-preprocessing/trace.json'
    output.write_text(json.dumps(calculate(), indent=2)+'\n')
    print(output)
