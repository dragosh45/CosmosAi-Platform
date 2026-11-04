"""Check arithmetic teaching examples against explicit expected results."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lab1_example', ROOT/'scripts/examples/lab1_preprocessing.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_interpolation_distributes_missing_points_and_handles_falling_signal():
    assert example.interpolate(2, 5, 2) == [3, 4]
    assert example.interpolate(4, 2, 1) == [3]
    with pytest.raises(ValueError):
        example.interpolate(2, 5, 0)


def test_normalization_uses_training_statistics_even_for_extreme_test_reading():
    result = example.normalize([1, 2, 3], [5, -100])
    assert result['mean'] == 2
    assert result['std'] == 1
    assert result['normalized'] == [3, -102]
    with pytest.raises(ValueError):
        example.normalize([2, 2, 2], [5])
    with pytest.raises(ValueError):
        example.normalize([1, float('nan')], [5])


def test_saved_lab1_trace_reproduces_the_calculation():
    trace = json.loads((ROOT/'apps/learning-examples/lab1-preprocessing/trace.json').read_text())
    assert trace == example.calculate()
    assert trace['lab_display_example']['normalized'] == pytest.approx(1.5314, abs=.0001)
