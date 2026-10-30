import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_check", Path(__file__).resolve().parent.parent / "scripts" / "checks" / "run_check.py")
run_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_check)


def test_to_count_recovers_exact_counts_from_float32_and_float64_percentages():
    assert run_check.to_count(92.26999878883362, 10000) == 9227      # float32-derived (overnight-2 J4)
    assert run_check.to_count(92.27, 10000) == 9227                  # exact python float
    assert run_check.to_count(92.64, 5000) == 4632


def test_to_count_refuses_non_integer_percentages():
    with pytest.raises(AssertionError):
        run_check.to_count(92.275, 10000)
