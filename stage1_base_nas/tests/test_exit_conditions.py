import math

import pytest
import torch

from utils.calibrate import calibrate_rule, thresholds_for
from utils.exit_conditions import RULES, margin, maxlogit, route
from utils.reference import compare_with_reference


def synthetic(n=3000, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    logits = []
    for strength in [1.0, 2.0, 4.0][: K] + [6.0]:
        z = torch.randn(n, 10, generator=g)
        z[torch.arange(n), y] += strength * torch.rand(n, generator=g) * 3
        logits.append(z)
    return logits, y


def test_margin_and_maxlogit():
    z = torch.tensor([[1.0, 5.0, 3.0], [0.0, 0.0, -1.0]])
    assert torch.equal(margin(z), torch.tensor([2.0, 0.0]))
    assert torch.equal(maxlogit(z), torch.tensor([5.0, 0.0]))


@pytest.mark.parametrize("rule", RULES)
def test_vectorised_equals_reference_loop_with_faulty_rows(rule):
    logits, y = synthetic()
    calib = calibrate_rule(rule, [z[:2000] for z in logits], y[:2000])
    test = [z[2000:].clone() for z in logits]
    test[0][:20] *= 1e6                               # exploded logits at exit 1
    test[0][20:30, 3] = float("inf")                  # non-finite at exit 1 -> skip exit 1
    test[1][30:40, 0] = float("nan")                  # non-finite at exit 2
    test[2][40:50, 1] = float("inf")                  # non-finite final -> flagged
    test[2][50:60] *= 1e-6                            # collapsed final logits
    for alpha in (0.5, 1.0, 2.0, 5.0):
        th = thresholds_for(calib, alpha)
        exit_id, pred = route(rule, test, th)
        mism, n = compare_with_reference(rule, test, th, exit_id, pred, n=1000)
        assert mism == 0, (rule, alpha, mism)
        assert (exit_id[40:50] != 2).all()            # non-finite final logits never accepted


@pytest.mark.parametrize("rule", RULES)
def test_val_false_alarm_rate_equals_alpha(rule):
    logits, y = synthetic(n=5000)
    calib = calibrate_rule(rule, logits, y)
    for alpha, fin in calib["final"].items():
        assert abs(fin["val_flag_rate"] - float(alpha)) <= 0.1, (alpha, fin["val_flag_rate"])


def test_exit_accuracy_criterion_on_val():
    logits, y = synthetic(n=5000)
    calib = calibrate_rule("margin_band", logits, y)
    for e in calib["exits"]:
        if e["val_exit_count"] >= 25:
            assert e["val_head_acc_on_exited"] >= e["val_final_acc_on_exited"] - 0.5
        assert math.isinf(e["t"]) or float(torch.tensor(e["t"], dtype=torch.float32)) == e["t"]


def test_exits_partition_and_flags():
    logits, y = synthetic()
    calib = calibrate_rule("margin_band", logits, y)
    exit_id, pred = route("margin_band", logits, thresholds_for(calib, 1.0))
    assert ((exit_id >= -1) & (exit_id <= 2)).all()
    assert (pred[exit_id == -1] == -1).all() and (pred[exit_id >= 0] >= 0).all()
