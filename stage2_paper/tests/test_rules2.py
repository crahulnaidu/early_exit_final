import math

import pytest
import torch

from utils import rules2
from utils.calibrate import calibrate_rule, thresholds_for
from utils.exit_conditions import route as route_old


def synthetic(n=4000, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    out = []
    for s in [1.0, 2.0, 3.0][:K - 1] + [5.0]:
        z = torch.randn(n, 10, generator=g)
        z[torch.arange(n), y] += s * torch.rand(n, generator=g) * 3
        out.append(z)
    return out, y


@pytest.mark.parametrize("new,old", [("margin_band", "margin_band"), ("maxlogit_band", "maxlogit_band"),
                                     ("energy_band", "energy")])
def test_generic_rules_reproduce_existing_calibration_and_routing(new, old):
    val, y = synthetic(seed=1)
    test, _ = synthetic(seed=2)
    test[0][:30] *= 1e6
    test[1][30:40, 2] = float("inf")
    for alpha in (0.5, 1.0, 5.0):
        c_old = calibrate_rule(old, val, y, alphas=(alpha,))
        c_new = rules2.calibrate(new, val, y, alphas=(alpha,))
        for eo, en in zip(c_old["exits"], c_new["exits"]):
            assert eo["t"] == en["t"] and eo["L"] == en["L"] and eo["U"] == en["U"]
        e1, p1 = route_old(old, test, thresholds_for(c_old, alpha))
        e2, p2, _ = rules2.route(c_new, test, alpha)
        assert torch.equal(e1, e2) and torch.equal(p1, p2)


@pytest.mark.parametrize("rule", [r for r in rules2.RULES if rules2.RULES[r].kind == "standard"])
def test_every_standard_rule_calibrates_routes_and_partitions(rule):
    val, y = synthetic(seed=3)
    cal = rules2.calibrate(rule, val, y, alphas=(1.0, 5.0))
    e, p, m = rules2.route(cal, val, 1.0)
    K = len(val) - 1
    assert ((e >= -1) & (e <= K)).all()
    assert (p[e == -1] == -1).all() and (p[e >= 0] >= 0).all()
    assert torch.equal(m[e >= 0], p[e >= 0]) and (m[e == -1] >= 0).all()
    far = 100 * (e == -1).float().mean().item()
    if rules2.RULES[rule].final == "none":
        assert far == 0
    elif rule != "margin_multiband":           # extra bands add their own small false-alarm budget
        assert abs(far - 1.0) <= 0.3


def test_plain_dnn_is_final_classifier():
    val, y = synthetic(seed=4)
    cal = rules2.calibrate("plain_dnn", val, y)
    e, p, m = rules2.route(cal, val)
    assert (e == len(val) - 1).all() and torch.equal(p, val[-1].argmax(-1))


@pytest.mark.parametrize("gamma,tau", [(0.5, 0.0), (0.7, 3.0), (0.9, 6.0), (0.6, 50.0)])
def test_mendnet_vectorised_equals_reference(gamma, tau):
    val, _ = synthetic(n=600, seed=5)
    e, p, m = rules2.route_mendnet(val, gamma, tau)
    rows = [z.tolist() for z in val]
    for i in range(600):
        re, rp, rm = rules2.mendnet_reference([r[i] for r in rows], gamma, tau)
        assert (int(e[i]), int(p[i]), int(m[i])) == (re, rp, rm), i


def test_mendnet_flags_when_no_exit_is_certain_and_accepts_exploded_logits():
    val, y = synthetic(n=2000, seed=6)
    cal = rules2.calibrate_mendnet_clean(val, y, alpha=1.0)
    e, _, _ = rules2.route_mendnet(val, cal["gamma"], cal["tau"])
    assert abs(100 * (e == -1).float().mean().item() - 1.0) <= 0.3
    boom = [z.clone() for z in val]
    boom[0] = boom[0] * 1e6                    # exponent-style blow-up at exit 1: one-sided rule accepts it
    e2, _, _ = rules2.route_mendnet(boom, cal["gamma"], cal["tau"])
    assert (e2 == 0).float().mean() > 0.9
