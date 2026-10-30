import importlib.util
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent.parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev = load("offline41", "41_offline_eval.py")
cs = load("campaign40", "40_campaign_store.py")


def test_sig_count_is_smallest_significant_count():
    for n, far in [(100, 0.01), (500, 0.01), (10000, 0.009)]:
        c = ev.sig_count(n, far)
        assert binom.sf(c - 1, n, far) < 0.01 <= binom.sf(c - 2, n, far)
    assert ev.sig_count(100, 0.0) == 1


def test_per_run_outcomes_partition_the_clean_correct_subset():
    g = torch.Generator().manual_seed(0)
    R, N, K = 5, 1000, 3
    y = torch.randint(0, 10, (1, N), generator=g)
    e = torch.randint(-1, K + 1, (R, N), generator=g)
    p = torch.where(e < 0, -1, torch.randint(0, 10, (R, N), generator=g))
    S = torch.rand(1, N, generator=g) < 0.9
    corr = torch.rand(R, N, generator=g) < 0.5
    costs = torch.arange(K + 2, dtype=torch.float64) + 1.0
    r = ev.per_run(e, p, p.clamp_min(0), y, S, corr, costs, 0.01, K)
    assert np.allclose(r["sdc_rate"] + r["flag_rate"] + r["masked_rate"], 100)
    assert (sum(r[f"exit_{j}"] for j in range(K + 1)) + r["n_flag"] == N).all()
    assert (r["latency_100"][r["n_flag"] == 0] == np.inf).all()


def test_agree_rule_pays_for_the_next_exit():
    pos, archs = [16, 22, 23], [{"n_conv": 0, "pooling": 1, "hidden_fc": 0}] * 3
    a = ev.cost_vector(pos, archs, False, "cpu")
    b = ev.cost_vector(pos, archs, True, "cpu")
    assert torch.equal(b[:2], a[1:3]) and b[2] == a[3] and b[-1] == a[3]


def test_int8_flip_and_undo_are_exact():
    p = torch.nn.Parameter(torch.randn(50))
    w = cs.Int8Weights({"G": [("w", p)]})
    before = p.data.clone()
    for bit in range(8):
        f = w.flip("w", p, 7, bit)
        assert f["new_code"] in range(-128, 128) and (f["old_code"] ^ f["new_code"]) & 0xFF == 1 << bit
        w.unflip("w", p, 7, f["old_code"], f["old"])
        assert torch.equal(p.data, before)
