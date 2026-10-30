import numpy as np
import torch

from faults.evaluator import SubsetEvaluator
from faults.inject import (EXP_BITS, LocationSampler, fault_locations, fault_seed, identical_to, snapshot)
from utils.flops import head_params


def images(n=24, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def test_locations_partition_all_faultable_weights(ee_model):
    locs = fault_locations(ee_model)
    assert list(locs) == ["G1", "G2", "G3", "HEADS"]
    counts = {k: sum(p.numel() for _, p in v) for k, v in locs.items()}
    assert counts == {"G1": 554_688, "G2": 1_179_648, "G3": 13_243_392, "HEADS": 1_101_568}
    names = [n for v in locs.values() for n, _ in v]
    assert len(names) == len(set(names))
    expected = {n for n, m in ee_model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}
    assert {n[: -len(".weight")] for n in names} == expected
    assert all(not n.endswith("bias") for n in names)


def test_same_seed_same_fault_and_bits_in_exponent(ee_model):
    s = LocationSampler("G2", fault_locations(ee_model)["G2"])
    a = [s.draw(np.random.default_rng(fault_seed(2, 3, 7))) for _ in range(1)]
    b = [s.draw(np.random.default_rng(fault_seed(2, 3, 7))) for _ in range(1)]
    assert a == b
    rng = np.random.default_rng(0)
    bits = {s.draw(rng)["bit"] for _ in range(2000)}
    assert bits == set(EXP_BITS)


def test_sampling_is_uniform_over_weights_not_layers(ee_model):
    s = LocationSampler("HEADS", fault_locations(ee_model)["HEADS"])
    rng = np.random.default_rng(0)
    hits = np.bincount([s.draw(rng)["tensor"] for _ in range(20000)], minlength=len(s.params))
    share = hits / hits.sum()
    expected = np.array([p.numel() for p in s.params]) / s.total
    assert np.allclose(share, expected, atol=0.01)


def test_flip_changes_exponent_only_and_undo_is_bit_identical(ee_model):
    locs = fault_locations(ee_model)
    snap = snapshot(locs)
    s = LocationSampler("G3", locs["G3"])
    rng = np.random.default_rng(5)
    faults = [s.flip(s.draw(rng)) for _ in range(8)]
    assert not identical_to(locs, snap)
    for f in faults:
        old = np.array(f["old"], dtype=np.float32).view(np.int32)
        new = np.array(f["new"], dtype=np.float32).view(np.int32)
        assert (old ^ new) == (1 << f["bit"]) and 23 <= f["bit"] <= 30
    for f in reversed(faults):
        s.unflip(f)
    assert identical_to(locs, snap)


def test_k0_and_segment_forward_equals_full_forward(ee_model):
    ev = SubsetEvaluator(ee_model, images(), batch_size=10)
    full = ev.full_logits()
    assert all(torch.equal(a, b) for a, b in zip(full, ev.clean_logits))       # k = 0: deterministic
    for loc in ("G1", "G2", "G3", "HEADS"):
        assert all(torch.equal(a, b) for a, b in zip(ev.logits(loc), ev.clean_logits)), loc
    locs = fault_locations(ee_model)
    for loc in ("G1", "G2", "G3", "HEADS"):
        s = LocationSampler(loc, locs[loc])
        f = s.flip(s.draw(np.random.default_rng(11)))
        fast, slow = ev.logits(loc), ev.full_logits()
        s.unflip(f)
        for a, b in zip(fast, slow):
            assert torch.equal(a, b) or (torch.isnan(a) == torch.isnan(b)).all(), loc


def test_k0_route_gives_no_sdc(ee_model):
    from utils.calibrate import calibrate_rule, thresholds_for
    from utils.exit_conditions import route
    ev = SubsetEvaluator(ee_model, images(200, 3), batch_size=50)
    y = ev.clean_logits[-1].argmax(-1)
    calib = calibrate_rule("margin_band", ev.clean_logits, y)
    th = thresholds_for(calib, 1.0)
    e1, p1 = route("margin_band", ev.clean_logits, th)
    e2, p2 = route("margin_band", ev.full_logits(), th)
    assert torch.equal(e1, e2) and torch.equal(p1, p2)


def _model(positions, archs):
    from models.backbone import build_backbone
    from models.searched_exit_model import SearchedMultiExitVGG
    b = build_backbone()
    return SearchedMultiExitVGG(b.features, b.avgpool, b.classifier, positions, archs).eval()


def test_three_groups_for_three_exits_and_two_for_one_exit():
    from faults.inject import location_ranges
    cfg = {"n_conv": 1, "conv_type": "depthwise_separable", "channel_reduction": 0.25, "pooling": 2,
           "hidden_fc": 64}
    m3 = _model([6, 16, 29], [cfg, "A1", cfg])
    assert dict(location_ranges(m3)) == {"G1": (0, 6), "G2": (7, 29), "G3": (30, 43)}
    locs = fault_locations(m3)
    assert list(locs) == ["G1", "G2", "G3", "HEADS"]
    total = sum(p.numel() for v in locs.values() for _, p in v)
    expected = sum(m.weight.numel() for m in m3.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
    assert total == expected
    m1 = _model([23], ["A3"])
    assert list(fault_locations(m1)) == ["G1", "G3", "HEADS"]           # G2 empty -> skipped


def test_segment_forward_equals_full_forward_three_exits():
    cfg = {"n_conv": 2, "conv_type": "standard", "channel_reduction": 0.125, "pooling": 4, "hidden_fc": 0}
    m = _model([5, 13, 33], [cfg, "A2", "A4"])
    ev = SubsetEvaluator(m, images(12), batch_size=5)
    locs = fault_locations(m)
    for loc in locs:
        assert all(torch.equal(a, b) for a, b in zip(ev.logits(loc), ev.clean_logits)), loc
        s = LocationSampler(loc, locs[loc])
        f = s.flip(s.draw(np.random.default_rng(3)))
        fast, slow = ev.logits(loc), ev.full_logits()
        s.unflip(f)
        for a, b in zip(fast, slow):
            assert torch.equal(a, b) or (torch.isnan(a) == torch.isnan(b)).all(), loc


def test_location_ids_fixed_for_seeds():
    from faults.inject import LOCATION_IDS
    assert LOCATION_IDS == {"G1": 1, "G2": 2, "G3": 3, "HEADS": 4}
