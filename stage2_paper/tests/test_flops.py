"""Analytic FLOPs (utils/flops.py) vs fvcore, per category, within 2 %.

fvcore counts conv/linear as MACs (we count 2*MAC), BatchNorm (eval) as 2*elements, adaptive avg-pool as input
elements, and does not count ReLU / max-pool. Those two are checked against their formula and shown to be < 1 %.
"""
import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from models.backbone import FEATURE_SHAPES, build_backbone
from models.exit_heads import HEAD_BUILDERS
from models.multi_exit import CANDIDATE_POSITIONS
from utils.flops import (BACKBONE_FLOPS, CLASSIFIER_FLOPS, TOTAL_BASE_FLOPS, backbone_flops_breakdown,
                         classifier_flops_breakdown, exit_costs, head_cost, head_flops_breakdown, head_params)


def fv(module, shape):
    f = FlopCountAnalysis(module.eval(), torch.zeros(1, *shape))
    f.unsupported_ops_warnings(False)
    f.uncalled_modules_warnings(False)
    ops = dict(f.by_operator())
    return {"conv": 2 * ops.get("conv", 0), "linear": 2 * (ops.get("linear", 0) + ops.get("addmm", 0)),
            "bn": ops.get("batch_norm", 0), "pool": ops.get("adaptive_avg_pool2d", 0)}


def close(a, b, tol=0.02):
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


def test_backbone_matches_fvcore():
    model = build_backbone()
    ours = backbone_flops_breakdown()
    cls = classifier_flops_breakdown()
    ref = fv(model, (3, 32, 32))
    assert close(ours["conv"], ref["conv"])
    assert close(cls["linear"], ref["linear"])
    assert close(ours["bn"], ref["bn"])
    assert close(cls["pool"], ref["pool"])           # final AdaptiveAvgPool2d(1)
    assert ours["conv"] == 626_688_000 or close(ours["conv"], 626.4e6, 0.002)
    assert cls["linear"] == 2 * (512 * 512 + 512 * 10) == 534_528
    unverified = ours["relu"] + ours["pool"] + cls["relu"]
    assert unverified / TOTAL_BASE_FLOPS < 0.01
    assert TOTAL_BASE_FLOPS == BACKBONE_FLOPS + CLASSIFIER_FLOPS


@pytest.mark.parametrize("pos", CANDIDATE_POSITIONS)
@pytest.mark.parametrize("arch", list(HEAD_BUILDERS))
def test_head_matches_fvcore_and_params(pos, arch):
    C, H, W = FEATURE_SHAPES[pos]
    head = HEAD_BUILDERS[arch](in_channels=C)
    ours = head_flops_breakdown(arch, C, H, W)
    ref = fv(head, (C, H, W))
    for cat in ("conv", "linear", "bn", "pool"):
        assert close(ours[cat], ref[cat]), (cat, ours[cat], ref[cat])
    assert head_params(arch, C) == sum(p.numel() for p in head.parameters())
    assert head_cost(pos, arch) == (sum(ours.values()), head_params(arch, C))


def test_head_flops_use_real_spatial_size():
    # A4 at position 16 sees an 8x8 map: its 1x1 conv must cost 2*256*64*8*8, not the old 4x4 assumption.
    assert head_flops_breakdown("A4", *FEATURE_SHAPES[16])["conv"] == 2 * 256 * 64 * 64


def test_feature_shapes_match_forward():
    model = build_backbone().eval()
    x = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            x = layer(x)
            assert tuple(x.shape[1:]) == FEATURE_SHAPES[i], i


def test_exit_costs_formula():
    costs = exit_costs([16, 23], ["A4", "A3"])
    h1, h2 = head_cost(16, "A4")[0], head_cost(23, "A3")[0]
    from utils.flops import CUMULATIVE_BACKBONE_FLOPS
    assert costs == [CUMULATIVE_BACKBONE_FLOPS[16] + h1, CUMULATIVE_BACKBONE_FLOPS[23] + h1 + h2,
                     TOTAL_BASE_FLOPS + h1 + h2]
