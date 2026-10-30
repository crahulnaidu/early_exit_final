import itertools
import random

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from models.backbone import FEATURE_SHAPES
from models.head_builder import (CONV_TYPES, DEFAULT_HEAD, HIDDEN, POOLINGS, REDUCTIONS, ConfigHead, cfg_key,
                                 config_head_flops_breakdown, config_head_params, fc_input_size, normalize_cfg)
from models.multi_exit import CANDIDATE_POSITIONS
from utils.flops import head_cost


def all_cfgs():
    out = [{"n_conv": 0, "pooling": p, "hidden_fc": h} for p in POOLINGS for h in HIDDEN]
    for n, t, r, p, h in itertools.product((1, 2), CONV_TYPES, REDUCTIONS, POOLINGS, HIDDEN):
        out.append({"n_conv": n, "conv_type": t, "channel_reduction": r, "pooling": p, "hidden_fc": h})
    return out


ALL = all_cfgs()
CASES = [(p, c) for p in (2, 42) for c in ALL]                                  # every cfg at the extremes
CASES += [(p, c) for p in CANDIDATE_POSITIONS for c in random.Random(0).sample(ALL, 12)]


def fv(module, shape):
    f = FlopCountAnalysis(module.eval(), torch.zeros(1, *shape))
    f.unsupported_ops_warnings(False)
    f.uncalled_modules_warnings(False)
    ops = dict(f.by_operator())
    return {"conv": 2 * ops.get("conv", 0), "linear": 2 * (ops.get("linear", 0) + ops.get("addmm", 0)),
            "bn": ops.get("batch_norm", 0), "pool": ops.get("adaptive_avg_pool2d", 0)}


def close(a, b, tol=0.02):
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


def test_space_size():
    assert len(ALL) == 12 + 2 * 2 * 3 * 3 * 4 == 156


@pytest.mark.parametrize("pos,cfg", CASES, ids=[cfg_key(p, c) for p, c in CASES])
def test_config_head_flops_params_vs_fvcore(pos, cfg):
    C, H, W = FEATURE_SHAPES[pos]
    head = ConfigHead(C, cfg)
    ours = config_head_flops_breakdown(cfg, C, H, W)
    ref = fv(head, (C, H, W))
    for cat in ("conv", "linear", "bn", "pool"):
        assert close(ours[cat], ref[cat]), (cat, ours[cat], ref[cat])
    assert config_head_params(cfg, C) == sum(p.numel() for p in head.parameters())
    assert head_cost(pos, cfg) == (sum(ours.values()), config_head_params(cfg, C))
    out = head(torch.zeros(2, C, H, W))
    assert out.shape == (2, 10)
    assert head.fc[0].in_features == fc_input_size(pos, cfg)


def test_normalize_drops_unused_conditionals():
    a = normalize_cfg({"n_conv": 0, "conv_type": "standard", "channel_reduction": 0.5, "pooling": 1, "hidden_fc": 0})
    assert a == {"n_conv": 0, "pooling": 1, "hidden_fc": 0}
    assert cfg_key(6, a) == cfg_key(6, {"n_conv": 0, "pooling": 1, "hidden_fc": 0, "conv_type": "x"})
    with pytest.raises(AssertionError):
        normalize_cfg({"n_conv": 1, "conv_type": "bad", "channel_reduction": 0.5, "pooling": 1, "hidden_fc": 0})


def test_default_head_equals_old_a3_cost():
    for pos in (6, 13, 23):
        assert head_cost(pos, DEFAULT_HEAD) == head_cost(pos, "A3")
