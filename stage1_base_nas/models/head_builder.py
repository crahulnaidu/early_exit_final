"""Configurable exit heads for the NAS search space (Step 4).

cfg = {"n_conv": 0|1|2,
       "conv_type": "standard" | "depthwise_separable",     (only if n_conv > 0)
       "channel_reduction": 0.5 | 0.25 | 0.125,             (only if n_conv > 0)
       "pooling": 1 | 2 | 4,                                 (1 = GAP, 2/4 = adaptive avg pool 2x2 / 4x4)
       "hidden_fc": 0 | 64 | 128 | 256}
Head: [conv block] x n_conv -> AdaptiveAvgPool2d(pooling) -> flatten -> [Linear(F, h) -> ReLU] -> Linear(., 10)
  standard conv block            Conv3x3(c_in -> c_out, no bias) + BN + ReLU
  depthwise_separable conv block DW Conv3x3(c_in, groups=c_in, no bias) + PW Conv1x1(c_in -> c_out, no bias) + BN + ReLU
  c_out = C * channel_reduction for every conv block (the second block keeps c_out); stride 1, padding 1.
FC input F = c_last * pooling^2, so channel reduction and pooling control how small the FC is.
"""
import json

import torch
import torch.nn as nn

from models.backbone import FEATURE_SHAPES, NUM_CLASSES
from models.exit_heads import HEAD_BUILDERS

N_CONV = (0, 1, 2)
CONV_TYPES = ("standard", "depthwise_separable")
REDUCTIONS = (0.5, 0.25, 0.125)
POOLINGS = (1, 2, 4)
HIDDEN = (0, 64, 128, 256)
DEFAULT_HEAD = {"n_conv": 0, "pooling": 4, "hidden_fc": 256}   # = old A3 (hand-placed baseline head)


def normalize_cfg(cfg):
    """Canonical cfg: conv_type / channel_reduction only present when n_conv > 0; validated."""
    n = int(cfg["n_conv"])
    assert n in N_CONV and int(cfg["pooling"]) in POOLINGS and int(cfg["hidden_fc"]) in HIDDEN, cfg
    out = {"n_conv": n, "pooling": int(cfg["pooling"]), "hidden_fc": int(cfg["hidden_fc"])}
    if n > 0:
        assert cfg["conv_type"] in CONV_TYPES and float(cfg["channel_reduction"]) in REDUCTIONS, cfg
        out["conv_type"] = cfg["conv_type"]
        out["channel_reduction"] = float(cfg["channel_reduction"])
    return out


def cfg_key(position, cfg):
    if isinstance(cfg, str):                       # old menu head ("A1".."A5")
        return f"p{position}_{cfg}"
    return f"p{position}_" + json.dumps(normalize_cfg(cfg), sort_keys=True)


def _conv_channels(C, cfg):
    return int(C * cfg["channel_reduction"]) if cfg["n_conv"] > 0 else C


class ConfigHead(nn.Module):
    def __init__(self, C, cfg, num_classes=NUM_CLASSES):
        super().__init__()
        cfg = normalize_cfg(cfg)
        layers, c_in = [], C
        c_out = _conv_channels(C, cfg)
        for _ in range(cfg["n_conv"]):
            if cfg["conv_type"] == "standard":
                layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)]
            else:
                layers += [nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
                           nn.Conv2d(c_in, c_out, 1, bias=False)]
            layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(cfg["pooling"])
        f = c_in * cfg["pooling"] ** 2
        h = cfg["hidden_fc"]
        self.fc = (nn.Sequential(nn.Linear(f, h), nn.ReLU(inplace=True), nn.Linear(h, num_classes)) if h
                   else nn.Sequential(nn.Linear(f, num_classes)))

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.convs(x)), 1))


def make_head(position, arch):
    """arch: old menu name ("A1".."A5") or a cfg dict."""
    C, H, W = FEATURE_SHAPES[position]
    if isinstance(arch, str):
        return HEAD_BUILDERS[arch](in_channels=C)
    return ConfigHead(C, arch)


def config_head_flops_breakdown(cfg, C, H, W, num_classes=NUM_CLASSES):
    """Same conventions as utils/flops.py (2*MAC, BN 2*elements, ReLU elements, pooling input elements)."""
    cfg = normalize_cfg(cfg)
    hw = H * W
    f = {"conv": 0, "linear": 0, "bn": 0, "relu": 0, "pool": 0}
    c_in, c_out = C, _conv_channels(C, cfg)
    for _ in range(cfg["n_conv"]):
        if cfg["conv_type"] == "standard":
            f["conv"] += 2 * c_in * c_out * 9 * hw
        else:
            f["conv"] += 2 * c_in * 9 * hw + 2 * c_in * c_out * hw
        f["bn"] += 2 * c_out * hw
        f["relu"] += c_out * hw
        c_in = c_out
    f["pool"] = c_in * hw
    feat = c_in * cfg["pooling"] ** 2
    h = cfg["hidden_fc"]
    if h:
        f["linear"] = 2 * (feat * h + h * num_classes)
        f["relu"] += h
    else:
        f["linear"] = 2 * feat * num_classes
    return f


def config_head_params(cfg, C, num_classes=NUM_CLASSES):
    cfg = normalize_cfg(cfg)
    p, c_in, c_out = 0, C, _conv_channels(C, cfg)
    for _ in range(cfg["n_conv"]):
        p += c_in * c_out * 9 if cfg["conv_type"] == "standard" else c_in * 9 + c_in * c_out
        p += 2 * c_out
        c_in = c_out
    feat = c_in * cfg["pooling"] ** 2
    h = cfg["hidden_fc"]
    p += (feat * h + h + h * num_classes + num_classes) if h else (feat * num_classes + num_classes)
    return p


def fc_input_size(position, cfg):
    C, _, _ = FEATURE_SHAPES[position]
    cfg = normalize_cfg(cfg)
    return _conv_channels(C, cfg) * cfg["pooling"] ** 2
