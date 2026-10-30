"""FP32 exponent-bit fault injection by location (Step 3).

Faultable tensors: the .weight of every Conv2d and Linear module (biases and BatchNorm are never faulted).
Locations for a multi-exit model with exits at p_1 < ... < p_K (always 3 backbone groups + heads):
  G1        features[0 .. p_1]                       (before / up to the first exit)
  G2        features[p_1+1 .. p_K]                   (all middle segments merged; empty -> absent if K = 1)
  G3        features[p_K+1 .. 43] + classifier       (after the last exit)
  HEADS     all exit-head weights
Within a location, a weight is chosen uniformly over ALL its weights (global index over the concatenated
tensors), never twice in the same run (one flip per weight), with an exponent bit drawn from 23..30.
"""
import hashlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

EXP_BITS = tuple(range(23, 31))
FAULTABLE = (nn.Conv2d, nn.Linear)


def _weights(module, prefix):
    return [(f"{prefix}.{name}.weight" if name else f"{prefix}.weight", m.weight)
            for name, m in module.named_modules() if isinstance(m, FAULTABLE)]


LOCATION_IDS = {"G1": 1, "G2": 2, "G3": 3, "HEADS": 4}   # fixed ids -> fault seeds independent of K


def location_ranges(model):
    """OrderedDict location -> (first, last) features index (inclusive); G2 omitted when there is one exit."""
    p = list(model.positions)
    ranges = OrderedDict([("G1", (0, p[0]))])
    if len(p) >= 2:
        ranges["G2"] = (p[0] + 1, p[-1])
    ranges["G3"] = (p[-1] + 1, len(model.features) - 1)
    return ranges


def fault_locations(model):
    """OrderedDict location -> list of (qualified weight name, Parameter)."""
    locs = OrderedDict()
    for loc, (lo, hi) in location_ranges(model).items():
        tensors = []
        for i in range(lo, hi + 1):
            tensors += _weights(model.features[i], f"features.{i}")
        if loc == "G3":
            tensors += _weights(model.classifier, "classifier")
        locs[loc] = tensors
    locs["HEADS"] = _weights(model.exit_heads, "exit_heads")
    return locs


def location_recompute(model, location):
    """(first features index to recompute, number of leading exits whose logits stay clean) for a fault in
    `location`; None for HEADS (only the heads are recomputed)."""
    if location == "HEADS":
        return None
    p = list(model.positions)
    return {"G1": (0, 0), "G2": (p[0] + 1, 1), "G3": (p[-1] + 1, len(p))}[location]


def fault_seed(location_id, k, run):
    """Deterministic seed per (location, k, run); independent of the exit rule, so rules see the same faults."""
    return 1_000_000 * location_id + 1000 * k + run


class LocationSampler:
    """Uniform sampling of (weight, exponent bit) over all faultable weights of one location."""

    def __init__(self, location, tensors):
        self.location = location
        self.names = [n for n, _ in tensors]
        self.params = [p for _, p in tensors]
        self.cum = np.cumsum([p.numel() for p in self.params])
        self.total = int(self.cum[-1])

    def draw(self, rng):
        g = int(rng.integers(0, self.total))
        t = int(np.searchsorted(self.cum, g, side="right"))
        local = g - (int(self.cum[t - 1]) if t > 0 else 0)
        bit = int(rng.integers(EXP_BITS[0], EXP_BITS[-1] + 1))
        return {"location": self.location, "tensor": t, "layer": self.names[t], "index": local,
                "global_index": g, "bit": bit}

    def flip(self, fault):
        """XOR one bit of one FP32 weight in place; records old/new values in the fault dict."""
        flat = self.params[fault["tensor"]].data.view(-1)
        ints = flat.view(torch.int32)
        fault["old"] = flat[fault["index"]].item()
        ints[fault["index"]] ^= (1 << fault["bit"])
        fault["new"] = flat[fault["index"]].item()
        return fault

    def unflip(self, fault):
        flat = self.params[fault["tensor"]].data.view(-1)
        flat.view(torch.int32)[fault["index"]] ^= (1 << fault["bit"])


def snapshot(locations):
    """Bit-exact copies (int32 views) of every faultable tensor."""
    return {name: p.data.view(-1).view(torch.int32).clone() for tensors in locations.values() for name, p in tensors}


def identical_to(locations, snap):
    return all(torch.equal(p.data.view(-1).view(torch.int32), snap[name])
               for tensors in locations.values() for name, p in tensors)


def checksum(locations):
    h = hashlib.sha256()
    for tensors in locations.values():
        for name, p in tensors:
            h.update(name.encode())
            h.update(p.data.detach().cpu().numpy().tobytes())
    return h.hexdigest()
