"""Cheap head-fault probe for fault-aware NAS (variant E).

For one candidate (exits + calibrated rule), N single-weight exponent faults (bits 23-30) are injected into the
exit-head weights, uniformly over all faultable head weights; a fault whose weight or head logits become
non-finite is rejected and redrawn. Only the faulty head is recomputed, from the cached clean VAL_SEL features.
head_fault_sdc = mean over the N faults of the SDC rate (%) on the trial's own clean-correct VAL_SEL subset
(golden = label; SDC = not flagged and prediction != label). Fault draws come from fixed seeds (seed_base + i), so
every trial sees the same random numbers (mapped onto its own head weights). Head weights live in memory only.
"""
import numpy as np
import torch
import torch.nn as nn

from models.head_builder import cfg_key
from utils.exit_conditions import route
from utils.proxy import train_proxy_head

EXP_BITS = (23, 31)
MAX_REDRAWS = 50


class HeadBank:
    """In-memory proxy-trained heads (never written to disk), keyed like the proxy memo."""

    def __init__(self, cache, epochs, base_seed):
        self.cache, self.epochs, self.base_seed = cache, epochs, base_seed
        self.heads = {}
        self.trained = 0

    def get(self, position, cfg):
        key = cfg_key(position, cfg)
        if key not in self.heads:
            _, head = train_proxy_head(self.cache, position, cfg, self.epochs, self.base_seed, return_head=True)
            for p in head.parameters():
                p.requires_grad_(False)
            self.heads[key] = head.eval()
            self.trained += 1
        return self.heads[key]


def probe_head_sdc(bank, exits, rule, th, sel_idx, y_sel, final_sel, n_faults, seed_base=777000):
    """Returns (mean SDC %, n clean-correct, n rejected faults, n faultable head weights)."""
    heads = [bank.get(p, c) for p, c in exits]          # trains missing heads (needs autograd)
    with torch.no_grad():
        return _probe(bank.cache, heads, exits, rule, th, sel_idx, y_sel, final_sel, n_faults, seed_base)


def _probe(cache, heads, exits, rule, th, sel_idx, y_sel, final_sel, n_faults, seed_base):
    dev = cache.device
    feats = [cache.feats("val", p)[sel_idx.to(dev)] for p, _ in exits]

    def head_logits(k, idx=None):
        x = feats[k] if idx is None else feats[k][idx]
        return torch.cat([heads[k](x[a:a + 1000].float()) for a in range(0, x.shape[0], 1000)])

    clean = [head_logits(k) for k in range(len(exits))]
    final = final_sel.to(dev)
    y = y_sel.to(dev)
    e0, p0 = route(rule, clean + [final], th)
    cc = torch.nonzero((e0 >= 0) & (p0 == y)).flatten()
    clean_cc = [z[cc] for z in clean]
    final_cc, y_cc = final[cc], y[cc]

    tensors = [(k, m.weight) for k, h in enumerate(heads) for m in h.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    cum = np.cumsum([w.numel() for _, w in tensors])
    total = int(cum[-1])
    sdc, rejected = [], 0
    for i in range(n_faults):
        rng = np.random.default_rng(seed_base + i)
        for _ in range(MAX_REDRAWS):
            g = int(rng.random() * total)
            bit = EXP_BITS[0] + int(rng.random() * (EXP_BITS[1] - EXP_BITS[0]))
            t = int(np.searchsorted(cum, g, side="right"))
            local = g - (int(cum[t - 1]) if t > 0 else 0)
            k, w = tensors[t]
            flat = w.data.view(-1)
            ints = flat.view(torch.int32)
            old = ints[local].clone()
            ints[local] ^= (1 << bit)
            ok = bool(torch.isfinite(flat[local]))
            if ok:
                zk = head_logits(k, cc)
                ok = bool(torch.isfinite(zk).all())
            if not ok:
                ints[local] = old
                rejected += 1
                continue
            logits = [zk if j == k else clean_cc[j] for j in range(len(exits))] + [final_cc]
            e, p = route(rule, logits, th)
            sdc.append(100.0 * ((e >= 0) & (p != y_cc)).float().mean().item())
            ints[local] = old
            assert torch.equal(ints[local], old)
            break
        else:
            raise RuntimeError(f"probe fault {i}: more than {MAX_REDRAWS} non-finite redraws")
    return float(np.mean(sdc)), int(cc.numel()), rejected, total
