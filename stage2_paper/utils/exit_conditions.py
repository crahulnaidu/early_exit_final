"""Exit rules, vectorised over a batch (no per-sample Python loops).

Every rule has a SCORE (confidence, higher = more confident) and a BAND variable:
  rule            score            band variable     comparisons only?
  margin_band     top1 - top2      max logit         yes (default)
  maxlogit_band   max logit        max logit         yes
  energy          -E(z)            E(z)              no: logsumexp (ablation only)
Intermediate exit k:  EXIT if finite(z) AND score >= t_k AND L_k <= band <= U_k, else CONTINUE
                      (non-finite logits at exit k -> that exit is skipped).
Final exit:           ACCEPT if finite(z) AND L_fin <= band <= U_fin, else FLAG.
"""
import torch

RULES = ("margin_band", "maxlogit_band", "energy")
FLAGGED = -1


def margin(z):
    top2 = torch.topk(z, 2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def maxlogit(z):
    return z.max(dim=-1).values


def finite(z):
    return torch.isfinite(z).all(dim=-1)


def compute_energy_score(logits, temperature=1.0):
    """E(z) = -T * logsumexp(z / T); lower = more confident (energy rule / ablation only)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def score_and_band(rule, z):
    if rule == "margin_band":
        return margin(z), maxlogit(z)
    if rule == "maxlogit_band":
        ml = maxlogit(z)
        return ml, ml
    if rule == "energy":
        e = compute_energy_score(z)
        return -e, e
    raise ValueError(f"unknown rule {rule}")


def exit_mask(rule, z, t, lo, hi):
    """Boolean [N]: samples that leave at this intermediate exit."""
    s, b = score_and_band(rule, z)
    return finite(z) & (s >= t) & (b >= lo) & (b <= hi)


def final_accept_mask(rule, z, lo, hi):
    _, b = score_and_band(rule, z)
    return finite(z) & (b >= lo) & (b <= hi)


def route(rule, logits_list, calib):
    """Route a batch through exits E1..EK then the final classifier.

    logits_list: [E1..EK, final], each [N, C]. calib: {"exits": [{"t","L","U"}...], "final": {"L","U"}}.
    Returns exit_id [N] (0..K-1 = early exit, K = final, -1 = flagged) and pred [N] (-1 if flagged).
    """
    K = len(logits_list) - 1
    n = logits_list[0].shape[0]
    device = logits_list[0].device
    exit_id = torch.full((n,), FLAGGED, dtype=torch.long, device=device)
    pred = torch.full((n,), -1, dtype=torch.long, device=device)
    undecided = torch.ones(n, dtype=torch.bool, device=device)

    for k in range(K):
        th = calib["exits"][k]
        ok = undecided & exit_mask(rule, logits_list[k], th["t"], th["L"], th["U"])
        exit_id[ok] = k
        pred[ok] = logits_list[k].argmax(dim=-1)[ok]
        undecided &= ~ok

    fin = calib["final"]
    ok = undecided & final_accept_mask(rule, logits_list[K], fin["L"], fin["U"])
    exit_id[ok] = K
    pred[ok] = logits_list[K].argmax(dim=-1)[ok]
    return exit_id, pred
