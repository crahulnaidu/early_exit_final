"""Threshold calibration on clean VALIDATION logits only (never test).

Per intermediate exit k (sequentially, on the samples that did not leave earlier):
  L_k, U_k = 0.1th / 99.9th percentile of the clean band variable at exit k (all val samples)
  t_k      = smallest candidate threshold such that, for t_k and every larger candidate, the val accuracy of the
             samples exiting at k is >= the final classifier's accuracy on those same samples - ACC_SLACK.
             Candidates = the 0, 0.5, ..., 100th percentiles of the score among eligible samples, then +inf
             (= exit disabled). Sets smaller than MIN_COUNT samples count as satisfied (too small to judge).
Final exit, per alpha: [L_fin, U_fin] on the band variable of the samples that reach the final classifier,
  alpha/2 of all val samples flagged in each tail, so the clean val false-alarm rate is alpha.
"""
import math

import numpy as np
import torch

from utils.exit_conditions import finite, route, score_and_band

ALPHAS = (0.5, 1.0, 2.0, 5.0)
DEFAULT_ALPHA = 1.0
ACC_SLACK = 0.5     # accuracy points
MIN_COUNT = 25      # minimum exiting samples for the accuracy criterion to be evaluated
BAND_PCT = (0.1, 99.9)


def _np(x):
    return x.detach().double().cpu().numpy()


def _f32(v):
    """Round a threshold to the nearest float32 so torch (float32) and Python comparisons agree exactly."""
    return float(np.float32(v)) if math.isfinite(v) else v


def calibrate_rule(rule, val_logits, val_targets, alphas=ALPHAS):
    """val_logits: [E1..EK, final] tensors [N, C] (clean val); val_targets: [N]."""
    K = len(val_logits) - 1
    n = val_targets.shape[0]
    final_correct = _np(val_logits[K].argmax(dim=-1).eq(val_targets))
    eligible = np.ones(n, dtype=bool)
    exits = []

    for k in range(K):
        z = val_logits[k]
        assert bool(finite(z).all()), f"non-finite clean val logits at exit {k + 1}"
        s, b = (_np(v) for v in score_and_band(rule, z))
        head_correct = _np(z.argmax(dim=-1).eq(val_targets))
        lo, hi = (_f32(float(v)) for v in np.percentile(b, BAND_PCT))
        cand = eligible & (b >= lo) & (b <= hi)

        grid = np.unique(np.percentile(s[cand], np.arange(0, 100.5, 0.5))) if cand.any() else np.array([])
        thresholds = list(grid) + [math.inf]
        ok = []
        for t in thresholds:
            sel = cand & (s >= t)
            m = int(sel.sum())
            if m < MIN_COUNT:
                ok.append(True)
            else:
                ok.append(100 * head_correct[sel].mean() >= 100 * final_correct[sel].mean() - ACC_SLACK)
        last_fail = max((i for i, v in enumerate(ok) if not v), default=-1)
        t_k = _f32(float(thresholds[last_fail + 1]))

        sel = cand & (s >= t_k)
        exits.append({
            "t": t_k, "L": lo, "U": hi,
            "val_exit_count": int(sel.sum()),
            "val_head_acc_on_exited": float(100 * head_correct[sel].mean()) if sel.any() else None,
            "val_final_acc_on_exited": float(100 * final_correct[sel].mean()) if sel.any() else None,
        })
        eligible &= ~sel

    _, b_fin = (_np(v) for v in score_and_band(rule, val_logits[K]))
    reach = b_fin[eligible]
    finals = {}
    for alpha in alphas:
        per_tail = alpha / 100.0 * n / 2.0
        if len(reach) == 0:                      # nothing reaches the final classifier on val
            finals[str(alpha)] = {"L": -math.inf, "U": math.inf, "alpha_reachable": False}
            continue
        q = 100.0 * per_tail / len(reach)
        if q >= 50.0:                            # fewer samples reach the final than alpha*N: flag all of them
            finals[str(alpha)] = {"L": math.inf, "U": -math.inf, "alpha_reachable": False}
            continue
        lo, hi = (_f32(float(v)) for v in np.percentile(reach, (q, 100.0 - q)))
        finals[str(alpha)] = {"L": lo, "U": hi, "alpha_reachable": True}

    calib = {"rule": rule, "exits": exits, "final": finals, "default_alpha": DEFAULT_ALPHA,
             "acc_slack": ACC_SLACK, "min_count": MIN_COUNT, "band_percentiles": list(BAND_PCT)}
    for alpha in alphas:
        exit_id, _ = route(rule, val_logits, thresholds_for(calib, alpha))
        finals[str(alpha)]["val_flag_rate"] = float(100.0 * (exit_id == -1).float().mean())
        finals[str(alpha)]["val_exit_fractions"] = [float(100.0 * (exit_id == j).float().mean())
                                                    for j in range(K + 1)]
    return calib


def thresholds_for(calib, alpha=None):
    """The {"exits", "final"} dict that utils.exit_conditions.route expects, for one alpha."""
    alpha = calib["default_alpha"] if alpha is None else alpha
    fin = calib["final"][str(float(alpha))]
    return {"exits": [{"t": e["t"], "L": e["L"], "U": e["U"]} for e in calib["exits"]],
            "final": {"L": fin["L"], "U": fin["U"]}}


def collect_logits(model, loader, device):
    """Clean logits [E1..EK, final] and targets for a whole loader (eval mode, no grad)."""
    model.eval()
    outs, ys = None, []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            if outs is None:
                outs = [[] for _ in logits]
            for i, z in enumerate(logits):
                outs[i].append(z.cpu())
            ys.append(y)
    return [torch.cat(o) for o in outs], torch.cat(ys)
