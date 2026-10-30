"""Slow, independent per-sample implementation of the exit rules.

Used only to check that the vectorised utils.exit_conditions.route gives identical results.
Arithmetic is done in float32 (numpy scalars) like the model's logits, and thresholds are float32-representable
(utils.calibrate rounds them), so every comparison is the same as in the vectorised version.
"""
import math

import numpy as np


def _score_band(rule, row):
    if rule == "margin_band":
        srt = sorted(row, reverse=True)
        return np.float32(srt[0]) - np.float32(srt[1]), np.float32(srt[0])
    if rule == "maxlogit_band":
        m = np.float32(max(row))
        return m, m
    if rule == "energy":
        x = np.asarray(row, dtype=np.float32)
        m = x.max()
        e = -(m + np.log(np.exp(x - m).sum(dtype=np.float32)))
        return -e, e
    raise ValueError(rule)


def _argmax(row):
    best = 0
    for i, v in enumerate(row):
        if v > row[best]:
            best = i
    return best


def route_one(rule, rows, th):
    """rows: [E1..EK, final] lists of float32 values for one sample. Returns (exit_id, pred)."""
    K = len(rows) - 1
    for k in range(K):
        row = rows[k]
        if not all(math.isfinite(v) for v in row):
            continue
        s, b = _score_band(rule, row)
        e = th["exits"][k]
        if s >= e["t"] and e["L"] <= b <= e["U"]:
            return k, _argmax(row)
    row = rows[K]
    if all(math.isfinite(v) for v in row):
        _, b = _score_band(rule, row)
        if th["final"]["L"] <= b <= th["final"]["U"]:
            return K, _argmax(row)
    return -1, -1


def compare_with_reference(rule, logits_list, th, exit_id, pred, n=500):
    """Run the slow loop on the first n samples; return the number of mismatching samples."""
    n = min(n, logits_list[0].shape[0])
    rows_all = [z[:n].float().cpu().tolist() for z in logits_list]
    mismatches = 0
    for i in range(n):
        e, p = route_one(rule, [r[i] for r in rows_all], th)
        if e != int(exit_id[i]) or p != int(pred[i]):
            mismatches += 1
    return mismatches, n
