"""Exact hypervolume for 2-3 minimisation objectives (small point sets) and Pareto filtering."""
import numpy as np


def pareto_mask(points):
    """Boolean mask of non-dominated rows (minimisation)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        if dominated_by.any():
            mask[i] = False
    return mask


def hv2d(points, ref):
    pts = np.asarray([p for p in np.asarray(points, dtype=float) if np.all(p < ref)])
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    area, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(area)


def hv3d(points, ref):
    ref = np.asarray(ref, dtype=float)
    pts = np.asarray([p for p in np.asarray(points, dtype=float).reshape(-1, 3) if np.all(p < ref)])
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 2], kind="stable")]
    vol = 0.0
    for i in range(len(pts)):
        z_next = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
        if z_next > pts[i, 2]:
            vol += hv2d(pts[: i + 1, :2], ref[:2]) * (z_next - pts[i, 2])
    return float(vol)
