"""Rule-independent fault campaign that STORES the logits of every exit, so all exit rules / baselines / mitigation /
detection metrics are evaluated offline on identical faults (41_offline_eval.py).

Model: --ee-ckpt (path; source checkpoints are read-only) on the frozen backbone. Images: the whole --split
(test: 10,000, val: 5,000). Outputs under results/campaigns/<tag>/ :
  clean_<split>.pt               clean logits [N, K+1, C] (float32), labels, positions, archs
  <fm>_<split>_<loc>_k<k>.npy    float32 [runs, N, K+1, C]   (fm = exponent | int8 | mendnet)
  <fm>_<split>_<loc>_k<k>.csv    one row per run: seed, faults (layer, index, bit, old, new), rejections, undo check
Fault models:
  exponent : FP32 weights, exponent bits 23-30, k = 1..8 distinct weights (one flip each), uniform within the
             location (G1 / G2 / G3 / HEADS), faults making any weight or logit non-finite on the split are rejected,
             logged and redrawn. Same seeds as the source campaigns (fault_seed); val uses an offset of 50,000,000.
  int8     : weight-only symmetric per-tensor INT8 (codes in [-127, 127], weights = code * scale); a fault flips one
             of the 8 bits of one weight's two's-complement code (bit 7 = sign); k = 1..8 distinct weights.
  mendnet  : MENDNet's model - sign-bit (bit 31) flips of FR = 10/30/50 % of the weights of one layer group:
             FP1 final classifier, FP2 internal classifier (exit-2 head), FP3 first conv after exit 1, FP4 first conv.
After every run all weights are restored and checked bit-identical. Resumable per (fm, loc, k) file.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import BACKBONE_CKPT
from dataset.datamodule import get_cifar10_dataloaders
from faults.evaluator import SubsetEvaluator
from faults.inject import LOCATION_IDS, LocationSampler, checksum, fault_locations, fault_seed, identical_to, snapshot
from models.searched_exit_model import load_searched_model
from utils.paths import safe_path
from utils.seed import set_seed

OUT_ROOT = Path("/home1/jtt/early_exit_final/stage2_paper/results/campaigns")
MAX_ATTEMPTS = 1000


def stack(logits):
    return torch.stack(logits, 1).float().cpu().numpy()          # [N, K+1, C]


def all_finite(logits):
    return all(bool(torch.isfinite(z).all()) for z in logits)


class Int8Weights:
    """Weight-only INT8: every faultable tensor is replaced by code * scale (code int8 kept on the device)."""

    def __init__(self, locations):
        self.codes, self.scales = {}, {}
        for tensors in locations.values():
            for name, p in tensors:
                s = p.data.abs().max().clamp_min(1e-12) / 127.0
                c = torch.round(p.data / s).clamp(-127, 127).to(torch.int8)
                p.data.copy_(c.float() * s)
                self.codes[name], self.scales[name] = c, s

    def flip(self, name, param, index, bit):
        flat = param.data.view(-1)
        codes = self.codes[name].view(-1)
        old_c = int(codes[index])
        u = (old_c & 0xFF) ^ (1 << bit)
        new_c = u - 256 if u >= 128 else u
        old_w = flat[index].item()
        codes[index] = new_c
        flat[index] = new_c * self.scales[name]
        return {"old": old_w, "new": flat[index].item(), "old_code": old_c, "new_code": new_c}

    def unflip(self, name, param, index, old_code, old_w):
        self.codes[name].view(-1)[index] = old_code
        param.data.view(-1)[index] = old_w


def mendnet_groups(model):
    p = list(model.positions)
    first_after = next(i for i in range(p[0] + 1, len(model.features)) if isinstance(model.features[i], nn.Conv2d))
    lin = [(f"classifier.{n}.weight", m.weight) for n, m in model.classifier.named_modules() if isinstance(m, nn.Linear)]
    mid = min(1, len(p) - 1)
    head = [(f"exit_heads.{mid}.{n}.weight", m.weight) for n, m in model.exit_heads[mid].named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]
    return {"FP1": lin, "FP2": head,
            "FP3": [(f"features.{first_after}.weight", model.features[first_after].weight)],
            "FP4": [("features.0.weight", model.features[0].weight)]}


def load_split(split, quick):
    tr, va, te = get_cifar10_dataloaders(batch_size=500, seed=42, num_workers=4, quick=quick)
    xs, ys = zip(*(te if split == "test" else va))
    return torch.cat(xs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ee-ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fault-model", default="exponent", choices=["exponent", "int8", "mendnet"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--locations", nargs="+", default=None)
    ap.add_argument("--fr", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.kmax, a.runs = 2, 2
    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT_ROOT / ("quick" if a.quick else "") / a.tag
    model, meta = load_searched_model(BACKBONE_CKPT if not a.quick else
                                      Path(str(BACKBONE_CKPT).replace("/checkpoints/", "/checkpoints/quick/")),
                                      a.ee_ckpt)
    model = model.to(dev).eval()
    locs = fault_locations(model)
    int8 = Int8Weights(locs) if a.fault_model == "int8" else None
    snap = snapshot(locs)
    sum0 = checksum(locs)

    # clean logits for both splits (calibration uses val, evaluation uses test); int8 -> quantised model
    for split in ("val", "test"):
        f = out / f"clean_{split}{'_int8' if int8 else ''}.pt"
        if not f.is_file():
            x, y = load_split(split, a.quick)
            ev = SubsetEvaluator(model, x.to(dev), 1000)
            torch.save({"logits": torch.from_numpy(stack(ev.clean_logits)), "labels": y,
                        "positions": meta["positions"], "archs": meta["archs"], "rule": meta.get("rule"),
                        "ee_ckpt": str(a.ee_ckpt), "int8": bool(int8)}, safe_path(f))
            acc = (ev.clean_logits[-1].argmax(-1).cpu() == y).float().mean().item() * 100
            print(f"[clean {split}{' int8' if int8 else ''}] n={len(y)} final-classifier acc {acc:.2f}%", flush=True)
            del ev
            torch.cuda.empty_cache()

    x, y = load_split(a.split, a.quick)
    ev = SubsetEvaluator(model, x.to(dev), 1000)
    offset = 50_000_000 if a.split == "val" else 0
    t_start = time.time()

    if a.fault_model == "mendnet":
        groups = mendnet_groups(model)
        for fp, tensors in groups.items():
            names = [n for n, _ in tensors]
            params = [p for _, p in tensors]
            sizes = np.array([p.numel() for p in params])
            cum = np.cumsum(sizes)
            for fr in a.fr:
                stem = out / f"mendnet_{a.split}_{fp}_fr{fr}"
                if Path(str(stem) + ".npy").is_file():
                    continue
                arr, rows = [], []
                for run in range(1, a.runs + 1):
                    seed = 900_000_000 + 1000 * int(fp[2]) * 100 + 1000 * fr + run + offset
                    rng = np.random.default_rng(seed)
                    m = int(round(fr / 100 * cum[-1]))
                    g = rng.choice(int(cum[-1]), size=m, replace=False)
                    t_idx = np.searchsorted(cum, g, side="right")
                    for t in range(len(params)):
                        loc_idx = g[t_idx == t] - (cum[t - 1] if t > 0 else 0)
                        if len(loc_idx):
                            ints = params[t].data.view(-1).view(torch.int32)
                            ii = torch.as_tensor(loc_idx, device=ints.device, dtype=torch.long)
                            ints[ii] ^= torch.tensor(-2 ** 31, dtype=torch.int32, device=ints.device)
                    logits = ev.full_logits()
                    arr.append(stack(logits))
                    for t in range(len(params)):
                        loc_idx = g[t_idx == t] - (cum[t - 1] if t > 0 else 0)
                        if len(loc_idx):
                            ints = params[t].data.view(-1).view(torch.int32)
                            ii = torch.as_tensor(loc_idx, device=ints.device, dtype=torch.long)
                            ints[ii] ^= torch.tensor(-2 ** 31, dtype=torch.int32, device=ints.device)
                    ok = identical_to(locs, snap)
                    rows.append({"fault_model": "mendnet", "position": fp, "fr": fr, "run": run, "seed": seed,
                                 "n_flipped": m, "layers": ";".join(names), "finite": all_finite(logits),
                                 "undo_ok": ok})
                    if not ok:
                        raise RuntimeError("undo failed")
                save_npy(Path(str(stem) + ".npy"), np.stack(arr))
                _write_rows(Path(str(stem) + ".csv"), rows)
                print(f"[mendnet {fp} FR {fr}%] {a.runs} runs | {(time.time() - t_start) / 60:.1f} min", flush=True)
    else:
        names = a.locations or list(locs.keys())
        bits = (0, 8) if int8 else (23, 31)
        for loc in names:
            sampler = LocationSampler(loc, locs[loc])
            for k in range(1, a.kmax + 1):
                stem = out / f"{a.fault_model}_{a.split}_{loc}_k{k}"
                if Path(str(stem) + ".npy").is_file():
                    continue
                arr, rows = [], []
                for run in range(1, a.runs + 1):
                    seed = fault_seed(LOCATION_IDS[loc], k, run) + offset
                    rng = np.random.default_rng(seed)
                    applied, rejected, used = [], [], set()
                    logits, n_att = ev.clean_logits, 0
                    while len(applied) < k:
                        n_att += 1
                        if n_att > MAX_ATTEMPTS:
                            raise RuntimeError(f"{loc} k={k} run={run}: too many attempts")
                        f = sampler.draw(rng)
                        if int8:
                            f["bit"] = int(rng.integers(*bits))
                        if f["global_index"] in used:
                            continue
                        if int8:
                            f.update(int8.flip(f["layer"], sampler.params[f["tensor"]], f["index"], f["bit"]))
                        else:
                            sampler.flip(f)
                        reason = None
                        if not np.isfinite(f["new"]):
                            reason = "weight_nonfinite"
                        else:
                            new = ev.logits(loc)
                            if not all_finite(new):
                                reason = "logits_nonfinite"
                        if reason:
                            if int8:
                                int8.unflip(f["layer"], sampler.params[f["tensor"]], f["index"], f["old_code"], f["old"])
                            else:
                                sampler.unflip(f)
                            rejected.append({**f, "reason": reason})
                            continue
                        used.add(f["global_index"])
                        applied.append(f)
                        logits = new
                    arr.append(stack(logits))
                    for f in reversed(applied):
                        if int8:
                            int8.unflip(f["layer"], sampler.params[f["tensor"]], f["index"], f["old_code"], f["old"])
                        else:
                            sampler.unflip(f)
                    ok = identical_to(locs, snap)
                    keys = ("layer", "index", "bit", "old", "new")
                    rows.append({"fault_model": a.fault_model, "location": loc, "k": k, "run": run, "seed": seed,
                                 "n_attempts": n_att, "n_rejected": len(rejected), "undo_ok": ok,
                                 "faults": json.dumps([{x: f[x] for x in keys} for f in applied]),
                                 "rejected": json.dumps([{**{x: f[x] for x in keys}, "reason": f["reason"]}
                                                         for f in rejected])})
                    if not ok:
                        raise RuntimeError("undo failed")
                save_npy(Path(str(stem) + ".npy"), np.stack(arr))
                _write_rows(Path(str(stem) + ".csv"), rows)
                print(f"[{a.fault_model} {a.split} {loc} k={k}] {a.runs} runs | {(time.time() - t_start) / 60:.1f} min",
                      flush=True)

    ok = checksum(locs) == sum0
    with open(safe_path(out / f"status_{a.fault_model}_{a.split}.json"), "w") as fh:
        json.dump({"tag": a.tag, "fault_model": a.fault_model, "split": a.split, "kmax": a.kmax, "runs": a.runs,
                   "ee_ckpt": str(a.ee_ckpt), "positions": meta["positions"], "checksum_ok": ok,
                   "minutes": round((time.time() - t_start) / 60, 2)}, fh, indent=2)
    print(f"[done] checksum ok: {ok}", flush=True)
    if not ok:
        raise SystemExit("checksum mismatch")


def save_npy(path, arr):
    """Atomic: write <name>.part then rename, so an interrupted write (e.g. disk full) never leaves a truncated file
    that the resume logic would treat as finished."""
    part = safe_path(Path(str(path) + ".part"))
    with open(part, "wb") as fh:
        np.save(fh, arr)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(part, safe_path(path))


def _write_rows(path, rows):
    with open(safe_path(path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
