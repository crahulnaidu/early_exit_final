"""Batch-1 wall-clock latency with REAL early stopping (the forward pass stops at the exit that accepts).

For every tag (results/campaigns/<tag>/clean_*.pt gives the checkpoint): margin_band at alpha 1 calibrated on the
clean val logits (rules2, same as 41), then per test image (test order):
  plain    : backbone features + classifier (no heads, no checks)       = the unprotected plain VGG-16
  ee_all   : all heads + final classifier computed, no stopping          = detection-only deployment
  ee_stop  : stop at the first accepting exit, else final classifier + final band check
GPU: first --n-gpu test images after 50 warm-up images, torch.cuda.synchronize per image. CPU: first --n-cpu images,
--threads intra-op threads. Writes results/latency/latency.csv (mean / p50 / p95 ms, speed-up vs plain) and
latency_per_image_<tag>.csv; checks that ee_stop predictions equal the offline routed predictions.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import BACKBONE_CKPT
from dataset.datamodule import get_cifar10_dataloaders
from models.searched_exit_model import load_searched_model
from utils import rules2
from utils.logit_store import load_clean_logits
from utils.paths import safe_path

CAMP = Path("/home1/jtt/early_exit_final/stage2_paper/results/campaigns")
OUT = Path("/home1/jtt/early_exit_final/stage2_paper/results/latency")
WARMUP = 50


def as_list(t):
    return [t[:, j] for j in range(t.shape[1])]


class Runner:
    def __init__(self, model, cal):
        self.m, self.cal = model, cal
        self.spec = rules2.RULES[cal["rule"]]
        self.pos = list(model.positions)
        self.fin = cal["final"]["1.0"]

    def _accept(self, z, e):
        s = self.spec
        if not bool(rules2.finite(z)) or float(s.score(z)) < e["t"]:
            return False
        if "L" in e and not (e["L"] <= float(s.band(z)) <= e["U"]):
            return False
        return True

    @torch.no_grad()
    def plain(self, x):
        m = self.m
        for layer in m.features:
            x = layer(x)
        return int(m.classifier(torch.flatten(m.avgpool(x), 1)).argmax(-1))

    @torch.no_grad()
    def ee(self, x, stop):
        m, h, out = self.m, 0, None
        for idx, layer in enumerate(m.features):
            x = layer(x)
            if h < len(self.pos) and idx == self.pos[h]:
                z = m.exit_heads[h](x)
                if out is None and self._accept(z, self.cal["exits"][h]):
                    out = (h, int(z.argmax(-1)))
                    if stop:
                        return out
                h += 1
        z = m.classifier(torch.flatten(m.avgpool(x), 1))
        if out is not None:
            return out
        b = float(self.spec.band(z))
        return (len(self.pos), int(z.argmax(-1))) if bool(rules2.finite(z)) and self.fin["L"] <= b <= self.fin["U"] \
            else (-1, -1)


def timed(fn, xs, dev):
    ts, outs = [], []
    for i, x in enumerate(xs):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = fn(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= WARMUP:
            ts.append(1000 * (time.perf_counter() - t0))
            outs.append(o)
    return np.array(ts), outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n-gpu", type=int, default=2000)
    ap.add_argument("--n-cpu", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_gpu, a.n_cpu = 60, 60
    torch.set_num_threads(a.threads)
    _, _, te = get_cifar10_dataloaders(batch_size=500, seed=42, num_workers=4, quick=a.quick)
    x_all = torch.cat([x for x, _ in te])
    base = BACKBONE_CKPT if not a.quick else Path(str(BACKBONE_CKPT).replace("/checkpoints/", "/checkpoints/quick/"))
    out = OUT / ("quick" if a.quick else "")
    rows = []
    devs = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for tag in a.tags:
        d = CAMP / ("quick" if a.quick else "") / tag
        v, t = load_clean_logits(d / "clean_val.pt"), load_clean_logits(d / "clean_test.pt")
        cal = rules2.calibrate("margin_band", as_list(v["logits"]), v["labels"], alphas=(1.0,))
        e_ref, p_ref, _ = rules2.route(cal, as_list(t["logits"]), 1.0)
        model, meta = load_searched_model(base, t["ee_ckpt"])
        per_img = []
        for dev in devs:
            m = model.to(dev).eval()
            r = Runner(m, cal)
            n = min(WARMUP + (a.n_gpu if dev.type == "cuda" else a.n_cpu), len(x_all))
            xs = [x_all[i:i + 1].to(dev) for i in range(n)]
            res = {}
            for mode, fn in (("plain", r.plain), ("ee_all", lambda x: r.ee(x, False)),
                             ("ee_stop", lambda x: r.ee(x, True))):
                res[mode] = timed(fn, xs, dev)
            ids = np.arange(WARMUP, n)
            got = np.array([o[1] for o in res["ee_stop"][1]])
            ex = np.array([o[0] for o in res["ee_stop"][1]])
            same = float((got == p_ref.numpy()[ids]).mean() * 100)
            same_exit = float((ex == e_ref.numpy()[ids]).mean() * 100)
            for mode, (ts, _) in res.items():
                rows.append({"tag": tag, "device": dev.type, "mode": mode, "n_images": len(ts),
                             "mean_ms": ts.mean(), "p50_ms": np.percentile(ts, 50), "p95_ms": np.percentile(ts, 95),
                             "speedup_vs_plain": res["plain"][0].mean() / ts.mean(),
                             "pred_agreement_with_offline_pct": same if mode == "ee_stop" else float("nan"),
                             "exit_agreement_with_offline_pct": same_exit if mode == "ee_stop" else float("nan"),
                             "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else f"cpu x{a.threads}"})
                per_img.append(pd.DataFrame({"device": dev.type, "mode": mode, "image": ids, "ms": ts}))
            print(f"[{tag} {dev.type}] plain {res['plain'][0].mean():.3f} ms | ee_all {res['ee_all'][0].mean():.3f} | "
                  f"ee_stop {res['ee_stop'][0].mean():.3f} | pred agreement {same:.2f}% exit {same_exit:.2f}%",
                  flush=True)
        pd.concat(per_img).to_csv(safe_path(out / f"latency_per_image_{tag}.csv"), index=False)
    pd.DataFrame(rows).to_csv(safe_path(out / "latency.csv"), index=False)


if __name__ == "__main__":
    main()
