"""Offline evaluation of every exit rule / baseline on the STORED campaign logits (40_campaign_store.py), so all
methods see identical faults. Calibration: clean VAL logits only (full 5,000 val images; int8 -> int8 clean val);
MENDNet faulty-tuned: (gamma, tau) grid on the VALIDATION exponent fault runs (never test).

Per tag (results/campaigns/<tag>/) writes results/offline/<tag>/ :
  clean_methods.csv       clean TEST per method: accuracy, false-alarm rate (FAR), FLOPs saved, clean-correct size
  runs_<fm>.csv.gz        one row per (fault file, run, method): SDC / flag / masked on the method's clean-correct
                          subset, all-image accuracy, mitigated accuracy, flags, image precision, FLOPs, exit counts,
                          run-level detection (3x FAR and binomial p<0.01), detection latency (windows 100 / 500)
  summary_<fm>.csv        per method x location (and ALL): means over runs, pooled detection ratio / precision,
                          run-level TPR on critical runs, FPR on benign runs, median latency
  per_bit_<fm>.csv / per_layer_<fm>.csv   k = 1 breakdown (exponent / int8) for the main methods
  calibrations.json       every calibration used
  reproduction.json       (tags nas / old1623) margin_band with the SOURCE calibration vs the source campaign counts
Definitions (DECISIONS.md P-series):
  clean-correct subset S_m: clean test images the method routes correctly without a flag (as in the source campaign);
  SDC = accepted and wrong on S_m; detection ratio = flagged / (flagged + SDC) on S_m;
  corrupted image = non-finite logits or an argmax change at any exit vs clean; precision = flagged & corrupted / flagged;
  M0 = the unprotected early-exit model (margin thresholds without any band, no final gate); M0 drop = clean - faulty
  all-image accuracy; critical run: M0 drop >= 1.0 point; benign run: M0 drop < 0.1 point;
  run detected: flag rate > 3 x clean FAR, or binomial P(X >= flags | N, FAR) < 0.01 (FAR = 0 -> any flag);
  latency: first non-overlapping window (W images, test order) whose flag count is significant at p < 0.01.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import SRC_OUTPUT_DIR, SRC_RESULTS_DIR
from utils import rules2
from utils.calibrate import thresholds_for
from utils.exit_conditions import route as route_old
from utils.flops import TOTAL_BASE_FLOPS, exit_costs
from utils.logit_store import load_clean_logits
from utils.paths import safe_path

CAMP = Path("/home1/jtt/early_exit_final/stage2_paper/results/campaigns")
OUT = Path("/home1/jtt/early_exit_final/stage2_paper/results/offline")
ALPHAS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
SWEEP_RULES = ["margin_band", "maxlogit_band", "energy_band", "msp_low", "energy_low", "margin_low", "entropy_low",
               "margin_multiband", "margin_band_agree"]
MAIN = ["plain_dnn", "sdn_msp", "msp_low@1", "energy_low@1", "energy_band@1", "maxlogit_band@1", "margin_band@1",
        "margin_multiband@1", "margin_band_agree@1", "mendnet_clean@1", "mendnet_faulty_val"]
WINDOWS = (100, 500)
CRIT, BENIGN = 1.0, 0.1
SOURCE = {"nas": (SRC_OUTPUT_DIR / "nas" / "calib_margin_band.json", SRC_RESULTS_DIR / "nas"),
          "old1623": (SRC_OUTPUT_DIR / "calib_margin_band.json", SRC_RESULTS_DIR)}


def as_list(t):                                       # [N, K+1, C] -> [E1..EK, final]
    return [t[:, j] for j in range(t.shape[1])]


def load_clean(d, int8, dev):
    sfx = "_int8" if int8 else ""
    v, t = load_clean_logits(d / f"clean_val{sfx}.pt"), load_clean_logits(d / f"clean_test{sfx}.pt")
    return (as_list(v["logits"].to(dev)), v["labels"].to(dev), as_list(t["logits"].to(dev)), t["labels"].to(dev),
            t["positions"], t["archs"])


def fault_files(d, fm, split):
    out = []
    for f in sorted(d.glob(f"{fm}_{split}_*.npy")):
        parts = f.stem.split("_")
        loc, k = parts[2], int(parts[3][1:] if parts[3].startswith("k") else parts[3][2:])
        out.append((f, loc, k))
    return out


def calibrations(val, yv, val_runs, ablations):
    m = {}
    for r, spec in rules2.RULES.items():
        if spec.kind == "plain":
            m["plain_dnn"] = (rules2.calibrate(r, val, yv), 1.0)
        elif spec.kind == "standard":
            alphas = ALPHAS if r in SWEEP_RULES else (1.0,)
            cal = rules2.calibrate(r, val, yv, alphas=alphas)
            for a in alphas:
                m[r if spec.final == "none" else f"{r}@{a:g}"] = (cal, a)
    for a in ALPHAS:
        m[f"mendnet_clean@{a:g}"] = (rules2.calibrate_mendnet_clean(val, yv, alpha=a), a)
    if val_runs:
        m["mendnet_faulty_val"] = (rules2.calibrate_mendnet_faulty(val_runs, yv), 1.0)
    if ablations:
        for sides in ("upper", "lower"):
            m[f"abl_band_{sides}@1"] = (rules2.calibrate("margin_band", val, yv, band_sides=sides), 1.0)
        m["abl_band_none@1"] = (rules2.calibrate("margin_band", val, yv, use_band=False), 1.0)
        for s in (0.0, 1.0):
            m[f"abl_slack{s:g}@1"] = (rules2.calibrate("margin_band", val, yv, slack=s), 1.0)
    m0 = rules2.calibrate("margin_band", val, yv, use_band=False)
    m0["final"] = {"1.0": {"gate": "none"}}
    m["M0_unprotected"] = (m0, 1.0)
    return m


def cost_vector(positions, archs, agree, dev):
    c = exit_costs(positions, archs)
    K = len(positions)
    idx = [min(k + 1, K) if agree else k for k in range(K)] + [K, K]     # last entry: flagged (-1) -> full cost
    return torch.tensor([float(c[i]) for i in idx], device=dev, dtype=torch.float64)


def sig_count(n, far):
    """Smallest count c with P(X >= c | n, far) < 0.01 (far = 0 -> 1)."""
    if far <= 0:
        return 1
    c = np.arange(n + 2)
    ok = binom.sf(c - 1, n, far) < 0.01
    return int(c[ok.argmax()]) if ok.any() else n + 1


def per_run(e, p, mt, y, S, corr, costs, far, K):
    """e/p/mt: [R, N] routed outputs. Returns a dict of numpy arrays over runs."""
    R, N = e.shape
    flag = e == -1
    acc = ~flag & (p == y)
    sdc = ~flag & (p != y) & S
    flS = flag & S
    nS = max(int(S.sum()), 1)
    r = {"sdc_rate": 100 * sdc.sum(1).double() / nS, "flag_rate": 100 * flS.sum(1).double() / nS,
         "masked_rate": 100 * (acc & S).sum(1).double() / nS,
         "n_sdc": sdc.sum(1), "n_flag_S": flS.sum(1),
         "acc_all": 100 * acc.double().mean(1), "flag_all": 100 * flag.double().mean(1),
         "mitig_acc": 100 * (mt == y).double().mean(1),
         "n_flag": flag.sum(1), "n_flag_corrupted": (flag & corr).sum(1), "n_corrupted": corr.sum(1),
         "saved_pct": 100 * (1 - costs[torch.where(e < 0, K + 1, e)].mean(1) / TOTAL_BASE_FLOPS)}
    for j in range(K + 1):
        r[f"exit_{j}"] = (e == j).sum(1)
    x = r["n_flag"].cpu().numpy()
    r["det_3x"] = torch.as_tensor(x > 3 * far * N) if far > 0 else torch.as_tensor(x > 0)
    r["det_binom"] = torch.as_tensor(binom.sf(x - 1, N, far) < 0.01) if far > 0 else torch.as_tensor(x > 0)
    for W in WINDOWS:
        c = sig_count(W, far)
        nw = N // W
        win = flag[:, :nw * W].reshape(R, nw, W).sum(2) >= c
        first = torch.where(win.any(1), win.float().argmax(1) + 1, torch.full((R,), math.inf, device=e.device))
        r[f"latency_{W}"] = first * W
    return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in r.items()}


def evaluate_tag(tag, a, dev):
    d = CAMP / ("quick" if a.quick else "") / tag
    out = OUT / ("quick" if a.quick else "") / tag
    fms = [fm for fm in ("exponent", "int8", "mendnet") if list(d.glob(f"{fm}_test_*.npy"))]
    status = {}
    for fm in fms:
        int8 = fm == "int8"
        val, yv, test, yt, positions, archs = load_clean(d, int8, dev)
        K = len(test) - 1
        val_runs = []
        for f, _, _ in fault_files(d, "int8" if int8 else "exponent", "val"):
            arr = torch.from_numpy(np.load(f)).to(dev)
            val_runs += [as_list(arr[r]) for r in range(arr.shape[0])]
        meths = calibrations(val, yv, val_runs, ablations=(a.ablations and tag in a.ablations))
        del val_runs
        cal_dump = {n: c for n, (c, _) in meths.items()}
        with open(safe_path(out / f"calibrations_{fm}.json"), "w") as fh:
            json.dump(cal_dump, fh, indent=1, default=float)
        clean_am = torch.stack([torch.nan_to_num(z, nan=-math.inf).argmax(-1) for z in test], 1)

        # clean test per method
        clean_rows, info = [], {}
        for name, (cal, al) in meths.items():
            e, p, mt = rules2.route(cal, test, al)
            S = (e >= 0) & (p == yt)
            far = float((e == -1).double().mean())
            costs = cost_vector(positions, archs, rules2.RULES[cal["rule"]].agree, dev)
            info[name] = (S, far, costs)
            clean_rows.append({"tag": tag, "fault_model": fm, "method": name, "rule": cal["rule"], "alpha": al,
                               "acc_all": 100 * float(S.double().mean()), "far_pct": 100 * far,
                               "mitig_acc": 100 * float((mt == yt).double().mean()), "n_clean_correct": int(S.sum()),
                               "saved_pct": 100 * (1 - float(costs[torch.where(e < 0, K + 1, e)].mean())
                                                   / TOTAL_BASE_FLOPS),
                               **{f"exit_{j}": int((e == j).sum()) for j in range(K + 1)},
                               **({"gamma": cal["gamma"], "tau": cal["tau"]} if cal.get("kind") == "mendnet" else {})})
        pd.DataFrame(clean_rows).to_csv(safe_path(out / f"clean_methods_{fm}.csv"), index=False)
        m0_clean = next(r["acc_all"] for r in clean_rows if r["method"] == "M0_unprotected")
        plain_clean = next(r["acc_all"] for r in clean_rows if r["method"] == "plain_dnn")

        rows = []
        for f, loc, k in fault_files(d, fm, "test"):
            arr = torch.from_numpy(np.load(f)).to(dev)                    # [R, N, K+1, C]
            R, N = arr.shape[:2]
            flat = [arr[:, :, j].reshape(R * N, -1) for j in range(K + 1)]
            am = torch.nan_to_num(arr, nan=-math.inf).argmax(-1)          # [R, N, K+1]
            corr = (am != clean_am.unsqueeze(0)).any(-1) | ~torch.isfinite(arr).all(-1).all(-1)
            base = None
            for name, (cal, al) in meths.items():
                e, p, mt = (t.view(R, N) for t in rules2.route(cal, flat, al))
                S, far, costs = info[name]
                r = per_run(e, p, mt, yt.unsqueeze(0), S.unsqueeze(0), corr, costs, far, K)
                if name == "M0_unprotected":
                    base = m0_clean - r["acc_all"]
                df = pd.DataFrame(r)
                df.insert(0, "run", np.arange(1, R + 1))
                df.insert(0, "method", name)
                df.insert(0, "k", k)
                df.insert(0, "location", loc)
                if name == "plain_dnn":
                    plain_drop = plain_clean - r["acc_all"]
                rows.append(df)
            for df in rows[-len(meths):]:
                df["m0_drop"] = base
                df["plain_drop"] = plain_drop
            print(f"[{tag} {fm}] {f.name}: {R} runs x {len(meths)} methods", flush=True)
            del arr, flat, am
            torch.cuda.empty_cache()
        runs = pd.concat(rows, ignore_index=True)
        runs.insert(0, "fault_model", fm)
        runs.insert(0, "tag", tag)
        runs.to_csv(safe_path(out / f"runs_{fm}.csv.gz"), index=False, compression="gzip")
        summarise(runs, out, fm)
        if fm in ("exponent", "int8"):
            breakdown(runs, d, out, fm)
        status[fm] = {"files": len(fault_files(d, fm, "test")), "rows": len(runs), "methods": len(meths)}
    if tag in SOURCE and "exponent" in fms and not a.quick:
        status["reproduction"] = reproduce(tag, d, out, dev)
    with open(safe_path(out / "status.json"), "w") as fh:
        json.dump(status, fh, indent=2, default=float)
    return status


def _summ(g):
    crit, ben = g[g.m0_drop >= CRIT], g[g.m0_drop < BENIGN]
    nsf = g.n_sdc.sum() + g.n_flag_S.sum()
    s = {"n_runs": len(g), "sdc_rate": g.sdc_rate.mean(), "sdc_rate_std": g.sdc_rate.std(),
         "flag_rate": g.flag_rate.mean(), "masked_rate": g.masked_rate.mean(), "acc_all": g.acc_all.mean(),
         "mitig_acc": g.mitig_acc.mean(), "flag_all": g.flag_all.mean(), "saved_pct": g.saved_pct.mean(),
         "det_ratio": 100 * g.n_flag_S.sum() / nsf if nsf else float("nan"),
         "precision": 100 * g.n_flag_corrupted.sum() / g.n_flag.sum() if g.n_flag.sum() else float("nan"),
         "n_critical": len(crit), "tpr_3x": 100 * crit.det_3x.mean() if len(crit) else float("nan"),
         "tpr_binom": 100 * crit.det_binom.mean() if len(crit) else float("nan"),
         "n_benign": len(ben), "fpr_3x": 100 * ben.det_3x.mean() if len(ben) else float("nan"),
         "fpr_binom": 100 * ben.det_binom.mean() if len(ben) else float("nan"),
         "max_sdc_rate": g.sdc_rate.max()}
    for W in WINDOWS:
        lat = crit[f"latency_{W}"]
        s[f"crit_detected_{W}"] = 100 * np.isfinite(lat).mean() if len(lat) else float("nan")
        s[f"crit_latency_median_{W}"] = float(np.median(lat[np.isfinite(lat)])) if np.isfinite(lat).any() else float("nan")
    return pd.Series(s)


def summarise(runs, out, fm):
    a = runs.groupby(["method", "location"]).apply(_summ, include_groups=False).reset_index()
    b = runs.groupby(["method"]).apply(_summ, include_groups=False).reset_index()
    b.insert(1, "location", "ALL")
    pd.concat([a, b]).to_csv(safe_path(out / f"summary_{fm}.csv"), index=False)


def breakdown(runs, d, out, fm):
    k1 = []
    for f in sorted(d.glob(f"{fm}_test_*_k1.csv")):
        c = pd.read_csv(f)
        fl = c.faults.map(json.loads)
        k1.append(pd.DataFrame({"location": c.location, "run": c.run, "bit": fl.map(lambda x: x[0]["bit"]),
                                "layer": fl.map(lambda x: x[0]["layer"])}))
    if not k1:
        return
    k1 = pd.concat(k1)
    m = runs[(runs.k == 1) & runs.method.isin(MAIN)].merge(k1, on=["location", "run"])
    cols = dict(n_runs=("run", "size"), sdc_rate=("sdc_rate", "mean"), flag_rate=("flag_rate", "mean"),
                acc_all=("acc_all", "mean"), mitig_acc=("mitig_acc", "mean"))
    m.groupby(["method", "bit"]).agg(**cols).reset_index().to_csv(safe_path(out / f"per_bit_{fm}.csv"), index=False)
    m.groupby(["method", "location", "layer"]).agg(**cols).reset_index().to_csv(
        safe_path(out / f"per_layer_{fm}.csv"), index=False)


def reproduce(tag, d, out, dev):
    """margin_band at alpha 1 with the SOURCE calibration on our stored logits vs the source campaign's recorded
    n_sdc / n_flagged, on runs whose applied faults (layer, index, bit) are identical."""
    cal_path, res_dir = SOURCE[tag]
    th = thresholds_for(json.load(open(cal_path)), 1.0)
    t = load_clean_logits(d / "clean_test.pt")
    test, yt = as_list(t["logits"].to(dev)), t["labels"].to(dev)
    e0, p0 = route_old("margin_band", test, th)
    S = (e0 >= 0) & (p0 == yt)
    tot = dict(runs=0, identical=0, exact=0, abs_diff_sdc=0, abs_diff_flag=0, images=0)
    for f, loc, k in fault_files(d, "exponent", "test"):
        src = pd.read_csv(res_dir / f"faults_margin_band_{loc}.csv")
        src = src[(src.k == k)].set_index("run")
        ours = pd.read_csv(str(f)[:-4] + ".csv").set_index("run")
        arr = torch.from_numpy(np.load(f)).to(dev)
        for run in ours.index:
            tot["runs"] += 1
            key = lambda js: [(x["layer"], x["index"], x["bit"]) for x in json.loads(js)]
            if run not in src.index or key(src.loc[run, "faults"]) != key(ours.loc[run, "faults"]):
                continue
            tot["identical"] += 1
            e, p = route_old("margin_band", as_list(arr[run - 1][S]), th)
            n_flag, n_sdc = int((e == -1).sum()), int(((e >= 0) & (p != yt[S])).sum())
            ds, df_ = abs(n_sdc - int(src.loc[run, "n_sdc"])), abs(n_flag - int(src.loc[run, "n_flagged"]))
            tot["exact"] += int(ds == 0 and df_ == 0)
            tot["abs_diff_sdc"] += ds
            tot["abs_diff_flag"] += df_
            tot["images"] += int(S.sum())
        del arr
    tot["n_subset_ours"] = int(S.sum())
    tot["n_subset_source"] = int(pd.read_csv(res_dir / "faults_margin_band_G1.csv").n_subset.iloc[0])
    with open(safe_path(out / "reproduction.json"), "w") as fh:
        json.dump(tot, fh, indent=2)
    print(f"[reproduction {tag}] {json.dumps(tot)}", flush=True)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--ablations", nargs="*", default=["nas"], help="tags that also get the band/slack ablations")
    ap.add_argument("--only-reproduce", action="store_true", help="just the source-reproduction check (early in chain)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tag in a.tags:
        if a.only_reproduce:
            reproduce(tag, CAMP / tag, OUT / tag, dev)
            continue
        st = evaluate_tag(tag, a, dev)
        print(f"[done {tag}] {json.dumps(st, default=float)}", flush=True)


if __name__ == "__main__":
    main()
