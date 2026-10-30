"""Step 3B: deeper analysis of a finished fault campaign (margin_band + energy) of one model.

Per (rule, location, k), from the campaign CSVs:
  coverage            = 100 - flag rate                    (share of clean-correct images still answered)
  selective accuracy  = correct / accepted                 (accuracy among answered images)
  detection ratio     = flagged / (flagged + SDC), pooled  (share of fault-affected images that were caught)
  median SDC, fraction of runs with any SDC, fraction of runs with any effect (SDC or flag)
Per exponent bit, from the k=1 runs (exactly one fault): mean SDC/flag, fraction of runs with any SDC / any effect.
Exposure per location: share of all faultable weights, and the exposure-weighted SDC (sum share * mean SDC).
Paired margin vs energy: every logged fault set (from the margin campaign) is re-applied, logits are computed once on
the union of both clean-correct subsets, and both rules are scored on their INTERSECTION; Wilcoxon signed-rank on the
per-run SDC (and flag) rates per location and per (location, k).
Validation: the re-applied faults must reproduce the margin campaign's recorded SDC / flag counts exactly.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from faults.evaluator import SubsetEvaluator
from faults.inject import EXP_BITS, fault_locations, identical_to, snapshot
from models.searched_exit_model import load_searched_model
from utils.calibrate import thresholds_for
from utils.exit_conditions import route
from utils.runmeta import add_common_args, ckpt_path, output_path, results_path, run_meta
from utils.seed import set_seed

RULES = ("margin_band", "energy")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})


def load_campaign(rule, args):
    frames = [pd.read_csv(p) for p in sorted(results_path("x", args.quick, args.tag).parent.glob(f"faults_{rule}_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no campaign CSVs for rule {rule}")
    return pd.concat(frames, ignore_index=True)


def csv_metrics(df):
    rows = []
    for (rule, loc, k), g in df.groupby(["rule", "location", "k"]):
        accepted = 100.0 - g.flag_rate
        sel = np.where(accepted > 0, 100.0 * g.correct_rate / accepted.replace(0, np.nan), np.nan)
        affected = (g.n_flagged + g.n_sdc).sum()
        rows.append({"rule": rule, "location": loc, "k": k, "runs": len(g),
                     "coverage_mean": accepted.mean(), "coverage_min": accepted.min(),
                     "selective_acc_mean": np.nanmean(sel), "selective_acc_min": np.nanmin(sel),
                     "detection_ratio_pooled": 100.0 * g.n_flagged.sum() / affected if affected else np.nan,
                     "sdc_mean": g.sdc_rate.mean(), "sdc_median": g.sdc_rate.median(), "sdc_max": g.sdc_rate.max(),
                     "pct_runs_any_sdc": 100.0 * (g.n_sdc > 0).mean(),
                     "pct_runs_any_effect": 100.0 * ((g.n_sdc + g.n_flagged) > 0).mean()})
    return pd.DataFrame(rows)


def per_bit_k1(df):
    d = df[df.k == 1].copy()
    d["bit"] = [json.loads(f)[0]["bit"] for f in d.faults]
    rows = []
    for (rule, loc, bit), g in d.groupby(["rule", "location", "bit"]):
        rows.append({"rule": rule, "location": loc, "bit": bit, "runs": len(g), "sdc_mean": g.sdc_rate.mean(),
                     "flag_mean": g.flag_rate.mean(), "pct_runs_any_sdc": 100.0 * (g.n_sdc > 0).mean(),
                     "pct_runs_any_effect": 100.0 * ((g.n_sdc + g.n_flagged) > 0).mean()})
    return pd.DataFrame(rows)


def exposure(metrics, sizes):
    total = sum(sizes.values())
    rows = []
    for (rule, k), g in metrics.groupby(["rule", "k"]):
        w = {r.location: r.sdc_mean for r in g.itertuples()}
        row = {"rule": rule, "k": k,
               "exposure_weighted_sdc": sum(sizes[l] / total * w.get(l, 0.0) for l in sizes)}
        for l in sizes:
            row[f"share_{l}"] = 100.0 * sizes[l] / total
            row[f"contrib_{l}"] = sizes[l] / total * w.get(l, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    nz = int((d != 0).sum())
    if nz == 0:
        return float("nan"), 1.0, 0
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue), nz


def paired_replay(args, dfs, device):
    """Re-apply each margin-campaign fault set; score both rules on the intersection subset."""
    calibs = {}
    for rule in RULES:
        with open(output_path(f"calib_{rule}.json", args.quick, args.tag)) as f:
            calibs[rule] = thresholds_for(json.load(f), args.alpha)
    cc = {}
    for rule in RULES:
        with open(results_path(f"clean_correct_{rule}.json", args.quick, args.tag)) as f:
            cc[rule] = json.load(f)["indices"]
    ra, rb = RULES
    union = sorted(set(cc[ra]) | set(cc[rb]))
    pos = {t: i for i, t in enumerate(union)}
    sub = {r: torch.tensor([pos[t] for t in cc[r]], device=device) for r in RULES}
    inter = sorted(set(cc[ra]) & set(cc[rb]))
    inter_idx = torch.tensor([pos[t] for t in inter], device=device)

    model, meta = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick),
                                      ckpt_path(args.ee_ckpt, args.quick))
    model = model.to(device).eval()
    K = len(meta["positions"])
    _, _, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size, seed=args.seed,
                                                num_workers=args.num_workers, quick=args.quick)
    xs, ys = zip(*test_loader)
    images, labels = torch.cat(xs), torch.cat(ys).to(device)
    ev = SubsetEvaluator(model, images[union].to(device), args.eval_batch)
    lab = labels[union]
    locs = fault_locations(model)
    params = {n: p for tensors in locs.values() for n, p in tensors}
    snap = snapshot(locs)
    sizes = {l: sum(p.numel() for _, p in t) for l, t in locs.items()}

    def score(rule, logits, idx):
        exit_id, pred = route(rule, [z[idx] for z in logits], calibs[rule])
        flagged = exit_id == -1
        sdc = ~flagged & (pred != lab[idx])
        return int(sdc.sum()), int(flagged.sum())

    m = dfs[ra].set_index(["location", "k", "run"])
    e = dfs[rb].set_index(["location", "k", "run"])
    rows, repro_bad, repro_e_bad, same_faults, nonfinite = [], 0, 0, 0, 0
    for (loc, k, run), r in m.sort_index().iterrows():
        faults = json.loads(r.faults)
        for f in faults:
            flat = params[f["layer"]].data.view(-1)
            assert flat[f["index"]].item() == f["old"] or np.isnan(f["old"]), f
            flat.view(torch.int32)[f["index"]] ^= (1 << f["bit"])
        logits = ev.logits(loc)
        for f in reversed(faults):
            params[f["layer"]].data.view(-1).view(torch.int32)[f["index"]] ^= (1 << f["bit"])
        if not all(bool(torch.isfinite(z).all()) for z in logits):
            nonfinite += 1
        s_m, f_m = score(ra, logits, sub[ra])
        repro_bad += int(s_m != r.n_sdc or f_m != r.n_flagged)
        er = e.loc[(loc, k, run)]
        identical = er.faults == r.faults
        same_faults += int(identical)
        if identical:
            s_e, f_e = score(rb, logits, sub[rb])
            repro_e_bad += int(s_e != er.n_sdc or f_e != er.n_flagged)
        n = inter_idx.numel()
        rows_m, rows_e = score(ra, logits, inter_idx), score(rb, logits, inter_idx)
        rows.append({"location": loc, "k": k, "run": run, "n_intersection": n, "same_faults_both_rules": identical,
                     "sdc_a": 100.0 * rows_m[0] / n, "sdc_b": 100.0 * rows_e[0] / n,
                     "flag_a": 100.0 * rows_m[1] / n, "flag_b": 100.0 * rows_e[1] / n})
    undo_ok = identical_to(locs, snap)
    status = {"runs": len(rows), "margin_reproduction_mismatches": repro_bad,
              "energy_reproduction_mismatches_on_identical_fault_runs": repro_e_bad,
              "runs_with_identical_faults_in_both_campaigns": same_faults, "nonfinite_replays": nonfinite,
              "undo_ok": undo_ok, "n_intersection": len(inter), "rule_a": ra, "rule_b": rb, "n_rule_a": len(cc[ra]),
              "n_rule_b": len(cc[rb]), "location_sizes": sizes}
    return pd.DataFrame(rows), status


def save(fig, name, args):
    for ext in ("pdf", "png"):
        fig.savefig(results_path(f"figures/{name}.{ext}", args.quick, args.tag), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figures(metrics, bits, paired, wil, args):
    for rule in RULES:
        d = metrics[metrics.rule == rule]
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for loc, color in zip(["G1", "G2", "G3", "HEADS"], SERIES):
            g = d[d.location == loc].sort_values("k")
            if len(g):
                ax.plot(g.k, g.detection_ratio_pooled, color=color, linewidth=2, marker="o", markersize=5, label=loc)
        ax.set_ylim(0, 102)
        ax.set_xlabel("k (faulty weights)")
        ax.set_ylabel("detection ratio (%) = flagged / (flagged + SDC)")
        ax.set_title(f"Fault-affected images that were caught, rule = {rule}", fontsize=10)
        ax.legend()
        save(fig, f"analysis_detection_ratio_{rule}", args)

        b = bits[bits.rule == rule]
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        locs = [l for l in ["G1", "G2", "G3", "HEADS"] if l in set(b.location)]
        width = 0.8 / max(len(locs), 1)
        for i, (loc, color) in enumerate(zip(locs, SERIES)):
            g = b[b.location == loc].sort_values("bit")
            ax.bar(g.bit + (i - (len(locs) - 1) / 2) * width, g.pct_runs_any_effect, width=width, color=color,
                   edgecolor=SURFACE, linewidth=1, label=loc)
        ax.set_xticks(list(EXP_BITS))
        ax.set_xlabel("flipped exponent bit (k = 1)")
        ax.set_ylabel("% of runs with any effect")
        ax.set_title(f"Single-fault effect by bit, rule = {rule}", fontsize=10)
        ax.legend()
        save(fig, f"analysis_k1_per_bit_{rule}", args)

    if paired is not None:
        g = paired.groupby("location")[["sdc_a", "sdc_b"]].mean()
        locs = [l for l in ["G1", "G2", "G3", "HEADS"] if l in g.index]
        x = np.arange(len(locs))
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.bar(x - 0.2, g.loc[locs, "sdc_a"], width=0.4, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
               label=RULES[0])
        ax.bar(x + 0.2, g.loc[locs, "sdc_b"], width=0.4, color=SERIES[1], edgecolor=SURFACE, linewidth=2,
               label=RULES[1])
        pv = wil[wil.k == "all"].set_index("location")["sdc_p"]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\np={pv.get(l, float('nan')):.2g}" for l in locs])
        ax.set_ylabel("mean SDC rate (%), all k, same faults")
        ax.set_title(f"{RULES[0]} vs {RULES[1]} on the shared clean-correct subset", fontsize=10)
        ax.legend()
        save(fig, "analysis_paired_sdc_a_vs_energy", args)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--ee-ckpt", default="ee_old16_23.pth")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eval-batch", type=int, default=1000)
    parser.add_argument("--rule-a", default="margin_band", help="deployed rule (its fault sets are replayed)")
    parser.add_argument("--rule-b", default="energy", help="comparison rule")
    args = parser.parse_args()
    global RULES
    RULES = (args.rule_a, args.rule_b)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dfs = {r: load_campaign(r, args) for r in RULES}
    df = pd.concat(dfs.values(), ignore_index=True)
    metrics = csv_metrics(df)
    bits = per_bit_k1(df)

    paired, status = paired_replay(args, dfs, device)
    sizes = status["location_sizes"]
    expo = exposure(metrics, sizes)

    wrows = []
    for loc, g in paired.groupby("location"):
        for kk, gg in [("all", g)] + [(str(k), x) for k, x in g.groupby("k")]:
            s_stat, s_p, s_n = wilcoxon(gg.sdc_a, gg.sdc_b)
            f_stat, f_p, f_n = wilcoxon(gg.flag_a, gg.flag_b)
            wrows.append({"location": loc, "k": kk, "runs": len(gg), "sdc_a_mean": gg.sdc_a.mean(),
                          "sdc_b_mean": gg.sdc_b.mean(), "sdc_W": s_stat, "sdc_p": s_p, "sdc_nonzero": s_n,
                          "flag_a_mean": gg.flag_a.mean(), "flag_b_mean": gg.flag_b.mean(),
                          "flag_W": f_stat, "flag_p": f_p, "flag_nonzero": f_n})
    wil = pd.DataFrame(wrows)

    for name, d in [("analysis_metrics", metrics), ("analysis_per_bit_k1", bits), ("analysis_exposure", expo),
                    ("analysis_wilcoxon", wil), ("paired_rules", paired)]:
        d.insert(0, "seed", args.seed)
        d.to_csv(results_path(f"{name}.csv", args.quick, args.tag), index=False)
    with open(results_path("analysis_tables.md", args.quick, args.tag), "w") as f:
        for name, d in [("Per (rule, location, k)", metrics), ("k = 1 by bit", bits), ("Exposure", expo),
                        ("Wilcoxon margin vs energy (intersection)", wil)]:
            f.write(f"## {name}\n\n{d.round(3).to_markdown(index=False)}\n\n")
    figures(metrics, bits, paired, wil, args)

    status["ok"] = (status["margin_reproduction_mismatches"] == 0 and status["nonfinite_replays"] == 0
                    and status["undo_ok"] and status["energy_reproduction_mismatches_on_identical_fault_runs"] == 0)
    with open(results_path("analysis_status.json", args.quick, args.tag), "w") as f:
        json.dump(run_meta(args, **status), f, indent=2)
    print(json.dumps(status, indent=2), flush=True)
    print(wil[wil.k == "all"].round(4).to_string(index=False), flush=True)
    if not status["ok"]:
        raise SystemExit("analysis validation failed")


if __name__ == "__main__":
    main()
