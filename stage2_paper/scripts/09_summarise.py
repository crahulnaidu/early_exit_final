"""Step 3 / J6: summary tables + figures for whatever campaign results exist.

Tables (results/):  fault_summary.csv/.md  per (rule, location, k): runs, mean/std/median/95% CI of correct, SDC,
                    flag and masked rates, % catastrophic runs (SDC > 10 %), rejection rate;
                    rejection_by_bit.csv per (rule, location, bit); clean_results_table.md from outputs/clean_eval.csv.
Figures (results/figures/, PDF + PNG): stacked correct/SDC/flag per location and k; SDC vs k per location
(mean +- std); rejection rate per location and bit; exit distribution clean vs faulty.
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
from scipy import stats

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from faults.inject import EXP_BITS
from utils.runmeta import output_path as _output_path, results_path as _results_path

TAG = ""   # set from --tag in main(): outputs/<tag>/, results/<tag>/


def output_path(name, quick):
    return _output_path(name, quick, TAG)


def results_path(name, quick):
    return _results_path(name, quick, TAG)

# Validated categorical palette (dataviz reference palette, light mode), assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OUTCOMES = [("correct_rate", "Correct (masked)"), ("sdc_rate", "SDC (silent)"), ("flag_rate", "Flagged (detected)")]
METRICS = ["correct_rate", "sdc_rate", "flag_rate", "masked_rate"]
LOC_ORDER = ["G1", "G2", "G3", "G4", "HEADS"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - half, x.mean() + half


def load_runs(quick):
    """Runs of FINISHED campaigns only (campaign_<rule>_status.json exists); partial campaigns are skipped."""
    frames = []
    for path in sorted(results_path("x", quick).parent.glob("faults_*_*.csv")):
        df = pd.read_csv(path)
        rule = df.rule.iloc[0]
        if not results_path(f"campaign_{rule}_status.json", quick).is_file():
            print(f"[skip] {path.name}: campaign '{rule}' not finished ({len(df)} runs so far)", flush=True)
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def summary_table(df):
    rows = []
    for (rule, loc, k), g in df.groupby(["rule", "location", "k"]):
        row = {"rule": rule, "location": loc, "k": k, "runs": len(g), "seed": ",".join(map(str, sorted(g.seed.unique())))}
        for m in METRICS:
            lo, hi = ci95(g[m])
            row.update({f"{m}_mean": g[m].mean(), f"{m}_std": g[m].std(ddof=1) if len(g) > 1 else float("nan"),
                        f"{m}_median": g[m].median(), f"{m}_ci95_lo": lo, f"{m}_ci95_hi": hi})
        row["pct_catastrophic_sdc_gt10"] = 100.0 * (g.sdc_rate > 10).mean()
        att = g[[f"attempts_bit{b}" for b in EXP_BITS]].to_numpy().sum()
        rej = g[[f"rejected_bit{b}" for b in EXP_BITS]].to_numpy().sum()
        row.update({"attempts": int(att), "rejected": int(rej), "rejection_rate_pct": 100.0 * rej / att if att else 0.0,
                    "undo_all_ok": bool(g.undo_ok.all())})
        rows.append(row)
    return pd.DataFrame(rows)


def rejection_by_bit(df):
    rows = []
    for (rule, loc), g in df.groupby(["rule", "location"]):
        for b in EXP_BITS:
            att, rej = int(g[f"attempts_bit{b}"].sum()), int(g[f"rejected_bit{b}"].sum())
            rows.append({"rule": rule, "location": loc, "bit": b, "attempts": att, "rejected": rej,
                         "rejection_rate_pct": 100.0 * rej / att if att else float("nan")})
    return pd.DataFrame(rows)


def ordered_locs(locs):
    return [l for l in LOC_ORDER if l in locs] + sorted(l for l in locs if l not in LOC_ORDER)


def save(fig, name, quick):
    for ext in ("pdf", "png"):
        fig.savefig(results_path(f"figures/{name}.{ext}", quick), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_stacked(summ, rule, quick):
    s = summ[summ.rule == rule]
    locs = ordered_locs(s.location.unique())
    fig, axes = plt.subplots(1, len(locs), figsize=(2.6 * len(locs), 2.8), sharey=True, squeeze=False)
    for ax, loc in zip(axes[0], locs):
        d = s[s.location == loc].sort_values("k")
        bottom = np.zeros(len(d))
        for (m, label), color in zip(OUTCOMES, SERIES):
            ax.bar(d.k, d[f"{m}_mean"], bottom=bottom, color=color, width=0.75, edgecolor=SURFACE, linewidth=2,
                   label=label)
            bottom += d[f"{m}_mean"].to_numpy()
        ax.set_title(loc, fontsize=9)
        ax.set_xlabel("k (faulty weights)")
        ax.set_xticks(d.k)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("% of clean-correct test images (mean)")
    axes[0][0].set_ylim(0, 100)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Outcome per fault location, rule = {rule}", fontsize=10, y=1.05)
    save(fig, f"stacked_outcomes_{rule}", quick)


def fig_sdc_vs_k(summ, rule, quick):
    s = summ[summ.rule == rule]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for loc, color in zip(ordered_locs(s.location.unique()), SERIES):
        d = s[s.location == loc].sort_values("k")
        ax.errorbar(d.k, d.sdc_rate_mean, yerr=d.sdc_rate_std.fillna(0), color=color, linewidth=2, marker="o",
                    markersize=5, capsize=3, label=loc)
    ax.set_xlabel("k (faulty weights)")
    ax.set_ylabel("SDC rate (%), mean ± std over runs")
    ax.set_title(f"SDC vs k per location, rule = {rule}", fontsize=10)
    ax.legend()
    save(fig, f"sdc_vs_k_{rule}", quick)


def fig_rejection(rej, rule, quick):
    r = rej[rej.rule == rule]
    locs = ordered_locs(r.location.unique())
    width = 0.8 / len(locs)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for i, (loc, color) in enumerate(zip(locs, SERIES)):
        d = r[r.location == loc].sort_values("bit")
        ax.bar(d.bit + (i - (len(locs) - 1) / 2) * width, d.rejection_rate_pct.fillna(0), width=width,
               color=color, edgecolor=SURFACE, linewidth=1, label=loc)
    ax.set_xticks(list(EXP_BITS))
    ax.set_xlabel("exponent bit (30 = MSB)")
    ax.set_ylabel("rejected as non-finite (% of attempts)")
    ax.set_title(f"Trivially detectable faults (finite-value check), rule = {rule}", fontsize=10)
    ax.legend()
    save(fig, f"rejection_by_bit_{rule}", quick)


def fig_exit_distribution(df, rule, K_names, quick):
    k0_path = results_path(f"k0_sanity_{rule}.json", quick)
    if not k0_path.is_file():
        return
    with open(k0_path) as f:
        k0 = json.load(f)
    cols = [f"exit_{n}" for n in K_names]
    d = df[df.rule == rule]
    n = d.n_subset.iloc[0]
    rows = [("clean", [100.0 * k0[c] / n for c in cols] + [100.0 * k0["n_flagged"] / n])]
    for loc in ordered_locs(d.location.unique()):
        g = d[d.location == loc]
        rows.append((f"faulty {loc}", [100.0 * g[c].mean() / n for c in cols] + [100.0 * g.n_flagged.mean() / n]))
    labels = [n_.replace("final", "final classifier") for n_ in K_names] + ["flagged"]
    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(rows) + 1.2))
    for i, (name, vals) in enumerate(rows):
        left = 0.0
        for j, v in enumerate(vals):
            ax.barh(i, v, left=left, color=SERIES[j], edgecolor=SURFACE, linewidth=2,
                    label=labels[j] if i == 0 else None)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of clean-correct test images (faulty = mean over all k and runs)")
    ax.set_title(f"Exit distribution, clean vs faulty, rule = {rule}", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, f"exit_distribution_{rule}", quick)


def clean_table(quick):
    path = output_path("clean_eval.csv", quick)
    if not path.is_file():
        return None
    c = pd.read_csv(path)
    keep = [col for col in c.columns if col.endswith("_count") or col in
            ("rule", "alpha", "accuracy", "accuracy_accepted", "flag_rate", "avg_flops", "saved_pct",
             "head_flops_overhead_pct", "head_params")]
    c = c[keep].copy()
    c["avg_flops"] = (c["avg_flops"] / 1e6).round(2)
    c = c.rename(columns={"avg_flops": "avg_MFLOPs"})
    with open(results_path("clean_results_table.md", quick), "w") as f:
        f.write(c.round(2).to_markdown(index=False))
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    global TAG
    TAG = args.tag

    clean = clean_table(args.quick)
    print("[clean table]", "written" if clean is not None else "outputs/clean_eval.csv missing", flush=True)

    df = load_runs(args.quick)
    if df is None:
        raise FileNotFoundError("no results/faults_<rule>_<location>.csv found - run 08_fault_campaign.py first")
    summ = summary_table(df)
    summ.to_csv(results_path("fault_summary.csv", args.quick), index=False)
    cols = ["rule", "location", "k", "runs", "correct_rate_mean", "sdc_rate_mean", "sdc_rate_std", "sdc_rate_median",
            "sdc_rate_ci95_lo", "sdc_rate_ci95_hi", "flag_rate_mean", "pct_catastrophic_sdc_gt10",
            "rejection_rate_pct", "undo_all_ok"]
    with open(results_path("fault_summary.md", args.quick), "w") as f:
        f.write(summ[cols].round(3).to_markdown(index=False))
    rej = rejection_by_bit(df)
    rej.to_csv(results_path("rejection_by_bit.csv", args.quick), index=False)

    K_names = [c[len("exit_"):] for c in df.columns if c.startswith("exit_")]
    for rule in sorted(df.rule.unique()):
        fig_stacked(summ, rule, args.quick)
        fig_sdc_vs_k(summ, rule, args.quick)
        fig_rejection(rej, rule, args.quick)
        fig_exit_distribution(df, rule, K_names, args.quick)
        done = summ[summ.rule == rule]
        print(f"[{rule}] {len(df[df.rule == rule])} runs | locations {ordered_locs(done.location.unique())} | "
              f"undo all ok: {bool(done.undo_all_ok.all())}", flush=True)
    print(f"[saved] {results_path('fault_summary.csv', args.quick).parent}", flush=True)


if __name__ == "__main__":
    main()
