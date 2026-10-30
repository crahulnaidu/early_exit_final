"""Step 6: compare the NAS model (tag nas) with the 16/A4 + 23/A3 validation model (no tag), for whatever exists.

Clean test results (outputs/[nas/]clean_eval.csv) at alpha = 1 %, and fault results per location and k
(results/[nas/]fault_summary.csv) for every rule campaign present in both. Writes results/nas/compare_models.md,
results/nas/compare_models.csv and figures results/nas/figures/compare_sdc_vs_k_<rule>.png|pdf.
Note: G1/G2/G3 contain different layers in the two models (they follow each model's exit positions).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.runmeta import output_path, results_path

MODELS = {"16/23 validation model": "", "NAS model": "nas"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})


def read(path):
    return pd.read_csv(path) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    md = ["# NAS model vs 16/23 validation model\n"]

    clean_rows = []
    for name, tag in MODELS.items():
        c = read(output_path("clean_eval.csv", q, tag))
        if c is None:
            md.append(f"- {name}: no clean evaluation found\n")
            continue
        c = c[c.alpha == 1.0].copy()
        c.insert(0, "model", name)
        clean_rows.append(c)
    if clean_rows:
        clean = pd.concat(clean_rows, ignore_index=True)
        cols = [c for c in ["model", "rule", "accuracy", "accuracy_accepted", "flag_rate", "avg_flops", "saved_pct",
                            "head_flops_overhead_pct", "head_params"] if c in clean.columns]
        clean = clean[cols].copy()
        clean["avg_flops"] = (clean.avg_flops / 1e6).round(2)
        md.append("## Clean test results (alpha = 1 %)\n\n" + clean.round(3).to_markdown(index=False) + "\n")
        clean.to_csv(results_path("compare_models_clean.csv", q, "nas"), index=False)
    cfg_path = output_path("chosen_config.json", q, "nas")
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = json.load(f)
        md.append(f"\nNAS model: exits at {cfg['positions']}, rule {cfg['rule']}, heads {json.dumps(cfg['archs'])}\n")

    faults = {}
    for name, tag in MODELS.items():
        s = read(results_path("fault_summary.csv", q, tag))
        if s is not None:
            faults[name] = s
    rows = []
    if len(faults) == 2:
        rules = sorted(set.intersection(*[set(s.rule) for s in faults.values()]))
        for rule in rules:
            for name, s in faults.items():
                for r in s[s.rule == rule].itertuples():
                    rows.append({"rule": rule, "model": name, "location": r.location, "k": r.k, "runs": r.runs,
                                 "sdc_mean": r.sdc_rate_mean, "sdc_std": r.sdc_rate_std,
                                 "flag_mean": r.flag_rate_mean, "correct_mean": r.correct_rate_mean,
                                 "pct_catastrophic": r.pct_catastrophic_sdc_gt10,
                                 "rejection_pct": r.rejection_rate_pct})
            d = pd.DataFrame([r for r in rows if r["rule"] == rule])
            locs = [l for l in ["G1", "G2", "G3", "HEADS"] if l in set(d.location)]
            fig, axes = plt.subplots(1, len(locs), figsize=(2.6 * len(locs), 2.8), sharey=True, squeeze=False)
            for ax, loc in zip(axes[0], locs):
                for (name, _), color in zip(MODELS.items(), SERIES):
                    g = d[(d.model == name) & (d.location == loc)].sort_values("k")
                    if len(g):
                        ax.errorbar(g.k, g.sdc_mean, yerr=g.sdc_std.fillna(0), color=color, linewidth=2, marker="o",
                                    markersize=4, capsize=2, label=name)
                ax.set_title(loc, fontsize=9)
                ax.set_xlabel("k")
            axes[0][0].set_ylabel("SDC rate (%), mean ± std")
            axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
            fig.suptitle(f"SDC vs k, NAS model vs 16/23 model, rule = {rule}", fontsize=10, y=1.05)
            for ext in ("pdf", "png"):
                fig.savefig(results_path(f"figures/compare_sdc_vs_k_{rule}.{ext}", q, "nas"), dpi=200,
                            bbox_inches="tight")
            plt.close(fig)
        comp = pd.DataFrame(rows)
        comp.to_csv(results_path("compare_models.csv", q, "nas"), index=False)
        piv = comp.pivot_table(index=["rule", "location", "k"], columns="model",
                               values=["sdc_mean", "flag_mean"]).round(3)
        piv.columns = [f"{a} | {b}" for a, b in piv.columns]
        md.append("\n## Faults: SDC and flag rate (%) per location and k\n\n" + piv.reset_index().to_markdown(index=False)
                  + "\n\nG1/G2/G3 follow each model's own exit positions, so the layers inside a group differ.\n")
    else:
        md.append(f"\nFault summaries available for: {list(faults) or 'none'} (need both to compare)\n")
    with open(results_path("compare_models.md", q, "nas"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md)[:3000], flush=True)


if __name__ == "__main__":
    main()
