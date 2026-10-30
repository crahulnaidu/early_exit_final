"""Step 5: Pareto fronts, hypervolume vs trials (same reference point for both samplers), and the fixed pick rule.

Minimisation form of the objectives: (100 - acc_sel, avg_flops / F_base, head_params / 1e6), feasible trials only.
Reference point (fixed before running, same for TPE and random):
  (100 - (backbone VAL_SEL acc - 1.0),  1.1,  1.1 * 3 * largest single-head params in the search space / 1e6).
Pick rule (fixed before running): among the TPE Pareto points within 1.0 point of the backbone VAL_SEL accuracy,
the lowest expected FLOPs; ties -> fewer head params. (The random sampler's pick is reported for comparison.)
Writes outputs/nas/chosen_config.json, pareto_<sampler>.csv, hypervolume.csv, nas_summary.json and figures.
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

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.backbone import FEATURE_SHAPES
from models.head_builder import CONV_TYPES, HIDDEN, POOLINGS, REDUCTIONS, config_head_params
from models.multi_exit import CANDIDATE_POSITIONS
from utils.flops import TOTAL_BASE_FLOPS
from utils.hypervolume import hv3d, pareto_mask
from utils.runmeta import output_path, results_path

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})


def max_head_params():
    best = 0
    for p in CANDIDATE_POSITIONS:
        C = FEATURE_SHAPES[p][0]
        for pool in POOLINGS:
            for h in HIDDEN:
                best = max(best, config_head_params({"n_conv": 0, "pooling": pool, "hidden_fc": h}, C))
                for n in (1, 2):
                    for t in CONV_TYPES:
                        for r in REDUCTIONS:
                            best = max(best, config_head_params({"n_conv": n, "conv_type": t, "channel_reduction": r,
                                                                 "pooling": pool, "hidden_fc": h}, C))
    return best


def objectives(df):
    return np.stack([100.0 - df.acc_sel.to_numpy(), df.avg_flops.to_numpy() / TOTAL_BASE_FLOPS,
                     df.head_params.to_numpy() / 1e6], axis=1)


def pareto(df):
    feas = df[df.feasible.astype(str) == "True"].copy()
    if feas.empty:
        return feas
    return feas[pareto_mask(objectives(feas))]


def hv_curve(df, ref):
    df = df.sort_values("trial")
    vals = []
    for t in range(1, len(df) + 1):
        feas = df.iloc[:t]
        feas = feas[feas.feasible.astype(str) == "True"]
        vals.append(hv3d(objectives(feas), ref) if len(feas) else 0.0)
    return vals


def pick(front, acc_min):
    cand = front[front.acc_sel >= acc_min]
    if cand.empty:
        return None
    return cand.sort_values(["avg_flops", "head_params"]).iloc[0]


def save(fig, name, quick):
    for ext in ("pdf", "png"):
        fig.savefig(results_path(f"figures/{name}.{ext}", quick, "nas"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    trials = {s: pd.read_csv(output_path(f"trials_{s}.csv", q, "nas")) for s in ("tpe", "random", "handplaced")}
    with open(output_path("meta_tpe.json", q, "nas")) as f:
        meta = json.load(f)
    acc_min = meta["acc_min"]
    params_ref = 1.1 * 3 * max_head_params() / 1e6
    ref = np.array([100.0 - acc_min, 1.1, params_ref])

    fronts, hv, picks = {}, {}, {}
    for s in ("tpe", "random"):
        fronts[s] = pareto(trials[s])
        fronts[s].to_csv(output_path(f"pareto_{s}.csv", q, "nas"), index=False)
        hv[s] = hv_curve(trials[s], ref)
        picks[s] = pick(fronts[s], acc_min)
    n = max(len(hv["tpe"]), len(hv["random"]))
    pd.DataFrame({"trial": range(1, n + 1), "hv_tpe": hv["tpe"] + [np.nan] * (n - len(hv["tpe"])),
                  "hv_random": hv["random"] + [np.nan] * (n - len(hv["random"]))}).to_csv(
        output_path("hypervolume.csv", q, "nas"), index=False)

    chosen = picks["tpe"]
    summary = {"reference_point": ref.tolist(), "acc_min": acc_min, "backbone_sel_acc": meta["backbone_sel_acc"],
               "F_base": TOTAL_BASE_FLOPS,
               "final_hv": {s: hv[s][-1] for s in hv}, "feasible_trials": {s: int((trials[s].feasible.astype(str) == "True").sum()) for s in trials},
               "pareto_size": {s: len(fronts[s]) for s in fronts},
               "pick_tpe": None if chosen is None else chosen.to_dict(),
               "pick_random": None if picks["random"] is None else picks["random"].to_dict(),
               "handplaced": trials["handplaced"].to_dict(orient="records")}
    with open(output_path("nas_summary.json", q, "nas"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ("final_hv", "feasible_trials", "pareto_size", "acc_min")}, indent=2))

    # Figures: hypervolume vs trials; accuracy vs FLOPs (all feasible trials, fronts, hand-placed, chosen).
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for s, color in zip(("tpe", "random"), SERIES):
        ax.plot(range(1, len(hv[s]) + 1), hv[s], color=color, linewidth=2, label=s.upper() if s == "tpe" else "random")
    ax.set_xlabel("trial")
    ax.set_ylabel("hypervolume (same reference point)")
    ax.set_title("Hypervolume of the feasible Pareto front", fontsize=10)
    ax.legend()
    save(fig, "nas_hypervolume_vs_trials", q)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for s, color in zip(("tpe", "random"), SERIES):
        t = trials[s]
        feas = t[t.feasible.astype(str) == "True"]
        ax.scatter(feas.avg_flops / 1e6, feas.acc_sel, s=10, color=color, alpha=0.35, linewidths=0)
        fr = fronts[s].sort_values("avg_flops")
        ax.plot(fr.avg_flops / 1e6, fr.acc_sel, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{'TPE' if s == 'tpe' else 'random'} Pareto front")
    hp = trials["handplaced"]
    ax.scatter(hp.avg_flops / 1e6, hp.acc_sel, s=60, marker="D", color=SERIES[2], edgecolors=SURFACE, linewidths=2,
               label="hand-placed 6/13/23", zorder=5)
    if chosen is not None:
        ax.scatter([chosen.avg_flops / 1e6], [chosen.acc_sel], s=140, marker="*", color=TEXT, label="chosen (TPE)",
                   zorder=6)
    ax.axhline(acc_min, color=TEXT2, linewidth=1, linestyle="--")
    ax.text(ax.get_xlim()[1], acc_min, " constraint", va="center", ha="left", color=TEXT2, fontsize=8)
    ax.set_xlabel("expected MFLOPs per image (VAL_SEL)")
    ax.set_ylabel("accuracy on VAL_SEL (%)")
    ax.set_title("NAS: accuracy vs expected FLOPs (feasible trials)", fontsize=10)
    ax.legend(loc="lower right")
    save(fig, "nas_pareto_acc_vs_flops", q)

    if chosen is None:
        raise SystemExit("no feasible TPE Pareto point within 1.0 point of the backbone: nothing to pick")
    cfg = {"positions": json.loads(chosen.positions), "archs": json.loads(chosen.cfgs), "rule": chosen.rule,
           "source": f"tpe trial {int(chosen.trial)}", "val_sel": {k: chosen[k] for k in
                                                                ("acc_sel", "acc_sel_deployed", "avg_flops", "saved_pct",
                                                                 "head_params", "exit_fractions")},
           "pick_rule": "lowest expected FLOPs among TPE Pareto points with acc_sel >= backbone_sel_acc - 1.0; "
                        "ties -> fewer head params"}
    with open(output_path("chosen_config.json", q, "nas"), "w") as f:
        json.dump(cfg, f, indent=2, default=float)
    print(f"[pick] {cfg['source']}: positions {cfg['positions']} rule {cfg['rule']} | acc {chosen.acc_sel:.2f}% | "
          f"saved {chosen.saved_pct:.2f}% | params {int(chosen.head_params):,}", flush=True)


if __name__ == "__main__":
    main()
