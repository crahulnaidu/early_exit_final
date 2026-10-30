"""Hypervolume over sampler seeds: TPE and random, 3 runs each (seed 1/2 = the main runs, + 2 extra runs each).

Uses the SAME reference point and objective normalisation as 14_pick_nas.py (functions imported from it).
Only runs whose trials file exists are used. The chosen model is NOT changed by these runs.
Writes outputs/nas/hv_seeds.csv (curves), outputs/nas/hv_seeds.json (final HV mean +- std, ddof=1) and
results/nas/figures/nas_hv_seeds.png|pdf.
"""
import argparse
import importlib.util
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

from utils.runmeta import output_path

spec = importlib.util.spec_from_file_location("pick_nas", root_dir / "scripts" / "14_pick_nas.py")
pick_nas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick_nas)

RUNS = {"tpe": ["tpe", "tpe_s11", "tpe_s21"], "random": ["random", "random_s12", "random_s22"]}
CHECKPOINTS = (50, 100, 150, 200, 250)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    with open(output_path("meta_tpe.json", q, "nas")) as f:
        acc_min = json.load(f)["acc_min"]
    ref = np.array([100.0 - acc_min, 1.1, 1.1 * 3 * pick_nas.max_head_params() / 1e6])

    curves, summary = {}, {"reference_point": ref.tolist(), "runs": {}, "final_hv": {}}
    for sampler, names in RUNS.items():
        finals = []
        for name in names:
            path = output_path(f"trials_{name}.csv", q, "nas")
            if not path.is_file():
                continue
            hv = pick_nas.hv_curve(pd.read_csv(path), ref)
            curves[name] = hv
            finals.append(hv[-1])
            summary["runs"][name] = {"trials": len(hv), "final_hv": hv[-1],
                                     **{f"hv_at_{c}": hv[c - 1] for c in CHECKPOINTS if c <= len(hv)}}
        if finals:
            summary["final_hv"][sampler] = {"n_runs": len(finals), "mean": float(np.mean(finals)),
                                            "std": float(np.std(finals, ddof=1)) if len(finals) > 1 else None,
                                            "values": finals}
    n = max(len(v) for v in curves.values())
    pd.DataFrame({k: v + [np.nan] * (n - len(v)) for k, v in curves.items()}).assign(trial=range(1, n + 1)).to_csv(
        output_path("hv_seeds.csv", q, "nas"), index=False)
    with open(output_path("hv_seeds.json", q, "nas"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for (sampler, names), color in zip(RUNS.items(), pick_nas.SERIES):
        runs = [curves[nm] for nm in names if nm in curves]
        if not runs:
            continue
        m = min(len(r) for r in runs)
        arr = np.array([r[:m] for r in runs])
        mean, std = arr.mean(0), (arr.std(0, ddof=1) if len(runs) > 1 else np.zeros(m))
        x = np.arange(1, m + 1)
        ax.plot(x, mean, color=color, linewidth=2, label=f"{'TPE' if sampler == 'tpe' else 'random'} (n={len(runs)})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("trial")
    ax.set_ylabel("hypervolume, mean ± std over seeds")
    ax.set_title("Hypervolume vs trials over sampler seeds", fontsize=10)
    ax.legend()
    for ext in ("pdf", "png"):
        fig.savefig(pick_nas.results_path(f"figures/nas_hv_seeds.{ext}", q, "nas"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["final_hv"], indent=2), flush=True)


if __name__ == "__main__":
    main()
