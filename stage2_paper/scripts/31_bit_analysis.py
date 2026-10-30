"""Phase 0d: why do moderate exponent flips (bit 26) escape the band while bit-30 flips are caught?

For both models (16/23 and NAS) the k = 1 G1 runs of the source's margin_band campaign are replayed (the logged
fault is re-applied, logits recomputed on the clean-correct test subset, rule margin_band, alpha = 1 %).
Per image whose outcome changed (SDC or flagged) we record the bit, the flip direction (0->1 grows |w| by 2^(2^(b-23)),
1->0 shrinks it), and at every exit the faulty max logit / margin relative to clean and to the calibrated band.
Outputs: results/phase0/0d_bit_runs.csv (per run), 0d_bit_summary.csv (per model x bit), 0d_bit_images.csv
(per changed image: max logit at its decision exit vs [L, U]) and figure results/phase0/figures/0d_bit26_vs_bit30.png.
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

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import BACKBONE_CKPT, EE_NAS_CKPT, EE_OLD16_23_CKPT, SRC_OUTPUT_DIR, SRC_RESULTS_DIR
from dataset.datamodule import get_cifar10_dataloaders
from faults.evaluator import SubsetEvaluator
from faults.inject import fault_locations, identical_to, snapshot
from models.searched_exit_model import load_searched_model
from utils.calibrate import thresholds_for
from utils.exit_conditions import margin, maxlogit, route
from utils.paths import safe_path
from utils.seed import set_seed

OUT = Path("/home1/jtt/early_exit_final/stage2_paper/results/phase0")
MODELS = {"16/23": (EE_OLD16_23_CKPT, SRC_OUTPUT_DIR, SRC_RESULTS_DIR),
          "NAS": (EE_NAS_CKPT, SRC_OUTPUT_DIR / "nas", SRC_RESULTS_DIR / "nas")}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="only 4 runs per model")
    args = parser.parse_args()
    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_cifar10_dataloaders(batch_size=500, seed=42, num_workers=4)
    xs, ys = zip(*test_loader)
    images, labels = torch.cat(xs), torch.cat(ys)

    run_rows, img_rows = [], []
    for name, (ee_ckpt, out_dir, res_dir) in MODELS.items():
        model, meta = load_searched_model(BACKBONE_CKPT, ee_ckpt)
        model = model.to(dev).eval()
        with open(out_dir / "calib_margin_band.json") as f:
            th = thresholds_for(json.load(f), 1.0)
        with open(res_dir / "clean_correct_margin_band.json") as f:
            idx = torch.tensor(json.load(f)["indices"])
        ev = SubsetEvaluator(model, images[idx].to(dev), 1000)
        golden = labels[idx].to(dev)
        clean = ev.clean_logits
        K = len(meta["positions"])
        locs = fault_locations(model)
        params = {n: p for t in locs.values() for n, p in t}
        snap = snapshot(locs)
        runs = pd.read_csv(res_dir / "faults_margin_band_G1.csv")
        runs = runs[runs.k == 1]
        if args.quick:
            runs = runs.head(4)
        for r in runs.itertuples():
            f = json.loads(r.faults)[0]
            flat = params[f["layer"]].data.view(-1)
            grow = abs(f["new"]) > abs(f["old"])
            flat.view(torch.int32)[f["index"]] ^= (1 << f["bit"])
            z = ev.logits("G1")
            flat.view(torch.int32)[f["index"]] ^= (1 << f["bit"])
            exit_id, pred = route("margin_band", z, th)
            flagged = exit_id == -1
            sdc = ~flagged & (pred != golden)
            changed = sdc | flagged
            ml = [maxlogit(x) for x in z]
            ml0 = [maxlogit(x) for x in clean]
            mg = [margin(x) for x in z]
            run = {"model": name, "k": 1, "run": r.run, "bit": f["bit"], "layer": f["layer"],
                   "direction": "grow" if grow else "shrink", "old": f["old"], "new": f["new"],
                   "n_sdc": int(sdc.sum()), "n_flagged": int(flagged.sum()), "n": int(golden.numel()),
                   "recorded_n_sdc": r.n_sdc, "recorded_n_flagged": r.n_flagged}
            for j in range(K + 1):
                name_j = f"E{j + 1}" if j < K else "final"
                ratio = (ml[j] / ml0[j].clamp(min=1e-6)).abs()
                run[f"median_maxlogit_ratio_{name_j}"] = float(ratio.median())
                run[f"max_maxlogit_ratio_{name_j}"] = float(ratio.max())
            run_rows.append(run)
            for i in torch.nonzero(changed).flatten().tolist():
                e = int(exit_id[i])
                j = e if e >= 0 else K           # flagged images are judged at the final exit
                band = th["exits"][j] if j < K else {"t": None, **th["final"]}
                img_rows.append({"model": name, "run": r.run, "bit": f["bit"], "direction": run["direction"],
                                 "outcome": "SDC" if bool(sdc[i]) else "flagged",
                                 "decision_exit": f"E{j + 1}" if j < K else "final",
                                 "maxlogit": float(ml[j][i]), "maxlogit_clean": float(ml0[j][i]),
                                 "margin": float(mg[j][i]), "L": band["L"], "U": band["U"], "t": band["t"]})
        assert identical_to(locs, snap), "undo failed"
        mism = sum(int(x["n_sdc"] != x["recorded_n_sdc"] or x["n_flagged"] != x["recorded_n_flagged"])
                   for x in run_rows if x["model"] == name)
        print(f"[{name}] {len(runs)} k=1 G1 runs replayed | reproduction mismatches vs campaign: {mism}", flush=True)

    runs_df, imgs = pd.DataFrame(run_rows), pd.DataFrame(img_rows)
    summary = []
    for (model, bit), g in runs_df.groupby(["model", "bit"]):
        gi = imgs[(imgs.model == model) & (imgs.bit == bit)] if len(imgs) else imgs
        n_ch = len(gi)
        summary.append({"model": model, "bit": bit, "runs": len(g), "grow_runs": int((g.direction == "grow").sum()),
                        "mean_sdc_pct": 100 * (g.n_sdc / g.n).mean(), "mean_flag_pct": 100 * (g.n_flagged / g.n).mean(),
                        "changed_images": n_ch,
                        "escape_rate_pct": 100 * (gi.outcome == "SDC").mean() if n_ch else np.nan,
                        "median_final_maxlogit_ratio": g.median_maxlogit_ratio_final.median(),
                        "max_final_maxlogit_ratio": g.max_maxlogit_ratio_final.max()})
    summary = pd.DataFrame(summary)
    runs_df.to_csv(safe_path(OUT / "0d_bit_runs.csv"), index=False)
    imgs.to_csv(safe_path(OUT / "0d_bit_images.csv"), index=False)
    summary.to_csv(safe_path(OUT / "0d_bit_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": TEXT2, "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
    if len(imgs):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
        for ax, bit in zip(axes, (26, 30)):
            g = imgs[(imgs.model == "NAS") & (imgs.bit == bit)]
            for (outcome, color) in (("SDC", SERIES[1]), ("flagged", SERIES[2])):
                v = g[g.outcome == outcome].maxlogit.replace([np.inf, -np.inf], np.nan).dropna()
                v = np.sign(v) * np.log10(1 + v.abs())
                if len(v):
                    ax.hist(v, bins=40, color=color, alpha=0.8, label=f"{outcome} (n={len(v)})")
            ax.set_title(f"NAS model, G1, bit {bit}", fontsize=10)
            ax.set_xlabel("max logit at decision exit, sign*log10(1+|x|)")
            ax.legend()
        axes[0].set_ylabel("changed images")
        for ext in ("png", "pdf"):
            fig.savefig(safe_path(OUT / "figures" / f"0d_bit26_vs_bit30.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
