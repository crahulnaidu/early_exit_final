"""Phase 0 (no new training), from the source project's finished results (read-only):

0a final hypervolume over 3 sampler seeds, TPE vs random (source outputs/nas/hv_seeds.json).
0b NAS model vs 16/23 model per location and k, margin_band campaign:
   G1-G3: same weights and same fault seeds -> paired by (k, run): Wilcoxon signed-rank on SDC and flag rate
          (all 800 pairs, plus a sensitivity run on pairs whose logged fault lists are identical);
   HEADS: different weights -> Mann-Whitney U. Holm correction within each metric family.
0c HEADS fault breakdown per head (which head causes SDC) and per exponent bit, both models
   (k = 1 runs: exactly one faulty weight; plus all-k runs whose faults all sit in one head).
Writes results/phase0/*.csv, results/phase0/phase0_tables.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import SRC_OUTPUT_DIR, SRC_RESULTS_DIR
from utils.paths import safe_path

OUT = Path("/home1/jtt/early_exit_final/stage2_paper/results/phase0")
MODELS = {"16/23": SRC_RESULTS_DIR, "NAS": SRC_RESULTS_DIR / "nas"}
LOCS = ["G1", "G2", "G3", "HEADS"]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if (d != 0).sum() == 0:
        return np.nan, 1.0, 0
    r = stats.wilcoxon(a, b, zero_method="wilcox")
    return float(r.statistic), float(r.pvalue), int((d != 0).sum())


def load(model, loc):
    return pd.read_csv(MODELS[model] / f"faults_margin_band_{loc}.csv")


def part_0a():
    with open(SRC_OUTPUT_DIR / "nas" / "hv_seeds.json") as f:
        hv = json.load(f)
    rows = [{"sampler": s, "n_runs": v["n_runs"], "hv_mean": v["mean"], "hv_std": v["std"],
             "values": ", ".join(f"{x:.3f}" for x in v["values"])} for s, v in hv["final_hv"].items()]
    return pd.DataFrame(rows), hv["reference_point"]


def part_0b():
    rows = []
    for loc in LOCS:
        a, b = load("16/23", loc), load("NAS", loc)
        m = a.merge(b, on=["k", "run"], suffixes=("_old", "_nas"))
        assert (m.fault_seed_old == m.fault_seed_nas).all()
        for kk, g in [("all", m)] + [(str(k), x) for k, x in m.groupby("k")]:
            for metric in ("sdc_rate", "flag_rate"):
                x, y = g[f"{metric}_old"], g[f"{metric}_nas"]
                row = {"location": loc, "k": kk, "metric": metric, "n": len(g), "mean_16_23": x.mean(),
                       "mean_NAS": y.mean(), "median_16_23": x.median(), "median_NAS": y.median()}
                if loc == "HEADS":
                    r = stats.mannwhitneyu(x, y, alternative="two-sided")
                    row.update(test="Mann-Whitney U", stat=float(r.statistic), p=float(r.pvalue), n_nonzero=None)
                else:
                    st, p, nz = wilcoxon(x, y)
                    same = g[g.faults_old == g.faults_nas]
                    st2, p2, nz2 = wilcoxon(same[f"{metric}_old"], same[f"{metric}_nas"])
                    row.update(test="Wilcoxon signed-rank (paired by k, run)", stat=st, p=p, n_nonzero=nz,
                               n_identical_faults=len(same), p_identical_faults_only=p2)
                rows.append(row)
    df = pd.DataFrame(rows)
    for metric in ("sdc_rate", "flag_rate"):
        sel = df.metric == metric
        df.loc[sel, "p_holm"] = holm(df.loc[sel, "p"])
    df["significant_holm_0.05"] = df.p_holm < 0.05
    df["NAS_better"] = np.where(df.metric == "sdc_rate", df.mean_NAS < df.mean_16_23, df.mean_NAS < df.mean_16_23)
    return df


def head_of(layer):
    return int(layer.split(".")[1])


def part_0c():
    rows_head, rows_bit = [], []
    for model in MODELS:
        d = load(model, "HEADS")
        faults = d.faults.apply(json.loads)
        d["heads"] = faults.apply(lambda fs: sorted({head_of(f["layer"]) for f in fs}))
        d["bits"] = faults.apply(lambda fs: [f["bit"] for f in fs])
        one = d[d.heads.apply(len) == 1].copy()
        one["head"] = one.heads.apply(lambda h: h[0] + 1)
        for (head, k1), g in [((h, "k=1"), x[x.k == 1]) for h, x in one.groupby("head")] + \
                             [((h, "all k, single-head runs"), x) for h, x in one.groupby("head")]:
            rows_head.append({"model": model, "faulty_head": f"E{head}", "runs": k1, "n": len(g),
                              "sdc_mean": g.sdc_rate.mean(), "flag_mean": g.flag_rate.mean(),
                              "pct_runs_any_sdc": 100 * (g.n_sdc > 0).mean(),
                              "sdc_per_fault": (g.sdc_rate / g.k).mean()})
        k1 = d[d.k == 1].copy()
        k1["bit"] = k1.bits.apply(lambda b: b[0])
        k1["head"] = k1.heads.apply(lambda h: h[0] + 1)
        for (bit, head), g in k1.groupby(["bit", "head"]):
            rows_bit.append({"model": model, "bit": bit, "faulty_head": f"E{head}", "n": len(g),
                             "sdc_mean": g.sdc_rate.mean(), "flag_mean": g.flag_rate.mean(),
                             "pct_runs_any_sdc": 100 * (g.n_sdc > 0).mean()})
        for bit, g in k1.groupby("bit"):
            rows_bit.append({"model": model, "bit": bit, "faulty_head": "any", "n": len(g),
                             "sdc_mean": g.sdc_rate.mean(), "flag_mean": g.flag_rate.mean(),
                             "pct_runs_any_sdc": 100 * (g.n_sdc > 0).mean()})
    return pd.DataFrame(rows_head), pd.DataFrame(rows_bit).sort_values(["model", "faulty_head", "bit"])


def main():
    hv, ref = part_0a()
    t0b = part_0b()
    heads, bits = part_0c()
    for name, df in [("0a_hv_seeds", hv), ("0b_paired_tests", t0b), ("0c_heads_per_head", heads),
                     ("0c_heads_per_bit", bits)]:
        df.to_csv(safe_path(OUT / f"{name}.csv"), index=False)
    summ = t0b[t0b.k == "all"][["location", "metric", "test", "n", "mean_16_23", "mean_NAS", "p", "p_holm",
                                "significant_holm_0.05", "n_identical_faults", "p_identical_faults_only"]]
    md = ["# Phase 0 tables (from the source's finished results)\n",
          f"## 0a Final hypervolume over 3 sampler seeds (reference point {ref})\n", hv.round(4).to_markdown(index=False),
          "\n## 0b NAS model vs 16/23 model, margin_band (pooled over k; per-k rows in 0b_paired_tests.csv)\n",
          summ.to_markdown(index=False, floatfmt=".4g"),
          "\nG1-G3: Wilcoxon signed-rank, paired by (k, run) with identical fault seeds; HEADS: Mann-Whitney U "
          "(different head weights). Holm correction over all (location, k) tests of the same metric.\n",
          "## 0c HEADS faults by faulty head\n", heads.to_markdown(index=False, floatfmt=".4g"),
          "\n## 0c HEADS faults by exponent bit (k = 1)\n", bits.to_markdown(index=False, floatfmt=".4g")]
    with open(safe_path(OUT / "phase0_tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
