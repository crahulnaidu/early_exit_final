"""Final paper tables and figures from the offline evaluation (41), latency (43), NAS variants (35) and the top-5
selection (44). Everything is read from existing result files; nothing is recomputed on the test set here.

Writes results/paper/paper_tables.md (all tables), one CSV per table and figures/*.png|pdf:
  T2 method comparison (final model, exponent faults, all locations)      T3 per-location SDC / flag
  T4 MENDNet fault model (sign flips, FR 10/30/50 %)                        T5 INT8 weight faults
  T6 alpha sweep (ROC points)                                               T7 k = 1 per-bit breakdown
  T8 band / slack ablation (nas)                                            T9 model comparison (margin_band @ 1 %)
  T10 seeds mean +- std                                                     T11 latency (batch 1)
  T12 hardware cost of each rule / model                                    T13 NAS variants (copied from 35)
  T14 top-5 full training (44)                                              reproduction check summary
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

from utils import rules2
from utils.flops import TOTAL_BASE_FLOPS, exit_costs, head_cost
from utils.paths import safe_path

W = Path("/home1/jtt/early_exit_final/stage2_paper")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MAIN = ["plain_dnn", "sdn_msp", "msp_low@1", "energy_low@1", "energy_band@1", "maxlogit_band@1", "margin_band@1",
        "margin_multiband@1", "margin_band_agree@1", "mendnet_clean@1", "mendnet_faulty_val"]
SHORT = ["plain_dnn", "sdn_msp", "energy_band@1", "margin_band@1", "mendnet_clean@1", "mendnet_faulty_val"]
FAMILIES = ["margin_band", "maxlogit_band", "energy_band", "msp_low", "energy_low", "mendnet_clean"]


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 2, "lines.markersize": 6})


def md(df, floatfmt=".3g"):
    return df.to_markdown(index=False, floatfmt=floatfmt) if len(df) else "_(no data)_"


class Book:
    def __init__(self, out):
        self.out, self.parts = out, []

    def add(self, name, title, df, note=""):
        if df is None:
            self.parts.append(f"## {name}. {title}\n\n_(missing input)_\n")
            return
        df.to_csv(safe_path(self.out / f"{name}.csv"), index=False)
        self.parts.append(f"## {name}. {title}\n\n{note + chr(10) + chr(10) if note else ''}{md(df)}\n")

    def save(self, header):
        with open(safe_path(self.out / "paper_tables.md"), "w") as f:
            f.write(header + "\n\n" + "\n".join(self.parts))


def rd(p):
    return pd.read_csv(p) if Path(p).is_file() else None


def savefig(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(safe_path(out / "figures" / f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-tag", default=None, help="default: from outputs/final/final_choice.json")
    ap.add_argument("--model-tags", nargs="+", default=["nas", "old1623", "final", "p3", "p4"])
    ap.add_argument("--seed-tags", nargs="*", default=["seed1", "seed2"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = "quick" if a.quick else ""
    off, out = W / "results" / "offline" / q, W / "results" / "paper" / q
    choice = json.load(open(W / "outputs/final" / ("final_choice_quick.json" if a.quick else "final_choice.json")))
    final = a.final_tag or ("nas" if choice["same_as_current_NAS_model"] else "final")
    style()
    book = Book(out)

    def summ(tag, fm="exponent"):
        return rd(off / tag / f"summary_{fm}.csv")

    def clean(tag, fm="exponent"):
        return rd(off / tag / f"clean_methods_{fm}.csv")

    # T2
    s, c = summ(final), clean(final)
    if s is not None:
        t = s[(s.location == "ALL") & s.method.isin(MAIN)].merge(
            c[["method", "acc_all", "far_pct", "saved_pct"]].rename(columns={"acc_all": "clean_acc",
                                                                           "saved_pct": "clean_saved_pct"}), on="method")
        t = t.set_index("method").loc[[m for m in MAIN if m in set(t.method)]].reset_index()
        book.add("T2", f"Method comparison on the final model ({final}), exponent faults G1-G3 + heads, k = 1..8",
                 t[["method", "clean_acc", "far_pct", "clean_saved_pct", "sdc_rate", "flag_rate", "det_ratio",
                    "precision", "tpr_binom", "fpr_binom", "crit_latency_median_100", "acc_all", "mitig_acc"]],
                 "SDC / flag on each method's clean-correct subset; det_ratio = flagged/(flagged+SDC); TPR/FPR = "
                 "run-level detection (binomial p<0.01) on critical (M0 drop >= 1 pt) / benign (< 0.1 pt) runs; "
                 "latency = median images until detection (windows of 100).")
        t3 = s[s.method.isin(MAIN) & (s.location != "ALL")].pivot(index="method", columns="location",
                                                                  values=["sdc_rate", "flag_rate"])
        t3.columns = [f"{a_}_{b}" for a_, b in t3.columns]
        book.add("T3", f"Per-location SDC / flag ({final})", t3.reset_index())
        # figure: SDC vs k
        r = rd(off / final / "runs_exponent.csv.gz")
        if r is not None:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for i, m in enumerate(SHORT):
                g = r[r.method == m].groupby("k").sdc_rate.mean()
                if len(g):
                    ax.plot(g.index, g.values, marker="o", color=SERIES[i], label=m)
            ax.set_xlabel("faulty weights k")
            ax.set_ylabel("SDC rate (%)")
            ax.set_title(f"SDC vs number of faults ({final})", color=INK, fontsize=10)
            ax.legend(fontsize=7)
            savefig(fig, out, "sdc_vs_k")
    else:
        book.add("T2", "Method comparison", None)

    # T4 MENDNet fault model, T5 INT8
    sm = rd(off / final / "runs_mendnet.csv.gz")
    if sm is not None:
        t4 = sm[sm.method.isin(SHORT)].groupby(["location", "k", "method"]).agg(
            acc_all=("acc_all", "mean"), mitig_acc=("mitig_acc", "mean"), flag_all=("flag_all", "mean"),
            sdc_rate=("sdc_rate", "mean")).reset_index().rename(columns={"location": "group", "k": "FR_pct"})
        book.add("T4", f"MENDNet fault model: sign flips of FR % of one group's weights ({final})", t4,
                 "FP1 final classifier, FP2 exit-2 head, FP3 first conv after exit 1, FP4 first conv. mitig_acc = "
                 "accuracy when flagged samples take the fallback label.")
    else:
        book.add("T4", "MENDNet fault model", None)
    s8, c8 = summ(final, "int8"), clean(final, "int8")
    if s8 is not None:
        t5 = s8[(s8.location == "ALL") & s8.method.isin(MAIN)].merge(
            c8[["method", "acc_all", "far_pct"]].rename(columns={"acc_all": "clean_acc"}), on="method")
        book.add("T5", f"INT8 weight faults (one bit of k INT8 codes, {final})",
                 t5[["method", "clean_acc", "far_pct", "sdc_rate", "flag_rate", "det_ratio", "tpr_binom", "acc_all",
                     "mitig_acc"]])
    else:
        book.add("T5", "INT8 weight faults", None)

    # T6 alpha sweep + ROC
    if s is not None:
        rows = []
        for fam in FAMILIES:
            for m in sorted(set(s.method)):
                if m.startswith(fam + "@"):
                    al = float(m.split("@")[1])
                    x = s[(s.method == m) & (s.location == "ALL")].iloc[0]
                    cf = c[c.method == m].iloc[0]
                    rows.append({"family": fam, "alpha": al, "clean_far": cf.far_pct, "clean_acc": cf.acc_all,
                                 "saved_pct": cf.saved_pct, "sdc_rate": x.sdc_rate, "det_ratio": x.det_ratio,
                                 "tpr_binom": x.tpr_binom, "fpr_binom": x.fpr_binom})
        t6 = pd.DataFrame(rows).sort_values(["family", "alpha"])
        book.add("T6", "alpha sweep (clean false-alarm budget) per rule family", t6)
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2))
        for i, fam in enumerate(FAMILIES):
            g = t6[t6.family == fam]
            axs[0].plot(g.clean_far, g.det_ratio, marker="o", color=SERIES[i], label=fam)
            axs[1].plot(g.clean_far, g.sdc_rate, marker="o", color=SERIES[i], label=fam)
        axs[0].set_ylabel("image detection ratio (%)")
        axs[1].set_ylabel("SDC rate (%)")
        for ax_ in axs:
            ax_.set_xscale("log")
            ax_.set_xlabel("clean false-alarm rate (%)")
        axs[0].set_title("Detection vs false alarms", color=INK, fontsize=10)
        axs[1].set_title("Residual SDC vs false alarms", color=INK, fontsize=10)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(FAMILIES), fontsize=7, bbox_to_anchor=(0.5, -0.08))
        savefig(fig, out, "roc_alpha_sweep")

    # T7 per bit
    pb = rd(off / final / "per_bit_exponent.csv")
    if pb is not None:
        t7 = pb[pb.method.isin(SHORT)].pivot(index="bit", columns="method", values="sdc_rate").reset_index()
        book.add("T7", f"k = 1 SDC rate (%) per flipped exponent bit ({final})", t7)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        meths = [m for m in SHORT if m in t7.columns]
        wbar = 0.8 / max(len(meths), 1)
        for i, m in enumerate(meths):
            ax.bar(t7.bit + (i - len(meths) / 2 + 0.5) * wbar, t7[m], width=wbar * 0.9, color=SERIES[i], label=m)
        ax.set_xlabel("exponent bit")
        ax.set_ylabel("SDC rate (%)")
        ax.set_title("Single-fault SDC per bit", color=INK, fontsize=10)
        ax.legend(fontsize=7)
        savefig(fig, out, "per_bit_k1")
    else:
        book.add("T7", "per-bit", None)

    # T8 ablation
    sn, cn = summ("nas"), clean("nas")
    if sn is not None:
        ab = ["margin_band@1", "abl_band_upper@1", "abl_band_lower@1", "abl_band_none@1", "abl_slack0@1",
              "abl_slack1@1", "M0_unprotected"]
        t8 = sn[(sn.location == "ALL") & sn.method.isin(ab)].merge(
            cn[["method", "acc_all", "far_pct", "saved_pct"]].rename(columns={"acc_all": "clean_acc",
                                                                            "saved_pct": "clean_saved"}), on="method")
        book.add("T8", "Ablation on the NAS model: band sides and calibration slack (margin_band, alpha 1 %)",
                 t8[["method", "clean_acc", "far_pct", "clean_saved", "sdc_rate", "flag_rate", "det_ratio",
                     "tpr_binom"]], "abl_slack* use slack 0 / 1 points (default 0.5).")

    # T9 models
    rows = []
    for tag in a.model_tags + [final]:
        x, cx = summ(tag), clean(tag)
        if x is None or tag in [r_["tag"] for r_ in rows]:
            continue
        cf = cx[cx.method == "margin_band@1"].iloc[0]
        meta = json.load(open(W / "results/campaigns" / q / tag / "status_exponent_test.json"))
        archs = rd_pt_archs(W / "results/campaigns" / q / tag / "clean_test.pt")
        hp = sum(head_cost(p, a_)[1] for p, a_ in zip(*archs))
        row = {"tag": tag, "positions": str(meta["positions"]), "head_params": hp, "clean_acc": cf.acc_all,
               "far_pct": cf.far_pct, "saved_pct": cf.saved_pct}
        for loc in ("G1", "G2", "G3", "HEADS", "ALL"):
            y_ = x[(x.method == "margin_band@1") & (x.location == loc)]
            if len(y_):
                row[f"sdc_{loc}"] = y_.sdc_rate.iloc[0]
                row[f"flag_{loc}"] = y_.flag_rate.iloc[0]
        rows.append(row)
    book.add("T9", "Models (margin_band, alpha 1 %): clean test + exponent-fault SDC/flag per location",
             pd.DataFrame(rows))

    # T10 seeds
    rows = []
    for tag in [final] + a.seed_tags:
        x, cx = summ(tag), clean(tag)
        if x is None:
            continue
        for m in ("margin_band@1", "energy_band@1", "mendnet_clean@1"):
            cf, y_ = cx[cx.method == m], x[(x.method == m) & (x.location == "ALL")]
            if len(cf) and len(y_):
                rows.append({"tag": tag, "method": m, "clean_acc": cf.acc_all.iloc[0], "far": cf.far_pct.iloc[0],
                             "saved": cf.saved_pct.iloc[0], "sdc": y_.sdc_rate.iloc[0], "flag": y_.flag_rate.iloc[0]})
    if rows:
        d = pd.DataFrame(rows)
        t10 = d.groupby("method").agg(n=("tag", "size"), **{f"{v}_{f}": (v, f) for v in
                                                           ("clean_acc", "far", "saved", "sdc", "flag")
                                                           for f in ("mean", "std")}).reset_index()
        book.add("T10", f"Head-training seeds ({final} seed 42 + {', '.join(a.seed_tags)}), mean +- std", t10,
                 "Seed tags use the first 20 runs per (location, k); the seed-42 row uses all 100.")
    else:
        book.add("T10", "Seeds", None)

    # T11 latency
    lat = rd(W / "results/latency" / q / "latency.csv")
    book.add("T11", "Batch-1 latency (ms), real early stopping, margin_band alpha 1 %", lat)
    if lat is not None:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        modes = ["plain", "ee_all", "ee_stop"]
        g = lat[lat.device == ("cuda" if "cuda" in set(lat.device) else "cpu")]
        tags = list(dict.fromkeys(g.tag))
        for i, mo in enumerate(modes):
            v = [g[(g.tag == t_) & (g["mode"] == mo)].mean_ms.mean() for t_ in tags]
            ax.bar(np.arange(len(tags)) + (i - 1) * 0.27, v, width=0.25, color=SERIES[i], label=mo)
        ax.set_xticks(np.arange(len(tags)), tags)
        ax.set_ylabel("mean latency (ms)")
        ax.set_title(f"Batch-1 latency ({g.device.iloc[0]})", color=INK, fontsize=10)
        ax.legend(fontsize=7)
        savefig(fig, out, "latency")

    # T12 hardware
    rows = []
    for tag in a.model_tags + [final]:
        f = W / "results/campaigns" / q / tag / "clean_test.pt"
        if not f.is_file() or tag in [r_["tag"] for r_ in rows]:
            continue
        pos, arch = rd_pt_archs(f)
        costs = exit_costs(pos, arch)
        rows.append({"tag": tag, "exits": len(pos), "head_params": sum(head_cost(p, a_)[1] for p, a_ in zip(pos, arch)),
                     "head_flops_overhead_pct": 100 * (costs[-1] - TOTAL_BASE_FLOPS) / TOTAL_BASE_FLOPS,
                     "decision_units_margin_band": f"{len(pos) + 1} x (top-2 comparator tree + 1 subtractor + "
                                                   f"3 comparators)",
                     "decision_units_mendnet": f"{len(pos) + 1} x (softmax: 10 exp + divider + max; logsumexp: "
                                               f"exp/log)"})
    book.add("T12", "Hardware cost (theoretical; HLS synthesis pending)", pd.DataFrame(rows))
    book.add("T12b", "Operations per exit decision of each rule",
             pd.DataFrame([{"rule": r, "ops": s_.ops} for r, s_ in rules2.RULES.items()]))

    # T13 / T14 / reproduction
    book.add("T13", "NAS variants (common hypervolume, 3 seeds)", rd(W / "results/nas_variants/T1_search_variants.csv"))
    book.add("T14", "Top-5 candidates after full training (VAL_SEL)",
             rd(W / "outputs/final" / ("top5_full_training_quick.csv" if a.quick else "top5_full_training.csv")))
    rep = []
    for tag in ("nas", "old1623"):
        f = off / tag / "reproduction.json"
        if f.is_file():
            rep.append({"tag": tag, **json.load(open(f))})
    book.add("R", "Reproduction of the source campaign (margin_band, source calibration, identical faults)",
             pd.DataFrame(rep))
    book.save(f"# Paper tables (VGG16-BN / CIFAR-10)\n\nFinal model: **{final}** ({choice['label']}, exits "
              f"{choice['positions']}). Generated by scripts/42_paper_tables.py.")
    print(f"[saved] {out / 'paper_tables.md'}", flush=True)


def rd_pt_archs(f):
    from utils.logit_store import load_clean_logits
    t = load_clean_logits(f)
    return t["positions"], t["archs"]


if __name__ == "__main__":
    main()
