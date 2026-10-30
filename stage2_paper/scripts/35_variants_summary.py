"""Phase 1/2 summary: variant table (T1), hypervolumes, picks P1-P4 (VAL_SEL), search figures.

Every 3-objective hypervolume uses the SAME objectives (acc_sel, avg_flops, head_params) and the SAME reference
point as the source's 14_pick_nas.py (acc_min from the source meta_tpe.json, params bound from the search space).
E variants also get an HV on their OWN objectives, reported separately ("own HV"):
  E1: 4 objectives (acc, FLOPs, params, head_fault_sdc), reference = common reference + SDC 20 %;
  E2: 3 objectives (acc, FLOPs, exposure), reference = (acc, FLOPs) as common, exposure = params bound x 20 %.
A1 (TPE) and A3 (random) reuse the source's 3 seed runs each (read-only); all other variants are read from
outputs/nas_variants/.
Pick rules (fixed before the runs):
  P1 lowest expected FLOPs among feasible Pareto points (3 obj, union of seeds) of the best A variant (highest mean
     final HV); ties -> fewer head params.          P2 lowest head params on the same front; ties -> lower FLOPs.
  P3 E1 (union of seeds, 4-obj Pareto, feasible) with FLOPs saved >= the 16/23 model's (same proxy scoring) ->
     lowest head_fault_sdc; ties -> lower FLOPs.   P4 best 2-exit point: C2 front, P1 rule.
The head-fault probe is (re)computed for every pick, the 16/23 model, the hand-placed design, the current NAS model
and the A1 seed-1 Pareto points (in-memory heads, same N faults and seeds as the E search).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from optuna._hypervolume import compute_hypervolume

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import SRC_OUTPUT_DIR, SRC_RESULTS_DIR
from faults.head_probe import HeadBank, probe_head_sdc
from models.head_builder import DEFAULT_HEAD, normalize_cfg
from utils.calibrate import calibrate_rule, thresholds_for
from utils.flops import TOTAL_BASE_FLOPS
from utils.hypervolume import pareto_mask
from utils.proxy import proxy_val_logits
from utils.runmeta import output_path, results_path

spec = importlib.util.spec_from_file_location("pick14", root_dir / "scripts" / "14_pick_nas.py")
pick14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick14)
spec = importlib.util.spec_from_file_location("nas13", root_dir / "scripts" / "13_nas_search.py")
nas13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nas13)

SRC_NAS = SRC_OUTPUT_DIR / "nas"
SDC_REF = 20.0
VARIANTS = {   # variant -> (description, group, list of (run label, path builder))
    "A1_tpe": ("TPE, full space", "A", [("s1", SRC_NAS / "trials_tpe.csv"), ("s11", SRC_NAS / "trials_tpe_s11.csv"),
                                        ("s21", SRC_NAS / "trials_tpe_s21.csv")]),
    "A2_nsga2": ("NSGA-II (pop 50), full space", "A", [3, 13, 23]),
    "A3_random": ("Random, full space", "A", [("s2", SRC_NAS / "trials_random.csv"),
                                              ("s12", SRC_NAS / "trials_random_s12.csv"),
                                              ("s22", SRC_NAS / "trials_random_s22.csv")]),
    "B1_menu": ("TPE, old menu heads A1-A5", "B", [1, 11, 21]),
    "B2_positions": ("TPE, positions only (default head)", "B", [1, 11, 21]),
    "B3_heads": ("TPE, heads only (positions 16, 23)", "B", [1, 11, 21]),
    "C1_one": ("TPE, exactly 1 exit", "C", [1, 11, 21]),
    "C2_two": ("TPE, exactly 2 exits", "C", [1, 11, 21]),
    "C3_three": ("TPE, exactly 3 exits", "C", [1, 11, 21]),
    "D1_margin": ("TPE, rule fixed margin_band", "D", [1]),
    "D2_maxlogit": ("TPE, rule fixed maxlogit_band", "D", [1]),
    "E1_sdc": ("TPE, 4 obj (+head_fault_sdc)", "E", [1, 11, 21]),
    "E2_exposure": ("TPE, 3 obj (acc, FLOPs, exposure)", "E", [1, 11, 21]),
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})


def runs_of(variant, q):
    _, _, spec_runs = VARIANTS[variant]
    out = []
    for item in spec_runs:
        if isinstance(item, tuple):
            label, path = item
            meta = path.parent / path.name.replace("trials_", "meta_").replace(".csv", ".json")
        else:
            label = f"s{item}"
            path = output_path(f"trials_{variant}_s{item}.csv", q, "nas_variants")
            meta = output_path(f"meta_{variant}_s{item}.json", q, "nas_variants")
        if path.is_file():
            df = pd.read_csv(path)
            df["run_label"] = label
            df["variant"] = variant
            minutes = json.load(open(meta)).get("minutes") if meta.is_file() else None
            out.append((label, df, minutes, str(path)))
    return out


def feasible(df):
    return df[df.feasible.astype(str) == "True"]


def front3(df):
    f = feasible(df)
    return f[pareto_mask(pick14.objectives(f))] if len(f) else f


def obj4(df):
    return np.concatenate([pick14.objectives(df), df[["head_fault_sdc"]].to_numpy()], axis=1)


def own_objectives(variant, df):
    """E1: (acc_err, FLOPs/F_base, params/1e6, head_fault_sdc); E2: (acc_err, FLOPs/F_base, exposure)."""
    base = pick14.objectives(df)
    if variant.startswith("E1"):
        return np.concatenate([base, df[["head_fault_sdc"]].to_numpy()], axis=1)
    return np.concatenate([base[:, :2], df[["exposure"]].to_numpy()], axis=1)


def hv_own(variant, df, ref):
    f = feasible(df)
    pts = own_objectives(variant, f) if len(f) else np.zeros((0, len(ref)))
    pts = pts[np.all(pts < ref, axis=1)]
    return float(compute_hypervolume(pts, ref)) if len(pts) else 0.0


def save(fig, name, q):
    for ext in ("png", "pdf"):
        fig.savefig(results_path(f"figures/{name}.{ext}", q, "nas_variants"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--probe-faults", type=int, default=40)
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()
    q = args.quick
    with open(SRC_NAS / "meta_tpe.json") as f:
        acc_min = json.load(f)["acc_min"]
    ref3 = np.array([100.0 - acc_min, 1.1, 1.1 * 3 * pick14.max_head_params() / 1e6])
    ref4 = np.concatenate([ref3, [SDC_REF]])
    own_refs = {"E1_sdc": ref4, "E2_exposure": np.array([ref3[0], ref3[1], ref3[2] * SDC_REF])}

    t1, curves, unions = [], {}, {}
    for variant, (desc, group, _) in VARIANTS.items():
        runs = runs_of(variant, q)
        if not runs:
            continue
        finals, npts, best_flops, minutes, own = [], [], [], [], []
        curves[variant] = []
        for label, df, mins, path in runs:
            hv = pick14.hv_curve(df, ref3)
            curves[variant].append(hv)
            finals.append(hv[-1])
            fr = front3(df)
            npts.append(len(fr))
            f = feasible(df)
            best_flops.append(f.avg_flops.min() / 1e6 if len(f) else np.nan)
            minutes.append(mins)
            if variant in own_refs:
                own.append(hv_own(variant, df, own_refs[variant]))
        unions[variant] = pd.concat([d for _, d, _, _ in runs], ignore_index=True)
        t1.append({"variant": variant, "group": group, "description": desc, "runs": len(runs),
                   "hv_mean": np.mean(finals), "hv_std": np.std(finals, ddof=1) if len(finals) > 1 else np.nan,
                   "pareto_pts_mean": np.mean(npts), "best_feasible_MFLOPs_mean": np.nanmean(best_flops),
                   "feasible_pct": 100 * np.mean([len(feasible(d)) / len(d) for _, d, _, _ in runs]),
                   "wall_min_mean": np.mean([m for m in minutes if m is not None]) if any(m is not None for m in minutes) else np.nan,
                   "own_hv_objectives": {"E1_sdc": "acc,FLOPs,params,head_fault_sdc",
                                         "E2_exposure": "acc,FLOPs,exposure"}.get(variant, ""),
                   "own_hv_mean": np.mean(own) if own else np.nan,
                   "own_hv_std": np.std(own, ddof=1) if len(own) > 1 else np.nan,
                   "sources": "; ".join(p for _, _, _, p in runs)})
    t1 = pd.DataFrame(t1)
    t1.to_csv(results_path("T1_search_variants.csv", q, "nas_variants"), index=False)

    # ---------------- picks ----------------
    a_vars = [v for v in ("A1_tpe", "A2_nsga2", "A3_random") if v in unions]
    best_a = t1[t1.variant.isin(a_vars)].sort_values("hv_mean", ascending=False).variant.iloc[0]
    fa = front3(unions[best_a].drop_duplicates(subset=["positions", "cfgs", "rule"]))
    extra = pd.read_csv(SRC_RESULTS_DIR / "nas" / "extra_val_table.csv")
    saved_1623 = float(extra[(extra.model == "16/A4 + 23/A3 model") & (extra.rule == "margin_band")].saved_pct.iloc[0])
    picks = {}
    if len(fa):
        cand = fa[fa.acc_sel >= acc_min]
        picks["P1"] = (cand.sort_values(["avg_flops", "head_params"]).iloc[0], f"{best_a}: lowest FLOPs")
        picks["P2"] = (cand.sort_values(["head_params", "avg_flops"]).iloc[0], f"{best_a}: lowest head params")
    if "E1_sdc" in unions:
        e = feasible(unions["E1_sdc"].drop_duplicates(subset=["positions", "cfgs", "rule"]))
        if len(e):
            e = e[pareto_mask(obj4(e))]
            c3 = e[(e.acc_sel >= acc_min) & (e.saved_pct >= saved_1623)]
            if len(c3):
                picks["P3"] = (c3.sort_values(["head_fault_sdc", "avg_flops"]).iloc[0],
                               f"E1: lowest head_fault_sdc with saved >= {saved_1623:.2f} %")
    if "C2_two" in unions:
        fc = front3(unions["C2_two"].drop_duplicates(subset=["positions", "cfgs", "rule"]))
        c4 = fc[fc.acc_sel >= acc_min]
        if len(c4):
            picks["P4"] = (c4.sort_values(["avg_flops", "head_params"]).iloc[0], "C2: lowest FLOPs")

    with open(SRC_NAS / "chosen_config.json") as f:
        current = json.load(f)
    refs = {"16/23 model": ([(16, "A4"), (23, "A3")], "margin_band"),
            "hand-placed 6/13/23": ([(p, normalize_cfg(DEFAULT_HEAD)) for p in (6, 13, 23)], "margin_band"),
            "current NAS model (source P1)": (list(zip(current["positions"], current["archs"])), current["rule"])}
    rows = []
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = nas13.Evaluator(SimpleNamespace(quick=q, proxy_epochs=8, seed=42, alpha=1.0), dev)
    bank = None if args.no_probe else HeadBank(ev.cache, 8, 42)

    def probe(exits, rule):
        if bank is None:
            return np.nan, np.nan
        logits = [proxy_val_logits(ev.cache, p, c, 8, 42)[0] for p, c in exits]
        calib = calibrate_rule(rule, [z[ev.cal] for z in logits + [ev.final]], ev.y[ev.cal], alphas=(1.0,))
        th = thresholds_for(calib, 1.0)
        sdc, _, _, n_w = probe_head_sdc(bank, exits, rule, th, ev.sel, ev.y[ev.sel], ev.final[ev.sel],
                                        args.probe_faults)
        return sdc, n_w

    for name, (r, rule_txt) in picks.items():
        exits = list(zip(json.loads(r.positions), json.loads(r.cfgs)))
        sdc, n_w = probe(exits, r.rule)
        same = json.loads(r.positions) == current["positions"] and json.loads(r.cfgs) == current["archs"] \
            and r.rule == current["rule"]
        rows.append({"pick": name, "rule_used": rule_txt, "source": f"{r.variant} {r.run_label} trial {int(r.trial)}", "positions": r.positions,
                     "heads": r.cfgs, "rule": r.rule, "acc_sel": r.acc_sel, "acc_sel_deployed": r.acc_sel_deployed,
                     "flag_sel": r.flag_sel_deployed, "saved_pct": r.saved_pct, "head_params": int(r.head_params),
                     "head_faultable_weights": n_w, "head_fault_sdc": sdc, "exit_fractions": r.exit_fractions,
                     "same_as_current_NAS_model": bool(same)})
    for name, (exits, rule) in refs.items():
        res = ev(exits, rule)
        sdc, n_w = probe(exits, rule)
        rows.append({"pick": name, "rule_used": "reference", "source": "", "positions": res["positions"],
                     "heads": res["cfgs"], "rule": rule, "acc_sel": res["acc_sel"],
                     "acc_sel_deployed": res["acc_sel_deployed"], "flag_sel": res["flag_sel_deployed"],
                     "saved_pct": res["saved_pct"], "head_params": int(res["head_params"]),
                     "head_faultable_weights": n_w, "head_fault_sdc": sdc, "exit_fractions": res["exit_fractions"],
                     "same_as_current_NAS_model": name.startswith("current")})
    picks_df = pd.DataFrame(rows)
    picks_df.to_csv(results_path("picks_val_sel.csv", q, "nas_variants"), index=False)
    with open(output_path("picks.json", q, "nas_variants"), "w") as f:
        json.dump({r["pick"]: {"positions": json.loads(r["positions"]), "archs": json.loads(r["heads"]),
                               "rule": r["rule"], "source": r["source"], "rule_used": r["rule_used"],
                               "same_as_current_NAS_model": r["same_as_current_NAS_model"]}
                   for r in rows if r["pick"].startswith("P")}, f, indent=2)

    # head-SDC vs params for the A1 seed-1 Pareto points (motivates E)
    sdc_points = []
    if "A1_tpe" in unions and bank is not None:
        fr = front3(runs_of("A1_tpe", q)[0][1])
        for r in fr.itertuples():
            sdc, n_w = probe(list(zip(json.loads(r.positions), json.loads(r.cfgs))), r.rule)
            sdc_points.append({"set": "A1 seed-1 Pareto", "head_params": r.head_params, "head_fault_sdc": sdc,
                               "acc_sel": r.acc_sel, "saved_pct": r.saved_pct})
    if "E1_sdc" in unions:
        e = feasible(unions["E1_sdc"])
        e = e[pareto_mask(obj4(e))] if len(e) else e
        for r in e.itertuples():
            sdc_points.append({"set": "E1 Pareto (4 obj)", "head_params": r.head_params,
                               "head_fault_sdc": r.head_fault_sdc, "acc_sel": r.acc_sel, "saved_pct": r.saved_pct})
    sdc_points = pd.DataFrame(sdc_points)
    sdc_points.to_csv(results_path("head_sdc_vs_params.csv", q, "nas_variants"), index=False)

    # ---------------- figures ----------------
    for group, title in (("A", "Search algorithm"), ("B", "Search space"), ("C", "Number of exits"),
                         ("D", "Exit rule"), ("E", "Fault-aware (3-objective HV)")):
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        members = [v for v in VARIANTS if VARIANTS[v][1] == group and v in curves]
        if group != "A" and "A1_tpe" in curves:
            members = ["A1_tpe"] + members
        for v, color in zip(members, SERIES):
            arr = curves[v]
            m = min(len(c) for c in arr)
            a = np.array([c[:m] for c in arr])
            mean = a.mean(0)
            std = a.std(0, ddof=1) if len(arr) > 1 else np.zeros(m)
            x = np.arange(1, m + 1)
            ax.plot(x, mean, color=color, linewidth=2, label=f"{v} (n={len(arr)})")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("trial")
        ax.set_ylabel("hypervolume (acc, FLOPs, params)")
        ax.set_title(f"{title}: HV vs trials, mean ± std over seeds", fontsize=10)
        ax.legend()
        save(fig, f"hv_vs_trials_{group}", q)

    for xcol, xlabel, fname in (("avg_flops", "expected MFLOPs (VAL_SEL)", "pareto_acc_flops"),
                                ("head_params", "head params", "pareto_acc_params")):
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        for v, color in zip([v for v in ("A1_tpe", "A2_nsga2", "A3_random") if v in unions], SERIES):
            fr = front3(unions[v])
            fr = fr.sort_values(xcol)
            xs = fr[xcol] / (1e6 if xcol == "avg_flops" else 1)
            ax.plot(xs, fr.acc_sel, color=color, linewidth=2, marker="o", markersize=3, label=f"{v} front (3 seeds)")
        for (name, (r, _)), mk in zip(picks.items(), ["*", "P", "X", "D"]):
            ax.scatter([r[xcol] / (1e6 if xcol == "avg_flops" else 1)], [r.acc_sel], marker=mk, s=110, color=TEXT,
                       zorder=6, label=name)
        for _, r in picks_df[picks_df.rule_used == "reference"].iterrows():
            if not r["pick"].startswith("current"):
                xv = (TOTAL_BASE_FLOPS * (1 - r.saved_pct / 100) / 1e6) if xcol == "avg_flops" else r.head_params
                ax.scatter([xv], [r.acc_sel], marker="s", s=50, color=SERIES[4], edgecolors=SURFACE, zorder=5)
                ax.annotate(r["pick"], (xv, r.acc_sel), fontsize=7, color=TEXT2, xytext=(4, -10),
                            textcoords="offset points")
        ax.axhline(acc_min, color=TEXT2, linewidth=1, linestyle="--")
        if xcol == "head_params":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy on VAL_SEL (%)")
        ax.set_title("Pareto fronts (feasible) and picks", fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
        save(fig, fname, q)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for v, color in zip(("C1_one", "C2_two", "C3_three", "A1_tpe"), SERIES):
        if v in unions:
            fr = front3(unions[v]).sort_values("avg_flops")
            ax.plot(fr.avg_flops / 1e6, fr.acc_sel, color=color, linewidth=2, marker="o", markersize=3, label=v)
    ax.axhline(acc_min, color=TEXT2, linewidth=1, linestyle="--")
    ax.set_xlabel("expected MFLOPs (VAL_SEL)")
    ax.set_ylabel("accuracy on VAL_SEL (%)")
    ax.set_title("Exit count: Pareto fronts (union of seeds)", fontsize=10)
    ax.legend()
    save(fig, "exit_count_fronts", q)

    pf = output_path("proxy_fidelity.csv", q, "nas_variants")
    if pf.is_file():
        d = pd.read_csv(pf)
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for e, color in zip((4, 8, 16), SERIES):
            ax.scatter(d[f"proxy{e}_val_acc"], d.full_val_acc_best, s=18, color=color, label=f"{e}-epoch proxy")
        ax.set_xlabel("proxy val accuracy (%)")
        ax.set_ylabel("full 60-epoch val accuracy (%)")
        ax.set_title("Proxy fidelity (20 head configs)", fontsize=10)
        ax.legend()
        save(fig, "proxy_fidelity", q)

    if len(sdc_points):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for (s, g), color in zip(sdc_points.groupby("set"), SERIES):
            ax.scatter(g.head_params, g.head_fault_sdc, s=18, color=color, label=s)
        ax.set_xscale("log")
        ax.set_xlabel("head params")
        ax.set_ylabel("head-fault probe SDC (%)")
        ax.set_title("Head fault sensitivity vs head size", fontsize=10)
        ax.legend()
        save(fig, "head_sdc_vs_params", q)

    md = ["# NAS variant study: search results (VAL_SEL, proxy-scored)\n",
          f"Reference point (acc_err, FLOPs/F_base, params/1e6) = {ref3.round(4).tolist()}; own HV: E1 adds SDC {SDC_REF} %, E2 uses exposure bound {ref3[2] * SDC_REF:.3f}.",
          f"Constraint: acc_sel >= {acc_min:.2f} %. Best A variant: {best_a}.\n",
          "## T1 Search variants\n",
          t1.drop(columns=["sources"]).to_markdown(index=False, floatfmt=".4g"),
          "\n## Picks (VAL_SEL) and reference designs\n",
          picks_df.drop(columns=["exit_fractions"]).to_markdown(index=False, floatfmt=".4g")]
    with open(results_path("variants_tables.md", q, "nas_variants"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
