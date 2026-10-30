"""C1: resolve the "unstable pick" by FULL training of the top candidates of the TPE front.

--prepare : candidates = the 5 lowest-FLOPs feasible Pareto points (acc_sel >= backbone - 1.0) of the union of the
            three source TPE runs (A1), plus TPE seed-1 trials 94 (current NAS model) and 239 (the 16-epoch-proxy pick)
            if not already included. Writes outputs/final/candidates.json and cand_<label>.json (05 --config input).
--select  : every candidate is fully trained (05 recipe, checkpoints/ee_top_<label>.pth). Each is scored exactly like a
            NAS trial but with its fully trained heads: thresholds calibrated on VAL_CAL (its own rule, alpha 1 %),
            exit-only accuracy / expected FLOPs / head params on VAL_SEL. Pick rule (unchanged): lowest FLOPs with
            acc >= backbone VAL_SEL acc - 1.0; ties -> fewer head params. Writes outputs/final/final_choice.json.
            If the winner is trial 94, the source's ee_nas.pth (already fully evaluated) is used as the final model.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import BACKBONE_CKPT, EE_NAS_CKPT, SRC_OUTPUT_DIR
from dataset.datamodule import get_cifar10_dataloaders
from models.searched_exit_model import load_searched_model
from utils import rules2
from utils.calibrate import collect_logits
from utils.flops import expected_flops, head_cost
from utils.hypervolume import pareto_mask
from utils.paths import safe_path
from utils.splits import load_subsplits

spec = importlib.util.spec_from_file_location("pick14", root_dir / "scripts" / "14_pick_nas.py")
pick14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick14)

OUT = Path("/home1/jtt/early_exit_final/stage2_paper/outputs/final")
CKPT = Path("/home1/jtt/early_exit_final/stage2_paper/checkpoints")
RUNS = [("s1", "trials_tpe.csv"), ("s11", "trials_tpe_s11.csv"), ("s21", "trials_tpe_s21.csv")]
MUST = [("s1", 94), ("s1", 239)]


def prepare():
    frames = []
    for label, f in RUNS:
        d = pd.read_csv(SRC_OUTPUT_DIR / "nas" / f)
        d["run_label"] = label
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    acc_min = json.load(open(SRC_OUTPUT_DIR / "nas" / "meta_tpe.json"))["acc_min"]
    feas = df[df.feasible.astype(str) == "True"].drop_duplicates(subset=["positions", "cfgs", "rule"])
    front = feas[pareto_mask(pick14.objectives(feas))]
    cand = front[front.acc_sel >= acc_min].sort_values(["avg_flops", "head_params"]).head(5)
    for lab, t in MUST:
        row = df[(df.run_label == lab) & (df.trial == t)]
        key = (row.positions.iloc[0], row.cfgs.iloc[0], row.rule.iloc[0])
        if not ((cand.positions == key[0]) & (cand.cfgs == key[1]) & (cand.rule == key[2])).any():
            cand = pd.concat([cand, row])
    out = []
    for r in cand.itertuples():
        label = f"{r.run_label}_t{int(r.trial)}"
        cfg = {"positions": json.loads(r.positions), "archs": json.loads(r.cfgs), "rule": r.rule, "label": label,
               "proxy_val_sel": {"acc_sel": r.acc_sel, "saved_pct": r.saved_pct, "head_params": int(r.head_params)}}
        with open(safe_path(OUT / f"cand_{label}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        out.append(cfg)
    with open(safe_path(OUT / "candidates.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(f"{c['label']}: {c['positions']} {c['rule']} proxy saved {c['proxy_val_sel']['saved_pct']:.2f}% "
                    f"acc {c['proxy_val_sel']['acc_sel']:.2f}%" for c in out))
    print("LABELS=" + " ".join(c["label"] for c in out))


def select(quick=False):
    cands = json.load(open(OUT / ("candidates_quick.json" if quick else "candidates.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = get_cifar10_dataloaders(batch_size=500, seed=42, num_workers=4, quick=quick)
    _, vs = load_subsplits()
    n_val = len(val_loader.dataset)
    cal_i = torch.tensor([i for i in vs["cal"] if i < n_val])
    sel_i = torch.tensor([i for i in vs["sel"] if i < n_val])
    base = BACKBONE_CKPT if not quick else Path(str(BACKBONE_CKPT).replace("/checkpoints/", "/checkpoints/quick/"))
    rows = []
    bb_sel = None
    for c in cands:
        ck = Path(c["ckpt"]) if "ckpt" in c else CKPT / f"ee_top_{c['label']}.pth"
        if c["label"] == "s1_t94" and not ck.is_file():          # = source ee_nas.pth (same 05 recipe, seed 42)
            ck = EE_NAS_CKPT
        model, meta = load_searched_model(base, ck)
        logits, y = collect_logits(model.to(dev), val_loader, dev)
        if bb_sel is None:
            bb_sel = 100 * logits[-1][sel_i].argmax(-1).eq(y[sel_i]).float().mean().item()
        cal = rules2.calibrate(c["rule"], [z[cal_i] for z in logits], y[cal_i], alphas=(1.0,))
        sel_logits = [z[sel_i] for z in logits]
        off = json.loads(json.dumps(cal))
        off["final"]["1.0"] = {"L": -math.inf, "U": math.inf, "extras": []}
        e_off, p_off, _ = rules2.route(off, sel_logits, 1.0)
        e_dep, p_dep, _ = rules2.route(cal, sel_logits, 1.0)
        acc = 100 * p_off.eq(y[sel_i]).float().mean().item()
        fl = expected_flops(c["positions"], c["archs"], e_dep)
        rows.append({"label": c["label"], "positions": c["positions"], "rule": c["rule"], "acc_sel": acc,
                     "acc_sel_deployed": 100 * (p_dep.eq(y[sel_i]) & (e_dep >= 0)).float().mean().item(),
                     "saved_pct": fl["saved_pct"], "avg_flops": fl["avg_flops"],
                     "head_params": sum(head_cost(p, a)[1] for p, a in zip(c["positions"], c["archs"])),
                     "head_val_accs_full": meta.get("head_val_accs"), "proxy": c["proxy_val_sel"],
                     "ckpt": str(ck)})
        print(f"[{c['label']}] full-training VAL_SEL acc {acc:.2f}% saved {fl['saved_pct']:.2f}% "
              f"(proxy: acc {c['proxy_val_sel']['acc_sel']:.2f}% saved {c['proxy_val_sel']['saved_pct']:.2f}%)", flush=True)
    df = pd.DataFrame(rows)
    feas = df[df.acc_sel >= bb_sel - 1.0]
    win = feas.sort_values(["avg_flops", "head_params"]).iloc[0] if len(feas) else None
    if win is None and quick:
        win = df.sort_values(["avg_flops", "head_params"]).iloc[0]
    if win is None:
        raise SystemExit("no fully trained candidate within 1.0 point of the backbone on VAL_SEL")
    c = next(x for x in cands if x["label"] == win["label"])
    same = win["label"] == "s1_t94"
    choice = {"label": win["label"], "positions": c["positions"], "archs": c["archs"], "rule": c["rule"],
              "ckpt": str(EE_NAS_CKPT) if same else win["ckpt"], "same_as_current_NAS_model": bool(same),
              "backbone_val_sel_acc": bb_sel, "acc_min": bb_sel - 1.0,
              "pick_rule": "lowest FLOPs among fully trained candidates with VAL_SEL acc >= backbone - 1.0; ties -> "
                           "fewer head params", "table": rows}
    sfx = "_quick" if quick else ""
    df.to_csv(safe_path(OUT / f"top5_full_training{sfx}.csv"), index=False)
    with open(safe_path(OUT / f"final_choice{sfx}.json"), "w") as f:
        json.dump(choice, f, indent=2, default=float)
    print(f"[final choice] {win['label']} (same as current NAS model: {same}) -> {choice['ckpt']}", flush=True)


def export_picks():
    """05 --config files for the Phase-1 picks P3 / P4 (outputs/nas_variants/picks.json)."""
    picks = json.load(open("/home1/jtt/early_exit_final/stage2_paper/outputs/nas_variants/picks.json"))
    for p in ("P3", "P4"):
        c = picks[p]
        cfg = {"positions": c["positions"], "archs": c["archs"], "rule": c["rule"], "label": p.lower(),
               "source": c.get("source")}
        with open(safe_path(OUT / f"cand_{p.lower()}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"{p}: {c['positions']} {c['rule']} ({c.get('source')})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "select", "picks"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    {"prepare": prepare, "select": lambda: select(a.quick), "picks": export_picks}[a.mode]()
