"""Extra analyses (no new full training).

a) One validation table, all rows scored IDENTICALLY to the NAS trials (8-epoch proxy heads on cached features,
   thresholds calibrated on VAL_CAL, metrics on VAL_SEL, alpha = 1 %):
     - chosen NAS model (3 exits, TPE pick)
     - best 2-exit TPE point under the same pick rule (Pareto front of the feasible 2-exit TPE trials, within
       1.0 point of the backbone, lowest FLOPs, ties -> fewer params); also whether it lies on the global TPE front
     - 16/A4 + 23/A3 validation model (margin_band and maxlogit_band)
     - hand-placed 6/13/23 with the default head (both rules)
   plus, where a fully trained checkpoint exists, its per-head val accuracy (full 5k val).
b) Head parameters / FLOPs / FC input size per head, and faultable weights per fault location (G1, G2, G3, HEADS),
   16/23 model vs NAS model.
Writes results/nas/extra_analyses.md, extra_val_table.csv, extra_heads.csv, extra_locations.csv.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from faults.inject import fault_locations, location_ranges
from models.head_builder import fc_input_size
from models.searched_exit_model import load_searched_model
from utils.ckpt import read_checkpoint_meta
from utils.flops import TOTAL_BASE_FLOPS, head_cost
from utils.hypervolume import pareto_mask
from utils.runmeta import ckpt_path, output_path, results_path


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, root_dir / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nas = load_script("13_nas_search")
pick = load_script("14_pick_nas")


def short(arch):
    if isinstance(arch, str):
        return arch
    c = arch
    conv = "" if c["n_conv"] == 0 else (f"{c['n_conv']}x{'DS' if c['conv_type'] == 'depthwise_separable' else 'std'}"
                                        f" C*{c['channel_reduction']:g}, ")
    pool = "GAP" if c["pooling"] == 1 else f"{c['pooling']}x{c['pooling']}"
    return f"{conv}{pool}, FC{c['hidden_fc'] or '-'}"


def row(label, res, note=""):
    return {"model": label, "positions": res["positions"], "heads": " | ".join(short(a) for a in json.loads(res["cfgs"])),
            "rule": res["rule"], "acc_sel": res["acc_sel"], "acc_sel_deployed": res["acc_sel_deployed"],
            "flag_sel_deployed": res["flag_sel_deployed"], "avg_MFLOPs": res["avg_flops"] / 1e6,
            "saved_pct": res["saved_pct"], "head_params": int(res["head_params"]),
            "head_flops_overhead_pct": res["head_flops_overhead_pct"], "exit_fractions": res["exit_fractions"],
            "feasible": res["feasible"], "note": note}


def first_linear_in(head):
    return next(m.in_features for m in head.modules() if isinstance(m, nn.Linear))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- a) validation table ----------------
    tpe = pd.read_csv(output_path("trials_tpe.csv", q, "nas"))
    hand = pd.read_csv(output_path("trials_handplaced.csv", q, "nas"))
    with open(output_path("meta_tpe.json", q, "nas")) as f:
        meta = json.load(f)
    acc_min = meta["acc_min"]
    with open(output_path("chosen_config.json", q, "nas")) as f:
        chosen_cfg = json.load(f)
    chosen_trial = int(chosen_cfg["source"].split()[-1])
    feas = tpe[tpe.feasible.astype(str) == "True"]
    global_front = feas[pareto_mask(pick.objectives(feas))]
    two = feas[feas.n_exits == 2]
    two_front = two[pareto_mask(pick.objectives(two))] if len(two) else two
    best2 = pick.pick(two_front, acc_min) if len(two_front) else None

    rows = [row("NAS chosen (TPE pick)", tpe[tpe.trial == chosen_trial].iloc[0], f"TPE trial {chosen_trial}")]
    if best2 is not None:
        on_front = int(best2.trial) in set(global_front.trial)
        rows.append(row("best 2-exit TPE point", best2, f"TPE trial {int(best2.trial)}; on global TPE Pareto front: "
                                                       f"{'yes' if on_front else 'no'}"))
    ev = nas.Evaluator(SimpleNamespace(quick=q, proxy_epochs=8, seed=42, alpha=1.0), device)
    for rule in ("margin_band", "maxlogit_band"):
        res = ev([(16, "A4"), (23, "A3")], rule)
        rows.append(row("16/A4 + 23/A3 model", res, "same proxy scoring"))
    for r in hand.itertuples():
        rows.append(row("hand-placed 6/13/23", r._asdict(), "default head (A3-equivalent)"))
    table = pd.DataFrame(rows)
    table.to_csv(results_path("extra_val_table.csv", q, "nas"), index=False)

    trained = []
    for name, ck in (("NAS chosen (fully trained)", "ee_nas.pth"), ("16/A4 + 23/A3 (fully trained)", "ee_old16_23.pth")):
        p = ckpt_path(ck, q)
        if p.is_file():
            m = read_checkpoint_meta(p)
            trained.append({"model": name, "positions": m["positions"],
                            "head_val_acc_full_5k": [round(a, 2) for a in m["head_val_accs"]],
                            "final_val_acc": round(m["final_val_acc"], 2), "best_epoch": m["epoch"]})
    trained = pd.DataFrame(trained)

    # ---------------- b) heads and fault locations ----------------
    head_rows, loc_rows = [], []
    for name, ck in (("16/A4 + 23/A3", "ee_old16_23.pth"), ("NAS chosen", "ee_nas.pth")):
        if not ckpt_path(ck, q).is_file():
            continue
        model, m = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", q), ckpt_path(ck, q))
        for j, (p, a, h) in enumerate(zip(m["positions"], m["archs"], model.exit_heads), start=1):
            fl, pr = head_cost(p, a)
            faultable = sum(x.weight.numel() for x in h.modules() if isinstance(x, (nn.Conv2d, nn.Linear)))
            head_rows.append({"model": name, "exit": j, "position": p, "head": short(a), "params": pr,
                              "faultable_weights": faultable, "MFLOPs": fl / 1e6,
                              "fc_input": fc_input_size(p, a) if isinstance(a, dict) else first_linear_in(h)})
        locs = fault_locations(model)
        total = sum(sum(t.numel() for _, t in v) for v in locs.values())
        ranges = location_ranges(model)
        for loc, tensors in locs.items():
            n = sum(t.numel() for _, t in tensors)
            loc_rows.append({"model": name, "location": loc,
                             "layers": f"features[{ranges[loc][0]}..{ranges[loc][1]}]" + (" + classifier" if loc == "G3" else "")
                             if loc in ranges else "exit heads",
                             "faultable_weights": n, "share_pct": 100.0 * n / total})
        loc_rows.append({"model": name, "location": "TOTAL", "layers": "", "faultable_weights": total, "share_pct": 100.0})
    heads, locs_df = pd.DataFrame(head_rows), pd.DataFrame(loc_rows)
    heads.to_csv(results_path("extra_heads.csv", q, "nas"), index=False)
    locs_df.to_csv(results_path("extra_locations.csv", q, "nas"), index=False)

    md = ["# Extra analyses (no new full training)\n",
          f"Backbone accuracy on VAL_SEL: {meta['backbone_sel_acc']:.2f} % -> constraint acc >= {acc_min:.2f} %. "
          f"F_base = {TOTAL_BASE_FLOPS / 1e6:.2f} MFLOPs.\n",
          "## a) Validation comparison (VAL_SEL, 2,500 images)\n",
          "All rows are scored like the NAS trials: 8-epoch proxy heads on cached features, thresholds calibrated on "
          "VAL_CAL, metrics on VAL_SEL. `acc_sel` = exits only (final band off, the NAS objective); "
          "`acc_sel_deployed` / `flag_sel_deployed` = with the alpha = 1 % final band. These are VALIDATION numbers, "
          "not test numbers.\n",
          table.drop(columns=["exit_fractions"]).round(3).to_markdown(index=False),
          "\nExit fractions (% of VAL_SEL per exit, then final, deployed rule):\n",
          table[["model", "rule", "exit_fractions"]].to_markdown(index=False)]
    if len(trained):
        md += ["\nFully trained heads (05 recipe, 60 epochs), head val accuracy alone on the full 5k val:\n",
               trained.to_markdown(index=False)]
    md += ["\n## b) Heads and faultable weights per location\n", heads.round(3).to_markdown(index=False), "",
           locs_df.round(2).to_markdown(index=False),
           "\nFaultable = Conv2d/Linear weights only. G1/G2/G3 follow each model's own exit positions."]
    with open(results_path("extra_analyses.md", q, "nas"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
