"""Phase 1E check: does the cheap head-fault probe rank configurations like a larger head campaign?

10 configurations from the feasible E1 Pareto front (4 objectives, union of the E1 seeds), spread evenly over the
range of the probe's head_fault_sdc. For each: the probe as used in the search (same N faults, seed base 777000)
and a 200-fault head campaign with DIFFERENT seeds (base 991000), same procedure (single exponent flips in the head
weights, NaN/Inf rejected, VAL_SEL clean-correct subset). Spearman / Kendall between the two.
Writes outputs/nas_variants/probe_validation.csv / .json.
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
from scipy import stats

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from faults.head_probe import HeadBank, probe_head_sdc
from utils.calibrate import calibrate_rule, thresholds_for
from utils.hypervolume import pareto_mask
from utils.proxy import proxy_val_logits
from utils.runmeta import output_path
from utils.seed import set_seed

spec = importlib.util.spec_from_file_location("nas13", root_dir / "scripts" / "13_nas_search.py")
nas13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nas13)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-configs", type=int, default=10)
    parser.add_argument("--campaign-faults", type=int, default=200)
    args = parser.parse_args()
    q = args.quick
    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(output_path("x", q, "nas_variants").parent.glob("trials_E1_sdc_s*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    probe_faults = int(df.probe_faults.iloc[0])
    feas = df[df.feasible.astype(str) == "True"].drop_duplicates(subset=["positions", "cfgs", "rule"])
    obj = np.stack([-feas.acc_sel, feas.avg_flops, feas.head_params, feas.head_fault_sdc], axis=1)
    front = feas[pareto_mask(obj)].sort_values("head_fault_sdc")
    n = min(args.n_configs if not q else 3, len(front))
    pick = front.iloc[np.unique(np.linspace(0, len(front) - 1, n).round().astype(int))]

    a = SimpleNamespace(quick=q, proxy_epochs=8, seed=42, alpha=1.0)
    ev = nas13.Evaluator(a, dev)
    bank = HeadBank(ev.cache, 8, 42)
    rows = []
    for r in pick.itertuples():
        positions, cfgs = json.loads(r.positions), json.loads(r.cfgs)
        exits = list(zip(positions, cfgs))
        logits = [proxy_val_logits(ev.cache, p, c, 8, 42)[0] for p, c in exits]
        calib = calibrate_rule(r.rule, [z[ev.cal] for z in logits + [ev.final]], ev.y[ev.cal], alphas=(1.0,))
        th = thresholds_for(calib, 1.0)
        args_common = (bank, exits, r.rule, th, ev.sel, ev.y[ev.sel], ev.final[ev.sel])
        p_sdc = probe_head_sdc(*args_common, probe_faults if not q else 4, seed_base=777000)[0]
        c_sdc = probe_head_sdc(*args_common, args.campaign_faults if not q else 8, seed_base=991000)[0]
        rows.append({"variant": r.variant, "sampler_seed": r.sampler_seed, "trial": r.trial, "positions": r.positions,
                     "rule": r.rule, "acc_sel": r.acc_sel, "head_params": r.head_params,
                     "search_probe_sdc": r.head_fault_sdc, "probe_sdc_recomputed": p_sdc,
                     "campaign_sdc": c_sdc})
        print(f"[{r.variant} s{r.sampler_seed} t{r.trial}] probe {p_sdc:.4f}% (search {r.head_fault_sdc:.4f}%) | "
              f"{args.campaign_faults}-fault campaign {c_sdc:.4f}%", flush=True)
    out = pd.DataFrame(rows)
    rho, p = stats.spearmanr(out.probe_sdc_recomputed, out.campaign_sdc) if len(out) > 2 else (np.nan, np.nan)
    tau, _ = stats.kendalltau(out.probe_sdc_recomputed, out.campaign_sdc) if len(out) > 2 else (np.nan, np.nan)
    out.to_csv(output_path("probe_validation.csv", q, "nas_variants"), index=False)
    res = {"probe_faults": probe_faults, "campaign_faults": args.campaign_faults, "n_configs": len(out),
           "spearman_rho": float(rho), "spearman_p": float(p), "kendall_tau": float(tau)}
    with open(output_path("probe_validation.json", q, "nas_variants"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
