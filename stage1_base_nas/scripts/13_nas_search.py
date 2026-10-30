"""Step 5: multi-objective NAS over exit placement, head architecture and exit rule (validation only).

Search space (per trial): n_exits 1..3; per exit: position (17 candidates), n_conv {0,1,2},
[conv_type, channel_reduction only if n_conv > 0], pooling {1 (GAP), 2, 4}, hidden_fc {0,64,128,256};
one rule per model {margin_band, maxlogit_band}. Positions are sorted; a repeated position keeps only the head of
the lower exit index (the effective config is logged). Thresholds are NOT searched: per trial they are calibrated
with the accuracy-aware rule (utils/calibrate.py) on VAL_CAL (2.5k).
Objectives on VAL_SEL (2.5k): maximise accuracy, minimise expected FLOPs (deployed rule), minimise head params.
Constraint: accuracy >= backbone accuracy on VAL_SEL - 1.0 (Optuna constraints_func for TPE).
Accuracy for the objective is measured with the final band OFF (alpha flags a fixed alpha*N of val images for any
architecture, see DECISIONS.md); the deployed accuracy with the alpha band is logged too.
Heads come from the 8-epoch proxy on cached features (utils/proxy.py); only VAL logits are memoised, never weights.
--sampler tpe | random | handplaced (exits 6, 13, 23 with the default A3-like head, both rules).
Writes outputs/nas/trials_<sampler>.csv (one row per trial) and outputs/nas/meta_<sampler>.json.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import optuna
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.head_builder import CONV_TYPES, DEFAULT_HEAD, HIDDEN, N_CONV, POOLINGS, REDUCTIONS, normalize_cfg
from models.multi_exit import CANDIDATE_POSITIONS
from utils.calibrate import calibrate_rule, thresholds_for
from utils.exit_conditions import route
from utils.flops import expected_flops, head_cost
from utils.proxy import FeatureCache, proxy_val_logits
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed
from utils.splits import load_subsplits

NAS_RULES = ("margin_band", "maxlogit_band")
MAX_EXITS = 3


class Evaluator:
    def __init__(self, args, device):
        self.args = args
        self.cache = FeatureCache(device, args.quick, ckpt_path("vgg16_bn_cifar10_base.pth", args.quick))
        _, vs = load_subsplits()
        n = self.cache.labels["val"].shape[0]
        self.cal = torch.tensor([i for i in vs["cal"] if i < n])
        self.sel = torch.tensor([i for i in vs["sel"] if i < n])
        self.y = self.cache.labels["val"].cpu()
        self.final = self.cache.final["val"].cpu()
        self.backbone_sel_acc = 100.0 * (self.final[self.sel].argmax(1) == self.y[self.sel]).float().mean().item()
        self.backbone_val_acc = 100.0 * (self.final.argmax(1) == self.y).float().mean().item()
        self.acc_min = self.backbone_sel_acc - 1.0

    def __call__(self, exits, rule):
        t0 = time.time()
        positions, cfgs = [p for p, _ in exits], [c for _, c in exits]
        new = 0
        head_logits = []
        for p, c in exits:
            z, hit = proxy_val_logits(self.cache, p, c, self.args.proxy_epochs, self.args.seed)
            head_logits.append(z)
            new += int(not hit)
        logits = head_logits + [self.final]
        alpha = self.args.alpha
        calib = calibrate_rule(rule, [z[self.cal] for z in logits], self.y[self.cal], alphas=(alpha,))
        th = thresholds_for(calib, alpha)
        sel_logits = [z[self.sel] for z in logits]
        ysel = self.y[self.sel]
        e_off, p_off = route(rule, sel_logits, {"exits": th["exits"], "final": {"L": -math.inf, "U": math.inf}})
        acc = 100.0 * (p_off == ysel).float().mean().item()
        e_dep, p_dep = route(rule, sel_logits, th)
        acc_dep = 100.0 * ((p_dep == ysel) & (e_dep >= 0)).float().mean().item()
        fl = expected_flops(positions, cfgs, e_dep)
        params = sum(head_cost(p, c)[1] for p, c in exits)
        K = len(exits)
        return {"n_exits": K, "positions": json.dumps(positions), "cfgs": json.dumps(cfgs), "rule": rule,
                "acc_sel": acc, "acc_sel_deployed": acc_dep,
                "flag_sel_deployed": 100.0 * (e_dep == -1).float().mean().item(),
                "avg_flops": fl["avg_flops"], "saved_pct": fl["saved_pct"],
                "head_flops_overhead_pct": fl["head_flops_overhead_pct"], "head_params": params,
                "constraint": self.acc_min - acc, "feasible": acc >= self.acc_min,
                "exit_fractions": json.dumps([round(100.0 * (e_dep == j).float().mean().item(), 3)
                                              for j in range(K + 1)]),
                "thresholds": json.dumps([e["t"] for e in calib["exits"]]),
                "new_proxy_heads": new, "seconds": round(time.time() - t0, 2)}


def suggest(trial):
    k = trial.suggest_int("n_exits", 1, MAX_EXITS)
    raw = []
    for j in range(1, k + 1):
        pos = trial.suggest_categorical(f"pos_{j}", CANDIDATE_POSITIONS)
        n_conv = trial.suggest_categorical(f"n_conv_{j}", list(N_CONV))
        cfg = {"n_conv": n_conv}
        if n_conv > 0:
            cfg["conv_type"] = trial.suggest_categorical(f"conv_type_{j}", list(CONV_TYPES))
            cfg["channel_reduction"] = trial.suggest_categorical(f"channel_reduction_{j}", list(REDUCTIONS))
        cfg["pooling"] = trial.suggest_categorical(f"pooling_{j}", list(POOLINGS))
        cfg["hidden_fc"] = trial.suggest_categorical(f"hidden_fc_{j}", list(HIDDEN))
        raw.append((pos, j, normalize_cfg(cfg)))
    rule = trial.suggest_categorical("rule", list(NAS_RULES))
    exits, seen = [], set()
    for pos, _, cfg in sorted(raw, key=lambda r: (r[0], r[1])):      # strictly increasing; duplicates -> first j
        if pos not in seen:
            seen.add(pos)
            exits.append((pos, cfg))
    return k, exits, rule


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--sampler", required=True, choices=["tpe", "random", "handplaced"])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--proxy-epochs", type=int, default=8)
    parser.add_argument("--run-name", default=None,
                        help="file name stem for trials/meta (default: the sampler name); used for extra seed runs")
    args = parser.parse_args()
    run_name = args.run_name or args.sampler
    if args.quick:
        args.trials = 4
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = Evaluator(args, device)
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else {"tpe": 1, "random": 2, "handplaced": 0}[args.sampler]
    print(f"[nas:{args.sampler}] backbone acc VAL_SEL {ev.backbone_sel_acc:.2f}% (full val {ev.backbone_val_acc:.2f}%) "
          f"-> constraint acc >= {ev.acc_min:.2f}% | CAL {len(ev.cal)} SEL {len(ev.sel)} | sampler seed {sampler_seed}",
          flush=True)

    out_csv = output_path(f"trials_{run_name}.csv", args.quick, "nas")
    fields = None
    t_start = time.time()

    def write(row):
        nonlocal fields
        mode = "w" if fields is None else "a"
        if fields is None:
            fields = list(row.keys())
        with open(out_csv, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if mode == "w":
                w.writeheader()
            w.writerow(row)

    if args.sampler == "handplaced":
        exits = [(p, normalize_cfg(DEFAULT_HEAD)) for p in (6, 13, 23)]
        for i, rule in enumerate(NAS_RULES):
            res = ev(exits, rule)
            write({"seed": args.seed, "sampler": "handplaced", "trial": i, "n_exits_requested": 3, **res})
            print(f"[handplaced {rule}] acc {res['acc_sel']:.2f}% | saved {res['saved_pct']:.2f}% | params "
                  f"{res['head_params']:,} | feasible {res['feasible']}", flush=True)
    else:
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(multivariate=True, seed=sampler_seed,
                                                 constraints_func=lambda t: t.user_attrs["constraint"])
        else:
            sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(directions=["maximize", "minimize", "minimize"], sampler=sampler)

        def objective(trial):
            k, exits, rule = suggest(trial)
            res = ev(exits, rule)
            trial.set_user_attr("constraint", [res["constraint"]])
            write({"seed": args.seed, "sampler": args.sampler, "trial": trial.number, "n_exits_requested": k, **res})
            if trial.number % 10 == 0 or args.quick:
                print(f"[{args.sampler} {trial.number:3d}] {res['positions']} {rule} acc {res['acc_sel']:.2f}% "
                      f"saved {res['saved_pct']:.1f}% params {res['head_params']:,} feas {res['feasible']} | "
                      f"{res['seconds']:.1f}s ({res['new_proxy_heads']} new) | {(time.time() - t_start) / 60:.1f} min",
                      flush=True)
            return res["acc_sel"], res["avg_flops"], res["head_params"]

        study.optimize(objective, n_trials=args.trials)

    meta = run_meta(args, sampler=args.sampler, run_name=run_name, sampler_seed=sampler_seed, trials=args.trials,
                    backbone_sel_acc=ev.backbone_sel_acc, backbone_val_acc=ev.backbone_val_acc,
                    acc_min=ev.acc_min, n_cal=len(ev.cal), n_sel=len(ev.sel),
                    minutes=round((time.time() - t_start) / 60, 2))
    with open(output_path(f"meta_{run_name}.json", args.quick, "nas"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[nas:{args.sampler}] done in {meta['minutes']} min -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
