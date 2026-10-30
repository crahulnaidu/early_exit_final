"""Phase 1: NAS search variants (proxy-scored on VAL_SEL, thresholds calibrated per trial on VAL_CAL).

Same evaluator, proxy (8 epochs), memo, VAL_CAL/VAL_SEL split and constraint as the source's 13_nas_search.py.
Options (combined per variant):
  --sampler   tpe (TPESampler multivariate) | nsga2 (NSGAIISampler, population 50) | random
  --space     full      : n_exits 1..3, per exit position + configurable head (n_conv, conv_type, reduction,
                          pooling, hidden_fc)                                      [= source search space]
              menu      : per exit position + head from the old menu {A1..A5}            (B1)
              positions : per exit position, head fixed to the default (A3-equivalent)  (B2)
              heads     : positions fixed at 16 and 23, configurable heads              (B3)
  --n-exits   0 = searched 1..3 (positions sampled per exit, sorted, duplicates dropped = source behaviour);
              1/2/3 = exactly that many exits (one categorical over all sorted position combinations)
  --rule      searched (default) | margin_band | maxlogit_band
  --fault-objective none | sdc (E1: 4 objectives acc, FLOPs, params, head_fault_sdc) |
              exposure (E2: 3 objectives acc, FLOPs, exposure = faultable head weights x head_fault_sdc / 1e6;
              head_fault_sdc is NOT a separate objective)
  --seeds     sampler seeds; one study per seed, run one after another in this process
Objectives (objective_spec): none -> (acc, FLOPs, params); sdc -> (acc, FLOPs, params, head_fault_sdc);
exposure -> (acc, FLOPs, exposure). Accuracy is maximised, everything else minimised.
Writes outputs/nas_variants/trials_<variant>_s<seed>.csv and meta_<variant>_s<seed>.json.
"""
import argparse
import csv
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import optuna
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from faults.head_probe import HeadBank, probe_head_sdc
from models.head_builder import CONV_TYPES, DEFAULT_HEAD, HIDDEN, N_CONV, POOLINGS, REDUCTIONS, normalize_cfg
from models.multi_exit import CANDIDATE_POSITIONS
from utils.calibrate import calibrate_rule, thresholds_for
from utils.runmeta import add_common_args, output_path, run_meta
from utils.seed import set_seed

spec = importlib.util.spec_from_file_location("nas13", root_dir / "scripts" / "13_nas_search.py")
nas13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nas13)

NAS_RULES = nas13.NAS_RULES
MENU = ["A1", "A2", "A3", "A4", "A5"]


def objective_spec(fault_objective):
    """(Optuna directions, result keys) of the search objectives for a fault-objective setting."""
    keys = {"none": ["acc_sel", "avg_flops", "head_params"],
            "sdc": ["acc_sel", "avg_flops", "head_params", "head_fault_sdc"],
            "exposure": ["acc_sel", "avg_flops", "exposure"]}[fault_objective]
    return ["maximize"] + ["minimize"] * (len(keys) - 1), keys


def suggest_head(trial, j, space):
    if space == "menu":
        return trial.suggest_categorical(f"arch_{j}", MENU)
    if space == "positions":
        return normalize_cfg(DEFAULT_HEAD)
    n_conv = trial.suggest_categorical(f"n_conv_{j}", list(N_CONV))
    cfg = {"n_conv": n_conv}
    if n_conv > 0:
        cfg["conv_type"] = trial.suggest_categorical(f"conv_type_{j}", list(CONV_TYPES))
        cfg["channel_reduction"] = trial.suggest_categorical(f"channel_reduction_{j}", list(REDUCTIONS))
    cfg["pooling"] = trial.suggest_categorical(f"pooling_{j}", list(POOLINGS))
    cfg["hidden_fc"] = trial.suggest_categorical(f"hidden_fc_{j}", list(HIDDEN))
    return normalize_cfg(cfg)


def make_suggest(args):
    combos = {k: ["-".join(map(str, c)) for c in itertools.combinations(CANDIDATE_POSITIONS, k)] for k in (1, 2, 3)}

    def suggest(trial):
        if args.space == "heads":
            positions = [16, 23]
        elif args.n_exits:
            positions = [int(p) for p in trial.suggest_categorical("positions", combos[args.n_exits]).split("-")]
        else:
            k = trial.suggest_int("n_exits", 1, 3)
            positions = [trial.suggest_categorical(f"pos_{j}", CANDIDATE_POSITIONS) for j in range(1, k + 1)]
        raw = [(p, j, suggest_head(trial, j, args.space)) for j, p in enumerate(positions, start=1)]
        rule = args.rule if args.rule != "searched" else trial.suggest_categorical("rule", list(NAS_RULES))
        exits, seen = [], set()
        for pos, _, cfg in sorted(raw, key=lambda r: (r[0], r[1])):
            if pos not in seen:
                seen.add(pos)
                exits.append((pos, cfg))
        return len(positions), exits, rule
    return suggest


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--variant", required=True, help="name, e.g. A2_nsga2")
    parser.add_argument("--sampler", required=True, choices=["tpe", "nsga2", "random"])
    parser.add_argument("--space", default="full", choices=["full", "menu", "positions", "heads"])
    parser.add_argument("--n-exits", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--rule", default="searched", choices=["searched", "margin_band", "maxlogit_band"])
    parser.add_argument("--fault-objective", default="none", choices=["none", "sdc", "exposure"])
    parser.add_argument("--probe-faults", type=int, default=40)
    parser.add_argument("--probe-timing", type=int, default=0, help="only time N trials with the probe, then exit")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--proxy-epochs", type=int, default=8)
    parser.add_argument("--preload-all", action="store_true",
                        help="load every cached position (train + val) onto the device first (worst-case memory)")
    parser.add_argument("--report-memory", action="store_true", help="print GPU name and peak memory at the end")
    args = parser.parse_args()
    if args.quick:
        args.trials, args.probe_faults = 4, 4
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = nas13.Evaluator(args, device)
    if args.preload_all:
        t0 = time.time()
        for p in CANDIDATE_POSITIONS:
            for split in ("train", "val"):
                ev.cache.feats(split, p)
        print(f"[preload] all {len(CANDIDATE_POSITIONS)} positions on {device} in {time.time() - t0:.1f}s", flush=True)
    bank = HeadBank(ev.cache, args.proxy_epochs, args.seed) if args.fault_objective != "none" else None
    suggest = make_suggest(args)
    directions, objective_keys = objective_spec(args.fault_objective)
    print(f"[{args.variant}] sampler={args.sampler} space={args.space} n_exits={args.n_exits or '1-3'} "
          f"rule={args.rule} fault_objective={args.fault_objective} seeds={args.seeds} | constraint acc >= "
          f"{ev.acc_min:.2f}% on VAL_SEL", flush=True)

    def evaluate(trial):
        k, exits, rule = suggest(trial)
        res = ev(exits, rule)
        res["n_exits_requested"] = k
        if bank:
            t0 = time.time()
            logits = [nas13.proxy_val_logits(ev.cache, p, c, args.proxy_epochs, args.seed)[0] for p, c in exits]
            calib = calibrate_rule(rule, [z[ev.cal] for z in logits + [ev.final]], ev.y[ev.cal], alphas=(args.alpha,))
            th = thresholds_for(calib, args.alpha)
            sdc, n_cc, rej, n_w = probe_head_sdc(bank, exits, rule, th, ev.sel, ev.y[ev.sel], ev.final[ev.sel],
                                                 args.probe_faults)
            res.update(head_fault_sdc=sdc, probe_n_clean_correct=n_cc, probe_rejected=rej,
                       head_faultable_weights=n_w, exposure=n_w * sdc / 1e6, probe_faults=args.probe_faults,
                       probe_seconds=round(time.time() - t0, 2), bank_heads=bank.trained)
        return res

    if args.probe_timing:
        sampler = optuna.samplers.RandomSampler(seed=999)
        study = optuna.create_study(directions=directions, sampler=sampler)
        times = []
        for _ in range(args.probe_timing):
            t = study.ask()
            t0 = time.time()
            r = evaluate(t)
            times.append(time.time() - t0)
            print(f"[probe-timing] trial {t.number}: total {times[-1]:.1f}s (probe {r['probe_seconds']}s, "
                  f"{r['probe_faults']} faults, sdc {r['head_fault_sdc']:.3f}%)", flush=True)
        print(f"PROBE_SECONDS_PER_TRIAL={sum(times) / len(times):.2f}", flush=True)
        report_memory(args, device)
        return

    for seed in args.seeds:
        run_name = f"{args.variant}_s{seed}"
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed,
                                                 constraints_func=lambda t: t.user_attrs["constraint"])
        elif args.sampler == "nsga2":
            sampler = optuna.samplers.NSGAIISampler(population_size=50, seed=seed,
                                                    constraints_func=lambda t: t.user_attrs["constraint"])
        else:
            sampler = optuna.samplers.RandomSampler(seed=seed)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(directions=directions, sampler=sampler)
        out_csv = output_path(f"trials_{run_name}.csv", args.quick, "nas_variants")
        fields = None
        t_start = time.time()

        def objective(trial):
            nonlocal fields
            res = evaluate(trial)
            trial.set_user_attr("constraint", [res["constraint"]])
            row = {"seed": args.seed, "variant": args.variant, "sampler": args.sampler, "sampler_seed": seed,
                   "trial": trial.number, **res}
            with open(out_csv, "w" if fields is None else "a", newline="") as f:
                if fields is None:
                    fields = list(row.keys())
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                else:
                    w = csv.DictWriter(f, fieldnames=fields)
                w.writerow(row)
            if trial.number % 25 == 0 or args.quick:
                print(f"[{run_name} {trial.number:3d}] {res['positions']} {res['rule']} acc {res['acc_sel']:.2f}% "
                      f"saved {res['saved_pct']:.1f}% params {res['head_params']:,} feas {res['feasible']}"
                      + (f" sdc {res['head_fault_sdc']:.3f}%" if bank else "")
                      + f" | {(time.time() - t_start) / 60:.1f} min", flush=True)
            return tuple(res[key] for key in objective_keys)

        study.optimize(objective, n_trials=args.trials)
        meta = run_meta(args, variant=args.variant, run_name=run_name, sampler=args.sampler, sampler_seed=seed,
                        space=args.space, n_exits=args.n_exits, rule=args.rule, fault_objective=args.fault_objective,
                        probe_faults=args.probe_faults if bank else None, trials=args.trials,
                        backbone_sel_acc=ev.backbone_sel_acc, acc_min=ev.acc_min,
                        minutes=round((time.time() - t_start) / 60, 2))
        with open(output_path(f"meta_{run_name}.json", args.quick, "nas_variants"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{run_name}] done in {meta['minutes']} min -> {out_csv}", flush=True)
    report_memory(args, device)


def report_memory(args, device):
    if args.report_memory and device.type == "cuda":
        print(f"GPU_NAME={torch.cuda.get_device_name(0)}", flush=True)
        print(f"PEAK_MEMORY_GB={torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
        print(f"TOTAL_MEMORY_GB={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}", flush=True)


if __name__ == "__main__":
    main()
