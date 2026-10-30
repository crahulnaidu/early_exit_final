"""Step 3: exponent-bit fault campaign on the clean-correct test subset (resumable).

For each location in {G1, .., G_{K+1}, HEADS}, k = 1..kmax faulty weights, run = 1..runs:
  draw k distinct weights uniformly in the location, one random exponent bit (23..30) each, flip them one by one;
  a flip is REJECTED (undone, logged, redrawn) if the new weight is non-finite or any logit at any exit on the
  subset becomes non-finite. Then route the faulty logits with the calibrated rule, record correct / SDC / flagged /
  masked rates, exit distribution and per-exit SDC, undo, and assert the weights are bit-identical.
One CSV row per run is appended to results/faults_<rule>_<location>.csv; finished runs are skipped on restart.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from faults.evaluator import SubsetEvaluator
from faults.inject import (EXP_BITS, LOCATION_IDS, LocationSampler, checksum, fault_locations, fault_seed,
                           identical_to, snapshot)
from models.searched_exit_model import load_searched_model
from utils.calibrate import thresholds_for
from utils.exit_conditions import route
from utils.runmeta import add_common_args, ckpt_path, output_path, results_path, run_meta
from utils.seed import set_seed

MAX_ATTEMPTS_PER_RUN = 1000


def all_finite(logits):
    return all(bool(torch.isfinite(z).all()) for z in logits)


def outcome(rule, th, logits, golden, K):
    exit_id, pred = route(rule, logits, th)
    flagged = exit_id == -1
    sdc = ~flagged & (pred != golden)
    masked = ~flagged & (pred == golden)
    n = golden.shape[0]
    res = {"correct_rate": 100.0 * masked.float().mean().item(),   # golden == label on the clean-correct subset
           "sdc_rate": 100.0 * sdc.float().mean().item(),
           "flag_rate": 100.0 * flagged.float().mean().item(),
           "masked_rate": 100.0 * masked.float().mean().item(),
           "n_flagged": int(flagged.sum())}
    for j in range(K + 1):
        name = f"E{j + 1}" if j < K else "final"
        res[f"exit_{name}"] = int((exit_id == j).sum())
        res[f"sdc_{name}"] = int((sdc & (exit_id == j)).sum())
    res["n_sdc"] = int(sdc.sum())
    assert res["n_sdc"] + int(masked.sum()) + res["n_flagged"] == n
    return res


def load_images(loader, device):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def done_runs(path):
    if not path.is_file():
        return set()
    with open(path) as f:
        return {(int(r["k"]), int(r["run"])) for r in csv.DictReader(f)}


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--ee-ckpt", default="ee_searched.pth", help="checkpoint file name under checkpoints/")
    parser.add_argument("--rule", default="margin_band", choices=["margin_band", "maxlogit_band", "energy"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--locations", nargs="+", default=None, help="default: all")
    parser.add_argument("--eval-batch", type=int, default=1000)
    args = parser.parse_args()
    if args.quick:
        args.kmax, args.runs = 2, 2

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    with open(output_path(f"calib_{args.rule}.json", args.quick, args.tag)) as f:
        calib = json.load(f)
    th = thresholds_for(calib, args.alpha)

    model, ee_meta = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick),
                                         ckpt_path(args.ee_ckpt, args.quick))
    model = model.to(device).eval()
    assert calib["positions"] == ee_meta["positions"], "calibration was made for a different model"
    K = len(ee_meta["positions"])

    # Golden = fault-free early-exit system on the full test set; clean-correct = correct and not flagged.
    _, _, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size, seed=args.seed,
                                                num_workers=args.num_workers, quick=args.quick)
    images, labels = load_images(test_loader, device)
    clean_all = SubsetEvaluator(model, images, args.eval_batch).clean_logits
    g_exit, g_pred = route(args.rule, clean_all, th)
    keep = (g_exit >= 0) & (g_pred == labels)
    idx = torch.nonzero(keep).flatten()
    with open(results_path(f"clean_correct_{args.rule}.json", args.quick, args.tag), "w") as f:
        json.dump(run_meta(args, rule=args.rule, alpha=args.alpha, n_test=int(labels.shape[0]),
                           n_clean_correct=int(idx.numel()), indices=idx.cpu().tolist()), f)
    print(f"[subset] rule={args.rule} alpha={args.alpha}% | clean-correct {idx.numel()}/{labels.shape[0]} "
          f"| exits at {ee_meta['positions']} {ee_meta['archs']}", flush=True)

    evaluator = SubsetEvaluator(model, images[idx], args.eval_batch)
    golden = g_pred[idx]
    del images, clean_all

    # k = 0 sanity: clean weights -> SDC 0, no flags, and the forward pass is deterministic.
    k0 = outcome(args.rule, th, evaluator.clean_logits, golden, K)
    again = evaluator.full_logits()
    k0["deterministic"] = all(torch.equal(a, b) for a, b in zip(again, evaluator.clean_logits))
    k0_ok = k0["n_sdc"] == 0 and k0["n_flagged"] == 0 and k0["deterministic"]
    with open(results_path(f"k0_sanity_{args.rule}.json", args.quick, args.tag), "w") as f:
        json.dump(run_meta(args, rule=args.rule, ok=k0_ok, **k0), f, indent=2)
    print(f"[k=0 sanity] SDC={k0['n_sdc']} flagged={k0['n_flagged']} deterministic={k0['deterministic']} "
          f"-> {'OK' if k0_ok else 'FAILED'}", flush=True)
    if not k0_ok:
        raise SystemExit("k=0 sanity check failed")

    locations = fault_locations(model)
    names = args.locations or list(locations.keys())
    snap = snapshot(locations)
    sum_start = checksum(locations)
    print(f"[weights] sha256 {sum_start} | locations " +
          ", ".join(f"{n}: {sum(p.numel() for _, p in locations[n]):,}" for n in names), flush=True)

    fieldnames = None
    for loc in locations.keys():
        if loc not in names:
            continue
        loc_id = LOCATION_IDS[loc]
        sampler = LocationSampler(loc, locations[loc])
        out_csv = results_path(f"faults_{args.rule}_{loc}.csv", args.quick, args.tag)
        finished = done_runs(out_csv)
        for k in range(1, args.kmax + 1):
            for run in range(1, args.runs + 1):
                if (k, run) in finished:
                    continue
                t0 = time.time()
                seed = fault_seed(loc_id, k, run)
                rng = np.random.default_rng(seed)
                applied, rejected, used = [], [], set()
                attempts = {b: 0 for b in EXP_BITS}
                rejects = {b: 0 for b in EXP_BITS}
                logits = evaluator.clean_logits
                n_attempts = 0
                while len(applied) < k:
                    n_attempts += 1
                    if n_attempts > MAX_ATTEMPTS_PER_RUN:
                        raise RuntimeError(f"{loc} k={k} run={run}: more than {MAX_ATTEMPTS_PER_RUN} attempts")
                    fault = sampler.draw(rng)
                    if fault["global_index"] in used:      # one flip per weight
                        continue
                    attempts[fault["bit"]] += 1
                    sampler.flip(fault)
                    reason = None
                    if not np.isfinite(fault["new"]):
                        reason = "weight_nonfinite"
                    else:
                        new_logits = evaluator.logits(loc)
                        if not all_finite(new_logits):
                            reason = "logits_nonfinite"
                    if reason:
                        sampler.unflip(fault)
                        rejects[fault["bit"]] += 1
                        rejected.append({**fault, "reason": reason})
                        continue
                    used.add(fault["global_index"])
                    applied.append(fault)
                    logits = new_logits

                res = outcome(args.rule, th, logits, golden, K)
                for fault in reversed(applied):
                    sampler.unflip(fault)
                undo_ok = identical_to(locations, snap)

                row = {"seed": args.seed, "rule": args.rule, "alpha": args.alpha, "location": loc, "k": k,
                       "run": run, "fault_seed": seed, "n_subset": int(golden.shape[0]), **res,
                       "n_attempts": sum(attempts.values()),
                       "n_rejected_weight_nonfinite": sum(r["reason"] == "weight_nonfinite" for r in rejected),
                       "n_rejected_logits_nonfinite": sum(r["reason"] == "logits_nonfinite" for r in rejected),
                       **{f"attempts_bit{b}": attempts[b] for b in EXP_BITS},
                       **{f"rejected_bit{b}": rejects[b] for b in EXP_BITS},
                       "undo_ok": undo_ok, "seconds": round(time.time() - t0, 3),
                       "faults": json.dumps([{x: f[x] for x in ("layer", "index", "bit", "old", "new")}
                                             for f in applied]),
                       "rejected": json.dumps([{x: f[x] for x in ("layer", "index", "bit", "old", "new", "reason")}
                                               for f in rejected])}
                new_file = not out_csv.is_file()
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new_file:
                        w.writeheader()
                    w.writerow(row)
                if not undo_ok:
                    raise RuntimeError(f"undo check failed at {loc} k={k} run={run}")
                if run % 25 == 0 or args.quick:
                    print(f"[{args.rule} {loc} k={k} run {run:3d}] correct {res['correct_rate']:.2f}% "
                          f"SDC {res['sdc_rate']:.2f}% flag {res['flag_rate']:.2f}% | rejected {len(rejected)} | "
                          f"{row['seconds']:.2f}s | elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)

    sum_end = checksum(locations)
    status = run_meta(args, rule=args.rule, alpha=args.alpha, locations=names, kmax=args.kmax, runs=args.runs,
                      checksum_start=sum_start, checksum_end=sum_end, checksum_ok=sum_start == sum_end,
                      minutes=round((time.time() - t_start) / 60, 2))
    with open(results_path(f"campaign_{args.rule}_status.json", args.quick, args.tag), "w") as f:
        json.dump(status, f, indent=2)
    print(f"[done] checksum start == end: {sum_start == sum_end} | {status['minutes']} min", flush=True)
    if sum_start != sum_end:
        raise SystemExit("final checksum mismatch")


if __name__ == "__main__":
    main()
