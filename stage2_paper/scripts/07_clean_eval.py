"""Step 2: clean TEST evaluation of every rule and alpha (vectorised), plus the slow-loop reference check.

Writes outputs/clean_eval.csv (one row per rule x alpha), outputs/clean_eval.json and outputs/reference_check.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from models.searched_exit_model import load_searched_model
from utils.calibrate import collect_logits, thresholds_for
from utils.exit_conditions import RULES, route
from utils.flops import TOTAL_BASE_FLOPS, expected_flops, head_cost
from utils.reference import compare_with_reference
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed


def summarise(rule, alpha, exit_id, pred, targets, positions, archs):
    K = len(positions)
    n = targets.shape[0]
    correct = (pred == targets) & (exit_id >= 0)
    row = {"rule": rule, "alpha": alpha, "n": n,
           "accuracy": 100.0 * correct.float().mean().item(),
           "flag_rate": 100.0 * (exit_id == -1).float().mean().item()}
    for k in range(K + 1):
        name = f"E{k+1}" if k < K else "final"
        sel = exit_id == k
        row[f"{name}_count"] = int(sel.sum())
        row[f"{name}_acc"] = 100.0 * correct[sel].float().mean().item() if sel.any() else float("nan")
    row["flagged_count"] = int((exit_id == -1).sum())
    accepted = exit_id >= 0
    row["accuracy_accepted"] = 100.0 * correct[accepted].float().mean().item() if accepted.any() else float("nan")
    row.update(expected_flops(positions, archs, exit_id.cpu()))
    row["head_params"] = sum(head_cost(p, a)[1] for p, a in zip(positions, archs))
    return row


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--ee-ckpt", default="ee_searched.pth", help="checkpoint file name under checkpoints/")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, quick=args.quick)
    model, ee_meta = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick),
                                         ckpt_path(args.ee_ckpt, args.quick))
    model = model.to(device)
    positions, archs = ee_meta["positions"], ee_meta["archs"]
    logits, targets = collect_logits(model, test_loader, device)
    backbone_correct = int(logits[-1].argmax(-1).eq(targets).sum())
    backbone_acc = 100.0 * backbone_correct / targets.shape[0]
    print(f"[test] n={targets.shape[0]} | backbone (final classifier) accuracy {backbone_acc:.2f}% | "
          f"F_base {TOTAL_BASE_FLOPS / 1e6:.2f} MFLOPs", flush=True)

    rows, ref = [], {}
    for rule in RULES:
        calib_path = output_path(f"calib_{rule}.json", args.quick, args.tag)
        if not calib_path.is_file():
            raise FileNotFoundError(f"{calib_path} not found. Run 06_calibrate.py first.")
        with open(calib_path) as f:
            calib = json.load(f)
        assert calib["positions"] == positions, "calibration was made for a different model"
        for alpha_key in calib["final"]:
            alpha = float(alpha_key)
            th = thresholds_for(calib, alpha)
            exit_id, pred = route(rule, logits, th)
            rows.append(summarise(rule, alpha, exit_id, pred, targets, positions, archs))
            mism, n_ref = compare_with_reference(rule, logits, th, exit_id, pred, n=500)
            ref[f"{rule}@{alpha}"] = {"mismatches": mism, "n": n_ref}
            r = rows[-1]
            print(f"[{rule:13s} a={alpha:>3}%] acc {r['accuracy']:.2f}% | flag {r['flag_rate']:.2f}% | "
                  f"exits {[r[f'E{k+1}_count'] for k in range(len(positions))] + [r['final_count']]} | "
                  f"saved {r['saved_pct']:.2f}% | ref mismatches {mism}/{n_ref}", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "seed", args.seed)
    df.to_csv(output_path("clean_eval.csv", args.quick, args.tag), index=False)
    meta = run_meta(args, positions=positions, archs=archs, backbone_test_acc=backbone_acc,
                    backbone_test_correct=backbone_correct, n_test=int(targets.shape[0]),
                    F_base=TOTAL_BASE_FLOPS)
    with open(output_path("clean_eval.json", args.quick, args.tag), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    with open(output_path("reference_check.json", args.quick, args.tag), "w") as f:
        json.dump({"meta": meta, "checks": ref, "all_identical": all(v["mismatches"] == 0 for v in ref.values())},
                  f, indent=2)
    print(f"[Saved] {output_path('clean_eval.csv', args.quick, args.tag)} and reference_check.json", flush=True)


if __name__ == "__main__":
    main()
