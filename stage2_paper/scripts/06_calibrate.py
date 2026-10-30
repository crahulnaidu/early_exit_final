"""Step 2: calibrate the exit rules on clean VALIDATION logits only.

Writes outputs/calib_<rule>.json for rule in {margin_band, maxlogit_band, energy}, each with t/L/U per exit and
[L_fin, U_fin] for alpha in {0.5, 1, 2, 5} %. See utils/calibrate.py for the exact procedure.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from models.searched_exit_model import load_searched_model
from utils.calibrate import ALPHAS, calibrate_rule, collect_logits
from utils.exit_conditions import RULES
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--ee-ckpt", default="ee_searched.pth", help="checkpoint file name under checkpoints/")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step 2] Calibrating exit rules on clean VAL | {device} | seed={args.seed} | quick={args.quick}", flush=True)

    _, val_loader, _ = get_cifar10_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, quick=args.quick)

    ee_ckpt = ckpt_path(args.ee_ckpt, args.quick)
    model, ee_meta = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick), ee_ckpt)
    model = model.to(device)
    val_logits, val_targets = collect_logits(model, val_loader, device)
    print(f"[val] {val_targets.shape[0]} samples | exits at {ee_meta['positions']} ({ee_meta['archs']})", flush=True)

    for rule in RULES:
        calib = calibrate_rule(rule, val_logits, val_targets, ALPHAS)
        calib["positions"], calib["archs"] = ee_meta["positions"], ee_meta["archs"]
        calib["meta"] = run_meta(args, ee_checkpoint=str(ee_ckpt), n_val=int(val_targets.shape[0]))
        out = output_path(f"calib_{rule}.json", args.quick, args.tag)
        with open(out, "w") as f:
            json.dump(calib, f, indent=2)
        ex = ", ".join(f"E{k+1}: t={e['t']:.4g} [{e['L']:.4g},{e['U']:.4g}] n={e['val_exit_count']}"
                       for k, e in enumerate(calib["exits"]))
        fin = " | ".join(f"a={a}%: [{v['L']:.4g},{v['U']:.4g}] FAR={v['val_flag_rate']:.2f}%"
                         for a, v in calib["final"].items())
        print(f"[{rule}] {ex}\n    final {fin}\n    -> {out}", flush=True)


if __name__ == "__main__":
    main()
