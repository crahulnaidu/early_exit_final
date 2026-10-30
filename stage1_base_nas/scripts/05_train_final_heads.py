"""Section 2.7: train the chosen exit heads on the frozen backbone (KD from the final classifier).

Recipe unchanged: 60 epochs, Adam lr 1e-3, wd 1e-4, cosine (eta_min 1e-5), KD T=3 alpha=0.3,
checkpoint = best mean head val accuracy (utils/train_heads.py).
Heads from --positions/--archs (menu names), or --config <json with positions, archs (names or NAS cfg dicts),
rule>, else outputs/chosen_config.json.
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
from models.searched_exit_model import build_searched_model
from utils.ckpt import save_checkpoint
from utils.freeze import head_parameters
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed
from utils.train_heads import kd_loss, train_heads  # noqa: F401  (kd_loss re-exported for older imports)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--positions", type=int, nargs="+", default=None)
    parser.add_argument("--archs", nargs="+", default=None)
    parser.add_argument("--config", default=None, help="JSON with positions/archs[/rule] (e.g. outputs/nas/chosen_config.json)")
    parser.add_argument("--out", default="ee_searched.pth", help="checkpoint file name under checkpoints/")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2

    rule = None
    if args.positions is None:
        config_path = Path(args.config) if args.config else output_path("chosen_config.json", args.quick)
        if not config_path.is_file():
            raise FileNotFoundError(f"{config_path} not found. Run the search or pass --positions/--archs.")
        with open(config_path) as f:
            chosen_config = json.load(f)
        positions, archs, rule = chosen_config["positions"], chosen_config["archs"], chosen_config.get("rule")
    else:
        positions, archs = args.positions, args.archs
        assert archs is not None and len(archs) == len(positions), "--archs must match --positions"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 2.7] Training exit heads on {device} | positions {positions} | archs {archs} | "
          f"seed={args.seed} | quick={args.quick}", flush=True)

    train_loader, val_loader, _ = get_cifar10_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, quick=args.quick)

    base_checkpoint = ckpt_path("vgg16_bn_cifar10_base.pth", args.quick)
    model = build_searched_model(base_checkpoint, positions, archs).to(device)
    print(f"[Loaded Frozen Base Backbone] {base_checkpoint}", flush=True)
    print(f"[Trainable Parameters] {sum(p.numel() for p in head_parameters(model)):,}", flush=True)

    saved_checkpoint = ckpt_path(args.out, args.quick)

    def on_improve(epoch, val_accs):
        save_checkpoint(saved_checkpoint, model, run_meta(
            args, positions=list(positions), archs=list(archs), rule=rule, epoch=epoch, head_val_accs=val_accs[:-1],
            final_val_acc=val_accs[-1], base_checkpoint=str(base_checkpoint)))

    train_heads(model, train_loader, val_loader, device, args.epochs, on_improve=on_improve,
                log=lambda s: print(s, flush=True))
    print(f"[Training Complete] Saved Trained Model to {saved_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
