import argparse
import sys
from pathlib import Path

import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from models.backbone import build_backbone
from models.multi_exit import SupernetMultiExitVGG
from utils.ckpt import load_checkpoint
from utils.freeze import BACKBONE_PARTS, assert_backbone_frozen
from utils.runmeta import add_common_args, ckpt_path, run_meta
from utils.seed import set_seed


def cache_split_logits(supernet, dataloader, split_name, device, args):
    print(f"--> Caching logits for split: {split_name} ...", flush=True)
    supernet.eval()

    all_logits = {k: [] for k in supernet.head_keys}
    all_logits["final"] = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits_dict = supernet(inputs)

            for key in all_logits.keys():
                all_logits[key].append(logits_dict[key].cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    cached_data = {
        "targets": torch.cat(all_targets, dim=0),
        "logits": {k: torch.cat(v, dim=0) for k, v in all_logits.items()},
        "meta": run_meta(args, split=split_name),
    }

    out_file = ckpt_path(f"cached_logits/{split_name}_cached_logits.pt", args.quick)
    torch.save(cached_data, out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"    Saved {split_name} logits to {out_file} ({size_mb:.2f} MB)", flush=True)


def main():
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 2.3] Caching Candidate Logits on {device} | seed={args.seed} | quick={args.quick}", flush=True)

    _, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, quick=args.quick)

    # Backbone first (strict), then the supernet checkpoint (strict); its backbone copy must be identical.
    base_ckpt = ckpt_path("vgg16_bn_cifar10_base.pth", args.quick)
    supernet_ckpt = ckpt_path("supernet_heads.pth", args.quick)
    vgg_base = build_backbone()
    load_checkpoint(base_ckpt, vgg_base)
    supernet = SupernetMultiExitVGG(vgg_base.features, vgg_base.avgpool, vgg_base.classifier)
    base_state = {k: v.clone() for k, v in supernet.state_dict().items() if k.split(".")[0] in BACKBONE_PARTS}
    load_checkpoint(supernet_ckpt, supernet)
    for k, v in base_state.items():
        assert torch.equal(supernet.state_dict()[k], v), f"backbone tensor {k} differs between checkpoints"
    assert_backbone_frozen(supernet)
    print(f"[Loaded] {base_ckpt} + {supernet_ckpt}", flush=True)

    supernet = supernet.to(device)
    cache_split_logits(supernet, val_loader, "val", device, args)
    cache_split_logits(supernet, test_loader, "test", device, args)
    print("  LOGIT CACHING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
