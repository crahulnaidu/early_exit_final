"""Section 1 / Step 1b: train the VGG16-BN backbone with the CIFAR-style classifier (models/backbone.py).

Recipe unchanged from the source 01: SGD lr 0.025, momentum 0.9, wd 5e-4, cosine over the epochs
(eta_min 1e-4), batch 128, grad-clip 1.0, NaN/Inf-loss batches skipped. Checkpoint = best VAL accuracy.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from models.backbone import build_backbone
from utils.ckpt import load_checkpoint, save_checkpoint
from utils.flops import BACKBONE_FLOPS, CLASSIFIER_FLOPS, TOTAL_BASE_FLOPS
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed


def evaluate(model, loader, device, counts=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += model(inputs).argmax(dim=-1).eq(targets).sum().item()
            total += targets.size(0)
    return (100.0 * correct / total, correct, total) if counts else 100.0 * correct / total


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 1] Training baseline VGG-16-BN (CIFAR head) on {device} | seed={args.seed} | "
          f"epochs={args.epochs} | quick={args.quick}", flush=True)

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, quick=args.quick)

    model = build_backbone().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    checkpoint_path = ckpt_path("vgg16_bn_cifar10_base.pth", args.quick)
    log_path = output_path("train_backbone_log.csv", args.quick)
    best_val_acc, best_epoch = -1.0, -1

    with open(log_path, "w", newline="") as log_f:
        log = csv.writer(log_f)
        log.writerow(["seed", "epoch", "train_loss", "skipped_batches", "val_acc", "best_val_acc", "epoch_seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running_loss, skipped = 0.0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)

                # Skip backward pass if loss is corrupted
                if torch.isnan(loss) or torch.isinf(loss):
                    skipped += 1
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()

            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                save_checkpoint(checkpoint_path, model, run_meta(
                    args, arch="vgg16_bn_cifar_head_512", epoch=epoch, val_acc=val_acc))
            seconds = time.time() - t0
            log.writerow([args.seed, epoch, running_loss / len(train_loader), skipped, val_acc, best_val_acc,
                          round(seconds, 2)])
            log_f.flush()
            print(f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {running_loss / len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.2f}% | Best Val: {best_val_acc:.2f}% | {seconds:.1f}s", flush=True)

    # Final test evaluation of the best-val checkpoint (test is never used for selection)
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    test_acc, test_correct, test_total = evaluate(model, test_loader, device, counts=True)

    results = run_meta(args, best_epoch=best_epoch, val_acc=best_val_acc, test_acc=test_acc, test_correct=test_correct,
                       test_total=test_total, params=n_params,
                       flops_backbone=BACKBONE_FLOPS, flops_classifier=CLASSIFIER_FLOPS,
                       flops_total=TOTAL_BASE_FLOPS, flops_convention="1 MAC = 2 FLOPs; see utils/flops.py",
                       checkpoint=str(checkpoint_path))
    with open(output_path("backbone_results.json", args.quick), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Baseline] best epoch {best_epoch} | Val {best_val_acc:.2f}% | Test {test_acc:.2f}% | "
          f"params {n_params:,} | FLOPs {TOTAL_BASE_FLOPS / 1e6:.2f} M "
          f"(features {BACKBONE_FLOPS / 1e6:.2f} M + classifier {CLASSIFIER_FLOPS / 1e6:.3f} M)", flush=True)
    print(f"[Saved] {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
