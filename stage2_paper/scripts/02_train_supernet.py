import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from models.backbone import build_backbone
from models.multi_exit import SupernetMultiExitVGG
from utils.ckpt import load_checkpoint, save_checkpoint
from utils.freeze import assert_backbone_frozen, assert_optimizer_heads_only, head_parameters
from utils.runmeta import add_common_args, ckpt_path, run_meta
from utils.seed import set_seed


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 2.2] Training Supernet Heads on {device} | seed={args.seed} | quick={args.quick}", flush=True)

    train_loader, val_loader, _ = get_cifar10_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, quick=args.quick)

    # Load frozen baseline backbone (FileNotFoundError if missing, strict=True)
    vgg_base = build_backbone()
    base_checkpoint = ckpt_path("vgg16_bn_cifar10_base.pth", args.quick)
    load_checkpoint(base_checkpoint, vgg_base)
    print(f"[Loaded Frozen Backbone Checkpoint] {base_checkpoint}", flush=True)

    supernet = SupernetMultiExitVGG(
        features=vgg_base.features,
        avgpool=vgg_base.avgpool,
        classifier=vgg_base.classifier
    )
    supernet = supernet.to(device)

    # Optimize ONLY candidate head parameters
    trainable_params = head_parameters(supernet)
    print(f"[Total Candidate Heads] {len(supernet.head_keys)} | Trainable Head Parameters: "
          f"{sum(p.numel() for p in trainable_params):,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
    assert_optimizer_heads_only(optimizer, supernet)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc_sum = -1.0
    supernet_checkpoint = ckpt_path("supernet_heads.pth", args.quick)

    for epoch in range(1, args.epochs + 1):
        supernet.train()
        assert_backbone_frozen(supernet)
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits_dict = supernet(inputs)

            # Total loss = sum of CE losses over all 85 candidate heads
            total_loss = 0.0
            for key, logits in logits_dict.items():
                if key != "final":
                    total_loss += criterion(logits, targets)

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        scheduler.step()

        # Validation check across candidate heads
        supernet.eval()
        head_corrects = {k: 0 for k in supernet.head_keys}
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits_dict = supernet(inputs)
                total_samples += targets.size(0)

                for k in supernet.head_keys:
                    preds = logits_dict[k].argmax(dim=-1)
                    head_corrects[k] += preds.eq(targets).sum().item()

        mean_val_acc = sum(100.0 * c / total_samples for c in head_corrects.values()) / len(head_corrects)

        if mean_val_acc > best_val_acc_sum:
            best_val_acc_sum = mean_val_acc
            save_checkpoint(supernet_checkpoint, supernet,
                            run_meta(args, epoch=epoch, mean_head_val_acc=mean_val_acc,
                                     base_checkpoint=str(base_checkpoint)))

        print(f"Epoch [{epoch:2d}/{args.epochs}] | Sum Loss: {running_loss/len(train_loader):.4f} | "
              f"Mean Head Val Acc: {mean_val_acc:.2f}% | Best Mean Val Acc: {best_val_acc_sum:.2f}%", flush=True)

    print(f"[Supernet Training Complete] Saved Trained Heads to: {supernet_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
