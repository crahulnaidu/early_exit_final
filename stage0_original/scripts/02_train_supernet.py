import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders, set_seed
from torchvision.models import vgg16_bn
from models.multi_exit import SupernetMultiExitVGG

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 2.2] Training Supernet Heads on Device: {device}", flush=True)

    train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=128, seed=42)

    # Load frozen baseline VGG-16 backbone + avgpool + classifier
    vgg_base = vgg16_bn(num_classes=10)
    base_checkpoint = "checkpoints/vgg16_bn_cifar10_base.pth"
    
    if Path(base_checkpoint).exists():
        vgg_base.load_state_dict(torch.load(base_checkpoint, map_location=device))
        print(f"[Loaded Frozen Backbone Checkpoint] {base_checkpoint}", flush=True)

    supernet = SupernetMultiExitVGG(
        features=vgg_base.features, 
        avgpool=vgg_base.avgpool, 
        classifier=vgg_base.classifier
    )
    supernet = supernet.to(device)

    # Optimize ONLY candidate head parameters
    trainable_params = [p for p in supernet.parameters() if p.requires_grad]
    print(f"[Total Candidate Heads] 85 | Trainable Head Parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_acc_sum = 0.0
    supernet_checkpoint = "checkpoints/supernet_heads.pth"
    Path("checkpoints").mkdir(exist_ok=True)

    print("\n==========================================================================================", flush=True)
    print("  SECTION 2.2: SUPERNET CANDIDATE HEADS TRAINING (40 EPOCHS)", flush=True)
    print("==========================================================================================", flush=True)

    for epoch in range(1, 41):
        supernet.train()
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
            torch.save(supernet.state_dict(), supernet_checkpoint)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:2d}/40] | Sum Loss: {running_loss/len(train_loader):.4f} | Mean Head Val Acc: {mean_val_acc:.2f}% | Best Mean Val Acc: {best_val_acc_sum:.2f}%", flush=True)

    print("\n==========================================================================================", flush=True)
    print(f"[Supernet Training Complete] Saved Trained Heads to: {supernet_checkpoint}", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
