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

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 1] Training Frozen Baseline VGG-16 on Device: {device}", flush=True)

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=128, seed=42)

    # Plain VGG-16 BN model
    model = vgg16_bn(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    # Learning rate adjusted to 0.025 with momentum 0.9 for stable convergence
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    best_val_acc = 0.0
    checkpoint_path = "checkpoints/vgg16_bn_cifar10_base.pth"
    Path("checkpoints").mkdir(exist_ok=True)

    print("\n==========================================================================================", flush=True)
    print("  SECTION 1: PLAIN VGG-16 BASELINE TRAINING (100 EPOCHS)", flush=True)
    print("==========================================================================================", flush=True)

    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Skip backward pass if loss is corrupted
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=-1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

        val_acc = 100.0 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/100] | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Best Val: {best_val_acc:.2f}%", flush=True)

    # Final Test Set Evaluation using Best Checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)
            test_correct += preds.eq(targets).sum().item()
            test_total += targets.size(0)

    test_acc = 100.0 * test_correct / test_total
    print("\n==========================================================================================", flush=True)
    print(f"[Baseline Final Results] Best Val Accuracy: {best_val_acc:.2f}% | Test Accuracy: {test_acc:.2f}%", flush=True)
    print(f"[Saved Checkpoint] {checkpoint_path}", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
