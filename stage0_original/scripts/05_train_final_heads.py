import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders, set_seed
from torchvision.models import vgg16_bn
from models.searched_exit_model import SearchedMultiExitVGG

def kd_loss(student_logits, teacher_logits, target, temperature=3.0, alpha=0.3):
    """
    Combined Cross-Entropy and Knowledge Distillation Loss.
    """
    ce_loss = F.cross_entropy(student_logits, target)
    p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl_div = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    return (1.0 - alpha) * ce_loss + alpha * kl_div

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Section 2.7] Final Training of Chosen Exit Heads on Device: {device}", flush=True)

    # Load NAS selected configuration
    config_path = "checkpoints/chosen_config.json"
    if not Path(config_path).exists():
        print(f"[Error] Configuration file {config_path} not found. Run Day 4 first!", flush=True)
        return

    with open(config_path, "r") as f:
        chosen_config = json.load(f)

    positions = chosen_config["positions"]
    archs = chosen_config["archs"]

    print(f"[Selected Positions] {positions} | [Selected Archs] {archs}", flush=True)

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=128, seed=42)

    # Load frozen base model
    vgg_base = vgg16_bn(num_classes=10)
    base_checkpoint = "checkpoints/vgg16_bn_cifar10_base.pth"
    if Path(base_checkpoint).exists():
        vgg_base.load_state_dict(torch.load(base_checkpoint, map_location=device))
        print(f"[Loaded Frozen Base Backbone] {base_checkpoint}", flush=True)

    model = SearchedMultiExitVGG(
        features=vgg_base.features,
        avgpool=vgg_base.avgpool,
        classifier=vgg_base.classifier,
        positions=positions,
        archs=archs
    ).to(device)

    # Optimize only head parameters
    trainable_params = [p for p in model.exit_heads.parameters() if p.requires_grad]
    print(f"[Trainable Parameters] {sum(p.numel() for p in trainable_params):,}", flush=True)

    optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    best_val_acc = 0.0
    saved_checkpoint = "checkpoints/ee_searched.pth"

    print("\n==========================================================================================", flush=True)
    print("  SECTION 2.7: FINAL HEAD TRAINING WITH KNOWLEDGE DISTILLATION (60 EPOCHS)", flush=True)
    print("==========================================================================================", flush=True)

    for epoch in range(1, 61):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits_list = model(inputs)
            teacher_logits = logits_list[-1] # Final classifier is Teacher

            loss = 0.0
            for k in range(len(positions)):
                loss += kd_loss(logits_list[k], teacher_logits, targets, temperature=3.0, alpha=0.3)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation set evaluation
        model.eval()
        head_corrects = [0] * (len(positions) + 1)
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits_list = model(inputs)
                total_samples += targets.size(0)

                for k, logits in enumerate(logits_list):
                    preds = logits.argmax(dim=-1)
                    head_corrects[k] += preds.eq(targets).sum().item()

        val_accs = [100.0 * c / total_samples for c in head_corrects]
        mean_head_acc = sum(val_accs[:-1]) / len(positions)

        if mean_head_acc > best_val_acc:
            best_val_acc = mean_head_acc
            torch.save(model.state_dict(), saved_checkpoint)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:2d}/60] | Loss: {running_loss/len(train_loader):.4f} | Head Accs: {[round(a, 2) for a in val_accs[:-1]]}% | Teacher: {val_accs[-1]:.2f}%", flush=True)

    print("\n==========================================================================================", flush=True)
    print(f"[Training Complete] Saved Trained Model to {saved_checkpoint}", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
