"""Full head training on the frozen backbone (05 recipe), shared by 05_train_final_heads.py and the proxy check.

Recipe: Adam lr 1e-3, wd 1e-4, cosine over `epochs` (eta_min 1e-5), loss per head = 0.7 CE + 0.3 KL(T=3) * T^2 with
the final classifier as teacher, augmented train loader, val evaluated every epoch.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.freeze import assert_backbone_frozen, assert_optimizer_heads_only, head_parameters


def kd_loss(student_logits, teacher_logits, target, temperature=3.0, alpha=0.3):
    """
    Combined Cross-Entropy and Knowledge Distillation Loss.
    """
    ce_loss = F.cross_entropy(student_logits, target)
    p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl_div = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    return (1.0 - alpha) * ce_loss + alpha * kl_div


def train_heads(model, train_loader, val_loader, device, epochs, on_improve=None, log=print):
    """Train all exit heads of `model` (forward -> [heads..., final]). Returns per-epoch head val accuracies and the
    per-head best val accuracy. on_improve(epoch, val_accs) is called when the MEAN head val accuracy improves."""
    trainable_params = head_parameters(model)
    optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
    assert_optimizer_heads_only(optimizer, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    n_heads = len(model.exit_heads)
    best_mean, history = -1.0, []
    best_per_head = [-1.0] * n_heads

    for epoch in range(1, epochs + 1):
        model.train()
        assert_backbone_frozen(model)
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits_list = model(inputs)
            teacher_logits = logits_list[-1]  # Final classifier is Teacher
            loss = 0.0
            for k in range(n_heads):
                loss += kd_loss(logits_list[k], teacher_logits, targets, temperature=3.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        head_corrects = [0] * (n_heads + 1)
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits_list = model(inputs)
                total_samples += targets.size(0)
                for k, logits in enumerate(logits_list):
                    head_corrects[k] += logits.argmax(dim=-1).eq(targets).sum().item()
        val_accs = [100.0 * c / total_samples for c in head_corrects]
        history.append(val_accs)
        best_per_head = [max(b, a) for b, a in zip(best_per_head, val_accs[:-1])]
        mean_head_acc = sum(val_accs[:-1]) / n_heads
        if mean_head_acc > best_mean:
            best_mean = mean_head_acc
            if on_improve is not None:
                on_improve(epoch, val_accs)
        log(f"Epoch [{epoch:2d}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | "
            f"Head Accs: {[round(a, 2) for a in val_accs[:-1]]}% | Teacher: {val_accs[-1]:.2f}%")
    return history, best_per_head
