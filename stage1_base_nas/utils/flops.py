"""Analytic FLOPs and parameter counts (no forward pass), for 32x32 CIFAR inputs and the backbone in
models/backbone.py.

Conventions (state these in the paper):
  conv / linear : 1 MAC = 2 FLOPs (conv 2*Cin/groups*Cout*k*k*Hout*Wout, linear 2*in*out); bias adds not counted
  BatchNorm     : 2 * C*H*W           (eval mode: scale + shift)
  ReLU          : 1 * C*H*W           (output elements)
  pooling       : 1 * C*H_in*W_in     (input elements; max-pool and adaptive avg-pool)
  dropout       : 0                   (identity at inference)
Head FLOPs use the ACTUAL feature-map size at the exit position.
"""
from models.backbone import FEATURE_SHAPES, LAYER_SPECS, NUM_CLASSES

CATEGORIES = ("conv", "linear", "bn", "relu", "pool")


def _empty():
    return {c: 0 for c in CATEGORIES}


def layer_flops(spec):
    kind, cin, cout, hi, wi, ho, wo = spec
    f = _empty()
    if kind == "conv":
        f["conv"] = 2 * cin * cout * 9 * ho * wo
    elif kind == "bn":
        f["bn"] = 2 * cout * ho * wo
    elif kind == "relu":
        f["relu"] = cout * ho * wo
    elif kind == "maxpool":
        f["pool"] = cin * hi * wi
    return f


def backbone_flops_breakdown(upto=None):
    """Sum per category over features[0..upto] (inclusive); upto=None -> all of features."""
    last = len(LAYER_SPECS) - 1 if upto is None else upto
    total = _empty()
    for spec in LAYER_SPECS[: last + 1]:
        for c, v in layer_flops(spec).items():
            total[c] += v
    return total


def classifier_flops_breakdown(num_classes=NUM_CLASSES):
    c, h, w = FEATURE_SHAPES[len(LAYER_SPECS) - 1]
    f = _empty()
    f["pool"] = c * h * w                                   # AdaptiveAvgPool2d(1)
    f["linear"] = 2 * (512 * 512 + 512 * num_classes)       # 534,528
    f["relu"] = 512
    return f


def head_flops_breakdown(arch, C, H, W, num_classes=NUM_CLASSES):
    f = _empty()
    hw = H * W
    if arch == "A1":    # GAP -> FC(C, 10)
        f["pool"] = C * hw
        f["linear"] = 2 * C * num_classes
    elif arch == "A2":  # adaptive pool 2x2 -> FC(4C, 10)
        f["pool"] = C * hw
        f["linear"] = 2 * 4 * C * num_classes
    elif arch == "A3":  # adaptive pool 4x4 -> FC(16C, 256) -> ReLU -> FC(256, 10)
        f["pool"] = C * hw
        f["linear"] = 2 * (16 * C * 256 + 256 * num_classes)
        f["relu"] = 256
    elif arch == "A4":  # conv1x1 C->oc + BN + ReLU -> pool 2x2 -> FC(4oc,128) -> ReLU -> FC(128,10)
        oc = max(16, C // 4)
        f["conv"] = 2 * C * oc * hw
        f["bn"] = 2 * oc * hw
        f["relu"] = oc * hw + 128
        f["pool"] = oc * hw
        f["linear"] = 2 * (4 * oc * 128 + 128 * num_classes)
    elif arch == "A5":  # dw3x3 + pw1x1 C->oc + BN + ReLU -> GAP -> FC(oc, 10)
        oc = max(16, C // 2)
        f["conv"] = 2 * C * 9 * hw + 2 * C * oc * hw
        f["bn"] = 2 * oc * hw
        f["relu"] = oc * hw
        f["pool"] = oc * hw
        f["linear"] = 2 * oc * num_classes
    else:
        raise ValueError(f"Unknown arch_name: {arch}")
    return f


def head_params(arch, C, num_classes=NUM_CLASSES):
    """Trainable parameters (weights + biases + BN affine) of a head, matching models/exit_heads.py."""
    if arch == "A1":
        return C * num_classes + num_classes
    if arch == "A2":
        return 4 * C * num_classes + num_classes
    if arch == "A3":
        return (16 * C * 256 + 256) + (256 * num_classes + num_classes)
    if arch == "A4":
        oc = max(16, C // 4)
        return C * oc + 2 * oc + (4 * oc * 128 + 128) + (128 * num_classes + num_classes)
    if arch == "A5":
        oc = max(16, C // 2)
        return 9 * C + C * oc + 2 * oc + (oc * num_classes + num_classes)
    raise ValueError(f"Unknown arch_name: {arch}")


def head_cost(position, arch):
    """(FLOPs, params) of head `arch` (menu name "A1".."A5" or a NAS cfg dict) attached after features[position]."""
    C, H, W = FEATURE_SHAPES[position]
    if isinstance(arch, dict):
        from models.head_builder import config_head_flops_breakdown, config_head_params
        return sum(config_head_flops_breakdown(arch, C, H, W).values()), config_head_params(arch, C)
    return sum(head_flops_breakdown(arch, C, H, W).values()), head_params(arch, C)


CUMULATIVE_BACKBONE_FLOPS = {i: sum(backbone_flops_breakdown(i).values()) for i in range(len(LAYER_SPECS))}
BACKBONE_FLOPS = CUMULATIVE_BACKBONE_FLOPS[len(LAYER_SPECS) - 1]
CLASSIFIER_FLOPS = sum(classifier_flops_breakdown().values())
TOTAL_BASE_FLOPS = BACKBONE_FLOPS + CLASSIFIER_FLOPS   # F_base: plain VGG-16, no heads


def exit_costs(positions, archs):
    """Per-image FLOPs when leaving at exit k (k = 0..K-1) or at the final classifier / flagged (index K).
    cost_k = backbone(0..p_k) + sum_{j<=k} head_j ; cost_fin = F_base + sum_j head_j (Section 3.4)."""
    head_flops = [head_cost(p, a)[0] for p, a in zip(positions, archs)]
    costs = [CUMULATIVE_BACKBONE_FLOPS[p] + sum(head_flops[: k + 1]) for k, p in enumerate(positions)]
    costs.append(TOTAL_BASE_FLOPS + sum(head_flops))
    return costs


def expected_flops(positions, archs, exit_ids):
    """Average theoretical per-sample FLOPs for routed samples (exit_id -1 = flagged, charged cost_fin)."""
    costs = exit_costs(positions, archs)
    K = len(positions)
    total = 0
    for k in range(K + 1):
        total += costs[k] * int((exit_ids == k).sum())
    total += costs[K] * int((exit_ids == -1).sum())
    avg = total / len(exit_ids)
    return {"avg_flops": avg, "saved_pct": 100.0 * (1.0 - avg / TOTAL_BASE_FLOPS),
            "head_flops_overhead_pct": 100.0 * (costs[K] - TOTAL_BASE_FLOPS) / TOTAL_BASE_FLOPS}
