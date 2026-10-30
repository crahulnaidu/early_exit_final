import torch

def get_vgg16_layer_flops():
    """
    Analytic FLOPs for VGG-16 with BN (CIFAR-10, 32x32 input).
    Calculates 2 * Cin * Cout * K * K * H * W for Conv2d layers.
    Returns:
        dict mapping layer index -> cumulative backbone FLOPs up to layer_idx
    """
    layers = [
        # Block 1
        ("conv", 3, 64, 3, 32), ("bn", 64, 32), ("relu", 64, 32),
        ("conv", 64, 64, 3, 32), ("bn", 64, 32), ("relu", 64, 32),
        ("pool", 64, 16), # Layer 6
        # Block 2
        ("conv", 64, 128, 3, 16), ("bn", 128, 16), ("relu", 128, 16),
        ("conv", 128, 128, 3, 16), ("bn", 128, 16), ("relu", 128, 16),
        ("pool", 128, 8), # Layer 13
        # Block 3
        ("conv", 128, 256, 3, 8), ("bn", 256, 8), ("relu", 256, 8),
        ("conv", 256, 256, 3, 8), ("bn", 256, 8), ("relu", 256, 8),
        ("conv", 256, 256, 3, 8), ("bn", 256, 8), ("relu", 256, 8),
        ("pool", 256, 4), # Layer 23
        # Block 4
        ("conv", 256, 512, 3, 4), ("bn", 512, 4), ("relu", 512, 4),
        ("conv", 512, 512, 3, 4), ("bn", 512, 4), ("relu", 512, 4),
        ("conv", 512, 512, 3, 4), ("bn", 512, 4), ("relu", 512, 4),
        ("pool", 512, 2), # Layer 33
        # Block 5
        ("conv", 512, 512, 3, 2), ("bn", 512, 2), ("relu", 512, 2),
        ("conv", 512, 512, 3, 2), ("bn", 512, 2), ("relu", 512, 2),
        ("conv", 512, 512, 3, 2), ("bn", 512, 2), ("relu", 512, 2),
        ("pool", 512, 1), # Layer 43
    ]

    cumulative_flops = {}
    current_flops = 0

    for idx, item in enumerate(layers):
        ltype = item[0]
        if ltype == "conv":
            _, cin, cout, k, h = item
            current_flops += 2 * cin * cout * k * k * h * h
        elif ltype == "bn":
            _, c, h = item
            current_flops += 2 * c * h * h
        elif ltype == "relu" or ltype == "pool":
            _, c, h = item
            current_flops += c * h * h

        cumulative_flops[idx] = current_flops

    final_classifier_flops = 2 * (512 * 7 * 7 * 4096 + 4096 * 4096 + 4096 * 10)
    total_base_flops = current_flops + final_classifier_flops

    return cumulative_flops, total_base_flops

# Compute top-level module constants
CUMULATIVE_BACKBONE_FLOPS, TOTAL_BASE_FLOPS = get_vgg16_layer_flops()

def get_head_flops_and_params(in_channels, arch_name, num_classes=10):
    """
    Computes exact analytic FLOPs and parameter counts for exit head architectures A1..A5.
    """
    if arch_name == "A1":
        flops = (in_channels * 1 * 1) + (2 * in_channels * num_classes)
        params = in_channels * num_classes + num_classes
    elif arch_name == "A2":
        flops = (in_channels * 4) + (2 * (in_channels * 4) * num_classes)
        params = (in_channels * 4) * num_classes + num_classes
    elif arch_name == "A3":
        flops = (in_channels * 16) + (2 * (in_channels * 16) * 256) + 256 + (2 * 256 * num_classes)
        params = (in_channels * 16 * 256 + 256) + (256 * num_classes + num_classes)
    elif arch_name == "A4":
        out_c = max(16, in_channels // 4)
        flops = (2 * in_channels * out_c * 1 * 1 * 4 * 4) + (2 * out_c * 4 * 4) + (out_c * 4 * 4) + \
                (out_c * 4) + (2 * (out_c * 4) * 128) + 128 + (2 * 128 * num_classes)
        params = (in_channels * out_c) + (2 * out_c) + (out_c * 4 * 128 + 128) + (128 * num_classes + num_classes)
    elif arch_name == "A5":
        out_c = max(16, in_channels // 2)
        flops = (2 * in_channels * 1 * 3 * 3 * 4 * 4) + (2 * in_channels * out_c * 1 * 1 * 4 * 4) + \
                (2 * out_c * 4 * 4) + (out_c * 4 * 4) + (out_c) + (2 * out_c * num_classes)
        params = (in_channels * 9) + (in_channels * out_c) + (2 * out_c) + (out_c * num_classes + num_classes)
    else:
        raise ValueError(f"Unknown arch_name: {arch_name}")

    return int(flops), int(params)
