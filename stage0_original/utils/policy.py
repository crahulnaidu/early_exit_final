import torch
from utils.flops import get_vgg16_layer_flops, get_head_flops_and_params
from models.multi_exit import get_channel_count_at_layer, CANDIDATE_POSITIONS
from torchvision.models import vgg16_bn

CUMULATIVE_BACKBONE_FLOPS, TOTAL_BASE_FLOPS = get_vgg16_layer_flops()

def evaluate_config_on_logits(cached_data, positions, archs, thresholds, criterion_type="logit_margin"):
    """
    Evaluates a candidate exit configuration on cached logits without forward passes.
    Returns:
        accuracy, avg_flops, saved_pct, head_overhead_flops_pct, exit_counts
    """
    targets = cached_data["targets"]
    N = targets.size(0)

    num_exits = len(positions)
    exit_counts = [0] * (num_exits + 1) # E1..EK, Final
    correct = 0

    # Retrieve channel dimensions
    vgg_base = vgg16_bn()
    in_channels = [get_channel_count_at_layer(vgg_base.features, pos) for pos in positions]

    # Calculate individual head FLOPs and costs
    head_flops = [get_head_flops_and_params(in_c, arch)[0] for in_c, arch in zip(in_channels, archs)]
    
    # cost_k = backbone(0..p_k) + sum_{j<=k} head_j
    cost_k = []
    for k in range(num_exits):
        cum_head_flops = sum(head_flops[:k+1])
        cost_k.append(CUMULATIVE_BACKBONE_FLOPS[positions[k]] + cum_head_flops)

    # cost_fin = backbone(all) + sum_{j<=K} head_j + classifier
    cost_fin = TOTAL_BASE_FLOPS + sum(head_flops)

    total_flops_spent = 0.0

    # Retrieve cached logits for selected heads
    head_logits = [cached_data["logits"][f"pos_{pos}_arch_{arch}"] for pos, arch in zip(positions, archs)]
    final_logits = cached_data["logits"]["final"]

    for i in range(N):
        exited = False
        for k in range(num_exits):
            logits = head_logits[k][i]
            
            if criterion_type == "logit_margin":
                top2 = torch.topk(logits, 2).values
                score = (top2[0] - top2[1]).item()
            elif criterion_type == "max_softmax":
                probs = torch.softmax(logits, dim=-1)
                score = torch.max(probs).item()
            elif criterion_type == "max_logit":
                score = torch.max(logits).item()
            else:
                raise ValueError(f"Unknown criterion: {criterion_type}")

            if score >= thresholds[k]:
                exited = True
                exit_counts[k] += 1
                total_flops_spent += cost_k[k]
                pred = logits.argmax().item()
                if pred == targets[i].item():
                    correct += 1
                break

        if not exited:
            exit_counts[-1] += 1
            total_flops_spent += cost_fin
            pred = final_logits[i].argmax().item()
            if pred == targets[i].item():
                correct += 1

    accuracy = 100.0 * correct / N
    avg_flops = total_flops_spent / N
    saved_pct = 100.0 * (1.0 - avg_flops / TOTAL_BASE_FLOPS)
    head_overhead_pct = 100.0 * (sum(head_flops) / TOTAL_BASE_FLOPS)

    return {
        "accuracy": accuracy,
        "avg_flops": avg_flops,
        "saved_pct": saved_pct,
        "head_overhead_pct": head_overhead_pct,
        "exit_counts": exit_counts,
    }
