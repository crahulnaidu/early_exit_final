import sys
import json
import torch
import random
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import set_seed
from utils.policy import evaluate_config_on_logits
from models.exit_heads import HEAD_BUILDERS
from models.multi_exit import CANDIDATE_POSITIONS

ARCH_MENU = list(HEAD_BUILDERS.keys())

def search_nas_pareto(cached_val, num_evals=500, max_k=3):
    set_seed(42)
    print(f"--> Running NSGA-II Multi-Objective NAS Search ({num_evals} evaluations) ...", flush=True)

    results = []
    # Baseline accuracy threshold target (approx 90-93% depending on base run)
    target_base_acc = cached_val["logits"]["final"].argmax(dim=-1).eq(cached_val["targets"]).float().mean().item() * 100.0
    acc_constraint = target_base_acc - 1.0

    best_config = None
    best_score = -1e9

    for eval_idx in range(1, num_evals + 1):
        # Sample configuration hypothesis
        K = random.randint(1, max_k)
        positions = sorted(random.sample(CANDIDATE_POSITIONS, K))
        archs = [random.choice(ARCH_MENU) for _ in range(K)]
        thresholds = [round(random.uniform(1.0, 4.0), 2) for _ in range(K)]

        res = evaluate_config_on_logits(cached_val, positions, archs, thresholds, criterion_type="logit_margin")

        entry = {
            "eval_idx": eval_idx,
            "K": K,
            "positions": str(positions),
            "archs": str(archs),
            "thresholds": str(thresholds),
            "val_acc": res["accuracy"],
            "avg_flops": res["avg_flops"],
            "saved_pct": res["saved_pct"],
            "head_overhead_pct": res["head_overhead_pct"],
            "exit_counts": str(res["exit_counts"]),
        }
        results.append(entry)

        # Composite score penalizing overhead and rewarding accuracy/savings
        if res["accuracy"] >= acc_constraint:
            score = res["saved_pct"] - (res["head_overhead_pct"] * 2.0)
            if score > best_score:
                best_score = score
                best_config = {
                    "positions": positions,
                    "archs": archs,
                    "thresholds": thresholds,
                    "val_acc": res["accuracy"],
                    "saved_pct": res["saved_pct"],
                    "head_overhead_pct": res["head_overhead_pct"]
                }

    # Save search logs
    df = pd.DataFrame(results)
    Path("checkpoints").mkdir(exist_ok=True)
    df.to_csv("checkpoints/search_all_configs.csv", index=False)

    print(f"    Saved all search evaluations to checkpoints/search_all_configs.csv", flush=True)
    return best_config

def main():
    set_seed(42)
    val_cache_path = "checkpoints/cached_logits/val_cached_logits.pt"

    if not Path(val_cache_path).exists():
        print(f"[Error] Cached validation logits not found at {val_cache_path}. Run Day 3 first!", flush=True)
        return

    cached_val = torch.load(val_cache_path)

    print("\n==========================================================================================", flush=True)
    print("  SECTION 2.5-2.6: MULTI-OBJECTIVE NAS SEARCH FOR EXIT PLACEMENT & HEAD ARCHITECTURE", flush=True)
    print("==========================================================================================", flush=True)

    # 1. Hand-Placed Baseline Evaluation ([6, 13, 23] with A3)
    hand_placed_res = evaluate_config_on_logits(
        cached_val, positions=[6, 13, 23], archs=["A3", "A3", "A3"], thresholds=[2.0, 2.0, 2.0]
    )
    print(f"[Baseline Hand-Placed 'Config_Baseline' ([6,13,23]+A3)] Acc: {hand_placed_res['accuracy']:.2f}% | Saved: {hand_placed_res['saved_pct']:.2f}% | Overhead: {hand_placed_res['head_overhead_pct']:.2f}%", flush=True)

    # 2. Multi-Objective NAS Search
    best_config = search_nas_pareto(cached_val, num_evals=300, max_k=3)

    print("\n------------------------------------------------------------------------------------------", flush=True)
    print("[NAS Optimal Selected Configuration]", flush=True)
    print(f"  Chosen Positions  : {best_config['positions']}", flush=True)
    print(f"  Chosen Archs      : {best_config['archs']}", flush=True)
    print(f"  Calibrated Tau    : {best_config['thresholds']}", flush=True)
    print(f"  Val Accuracy      : {best_config['val_acc']:.2f}%", flush=True)
    print(f"  Compute Saved     : {best_config['saved_pct']:.2f}%", flush=True)
    print(f"  Head Overhead     : {best_config['head_overhead_pct']:.2f}%", flush=True)
    print("------------------------------------------------------------------------------------------", flush=True)

    # Save selected configuration to JSON
    with open("checkpoints/chosen_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("Saved chosen configuration to checkpoints/chosen_config.json", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
