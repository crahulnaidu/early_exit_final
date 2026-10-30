"""Section 2.5-2.6 search (unchanged logic: random sampling; replaced by Optuna in Step 5)."""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.exit_heads import HEAD_BUILDERS
from models.multi_exit import CANDIDATE_POSITIONS
from utils.policy import evaluate_config_on_logits
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed

ARCH_MENU = list(HEAD_BUILDERS.keys())


def search_random(cached_val, args, num_evals=500, max_k=3):
    set_seed(args.seed)
    print(f"--> Running random search ({num_evals} evaluations) ...", flush=True)

    results = []
    target_base_acc = cached_val["logits"]["final"].argmax(dim=-1).eq(cached_val["targets"]).float().mean().item() * 100.0
    acc_constraint = target_base_acc - 1.0

    best_config = None
    best_score = -1e9
    best_any, best_any_acc = None, -1.0

    for eval_idx in range(1, num_evals + 1):
        # Sample configuration hypothesis
        K = random.randint(1, max_k)
        positions = sorted(random.sample(CANDIDATE_POSITIONS, K))
        archs = [random.choice(ARCH_MENU) for _ in range(K)]
        thresholds = [round(random.uniform(1.0, 4.0), 2) for _ in range(K)]

        res = evaluate_config_on_logits(cached_val, positions, archs, thresholds, criterion_type="logit_margin")

        entry = {
            "seed": args.seed,
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

        config = {"positions": positions, "archs": archs, "thresholds": thresholds, "val_acc": res["accuracy"],
                  "saved_pct": res["saved_pct"], "head_overhead_pct": res["head_overhead_pct"]}
        if res["accuracy"] > best_any_acc:
            best_any, best_any_acc = config, res["accuracy"]

        # Composite score penalizing overhead and rewarding accuracy/savings
        if res["accuracy"] >= acc_constraint:
            score = res["saved_pct"] - (res["head_overhead_pct"] * 2.0)
            if score > best_score:
                best_score = score
                best_config = config

    df = pd.DataFrame(results)
    out_csv = output_path("search_all_configs.csv", args.quick)
    df.to_csv(out_csv, index=False)
    print(f"    Saved all search evaluations to {out_csv}", flush=True)

    if best_config is None:
        if not args.quick:
            raise RuntimeError(f"No sampled configuration reached the accuracy constraint ({acc_constraint:.2f}%).")
        print("[QUICK MODE ONLY] no feasible config (untrained heads); using the most accurate one.", flush=True)
        best_config = best_any
    return best_config


def main():
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    val_cache_path = ckpt_path("cached_logits/val_cached_logits.pt", args.quick)
    if not val_cache_path.is_file():
        raise FileNotFoundError(f"Cached validation logits not found at {val_cache_path}. Run 03_cache_logits.py first.")
    cached_val = torch.load(val_cache_path)

    # 1. Hand-Placed Baseline Evaluation ([6, 13, 23] with A3)
    hand_placed_res = evaluate_config_on_logits(
        cached_val, positions=[6, 13, 23], archs=["A3", "A3", "A3"], thresholds=[2.0, 2.0, 2.0]
    )
    print(f"[Baseline Hand-Placed ([6,13,23]+A3)] Acc: {hand_placed_res['accuracy']:.2f}% | "
          f"Saved: {hand_placed_res['saved_pct']:.2f}% | Overhead: {hand_placed_res['head_overhead_pct']:.2f}%", flush=True)

    # 2. Search
    best_config = search_random(cached_val, args, num_evals=2 if args.quick else 300, max_k=3)
    best_config["meta"] = run_meta(args, hand_placed=hand_placed_res)

    print(f"[Selected] positions {best_config['positions']} | archs {best_config['archs']} | "
          f"tau {best_config['thresholds']} | val {best_config['val_acc']:.2f}% | saved {best_config['saved_pct']:.2f}% | "
          f"overhead {best_config['head_overhead_pct']:.2f}%", flush=True)

    out_json = output_path("chosen_config.json", args.quick)
    with open(out_json, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"Saved chosen configuration to {out_json}", flush=True)


if __name__ == "__main__":
    main()
