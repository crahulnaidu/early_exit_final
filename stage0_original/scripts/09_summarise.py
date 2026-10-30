import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def main():
    print("\n==========================================================================================", flush=True)
    print("  SECTION 6.5 & 6.6: FAULT CAMPAIGN SUMMARY & FIGURE GENERATION", flush=True)
    print("==========================================================================================", flush=True)

    csv_path = "checkpoints/fault_campaign_mode_A.csv"
    if not Path(csv_path).exists():
        print(f"[Error] Fault campaign CSV {csv_path} not found. Run Day 8 first!", flush=True)
        return

    df = pd.read_csv(csv_path)

    # 1. Statistical Aggregation per k
    summary_list = []
    k_values = sorted(df['k'].unique())

    print("\n--- STATISTICAL SUMMARY TABLE (MODE A: EXPONENT BIT FLIPS) ---", flush=True)
    print(f"{'k':<3} | {'Acc (Mean ± Std)':<18} | {'Acc (Median)':<12} | {'Acc (Min)':<10} | {'SDC Rate (%)':<14} | {'Flag Rate (%)':<14}", flush=True)
    print("-" * 85, flush=True)

    for k in k_values:
        sub = df[df['k'] == k]
        
        acc_mean = sub['accuracy'].mean()
        acc_std = sub['accuracy'].std(ddof=1) if len(sub) > 1 else 0.0
        acc_med = sub['accuracy'].median()
        acc_min = sub['accuracy'].min()
        
        sdc_mean = sub['sdc_rate'].mean()
        flag_mean = sub['flag_rate'].mean()

        summary_list.append({
            "k": k,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "acc_median": acc_med,
            "acc_min": acc_min,
            "sdc_rate_mean": sdc_mean,
            "flag_rate_mean": flag_mean
        })

        print(f"{k:<3} | {acc_mean:6.2f} ± {acc_std:5.2f}%    | {acc_med:10.2f}%  | {acc_min:8.2f}%  | {sdc_mean:12.2f}%   | {flag_mean:12.2f}%", flush=True)

    summary_df = pd.DataFrame(summary_list)
    summary_csv = "checkpoints/fault_campaign_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[Saved Summary Table] {summary_csv}", flush=True)

    # 2. Plotting Figure F3 & F4 (Accuracy, SDC Rate, Flag Rate vs k)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(summary_df['k'], summary_df['acc_mean'], yerr=summary_df['acc_std'], fmt='-o', capsize=5, label='Mean Acc ± Std', color='navy')
    plt.plot(summary_df['k'], summary_df['acc_median'], '--s', label='Median Acc', color='teal')
    plt.plot(summary_df['k'], summary_df['acc_min'], ':x', label='Worst Case (Min)', color='crimson')
    plt.xlabel("Number of Faulty Weights (k)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Fault Intensity (Mode A)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(summary_df['k'], summary_df['sdc_rate_mean'], '-o', label='SDC Rate (Undetected Errors)', color='darkorange')
    plt.plot(summary_df['k'], summary_df['flag_rate_mean'], '-s', label='Flag Rate (Detected Faults)', color='forestgreen')
    plt.xlabel("Number of Faulty Weights (k)")
    plt.ylabel("Percentage (%)")
    plt.title("SDC Rate vs. Flag Rate (Mode A)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    fig_path = "checkpoints/figure_fault_resilience.png"
    plt.savefig(fig_path, dpi=300)
    print(f"[Saved Plot] {fig_path}", flush=True)
    print("==========================================================================================", flush=True)

if __name__ == "__main__":
    main()
