"""Paper figures (PNG, 300 dpi) from the numbers in stage2_paper/results/paper/paper_tables.md (T6, T7, T11)
and results/offline/final/runs_exponent.csv.gz (per-k means)."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path("/home1/jtt/early_exit_final/stage2_paper/results/offline/final/runs_exponent.csv.gz")
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
# fixed entity -> colour (categorical slots in order) and marker (secondary encoding)
STYLE = {
    "margin_band@1":   ("margin_band (ours)", "#2a78d6", "o"),
    "sdn_msp":         ("sdn_msp", "#eb6834", "s"),
    "mendnet_clean@1": ("mendnet_clean", "#1baf7a", "^"),
    "energy_low@1":    ("energy_low", "#eda100", "D"),
    "maxlogit_band@1": ("maxlogit_band", "#e87ba4", "P"),
    "energy_band@1":   ("energy_band", "#008300", "X"),
    "msp_low@1":       ("msp_low", "#6250d6", "v"),
    "plain_dnn":       ("plain_dnn", "#7a7974", "o"),
}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def line(ax, x, y, key):
    lab, c, m = STYLE[key]
    plain = key == "plain_dnn"
    ax.plot(x, y, color=c, marker=m, ms=5, lw=2.4 if key == "margin_band@1" else 1.6,
            ls="--" if plain else "-", label=lab, zorder=3 if key == "margin_band@1" else 2,
            markeredgecolor="white", markeredgewidth=0.8)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# 1, 2: SDC and accuracy vs k
r = pd.read_csv(RUNS, usecols=["method", "k", "sdc_rate", "acc_all"])
K5 = ["plain_dnn", "sdn_msp", "mendnet_clean@1", "energy_low@1", "margin_band@1"]
g = r[r.method.isin(K5)].groupby(["method", "k"])[["sdc_rate", "acc_all"]].mean()
for col, name, ylab, log in [("sdc_rate", "sdc_vs_k", "SDC rate (%)", True),
                              ("acc_all", "acc_vs_k", "accuracy under faults (%)", False)]:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for m in K5:
        s = g.loc[m][col]
        line(ax, s.index, s.values, m)
    ax.set_xlabel("number of faulty weights k"); ax.set_ylabel(ylab); ax.set_xticks(range(1, 9))
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=7, loc="lower right" if log else "lower left")
    save(fig, name)

# 3: per-bit SDC, k = 1 (T7); zero values omitted on the log axis
bits = list(range(23, 31))
T7 = {"plain_dnn": [0.000761, 0.00176, 0.0263, 0.24, 0.000231, 0, 0, 36.4],
      "sdn_msp": [0.00172, 0.00482, 0.0578, 0.588, 0.000927, 0.00142, 0.000991, 15.6],
      "mendnet_clean@1": [0.00211, 0.00584, 0.0347, 0.493, 0.00163, 0.00214, 0.000993, 18.5],
      "margin_band@1": [0.00423, 0.00739, 0.053, 0.547, 0.00163, 0.00429, 0.00219, 0.101]}
fig, ax = plt.subplots(figsize=(7, 2.6))
w = 0.2
for i, (m, v) in enumerate(T7.items()):
    lab, c, _ = STYLE[m]
    xs = [b + (i - 1.5) * w for b, y in zip(bits, v) if y > 0]
    ys = [y for y in v if y > 0]
    ax.bar(xs, ys, width=w * 0.9, color=c, label=lab, zorder=2)
ax.set_yscale("log"); ax.set_ylim(1e-4, 100); ax.set_xticks(bits)
ax.set_xlabel("flipped exponent bit (k = 1)"); ax.set_ylabel("SDC rate (%)")
ax.grid(axis="x", visible=False)
ax.annotate("36.4", (30 - 1.5 * w, 36.4), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK)
ax.annotate("0.10", (30 + 1.5 * w, 0.101), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK)
ax.legend(fontsize=7, ncol=4, loc="upper left")
save(fig, "per_bit")

# 4: SDC vs clean FAR over the alpha sweep (T6)
T6 = {"margin_band@1": ([0.28, 0.43, 0.95, 1.77, 4.85, 5.29], [0.233, 0.231, 0.223, 0.21, 0.14, 0.132]),
      "maxlogit_band@1": ([0.39, 0.55, 0.99, 2, 4.83, 10], [0.188, 0.187, 0.181, 0.171, 0.144, 0.104]),
      "energy_band@1": ([0.4, 0.55, 0.96, 2, 5.03, 9.53], [0.188, 0.187, 0.181, 0.173, 0.146, 0.104]),
      "msp_low@1": ([0.28, 0.44, 0.96, 1.88, 5.11, 6.54], [6.29, 6.27, 6.23, 6.17, 5.86, 5.7]),
      "energy_low@1": ([0.4, 0.62, 1.16, 2.15, 5.84, 9.54], [7.19, 7.17, 7.14, 7.08, 6.79, 6.41]),
      "mendnet_clean@1": ([0.36, 0.52, 1.12, 2.14, 5.42, 10.6], [7.02, 7.13, 7.35, 7.69, 8.47, 11.0])}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for m, (x, y) in T6.items():
    line(ax, x, y, m)
ax.set_yscale("log"); ax.set_xlabel("clean false-alarm rate (%)"); ax.set_ylabel("SDC rate (%)")
ax.text(10.8, 0.3, "two-sided bands", ha="right", fontsize=7, color=MUTED)
ax.text(10.8, 3.9, "one-sided detectors", ha="right", fontsize=7, color=MUTED)
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "alpha_sweep")

# 5: latency of the final model (T11)
fig, axs = plt.subplots(1, 2, figsize=(6, 2.4))
for ax, (title, vals) in zip(axs, [("CPU (8 threads)", [11.4, 12.5, 3.42]), ("GPU (A100)", [1.91, 3.03, 1.37])]):
    labels = ["plain", "all exits", "early stop"]
    bars = ax.bar(labels, vals, width=0.55, color=["#7a7974", "#9ec5f4", "#2a78d6"], zorder=2)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:g} ms", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, color=INK)
    ax.set_title(title, fontsize=9, color=INK); ax.set_ylabel("mean latency (ms)"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.18)
axs[0].annotate("3.3x faster", (2, 3.42), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
axs[1].annotate("1.4x faster", (2, 1.37), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
fig.tight_layout()
save(fig, "latency")
print(sorted(p.name for p in OUT.iterdir()))
