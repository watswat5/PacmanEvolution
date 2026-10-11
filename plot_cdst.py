#!/usr/bin/env python3
"""Chart a run's game outcomes (CDST: cleared, died, stagnated, timed out).

usage: python plot_cdst.py <seed folder or history.csv> [output.png]

Top: share (0 to 1) of all population training games per generation ending
in each outcome (stacked), so runs with different numbers of games compare
directly. Bottom: the generation champion's validation outcomes as a
rolling rate. Runs over 2000 generations are smoothed with a rolling mean of
generations/250 (BOARD_GRAPH_PLOT_WINDOW overrides; 1 = raw).
"""
import csv, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

source = Path(sys.argv[1])
if source.is_dir():
    source = source / "history.csv"
output = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name("cdst.png")
rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
x = np.array([float(r["generation"]) for r in rows])
window = int(os.environ.get("BOARD_GRAPH_PLOT_WINDOW", "0") or 0) or (
    1 if len(x) <= 2000 else max(2, len(x) // 250))

def rolling(values):
    values = np.asarray(values, dtype=float)
    if window == 1:
        return x, values
    sums = np.cumsum(np.insert(values, 0, 0.0))
    mean = (sums[window:] - sums[:-window]) / window
    return x[window // 2: window // 2 + len(mean)], mean

OUTCOMES = (("cleared", "Cleared", "#16a34a"), ("died", "Died", "#dc2626"),
            ("stagnated", "Stagnated", "#f59e0b"), ("timed_out", "Timed out", "#64748b"))
counts = {k: np.array([float(r[k]) for r in rows]) for k, _, _ in OUTCOMES}
total = sum(counts.values()); total[total == 0] = 1
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
shares = [rolling(counts[k] / total) for k, _, _ in OUTCOMES]
axes[0].stackplot(shares[0][0], *[s for _, s in shares],
                  labels=[label for _, label, _ in OUTCOMES],
                  colors=[c for _, _, c in OUTCOMES], alpha=0.85)
games = int(total[-1])
smooth = "" if window == 1 else f", {window}-generation rolling mean"
axes[0].set(title=f"Population training games by outcome ({games:,} games per generation{smooth})",
            ylabel="Share of games", ylim=(0, 1))
axes[0].legend(loc="upper left", ncol=4, frameon=True)
for k, label, color in OUTCOMES:
    key = f"validation_{k}"
    if key in rows[0] and any(float(r[key]) for r in rows):
        valid = np.array([float(r[key]) for r in rows])
        valid_total = sum(np.array([float(r[f"validation_{o}"]) for r in rows]) for o, _, _ in OUTCOMES)
        valid_total[valid_total == 0] = 1
        axes[1].plot(*rolling(valid / valid_total), label=label, color=color, linewidth=1.8)
axes[1].set(title="Generation champion on its validation board(s)" + smooth,
            ylabel="Share of champions", xlabel="Generation", ylim=(0, 1))
axes[1].legend(loc="upper left", ncol=4, frameon=True)
fig.savefig(output, dpi=160); plt.close(fig)
print(output.resolve())
