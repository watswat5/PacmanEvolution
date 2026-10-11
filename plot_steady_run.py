#!/usr/bin/env python3
"""Chart a steady-state run's history.csv and diversity.csv.

    python plot_steady_run.py runs/<run>/seed_42 [--snapshot] [--smooth 50]

--snapshot first copies both files into snapshot_<date>_<time>/ beside them
(safe while the run is still writing) and charts the copies there. Writes one
figure of six panels, steady_run.png: window fitness, outcomes of the window
games, tree size, tree depth, move diversity, and distinct behaviours and
expressions. History series are drawn as a trailing rolling mean (--smooth
generations; 1 disables); the diversity series are drawn as recorded.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_experiment_comparison import GRID, INK, MUTED, SECONDARY, SURFACE, rolling_mean

# Fixed categorical order of the project's validated palette.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # A copy taken mid-write can end in a half-written line.
    return [row for row in rows if None not in row.values() and "" not in row.values()]


def panel(axis, generations, series, title, ylabel, smooth, percent=False, markers=()):
    axis.set_facecolor(SURFACE)
    for generation, text in markers:
        axis.axvline(generation, color=MUTED, linewidth=1, linestyle=(0, (1, 3)))
        axis.annotate(text, xy=(generation, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                      textcoords="offset points", ha="left", va="top", fontsize=8, color=MUTED)
    for (label, values), color in zip(series, (INK,) + SERIES if len(series) <= 2 else SERIES):
        if smooth > 1:
            values = rolling_mean(values, smooth)
        if percent:
            values = [100 * value for value in values]
        axis.plot(generations, values, color=color, linewidth=2 if color == INK else 1.6, label=label,
                  solid_capstyle="round")
    axis.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold")
    axis.set_ylabel(ylabel, color=SECONDARY, fontsize=9.5)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axis.margins(x=0.01)
    if len(series) > 1:
        legend = axis.legend(loc="best", frameon=False, fontsize=8.5, handlelength=1.4)
        for text in legend.get_texts():
            text.set_color(INK)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a seed folder holding history.csv and diversity.csv")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--smooth", type=int, default=50)
    parser.add_argument("--mark", action="append", default=[], metavar="GENERATION:TEXT",
                        help="draw a labelled vertical line, e.g. 1963:resumed")
    args = parser.parse_args()

    folder = args.run
    if args.snapshot:
        folder = args.run / f"snapshot_{datetime.now():%Y%m%d_%H%M}"
        folder.mkdir(exist_ok=True)
        for name in ("history.csv", "diversity.csv"):
            if (args.run / name).exists():
                shutil.copyfile(args.run / name, folder / name)
    history = read_rows(folder / "history.csv")
    diversity = read_rows(folder / "diversity.csv")
    if not history:
        raise SystemExit(f"No history.csv rows in {folder}")
    markers = [(int(text.split(":", 1)[0]), text.split(":", 1)[1]) for text in args.mark]

    generation = [int(row["generation"]) for row in history]
    column = lambda key: [float(row[key]) for row in history]
    games = [sum(int(row[key]) for key in ("cleared", "died", "stagnated", "timed_out")) for row in history]
    share = lambda key: [int(row[key]) / total for row, total in zip(history, games)]

    figure, axes = plt.subplots(3, 2, figsize=(15, 13), facecolor=SURFACE)
    smooth_note = f"{args.smooth}-generation rolling mean" if args.smooth > 1 else "as recorded"
    panel(axes[0][0], generation, [("Best tree", column("best_fitness")), ("Population average", column("average_fitness"))],
          "Fitness over the board window", "Fitness", args.smooth, markers=markers)
    panel(axes[0][1], generation, [("Cleared", share("cleared")), ("Died", share("died")),
                                   ("Stagnated", share("stagnated")), ("Timed out", share("timed_out"))],
          "Outcomes of the population's window games", "Share of games (%)", args.smooth, percent=True, markers=markers)
    panel(axes[1][0], generation, [("Population average", column("average_node_count")), ("Best tree", column("best_node_count"))],
          "Tree size", "Nodes", args.smooth, markers=markers)
    panel(axes[1][1], generation, [("Population average", column("average_tree_depth")), ("Best tree", column("best_tree_depth"))],
          "Tree depth", "Depth (levels)", args.smooth, markers=markers)
    if diversity:
        diversity_generation = [int(row["generation"]) for row in diversity]
        panel(axes[2][0], diversity_generation, [("Move diversity", [float(row["move_diversity"]) for row in diversity])],
              "Move diversity: chance two trees pick different moves", "Share of situations (%)", 1, percent=True,
              markers=markers)
        panel(axes[2][1], diversity_generation,
              [("Distinct behaviours", [float(row["distinct_behaviours"]) for row in diversity]),
               ("Distinct expressions", [float(row["distinct_expressions"]) for row in diversity])],
              "Distinct trees in the population", "Trees", 1, markers=markers)
        for axis in axes[2]:
            axis.set_xlim(axes[0][0].get_xlim())
    else:
        for axis in axes[2]:
            axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Generation", color=SECONDARY, fontsize=9.5)
    last = history[-1]
    seed_folder = folder.parent if folder.name.startswith("snapshot_") else folder
    figure.suptitle(f"Steady-state run: {seed_folder.parent.name}/{seed_folder.name}", x=0.04, y=0.985, ha="left",
                    fontsize=15, color=INK, fontweight="bold")
    figure.text(0.04, 0.955, f"Generations 0 to {last['generation']}; fitness, outcomes, size and depth: {smooth_note}; "
                f"diversity from generation {diversity[0]['generation'] if diversity else '-'}, as recorded",
                fontsize=10, color=SECONDARY, ha="left")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.05, hspace=0.32, wspace=0.16)
    output = folder / "steady_run.png"
    figure.savefig(output, dpi=120, facecolor=SURFACE)
    print(f"Wrote {output} ({len(history)} generations, {len(diversity)} diversity rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
