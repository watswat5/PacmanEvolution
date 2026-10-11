#!/usr/bin/env python3
"""Plot every condition of an experiment on shared axes, one chart per metric.

    python plot_experiment_comparison.py runs/experiment_ablation --seed 101
    python plot_experiment_comparison.py runs/experiment_ablation --seed all

Reads <directory>/<condition>/seed_<seed>/history.csv and writes one PNG per
metric to <directory>/comparison_seed_<seed>/. Each chart has one line per
condition, so a setting's effect on that metric can be read directly.

With --seed all (the default) each line is the mean over every seed of that
condition, written to <directory>/comparison_all_seeds/, together with
summary_*.png charts that show every seed's end result as a dot beside the
condition's mean. Those are the charts to judge an effect by: a condition
whose dots overlap the baseline's has not shown one. --fresh-boards N adds
the fresh-board summaries (slow the first time; cached in fresh_boards.csv).

With --islands the directory is an island run (seed_<seed>/ folders directly
inside it): each chart has one line per island plus their mean, with the
migration rounds marked, written to <directory>/comparison_islands/.

Per-generation metrics that jump with the rotating boards are drawn as a
rolling mean (--smooth generations, default 5; 1 disables). Tree depth, node
count and the best-so-far curves are drawn as recorded.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour follows the condition, not its rank, so a condition keeps its colour
# in every chart and for every seed. The baseline is the reference and is drawn
# in ink; the others take the validated categorical palette in fixed order.
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CONDITION_COLORS = {
    "baseline": INK,
    "repair": "#2a78d6",
    "mutation10": "#eb6834",
    "step100": "#1baf7a",
    "small_inserts": "#eda100",
    "global": "#e87ba4",
    "grid8x8": "#008300",
    "no_junction": "#4a3aa7",
    # Endgame experiment (a separate directory, so the colours can repeat).
    "scaled_stagnation": "#2a78d6",
    "late_pellets": "#eb6834",
    "endgame": "#1baf7a",
    "baseline_depth16": "#eda100",
}
EXTRA_COLOR = "#e34948"  # any condition not listed above
# Islands take the categorical palette in seed order.
ISLAND_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MEAN_LABEL = "mean of islands"
# Lines drawn in ink and slightly heavier, as the reference for the others.
REFERENCE_LINES = {"baseline", MEAN_LABEL}

# (file name, title, y label, value function, smoothed?, note)
FITNESS_NOTE = (
    "step100 charges 100 per non-scoring step instead of 20, "
    "so its fitness runs lower for the same play."
)


def rate(key: str):
    def value(row: dict) -> float:
        games = sum(int(row[name]) for name in ("cleared", "died", "stagnated", "timed_out"))
        return 100.0 * int(row[key]) / max(1, games)
    return value


def column(key: str, scale: float = 1.0):
    return lambda row: float(row[key]) * scale


METRICS = [
    ("fitness_best", "Best training fitness", "Fitness", column("best_fitness"), True, FITNESS_NOTE),
    ("fitness_average", "Average training fitness", "Fitness", column("average_fitness"), True, FITNESS_NOTE),
    ("fitness_worst", "Worst training fitness", "Fitness", column("worst_fitness"), True, FITNESS_NOTE),
    ("champion_10_board", "Champion fitness on the 10 tournament boards", "Fitness",
     column("tournament_fitness"), True, FITNESS_NOTE),
    ("champion_10_board_best_so_far", "Best champion so far on the 10 tournament boards", "Fitness",
     column("best_tournament_fitness_so_far"), False, FITNESS_NOTE),
    ("champion_10_board_pellets", "Champion pellets eaten on the 10 tournament boards", "Pellets eaten (%)",
     column("tournament_pellet_ratio", 100.0), True, ""),
    ("champion_validation", "Champion fitness on the original maze", "Fitness",
     column("validation_fitness"), True, FITNESS_NOTE + " One board, so this is noisy."),
    ("outcome_cleared", "Training games cleared", "Share of games (%)", rate("cleared"), True, ""),
    ("outcome_died", "Training games ending in death", "Share of games (%)", rate("died"), True, ""),
    ("outcome_stagnated", "Training games cut off for stagnation", "Share of games (%)",
     rate("stagnated"), True, ""),
    ("tree_depth_average", "Average tree depth", "Depth (levels)", column("average_tree_depth"), False, ""),
    ("tree_depth_best", "Depth of the best training tree", "Depth (levels)", column("best_tree_depth"), True, ""),
    ("node_count_average", "Average node count", "Nodes", column("average_node_count"), False, ""),
    ("node_count_best", "Node count of the best training tree", "Nodes", column("best_node_count"), True, ""),
]


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Trailing mean, so a point never uses generations after it."""
    result, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def condition_order(name: str) -> tuple:
    """Baseline first, then the fixed palette order, then anything else."""
    order = list(CONDITION_COLORS)
    return (order.index(name) if name in order else len(order), name)


def read_all(directory: Path) -> dict[str, dict[str, list[dict]]]:
    """{condition: {seed: history rows}} for every run with a history.csv."""
    runs: dict[str, dict[str, list[dict]]] = {}
    for condition in sorted(path for path in directory.iterdir() if path.is_dir()):
        for history in sorted(condition.glob("seed_*/history.csv")):
            with history.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                seed = history.parent.name.removeprefix("seed_")
                runs.setdefault(condition.name, {})[seed] = rows
    return dict(sorted(runs.items(), key=lambda item: condition_order(item[0])))


def spread_labels(positions: list[float], gap: float) -> list[float]:
    """Nudge end-of-line labels apart so they do not overlap."""
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    placed = list(positions)
    for previous, current in zip(order, order[1:]):
        if placed[current] - placed[previous] < gap:
            placed[current] = placed[previous] + gap
    return placed


def plot_metric(runs, metric, seeds: list[str], smooth: int, output: Path,
                subtitle: str | None = None, mean_line: bool = False,
                markers: tuple[int, ...] = (), generation_of=lambda index: index) -> None:
    """One line per condition: that seed's run, or the mean over ``seeds``.

    ``mean_line`` adds the mean of all the lines (used for islands), and
    ``markers`` draws a faint vertical line at each of those generations.
    ``generation_of`` maps a row's position to its generation, for series that
    do not have one row per generation.
    """
    name, title, ylabel, value, smoothed, note = metric
    figure, axis = plt.subplots(figsize=(12, 6.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for generation in markers:
        axis.axvline(generation, color=MUTED, alpha=0.55, linewidth=1.0, linestyle=(0, (1, 3)), zorder=1)
    ends = []
    lines = {}
    for condition, by_seed in runs.items():
        lines[condition] = [[value(row) for row in by_seed[seed]] for seed in seeds if seed in by_seed]
    if mean_line:
        # Only over the generations every line has reached.
        shared = min(len(series[0]) for series in lines.values() if series)
        lines[MEAN_LABEL] = [series[0][:shared] for series in lines.values() if series]
    for condition, series in lines.items():
        if not series:
            continue
        # Average generation by generation over the seeds that reached it.
        length = max(len(values) for values in series)
        generations = [generation_of(index) for index in range(length)]
        values = [
            sum(run[index] for run in series if index < len(run))
            / sum(index < len(run) for run in series)
            for index in range(length)
        ]
        if smoothed and smooth > 1:
            values = rolling_mean(values, smooth)
        color = INK if condition == MEAN_LABEL else CONDITION_COLORS.get(condition, EXTRA_COLOR)
        axis.plot(
            generations, values, color=color, label=condition,
            linewidth=2.6 if condition in REFERENCE_LINES else 2.0,
            zorder=3 if condition in REFERENCE_LINES else 2,
            solid_capstyle="round", solid_joinstyle="round",
        )
        ends.append((condition, generations[-1], values[-1], color))

    # Direct labels at the line ends; the legend repeats them, so identity
    # never rests on colour alone.
    low, high = axis.get_ylim()
    label_y = spread_labels([end[2] for end in ends], (high - low) * 0.035)
    for (condition, x, _, color), y in zip(ends, label_y):
        # A colour swatch beside the text carries identity; the text stays ink.
        axis.annotate(
            "\u25a0", xy=(x, y), xytext=(7, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=color, annotation_clip=False,
        )
        axis.annotate(
            condition, xy=(x, y), xytext=(19, 0), textcoords="offset points",
            va="center", ha="left", fontsize=10, color=INK, annotation_clip=False,
        )

    if subtitle is not None:
        pass
    elif len(seeds) == 1:
        subtitle = f"Seed {seeds[0]}, one line per condition"
    else:
        subtitle = f"Mean of {len(seeds)} seeds ({', '.join(seeds)}), one line per condition"
    if smoothed and smooth > 1 and "rolling mean" not in subtitle:
        subtitle += f"; {smooth}-generation rolling mean"
    figure.suptitle(title, x=0.065, y=0.975, ha="left", fontsize=15, color=INK, fontweight="bold")
    axis.set_title(subtitle, loc="left", fontsize=10.5, color=SECONDARY, pad=10)
    axis.set_xlabel("Generation", color=SECONDARY, fontsize=10.5)
    axis.set_ylabel(ylabel, color=SECONDARY, fontsize=10.5)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9.5, length=0)
    axis.margins(x=0.01)
    legend = axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=min(8, len(lines)),
        frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    if note:
        figure.text(0.065, 0.012, note, fontsize=9, color=MUTED, ha="left", va="bottom")
    # Leave room on the right for the longest end-of-line label.
    longest = max((len(name) for name in lines), default=0)
    figure.subplots_adjust(left=0.065, right=min(0.885, 0.97 - 0.0075 * longest), top=0.87, bottom=0.17)
    figure.savefig(output / f"{name}.png", dpi=130, facecolor=SURFACE)
    plt.close(figure)


SEED_MARKERS = ("o", "s", "^", "D", "v", "P")


def plot_summary(name, title, subtitle, xlabel, values, seeds, output, note="", fmt="{:,.0f}"):
    """Dot plot: every seed's end result per condition, with the condition mean.

    ``values`` is {condition: {seed: number}}. Conditions are sorted by mean.
    """
    means = {
        condition: sum(by_seed.values()) / len(by_seed)
        for condition, by_seed in values.items() if by_seed
    }
    ranked = sorted(means, key=means.get)
    figure, axis = plt.subplots(figsize=(12, 0.62 * len(ranked) + 2.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    reference = next((name for name in means if name in REFERENCE_LINES), None)
    if reference is not None:
        axis.axvline(means[reference], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    for row, condition in enumerate(ranked):
        color = CONDITION_COLORS.get(condition, EXTRA_COLOR)
        numbers = values[condition]
        axis.plot([min(numbers.values()), max(numbers.values())], [row, row],
                  color=color, linewidth=2, alpha=0.35, zorder=2, solid_capstyle="round")
        for index, seed in enumerate(seeds):
            if seed in numbers:
                axis.plot([numbers[seed]], [row], marker=SEED_MARKERS[index % len(SEED_MARKERS)],
                          markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                          linestyle="none", zorder=3)
        axis.plot([means[condition]], [row], marker="|", markersize=26, markeredgewidth=2.6,
                  color=INK, linestyle="none", zorder=4)
        axis.annotate(fmt.format(means[condition]), xy=(1.012, row), xycoords=("axes fraction", "data"),
                      va="center", ha="left", fontsize=10.5, color=INK, annotation_clip=False)
    axis.annotate("mean", xy=(1.012, len(ranked) - 0.28), xycoords=("axes fraction", "data"),
                  va="bottom", ha="left", fontsize=9, color=MUTED, annotation_clip=False)
    axis.set_yticks(range(len(ranked)))
    axis.set_yticklabels(ranked, fontsize=11, color=INK)
    axis.set_ylim(-0.6, len(ranked) - 0.4)
    axis.set_xlabel(xlabel, color=SECONDARY, fontsize=10.5)
    axis.grid(True, axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="x", colors=MUTED, labelsize=9.5, length=0)
    axis.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], marker=SEED_MARKERS[index % len(SEED_MARKERS)], linestyle="none",
                   markersize=8, color=SECONDARY, label=f"seed {seed}")
        for index, seed in enumerate(seeds)
    ]
    handles.append(plt.Line2D([], [], marker="|", linestyle="none", markersize=14,
                              markeredgewidth=2.4, color=INK, label="mean of seeds"))
    if reference is not None:
        handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                                  label=f"{reference} mean"))
    height = figure.get_size_inches()[1]
    legend = figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.42 / height),
                           ncol=len(handles), frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)
    figure.suptitle(title, x=0.012, y=1 - 0.22 / height, ha="left", fontsize=15, color=INK, fontweight="bold")
    figure.text(0.012, 1 - 0.62 / height, subtitle, fontsize=10.5, color=SECONDARY, ha="left", va="top")
    if note:
        figure.text(0.012, 0.012, note, fontsize=9, color=MUTED, ha="left", va="bottom")
    figure.subplots_adjust(left=0.115, right=0.915, top=1 - 1.0 / height, bottom=1.25 / height)
    figure.savefig(output / f"{name}.png", dpi=130, facecolor=SURFACE)
    plt.close(figure)


def fresh_board_results(directory: Path, runs, board_count: int,
                        stagnation_scaling: str = "fixed") -> dict[tuple[str, str], tuple[float, int]]:
    """Each run's best tree on fresh random boards, cached in fresh_boards.csv."""
    # One cache per stagnation rule, because the rule changes the results.
    suffix = "" if stagnation_scaling == "fixed" else f"_{stagnation_scaling}"
    cache_path = directory / f"fresh_boards{suffix}.csv"
    cache = {}
    if cache_path.exists():
        with cache_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["boards"]) == board_count:
                    cache[(row["condition"], row["seed"])] = (float(row["pellets"]), int(row["clears"]))
    import summarize_experiments

    for condition, by_seed in runs.items():
        for seed in by_seed:
            if (condition, seed) in cache:
                continue
            result = summarize_experiments.fresh_board_result(
                directory / condition / f"seed_{seed}", board_count, stagnation_scaling
            )
            if result is None:
                continue
            cache[(condition, seed)] = result
            # Rewrite after every run so an interrupted pass keeps its work.
            with cache_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["condition", "seed", "boards", "pellets", "clears"])
                for (name, run_seed), (pellets, clears) in sorted(cache.items()):
                    writer.writerow([name, run_seed, board_count, f"{pellets:.6f}", clears])
    return cache


def plot_summaries(directory: Path, runs, seeds: list[str], output: Path, fresh_boards: int,
                   stagnation_scaling: str = "fixed") -> int:
    seed_text = f"Each dot is one seed ({', '.join(seeds)}); the bar marks the condition's mean"

    def per_run(function) -> dict:
        return {condition: {seed: function(rows) for seed, rows in by_seed.items()}
                for condition, by_seed in runs.items()}

    def last_20_rate(key):
        return lambda rows: sum(rate(key)(row) for row in rows[-20:]) / len(rows[-20:])

    charts = [
        ("summary_best_10_board", "Best champion on the 10 tournament boards", "Fitness",
         per_run(lambda rows: max(float(row["tournament_fitness"]) for row in rows)), FITNESS_NOTE, "{:,.0f}"),
        ("summary_pellets_10_board", "Pellets eaten by that champion on the 10 tournament boards",
         "Pellets eaten (%)",
         per_run(lambda rows: 100 * float(max(rows, key=lambda row: float(row["tournament_fitness"]))["tournament_pellet_ratio"])),
         "", "{:.1f}%"),
        ("summary_clear_rate", "Training games cleared, last 20 generations", "Share of games (%)",
         per_run(last_20_rate("cleared")), "", "{:.1f}%"),
        ("summary_death_rate", "Training games ending in death, last 20 generations", "Share of games (%)",
         per_run(last_20_rate("died")), "", "{:.1f}%"),
        ("summary_node_count", "Average tree size in the final generation", "Nodes",
         per_run(lambda rows: float(rows[-1]["average_node_count"])), "", "{:.0f}"),
        ("summary_tree_depth", "Average tree depth in the final generation", "Depth (levels)",
         per_run(lambda rows: float(rows[-1]["average_tree_depth"])), "", "{:.1f}"),
    ]
    if fresh_boards:
        fresh = fresh_board_results(directory, runs, fresh_boards, stagnation_scaling)
        note = (f"Each run's best_overall tree on {fresh_boards} random boards that no run "
                "trained or was selected on. This is the fairest comparison.")
        charts += [
            ("summary_fresh_pellets", f"Pellets eaten on {fresh_boards} fresh boards", "Pellets eaten (%)",
             {condition: {seed: 100 * fresh[(condition, seed)][0] for seed in by_seed if (condition, seed) in fresh}
              for condition, by_seed in runs.items()}, note, "{:.1f}%"),
            ("summary_fresh_clears", f"Fresh boards cleared, out of {fresh_boards}", "Boards cleared",
             {condition: {seed: fresh[(condition, seed)][1] for seed in by_seed if (condition, seed) in fresh}
              for condition, by_seed in runs.items()}, note, "{:.1f}"),
        ]
    for name, title, xlabel, values, note, fmt in charts:
        plot_summary(name, title, seed_text, xlabel, values, seeds, output, note, fmt)
    return len(charts)


def plot_islands(directory: Path, smooth: int, output: Path | None) -> int:
    """One line per island, their mean, and the migration rounds."""
    import json

    runs = {}
    for history in sorted(directory.glob("seed_*/history.csv"),
                          key=lambda path: int(path.parent.name.removeprefix("seed_"))):
        with history.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            runs[f"island {history.parent.name.removeprefix('seed_')}"] = {"run": rows}
    if not runs:
        raise SystemExit(f"No seed_*/history.csv under {directory}")
    for index, name in enumerate(runs):
        CONDITION_COLORS[name] = ISLAND_PALETTE[index % len(ISLAND_PALETTE)]
    state = directory / "island_state.json"
    migrations = tuple(json.loads(state.read_text())["migrated"]) if state.exists() else ()
    generations = min(len(by_seed["run"]) for by_seed in runs.values())
    subtitle = f"{len(runs)} islands through generation {generations}, with their mean"
    if migrations:
        subtitle += "; dotted lines mark migrations"
    output = output or directory / "comparison_islands"
    output.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        name, title, ylabel, value, smoothed, note = metric
        text = subtitle + (f"; {smooth}-generation rolling mean" if smoothed and smooth > 1 else "")
        # The fitness footnote is about an ablation condition, not islands.
        plot_metric(runs, (name, title, ylabel, value, smoothed, ""), ["run"], smooth, output,
                    subtitle=text, mean_line=True, markers=migrations)
    print(f"Wrote {len(METRICS)} island charts to {output} (through generation {generations})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", default="all", help="one seed, or 'all' for the mean over seeds")
    parser.add_argument("--fresh-boards", type=int, default=0,
                        help="with --seed all: also summarise each run's best tree on N fresh boards")
    parser.add_argument("--stagnation-scaling", choices=("fixed", "inverse"), default="fixed",
                        help="stagnation cutoff used on the fresh boards (default: fixed)")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--output", type=Path, help="default: <directory>/comparison_seed_<seed>")
    parser.add_argument("--islands", action="store_true",
                        help="the directory is an island run: one line per island plus their mean")
    args = parser.parse_args()
    if args.islands:
        return plot_islands(args.directory, args.smooth, args.output)
    runs = read_all(args.directory)
    all_seeds = sorted({seed for by_seed in runs.values() for seed in by_seed})
    if args.seed == "all":
        seeds, folder = all_seeds, "comparison_all_seeds"
    else:
        seeds, folder = [args.seed], f"comparison_seed_{args.seed}"
        runs = {name: by_seed for name, by_seed in runs.items() if args.seed in by_seed}
    if not runs or not seeds:
        raise SystemExit(f"No seed_{args.seed}/history.csv under {args.directory}")
    output = args.output or args.directory / folder
    output.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(runs, metric, seeds, args.smooth, output)
    count = len(METRICS)
    if args.seed == "all":
        count += plot_summaries(args.directory, runs, seeds, output, args.fresh_boards,
                                args.stagnation_scaling)
    lengths = ", ".join(f"{name} {len(by_seed)}" for name, by_seed in runs.items())
    print(f"Wrote {count} charts to {output}\nSeeds per condition: {lengths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
