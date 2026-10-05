#!/usr/bin/env python3
"""Compare multi-seed runs with each other, for example islands against
separate seeds.

    python compare_runs.py islands=runs/<island run> separate=runs/<overnight run> --through 100

Each run is a folder with seed_NN/ subfolders. Only seeds present in every run
are used, and only generations 0 .. --through-1, so the runs are compared over
the same span. Writes one chart per metric (one line per run: the mean over
its seeds) and summary dot plots (every seed shown) to --output.

Charts of the 10-board tournament score are left out unless --tournament is
given, because runs can use different tournament boards (island runs do).
Training fitness, outcome rates and the original-maze score are comparable
when the runs share seeds, since a seed fixes the training boards.

--fresh-boards N also plays each seed's best champion from that span (by its
own run's tournament score) on N fresh boards, the same for every run.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import statistics
from pathlib import Path

import plot_experiment_comparison as charts

RUN_COLORS = (charts.INK, "#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TOURNAMENT_METRICS = {"champion_10_board", "champion_10_board_best_so_far", "champion_10_board_pellets"}
FRESH_BOARD_SEED = 2_200_000_000  # separate from the playoff's board sets


def read_run(directory: Path, through: int) -> dict[str, list[dict]]:
    runs = {}
    for history in sorted(directory.glob("seed_*/history.csv")):
        with history.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[:through]
        if rows:
            runs[history.parent.name.removeprefix("seed_")] = rows
    return runs


def best_champion(directory: Path, seed: str, rows: list[dict]):
    best = max(rows, key=lambda row: float(row["tournament_fitness"]))
    path = directory / f"seed_{seed}" / f"generation_{int(best['generation']):04d}_best.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle), int(best["generation"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="LABEL=DIRECTORY, the reference run first")
    parser.add_argument("--through", type=int, required=True, help="use generations 0 .. N-1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--tournament", action="store_true",
                        help="include tournament-score charts (only if the runs share tournament boards)")
    parser.add_argument("--fresh-boards", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    directories = {}
    for item in args.runs:
        label, _, directory = item.partition("=")
        directories[label] = Path(directory)
    data = {label: read_run(directory, args.through) for label, directory in directories.items()}
    seeds = sorted(set.intersection(*(set(run) for run in data.values())), key=int)
    short = {label: [seed for seed in seeds if len(run[seed]) < args.through] for label, run in data.items()}
    if not seeds:
        raise SystemExit("The runs have no seed in common")
    for label, missing in short.items():
        if missing:
            print(f"Note: {label} has fewer than {args.through} generations for seed(s) {', '.join(missing)}")
    runs = {label: {seed: run[seed] for seed in seeds} for label, run in data.items()}
    for index, label in enumerate(runs):
        charts.CONDITION_COLORS[label] = RUN_COLORS[index % len(RUN_COLORS)]
    charts.REFERENCE_LINES.add(next(iter(runs)))
    args.output.mkdir(parents=True, exist_ok=True)

    subtitle = f"Mean of seeds {', '.join(seeds)}, generations 0 to {args.through - 1}, one line per run"
    count = 0
    for metric in charts.METRICS:
        name, title, ylabel, value, smoothed, _ = metric
        if name in TOURNAMENT_METRICS and not args.tournament:
            continue
        text = subtitle + (f"; {args.smooth}-generation rolling mean" if smoothed and args.smooth > 1 else "")
        charts.plot_metric(runs, (name, title, ylabel, value, smoothed, ""), seeds, args.smooth,
                           args.output, subtitle=text)
        count += 1

    # Summaries over the last 20 generations of the span, one dot per seed.
    seed_text = f"Each dot is one seed ({', '.join(seeds)}); the bar marks the run's mean"
    span = f"generations {max(0, args.through - 20)} to {args.through - 1}"

    def per_run(function):
        return {label: {seed: function(rows) for seed, rows in by_seed.items()} for label, by_seed in runs.items()}

    def tail_mean(value):
        return lambda rows: statistics.mean(value(row) for row in rows[-20:])

    summaries = [
        ("summary_fitness_best", f"Best training fitness, {span}", "Fitness",
         per_run(tail_mean(charts.column("best_fitness"))), "{:,.0f}"),
        ("summary_fitness_average", f"Average training fitness, {span}", "Fitness",
         per_run(tail_mean(charts.column("average_fitness"))), "{:,.0f}"),
        ("summary_clear_rate", f"Training games cleared, {span}", "Share of games (%)",
         per_run(tail_mean(charts.rate("cleared"))), "{:.1f}%"),
        ("summary_death_rate", f"Training games ending in death, {span}", "Share of games (%)",
         per_run(tail_mean(charts.rate("died"))), "{:.1f}%"),
        ("summary_node_count", f"Average tree size, generation {args.through - 1}", "Nodes",
         per_run(lambda rows: float(rows[-1]["average_node_count"])), "{:.0f}"),
    ]
    fresh_rows = []
    if args.fresh_boards:
        import os
        from concurrent.futures import ProcessPoolExecutor

        import playoff

        entries = []
        for label, by_seed in runs.items():
            for seed, rows in by_seed.items():
                tree, generation = best_champion(directories[label], seed, rows)
                entries.append((label, seed, generation, tree))
        workers = args.workers or len(os.sched_getaffinity(0))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = playoff.play([entry[3] for entry in entries],
                                   [FRESH_BOARD_SEED + index for index in range(args.fresh_boards)],
                                   executor, workers)
        pellets = {label: {} for label in runs}
        clears = {label: {} for label in runs}
        for (label, seed, generation, tree), result in zip(entries, results):
            pellets[label][seed] = 100 * result["pellets"]
            clears[label][seed] = result["cleared"]
            fresh_rows.append([label, seed, generation, len(tree), f"{result['pellets']:.4f}",
                               result["cleared"], result["died"], result["stagnated"], f"{result['fitness']:.0f}"])
        note = (f"Each seed's best champion from generations 0 to {args.through - 1} on "
                f"{args.fresh_boards} fresh boards, the same boards for every run.")
        summaries += [
            ("summary_fresh_pellets", f"Pellets eaten on {args.fresh_boards} fresh boards", "Pellets eaten (%)",
             pellets, "{:.1f}%"),
            ("summary_fresh_clears", f"Fresh boards cleared, out of {args.fresh_boards}", "Boards cleared",
             clears, "{:.1f}"),
        ]
        with (args.output / "fresh_boards.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "seed", "champion_generation", "nodes", "pellets", "cleared", "died",
                             "stagnated", "fitness"])
            writer.writerows(fresh_rows)
    for name, title, xlabel, values, fmt in summaries:
        note = ""
        if name.startswith("summary_fresh"):
            note = (f"Each seed's best champion from generations 0 to {args.through - 1} on "
                    f"{args.fresh_boards} fresh boards, the same boards for every run.")
        charts.plot_summary(name, title, seed_text, xlabel, values, seeds, args.output, note, fmt)
        count += 1
        means = {label: statistics.mean(by_seed.values()) for label, by_seed in values.items()}
        print(f"{title}: " + "; ".join(
            f"{label} {fmt.format(means[label])} ({' '.join(fmt.format(by_seed[seed]) for seed in seeds)})"
            for label, by_seed in values.items()))
    print(f"\nWrote {count} charts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
