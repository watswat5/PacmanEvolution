#!/usr/bin/env python3
"""How each run's champions play full games on fresh boards, over generations.

    python champion_curves.py runs/experiment_endgame --stagnation-scaling inverse

For every <directory>/<condition>/seed_<seed>/ run, the champion of every
--every-th generation (generation_NNNN_best.pkl) plays the same --boards fresh
full-game boards, under one stagnation rule for all runs. Writes one chart per
outcome to <directory>/champion_curves/: pellets eaten, boards cleared, deaths,
and games that ended neither way. One line per condition, the mean over seeds
(--seed N charts one seed, in a seed_N subfolder).

Use this where training numbers are not comparable between conditions: runs
with endgame training boards, a different pellet scale, or a different
stagnation rule. Results are cached in the output folder, so rerunning during
an experiment only plays the new generations. Runs in progress are included up
to their latest champion.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import GeneticAlgo.train_board_graph as trainer

# Saved trees and checkpoints refer to classes of the trainer run as a script.
sys.modules["__main__"].GenerationStats = trainer.GenerationStats

import playoff
import plot_experiment_comparison as charts

BOARD_SEED = 2_300_000_000  # separate from the playoff and comparison boards
FIELDS = ["condition", "seed", "generation", "pellets", "cleared", "died", "stagnated", "timed_out"]


def sampled_generations(run: Path, every: int) -> list[int]:
    available = sorted(int(path.name.split("_")[1]) for path in run.glob("generation_*_best.pkl"))
    return [generation for generation in available if (generation + 1) % every == 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--every", type=int, default=10, help="use every N-th generation's champion (default: 10)")
    parser.add_argument("--boards", type=int, default=30, help="fresh boards each champion plays (default: 30)")
    parser.add_argument("--stagnation-scaling", choices=("fixed", "inverse"), default="fixed",
                        help="stagnation cutoff for these games (default: fixed)")
    parser.add_argument("--seed", default="all",
                        help="chart one seed only, e.g. while later seeds are still running (default: all)")
    parser.add_argument("--workers", type=int, default=0, help="processes (default: all CPUs)")
    parser.add_argument("--output", type=Path, help="default: <directory>/champion_curves")
    args = parser.parse_args()

    output = args.output or args.directory / "champion_curves"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / f"champion_curves_{args.stagnation_scaling}_{args.boards}_boards.csv"
    cache: dict[tuple[str, str, int], dict] = {}
    if cache_path.exists():
        with cache_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cache[(row["condition"], row["seed"], int(row["generation"]))] = row

    def save() -> None:
        with cache_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(cache[key] for key in sorted(cache))

    runs = sorted(path for path in args.directory.glob("*/seed_*") if path.is_dir())
    board_seeds = [BOARD_SEED + index for index in range(args.boards)]
    workers = args.workers or len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for run in runs:
            condition, seed = run.parent.name, run.name.removeprefix("seed_")
            missing = [generation for generation in sampled_generations(run, args.every)
                       if (condition, seed, generation) not in cache]
            if not missing:
                continue
            trees = {}
            for generation in missing:
                with (run / f"generation_{generation:04d}_best.pkl").open("rb") as handle:
                    trees[generation] = pickle.load(handle)
            # A champion often holds for several generations: play each tree once.
            distinct = {str(tree): tree for tree in trees.values()}
            results = dict(zip(distinct, playoff.play(list(distinct.values()), board_seeds, executor, workers,
                                                     args.stagnation_scaling)))
            for generation, tree in trees.items():
                result = results[str(tree)]
                cache[(condition, seed, generation)] = {
                    "condition": condition, "seed": seed, "generation": generation,
                    "pellets": f"{result['pellets']:.4f}", "cleared": result["cleared"],
                    "died": result["died"], "stagnated": result["stagnated"], "timed_out": result["timed_out"],
                }
            save()
            print(f"{condition} seed {seed}: {len(missing)} generations, {len(distinct)} distinct champions")
    if not cache:
        raise SystemExit(f"No generation_NNNN_best.pkl files under {args.directory}/<condition>/seed_<seed>/")

    by_run: dict[str, dict[str, list[dict]]] = {}
    for (condition, seed, _), row in sorted(cache.items()):
        by_run.setdefault(condition, {}).setdefault(seed, []).append(row)
    by_run = dict(sorted(by_run.items(), key=lambda item: charts.condition_order(item[0])))
    seeds = sorted({seed for by_seed in by_run.values() for seed in by_seed})
    if args.seed != "all":
        seeds = [args.seed]
        by_run = {name: by_seed for name, by_seed in by_run.items() if args.seed in by_seed}
        if not by_run:
            raise SystemExit(f"No champions for seed {args.seed}")

    def share(*keys):
        return lambda row: 100.0 * sum(int(row[key]) for key in keys) / args.boards

    rule = ("cutoff lengthens as the board empties" if args.stagnation_scaling == "inverse"
            else "50-step cutoff")
    subtitle = (f"Every {args.every}th generation's champion on the same {args.boards} fresh full-game boards "
                f"({rule}); {'seed' if len(seeds) == 1 else 'mean of seeds'} {', '.join(seeds)}")
    note = "A run in progress is drawn up to its latest champion; the mean uses the seeds that have reached each point."
    metrics = [
        ("curve_pellets", "Champion: pellets eaten on fresh boards", "Pellets eaten (%)",
         lambda row: 100.0 * float(row["pellets"]), False, note),
        ("curve_cleared", "Champion: fresh boards cleared", "Share of boards (%)", share("cleared"), False, note),
        ("curve_died", "Champion: fresh-board games ending in death", "Share of boards (%)", share("died"), False, note),
        ("curve_unfinished", "Champion: fresh-board games cut off or out of steps", "Share of boards (%)",
         share("stagnated", "timed_out"), False, note),
    ]
    chart_output = output if args.seed == "all" else output / f"seed_{args.seed}"
    chart_output.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        charts.plot_metric(by_run, metric, seeds, 1, chart_output, subtitle=subtitle,
                           generation_of=lambda index: (index + 1) * args.every - 1)
    print(f"Wrote {len(metrics)} charts to {chart_output}; results cached in {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
