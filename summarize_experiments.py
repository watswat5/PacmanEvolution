#!/usr/bin/env python3
"""Compare the conditions of an experiment directory.

Reads <directory>/<condition>/seed_<seed>/history.csv (as written by
run_board_graph_conditions.sh) and prints one row per condition, best first,
with the difference from the baseline. Writes the same table to
<directory>/summary.csv.

With --fresh-boards N, each run's best_overall.pkl also plays N random boards
that no run trained or was selected on, which is the fairest comparison.
--stagnation-scaling inverse plays those boards with the stagnation cutoff
that lengthens as the board empties; use it when any condition trained that
way, so every condition is judged by the same rule.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import statistics
from pathlib import Path

FRESH_BOARD_SEED = 9000


def read_run(run: Path) -> dict | None:
    history = run / "history.csv"
    if not history.exists():
        return None
    with history.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["tournament_fitness"]))
    games = sum(
        int(rows[-1][key]) for key in ("cleared", "died", "stagnated", "timed_out")
    )
    recent = rows[-20:]
    return {
        "generations": len(rows),
        "finished": (run / "summary.txt").exists(),
        "best": float(best["tournament_fitness"]),
        "best_generation": int(best["generation"]),
        "pellets": float(best["tournament_pellet_ratio"]),
        "clear_rate": statistics.mean(int(row["cleared"]) for row in recent) / max(1, games),
        "nodes": float(rows[-1]["average_node_count"]),
    }


def fresh_board_result(
    run: Path, board_count: int, stagnation_scaling: str = "fixed"
) -> tuple[float, int] | None:
    best = run / "best_overall.pkl"
    if not best.exists():
        return None
    import GeneticAlgo.train_board_graph as trainer

    with best.open("rb") as handle:
        tree = pickle.load(handle)
    primitive_set = trainer.build_primitive_set()
    rules = trainer.EvaluationRules(stagnation_scaling=stagnation_scaling)
    ratios, clears = [], 0
    for seed in range(FRESH_BOARD_SEED, FRESH_BOARD_SEED + board_count):
        board = trainer.generate_board(seed)
        _, _, pellet_ratios, labels = trainer.evaluate_population_lockstep(
            [tree], board, 1000, primitive_set, rules=rules
        )
        ratios.append(pellet_ratios[0])
        clears += labels[0] == "cleared"
    return statistics.mean(ratios), clears


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--fresh-boards", type=int, default=0)
    parser.add_argument("--stagnation-scaling", choices=("fixed", "inverse"), default="fixed",
                        help="stagnation cutoff used on the fresh boards (default: fixed)")
    args = parser.parse_args()

    table = []
    for condition in sorted(path for path in args.directory.iterdir() if path.is_dir()):
        runs = []
        for run in sorted(condition.glob("seed_*")):
            result = read_run(run)
            if result is None:
                continue
            result["seed"] = run.name.removeprefix("seed_")
            if args.fresh_boards:
                result["fresh"] = fresh_board_result(
                    run, args.fresh_boards, args.stagnation_scaling
                )
            runs.append(result)
        if runs:
            table.append((condition.name, runs))
    if not table:
        print(f"No runs with a history.csv under {args.directory}")
        return 1

    def mean(runs: list[dict], key: str) -> float:
        return statistics.mean(run[key] for run in runs)

    baseline = next((runs for name, runs in table if name == args.baseline), None)
    baseline_best = mean(baseline, "best") if baseline else None
    table.sort(key=lambda item: mean(item[1], "best"), reverse=True)

    header = [
        "condition", "seeds", "finished", "mean_best_10_board", "vs_baseline",
        "per_seed_best", "mean_pellets", "clear_rate_last_20", "mean_nodes",
    ]
    if args.fresh_boards:
        header += ["fresh_pellets", "fresh_clears"]
    rows = []
    for name, runs in table:
        best = mean(runs, "best")
        row = [
            name,
            len(runs),
            sum(run["finished"] for run in runs),
            f"{best:.0f}",
            "" if baseline_best is None else f"{best - baseline_best:+.0f}",
            " ".join(f"{run['best']:.0f}" for run in runs),
            f"{mean(runs, 'pellets'):.1%}",
            f"{mean(runs, 'clear_rate'):.1%}",
            f"{mean(runs, 'nodes'):.0f}",
        ]
        if args.fresh_boards:
            fresh = [run["fresh"] for run in runs if run.get("fresh")]
            row += (
                [
                    f"{statistics.mean(ratio for ratio, _ in fresh):.1%}",
                    f"{sum(clears for _, clears in fresh)}/{len(fresh) * args.fresh_boards}",
                ]
                if fresh
                else ["", ""]
            )
        rows.append(row)

    widths = [max(len(str(value)) for value in column) for column in zip(header, *rows)]
    for line in (header, *rows):
        print("  ".join(str(value).ljust(width) for value, width in zip(line, widths)))
    print(
        "\nmean_best_10_board: best champion score on the 10 tournament boards, "
        "averaged over seeds. Unfinished runs are included at their current best."
    )
    with (args.directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
