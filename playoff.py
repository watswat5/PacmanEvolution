#!/usr/bin/env python3
"""Playoff: find the most generally successful tree among several runs.

    python playoff.py runs/board_graph_overnight_20261002_012238_GTc622
    python playoff.py runs/<run A> runs/<run B> --top 10 --boards 100

Each argument is a run folder, or a folder that contains runs (an overnight or
island run with seed_NN/ folders, or an experiment with <condition>/seed_NN/).
From every run it takes the --top best *distinct* generation champions by
their training-time tournament score, then plays all of them on the same
--boards fresh random boards that no run trained or was selected on.

The winner is then played on a second, separate set of --final-boards boards.
Quote that second result: the best of many trees on any one set of boards is
flattered a little by having been picked on it.

Writes playoff.csv (every candidate, ranked) and playoff_winner.pkl to
--output (default: the run folder, or the common parent of several).
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import GeneticAlgo.train_board_graph as trainer

# Far above every training and tournament board seed (those stay below about
# 1.1e9 even for 1000-generation runs), so playoff boards are always unseen.
SELECTION_BOARD_SEED = 2_000_000_000
FINAL_BOARD_SEED = 2_100_000_000
STEP_LIMIT = 1000
STEP_COST = 20.0  # the overnight setting; only affects the fitness column


def find_runs(roots: list[Path]) -> list[Path]:
    runs = []
    for root in roots:
        found = sorted({path.parent for path in root.rglob("generation_*_best.pkl")})
        if not found:
            raise SystemExit(f"No generation_NNNN_best.pkl files under {root}")
        runs.extend(found)
    return runs


def candidates_from(run: Path, top: int, through: int | None = None) -> list[dict]:
    """The run's ``top`` best distinct generation champions by tournament score.

    ``through`` limits the choice to generations 0 .. through-1.
    """
    scores = {}
    history = run / "history.csv"
    if history.exists():
        with history.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                scores[int(row["generation"])] = float(row["tournament_fitness"])
    overall = None
    if (run / "best_overall.pkl").exists():
        with (run / "best_overall.pkl").open("rb") as handle:
            overall = str(pickle.load(handle))
    distinct: dict[str, dict] = {}
    for path in sorted(run.glob("generation_*_best.pkl")):
        generation = int(path.name.split("_")[1])
        if through is not None and generation >= through:
            continue
        with path.open("rb") as handle:
            tree = pickle.load(handle)
        expression = str(tree)
        score = scores.get(generation, float("-inf"))
        entry = distinct.get(expression)
        if entry is None:
            distinct[expression] = {
                "run": run, "tree": tree, "first_generation": generation,
                "generations_as_champion": 1, "tournament_score": score,
                "was_best_overall": expression == overall,
            }
        else:
            entry["generations_as_champion"] += 1
            entry["tournament_score"] = max(entry["tournament_score"], score)
    ranked = sorted(distinct.values(), key=lambda entry: entry["tournament_score"], reverse=True)
    return ranked[:top]


def play(trees: list, board_seeds: list[int], executor, workers: int,
         stagnation_scaling: str = "fixed") -> list[dict]:
    """Play every tree on every board; return per-tree averages and outcomes."""
    boards = [(seed, trainer.generate_board(seed)) for seed in board_seeds]
    population = [copy.deepcopy(tree) for tree in trees]
    fitnesses, _, pellets, labels = trainer.evaluate_population_on_boards(
        population, boards, STEP_LIMIT, trainer.build_primitive_set(),
        parsimony=0.0, stagnation_penalty=STEP_COST, stagnation_grace_steps=0,
        stagnation_termination_steps=50, random_tie_break=False,
        executor=executor, workers=workers,
        rules=trainer.EvaluationRules(stagnation_scaling=stagnation_scaling),
    )
    results = []
    for fitness, pellet_ratio, outcome in zip(fitnesses, pellets, labels):
        counts = Counter(outcome)
        results.append({
            "fitness": fitness, "pellets": pellet_ratio, "cleared": counts["cleared"],
            "died": counts["died"], "stagnated": counts["stagnated"], "timed_out": counts["timed_out"],
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=10, help="distinct champions taken from each run (default: 10)")
    parser.add_argument("--boards", type=int, default=100, help="fresh boards every candidate plays (default: 100)")
    parser.add_argument("--final-boards", type=int, default=100,
                        help="separate fresh boards the winner then plays (default: 100; 0 skips)")
    parser.add_argument("--rank-by", choices=("fitness", "pellets", "cleared"), default="fitness")
    parser.add_argument("--workers", type=int, default=0, help="processes (default: all CPUs)")
    parser.add_argument("--stagnation-scaling", choices=("fixed", "inverse"), default="fixed",
                        help="inverse: the stagnation cutoff lengthens as the board empties; use it "
                             "when any candidate trained that way (default: fixed)")
    parser.add_argument("--through", type=int,
                        help="only champions from generations 0 .. N-1, to compare runs of different lengths")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = find_runs(args.runs)
    common = Path(os.path.commonpath([str(run) for run in runs]))
    output = args.output or (args.runs[0] if len(args.runs) == 1 else common)
    output.mkdir(parents=True, exist_ok=True)
    name = lambda run: str(run.relative_to(common)) if run != common else run.name

    candidates = [entry for run in runs for entry in candidates_from(run, args.top, args.through)]
    print(f"{len(candidates)} candidates from {len(runs)} runs, {args.boards} fresh boards each")
    workers = args.workers or len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = play([entry["tree"] for entry in candidates],
                       [SELECTION_BOARD_SEED + index for index in range(args.boards)], executor, workers,
                       args.stagnation_scaling)
        for entry, result in zip(candidates, results):
            entry.update(result)
        candidates.sort(key=lambda entry: (entry[args.rank_by], entry["pellets"]), reverse=True)
        winner = candidates[0]
        final = None
        if args.final_boards:
            final = play([winner["tree"]],
                         [FINAL_BOARD_SEED + index for index in range(args.final_boards)], executor, workers,
                         args.stagnation_scaling)[0]

    header = ["rank", "run", "first_generation", "generations_as_champion", "nodes", "was_best_overall",
              "tournament_score", "playoff_fitness", "pellets", "cleared", "died", "stagnated", "timed_out"]
    rows = [[rank, name(entry["run"]), entry["first_generation"], entry["generations_as_champion"],
             len(entry["tree"]), "yes" if entry["was_best_overall"] else "",
             f"{entry['tournament_score']:.0f}", f"{entry['fitness']:.0f}", f"{entry['pellets']:.1%}",
             entry["cleared"], entry["died"], entry["stagnated"], entry["timed_out"]]
            for rank, entry in enumerate(candidates, start=1)]
    with (output / "playoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    with (output / "playoff_winner.pkl").open("wb") as handle:
        pickle.dump(winner["tree"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    shown = rows[:15]
    widths = [max(len(str(value)) for value in column) for column in zip(header, *shown)]
    print(f"\nTop {len(shown)} of {len(rows)} on the {args.boards} playoff boards (ranked by {args.rank_by}):")
    for line in (header, *shown):
        print("  ".join(str(value).ljust(width) for value, width in zip(line, widths)))

    print("\nBest candidate from each run, and where that run's saved best_overall placed:")
    for run in runs:
        own = [(rank, entry) for rank, entry in enumerate(candidates, start=1) if entry["run"] == run]
        best_rank, best = own[0]
        saved = next(((rank, entry) for rank, entry in own if entry["was_best_overall"]), None)
        saved_text = (f"best_overall placed {saved[0]} ({saved[1]['pellets']:.1%} pellets, {saved[1]['cleared']} clears)"
                      if saved else "best_overall was not among its candidates")
        same = " (the same tree)" if saved and saved[0] == best_rank else ""
        print(f"  {name(run)}: best placed {best_rank} with {best['pellets']:.1%} pellets, "
              f"{best['cleared']} clears, from generation {best['first_generation']}; {saved_text}{same}")

    print(f"\nWinner: {name(winner['run'])}, generation {winner['first_generation']}, {len(winner['tree'])} nodes")
    print(f"  on the {args.boards} playoff boards: {winner['pellets']:.1%} pellets, "
          f"{winner['cleared']} cleared, {winner['died']} died, {winner['stagnated']} stagnated")
    if final:
        print(f"  on {args.final_boards} further fresh boards (quote this): {final['pellets']:.1%} pellets, "
              f"{final['cleared']} cleared, {final['died']} died, {final['stagnated']} stagnated, "
              f"fitness {final['fitness']:.0f}")
    print(f"\nWrote {output / 'playoff.csv'} and {output / 'playoff_winner.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
