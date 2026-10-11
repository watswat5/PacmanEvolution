#!/usr/bin/env python3
"""Compare the C++ engine with the Python game, and time both.

    python -m FastCore.check [--trees 60] [--boards 8] [--threads N]
    python -m FastCore.check --features [--trees 60] [--boards 8]

Plays seeded random trees plus saved champions (runs/**/best_overall.pkl,
runs/playoff_winner.pkl, if present) on random boards and the original maze
under the run scripts' settings, with each engine, and reports how often the
two agree on the outcome, pellets and fitness, and how long each took.

--features instead plays each tree step by step in both engines and compares
every legal move and every move feature (the optional ones included) at
every step, which tests the features directly rather than through outcomes.
"""
from __future__ import annotations

import argparse
import copy
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

import GeneticAlgo.train_board_graph as trainer
from PacmanGame import ACTIONS, MOVE_FEATURE_NAMES, PacmanGame, read_board

from . import backend

PROJECT = Path(__file__).resolve().parent.parent
SETTINGS = dict(stagnation_penalty=20.0, stagnation_grace_steps=0, stagnation_termination_steps=50)


def trees(count: int, primitive_set) -> list:
    random.seed(4321)
    toolbox = trainer.build_toolbox(primitive_set, 2, 8, 20, 0.5, 0.8, full_min_depth=4,
                                    grow_terminal_probability=0.4)
    result = toolbox.population(n=count)
    saved = [PROJECT / "runs" / "playoff_winner.pkl"] + sorted((PROJECT / "runs").glob("*/seed_*/best_overall.pkl"))
    for path in saved[:40]:
        if path.exists():
            with path.open("rb") as handle:
                result.append(pickle.load(handle))
    return result


def evasion_trees(primitive_set) -> list:
    """Trees that lean on the evasion features, so they steer the game."""
    texts = [
        "safe_territory",
        "add(safe_territory, neg(pincer))",
        "sub(mul(danger_ghost_approaching, danger_ghost_distance), pellet_distance)",
        "if_lte(pincer, constant_1, add(safe_territory, neg(pellet_distance)), danger_ghost_distance)",
        "add(mul(safe_territory, constant_4), sub(pellets_ahead, mul(danger_ghost_approaching, constant_3)))",
        "if_lte(pellets_left, constant_2, neg(pellet_distance), add(safe_territory, neg(pellet_distance)))",
        "sub(neg(safe_pellet_distance), ghost_path_conflict)",
        "if_lte(ghost_path_conflict, constant_0, neg(safe_pellet_distance), sub(safe_territory, ghost_path_conflict))",
        "add(onward_safe_exits, sub(blinky_distance, safe_power_pellet_distance))",
        "if_lte(elroy, constant_0, sub(onward_safe_exits, pellet_distance), add(blinky_distance, neg(safe_power_pellet_distance)))",
    ]
    return [trainer.Individual(trainer.gp.PrimitiveTree.from_string(text, primitive_set)) for text in texts]


def compare_features(population, boards, primitive_set, steps: int = 400) -> int:
    """Step both engines together; return 0 when every move and feature matches."""
    games = steps_compared = mismatched_games = 0
    worst = 0.0
    first_report = None
    for board_index, board in enumerate(boards):
        arguments = backend.start_arguments(board)
        for tree_index, tree in enumerate(population):
            code = backend.encode(tree, primitive_set)
            cpp_steps = backend.core.trace_features(code, *arguments, steps)
            game = PacmanGame(copy.deepcopy(board))
            policy = trainer.compile_policy(tree, primitive_set)
            rng = random.Random(0)
            mismatch = None
            for step_index, cpp in enumerate(cpp_steps):
                if game.game_over or game.won:
                    mismatch = (step_index, "Python game ended first")
                    break
                observations = game.get_move_observations()
                py_moves = [ACTIONS.index(action) for action in observations]
                cpp_moves = [move for move, _ in cpp]
                if py_moves != cpp_moves:
                    mismatch = (step_index, f"legal moves {py_moves} vs {cpp_moves}")
                    break
                for (move, row), action in zip(cpp, observations):
                    for name, value in zip(MOVE_FEATURE_NAMES, row):
                        gap = abs(observations[action][name] - value)
                        worst = max(worst, gap)
                        if gap > 1e-9 and mismatch is None:
                            mismatch = (step_index, f"{action} {name}: Python {observations[action][name]!r}, C++ {value!r}")
                if mismatch:
                    break
                steps_compared += 1
                move, _ = trainer.choose_move(tree, observations, primitive_set, game.player_direction, rng, policy=policy)
                game.step(move or "stay")
            else:
                if len(cpp_steps) < steps and not (game.game_over or game.won):
                    mismatch = (len(cpp_steps), "C++ game ended first")
            games += 1
            if mismatch:
                mismatched_games += 1
                first_report = first_report or (board_index, tree_index, *mismatch)
    print(f"{games} games ({len(population)} trees x {len(boards)} boards), {steps_compared} steps compared")
    print(f"  games where every move and feature matched: {games - mismatched_games}/{games}; "
          f"largest feature gap {worst:.3g}")
    if first_report:
        print("  first difference: board {}, tree {}, step {}: {}".format(*first_report))
    return 0 if not mismatched_games else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees", type=int, default=60)
    parser.add_argument("--boards", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--features", action="store_true",
                        help="compare per-step moves and features instead of game outcomes")
    args = parser.parse_args()
    if args.threads:
        backend.THREADS = args.threads
    sys.modules["__main__"].GenerationStats = trainer.GenerationStats
    primitive_set = trainer.build_primitive_set()
    population = trees(args.trees, primitive_set)
    boards = [trainer.generate_board(seed) for seed in range(7700, 7700 + args.boards)]
    boards.append(read_board(trainer.VALIDATION_BOARD_PATH))
    if args.features:
        return compare_features(evasion_trees(primitive_set) + population, boards, primitive_set)
    rules = [trainer.DEFAULT_RULES, trainer.EvaluationRules("inverse", 0.5)]
    python_seconds = cpp_seconds = 0.0
    games = same_label = same_pellets = 0
    fitness_gaps = []
    for index, board in enumerate(boards):
        rule = rules[index % 2]
        started = time.perf_counter()
        py_fit, _, py_ratio, py_label = trainer.evaluate_population_lockstep(
            [copy.deepcopy(tree) for tree in population], board, 1000, primitive_set, rules=rule, **SETTINGS)
        python_seconds += time.perf_counter() - started
        started = time.perf_counter()
        cpp_fit, _, cpp_ratio, cpp_label = backend.evaluate_population_lockstep(
            [copy.deepcopy(tree) for tree in population], board, 1000, primitive_set, rules=rule, **SETTINGS)
        cpp_seconds += time.perf_counter() - started
        for a, b, c, d, e, f in zip(py_fit, cpp_fit, py_ratio, cpp_ratio, py_label, cpp_label):
            games += 1
            same_label += e == f
            same_pellets += c == d
            fitness_gaps.append(abs(a - b))
    exact = sum(gap == 0 for gap in fitness_gaps)
    differing = sorted(gap for gap in fitness_gaps if gap)
    print(f"{games} games ({len(population)} trees x {len(boards)} boards, half with scaled stagnation "
          f"and late-pellet weighting)")
    print(f"  same outcome: {same_label}/{games}   same pellets: {same_pellets}/{games}   "
          f"identical fitness: {exact}/{games}")
    if differing:
        print(f"  where fitness differs: median gap {statistics.median(differing):.6g}, largest {differing[-1]:.6g}")
    print(f"  Python {python_seconds:.1f} s, C++ ({backend.THREADS} threads) {cpp_seconds:.2f} s: "
          f"{python_seconds / max(cpp_seconds, 1e-9):.0f}x faster")
    return 0 if same_label == games else 1


if __name__ == "__main__":
    raise SystemExit(main())
