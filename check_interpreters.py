#!/usr/bin/env python3
"""Check that two Python interpreters play Pac-Man identically, and time them.

    python check_interpreters.py --output cpython.pkl       (run under each one)
    python check_interpreters.py --compare cpython.pkl pypy.pkl

The workload: 32 seeded random trees (plus runs/playoff_winner.pkl and the
best_overall.pkl of up to 16 runs, if they exist) each play 6 random boards and the original maze, and 6 of them are
replayed move by move with every observation and board state hashed. The
diversity figures of the 32 trees are included. ``--compare`` reports whether
everything matches and how long each interpreter took.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import pickle
import platform
import random
import sys
import time
from pathlib import Path

import GeneticAlgo.train_board_graph as trainer
from PacmanGame import PacmanGame, read_board

PROJECT = Path(__file__).resolve().parent


def workload() -> dict:
    primitive_set = trainer.build_primitive_set()
    random.seed(1234)
    toolbox = trainer.build_toolbox(primitive_set, 2, 8, 20, 0.5, 0.8, full_min_depth=4,
                                    grow_terminal_probability=0.4)
    trees = toolbox.population(n=32)
    # Evolved trees play long games, which random ones do not: add the saved
    # best trees of up to 16 runs, if there are any.
    saved = [PROJECT / "runs" / "playoff_winner.pkl"]
    saved += sorted((PROJECT / "runs").glob("*/seed_*/best_overall.pkl"))[:16]
    for path in saved:
        if path.exists():
            with path.open("rb") as handle:
                trees.append(pickle.load(handle))
    boards = [(seed, trainer.generate_board(seed)) for seed in range(9100, 9106)]
    boards.append((0, read_board(trainer.VALIDATION_BOARD_PATH)))
    results: dict = {"trees": [str(tree) for tree in trees]}
    started = time.perf_counter()
    for seed, board in boards:
        results[("games", seed)] = trainer.evaluate_population_lockstep(
            [copy.deepcopy(tree) for tree in trees], board, 1000, primitive_set,
            stagnation_penalty=20.0, seed=seed,
        )
    games_seconds = time.perf_counter() - started
    for index, tree in enumerate(trees[:6]):
        for seed, board in boards[:2] + boards[-1:]:
            game, rng, digest = PacmanGame(board.copy()), random.Random(0), hashlib.sha256()
            for _ in range(1000):
                if game.game_over or game.won:
                    break
                observations = game.get_move_observations()
                digest.update(repr(sorted((move, sorted(features.items()))
                                          for move, features in observations.items())).encode())
                move, _ = trainer.choose_move(tree, observations, primitive_set, game.player_direction, rng)
                game.step(move or "stay")
                state = game.board
                digest.update(repr((state.player_pos, state.ghost_positions, state.score, state.pellets_eaten,
                                    state.player_powerup_turns, state.game_over, state.ghost_scared,
                                    state.tiles)).encode())
            results[("trajectory", index, seed)] = digest.hexdigest()
    results["diversity"] = trainer.population_diversity(trees, primitive_set)
    results["_timing"] = {"interpreter": f"{platform.python_implementation()} {platform.python_version()}",
                          "games": len(trees) * len(boards), "games_seconds": games_seconds,
                          "total_seconds": time.perf_counter() - started}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = (pickle.load(path.open("rb")) for path in args.compare)
        timings = first.pop("_timing"), second.pop("_timing")
        different = sorted(str(key) for key in first.keys() | second.keys() if first.get(key) != second.get(key))
        for timing in timings:
            print(f"{timing['interpreter']}: {timing['games']} games in {timing['games_seconds']:.1f} s, "
                  f"everything in {timing['total_seconds']:.1f} s")
        print(f"Speed-up on the games: {timings[0]['games_seconds'] / timings[1]['games_seconds']:.2f}x")
        if different:
            print(f"DIFFERENT: {len(different)} of {len(first)} results differ, e.g. {different[:5]}")
            return 1
        print(f"IDENTICAL: all {len(first)} results match")
        return 0
    if args.output is None:
        parser.error("give --output or --compare")
    results = workload()
    with args.output.open("wb") as handle:
        pickle.dump(results, handle, protocol=4)
    timing = results["_timing"]
    print(f"{timing['interpreter']}: {timing['games']} games in {timing['games_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
