#!/usr/bin/env python3
"""Find the smallest tree that clears the original Pac-Man maze.

    python -m GeneticAlgo.train_original_board --output-directory runs/original_x

Training uses one board only, the original maze, on which a game is fully
deterministic: a tree always gets the same result. That makes the goal exact:

1. a tree that clears the maze beats every tree that does not;
2. among trees that clear it, fewer nodes is better (then less wasted time);
3. among trees that do not, the usual game fitness decides, with no size cost,
   so nothing holds a tree back from growing until it can clear.

Evolution is the cellular grid (a child replaces its parent only if it ranks
higher), with mutations that also shrink trees and the tree-repair step on.
Only new trees are played, since an unchanged tree's result cannot change.
When the run ends, the smallest clearing tree is pruned: every way of
replacing one of its subtrees by one of that subtree's own arguments or by a
single terminal is tried, the smallest result that still clears is kept, and
this repeats until nothing smaller clears.

--start-from takes saved trees or run folders; trees from them that clear the
maze join the starting population, so the search can begin at the shrinking
stage. --prune-only FILE.pkl just prunes one tree and writes the result.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import random
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from deap import gp

from PacmanGame import read_board

from . import train_board_graph as trainer
from .gp_core import repair_individual

BOARD_PATH = trainer.VALIDATION_BOARD_PATH
# Above any game fitness (at most 130,000), so every clear outranks every
# non-clear, and one node fewer always outranks any game-fitness difference.
CLEAR_BASE = 1_000_000.0
CHECKPOINT_NAME = "checkpoint.pkl"
HISTORY_FIELDS = [
    "generation", "clearing_trees", "smallest_clear_nodes", "best_cleared", "best_nodes",
    "best_pellet_ratio", "best_game_fitness", "average_nodes", "distinct_expressions", "games_played",
]


class Scorer:
    """Plays trees on the original maze and ranks them for minimisation."""

    def __init__(self, args: argparse.Namespace, primitive_set, executor, workers: int):
        self.args = args
        self.primitive_set = primitive_set
        self.executor = executor
        self.workers = workers
        self.board = read_board(BOARD_PATH)
        self.games_played = 0

    def play(self, trees: list) -> list[tuple[float, float, bool]]:
        """(game fitness without any size cost, pellet ratio, cleared) per tree."""
        if not trees:
            return []
        self.games_played += len(trees)
        fitnesses, _, pellet_ratios, labels = trainer.evaluate_population_on_boards(
            trees, [(0, self.board)], self.args.timestep_limit, self.primitive_set,
            parsimony=0.0, stagnation_penalty=self.args.stagnation_penalty,
            stagnation_grace_steps=0, stagnation_termination_steps=self.args.stagnation_termination_steps,
            random_tie_break=False, size_penalty=0.0,
            executor=self.executor, workers=self.workers,
        )
        return [(fitness, ratio, label[0] == "cleared")
                for fitness, ratio, label in zip(fitnesses, pellet_ratios, labels)]

    def score(self, trees: list, known: dict[str, tuple] | None = None) -> None:
        """Set each tree's fitness and ``result``; ``known`` maps expressions already played."""
        known = {} if known is None else known
        unseen: dict[str, object] = {}
        for tree in trees:
            expression = str(tree)
            if expression not in known:
                unseen.setdefault(expression, tree)
        known.update(zip(unseen, self.play(list(unseen.values()))))
        for tree in trees:
            game_fitness, ratio, cleared = known[str(tree)]
            tree.result = (game_fitness, ratio, cleared)
            tree.fitness.values = (
                (CLEAR_BASE - len(tree), game_fitness) if cleared else (game_fitness, -float(len(tree)))
            )


def hoist(individual):
    """Replace the tree by one of its own subtrees."""
    if len(individual) > 1:
        subtree = individual[individual.searchSubtree(random.randrange(1, len(individual)))]
        individual[slice(0, len(individual))] = subtree
    return (individual,)


def minimising_mutation(individual, pset, max_height: int, min_depth: int):
    """Usual subtree mutation 40% of the time; otherwise a move that shrinks or tweaks."""
    roll = random.random()
    if roll < 0.40:
        return trainer.mutate_within_depth(individual, pset, max_height, min_depth)
    if roll < 0.70:
        return gp.mutShrink(individual)
    if roll < 0.90:
        return hoist(individual)
    return gp.mutNodeReplacement(individual, pset)


def build_toolbox(args: argparse.Namespace, primitive_set):
    toolbox = trainer.build_toolbox(
        primitive_set, args.min_initial_depth, args.max_initial_depth, args.max_evolved_depth,
        0.5, 0.8, repair=False, full_min_depth=args.full_min_initial_depth,
        grow_terminal_probability=args.grow_terminal_probability,
    )
    toolbox.register("mutate", minimising_mutation, pset=primitive_set,
                     max_height=args.max_evolved_depth - 1, min_depth=1)
    limit = gp.staticLimit(key=lambda tree: tree.height, max_value=args.max_evolved_depth - 1)

    def repaired(operator):
        def wrapper(*operands, **keywords):
            return tuple(repair_individual(child, primitive_set) for child in operator(*operands, **keywords))
        return wrapper

    toolbox.decorate("mutate", limit, repaired)
    toolbox.decorate("mate", repaired)
    return toolbox


def load_trees(sources: list[Path]) -> list:
    trees, seen = [], set()
    for source in sources:
        paths = sorted(source.rglob("*.pkl")) if source.is_dir() else [source]
        for path in paths:
            if path.name == CHECKPOINT_NAME:
                continue
            with path.open("rb") as handle:
                tree = pickle.load(handle)
            # Not trainer.Individual: unpickling a tree makes DEAP rebuild that class.
            if isinstance(tree, gp.PrimitiveTree) and str(tree) not in seen:
                seen.add(str(tree))
                tree = copy.deepcopy(tree)
                del tree.fitness.values
                trees.append(tree)
    return trees


def prune(tree, scorer: Scorer, primitive_set, log=print):
    """Greedy pruning: the smallest one-step reduction that still clears, repeated."""
    best = copy.deepcopy(tree)
    scorer.score([best])
    if not best.result[2]:
        log("Pruning skipped: the tree does not clear the original maze.")
        return best
    terminals = [terminal for terminal in primitive_set.terminals[primitive_set.ret]]
    while True:
        candidates: dict[str, object] = {}
        for index in range(len(best)):
            span = best.searchSubtree(index)
            if span.stop - span.start < 2:
                continue  # a single terminal cannot get smaller
            replacements = []
            child = index + 1
            for _ in range(best[index].arity):  # each argument of this node
                child_span = best.searchSubtree(child)
                replacements.append(best[child_span])
                child = child_span.stop
            # Terminals may be classes (ephemeral-style) or instances in DEAP.
            replacements.extend([terminal() if isinstance(terminal, type) else terminal] for terminal in terminals)
            for replacement in replacements:
                candidate = copy.deepcopy(best)
                candidate[span] = replacement
                repair_individual(candidate, primitive_set)
                if len(candidate) < len(best):
                    candidates.setdefault(str(candidate), candidate)
        trees = list(candidates.values())
        scorer.score(trees)
        clearing = [candidate for candidate in trees if candidate.result[2]]
        if not clearing:
            return best
        smallest = max(clearing, key=lambda candidate: candidate.fitness)
        log(f"prune: {len(best)} -> {len(smallest)} nodes ({len(clearing)} of {len(trees)} reductions still clear)")
        best = smallest


def save_tree(tree, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)


def describe(tree) -> str:
    game_fitness, ratio, cleared = tree.result
    return (f"{'clears' if cleared else 'does not clear'} the original maze, {len(tree)} nodes, "
            f"depth {tree.height + 1}, {ratio:.1%} of pellets, game fitness {game_fitness:.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--population-size", type=int, default=400)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="processes (default: all CPUs)")
    parser.add_argument("--start-from", type=Path, nargs="*", default=[],
                        help="saved trees or run folders; those that clear the maze join the starting population")
    parser.add_argument("--target-nodes", type=int, help="stop once a clearing tree is this small")
    parser.add_argument("--timestep-limit", type=int, default=1000)
    parser.add_argument("--stagnation-penalty", type=float, default=20.0)
    parser.add_argument("--stagnation-termination-steps", type=int, default=50)
    parser.add_argument("--min-initial-depth", type=int, default=2)
    parser.add_argument("--max-initial-depth", type=int, default=8)
    parser.add_argument("--full-min-initial-depth", type=int, default=4)
    parser.add_argument("--grow-terminal-probability", type=float, default=0.4)
    parser.add_argument("--max-evolved-depth", type=int, default=20)
    parser.add_argument("--crossover-probability", type=float, default=0.5)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint.pkl in --output-directory")
    parser.add_argument("--no-prune", action="store_true", help="skip the final pruning pass")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--prune-only", type=Path, help="prune this saved tree and exit")
    args = parser.parse_args()
    if args.prune_only is None and args.output_directory is None:
        parser.error("--output-directory is required")
    if not 0 <= args.crossover_probability <= 1:
        parser.error("--crossover-probability must be between 0 and 1")
    return args


def finish(args, output: Path, scorer: Scorer, primitive_set, population) -> None:
    """Write the best tree, its pruned version, a summary and a video."""
    clearing = [tree for tree in population if tree.result[2]]
    best = max(population, key=lambda tree: tree.fitness)
    save_tree(best, output / "best.pkl")
    lines = [f"Best tree: {describe(best)}", str(best), ""]
    final = best
    if clearing:
        save_tree(best, output / "smallest_clear.pkl")
        if not args.no_prune:
            final = prune(best, scorer, primitive_set)
            save_tree(final, output / "smallest_clear_pruned.pkl")
            lines += [f"After pruning: {describe(final)}", str(final), ""]
    else:
        lines.append("No tree cleared the original maze, so nothing was minimised.")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    if not args.no_video:
        name = "smallest_clear_pruned" if final is not best else "best"
        save_tree(final, output / f"{name}.pkl")
        subprocess.run(
            [trainer.artifact_python(), "-m", "GeneticAlgo.watch_board_graph", str(output / f"{name}.pkl"),
             "--board", str(BOARD_PATH), "--video", str(output / f"{name}.mp4"),
             "--scale", "2", "--max-steps", str(args.timestep_limit)],
            cwd=trainer.PROJECT_DIR, check=False,
        )


def main() -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args = parse_args()
    primitive_set = trainer.build_primitive_set(trainer.FEATURE_NAMES)
    workers = args.workers or len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scorer = Scorer(args, primitive_set, executor, workers)
        if args.prune_only is not None:
            tree, = load_trees([args.prune_only])
            pruned = prune(tree, scorer, primitive_set)
            print(f"{describe(pruned)}\n{pruned}")
            if not pruned.result[2]:
                return 1
            target = args.prune_only.with_name(args.prune_only.stem + "_pruned.pkl")
            save_tree(pruned, target)
            print(f"Wrote {target}")
            return 0

        output = args.output_directory
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / CHECKPOINT_NAME
        random.seed(args.seed)
        toolbox = build_toolbox(args, primitive_set)
        history: list[dict] = []
        start = 0
        if args.resume and checkpoint_path.exists():
            with checkpoint_path.open("rb") as handle:
                state = pickle.load(handle)
            population, history, start = state["population"], state["history"], state["next_generation"]
            scorer.games_played = state["games_played"]
            random.setstate(state["random_state"])
            print(f"Resuming at generation {start} of {args.generations}")
        else:
            population = toolbox.population(n=args.population_size)
            for tree in population:
                repair_individual(tree, primitive_set)
            scorer.score(population)
            given = load_trees(args.start_from)
            scorer.score(given)
            usable = sorted((tree for tree in given if tree.result[2]), key=lambda tree: tree.fitness, reverse=True)
            print(f"Starting trees: {len(given)} loaded, {len(usable)} clear the original maze")
            # The loaded trees fill every other cell of at most half the grid,
            # so each has random neighbours to mix with.
            for slot, tree in zip(range(0, args.population_size, 2), usable):
                population[slot] = tree
        print(f"Original-maze minimiser: {args.population_size} trees, {workers} workers, depth limit {args.max_evolved_depth}")

        try:
            for generation in range(start, args.generations):
                children = trainer.cellular_offspring(
                    population, toolbox, args.crossover_probability, 1.0, args.tournament_size
                )
                scorer.score(children, {str(tree): tree.result for tree in population})
                population = [
                    child if child.fitness > parent.fitness else parent
                    for parent, child in zip(population, children)
                ]
                best = max(population, key=lambda tree: tree.fitness)
                clearing = [tree for tree in population if tree.result[2]]
                row = {
                    "generation": generation,
                    "clearing_trees": len(clearing),
                    "smallest_clear_nodes": min((len(tree) for tree in clearing), default=""),
                    "best_cleared": int(best.result[2]),
                    "best_nodes": len(best),
                    "best_pellet_ratio": round(best.result[1], 4),
                    "best_game_fitness": round(best.result[0], 1),
                    "average_nodes": round(sum(len(tree) for tree in population) / len(population), 1),
                    "distinct_expressions": len({str(tree) for tree in population}),
                    "games_played": scorer.games_played,
                }
                history.append(row)
                with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
                    writer.writeheader()
                    writer.writerows(history)
                save_tree(best, output / "best.pkl")
                temporary = checkpoint_path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump({"population": population, "history": history, "next_generation": generation + 1,
                                 "games_played": scorer.games_played, "random_state": random.getstate()},
                                handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, checkpoint_path)
                smallest = row["smallest_clear_nodes"]
                print(
                    f"g={generation:04d} clears={len(clearing)}/{len(population)} "
                    f"smallest-clear={smallest if smallest != '' else '-'} "
                    f"best={'clear' if best.result[2] else f'{best.result[1]:.1%}'}/{len(best)}n "
                    f"avg-n={row['average_nodes']} distinct={row['distinct_expressions']}",
                    flush=True,
                )
                if args.target_nodes and smallest != "" and smallest <= args.target_nodes:
                    print(f"Reached the target of {args.target_nodes} nodes.")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted; finishing with the trees found so far.")
        finish(args, output, scorer, primitive_set, population)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
