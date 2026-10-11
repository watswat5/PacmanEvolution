#!/usr/bin/env python3
"""Exchange trees between island populations, by editing their checkpoints.

    python migrate_islands.py runs/<island run> --generation 20

Each island is an ordinary training run in <directory>/seed_<seed>/ that has
stopped at the same generation. Every island sends ``--migrants`` randomly
chosen trees to every other island, where they replace the worst trees. The
islands then continue from their checkpoints with ``--resume``.

- Migrants are chosen from the populations as they were before any arrivals,
  so a tree is never passed on twice in one round.
- A migrant's old fitness means nothing on its new island (each island scores
  on its own boards), so it arrives unscored; the trainer scores every tree at
  the start of the next generation anyway.
- "Worst" is by the fitness each island's trees earned in the generation just
  finished. If a population has no fitness (global method), random trees are
  replaced instead.
- The round is recorded in <directory>/island_state.json, so running the same
  round twice does nothing, and each move is logged in migrations.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import random
import sys
from pathlib import Path

import GeneticAlgo.train_board_graph as trainer

# Checkpoints written by ``python -m GeneticAlgo.train_board_graph`` refer to
# the history rows as __main__.GenerationStats.
sys.modules["__main__"].GenerationStats = trainer.GenerationStats

STATE_NAME = "island_state.json"
LOG_NAME = "migrations.csv"


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def without_scores(tree):
    """A migrant plays its new island's boards itself: drop the per-board
    results it brings from home (train_board_graph's board cache)."""
    tree.board_scores = {}
    return tree


def fitness_of(tree) -> float | None:
    return tree.fitness.values[0] if tree.fitness.valid else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--generation", type=int, required=True,
                        help="generation every island must have completed")
    parser.add_argument("--migrants", type=int, default=1,
                        help="trees each island sends to each other island (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing migrants")
    args = parser.parse_args()

    state_path = args.directory / STATE_NAME
    state = json.loads(state_path.read_text()) if state_path.exists() else {"migrated": []}
    if args.generation in state["migrated"]:
        print(f"Migration after generation {args.generation} was already done; nothing to do.")
        return 0

    paths = sorted(args.directory.glob("seed_*/checkpoint.pkl"),
                   key=lambda path: int(path.parent.name.removeprefix("seed_")))
    if len(paths) < 2:
        raise SystemExit(f"Need at least two islands with a checkpoint under {args.directory}")
    islands = {path.parent.name.removeprefix("seed_"): load(path) for path in paths}
    behind = {name: data["next_generation"] for name, data in islands.items()
              if data["next_generation"] != args.generation}
    if behind:
        raise SystemExit(
            f"Not migrating: every island must be at generation {args.generation}, but "
            + ", ".join(f"seed {name} is at {generation}" for name, generation in behind.items())
        )
    size = min(len(data["population"]) for data in islands.values())
    arrivals = args.migrants * (len(islands) - 1)
    if arrivals >= size:
        raise SystemExit("Too many migrants for the population size")

    rng = random.Random(f"{args.seed}-{args.generation}")
    # Choose every migrant before anything is replaced.
    outgoing = {
        (source, target): [without_scores(copy.deepcopy(tree)) for tree in rng.sample(data["population"], args.migrants)]
        for source, data in islands.items()
        for target in islands if target != source
    }
    rows = []
    for target, data in islands.items():
        population = data["population"]
        scored = all(tree.fitness.valid for tree in population)
        if scored:
            slots = sorted(range(len(population)), key=lambda index: population[index].fitness)[:arrivals]
        else:
            slots = rng.sample(range(len(population)), arrivals)
        incoming = [(source, tree) for (source, to), trees in outgoing.items() if to == target for tree in trees]
        for slot, (source, tree) in zip(slots, incoming):
            rows.append([args.generation, source, target, len(tree), tree.height + 1,
                         fitness_of(tree), slot, fitness_of(population[slot])])
            del tree.fitness.values
            population[slot] = tree

    # Write every checkpoint to a temporary file first, then swap them in, so
    # a crash cannot leave some islands migrated and others not.
    temporary = []
    for path in paths:
        data = islands[path.parent.name.removeprefix("seed_")]
        target = path.with_suffix(".migrating")
        with target.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.append((target, path))
    for target, path in temporary:
        os.replace(target, path)
    state["migrated"].append(args.generation)
    state_path.write_text(json.dumps(state))

    log_path = args.directory / LOG_NAME
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["after_generation", "from_seed", "to_seed", "migrant_nodes", "migrant_depth",
                             "migrant_fitness_at_home", "replaced_cell", "replaced_fitness"])
        writer.writerows(rows)
    print(f"Migration after generation {args.generation}: {len(rows)} trees moved between "
          f"{len(islands)} islands ({args.migrants} from each island to each other island).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
