#!/usr/bin/env python3
"""Train DEAP move-scoring trees on shared mazes per generation."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import pickle
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from deap import algorithms, base, creator, gp, tools

try:
    from .gp_core import build_primitive_set, safe_number, select_over
except ImportError:
    from gp_core import build_primitive_set, safe_number, select_over

try:
    from PacmanGame import (
        ACTIONS,
        Board,
        PacmanGame,
        precompute_board_graph,
        read_board,
        write_board,
    )
    from PacmanGame.generate_tetris_board import generate_board
except ImportError as exc:
    raise ImportError("run this script from the project root") from exc


PROJECT_DIR = Path(__file__).resolve().parent.parent
VALIDATION_BOARD_PATH = PROJECT_DIR / "PacmanGame" / "test_board.txt"
GENERATION_SEED_STRIDE = 1_000_003
FEATURE_NAMES = (
    "pellet_here",
    "pellet_distance",
    "power_pellet_distance",
    "danger_ghost_distance",
    "edible_ghost_distance",
    "is_reverse",
    "is_forward",
    "candidate_degree",
    "pellets_ahead",
    "steps_to_junction",
    "pellet_center_direction",
)
DEATH_PENALTY = 10_000.0
NON_SCORING_MOVE_PENALTY = 100.0
NORMALIZED_CLEAR_SCORE = 100_000.0


def _ensure_deap_types() -> None:
    if not hasattr(creator, "BoardGraphFitnessMax"):
        creator.create("BoardGraphFitnessMax", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "BoardGraphIndividual"):
        creator.create(
            "BoardGraphIndividual",
            gp.PrimitiveTree,
            fitness=creator.BoardGraphFitnessMax,
        )


_ensure_deap_types()
Individual = creator.BoardGraphIndividual


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    cleared: int
    died: int
    timed_out: int
    validation_fitness: float
    validation_pellet_ratio: float
    best_validation_fitness_so_far: float
    best_validation_pellet_ratio_so_far: float
    validation_cleared: int
    validation_died: int
    validation_timed_out: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    average_tree_depth: float
    best_tree_depth: int
    average_node_count: float
    best_node_count: int


@dataclass(frozen=True)
class EvolutionResult:
    best: Individual
    population: list[Individual]
    generations_completed: int
    stopped_early: bool
    history: tuple[GenerationStats, ...]


@dataclass(frozen=True)
class OutcomeCounts:
    cleared: int
    died: int
    timed_out: int


def build_toolbox(
    primitive_set: gp.PrimitiveSet,
    min_depth: int,
    max_depth: int,
    max_evolved_depth: int,
    top_fraction: float,
    top_probability: float,
) -> base.Toolbox:
    """Build operators for a single move-scoring PrimitiveTree."""
    toolbox = base.Toolbox()
    toolbox.register(
        "expression",
        gp.genHalfAndHalf,
        pset=primitive_set,
        min_=min_depth - 1,
        max_=max_depth - 1,
    )
    toolbox.register("individual", tools.initIterate, Individual, toolbox.expression)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "select",
        select_over,
        top_fraction=top_fraction,
        top_probability=top_probability,
    )
    toolbox.register("mate", gp.cxOnePoint)
    expression = partial(
        gp.genGrow, pset=primitive_set, min_=0, max_=max_evolved_depth - 1
    )
    toolbox.register("mutate", gp.mutUniform, expr=expression, pset=primitive_set)
    height_limit = gp.staticLimit(
        key=lambda tree: tree.height, max_value=max_evolved_depth - 1
    )
    toolbox.decorate("mate", height_limit)
    toolbox.decorate("mutate", height_limit)
    return toolbox


@lru_cache(maxsize=8192)
def _compile_tree(expression: str, primitive_set: gp.PrimitiveSet) -> Callable:
    return gp.compile(expr=expression, pset=primitive_set)


def choose_move(
    individual: Individual,
    observations: dict[str, dict[str, float]],
    primitive_set: gp.PrimitiveSet,
    last_direction: str,
    rng: random.Random,
    *,
    random_tie_break: bool = False,
) -> tuple[str | None, dict[str, float]]:
    """Score every legal candidate with the same tree."""
    if not observations:
        return None, {}
    try:
        policy = _compile_tree(str(individual), primitive_set)
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        policy = None
    scores = {}
    for action, features in observations.items():
        arguments = [features.get(name, 0.0) for name in primitive_set.arguments]
        try:
            value = 0.0 if policy is None else float(policy(*arguments))
            scores[action] = safe_number(value)
        except (ArithmeticError, OverflowError, TypeError, ValueError):
            scores[action] = 0.0
    best_score = max(scores.values())
    candidates = [action for action, score in scores.items() if score == best_score]
    if random_tie_break and len(candidates) > 1:
        return rng.choice(candidates), scores
    if last_direction in candidates:
        return last_direction, scores
    return min(candidates, key=ACTIONS.index), scores


def calculate_fitness(
    game: PacmanGame,
    starting_pellets_eaten: int,
    starting_pellets_remaining: int,
    individual: Individual,
    steps_taken: int,
    timestep_limit: int,
    stagnation_cost: float,
) -> float:
    died_or_terminated = game.game_over or (
        steps_taken < timestep_limit and not game.won
    )
    pellets_consumed = game.board.pellets_eaten - starting_pellets_eaten
    completion_ratio = pellets_consumed / max(1, starting_pellets_remaining)
    return (
        NORMALIZED_CLEAR_SCORE * completion_ratio
        - DEATH_PENALTY * died_or_terminated
        - stagnation_cost
    )


def evaluate_population_lockstep(
    population: Sequence[Individual],
    starting_board: Board,
    timestep_limit: int,
    primitive_set: gp.PrimitiveSet,
    *,
    parsimony: float = 0.0,
    stagnation_penalty: float = NON_SCORING_MOVE_PENALTY,
    stagnation_grace_steps: int = 0,
    stagnation_termination_steps: int = 50,
    seed: int = 0,
    index_offset: int = 0,
    random_tie_break: bool = False,
) -> tuple[list[float], OutcomeCounts, list[float], list[str]]:
    """Evaluate all individuals while sharing one precomputed static graph."""
    graph = precompute_board_graph(starting_board)
    games = [
        PacmanGame(starting_board.copy(), board_graph=graph)
        for _ in population
    ]
    starting_pellets_eaten = [game.board.pellets_eaten for game in games]
    starting_pellets_remaining = [game.board.pellets_remaining for game in games]
    scoreless_steps = [0] * len(games)
    steps_taken = [0] * len(games)
    stagnation_costs = [0.0] * len(games)
    stagnated = [False] * len(games)
    rngs = [random.Random(seed + index_offset + index) for index in range(len(games))]
    active = list(range(len(games)))

    for _ in range(timestep_limit):
        for index in active:
            game = games[index]
            move, _ = choose_move(
                population[index],
                game.get_move_observations(),
                primitive_set,
                game.player_direction,
                rngs[index],
                random_tie_break=random_tie_break,
            )
            score_before = game.board.score
            game.step(move or "stay")
            steps_taken[index] += 1
            if game.board.score > score_before:
                scoreless_steps[index] = 0
            else:
                scoreless_steps[index] += 1
                if scoreless_steps[index] > stagnation_grace_steps:
                    stagnation_costs[index] += stagnation_penalty
                if scoreless_steps[index] >= stagnation_termination_steps:
                    stagnated[index] = True
        active = [
            index
            for index in active
            if not (games[index].game_over or games[index].won or stagnated[index])
        ]
        if not active:
            break

    fitnesses = [
        calculate_fitness(
            game,
            starting_pellets_eaten[index],
            starting_pellets_remaining[index],
            population[index],
            steps_taken[index],
            timestep_limit,
            stagnation_costs[index],
        )
        for index, game in enumerate(games)
    ]
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = (fitness, -parsimony * len(individual))
    outcomes = OutcomeCounts(
        cleared=sum(game.won for game in games),
        died=sum(game.game_over for game in games),
        timed_out=sum(not game.won and not game.game_over for game in games),
    )
    outcome_labels = [
        "cleared" if game.won else "died" if game.game_over else "timed_out"
        for game in games
    ]
    completion_ratios = [
        (game.board.pellets_eaten - starting_pellets_eaten[index])
        / max(1, starting_pellets_remaining[index])
        for index, game in enumerate(games)
    ]
    return fitnesses, outcomes, completion_ratios, outcome_labels


def count_outcomes(labels: Sequence[Sequence[str]]) -> OutcomeCounts:
    """Count outcomes across all evaluated boards and individuals."""
    flat_labels = [label for individual_labels in labels for label in individual_labels]
    return OutcomeCounts(
        cleared=flat_labels.count("cleared"),
        died=flat_labels.count("died"),
        timed_out=flat_labels.count("timed_out"),
    )


def _evaluate_population_chunk(task: tuple) -> tuple[int, list[float], list[float], list[str]]:
    """Evaluate a contiguous population slice in a worker process."""
    (
        start, individuals, board, timestep_limit, primitive_set, parsimony,
        stagnation_penalty, stagnation_grace_steps,
        stagnation_termination_steps, seed, random_tie_break,
    ) = task
    fitnesses, _, pellet_ratios, labels = evaluate_population_lockstep(
        individuals,
        board,
        timestep_limit,
        primitive_set,
        parsimony=parsimony,
        stagnation_penalty=stagnation_penalty,
        stagnation_grace_steps=stagnation_grace_steps,
        stagnation_termination_steps=stagnation_termination_steps,
        seed=seed,
        index_offset=start,
        random_tie_break=random_tie_break,
    )
    return start, fitnesses, pellet_ratios, labels


def evaluate_population_on_boards(
    population: Sequence[Individual],
    boards: Sequence[tuple[int, Board]],
    timestep_limit: int,
    primitive_set: gp.PrimitiveSet,
    *,
    parsimony: float,
    stagnation_penalty: float,
    stagnation_grace_steps: int,
    stagnation_termination_steps: int,
    random_tie_break: bool,
    executor: ProcessPoolExecutor | None = None,
    workers: int = 1,
) -> tuple[list[float], OutcomeCounts, list[float], list[tuple[str, ...]]]:
    """Evaluate every candidate on the same boards and average its fitness."""
    fitness_totals = [0.0] * len(population)
    pellet_totals = [0.0] * len(population)
    labels: list[list[str]] = [[] for _ in population]
    for layout_seed, board in boards:
        if executor is None:
            fitnesses, _, pellet_ratios, board_labels = evaluate_population_lockstep(
                population,
                board,
                timestep_limit,
                primitive_set,
                parsimony=parsimony,
                stagnation_penalty=stagnation_penalty,
                stagnation_grace_steps=stagnation_grace_steps,
                stagnation_termination_steps=stagnation_termination_steps,
                seed=layout_seed,
                random_tie_break=random_tie_break,
            )
        else:
            fitnesses = [0.0] * len(population)
            pellet_ratios = [0.0] * len(population)
            board_labels = [""] * len(population)
            chunk_size = max(1, math.ceil(len(population) / (workers * 2)))
            tasks = (
                (
                    start, population[start:start + chunk_size], board,
                    timestep_limit, primitive_set, parsimony, stagnation_penalty,
                    stagnation_grace_steps, stagnation_termination_steps,
                    layout_seed, random_tie_break,
                )
                for start in range(0, len(population), chunk_size)
            )
            for start, chunk_fitnesses, chunk_ratios, chunk_labels in executor.map(
                _evaluate_population_chunk, tasks
            ):
                stop = start + len(chunk_fitnesses)
                fitnesses[start:stop] = chunk_fitnesses
                pellet_ratios[start:stop] = chunk_ratios
                board_labels[start:stop] = chunk_labels
        for index, (fitness, ratio, label) in enumerate(
            zip(fitnesses, pellet_ratios, board_labels)
        ):
            fitness_totals[index] += fitness
            pellet_totals[index] += ratio
            labels[index].append(label)
    board_count = len(boards)
    average_fitnesses = [total / board_count for total in fitness_totals]
    average_pellet_ratios = [total / board_count for total in pellet_totals]
    for individual, fitness in zip(population, average_fitnesses):
        individual.fitness.values = (fitness, -parsimony * len(individual))
    return (
        average_fitnesses,
        count_outcomes(labels),
        average_pellet_ratios,
        [tuple(individual_labels) for individual_labels in labels],
    )


def population_statistics(
    population: Sequence[Individual],
    generation: int,
    outcomes: OutcomeCounts,
    validation_fitness: float,
    validation_pellet_ratio: float,
    best_validation_fitness_so_far: float,
    best_validation_pellet_ratio_so_far: float,
    validation_outcomes: OutcomeCounts,
) -> GenerationStats:
    fitnesses = np.asarray(
        [individual.fitness.values[0] for individual in population],
        dtype=np.float64,
    )
    depths = np.asarray([tree.height + 1 for tree in population], dtype=np.float64)
    sizes = np.asarray([len(tree) for tree in population], dtype=np.float64)
    best = max(population, key=lambda tree: tree.fitness)
    return GenerationStats(
        generation=generation,
        cleared=outcomes.cleared,
        died=outcomes.died,
        timed_out=outcomes.timed_out,
        validation_fitness=validation_fitness,
        validation_pellet_ratio=validation_pellet_ratio,
        best_validation_fitness_so_far=best_validation_fitness_so_far,
        best_validation_pellet_ratio_so_far=best_validation_pellet_ratio_so_far,
        validation_cleared=validation_outcomes.cleared,
        validation_died=validation_outcomes.died,
        validation_timed_out=validation_outcomes.timed_out,
        best_fitness=float(fitnesses.max()),
        average_fitness=float(fitnesses.mean()),
        worst_fitness=float(fitnesses.min()),
        average_tree_depth=float(depths.mean()),
        best_tree_depth=best.height + 1,
        average_node_count=float(sizes.mean()),
        best_node_count=len(best),
    )


def write_history(path: Path, history: Sequence[GenerationStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(GenerationStats.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in history)


def save_individual(individual: Individual, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(individual, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generation_board(seed: int, generation: int, board_index: int = 0) -> tuple[int, Board]:
    """Create one reproducible layout shared by the entire generation."""
    layout_seed = seed + generation * GENERATION_SEED_STRIDE + board_index
    return layout_seed, generate_board(layout_seed)


def cellular_grid_shape(population_size: int) -> tuple[int, int]:
    """Return the closest-to-square exact rectangular grid for a population."""
    if population_size < 1:
        raise ValueError("population_size must be positive")
    rows = math.isqrt(population_size)
    while population_size % rows:
        rows -= 1
    return rows, population_size // rows


def moore_neighborhood(
    index: int, population_size: int, *, include_self: bool = True
) -> tuple[int, ...]:
    """Return radius-one Moore neighbors on a toroidal rectangular grid."""
    rows, columns = cellular_grid_shape(population_size)
    if not 0 <= index < population_size:
        raise IndexError("population index outside cellular grid")
    row, column = divmod(index, columns)
    neighbors = {
        ((row + dy) % rows) * columns + ((column + dx) % columns)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if include_self or dx or dy
    }
    if not include_self:
        neighbors.discard(index)
    return tuple(sorted(neighbors))


def _local_tournament(
    population: Sequence[Individual],
    indices: Sequence[int],
    tournament_size: int,
) -> Individual:
    aspirants = random.choices(indices, k=tournament_size)
    return max((population[index] for index in aspirants), key=lambda tree: tree.fitness)


def cellular_offspring(
    population: Sequence[Individual],
    toolbox: base.Toolbox,
    crossover_probability: float,
    mutation_probability: float,
    tournament_size: int,
) -> list[Individual]:
    """Create one locally selected offspring for every toroidal grid cell."""
    offspring = []
    for index in range(len(population)):
        neighborhood = moore_neighborhood(index, len(population))
        if random.random() < crossover_probability:
            # Keep crossover spatially strict: the focal cell always mates
            # with an immediately adjacent cell, never a global parent.
            first = population[index]
            partner_indices = (
                moore_neighborhood(index, len(population), include_self=False)
                or neighborhood
            )
            second = _local_tournament(
                population, partner_indices, tournament_size
            )
            child, sibling = toolbox.mate(copy.deepcopy(first), copy.deepcopy(second))
            offspring.append(random.choice((child, sibling)))
        else:
            first = _local_tournament(population, neighborhood, tournament_size)
            child = copy.deepcopy(first)
            if mutation_probability > 0:
                child, = toolbox.mutate(child)
            offspring.append(child)
    return offspring


def evolve(
    primitive_set: gp.PrimitiveSet,
    args: argparse.Namespace,
    output_directory: Path,
) -> EvolutionResult:
    random.seed(args.seed)
    toolbox = build_toolbox(
        primitive_set,
        args.min_initial_depth,
        args.max_initial_depth,
        args.max_evolved_depth,
        args.top_fraction,
        args.top_probability,
    )
    population = toolbox.population(n=args.population_size)
    hall_of_fame = tools.HallOfFame(1)
    history = []
    stopped_early = False
    best_validation_fitness = float("-inf")
    best_validation_rank = (float("-inf"), float("-inf"))
    best_validation_pellet_ratio = 0.0
    validation_board = read_board(VALIDATION_BOARD_PATH)
    precompute_board_graph(validation_board)
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    worker_count = min(args.population_size, args.workers or available_cpus)
    print(f"Evaluation workers: {worker_count}")
    worker_pool = (
        ProcessPoolExecutor(max_workers=worker_count)
        if worker_count > 1
        else nullcontext(None)
    )

    with worker_pool as executor:
        for generation in range(args.generations):
            boards = [
                generation_board(args.seed, generation, board_index)
                for board_index in range(args.boards_per_generation)
            ]
            layout_seed = boards[0][0]
            for board_index, (_, board) in enumerate(boards):
                suffix = "" if board_index == 0 else f"_board_{board_index:04d}"
                layout_path = (
                    output_directory / "layouts" / f"generation_{generation:04d}{suffix}.txt"
                )
                write_board(board, layout_path)
            fitnesses, outcomes, _, parent_outcome_labels = evaluate_population_on_boards(
                population,
                boards,
                args.timestep_limit,
                primitive_set,
                parsimony=args.parsimony,
                stagnation_penalty=args.stagnation_penalty,
                stagnation_grace_steps=args.stagnation_grace_steps,
                stagnation_termination_steps=args.stagnation_termination_steps,
                random_tie_break=args.random_tie_break,
                executor=executor,
                workers=worker_count,
            )
            for individual, fitness in zip(population, fitnesses):
                individual.fitness.values = (fitness, -args.parsimony * len(individual))

            if args.evolution_method == "cellular":
                children = cellular_offspring(
                    population,
                    toolbox,
                    args.crossover_probability,
                    args.mutation_probability,
                    args.cellular_tournament_size,
                )
                (
                    _,
                    _,
                    _,
                    child_outcome_labels,
                ) = evaluate_population_on_boards(
                    children,
                    boards,
                    args.timestep_limit,
                    primitive_set,
                    parsimony=args.parsimony,
                    stagnation_penalty=args.stagnation_penalty,
                    stagnation_grace_steps=args.stagnation_grace_steps,
                    stagnation_termination_steps=args.stagnation_termination_steps,
                    random_tie_break=args.random_tie_break,
                    executor=executor,
                    workers=worker_count,
                )
                selected_population = []
                selected_outcomes = []
                for index, (parent, child) in enumerate(zip(population, children)):
                    if child.fitness > parent.fitness:
                        selected_population.append(child)
                        selected_outcomes.append(child_outcome_labels[index])
                    else:
                        selected_population.append(parent)
                        selected_outcomes.append(parent_outcome_labels[index])
                population = selected_population
                outcomes = count_outcomes(selected_outcomes)

            generation_best = tools.selBest(population, 1)[0]
            validation_candidate = copy.deepcopy(generation_best)
            (
                validation_fitnesses,
                validation_outcomes,
                validation_pellet_ratios,
                _,
            ) = evaluate_population_lockstep(
                [validation_candidate],
                validation_board,
                args.timestep_limit,
                primitive_set,
                parsimony=args.parsimony,
                stagnation_penalty=args.stagnation_penalty,
                stagnation_grace_steps=args.stagnation_grace_steps,
                stagnation_termination_steps=args.stagnation_termination_steps,
                seed=args.seed,
                random_tie_break=args.random_tie_break,
            )
            validation_fitness = validation_fitnesses[0]
            validation_pellet_ratio = validation_pellet_ratios[0]
            validation_rank = validation_candidate.fitness.wvalues
            is_overall_best = validation_rank > best_validation_rank
            if is_overall_best:
                best_validation_rank = validation_rank
                best_validation_fitness = validation_fitness
                best_validation_pellet_ratio = validation_pellet_ratio
            stats = population_statistics(
                population,
                generation,
                outcomes,
                validation_fitness,
                validation_pellet_ratio,
                best_validation_fitness,
                best_validation_pellet_ratio,
                validation_outcomes,
            )
            history.append(stats)
            write_history(output_directory / "history.csv", history)
            hall_of_fame.update([validation_candidate])
            save_individual(
                generation_best,
                output_directory / f"generation_{generation:04d}_best.pkl",
            )
            save_individual(hall_of_fame[0], output_directory / "best_overall.pkl")
            if is_overall_best:
                write_board(
                    validation_board, output_directory / "best_overall_board.txt"
                )
            print(
                f"g={generation:04d} seed={layout_seed} "
                f"fit={stats.best_fitness:.0f}/{stats.average_fitness:.0f} "
                f"val={stats.validation_fitness:.0f}/"
                f"{stats.best_validation_fitness_so_far:.0f} "
                f"p={stats.validation_pellet_ratio:.1%}/"
                f"{stats.best_validation_pellet_ratio_so_far:.1%} "
                f"C/D/T={stats.cleared}/{stats.died}/{stats.timed_out} "
                f"n={stats.average_node_count:.1f}"
            )
            if (
                args.target_fitness is not None
                and stats.validation_fitness >= args.target_fitness
            ):
                stopped_early = True
                break
            if generation < args.generations - 1 and args.evolution_method == "global":
                parents = toolbox.select(population, args.population_size)
                population = algorithms.varOr(
                    parents,
                    toolbox,
                    lambda_=args.population_size,
                    cxpb=args.crossover_probability,
                    mutpb=args.mutation_probability,
                )

    return EvolutionResult(
        best=hall_of_fame[0],
        population=population,
        generations_completed=len(history),
        stopped_early=stopped_early,
        history=tuple(history),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="evaluation processes (default: all available CPUs; 1 disables parallelism)",
    )
    parser.add_argument(
        "--evolution-method",
        choices=("global", "cellular"),
        default="global",
        help="global over-selection or cellular GP on a toroidal grid",
    )
    parser.add_argument(
        "--cellular-tournament-size",
        type=int,
        default=3,
        help="local tournament size for cellular evolution (default: 3)",
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--boards-per-generation",
        type=int,
        default=1,
        help="number of shared training boards per generation (default: 1)",
    )
    parser.add_argument("--timestep-limit", type=int, default=1000)
    parser.add_argument("--stagnation-penalty", type=float, default=100.0)
    parser.add_argument("--stagnation-grace-steps", type=int, default=0)
    parser.add_argument("--stagnation-termination-steps", type=int, default=50)
    parser.add_argument("--min-initial-depth", type=int, default=2)
    parser.add_argument("--max-initial-depth", type=int, default=5)
    parser.add_argument("--max-evolved-depth", type=int, default=10)
    parser.add_argument("--crossover-probability", type=float, default=0.97)
    parser.add_argument("--mutation-probability", type=float, default=0.03)
    parser.add_argument("--top-fraction", type=float, default=0.32)
    parser.add_argument("--top-probability", type=float, default=0.8)
    parser.add_argument(
        "--parsimony",
        type=float,
        default=1.0,
        help=(
            "tie-break weight per node; positive values prefer smaller trees "
            "only when game scores tie (default: 1)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-fitness", type=float)
    parser.add_argument("--random-tie-break", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-scale", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.parsimony < 0:
        raise ValueError("parsimony must be nonnegative")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.population_size < 1 or args.generations < 1:
        raise ValueError("population size and generations must be positive")
    if not 1 <= args.boards_per_generation < GENERATION_SEED_STRIDE:
        raise ValueError("boards-per-generation must be between 1 and 1000002")
    if args.cellular_tournament_size < 1:
        raise ValueError("cellular-tournament-size must be positive")
    if args.timestep_limit < 1:
        raise ValueError("timestep-limit must be positive")
    if args.video_fps <= 0:
        raise ValueError("video-fps must be positive")
    if args.min_initial_depth < 1 or args.max_initial_depth < args.min_initial_depth:
        raise ValueError("invalid initial depth range")
    if args.max_evolved_depth < args.max_initial_depth:
        raise ValueError("max evolved depth cannot be below initial depth")
    if not math.isclose(
        args.crossover_probability + args.mutation_probability, 1.0
    ):
        raise ValueError("crossover and mutation probabilities must total 1")


def generate_artifacts(args: argparse.Namespace, output: Path) -> bool:
    history, best = output / "history.csv", output / "best_overall.pkl"
    if not history.exists() or not best.exists():
        print("No generation completed; there is no evaluated individual to render.", file=sys.stderr)
        return False
    subprocess.run([sys.executable, str(PROJECT_DIR / "plot_board_graph.py"), str(history), str(output), str(best)], cwd=PROJECT_DIR, check=True)
    if not args.no_video:
        command = [sys.executable, "-m", "GeneticAlgo.watch_board_graph", str(best), "--board", str(output / "best_overall_board.txt"), "--video", str(output / "best_overall.mp4"), "--fps", str(args.video_fps), "--scale", str(args.video_scale), "--max-steps", str(args.timestep_limit), "--seed", str(args.seed)]
        if args.random_tie_break:
            command.append("--random-tie-break")
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    return True


def main() -> int:
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_directory or Path("runs") / f"board_graph_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    primitive_set = build_primitive_set(FEATURE_NAMES, constants=(-1, 0, 1))
    print(f"Run artifacts: {output}")
    print(f"Terminals: {', '.join(FEATURE_NAMES)}")
    print(f"Training boards per generation: {args.boards_per_generation}")
    if args.evolution_method == "cellular":
        rows, columns = cellular_grid_shape(args.population_size)
        print(
            f"Evolution: cellular {rows}x{columns} toroidal grid, "
            f"Moore neighborhood, tournament={args.cellular_tournament_size}"
        )
    else:
        print("Evolution: global over-selection")
    try:
        result = evolve(primitive_set, args, output)
    except KeyboardInterrupt:
        print("\nTraining interrupted; generating artifacts from the latest completed generation.")
        generate_artifacts(args, output)
        return 130
    generate_artifacts(args, output)
    print(
        f"Finished after {result.generations_completed} generations. "
        f"Best fitness: {result.best.fitness.values[0]}"
    )
    print(f"Best tree: {result.best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
