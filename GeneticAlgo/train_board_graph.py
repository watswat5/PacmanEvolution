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
import signal
import subprocess
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from deap import algorithms, base, creator, gp, tools

try:
    from .gp_core import (
        build_move_primitive_set, repair_individual, safe_number, select_over,
    )
except ImportError:
    from gp_core import (
        build_move_primitive_set, repair_individual, safe_number, select_over,
    )

try:
    from PacmanGame import (
        ACTIONS,
        BASE_MOVE_FEATURE_NAMES,
        EVASION_FEATURE_NAMES,
        OPTIONAL_FEATURE_NAMES,
        MOVE_FEATURE_NAMES,
        Board,
        PacmanGame,
        precompute_board_graph,
        read_board,
        write_board,
    )
    from PacmanGame.generate_tetris_board import generate_board
    from PacmanGame.render_board import PELLET, PLAYER_LOCATION, POWER_PELLET
except ImportError as exc:
    raise ImportError("run this script from the project root") from exc


PROJECT_DIR = Path(__file__).resolve().parent.parent
VALIDATION_BOARD_PATH = PROJECT_DIR / "PacmanGame" / "test_board.txt"
GENERATION_SEED_STRIDE = 1_000_003
# Default training terminals: the original 19 features. The evasion features
# (EVASION_FEATURE_NAMES) join them only with --add-features.
FEATURE_NAMES = BASE_MOVE_FEATURE_NAMES
# Context entered around each batch of evaluations; the C++ backend uses it to
# convert each tree once per batch instead of once per board.
EVALUATION_SCOPE: Callable = nullcontext
# Replaces the "Evaluation workers" start-up line when another engine plays the
# games (FastCore/backend.py sets it).
EVALUATION_NOTE: str | None = None
DEATH_PENALTY = 10_000.0
NON_SCORING_MOVE_PENALTY = 100.0
REVISIT_PENALTY = 50.0
NORMALIZED_CLEAR_SCORE = 100_000.0
# Fraction of NORMALIZED_CLEAR_SCORE added when every pellet is eaten.
CLEAR_BONUS = 0.30
# Extra cost of dying or stagnating, as a fraction of the value of the pellets
# left uneaten, so an early death costs more than a late one.
DEATH_PELLET_PENALTY = 0.25
# Distinct top training trees that compete for generation champion on the
# fixed tournament boards.
CHAMPION_CANDIDATES = 10
# Size cost subtracted from the main fitness (``parsimony`` is a tie-break
# only). The per-node rate grows with the tree: it equals SIZE_PENALTY at
# SIZE_PENALTY_REFERENCE_NODES and is proportionally lower below that, so small
# trees are almost free and only bloat is expensive.
SIZE_PENALTY = 10.0
# Smallest depth of the random subtree that a mutation inserts. The largest is
# whatever still fits under the depth limit at the mutation point.
MUTATION_MIN_DEPTH = 1
SIZE_PENALTY_REFERENCE_NODES = 400
# With --late-pellet-share, the last LATE_PELLET_FRACTION of a board's pellets
# carries that share of the pellet score (0.5: the last 20% is worth as much as
# the first 80%).
LATE_PELLET_FRACTION = 0.20
# Fraction of the pellets already eaten when an endgame training board starts.
ENDGAME_START = 0.75


@dataclass(frozen=True)
class EvaluationRules:
    """Optional scoring rules; the defaults reproduce the original behaviour.

    stagnation_scaling: "fixed" cuts a game off after
    ``stagnation_termination_steps`` scoreless steps. "inverse" divides that
    limit by the fraction of the board's pellets still uneaten, so it is the
    same on a full board, double at half, and ten times longer with 10% left.
    late_pellet_share: see ``pellet_value``; None scores pellets linearly.
    """

    stagnation_scaling: str = "fixed"
    late_pellet_share: float | None = None


DEFAULT_RULES = EvaluationRules()


def pellet_value(completion_ratio: float, late_pellet_share: float | None = None) -> float:
    """Pellet score in [0, 1] for the fraction of a game's pellets eaten.

    Linear when ``late_pellet_share`` is None. Otherwise two straight lines:
    the first 80% of the pellets earn ``1 - late_pellet_share`` in total and
    the last 20% earn ``late_pellet_share``.
    """
    if late_pellet_share is None:
        return completion_ratio
    if completion_ratio >= 1.0:
        return 1.0
    early = 1.0 - LATE_PELLET_FRACTION
    if completion_ratio <= early:
        return (1.0 - late_pellet_share) * completion_ratio / early
    return (1.0 - late_pellet_share) + late_pellet_share * (
        (completion_ratio - early) / LATE_PELLET_FRACTION
    )


def stagnation_limit(
    base_steps: int, pellets_remaining: int, pellets_total: int, scaling: str = "fixed"
) -> float:
    """Scoreless steps allowed before a game is cut off for stagnation."""
    if scaling == "fixed":
        return base_steps
    return base_steps * pellets_total / max(1, pellets_remaining)


ENDGAME_STYLES = ("sweep", "islands")


def _island_pellets(
    board: Board,
    neighbors,
    pellets: set[tuple[int, int]],
    keep: int,
    variant: int,
) -> set[tuple[int, int]]:
    """Choose ``keep`` pellets as two to four separate patches near corners.

    Each patch starts at the pellet nearest a maze corner and grows outwards
    along the corridors, so it is one connected piece. The patches take turns
    growing, which keeps them about the same size. ``variant`` picks the
    corners, through its own random generator.
    """
    rng = random.Random(variant)
    corners = [
        (0, 0), (board.width - 1, 0),
        (0, board.height - 1), (board.width - 1, board.height - 1),
    ]
    chosen = rng.sample(corners, rng.choice((2, 3, 4)))
    growth_orders = []
    for corner_x, corner_y in chosen:
        centre = min(
            pellets,
            key=lambda point: (abs(point[0] - corner_x) + abs(point[1] - corner_y), point),
        )
        seen = {centre}
        queue = deque([centre])
        order = []
        while queue:
            point = queue.popleft()
            if point in pellets:
                order.append(point)
            for adjacent in neighbors.get(point, {}).values():
                if adjacent not in seen:
                    seen.add(adjacent)
                    queue.append(adjacent)
        growth_orders.append(deque(order))
    kept: set[tuple[int, int]] = set()
    while len(kept) < keep and any(growth_orders):
        for order in growth_orders:
            while order and order[0] in kept:
                order.popleft()
            if order and len(kept) < keep:
                kept.add(order.popleft())
    return kept


def endgame_board(
    board: Board,
    eaten_fraction: float = ENDGAME_START,
    style: str = "sweep",
    variant: int = 0,
) -> Board:
    """Return ``board`` as it looks once ``eaten_fraction`` of it is eaten.

    A ghost-free walker starts where Pac-Man does and always goes to the
    nearest pellet; Pac-Man starts where the walker stopped. Two styles:

    - "sweep": the walker eats until the fraction is reached, which leaves
      whole corridors, mostly in one or two connected regions.
    - "islands": two to four patches of pellets near corners are set aside
      first (``_island_pellets``; ``variant`` picks the corners) and the walker
      eats everything else, going around the patches where it can.

    Both are deterministic. ``pellets_eaten`` records the head start, so the
    game releases every ghost at once and stagnation scaling still sees the
    whole board; fitness counts only the pellets eaten from here on.
    """
    if style not in ENDGAME_STYLES:
        raise ValueError(f"unknown endgame style: {style}")
    collectible = PELLET | POWER_PELLET
    neighbors = precompute_board_graph(board).neighbors
    tiles = [row.copy() for row in board.tiles]
    pellets = {(x, y) for x, y, value in board.iter_tiles() if value & collectible}
    total = len(pellets)
    to_eat = min(max(0, total - 1), int(total * eaten_fraction))
    kept = (
        _island_pellets(board, neighbors, pellets, total - to_eat, variant)
        if style == "islands"
        else set()
    )
    position = board.player_pos
    direction = next(
        (action for action in ACTIONS if getattr(board, f"player_{action}")), "left"
    )

    def nearest_pellet(avoid: set) -> tuple | None:
        """Breadth-first search to the nearest edible pellet: (tile, last step)."""
        arrived_by = {position: direction}
        queue = deque([position])
        while queue:
            point = queue.popleft()
            if (
                point != position
                and point not in kept
                and tiles[point[1]][point[0]] & collectible
            ):
                return point, arrived_by[point]
            for action, adjacent in neighbors.get(point, {}).items():
                if adjacent not in arrived_by and adjacent not in avoid:
                    arrived_by[adjacent] = action
                    queue.append(adjacent)
        return None

    eaten = 0
    while eaten < to_eat:
        # Walk around the set-aside patches; cross one only if there is no
        # other way to the next pellet.
        found = nearest_pellet(kept) or (nearest_pellet(set()) if kept else None)
        if found is None:
            break  # the remaining pellets cannot be reached from here
        position, direction = found
        tiles[position[1]][position[0]] &= ~collectible
        eaten += 1
    if any(value & PLAYER_LOCATION for row in tiles for value in row):
        for row in tiles:
            for index, value in enumerate(row):
                row[index] = value & ~PLAYER_LOCATION
        tiles[position[1]][position[0]] |= PLAYER_LOCATION
    return board.copy(
        tiles=tiles,
        player_pos=position,
        pellets_eaten=board.pellets_eaten + eaten,
        pellets_remaining=total - eaten,
        **{f"player_{action}": action == direction for action in ACTIONS},
    )


def is_endgame_slot(
    sequence_index: int, boards_per_generation: int, endgame_boards: int
) -> bool:
    """Whether the ``sequence_index``-th training board of a run is an endgame.

    The pattern repeats every ``boards_per_generation`` boards and spreads the
    endgames evenly, so however the boards rotate, the set in use always holds
    exactly ``endgame_boards`` of them.
    """
    slot = sequence_index % boards_per_generation
    return (
        (slot + 1) * endgame_boards // boards_per_generation
        > slot * endgame_boards // boards_per_generation
    )


def endgame_style_for(
    sequence_index: int,
    boards_per_generation: int,
    endgame_boards: int,
    styles: Sequence[str],
) -> str:
    """The endgame style of a training board, or "" for a full game.

    Successive endgame boards cycle through ``styles`` in order.
    """
    if not is_endgame_slot(sequence_index, boards_per_generation, endgame_boards):
        return ""
    slot = sequence_index % boards_per_generation
    ordinal = (
        sequence_index // boards_per_generation * endgame_boards
        + slot * endgame_boards // boards_per_generation
    )
    return styles[ordinal % len(styles)]


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


def _tree_height(tree) -> int:
    """``PrimitiveTree.height`` (depth of the deepest node, root 0), the same
    algorithm without the per-node ``max()`` call; about twice as fast.
    Height is checked for every child (DEAP's ``staticLimit``) and every tree
    in the statistics, so it is one of the larger Python costs per generation."""
    stack = [0]
    pop = stack.pop
    extend = stack.extend
    deepest = 0
    for node in tree:
        depth = pop()
        if depth > deepest:
            deepest = depth
        arity = node.arity
        if arity:
            extend((depth + 1,) * arity)
    return deepest


# Set on DEAP's PrimitiveTree itself: unpickled trees belong to a class DEAP
# recreates on load, not to Individual. Same values, so other code is unaffected.
gp.PrimitiveTree.height = property(_tree_height)


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    cleared: int
    died: int
    stagnated: int
    timed_out: int
    validation_fitness: float
    validation_pellet_ratio: float
    best_validation_fitness_so_far: float
    best_validation_pellet_ratio_so_far: float
    validation_cleared: int
    validation_died: int
    validation_stagnated: int
    validation_timed_out: int
    tournament_fitness: float
    tournament_pellet_ratio: float
    best_tournament_fitness_so_far: float
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
    stagnated: int
    timed_out: int


def node_depths(tree: Sequence) -> list[int]:
    """Depth of every node of a prefix-ordered tree (the root is 0)."""
    depths = []
    pending = [0]
    for node in tree:
        depth = pending.pop()
        depths.append(depth)
        pending.extend([depth + 1] * node.arity)
    return depths


def mutate_within_depth(
    individual: Individual,
    pset: gp.PrimitiveSet,
    max_height: int,
    min_depth: int = MUTATION_MIN_DEPTH,
) -> tuple[Individual]:
    """Replace a random subtree with a new one that fits the depth limit.

    The new subtree's depth ranges from ``min_depth`` to the room left
    below the chosen node, so a mutation is never discarded for being too deep.
    With a minimum depth of 0, genGrow stops at a single terminal 72% of the
    time (24 terminals to 10 functions), which steadily shrinks trees; a node
    already at the limit can still only become a terminal.
    """
    index = random.randrange(len(individual))
    room = max(0, max_height - node_depths(individual)[index])
    subtree = gp.genGrow(pset, min_=min(min_depth, room), max_=room)
    individual[individual.searchSubtree(index)] = subtree
    return (individual,)


def initial_expression(
    pset: gp.PrimitiveSet,
    min_: int,
    max_: int,
    full_min: int | None = None,
    terminal_probability: float | None = None,
) -> list:
    """Ramped half-and-half with two adjustable parts (heights, not depths).

    Half the trees are "full", with a height drawn evenly from ``full_min``
    (default ``min_``) to ``max_``. The other half "grow": height drawn from
    ``min_`` to ``max_``, and below ``min_`` each node becomes a terminal with
    ``terminal_probability`` (default: the terminals' share of the primitive
    set, which is DEAP's rule and about 0.71 here).
    """
    if random.random() < 0.5:
        return gp.genFull(pset, min_ if full_min is None else full_min, max_)
    if terminal_probability is None:
        return gp.genGrow(pset, min_, max_)

    def stop(height: int, depth: int) -> bool:
        return depth == height or (
            depth >= min_ and random.random() < terminal_probability
        )

    return gp.generate(pset, min_, max_, stop)


def build_toolbox(
    primitive_set: gp.PrimitiveSet,
    min_depth: int,
    max_depth: int,
    max_evolved_depth: int,
    top_fraction: float,
    top_probability: float,
    repair: bool = False,
    mutation_min_depth: int = MUTATION_MIN_DEPTH,
    full_min_depth: int | None = None,
    grow_terminal_probability: float | None = None,
) -> base.Toolbox:
    """Build operators for a single move-scoring PrimitiveTree."""
    toolbox = base.Toolbox()
    if full_min_depth is None and grow_terminal_probability is None:
        # DEAP's own generator, so existing seeds give the same population.
        toolbox.register(
            "expression",
            gp.genHalfAndHalf,
            pset=primitive_set,
            min_=min_depth - 1,
            max_=max_depth - 1,
        )
    else:
        toolbox.register(
            "expression",
            initial_expression,
            primitive_set,
            min_depth - 1,
            max_depth - 1,
            None if full_min_depth is None else full_min_depth - 1,
            grow_terminal_probability,
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
    toolbox.register(
        "mutate",
        mutate_within_depth,
        pset=primitive_set,
        max_height=max_evolved_depth - 1,
        min_depth=mutation_min_depth,
    )
    height_limit = gp.staticLimit(
        key=lambda tree: tree.height, max_value=max_evolved_depth - 1
    )
    toolbox.decorate("mate", height_limit)
    toolbox.decorate("mutate", height_limit)
    if repair:
        # Collapse trivial chains (abs(abs x), x * 1, ...) in every offspring.
        # Behaviour is unchanged; the tree just gets smaller and shallower.
        def repaired(operator: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return tuple(
                    repair_individual(child, primitive_set)
                    for child in operator(*args, **kwargs)
                )
            return wrapper

        toolbox.decorate("mate", repaired)
        toolbox.decorate("mutate", repaired)
    return toolbox


def build_primitive_set(feature_names=MOVE_FEATURE_NAMES) -> gp.PrimitiveSet:
    """Primitive set over ``feature_names``; the default (every feature) can
    run any saved tree. Training builds its own from FEATURE_NAMES and
    --add-features / --exclude-features."""
    return build_move_primitive_set(feature_names)


# Compiled trees are cached by expression and argument names, not by the
# primitive-set object: a worker receives a freshly unpickled primitive set
# with every task, and keying on the object would never hit across tasks and
# would keep every one of those copies alive.
_PRIMITIVE_SETS: dict[tuple[str, ...], gp.PrimitiveSet] = {}


@lru_cache(maxsize=2048)
def _compile_expression(expression: str, arguments: tuple[str, ...]) -> Callable:
    return gp.compile(expr=expression, pset=_PRIMITIVE_SETS[arguments])


def _compile_tree(expression: str, primitive_set: gp.PrimitiveSet) -> Callable:
    arguments = tuple(primitive_set.arguments)
    _PRIMITIVE_SETS.setdefault(arguments, primitive_set)
    return _compile_expression(expression, arguments)


def compile_policy(individual: Individual, primitive_set: gp.PrimitiveSet) -> Callable | None:
    """The tree as a callable, or None if it cannot be compiled."""
    try:
        return _compile_tree(str(individual), primitive_set)
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        return None


_COMPILE = object()


def choose_move(
    individual: Individual,
    observations: dict[str, dict[str, float]],
    primitive_set: gp.PrimitiveSet,
    last_direction: str,
    rng: random.Random,
    *,
    random_tie_break: bool = False,
    policy: Callable | None | object = _COMPILE,
) -> tuple[str | None, dict[str, float]]:
    """Score every legal candidate with the same tree.

    ``policy`` is the tree compiled with ``compile_policy``; pass it when
    calling this every step, since compiling means writing the tree out as
    text, which costs more than the move itself.
    """
    if not observations:
        return None, {}
    if policy is _COMPILE:
        policy = compile_policy(individual, primitive_set)
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


def size_cost(node_count: int, size_penalty: float = SIZE_PENALTY) -> float:
    """Quadratic size cost: ``size_penalty`` per node at the reference size."""
    return size_penalty * node_count * node_count / SIZE_PENALTY_REFERENCE_NODES


def calculate_fitness(
    game: PacmanGame,
    starting_pellets_eaten: int,
    starting_pellets_remaining: int,
    individual: Individual,
    steps_taken: int,
    timestep_limit: int,
    stagnation_cost: float,
    clear_bonus: float = CLEAR_BONUS,
    stagnated: bool = False,
    death_penalty: float = DEATH_PENALTY,
    death_pellet_penalty: float = DEATH_PELLET_PENALTY,
    size_penalty: float = SIZE_PENALTY,
    late_pellet_share: float | None = None,
) -> float:
    pellets_consumed = game.board.pellets_eaten - starting_pellets_eaten
    # Relative to the pellets on the board at the start, so an endgame board
    # scores 0 where it starts and the same as a full game when cleared.
    completion_ratio = pellet_value(
        pellets_consumed / max(1, starting_pellets_remaining), late_pellet_share
    )
    # A stagnation cutoff costs the same as dying, so looping never beats death.
    terminated_early = (game.game_over and not game.won) or stagnated
    return (
        NORMALIZED_CLEAR_SCORE * completion_ratio
        + NORMALIZED_CLEAR_SCORE * clear_bonus * game.won
        - terminated_early * (
            death_penalty
            + death_pellet_penalty * NORMALIZED_CLEAR_SCORE * (1.0 - completion_ratio)
        )
        - stagnation_cost
        - size_cost(len(individual), size_penalty)
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
    revisit_penalty: float = REVISIT_PENALTY,
    clear_bonus: float = CLEAR_BONUS,
    death_penalty: float = DEATH_PENALTY,
    death_pellet_penalty: float = DEATH_PELLET_PENALTY,
    size_penalty: float = SIZE_PENALTY,
    seed: int = 0,
    index_offset: int = 0,
    random_tie_break: bool = False,
    rules: EvaluationRules = DEFAULT_RULES,
) -> tuple[list[float], OutcomeCounts, list[float], list[str]]:
    """Evaluate all individuals while sharing one precomputed static graph."""
    graph = precompute_board_graph(starting_board)
    games = [
        PacmanGame(starting_board.copy(), board_graph=graph)
        for _ in population
    ]
    starting_pellets_eaten = [game.board.pellets_eaten for game in games]
    starting_pellets_remaining = [game.board.pellets_remaining for game in games]
    # The whole board, including an endgame board's head start.
    total_pellets = starting_board.pellets_eaten + starting_board.pellets_remaining
    scoreless_steps = [0] * len(games)
    steps_taken = [0] * len(games)
    stagnation_costs = [0.0] * len(games)
    stagnated = [False] * len(games)
    rngs = [random.Random(seed + index_offset + index) for index in range(len(games))]
    policies = [compile_policy(individual, primitive_set) for individual in population]
    active = list(range(len(games)))

    for _ in range(timestep_limit):
        for index in active:
            game = games[index]
            observations = game.get_move_observations()
            move, _ = choose_move(
                population[index],
                observations,
                primitive_set,
                game.player_direction,
                rngs[index],
                random_tie_break=random_tie_break,
                policy=policies[index],
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
                # Extra cost for dithering: stepping back onto a tile visited
                # within VISIT_WINDOW steps without scoring anything.
                if move and observations[move]["recently_visited"] > 0:
                    stagnation_costs[index] += revisit_penalty
                if scoreless_steps[index] >= stagnation_limit(
                    stagnation_termination_steps,
                    game.board.pellets_remaining,
                    total_pellets,
                    rules.stagnation_scaling,
                ):
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
            clear_bonus,
            stagnated[index],
            death_penalty,
            death_pellet_penalty,
            size_penalty,
            rules.late_pellet_share,
        )
        for index, game in enumerate(games)
    ]
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = (fitness, -parsimony * len(individual))
    outcome_labels = [
        "cleared" if game.won
        else "died" if game.game_over
        else "stagnated" if stagnated[index]
        else "timed_out"
        for index, game in enumerate(games)
    ]
    outcomes = OutcomeCounts(
        cleared=outcome_labels.count("cleared"),
        died=outcome_labels.count("died"),
        stagnated=outcome_labels.count("stagnated"),
        timed_out=outcome_labels.count("timed_out"),
    )
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
        stagnated=flat_labels.count("stagnated"),
        timed_out=flat_labels.count("timed_out"),
    )


def _evaluate_population_chunk(task: tuple) -> tuple[int, list[float], list[float], list[str]]:
    """Evaluate a contiguous population slice in a worker process."""
    (
        start, individuals, board, timestep_limit, primitive_set, parsimony,
        stagnation_penalty, stagnation_grace_steps,
        stagnation_termination_steps, revisit_penalty, clear_bonus,
        death_penalty, death_pellet_penalty, size_penalty, seed,
        random_tie_break, rules,
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
        revisit_penalty=revisit_penalty,
        clear_bonus=clear_bonus,
        death_penalty=death_penalty,
        death_pellet_penalty=death_pellet_penalty,
        size_penalty=size_penalty,
        seed=seed,
        index_offset=start,
        random_tie_break=random_tie_break,
        rules=rules,
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
    revisit_penalty: float = REVISIT_PENALTY,
    clear_bonus: float = CLEAR_BONUS,
    death_penalty: float = DEATH_PENALTY,
    death_pellet_penalty: float = DEATH_PELLET_PENALTY,
    size_penalty: float = SIZE_PENALTY,
    executor: ProcessPoolExecutor | None = None,
    workers: int = 1,
    rules: EvaluationRules = DEFAULT_RULES,
) -> tuple[list[float], OutcomeCounts, list[float], list[tuple[str, ...]]]:
    """Evaluate every candidate on the same boards and average its fitness."""
    with EVALUATION_SCOPE():
        return _evaluate_population_on_boards(
            population, boards, timestep_limit, primitive_set,
            parsimony=parsimony, stagnation_penalty=stagnation_penalty,
            stagnation_grace_steps=stagnation_grace_steps,
            stagnation_termination_steps=stagnation_termination_steps,
            random_tie_break=random_tie_break, revisit_penalty=revisit_penalty,
            clear_bonus=clear_bonus, death_penalty=death_penalty,
            death_pellet_penalty=death_pellet_penalty, size_penalty=size_penalty,
            executor=executor, workers=workers, rules=rules,
        )


def _evaluate_population_on_boards(
    population, boards, timestep_limit, primitive_set, *, parsimony,
    stagnation_penalty, stagnation_grace_steps, stagnation_termination_steps,
    random_tie_break, revisit_penalty, clear_bonus, death_penalty,
    death_pellet_penalty, size_penalty, executor, workers, rules,
):
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
                revisit_penalty=revisit_penalty,
                clear_bonus=clear_bonus,
                death_penalty=death_penalty,
                death_pellet_penalty=death_pellet_penalty,
                size_penalty=size_penalty,
                seed=layout_seed,
                random_tie_break=random_tie_break,
                rules=rules,
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
                    revisit_penalty, clear_bonus, death_penalty,
                    death_pellet_penalty, size_penalty, layout_seed,
                    random_tie_break, rules,
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


class BoardScores(dict):
    """A tree's cached per-board results. Never changed in place (a new one
    is made whenever results are added or dropped), so copies of a tree may
    share it: deepcopy returns the same object. Copying a tree is otherwise
    dominated by copying these, once per parent copy in every crossover."""

    def __deepcopy__(self, memo):
        return self


def evaluate_population_with_board_cache(
    population: Sequence[Individual],
    boards: Sequence[tuple[int, Board]],
    timestep_limit: int,
    primitive_set: gp.PrimitiveSet,
    *,
    parsimony: float,
    running_sum: bool = False,
    **settings,
) -> tuple[list[float], OutcomeCounts, list[float], list[tuple[str, ...]]]:
    """``evaluate_population_on_boards`` that never replays a game.

    Each tree remembers its result on every board in ``boards`` (in its
    ``board_scores``, keyed by layout seed) and plays only the boards it has
    not seen. With a rolling window of boards, the population therefore plays
    one new game per generation while a new child plays the whole window, and
    everyone's fitness is the mean over the same boards. Results for boards
    that have left the window are forgotten. Takes the same settings as
    ``evaluate_population_on_boards``; relies on games being deterministic, so
    it must not be combined with random tie-breaking.

    ``running_sum`` averages exactly as ``evaluate_population_on_boards``
    does (adding board by board), so the cellular method gets bit-identical
    fitness with or without the cache. Steady state keeps ``sum()``, which
    on Python 3.12 rounds differently in the last digit, so its earlier runs
    stay reproducible.
    """
    with EVALUATION_SCOPE():
        _fill_board_cache(population, boards, timestep_limit, primitive_set,
                          parsimony=parsimony, **settings)
    window = [layout_seed for layout_seed, _ in boards]
    return _cached_averages(population, window, parsimony, running_sum)


def _fill_board_cache(population, boards, timestep_limit, primitive_set, *, parsimony, **settings):
    for entry in boards:
        layout_seed = entry[0]
        unseen = [
            tree for tree in population
            if layout_seed not in getattr(tree, "board_scores", {})
        ]
        if not unseen:
            continue
        fitnesses, _, pellet_ratios, labels = evaluate_population_on_boards(
            unseen, [entry], timestep_limit, primitive_set,
            parsimony=parsimony, **settings,
        )
        for tree, fitness, pellet_ratio, label in zip(
            unseen, fitnesses, pellet_ratios, labels
        ):
            tree.board_scores = BoardScores({
                **getattr(tree, "board_scores", {}),
                layout_seed: (fitness, pellet_ratio, label[0]),
            })


def _cached_averages(population, window, parsimony, running_sum):
    average_fitnesses, average_pellet_ratios, all_labels = [], [], []
    for tree in population:
        tree.board_scores = BoardScores(
            (seed, tree.board_scores[seed]) for seed in window
        )
        results = list(tree.board_scores.values())
        if running_sum:
            fitness_total = pellet_total = 0.0
            for result in results:
                fitness_total += result[0]
                pellet_total += result[1]
            fitness = fitness_total / len(results)
            pellet_ratio = pellet_total / len(results)
        else:
            fitness = sum(result[0] for result in results) / len(results)
            pellet_ratio = sum(result[1] for result in results) / len(results)
        tree.fitness.values = (fitness, -parsimony * len(tree))
        average_fitnesses.append(fitness)
        average_pellet_ratios.append(pellet_ratio)
        all_labels.append(tuple(result[2] for result in results))
    return average_fitnesses, count_outcomes(all_labels), average_pellet_ratios, all_labels


def population_statistics(
    population: Sequence[Individual],
    generation: int,
    outcomes: OutcomeCounts,
    validation_fitness: float,
    validation_pellet_ratio: float,
    best_validation_fitness_so_far: float,
    best_validation_pellet_ratio_so_far: float,
    validation_outcomes: OutcomeCounts,
    tournament_fitness: float,
    tournament_pellet_ratio: float,
    best_tournament_fitness_so_far: float,
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
        stagnated=outcomes.stagnated,
        timed_out=outcomes.timed_out,
        validation_fitness=validation_fitness,
        validation_pellet_ratio=validation_pellet_ratio,
        best_validation_fitness_so_far=best_validation_fitness_so_far,
        best_validation_pellet_ratio_so_far=best_validation_pellet_ratio_so_far,
        validation_cleared=validation_outcomes.cleared,
        validation_died=validation_outcomes.died,
        validation_stagnated=validation_outcomes.stagnated,
        validation_timed_out=validation_outcomes.timed_out,
        tournament_fitness=tournament_fitness,
        tournament_pellet_ratio=tournament_pellet_ratio,
        best_tournament_fitness_so_far=best_tournament_fitness_so_far,
        best_fitness=float(fitnesses.max()),
        average_fitness=float(fitnesses.mean()),
        worst_fitness=float(fitnesses.min()),
        average_tree_depth=float(depths.mean()),
        best_tree_depth=best.height + 1,
        average_node_count=float(sizes.mean()),
        best_node_count=len(best),
    )


# Rows of each history file this process has written, so later calls only
# append the new rows instead of rewriting a file that grows every generation.
_HISTORY_WRITTEN: dict[Path, tuple[int, int]] = {}


def outcome_shares(stats: GenerationStats) -> str:
    """Cleared/died/stagnated/timed-out as shares of the generation's games
    (0 to 1), so runs with different numbers of games or boards compare
    directly. history.csv keeps the raw counts."""
    counts = (stats.cleared, stats.died, stats.stagnated, stats.timed_out)
    games = sum(counts) or 1
    return "/".join(f"{count / games:.3f}" for count in counts)


def write_history(path: Path, history: Sequence[GenerationStats]) -> None:
    """Write history.csv; appends when only new rows were added since this
    process last wrote it (the file comes out the same as a full rewrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(GenerationStats.__dataclass_fields__)
    written = _HISTORY_WRITTEN.get(path)
    if (
        written is not None
        and written[0] <= len(history)
        and path.exists()
        and path.stat().st_size == written[1]
    ):
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerows(
                asdict(row) for row in history[written[0]:]
            )
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(row) for row in history)
    _HISTORY_WRITTEN[path] = (len(history), path.stat().st_size)


def write_diversity(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_individual(individual: Individual, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(individual, handle, protocol=pickle.HIGHEST_PROTOCOL)


CHECKPOINT_NAME = "checkpoint.pkl"
# Settings that must match for a checkpoint to be resumed.
CHECKPOINT_SETTINGS = (
    "seed", "population_size", "evolution_method", "boards_per_generation",
    "boards_replaced_per_generation", "timestep_limit",
)


# Newer settings, with the value assumed for checkpoints saved before they
# existed.
CHECKPOINT_OPTIONAL_SETTINGS = {
    "stagnation_scaling": "fixed",
    "late_pellet_share": None,
    "endgame_boards": 0,
    "endgame_start": ENDGAME_START,
    "endgame_styles": "sweep",
    "steady_duel_boards": 1,
    "add_features": "",
}


def save_checkpoint(path: Path, state: dict) -> None:
    """Write the run state atomically so a crash never leaves half a file."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(path: Path, args: argparse.Namespace) -> dict | None:
    """Return the saved run state, or None when there is nothing to resume."""
    if not path.exists():
        return None
    with path.open("rb") as handle:
        state = pickle.load(handle)
    saved = {**CHECKPOINT_OPTIONAL_SETTINGS, **state["settings"]}
    for name in (*CHECKPOINT_SETTINGS, *CHECKPOINT_OPTIONAL_SETTINGS):
        if saved[name] != getattr(args, name):
            raise ValueError(
                f"cannot resume {path}: it was saved with {name}="
                f"{saved[name]!r}, but this run uses "
                f"{getattr(args, name)!r}"
            )
    return state


def champion_candidates(
    population: Sequence[Individual], count: int
) -> list[Individual]:
    """Copy the ``count`` best distinct trees by training fitness."""
    chosen: list[Individual] = []
    seen: set[str] = set()
    for tree in sorted(population, key=lambda tree: tree.fitness, reverse=True):
        expression = str(tree)
        if expression in seen:
            continue
        seen.add(expression)
        chosen.append(copy.deepcopy(tree))
        if len(chosen) == count:
            break
    return chosen


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


# --- Population diversity ---------------------------------------------------
# Measured on what trees do, not on how they are written: every tree is asked
# for its move in the same fixed sample of game situations.
DIVERSITY_STATES = 300
DIVERSITY_BOARDS = 6
DIVERSITY_BOARD_SEED = 2_400_000_000  # clear of training and playoff boards
DIVERSITY_STEPS_PER_BOARD = 250


@lru_cache(maxsize=1)
def diversity_states() -> tuple[tuple[dict[str, dict[str, float]], str], ...]:
    """A fixed sample of (move observations, current direction) situations.

    The same for every run, so diversity numbers are comparable between runs.
    A wandering Pac-Man that usually takes an adjacent pellet plays a few
    random boards; situations with a real choice (two or more legal moves) are
    recorded. Uses its own random generator and leaves the global one alone.
    """
    global_state = random.getstate()
    try:
        rng = random.Random(DIVERSITY_BOARD_SEED)
        states = []
        for index in range(DIVERSITY_BOARDS):
            game = PacmanGame(generate_board(DIVERSITY_BOARD_SEED + index))
            for _ in range(DIVERSITY_STEPS_PER_BOARD):
                if game.game_over or game.won:
                    break
                observations = game.get_move_observations()
                if not observations:
                    game.step("stay")
                    continue
                if len(observations) >= 2:
                    states.append((
                        {move: dict(features) for move, features in observations.items()},
                        game.player_direction,
                    ))
                moves = sorted(observations)
                with_pellet = [
                    move for move in moves if observations[move].get("pellet_here", 0) > 0
                ]
                game.step(rng.choice(with_pellet if with_pellet and rng.random() < 0.8 else moves))
        return tuple(rng.sample(states, min(DIVERSITY_STATES, len(states))))
    finally:
        random.setstate(global_state)


def move_signature(tree: Individual, primitive_set: gp.PrimitiveSet) -> tuple[str, ...]:
    """The move ``tree`` picks in each sampled situation (remembered on it)."""
    signature = getattr(tree, "move_signature", None)
    if signature is None:
        rng = random.Random(0)  # unused: ties are not broken at random
        policy = compile_policy(tree, primitive_set)
        signature = tuple(
            choose_move(tree, observations, primitive_set, direction, rng, policy=policy)[0]
            for observations, direction in diversity_states()
        )
        tree.move_signature = signature
    return signature


def population_diversity(
    population: Sequence[Individual], primitive_set: gp.PrimitiveSet
) -> tuple[float, int]:
    """(move diversity, distinct behaviours) for a population.

    Move diversity is the chance that two different trees picked at random
    choose different moves in a sampled situation: 0 when every tree plays
    alike. Distinct behaviours counts the trees that differ from every other
    in at least one sampled situation, counting each group of identical
    players once.
    """
    signatures = [move_signature(tree, primitive_set) for tree in population]
    size = len(signatures)
    if size < 2:
        return 0.0, len(set(signatures))
    pairs = size * (size - 1)
    agreeing = sum(
        sum(count * (count - 1) for count in Counter(moves).values())
        for moves in zip(*signatures)
    )
    return 1.0 - agreeing / (pairs * len(signatures[0])), len(set(signatures))


def steady_state_children(
    population: Sequence[Individual],
    toolbox: base.Toolbox,
    count: int,
    crossover_probability: float,
    tournament_size: int,
) -> list[Individual]:
    """Breed ``count`` children, each from two tournament-selected parents.

    With ``crossover_probability`` the child is one of the two crossover
    offspring, chosen at random; otherwise it is a mutated copy of the first
    parent. The parents must already have a fitness.
    """
    children = []
    for _ in range(count):
        first, second = (
            max(random.sample(list(population), min(tournament_size, len(population))),
                key=lambda tree: tree.fitness)
            for _ in range(2)
        )
        if random.random() < crossover_probability:
            child = random.choice(
                toolbox.mate(copy.deepcopy(first), copy.deepcopy(second))
            )
        else:
            child, = toolbox.mutate(copy.deepcopy(first))
        del child.fitness.values
        # A copy of a parent carries the parent's remembered board results.
        child.board_scores = {}
        child.move_signature = None
        children.append(child)
    return children


def steady_state_replace(
    population: list[Individual],
    children: Sequence[Individual],
    worst_fraction: float,
    pool_size: int,
    duel_score: Callable[[Individual], float] | None = None,
    behaviour: Callable[[Individual], object] | None = None,
    replace_on_tie: bool = False,
) -> list[tuple[int, int]]:
    """Insert each child that beats a randomly drawn weak tree; edit in place.

    For every child in turn: draw ``pool_size`` trees at random from the worst
    ``worst_fraction`` of the population, take the worst of those, and replace
    it if the child is strictly fitter. Everything must have been scored on
    the same boards. Returns (population index, child index) per replacement.

    The bottom of the population is always ranked by fitness. ``duel_score``,
    if given, replaces fitness for the contest itself: which of the drawn
    trees is worst, and whether the child beats it. It is called only for the
    child and the drawn trees, so it can afford extra games.

    ``behaviour``, if given, returns a key that is equal for trees that played
    identically. A child whose key matches a tree already in the population is
    dropped without a contest, so copies of one strategy cannot multiply.

    ``replace_on_tie`` lets a child that exactly equals its opponent's score
    (the first fitness value, which includes the size cost) take its place, so
    a different genome that plays as well can drift into the population.
    Otherwise the child must be strictly better, and on equal scores the
    smaller tree wins through the parsimony tie-break.
    """
    score = duel_score or (lambda tree: tree.fitness)
    present = (
        Counter(behaviour(tree) for tree in population) if behaviour else None
    )
    replaced = []
    worst_count = max(1, min(len(population), round(len(population) * worst_fraction)))
    for child_index, child in enumerate(children):
        if present is not None and present[behaviour(child)] > 0:
            continue
        weakest = sorted(range(len(population)), key=lambda index: population[index].fitness)
        pool = random.sample(weakest[:worst_count], min(pool_size, worst_count))
        victim = min(pool, key=lambda index: score(population[index]))
        if replace_on_tie:
            child_score, victim_score = score(child), score(population[victim])
            if duel_score is None:
                child_score, victim_score = child_score.values[0], victim_score.values[0]
            wins = child_score >= victim_score
        else:
            wins = score(child) > score(population[victim])
        if wins:
            if present is not None:
                present[behaviour(population[victim])] -= 1
                present[behaviour(child)] += 1
            population[victim] = child
            replaced.append((victim, child_index))
    return replaced


def evolve(
    primitive_set: gp.PrimitiveSet,
    args: argparse.Namespace,
    output_directory: Path,
) -> EvolutionResult:
    checkpoint_path = output_directory / CHECKPOINT_NAME
    checkpoint = load_checkpoint(checkpoint_path, args) if args.resume else None
    random.seed(args.seed)
    toolbox = build_toolbox(
        primitive_set,
        args.min_initial_depth,
        args.max_initial_depth,
        args.max_evolved_depth,
        args.top_fraction,
        args.top_probability,
        repair=args.repair,
        mutation_min_depth=args.mutation_min_depth,
        full_min_depth=args.full_min_initial_depth,
        grow_terminal_probability=args.grow_terminal_probability,
    )
    population = toolbox.population(n=args.population_size)
    if args.repair:
        for individual in population:
            repair_individual(individual, primitive_set)
    hall_of_fame = tools.HallOfFame(1)
    history = []
    stopped_early = False
    best_validation_fitness = float("-inf")
    best_validation_rank = (float("-inf"), float("-inf"))
    best_validation_pellet_ratio = 0.0
    best_tournament_rank = (float("-inf"), float("-inf"))
    best_tournament_fitness = float("-inf")
    validation_board = read_board(VALIDATION_BOARD_PATH)
    precompute_board_graph(validation_board)
    # Fixed seeded boards make champions from different generations comparable.
    # --tournament-seed gives several runs the same boards (island runs need
    # this to compare champions); otherwise they derive from the run's seed.
    tournament_base = (
        args.tournament_seed
        if args.tournament_seed is not None
        else args.seed + (args.generations + 1) * GENERATION_SEED_STRIDE
    )
    tournament_seeds = [tournament_base + index for index in range(10)]
    boards: list[tuple[int, Board]] = []
    # Parallel to ``boards``: each one's endgame style, "" for a full game.
    endgame_flags: list[str] = []
    styles = endgame_styles(args)
    rules = EvaluationRules(
        stagnation_scaling=args.stagnation_scaling,
        late_pellet_share=args.late_pellet_share,
    )

    def training_board(layout_seed: int, style: str) -> Board:
        board = generate_board(layout_seed)
        if not style:
            return board
        return endgame_board(board, args.endgame_start, style, variant=layout_seed)

    start_generation = 0
    if checkpoint is not None:
        # Continue exactly where the last completed generation left off.
        population = checkpoint["population"]
        random.setstate(checkpoint["random_state"])
        history = list(checkpoint["history"])
        hall_of_fame.update([checkpoint["best"]])
        stopped_early = checkpoint["stopped_early"]
        best_validation_fitness = checkpoint["best_validation_fitness"]
        best_validation_rank = checkpoint["best_validation_rank"]
        best_validation_pellet_ratio = checkpoint["best_validation_pellet_ratio"]
        best_tournament_rank = checkpoint["best_tournament_rank"]
        best_tournament_fitness = checkpoint["best_tournament_fitness"]
        # Keep the original tournament boards even if --generations changed.
        tournament_seeds = checkpoint["tournament_seeds"]
        endgame_flags = list(
            checkpoint.get("board_endgame", [""] * len(checkpoint["board_seeds"]))
        )
        boards = [
            (seed, training_board(seed, endgame))
            for seed, endgame in zip(checkpoint["board_seeds"], endgame_flags)
        ]
        start_generation = checkpoint["next_generation"]
        print(
            f"Resuming from {checkpoint_path} at generation {start_generation} "
            f"of {args.generations}"
        )
        if stopped_early or start_generation >= args.generations:
            return EvolutionResult(
                best=hall_of_fame[0],
                population=population,
                generations_completed=len(history),
                stopped_early=stopped_early,
                history=tuple(history),
            )
        if args.evolution_method == "global" and not checkpoint["bred"]:
            # The run ended on its last generation, which skips breeding.
            parents = toolbox.select(population, args.population_size)
            population = algorithms.varOr(
                parents,
                toolbox,
                lambda_=args.population_size,
                cxpb=args.crossover_probability,
                mutpb=args.mutation_probability,
            )
    tournament_boards = [(seed, generate_board(seed)) for seed in tournament_seeds]
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    worker_count = min(args.population_size, args.workers or available_cpus)
    print(EVALUATION_NOTE or f"Evaluation workers: {worker_count}")
    worker_pool = (
        ProcessPoolExecutor(max_workers=worker_count)
        if worker_count > 1
        else nullcontext(None)
    )

    boards_replaced = (
        args.boards_replaced_per_generation or args.boards_per_generation
    )
    # Steady state scores every tree on one game only: no champion tournament
    # and no validation game.
    raw_champion = args.evolution_method == "steady"
    # Steady state keeps each tree's result per board, so with a rolling window
    # of boards only unseen games are played.
    # Cellular survivors keep their results too (--no-board-cache turns this
    # off): a parent replays only the boards that rotated in, a child plays
    # all of them. Same results as replaying, since games are deterministic.
    cellular_cache = (
        args.evolution_method == "cellular"
        and args.board_cache
        and not args.random_tie_break
    )
    if raw_champion:
        evaluate_training = evaluate_population_with_board_cache
    elif cellular_cache:
        def evaluate_training(*arguments, **settings):
            return evaluate_population_with_board_cache(
                *arguments, running_sum=True, **settings
            )
    else:
        evaluate_training = evaluate_population_on_boards
    steady_replacements = (
        checkpoint.get("steady_replacements", 0) if checkpoint is not None else 0
    )
    diversity_rows = list(checkpoint.get("diversity", [])) if checkpoint is not None else []
    with worker_pool as executor:
        for generation in range(start_generation, args.generations):
            # Rotate the training boards: after generation 0 only the oldest
            # ``boards_replaced`` are swapped for fresh layouts, so consecutive
            # generations are scored on mostly the same boards.
            new_board_count = (
                args.boards_per_generation if generation == 0 else boards_replaced
            )
            # Position of the first new board in the run's sequence of
            # training boards; it decides which ones are endgame starts.
            first_sequence_index = (
                0
                if generation == 0
                else args.boards_per_generation + (generation - 1) * boards_replaced
            )
            new_flags = [
                endgame_style_for(
                    first_sequence_index + board_index,
                    args.boards_per_generation,
                    args.endgame_boards,
                    styles,
                )
                for board_index in range(new_board_count)
            ]
            new_seeds = [
                args.seed + generation * GENERATION_SEED_STRIDE + board_index
                for board_index in range(new_board_count)
            ]
            boards = (
                boards
                + [
                    (layout_seed, training_board(layout_seed, endgame))
                    for layout_seed, endgame in zip(new_seeds, new_flags)
                ]
            )[-args.boards_per_generation:]
            endgame_flags = (endgame_flags + new_flags)[-args.boards_per_generation:]
            layout_seed = boards[0][0]
            # --save-interval N keeps the per-generation files (layouts and
            # the generation's best tree) for every N-th generation only.
            keep_files = (generation + 1) % args.save_interval == 0
            # Steady state adds one board per generation; saving the newest is
            # enough to rebuild any window.
            saved_boards = boards[-1:] if raw_champion else boards
            for board_index, (_, board) in enumerate(saved_boards if keep_files else ()):
                suffix = "" if board_index == 0 else f"_board_{board_index:04d}"
                layout_path = (
                    output_directory / "layouts" / f"generation_{generation:04d}{suffix}.txt"
                )
                write_board(board, layout_path)
            (
                fitnesses, outcomes, parent_pellet_ratios, parent_outcome_labels
            ) = evaluate_training(
                population,
                boards,
                args.timestep_limit,
                primitive_set,
                parsimony=args.parsimony,
                stagnation_penalty=args.stagnation_penalty,
                stagnation_grace_steps=args.stagnation_grace_steps,
                stagnation_termination_steps=args.stagnation_termination_steps,
                revisit_penalty=args.revisit_penalty,
                clear_bonus=args.clear_bonus,
                death_penalty=args.death_penalty,
                death_pellet_penalty=args.death_pellet_penalty,
                size_penalty=args.size_penalty,
                random_tie_break=args.random_tie_break,
                executor=executor,
                workers=worker_count,
                rules=rules,
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
                for child in children:
                    # Copies of a parent carry its results; a child plays
                    # every board itself.
                    child.board_scores = {}
                (
                    _,
                    _,
                    _,
                    child_outcome_labels,
                ) = evaluate_training(
                    children,
                    boards,
                    args.timestep_limit,
                    primitive_set,
                    parsimony=args.parsimony,
                    stagnation_penalty=args.stagnation_penalty,
                    stagnation_grace_steps=args.stagnation_grace_steps,
                    stagnation_termination_steps=args.stagnation_termination_steps,
                    revisit_penalty=args.revisit_penalty,
                    clear_bonus=args.clear_bonus,
                    death_penalty=args.death_penalty,
                    death_pellet_penalty=args.death_pellet_penalty,
                    size_penalty=args.size_penalty,
                    random_tie_break=args.random_tie_break,
                    executor=executor,
                    workers=worker_count,
                    rules=rules,
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

            elif args.evolution_method == "steady":
                # Steady state: a few children per board, each of which must
                # beat a weak tree drawn from the bottom of the population.
                children = steady_state_children(
                    population,
                    toolbox,
                    args.steady_children,
                    args.crossover_probability,
                    args.steady_tournament_size,
                )
                (
                    _,
                    _,
                    child_pellet_ratios,
                    child_outcome_labels,
                ) = evaluate_training(
                    children,
                    boards,
                    args.timestep_limit,
                    primitive_set,
                    parsimony=args.parsimony,
                    stagnation_penalty=args.stagnation_penalty,
                    stagnation_grace_steps=args.stagnation_grace_steps,
                    stagnation_termination_steps=args.stagnation_termination_steps,
                    revisit_penalty=args.revisit_penalty,
                    clear_bonus=args.clear_bonus,
                    death_penalty=args.death_penalty,
                    death_pellet_penalty=args.death_pellet_penalty,
                    size_penalty=args.size_penalty,
                    random_tie_break=args.random_tie_break,
                    executor=executor,
                    workers=worker_count,
                    rules=rules,
                )
                selected_outcomes = list(parent_outcome_labels)
                selected_pellets = list(parent_pellet_ratios)
                duel_score = None
                if args.steady_duel_boards > len(boards):
                    # The contest between a child and its three opponents is
                    # decided on more games than everyone else plays: this
                    # generation's boards plus extra ones only those trees see.
                    duel_boards = [
                        generation_board(
                            args.seed, generation, args.boards_per_generation + extra
                        )
                        for extra in range(args.steady_duel_boards - len(boards))
                    ]
                    duel_scores: dict[int, tuple[Individual, float]] = {}

                    def duel_score(tree: Individual) -> float:
                        # Keyed by identity; the tree is kept so the key
                        # cannot be reused while this generation lasts.
                        if id(tree) not in duel_scores:
                            extra_fitness, _, _, _ = evaluate_population_on_boards(
                                [copy.deepcopy(tree)],
                                duel_boards,
                                args.timestep_limit,
                                primitive_set,
                                parsimony=args.parsimony,
                                stagnation_penalty=args.stagnation_penalty,
                                stagnation_grace_steps=args.stagnation_grace_steps,
                                stagnation_termination_steps=args.stagnation_termination_steps,
                                revisit_penalty=args.revisit_penalty,
                                clear_bonus=args.clear_bonus,
                                death_penalty=args.death_penalty,
                                death_pellet_penalty=args.death_pellet_penalty,
                                size_penalty=args.size_penalty,
                                random_tie_break=args.random_tie_break,
                                rules=rules,
                            )
                            total = (
                                tree.fitness.values[0] * len(boards)
                                + extra_fitness[0] * len(duel_boards)
                            )
                            duel_scores[id(tree)] = (
                                tree, total / (len(boards) + len(duel_boards))
                            )
                        return duel_scores[id(tree)][1]

                replacements = steady_state_replace(
                    population,
                    children,
                    args.steady_worst_fraction,
                    args.steady_replacement_pool,
                    duel_score,
                    # Fitness with the size cost put back is the game result
                    # alone: equal on every board means the same play.
                    (
                        (lambda tree: round(
                            tree.fitness.values[0]
                            + size_cost(len(tree), args.size_penalty), 6
                        ))
                        if args.steady_reject_duplicates
                        else None
                    ),
                    args.steady_replace_on_tie,
                )
                for population_index, child_index in replacements:
                    selected_outcomes[population_index] = child_outcome_labels[child_index]
                    selected_pellets[population_index] = child_pellet_ratios[child_index]
                outcomes = count_outcomes(selected_outcomes)
                steady_replacements += len(replacements)
                move_diversity, behaviours = population_diversity(population, primitive_set)
                diversity_rows.append({
                    "generation": generation,
                    "move_diversity": round(move_diversity, 4),
                    "distinct_behaviours": behaviours,
                    "distinct_expressions": len({str(tree) for tree in population}),
                })
                write_diversity(output_directory / "diversity.csv", diversity_rows)

            if raw_champion:
                # Steady state: one game per tree and nothing else. The
                # generation's best is simply the highest raw fitness on this
                # generation's board, and the "tournament" and "validation"
                # columns repeat that one game.
                best_index = max(
                    range(len(population)), key=lambda index: population[index].fitness
                )
                generation_best = copy.deepcopy(population[best_index])
                tournament_fitness = validation_fitness = generation_best.fitness.values[0]
                tournament_pellet_ratio = validation_pellet_ratio = selected_pellets[best_index]
                validation_outcomes = count_outcomes([selected_outcomes[best_index]])
                tournament_rank = generation_best.fitness.wvalues
                is_tournament_best = tournament_rank > best_tournament_rank
                if is_tournament_best:
                    best_tournament_rank = best_validation_rank = tournament_rank
                    best_tournament_fitness = best_validation_fitness = tournament_fitness
                    best_validation_pellet_ratio = tournament_pellet_ratio
            else:
                # Pick the generation champion on the fixed tournament boards, not
                # on this generation's few training boards.
                candidates = champion_candidates(population, args.champion_candidates)
                (
                    tournament_fitnesses,
                    _,
                    tournament_pellet_ratios,
                    _,
                ) = evaluate_population_on_boards(
                    candidates,
                    tournament_boards,
                    args.timestep_limit,
                    primitive_set,
                    parsimony=args.parsimony,
                    stagnation_penalty=args.stagnation_penalty,
                    stagnation_grace_steps=args.stagnation_grace_steps,
                    stagnation_termination_steps=args.stagnation_termination_steps,
                    revisit_penalty=args.revisit_penalty,
                    clear_bonus=args.clear_bonus,
                    death_penalty=args.death_penalty,
                    death_pellet_penalty=args.death_pellet_penalty,
                    size_penalty=args.size_penalty,
                    random_tie_break=args.random_tie_break,
                    executor=executor,
                    workers=worker_count,
                    rules=rules,
                )
                champion_index = max(
                    range(len(candidates)), key=lambda index: candidates[index].fitness
                )
                generation_best = candidates[champion_index]
                tournament_fitness = tournament_fitnesses[champion_index]
                tournament_pellet_ratio = tournament_pellet_ratios[champion_index]
                tournament_rank = generation_best.fitness.wvalues
                is_tournament_best = tournament_rank > best_tournament_rank
                if is_tournament_best:
                    best_tournament_rank = tournament_rank
                    best_tournament_fitness = tournament_fitness
                # Validate a copy so the champion keeps its tournament fitness.
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
                    revisit_penalty=args.revisit_penalty,
                    clear_bonus=args.clear_bonus,
                    death_penalty=args.death_penalty,
                    death_pellet_penalty=args.death_pellet_penalty,
                    size_penalty=args.size_penalty,
                    seed=args.seed,
                    random_tie_break=args.random_tie_break,
                    rules=rules,
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
                tournament_fitness,
                tournament_pellet_ratio,
                best_tournament_fitness,
            )
            history.append(stats)
            write_history(output_directory / "history.csv", history)
            if is_tournament_best:
                hall_of_fame.clear()
                hall_of_fame.update([generation_best])
            if keep_files:
                save_individual(
                    generation_best,
                    output_directory / f"generation_{generation:04d}_best.pkl",
                )
            save_individual(hall_of_fame[0], output_directory / "best_overall.pkl")
            if raw_champion:
                print(
                    f"g={generation:04d} seed={layout_seed} "
                    f"fit={stats.best_fitness:.0f}/{stats.average_fitness:.0f} "
                    f"best-ever={stats.best_tournament_fitness_so_far:.0f} "
                    f"p={stats.tournament_pellet_ratio:.1%} "
                    f"C/D/S/T={outcome_shares(stats)} "
                    f"n={stats.average_node_count:.1f} "
                    f"accepted={steady_replacements} "
                    f"diversity={move_diversity:.1%} behaviours={behaviours}"
                )
            else:
                print(
                    f"g={generation:04d} seed={layout_seed} "
                    f"fit={stats.best_fitness:.0f}/{stats.average_fitness:.0f} "
                    f"val={stats.validation_fitness:.0f}/"
                    f"{stats.best_validation_fitness_so_far:.0f} "
                    f"p={stats.validation_pellet_ratio:.1%}/"
                    f"{stats.best_validation_pellet_ratio_so_far:.1%} "
                    f"10-board={stats.tournament_fitness:.0f}/"
                    f"{stats.best_tournament_fitness_so_far:.0f} "
                    f"C/D/S/T={outcome_shares(stats)} "
                    f"n={stats.average_node_count:.1f}"
                )
            stopped_early = (
                args.target_fitness is not None
                and stats.validation_fitness >= args.target_fitness
            )
            bred = (
                not stopped_early
                and generation < args.generations - 1
                and args.evolution_method == "global"
            )
            if bred:
                parents = toolbox.select(population, args.population_size)
                population = algorithms.varOr(
                    parents,
                    toolbox,
                    lambda_=args.population_size,
                    cxpb=args.crossover_probability,
                    mutpb=args.mutation_probability,
                )
            # Saved after every generation; --resume continues from here.
            save_checkpoint(
                checkpoint_path,
                {
                    "settings": {
                        name: getattr(args, name)
                        for name in (*CHECKPOINT_SETTINGS, *CHECKPOINT_OPTIONAL_SETTINGS)
                    },
                    "next_generation": generation + 1,
                    "population": population,
                    "bred": bred,
                    "random_state": random.getstate(),
                    "history": history,
                    "best": hall_of_fame[0],
                    "stopped_early": stopped_early,
                    "best_validation_fitness": best_validation_fitness,
                    "best_validation_rank": best_validation_rank,
                    "best_validation_pellet_ratio": best_validation_pellet_ratio,
                    "best_tournament_rank": best_tournament_rank,
                    "best_tournament_fitness": best_tournament_fitness,
                    "tournament_seeds": tournament_seeds,
                    "board_seeds": [seed for seed, _ in boards],
                    "board_endgame": endgame_flags,
                    "steady_replacements": steady_replacements,
                    "diversity": diversity_rows,
                },
            )
            if stopped_early:
                break

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
        choices=("global", "cellular", "steady"),
        default="global",
        help=(
            "global over-selection, cellular GP on a toroidal grid, or steady "
            "state (a few children per generation replace weak trees)"
        ),
    )
    parser.add_argument(
        "--steady-children",
        type=int,
        default=1,
        help=(
            "steady state: children bred per generation, all scored on that "
            "generation's boards (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--steady-tournament-size",
        type=int,
        default=3,
        help="steady state: tournament size for choosing each parent (default: %(default)s)",
    )
    parser.add_argument(
        "--steady-worst-fraction",
        type=float,
        default=0.20,
        help=(
            "steady state: the bottom share of the population a child's "
            "opponent is drawn from (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--steady-reject-duplicates",
        action="store_true",
        help=(
            "steady state: drop a child that played this generation's boards "
            "exactly like a tree already in the population, whatever its "
            "expression looks like (default: off)"
        ),
    )
    parser.add_argument(
        "--steady-replace-on-tie",
        action="store_true",
        help=(
            "steady state: a child whose score equals its opponent's replaces "
            "it (default: the child must be strictly better)"
        ),
    )
    parser.add_argument(
        "--steady-duel-boards",
        type=int,
        default=1,
        help=(
            "steady state: games that decide whether a child replaces its "
            "opponent. Above --boards-per-generation, the child and the drawn "
            "trees (and only they) play the extra boards (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--steady-replacement-pool",
        type=int,
        default=3,
        help=(
            "steady state: trees drawn at random from that bottom share; the "
            "child replaces the worst of them if it is fitter (default: %(default)s)"
        ),
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
    parser.add_argument(
        "--boards-replaced-per-generation",
        type=int,
        help=(
            "training boards swapped for fresh layouts each generation; the "
            "rest carry over (default: replace all of them)"
        ),
    )
    parser.add_argument(
        "--champion-candidates",
        type=int,
        default=CHAMPION_CANDIDATES,
        help=(
            "distinct top training trees scored on the 10 tournament boards to "
            "choose each generation's champion (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help=(
            "write generation_NNNN_best.pkl and the layout files for every "
            "N-th generation only (N-1, 2N-1, ...); best_overall.pkl, "
            "history.csv and the checkpoint are still written every "
            "generation (default: %(default)s)"
        ),
    )
    parser.add_argument("--timestep-limit", type=int, default=1000)
    parser.add_argument("--stagnation-penalty", type=float, default=100.0)
    parser.add_argument("--stagnation-grace-steps", type=int, default=0)
    parser.add_argument("--stagnation-termination-steps", type=int, default=50)
    parser.add_argument(
        "--stagnation-scaling",
        choices=("fixed", "inverse"),
        default="fixed",
        help=(
            "inverse: divide --stagnation-termination-steps by the fraction of "
            "the board's pellets still uneaten, so the cutoff gets longer as "
            "the board empties (default: fixed)"
        ),
    )
    parser.add_argument(
        "--late-pellet-share",
        type=float,
        help=(
            "share of the pellet score carried by the last 20%% of the "
            "pellets; 0.5 makes them worth as much as the first 80%% "
            "(default: linear, which equals 0.2)"
        ),
    )
    parser.add_argument(
        "--endgame-boards",
        type=int,
        default=0,
        help=(
            "how many of the --boards-per-generation training boards start "
            "as endgames; the rest are full games (default: 0)"
        ),
    )
    parser.add_argument(
        "--endgame-styles",
        default="sweep",
        help=(
            "comma-separated ways to build endgame boards, used in turn: "
            "sweep (the leftovers of eating the nearest pellet) and islands "
            "(separate patches near corners) (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--endgame-start",
        type=float,
        default=ENDGAME_START,
        help=(
            "fraction of the pellets already eaten when an endgame board "
            "starts (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--revisit-penalty",
        type=float,
        default=REVISIT_PENALTY,
        help=(
            "extra cost when a non-scoring move returns to a tile visited in "
            "the last VISIT_WINDOW steps (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--clear-bonus",
        type=float,
        default=CLEAR_BONUS,
        help=(
            "extra fitness for eating every pellet, as a fraction of the "
            "full-clear score (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=DEATH_PENALTY,
        help="flat cost of dying or stagnating (default: %(default)s)",
    )
    parser.add_argument(
        "--death-pellet-penalty",
        type=float,
        default=DEATH_PELLET_PENALTY,
        help=(
            "extra cost of dying or stagnating, as a fraction of the value of "
            "the uneaten pellets (default: %(default)s)"
        ),
    )
    parser.add_argument("--min-initial-depth", type=int, default=2)
    parser.add_argument("--max-initial-depth", type=int, default=5)
    parser.add_argument(
        "--full-min-initial-depth",
        type=int,
        help=(
            "smallest depth of the 'full' half of the starting population "
            "(default: --min-initial-depth)"
        ),
    )
    parser.add_argument(
        "--grow-terminal-probability",
        type=float,
        help=(
            "chance that a node in the 'grow' half of the starting population "
            "is a terminal; 0.4 gives a 60/40 function/terminal split "
            "(default: the terminals' share of the primitive set, about 0.71)"
        ),
    )
    parser.add_argument("--max-evolved-depth", type=int, default=10)
    parser.add_argument("--crossover-probability", type=float, default=0.97)
    parser.add_argument("--mutation-probability", type=float, default=0.03)
    parser.add_argument("--top-fraction", type=float, default=0.32)
    parser.add_argument("--top-probability", type=float, default=0.8)
    parser.add_argument(
        "--size-penalty",
        type=float,
        default=SIZE_PENALTY,
        help=(
            "fitness subtracted per node for a 400-node tree; the per-node rate "
            "scales with tree size, so small trees pay almost nothing "
            "(default: %(default)s)"
        ),
    )
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
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "collapse trivial chains such as abs(abs(x)) or mul(x, 1) in new "
            "trees; this never changes a tree's behaviour (default: off)"
        ),
    )
    parser.add_argument(
        "--mutation-min-depth",
        type=int,
        default=MUTATION_MIN_DEPTH,
        help=(
            "smallest depth of the subtree a mutation inserts; 0 allows "
            "single-node inserts (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-board-cache",
        dest="board_cache",
        action="store_false",
        help=(
            "cellular: replay every board every generation instead of keeping "
            "each surviving tree's results (same results, slower; for checking)"
        ),
    )
    parser.add_argument(
        "--add-features",
        default="",
        help=(
            "comma-separated optional features to add to the default terminals "
            f"({', '.join(OPTIONAL_FEATURE_NAMES)}); 'all' stands for the three "
            "evasion features, e.g. 'all,pellets_left' (default: none)"
        ),
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help=(
            "comma-separated feature names to leave out of the primitive set, "
            "for ablation runs (default: none)"
        ),
    )
    parser.add_argument("--target-fitness", type=float)
    parser.add_argument(
        "--tournament-seed",
        type=int,
        help=(
            "seed of the 10 tournament boards; set the same value on several "
            "runs to score their champions on the same boards (default: "
            "derived from --seed and --generations)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue from checkpoint.pkl in --output-directory if it exists "
            "(saved after every generation); otherwise start a new run"
        ),
    )
    parser.add_argument("--random-tie-break", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-scale", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def endgame_styles(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(name.strip() for name in args.endgame_styles.split(",") if name.strip())


def added_features(args: argparse.Namespace) -> tuple[str, ...]:
    names = tuple(name.strip() for name in args.add_features.split(",") if name.strip())
    expanded = []
    for name in names:
        expanded.extend(EVASION_FEATURE_NAMES if name == "all" else (name,))
    return tuple(expanded)


def training_features(args: argparse.Namespace) -> tuple[str, ...]:
    """The run's terminals: defaults plus --add-features, minus --exclude-features."""
    excluded = set(excluded_features(args))
    added = added_features(args)
    return tuple(
        name for name in MOVE_FEATURE_NAMES
        if (name in FEATURE_NAMES or name in added) and name not in excluded
    )


def excluded_features(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(name.strip() for name in args.exclude_features.split(",") if name.strip())


def validate_args(args: argparse.Namespace) -> None:
    if args.parsimony < 0:
        raise ValueError("parsimony must be nonnegative")
    if args.revisit_penalty < 0:
        raise ValueError("revisit-penalty must be nonnegative")
    if args.clear_bonus < 0:
        raise ValueError("clear-bonus must be nonnegative")
    if args.death_penalty < 0 or args.death_pellet_penalty < 0:
        raise ValueError("death penalties must be nonnegative")
    if args.mutation_min_depth < 0:
        raise ValueError("mutation-min-depth must be nonnegative")
    unknown_added = set(added_features(args)) - set(OPTIONAL_FEATURE_NAMES)
    if unknown_added:
        raise ValueError(f"unknown --add-features: {', '.join(sorted(unknown_added))}")
    unknown = set(excluded_features(args)) - set(MOVE_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature(s) to exclude: {', '.join(sorted(unknown))}")
    if args.size_penalty < 0:
        raise ValueError("size-penalty must be nonnegative")
    if args.late_pellet_share is not None and not 0 < args.late_pellet_share < 1:
        raise ValueError("late-pellet-share must be between 0 and 1")
    if not 0 <= args.endgame_boards <= args.boards_per_generation:
        raise ValueError("endgame-boards must be between 0 and boards-per-generation")
    if not 0 < args.endgame_start < 1:
        raise ValueError("endgame-start must be between 0 and 1")
    if not endgame_styles(args) or set(endgame_styles(args)) - set(ENDGAME_STYLES):
        raise ValueError(f"endgame-styles must be chosen from {', '.join(ENDGAME_STYLES)}")
    if args.champion_candidates < 1:
        raise ValueError("champion-candidates must be positive")
    replaced = args.boards_replaced_per_generation
    if replaced is not None and not 1 <= replaced <= args.boards_per_generation:
        raise ValueError(
            "boards-replaced-per-generation must be between 1 and "
            "boards-per-generation"
        )
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.population_size < 1 or args.generations < 1:
        raise ValueError("population size and generations must be positive")
    if not 1 <= args.boards_per_generation < GENERATION_SEED_STRIDE:
        raise ValueError("boards-per-generation must be between 1 and 1000002")
    if args.cellular_tournament_size < 1:
        raise ValueError("cellular-tournament-size must be positive")
    if min(
        args.steady_children, args.steady_tournament_size,
        args.steady_replacement_pool, args.steady_duel_boards,
    ) < 1:
        raise ValueError("steady-state counts must be positive")
    if not 0 < args.steady_worst_fraction <= 1:
        raise ValueError("steady-worst-fraction must be above 0 and at most 1")
    if args.evolution_method == "steady" and args.random_tie_break:
        raise ValueError("steady state remembers game results, so it cannot use random-tie-break")
    if args.save_interval < 1:
        raise ValueError("save-interval must be positive")
    if args.timestep_limit < 1:
        raise ValueError("timestep-limit must be positive")
    if args.video_fps <= 0:
        raise ValueError("video-fps must be positive")
    if args.min_initial_depth < 1 or args.max_initial_depth < args.min_initial_depth:
        raise ValueError("invalid initial depth range")
    if args.full_min_initial_depth is not None and not (
        args.min_initial_depth <= args.full_min_initial_depth <= args.max_initial_depth
    ):
        raise ValueError("full-min-initial-depth must lie within the initial depth range")
    if args.grow_terminal_probability is not None and not 0 < args.grow_terminal_probability < 1:
        raise ValueError("grow-terminal-probability must be between 0 and 1")
    if args.max_evolved_depth < args.max_initial_depth:
        raise ValueError("max evolved depth cannot be below initial depth")
    if not math.isclose(
        args.crossover_probability + args.mutation_probability, 1.0
    ):
        raise ValueError("crossover and mutation probabilities must total 1")


def artifact_python() -> str:
    """Interpreter for the plots and videos written after training.

    Under PyPy, which usually lacks matplotlib and Pillow, that is the normal
    environment's Python; otherwise the one running now.
    """
    import platform

    regular = PROJECT_DIR / "GeneticAlgo" / ".venv" / "bin" / "python"
    if platform.python_implementation() == "PyPy" and regular.exists():
        return str(regular)
    return sys.executable


def generate_artifacts(args: argparse.Namespace, output: Path) -> bool:
    history, best = output / "history.csv", output / "best_overall.pkl"
    if not history.exists() or not best.exists():
        print("No generation completed; there is no evaluated individual to render.", file=sys.stderr)
        return False
    subprocess.run([artifact_python(), str(PROJECT_DIR / "plot_board_graph.py"), str(history), str(output), str(best)], cwd=PROJECT_DIR, check=True)
    if not args.no_video:
        # The final video always plays the original arcade maze.
        command = [artifact_python(), "-m", "GeneticAlgo.watch_board_graph", str(best), "--board", str(VALIDATION_BOARD_PATH), "--video", str(output / "best_overall.mp4"), "--fps", str(args.video_fps), "--scale", str(args.video_scale), "--max-steps", str(args.timestep_limit), "--seed", str(args.seed)]
        if args.random_tie_break:
            command.append("--random-tie-break")
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    return True


def main() -> int:
    # A shell starts background jobs with SIGINT ignored, and Python then never
    # installs its handler. Restore it so the run scripts can interrupt a
    # detached run and still get the final plots and video.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_directory or Path("runs") / f"board_graph_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    feature_names = training_features(args)
    primitive_set = build_primitive_set(feature_names)
    print(f"Run artifacts: {output}")
    print(f"Terminals: {', '.join(feature_names)}")
    if args.full_min_initial_depth is not None or args.grow_terminal_probability is not None:
        print(
            "Starting trees: full half from depth "
            f"{args.full_min_initial_depth or args.min_initial_depth}, grow half "
            f"with terminal probability {args.grow_terminal_probability or 'default'}"
        )
    print(
        f"Tree repair: {'on' if args.repair else 'off'}; "
        f"mutation inserts from depth {args.mutation_min_depth}"
    )
    print(
        f"Training boards per generation: {args.boards_per_generation} "
        f"({args.boards_replaced_per_generation or args.boards_per_generation} "
        "replaced each generation)"
    )
    if args.endgame_boards:
        print(
            f"Of those, {args.endgame_boards} start as endgames with "
            f"{args.endgame_start:.0%} of the pellets eaten "
            f"(styles: {', '.join(endgame_styles(args))})"
        )
    if args.stagnation_scaling != "fixed" or args.late_pellet_share is not None:
        print(
            f"Stagnation cutoff: {args.stagnation_scaling}; late pellet share: "
            f"{'linear' if args.late_pellet_share is None else args.late_pellet_share}"
        )
    print(f"Champion candidates per generation: {args.champion_candidates}")
    if args.evolution_method == "cellular":
        rows, columns = cellular_grid_shape(args.population_size)
        print(
            f"Evolution: cellular {rows}x{columns} toroidal grid, "
            f"Moore neighborhood, tournament={args.cellular_tournament_size}"
        )
    elif args.evolution_method == "steady":
        print(
            f"Evolution: steady state, {args.steady_children} child(ren) per "
            f"generation, parent tournament={args.steady_tournament_size}; each "
            f"child must beat the worst of {args.steady_replacement_pool} drawn "
            f"from the bottom {args.steady_worst_fraction:.0%}, over "
            f"{max(args.steady_duel_boards, args.boards_per_generation)} game(s)"
            + ("; duplicates of existing play are dropped" if args.steady_reject_duplicates else "")
            + ("; a tie goes to the child" if args.steady_replace_on_tie else "")
        )
    else:
        print("Evolution: global over-selection")
    try:
        result = evolve(primitive_set, args, output)
    except KeyboardInterrupt:
        print("\nTraining interrupted; generating artifacts from the latest completed generation.")
        generate_artifacts(args, output)
        return 130
    summary = (
        f"Finished after {result.generations_completed} generations. "
        f"Best fitness: {result.best.fitness.values[0]}\n"
        f"Best tree: {result.best}\n"
    )
    summary_path = output / "summary.txt"
    # A resumed run that had already finished keeps its existing artifacts.
    if not (summary_path.exists() and summary_path.read_text() == summary):
        generate_artifacts(args, output)
        summary_path.write_text(summary)
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
