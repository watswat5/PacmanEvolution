"""Python side of the C++ game engine.

``evaluate_population_lockstep`` has the same signature and results as
``GeneticAlgo.train_board_graph.evaluate_population_lockstep`` but plays the
games in C++ on several threads. ``install`` swaps it into the trainer module,
so the unchanged trainers (and everything that calls them) use it.

Only the training games move to C++. Watching, videos, the versus mode and
the diversity measurement still run the Python game, so rendering is
unchanged. The two games follow the same rules and agree closely, not bit for
bit (see FastCore/README.md).
"""
from __future__ import annotations

import os
from typing import Sequence

from deap import gp

import GeneticAlgo.train_board_graph as trainer
from PacmanGame import MOVE_FEATURE_NAMES
from PacmanGame.board_graph import _layout_key
from PacmanGame import pacman as python_game

try:
    from . import _pacman_core as core
except ImportError as error:  # not built yet, or built for another Python
    raise ImportError(
        "FastCore's C++ engine is not built for this Python; run FastCore/build.sh "
        f"(see FastCore/README.md). Original error: {error}"
    ) from error

OPCODES = {"add": 0, "sub": 1, "mul": 2, "div": 3, "min": 4, "max": 5, "abs": 6,
           "neg": 7, "sqrt": 8, "if_lte": 9}
FEATURE_OPCODE, CONSTANT_OPCODE = 10, 11
FEATURE_INDEX = {name: index for index, name in enumerate(MOVE_FEATURE_NAMES)}
LABELS = ("cleared", "died", "stagnated", "timed_out")
GHOST_KEYS = ("blinky_pos", "pinky_pos", "inky_pos", "clyde_pos")
ACTIONS = ("up", "down", "left", "right")

THREADS = os.cpu_count() or 1


def encode(tree, primitive_set: gp.PrimitiveSet) -> tuple[list[int], list[float]]:
    """A DEAP tree as the engine's prefix opcodes and values."""
    ops, values = [], []
    for node in tree:
        if isinstance(node, gp.Primitive):
            if node.name not in OPCODES:
                raise ValueError(f"the C++ engine has no primitive {node.name!r}")
            ops.append(OPCODES[node.name])
            values.append(0.0)
        elif node.value in FEATURE_INDEX:  # an argument: its value is the feature name
            if FEATURE_INDEX[node.value] >= core.FEATURES:
                raise ImportError(
                    f"this FastCore build predates the feature {node.value!r}; "
                    "rebuild it with FastCore/build.sh"
                )
            ops.append(FEATURE_OPCODE)
            values.append(float(FEATURE_INDEX[node.value]))
        else:
            # Named constants are symbolic: the value lives in the set's context.
            value = primitive_set.context.get(node.value, node.value) if isinstance(node.value, str) else node.value
            ops.append(CONSTANT_OPCODE)
            values.append(float(value))
    return ops, values


# Encodings by tree content. DEAP trees share their node objects, so the ids of
# a tree's nodes identify its content; each tree is encoded once, not once per
# board. Cleared when it grows large.
_ENCODINGS: dict[tuple, tuple[list[int], list[float]]] = {}


# Inside evaluation_scope() (one batch of evaluations, during which no tree
# changes), encodings are also remembered by tree object, so a tree playing 16
# boards is looked up by content once rather than 16 times.
_SCOPE: dict[int, tuple] | None = None


class evaluation_scope:
    """trainer.EVALUATION_SCOPE for the C++ engine (nests safely)."""

    def __enter__(self):
        global _SCOPE
        self._outer = _SCOPE
        if _SCOPE is None:
            _SCOPE = {}
        return self

    def __exit__(self, *exc):
        global _SCOPE
        _SCOPE = self._outer
        return False


def encoded(tree, primitive_set: gp.PrimitiveSet) -> tuple[list[int], list[float]]:
    if _SCOPE is not None:
        hit = _SCOPE.get(id(tree))
        if hit is not None and hit[0] is tree:
            return hit[1]
        result = _encoded_by_content(tree, primitive_set)
        _SCOPE[id(tree)] = (tree, result)
        return result
    return _encoded_by_content(tree, primitive_set)


def _encoded_by_content(tree, primitive_set: gp.PrimitiveSet) -> tuple[list[int], list[float]]:
    key = tuple(map(id, tree))
    result = _ENCODINGS.get(key)
    if result is None:
        if len(_ENCODINGS) > 50_000:
            _ENCODINGS.clear()
        result = _ENCODINGS[key] = encode(tree, primitive_set)
    return result


def start_arguments(board) -> tuple:
    width = board.width
    cell = lambda point: point[1] * width + point[0]
    direction = next(index for index, action in enumerate(ACTIONS) if getattr(board, f"player_{action}"))
    scared = board.ghost_scared
    return (
        [value for row in board.tiles for value in row], width, board.height,
        cell(board.player_pos), direction,
        [cell(getattr(board, key)) for key in GHOST_KEYS],
        [bool(scared[key.removesuffix("_pos")]) for key in GHOST_KEYS],
        board.player_powerup_turns, board.pellets_eaten, board.pellets_remaining, board.score,
        _layout_key(board) == python_game._original_layout(),
        python_game.POWERUP_DURATION,
    )


def evaluate_population_lockstep(
    population: Sequence,
    starting_board,
    timestep_limit: int,
    primitive_set: gp.PrimitiveSet,
    *,
    parsimony: float = 0.0,
    stagnation_penalty: float = trainer.NON_SCORING_MOVE_PENALTY,
    stagnation_grace_steps: int = 0,
    stagnation_termination_steps: int = 50,
    revisit_penalty: float = trainer.REVISIT_PENALTY,
    clear_bonus: float = trainer.CLEAR_BONUS,
    death_penalty: float = trainer.DEATH_PENALTY,
    death_pellet_penalty: float = trainer.DEATH_PELLET_PENALTY,
    size_penalty: float = trainer.SIZE_PENALTY,
    seed: int = 0,
    index_offset: int = 0,
    random_tie_break: bool = False,
    rules=trainer.DEFAULT_RULES,
):
    """C++ replacement for the trainer's evaluate_population_lockstep."""
    if random_tie_break:
        return _python_lockstep(
            population, starting_board, timestep_limit, primitive_set, parsimony=parsimony,
            stagnation_penalty=stagnation_penalty, stagnation_grace_steps=stagnation_grace_steps,
            stagnation_termination_steps=stagnation_termination_steps, revisit_penalty=revisit_penalty,
            clear_bonus=clear_bonus, death_penalty=death_penalty, death_pellet_penalty=death_pellet_penalty,
            size_penalty=size_penalty, seed=seed, index_offset=index_offset, random_tie_break=True, rules=rules,
        )
    settings = {
        "timestep_limit": int(timestep_limit),
        "stagnation_penalty": float(stagnation_penalty),
        "revisit_penalty": float(revisit_penalty),
        "stagnation_grace_steps": int(stagnation_grace_steps),
        "stagnation_termination_steps": int(stagnation_termination_steps),
        "scaled_stagnation": rules.stagnation_scaling == "inverse",
        "late_pellet_share": -1.0 if rules.late_pellet_share is None else float(rules.late_pellet_share),
        "clear_bonus": float(clear_bonus),
        "death_penalty": float(death_penalty),
        "death_pellet_penalty": float(death_pellet_penalty),
        "size_penalty": float(size_penalty),
    }
    fitnesses, ratios, labels, _, _ = core.evaluate_population(
        [encoded(tree, primitive_set) for tree in population], *start_arguments(starting_board), settings, THREADS
    )
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = (fitness, -parsimony * len(individual))
    names = [LABELS[label] for label in labels]
    outcomes = trainer.OutcomeCounts(
        cleared=names.count("cleared"), died=names.count("died"),
        stagnated=names.count("stagnated"), timed_out=names.count("timed_out"),
    )
    return list(fitnesses), outcomes, list(ratios), names


_python_scope = trainer.EVALUATION_SCOPE
_python_lockstep = trainer.evaluate_population_lockstep
_python_on_boards = trainer.evaluate_population_on_boards


def _on_boards_in_process(*args, executor=None, workers: int = 1, **kwargs):
    # The C++ engine has its own threads, so worker processes are not used.
    return _python_on_boards(*args, executor=None, workers=1, **kwargs)


def install(threads: int | None = None) -> None:
    """Make the trainer module play its training games in C++."""
    global THREADS
    if threads:
        THREADS = threads
    trainer.evaluate_population_lockstep = evaluate_population_lockstep
    trainer.evaluate_population_on_boards = _on_boards_in_process
    trainer.EVALUATION_SCOPE = evaluation_scope
    trainer.EVALUATION_NOTE = f"Evaluation: C++ engine on {THREADS} threads (one process; no worker processes needed)"


def uninstall() -> None:
    trainer.evaluate_population_lockstep = _python_lockstep
    trainer.evaluate_population_on_boards = _python_on_boards
    trainer.EVALUATION_NOTE = None
    trainer.EVALUATION_SCOPE = _python_scope
