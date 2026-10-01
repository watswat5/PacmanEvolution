"""Train four-tree Pac-Man policies using DEAP genetic programming."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from deap import algorithms, tools

PROJECT_DIR = Path(__file__).resolve().parent.parent
AUTOENCODER_DIR = PROJECT_DIR / "Autoencoder"
DEFAULT_INITIAL_BOARD = PROJECT_DIR / "PacmanGame" / "test_board.txt"
SUPPORTED_LATENT_DIMS = (4, 8, 16, 32)

try:
    from .gp_core import (
        ACTIONS,
        Individual,
        action_tree,
        build_primitive_set,
        build_toolbox,
        evaluate_individual_move,
        fitness_value,
        individual_depth,
        individual_node_count,
        save_individual,
        set_fitness,
        tree_depth,
        tree_size,
    )
except ImportError:  # Direct import with GeneticAlgo on sys.path.
    from gp_core import (
        ACTIONS,
        Individual,
        action_tree,
        build_primitive_set,
        build_toolbox,
        evaluate_individual_move,
        fitness_value,
        individual_depth,
        individual_node_count,
        save_individual,
        set_fitness,
        tree_depth,
        tree_size,
    )

try:
    from PacmanGame import Board, PacmanGame, precompute_board_graph, read_board
except ImportError as exc:
    raise ImportError("run GP training from the project root so PacmanGame is importable") from exc


@dataclass(frozen=True)
class EncoderMetadata:
    """Mapping between a Board and the encoder's collectible input vector."""

    positions: np.ndarray
    values: np.ndarray
    board_shape: tuple[int, int]

    @classmethod
    def load(cls, path: str | Path) -> "EncoderMetadata":
        with np.load(path) as data:
            shape = tuple(int(value) for value in data["board_shape"])
            return cls(
                positions=data["collectible_positions"].copy(),
                values=data["collectible_values"].copy(),
                board_shape=shape,
            )


@dataclass(frozen=True)
class EncoderBundle:
    """A dimension-checked encoder and its board-input metadata."""

    latent_dim: int
    encoder: Any
    metadata: EncoderMetadata
    directory: Path

    @classmethod
    def load(
        cls,
        latent_dim: int,
        models_directory: str | Path = AUTOENCODER_DIR,
    ) -> "EncoderBundle":
        if latent_dim not in SUPPORTED_LATENT_DIMS:
            raise ValueError(
                f"latent_dim must be one of {SUPPORTED_LATENT_DIMS}, got {latent_dim}"
            )
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load an encoder") from exc

        root = Path(models_directory)
        candidates = [
            root / f"training_{latent_dim}",
            root / f"training_{latent_dim}d",  # Earlier naming convention.
        ]
        if latent_dim == 16:
            candidates.append(root / "training")  # Existing 16-D model.
        directory = next(
            (candidate for candidate in candidates if (candidate / "encoder.keras").is_file()),
            None,
        )
        if directory is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"no {latent_dim}-D encoder found; searched: {searched}"
            )

        encoder = tf.keras.models.load_model(directory / "encoder.keras")
        output_dim = int(encoder.output_shape[-1])
        if output_dim != latent_dim:
            raise ValueError(
                f"{directory / 'encoder.keras'} outputs {output_dim} values, "
                f"expected {latent_dim}"
            )
        metadata = EncoderMetadata.load(directory / "board_metadata.npz")
        return cls(latent_dim, encoder, metadata, directory)


@dataclass(frozen=True)
class EvolutionResult:
    """Final state returned by a complete GP evolution run."""

    best: Individual
    population: list[Individual]
    generations_completed: int
    stopped_early: bool
    history: tuple["GenerationStats", ...]


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    average_tree_depth: float
    best_tree_depth: int
    average_node_count: float
    best_node_count: int
    average_up_tree_depth: float
    best_up_tree_depth: int
    average_up_node_count: float
    best_up_node_count: int
    average_down_tree_depth: float
    best_down_tree_depth: int
    average_down_node_count: float
    best_down_node_count: int
    average_left_tree_depth: float
    best_left_tree_depth: int
    average_left_node_count: float
    best_left_node_count: int
    average_right_tree_depth: float
    best_right_tree_depth: int
    average_right_node_count: float
    best_right_node_count: int


def population_statistics(
    population: Sequence[Individual], generation: int
) -> GenerationStats:
    if not population:
        raise ValueError("population cannot be empty")
    if any(not individual.fitness.valid for individual in population):
        raise ValueError("every individual must have fitness before logging")
    best = max(population, key=lambda individual: individual.fitness)
    fitnesses = np.asarray(
        [fitness_value(individual) for individual in population], dtype=np.float64
    )
    depths = np.asarray(
        [individual_depth(individual) for individual in population], dtype=np.float64
    )
    sizes = np.asarray(
        [individual_node_count(individual) for individual in population], dtype=np.float64
    )
    per_action_depths = {
        action: np.asarray(
            [tree_depth(action_tree(individual, action)) for individual in population],
            dtype=np.float64,
        )
        for action in ACTIONS
    }
    per_action_sizes = {
        action: np.asarray(
            [tree_size(action_tree(individual, action)) for individual in population],
            dtype=np.float64,
        )
        for action in ACTIONS
    }
    return GenerationStats(
        generation=generation,
        best_fitness=float(fitnesses.max()),
        average_fitness=float(fitnesses.mean()),
        worst_fitness=float(fitnesses.min()),
        average_tree_depth=float(depths.mean()),
        best_tree_depth=individual_depth(best),
        average_node_count=float(sizes.mean()),
        best_node_count=individual_node_count(best),
        average_up_tree_depth=float(per_action_depths["up"].mean()),
        best_up_tree_depth=tree_depth(action_tree(best, "up")),
        average_up_node_count=float(per_action_sizes["up"].mean()),
        best_up_node_count=tree_size(action_tree(best, "up")),
        average_down_tree_depth=float(per_action_depths["down"].mean()),
        best_down_tree_depth=tree_depth(action_tree(best, "down")),
        average_down_node_count=float(per_action_sizes["down"].mean()),
        best_down_node_count=tree_size(action_tree(best, "down")),
        average_left_tree_depth=float(per_action_depths["left"].mean()),
        best_left_tree_depth=tree_depth(action_tree(best, "left")),
        average_left_node_count=float(per_action_sizes["left"].mean()),
        best_left_node_count=tree_size(action_tree(best, "left")),
        average_right_tree_depth=float(per_action_depths["right"].mean()),
        best_right_tree_depth=tree_depth(action_tree(best, "right")),
        average_right_node_count=float(per_action_sizes["right"].mean()),
        best_right_node_count=tree_size(action_tree(best, "right")),
    )


def write_generation_history(
    path: str | Path, history: Sequence[GenerationStats]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GenerationStats.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for statistics in history:
            writer.writerow(
                {field: getattr(statistics, field) for field in fieldnames}
            )


def boards_to_encoder_batch(
    boards: Sequence[Board], metadata: EncoderMetadata
) -> np.ndarray:
    """Convert many game boards into one float32 encoder input batch."""
    if not boards:
        return np.empty((0, len(metadata.positions)), dtype=np.float32)
    batch = np.empty((len(boards), len(metadata.positions)), dtype=np.float32)
    for board_index, board in enumerate(boards):
        if board.shape != metadata.board_shape:
            raise ValueError(
                f"board shape {board.shape} does not match encoder shape "
                f"{metadata.board_shape}"
            )
        for input_index, ((row, column), value) in enumerate(
            zip(metadata.positions, metadata.values)
        ):
            batch[board_index, input_index] = float(
                board.tiles[int(row)][int(column)] & int(value) == int(value)
            )
    return batch


def encode_boards_batch(
    boards: Sequence[Board], encoder: Any, metadata: EncoderMetadata
) -> np.ndarray:
    """Encode a whole population in one TensorFlow model invocation."""
    inputs = boards_to_encoder_batch(boards, metadata)
    if len(inputs) == 0:
        output_shape = getattr(encoder, "output_shape", (None, 0))
        return np.empty((0, int(output_shape[-1])), dtype=np.float32)
    encoded = encoder(inputs, training=False)
    return np.asarray(encoded.numpy() if hasattr(encoded, "numpy") else encoded)


def encode_board(board: Board, encoder: Any, metadata: EncoderMetadata) -> np.ndarray:
    """Encode one board through the same batch path used during evolution."""
    return encode_boards_batch([board], encoder, metadata)[0]


def terminal_values(game: PacmanGame, latent: np.ndarray) -> dict[str, float]:
    """Build normalized GP terminals from one game and its latent board vector."""
    board = game.board
    x_scale = max(1, board.width - 1)
    y_scale = max(1, board.height - 1)
    values = {f"z{index}": float(value) for index, value in enumerate(latent)}
    values.update(
        {
            "player_x": board.player_pos[0] / x_scale,
            "player_y": board.player_pos[1] / y_scale,
            "powerup_timer": board.player_powerup_turns / 40.0,
            "player_up": float(board.player_up),
            "player_down": float(board.player_down),
            "player_left": float(board.player_left),
            "player_right": float(board.player_right),
            "inky_x": board.inky_pos[0] / x_scale,
            "inky_y": board.inky_pos[1] / y_scale,
            "blinky_x": board.blinky_pos[0] / x_scale,
            "blinky_y": board.blinky_pos[1] / y_scale,
            "pinky_x": board.pinky_pos[0] / x_scale,
            "pinky_y": board.pinky_pos[1] / y_scale,
            "clyde_x": board.clyde_pos[0] / x_scale,
            "clyde_y": board.clyde_pos[1] / y_scale,
        }
    )
    values.update(
        {action: float(legal) for action, legal in game.legal_moves().items()}
    )
    return values


DEATH_PENALTY = 10_000.0
NON_SCORING_MOVE_PENALTY = 100.0


def calculate_fitness(
    score_delta: int,
    died: bool,
    individual: Individual,
    parsimony: float = 0.0,
) -> float:
    return (
        float(score_delta)
        - DEATH_PENALTY * died
        - parsimony * individual_node_count(individual)
    )


_WORKER_POPULATION: Sequence[Individual] = ()
_WORKER_FUNCTION_SET = None
_EPISODE_ENCODER_BUNDLE: EncoderBundle | None = None


def _initialize_evaluation_worker(population, function_set) -> None:
    """Install read-only generation data once in each evaluation process."""
    global _WORKER_POPULATION, _WORKER_FUNCTION_SET
    _WORKER_POPULATION = population
    _WORKER_FUNCTION_SET = function_set


def _evaluate_move_chunk(tasks) -> list[tuple[int, str | None]]:
    """Evaluate a chunk of individuals outside the main Python GIL."""
    results = []
    for index, values, legal_moves, random_seed, random_tie_break in tasks:
        move, _ = evaluate_individual_move(
            _WORKER_POPULATION[index],
            values,
            legal_moves,
            _WORKER_FUNCTION_SET,
            random.Random(random_seed),
            random_tie_break=random_tie_break,
        )
        results.append((index, move))
    return results


def _initialize_episode_worker(latent_dim: int, models_directory: str) -> None:
    """Load one CPU encoder per persistent episode worker."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    global _EPISODE_ENCODER_BUNDLE
    _EPISODE_ENCODER_BUNDLE = EncoderBundle.load(latent_dim, models_directory)


def _evaluate_episode_chunk(task) -> list[tuple[int, float]]:
    """Run complete rollouts for one population chunk in a worker process."""
    (
        indices, individuals, board, timestep_limit, function_set, parsimony,
        stagnation_penalty, stagnation_grace_steps,
        stagnation_termination_steps, seed,
    ) = task
    assert _EPISODE_ENCODER_BUNDLE is not None
    fitnesses = evaluate_population_lockstep(
        individuals,
        board,
        timestep_limit,
        _EPISODE_ENCODER_BUNDLE.encoder,
        _EPISODE_ENCODER_BUNDLE.metadata,
        function_set=function_set,
        parsimony=parsimony,
        stagnation_penalty=stagnation_penalty,
        stagnation_grace_steps=stagnation_grace_steps,
        stagnation_termination_steps=stagnation_termination_steps,
        seed=seed,
        worker_count=1,
    )
    return list(zip(indices, fitnesses))


def _evaluate_population_by_episode(
    executor: ProcessPoolExecutor,
    population: Sequence[Individual],
    starting_board: Board,
    timestep_limit: int,
    function_set,
    parsimony: float,
    stagnation_penalty: float,
    stagnation_grace_steps: int,
    stagnation_termination_steps: int,
    seed: int,
    worker_count: int,
) -> list[float]:
    """Distribute self-contained game batches across persistent processes."""
    indexed = list(enumerate(population))
    chunk_count = min(worker_count, len(indexed))
    chunks = [indexed[offset::chunk_count] for offset in range(chunk_count)]
    tasks = [
        (
            [index for index, _ in chunk],
            [individual for _, individual in chunk],
            starting_board,
            timestep_limit,
            function_set,
            parsimony,
            stagnation_penalty,
            stagnation_grace_steps,
            stagnation_termination_steps,
            seed + offset * 1_000_003,
        )
        for offset, chunk in enumerate(chunks)
    ]
    fitnesses = [0.0] * len(population)
    for results in executor.map(_evaluate_episode_chunk, tasks):
        for index, fitness in results:
            fitnesses[index] = fitness
    return fitnesses


def evaluate_population_lockstep(
    population: Sequence[Individual],
    starting_board: Board,
    timestep_limit: int,
    encoder: Any,
    metadata: EncoderMetadata,
    *,
    function_set=None,
    parsimony: float = 0.0,
    stagnation_penalty: float = NON_SCORING_MOVE_PENALTY,
    stagnation_grace_steps: int = 0,
    stagnation_termination_steps: int = 50,
    seed: int = 0,
    random_tie_break: bool = False,
    worker_count: int = 1,
) -> list[float]:
    """Evaluate a population one shared timestep and encoder batch at a time.

    Actions for timestep ``t`` use latents encoded after timestep ``t - 1``.
    All active games step before a single batch encodes their resulting boards.
    """
    if timestep_limit < 1:
        raise ValueError("timestep_limit must be positive")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if stagnation_penalty < 0:
        raise ValueError("stagnation_penalty cannot be negative")
    if stagnation_grace_steps < 0:
        raise ValueError("stagnation_grace_steps cannot be negative")
    if stagnation_termination_steps < 1:
        raise ValueError("stagnation_termination_steps must be positive")
    if not population:
        return []
    if function_set is None:
        raise ValueError("a DEAP primitive set is required")
    board_graph = precompute_board_graph(starting_board)
    games = [
        PacmanGame(starting_board.copy(), board_graph=board_graph)
        for _ in population
    ]
    starting_scores = [game.board.score for game in games]
    steps_since_score = [0] * len(games)
    steps_taken = [0] * len(games)
    stagnation_costs = [0.0] * len(games)
    terminated_for_stagnation = [False] * len(games)
    rngs = [random.Random(seed + index) for index in range(len(population))]

    active = list(range(len(population)))
    initial_latents = encode_boards_batch(
        [games[index].board for index in active], encoder, metadata
    )
    latents = {
        population_index: initial_latents[batch_index]
        for batch_index, population_index in enumerate(active)
    }

    executor = None
    if worker_count > 1:
        # Spawn avoids forking TensorFlow/CUDA state into workers. Workers only
        # evaluate GP trees; encoding remains batched in the main process.
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_evaluation_worker,
            initargs=(population, function_set),
        )
    try:
        for timestep in range(timestep_limit):
            # Phase 1: evaluate action trees in separate Python processes, then
            # advance games centrally. TensorFlow never enters worker processes.
            if executor is None:
                moves = []
                for index in active:
                    move, _ = evaluate_individual_move(
                        population[index],
                        terminal_values(games[index], latents[index]),
                        games[index].legal_moves(),
                        function_set,
                        rngs[index],
                        random_tie_break=random_tie_break,
                    )
                    moves.append((index, move))
            else:
                tasks = [
                    (
                        index,
                        terminal_values(games[index], latents[index]),
                        games[index].legal_moves(),
                        seed + index * 1_000_003 + timestep,
                        random_tie_break,
                    )
                    for index in active
                ]
                chunk_count = min(worker_count, len(tasks))
                task_chunks = [tasks[offset::chunk_count] for offset in range(chunk_count)]
                move_chunks = executor.map(_evaluate_move_chunk, task_chunks)
                moves = [move for chunk in move_chunks for move in chunk]

            for index, move in moves:
                game = games[index]
                score_before_step = game.board.score
                game.step(move if move is not None else "stay")
                steps_taken[index] += 1
                if game.board.score > score_before_step:
                    steps_since_score[index] = 0
                else:
                    steps_since_score[index] += 1
                    if steps_since_score[index] > stagnation_grace_steps:
                        stagnation_costs[index] += stagnation_penalty
                    if steps_since_score[index] >= stagnation_termination_steps:
                        terminated_for_stagnation[index] = True

            # Phase 2: terminal games leave the batch; all remaining results are
            # encoded centrally exactly once before the next population timestep.
            active = [
                index for index in active
                if not (
                    games[index].game_over
                    or games[index].won
                    or terminated_for_stagnation[index]
                )
            ]
            if not active or timestep == timestep_limit - 1:
                break
            encoded = encode_boards_batch(
                [games[index].board for index in active], encoder, metadata
            )
            latents = {
                population_index: encoded[batch_index]
                for batch_index, population_index in enumerate(active)
            }
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    fitnesses = [
        calculate_fitness(
            game.board.score - starting_scores[index],
            game.game_over or (steps_taken[index] < timestep_limit and not game.won),
            population[index],
            parsimony,
        ) - stagnation_costs[index]
        for index, game in enumerate(games)
    ]
    for individual, fitness in zip(population, fitnesses):
        set_fitness(individual, fitness)
    return fitnesses


def evolve_pacman(
    starting_boards: Sequence[Board],
    encoder_bundle: EncoderBundle,
    function_set,
    *,
    population_size: int,
    generations: int,
    timestep_limit: int,
    initial_board: Board | None = None,
    full_game_timestep_limit: int = 1_000,
    sampled_state_count: int = 3,
    full_game_weight: float = 0.8,
    stagnation_penalty: float = NON_SCORING_MOVE_PENALTY,
    stagnation_grace_steps: int = 0,
    stagnation_termination_steps: int = 50,
    min_initial_depth: int = 2,
    max_initial_depth: int = 5,
    max_evolved_depth: int = 10,
    crossover_probability: float = 0.97,
    mutation_probability: float = 0.03,
    top_fraction: float = 0.32,
    top_probability: float = 0.8,
    parsimony: float = 0.0,
    worker_count: int = 1,
    seed: int = 0,
    target_fitness: float | None = None,
    checkpoint_directory: str | Path | None = None,
    log_path: str | Path | None = None,
    verbose: bool = True,
) -> EvolutionResult:
    """Run GP evolution using both sampled states and the initial game state.

    Every generation evaluates every individual on shared sampled training
    states and on a game beginning at ``initial_board``. Fitness is a weighted
    combination of the initial-game result and mean sampled-state result.
    """
    if not starting_boards:
        raise ValueError("at least one starting board is required")
    if population_size < 1 or generations < 1:
        raise ValueError("population_size and generations must be positive")
    if min_initial_depth < 1 or max_initial_depth < min_initial_depth:
        raise ValueError("invalid initialization depth range")
    if max_evolved_depth < max_initial_depth:
        raise ValueError("max_evolved_depth cannot be below max_initial_depth")
    if full_game_timestep_limit < 1:
        raise ValueError("full_game_timestep_limit must be positive")
    if sampled_state_count < 1:
        raise ValueError("sampled_state_count must be positive")
    if not 0.0 <= full_game_weight <= 1.0:
        raise ValueError("full_game_weight must be between 0 and 1")
    if stagnation_penalty < 0:
        raise ValueError("stagnation_penalty cannot be negative")
    if stagnation_grace_steps < 0:
        raise ValueError("stagnation_grace_steps cannot be negative")
    if stagnation_termination_steps < 1:
        raise ValueError("stagnation_termination_steps must be positive")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be greater than 0 and at most 1")
    if not 0.0 <= top_probability <= 1.0:
        raise ValueError("top_probability must be between 0 and 1")
    if initial_board is None:
        initial_board = read_board(DEFAULT_INITIAL_BOARD)

    rng = random.Random(seed)
    random.seed(seed)
    toolbox = build_toolbox(
        function_set,
        min_initial_depth,
        max_initial_depth,
        max_evolved_depth,
        top_fraction,
        top_probability,
    )
    population = toolbox.population(n=population_size)
    checkpoint_path = Path(checkpoint_directory) if checkpoint_directory else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    hall_of_fame = tools.HallOfFame(1)
    stopped_early = False
    generations_completed = 0
    history: list[GenerationStats] = []
    episode_executor = None
    if worker_count > 1:
        episode_executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_episode_worker,
            initargs=(encoder_bundle.latent_dim, str(encoder_bundle.directory.parent)),
        )

    for generation in range(generations):
        sampled_boards = rng.sample(
            list(starting_boards), k=min(sampled_state_count, len(starting_boards))
        )

        generation_seed = seed + generation * 1_000_003
        sampled_runs: list[list[float]] = []
        if episode_executor is None:
            for sample_index, board in enumerate(sampled_boards):
                sampled_runs.append(evaluate_population_lockstep(
                    population, board, timestep_limit,
                    encoder_bundle.encoder, encoder_bundle.metadata,
                    function_set=function_set, parsimony=parsimony,
                    stagnation_penalty=stagnation_penalty,
                    stagnation_grace_steps=stagnation_grace_steps,
                    stagnation_termination_steps=stagnation_termination_steps,
                    seed=generation_seed + sample_index, worker_count=1,
                ))
            full_game_fitnesses = evaluate_population_lockstep(
                population, initial_board, full_game_timestep_limit,
                encoder_bundle.encoder, encoder_bundle.metadata,
                function_set=function_set, parsimony=parsimony,
                stagnation_penalty=stagnation_penalty,
                stagnation_grace_steps=stagnation_grace_steps,
                stagnation_termination_steps=stagnation_termination_steps,
                seed=generation_seed, worker_count=1,
            )
        else:
            for sample_index, board in enumerate(sampled_boards):
                sampled_runs.append(_evaluate_population_by_episode(
                    episode_executor, population, board, timestep_limit,
                    function_set, parsimony, stagnation_penalty,
                    stagnation_grace_steps, stagnation_termination_steps,
                    generation_seed + sample_index, worker_count,
                ))
            full_game_fitnesses = _evaluate_population_by_episode(
                episode_executor, population, initial_board,
                full_game_timestep_limit, function_set, parsimony,
                stagnation_penalty, stagnation_grace_steps,
                stagnation_termination_steps,
                generation_seed, worker_count,
            )
        sampled_fitnesses = [
            float(np.mean(values)) for values in zip(*sampled_runs)
        ]
        fitnesses = [
            full_game_weight * full_game + (1.0 - full_game_weight) * sampled
            for sampled, full_game in zip(sampled_fitnesses, full_game_fitnesses)
        ]
        for individual, fitness in zip(population, fitnesses):
            set_fitness(individual, fitness)

        statistics = population_statistics(population, generation)
        history.append(statistics)
        if log_path is not None:
            write_generation_history(log_path, history)
        if verbose:
            print(
                f"generation={generation:04d} "
                f"best={statistics.best_fitness:.3f} "
                f"average={statistics.average_fitness:.3f} "
                f"worst={statistics.worst_fitness:.3f} "
                f"avg_nodes={statistics.average_node_count:.1f}"
            )

        hall_of_fame.update(population)
        generation_best = tools.selBest(population, 1)[0]

        generations_completed = generation + 1
        if checkpoint_path is not None:
            save_individual(
                generation_best,
                checkpoint_path / f"generation_{generation:04d}_best.pkl",
            )
            save_individual(hall_of_fame[0], checkpoint_path / "best_overall.pkl")

        if (
            target_fitness is not None
            and fitness_value(generation_best) >= target_fitness
        ):
            stopped_early = True
            break

        if generation < generations - 1:
            parents = toolbox.select(population, population_size)
            population = algorithms.varOr(
                parents,
                toolbox,
                lambda_=population_size,
                cxpb=crossover_probability,
                mutpb=mutation_probability,
            )

    if episode_executor is not None:
        episode_executor.shutdown(wait=True)
    return EvolutionResult(
        best=hall_of_fame[0],
        population=population,
        generations_completed=generations_completed,
        stopped_early=stopped_early,
        history=tuple(history),
    )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("board", type=Path)

    parser.add_argument("--latent-dim", type=int, default=16,
                        choices=SUPPORTED_LATENT_DIMS)

    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--timestep-limit", type=int, default=1000)
    parser.add_argument(
        "--initial-board",
        type=Path,
        default=DEFAULT_INITIAL_BOARD,
        help="initial Pac-Man state used for every individual's full-game rollout",
    )
    parser.add_argument(
        "--full-game-timestep-limit",
        type=int,
        default=1_000,
        help="safety ceiling for a full-game rollout (default: 1000)",
    )
    parser.add_argument(
        "--sampled-state-count",
        type=int,
        default=3,
        help="shared intermediate states evaluated per generation (default: 3)",
    )
    parser.add_argument(
        "--full-game-weight",
        type=float,
        default=0.8,
        help="fitness weight assigned to the initial-board game (default: 0.8)",
    )
    parser.add_argument(
        "--stagnation-penalty",
        type=float,
        default=NON_SCORING_MOVE_PENALTY,
        help="fitness cost per scoreless turn (default: 100, one tenth of a pellet)",
    )
    parser.add_argument(
        "--stagnation-grace-steps",
        type=int,
        default=0,
        help="scoreless turns allowed before penalties (default: 0)",
    )
    parser.add_argument(
        "--stagnation-termination-steps",
        type=int,
        default=50,
        help="end an evaluation after this many scoreless turns (default: 50)",
    )

    parser.add_argument("--min-initial-depth", type=int, default=2)
    parser.add_argument("--max-initial-depth", type=int, default=5)
    parser.add_argument("--max-evolved-depth", type=int, default=10)

    parser.add_argument("--crossover-probability", type=float, default=0.97)
    parser.add_argument("--mutation-probability", type=float, default=0.03)

    parser.add_argument("--top-fraction", type=float, default=0.32)
    parser.add_argument("--top-probability", type=float, default=0.8)

    parser.add_argument("--parsimony", type=float, default=0.0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--target-fitness", type=float)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "directory for all run artifacts; defaults to --checkpoint-directory, "
            "the --log-path parent, or runs/run_<timestamp>"
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        help="frame rate for best_overall.mp4 (default: 10)",
    )
    parser.add_argument(
        "--video-scale",
        type=int,
        default=1,
        choices=range(1, 5),
        help="integer render scale for best_overall.mp4 (default: 1)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="skip recording best_overall.mp4"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    if args.max_initial_depth > args.max_evolved_depth:
        raise ValueError("max_initial_depth cannot exceed max_evolved_depth")

    if not math.isclose(
        args.crossover_probability + args.mutation_probability,
        1.0
    ):
        raise ValueError("crossover + mutation probabilities must equal 1")
    if args.video_fps <= 0:
        raise ValueError("video_fps must be positive")

    if args.output_directory is not None:
        output_directory = args.output_directory
    elif args.checkpoint_directory is not None:
        output_directory = args.checkpoint_directory
    elif args.log_path is not None:
        output_directory = args.log_path.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path("runs") / f"run_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = args.checkpoint_directory or output_directory
    log_path = args.log_path or output_directory / "history.csv"
    print(f"Run artifacts: {output_directory}")

    board = []
    #Given the boards arg, which is a directory, choose one random board for every generation. If not enough exist, just choose them all.
    for board_file in args.board.iterdir():
        if not board_file.is_file() or not board_file.name.endswith(".txt"):
            continue
        print(f"Training on board: {board_file}")
        board.append(read_board(board_file))
    if not board:
        raise ValueError(f"No valid board files found in {args.board}")

    encoder_bundle = EncoderBundle.load(args.latent_dim)

    terminal_names = [
        "player_x",
        "player_y",
        "powerup_timer",
        "player_up", "player_down", "player_left", "player_right",
        "inky_x", "inky_y",
        "blinky_x", "blinky_y",
        "pinky_x", "pinky_y",
        "clyde_x", "clyde_y",
        "up", "down", "left", "right",
    ]

    terminal_names += [
        f"z{i}" for i in range(args.latent_dim)
    ]

    function_set = build_primitive_set(terminal_names, constants=[-1, 0, 1])

    result = evolve_pacman(
        board,
        encoder_bundle,
        function_set,

        population_size=args.population_size,
        generations=args.generations,
        timestep_limit=args.timestep_limit,
        initial_board=read_board(args.initial_board),
        full_game_timestep_limit=args.full_game_timestep_limit,
        sampled_state_count=args.sampled_state_count,
        full_game_weight=args.full_game_weight,
        stagnation_penalty=args.stagnation_penalty,
        stagnation_grace_steps=args.stagnation_grace_steps,
        stagnation_termination_steps=args.stagnation_termination_steps,

        min_initial_depth=args.min_initial_depth,
        max_initial_depth=args.max_initial_depth,
        max_evolved_depth=args.max_evolved_depth,

        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,

        top_fraction=args.top_fraction,
        top_probability=args.top_probability,

        parsimony=args.parsimony,
        worker_count=args.worker_count,
        seed=args.seed,

        target_fitness=args.target_fitness,
        checkpoint_directory=checkpoint_directory,
        log_path=log_path,
    )
    best_path = output_directory / "best_overall.pkl"
    save_individual(result.best, best_path)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "plot_gp_long.py"),
            str(log_path),
            str(output_directory),
            str(best_path),
        ],
        check=True,
    )
    if not args.no_video:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "GeneticAlgo.watch_individual",
                str(best_path),
                "--board",
                str(args.initial_board),
                "--latent-dim",
                str(args.latent_dim),
                "--video",
                str(output_directory / "best_overall.mp4"),
                "--fps",
                str(args.video_fps),
                "--scale",
                str(args.video_scale),
                "--max-steps",
                str(args.full_game_timestep_limit),
                "--seed",
                str(args.seed),
            ],
            cwd=PROJECT_DIR,
            check=True,
        )
    print(
        f"Finished after {result.generations_completed} generations. "
        f"Best fitness: {fitness_value(result.best)}"
    )

if __name__ == "__main__":
    main()
