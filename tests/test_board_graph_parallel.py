"""Parallel board evaluation must preserve serial fitness and outcomes."""

import random
import unittest
from concurrent.futures import ProcessPoolExecutor

from deap import creator

from GeneticAlgo.train_board_graph import (
    FEATURE_NAMES,
    VALIDATION_BOARD_PATH,
    build_primitive_set,
    build_toolbox,
    evaluate_population_on_boards,
)
from PacmanGame import read_board


class BoardGraphParallelTests(unittest.TestCase):
    def test_tree_size_breaks_exact_score_ties_only(self) -> None:
        long_tree = creator.BoardGraphFitnessMax()
        short_tree = creator.BoardGraphFitnessMax()
        higher_score = creator.BoardGraphFitnessMax()
        long_tree.values = (100.0, -1000.0)
        short_tree.values = (100.0, -5.0)
        higher_score.values = (100.01, -10000.0)
        self.assertGreater(short_tree, long_tree)
        self.assertGreater(higher_score, short_tree)

    def test_parallel_matches_serial_with_random_ties_and_multiple_boards(self) -> None:
        random.seed(42)
        primitive_set = build_primitive_set(FEATURE_NAMES, constants=(-1, 0, 1))
        toolbox = build_toolbox(primitive_set, 2, 3, 4, 0.5, 0.8)
        population = toolbox.population(n=12)
        board = read_board(VALIDATION_BOARD_PATH)
        boards = [(42, board), (43, board.copy())]
        options = dict(
            parsimony=0.01,
            stagnation_penalty=100.0,
            stagnation_grace_steps=0,
            stagnation_termination_steps=10,
            random_tie_break=True,
        )
        serial = evaluate_population_on_boards(
            population, boards, 20, primitive_set, **options
        )
        with ProcessPoolExecutor(max_workers=3) as executor:
            parallel = evaluate_population_on_boards(
                population, boards, 20, primitive_set,
                executor=executor, workers=3, **options
            )
        self.assertEqual(parallel, serial)
        self.assertEqual(
            [individual.fitness.values[0] for individual in population],
            serial[0],
        )


if __name__ == "__main__":
    unittest.main()
