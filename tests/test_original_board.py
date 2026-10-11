"""The original-maze minimiser: ranking, shrinking moves and pruning."""
import argparse
import random
import unittest

from deap import creator, gp

import GeneticAlgo.train_original_board as minimiser
from GeneticAlgo.train_board_graph import build_primitive_set

SETTINGS = argparse.Namespace(
    timestep_limit=1000, stagnation_penalty=20.0, stagnation_termination_steps=50,
)


class MinimiserTests(unittest.TestCase):
    def setUp(self):
        self.primitive_set = build_primitive_set()
        self.scorer = minimiser.Scorer(SETTINGS, self.primitive_set, None, 1)

    def tree(self, expression):
        return creator.BoardGraphIndividual(
            gp.PrimitiveTree.from_string(expression, self.primitive_set)
        )

    def test_a_clear_beats_any_non_clear_and_smaller_clears_win(self):
        small = self.tree("pellet_here")
        large = self.tree("add(pellet_here, pellet_distance)")
        failing = self.tree("neg(pellet_here)")
        known = {str(small): (100000.0, 1.0, True), str(large): (129000.0, 1.0, True),
                 str(failing): (99000.0, 0.99, False)}
        self.scorer.score([small, large, failing], known)
        self.assertEqual(self.scorer.games_played, 0)
        self.assertGreater(small.fitness, large.fitness)
        self.assertGreater(large.fitness, failing.fitness)

    def test_non_clears_rank_by_game_fitness_then_fewer_nodes(self):
        better = self.tree("add(pellet_here, pellet_distance)")
        worse = self.tree("pellet_here")
        same_small = self.tree("is_reverse")
        same_large = self.tree("neg(is_reverse)")
        known = {str(better): (50000.0, 0.5, False), str(worse): (40000.0, 0.4, False),
                 str(same_small): (30000.0, 0.3, False), str(same_large): (30000.0, 0.3, False)}
        self.scorer.score([better, worse, same_small, same_large], known)
        self.assertGreater(better.fitness, worse.fitness)
        self.assertGreater(same_small.fitness, same_large.fitness)

    def test_hoist_returns_a_subtree(self):
        random.seed(2)
        tree = self.tree("add(mul(pellet_here, is_reverse), neg(pellet_distance))")
        original = str(tree)
        hoisted, = minimiser.hoist(tree)
        self.assertLess(len(hoisted), 7)
        self.assertIn(str(hoisted), original)

    def test_pruning_keeps_the_clear_and_never_grows_the_tree(self):
        # Found by pruning an island-run champion; clears the original maze.
        tree = self.tree(
            "sub(sqrt(div(danger_ghost_distance, pellet_distance)), "
            "div(frightened_time, power_pellet_distance))"
        )
        pruned = minimiser.prune(tree, self.scorer, self.primitive_set, log=lambda line: None)
        self.assertTrue(pruned.result[2])
        self.assertLessEqual(len(pruned), len(tree))

    def test_pruning_leaves_a_failing_tree_alone(self):
        tree = self.tree("neg(pellet_here)")
        pruned = minimiser.prune(tree, self.scorer, self.primitive_set, log=lambda line: None)
        self.assertEqual(str(pruned), str(tree))
        self.assertFalse(pruned.result[2])


if __name__ == "__main__":
    unittest.main()
