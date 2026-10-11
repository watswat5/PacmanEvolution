"""Steady-state breeding and replacement."""
import random
import unittest

import GeneticAlgo.train_board_graph as trainer
from GeneticAlgo.train_board_graph import (
    build_primitive_set,
    build_toolbox,
    steady_state_children,
    steady_state_replace,
)


def scored_population(size=20):
    random.seed(3)
    primitive_set = build_primitive_set()
    toolbox = build_toolbox(primitive_set, 2, 4, 10, 0.5, 0.8)
    population = toolbox.population(n=size)
    for index, tree in enumerate(population):
        tree.fitness.values = (float(index), 0.0)
    return population, toolbox


class SteadyStateTests(unittest.TestCase):
    def test_children_are_new_unscored_trees(self):
        population, toolbox = scored_population()
        before = [str(tree) for tree in population]
        children = steady_state_children(population, toolbox, 6, 0.7, 3)
        self.assertEqual(len(children), 6)
        self.assertTrue(all(not child.fitness.valid for child in children))
        self.assertEqual(before, [str(tree) for tree in population])
        self.assertTrue(all(tree.fitness.valid for tree in population))

    def test_a_better_child_replaces_one_of_the_worst_fifth(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (100.0, 0.0)
        replaced = steady_state_replace(population, [child], 0.20, 3)
        self.assertEqual(len(replaced), 1)
        index, child_index = replaced[0]
        self.assertEqual(child_index, 0)
        self.assertLess(index, 4)  # fitness 0..3 are the worst 20% of 20
        self.assertIs(population[index], child)
        self.assertEqual(len(population), 20)

    def test_the_victim_is_the_worst_of_the_drawn_pool(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (100.0, 0.0)
        # With the whole bottom fifth drawn, the victim is the very worst tree.
        (index, _), = steady_state_replace(population, [child], 0.20, 4)
        self.assertEqual(index, 0)

    def test_a_child_no_better_than_its_opponent_is_dropped(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (-1.0, 0.0)
        original = list(population)
        self.assertEqual(steady_state_replace(population, [child], 0.20, 3), [])
        self.assertEqual(population, original)
        child.fitness.values = (0.0, 0.0)  # ties with the worst tree: not strictly better
        self.assertEqual(steady_state_replace(population, [child], 0.20, 4), [])

    def test_duel_score_decides_the_contest_but_not_the_bottom_fifth(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (-5.0, 0.0)  # worse than everyone on the shared board
        # Over the extra games the child is strong and tree 2 is the weakest.
        duel = {id(child): 50.0, id(population[0]): 30.0, id(population[1]): 20.0,
                id(population[2]): 10.0, id(population[3]): 40.0}
        asked = []

        def duel_score(tree):
            asked.append(id(tree))
            return duel[id(tree)]

        (index, _), = steady_state_replace(population, [child], 0.20, 4, duel_score)
        self.assertEqual(index, 2)
        # Only the child and the four drawn trees were ever scored.
        self.assertLessEqual(set(asked), set(duel))

    def test_a_child_that_loses_the_duel_is_dropped(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (100.0, 0.0)  # lucky on the shared board
        self.assertEqual(
            steady_state_replace(population, [child], 0.20, 4,
                                 lambda tree: -1.0 if tree is child else 0.0),
            [],
        )

    def test_a_child_that_plays_like_an_existing_tree_is_dropped(self):
        population, toolbox = scored_population()
        children = steady_state_children(population, toolbox, 3, 1.0, 3)
        for child, value in zip(children, (7.0, 50.0, 50.0)):
            child.fitness.values = (value, 0.0)
        behaviour = lambda tree: tree.fitness.values[0]
        replaced = steady_state_replace(population, children, 0.20, 4, None, behaviour)
        # 7.0 matches population tree 7; the second 50.0 matches the first one in.
        self.assertEqual([child_index for _, child_index in replaced], [1])

    def test_a_tie_goes_to_the_child_only_when_asked(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (0.0, -1e9)  # ties the worst tree; much larger
        self.assertEqual(steady_state_replace(population, [child], 0.20, 4), [])
        (index, _), = steady_state_replace(population, [child], 0.20, 4, replace_on_tie=True)
        self.assertEqual(index, 0)
        self.assertIs(population[0], child)

    def test_a_tie_in_the_duel_also_goes_to_the_child(self):
        population, toolbox = scored_population()
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        child.fitness.values = (-5.0, 0.0)
        duel = lambda tree: 10.0
        self.assertEqual(steady_state_replace(population, [child], 0.20, 4, duel), [])
        self.assertEqual(len(steady_state_replace(population, [child], 0.20, 4, duel, None, True)), 1)

    def test_several_children_are_placed_one_after_another(self):
        population, toolbox = scored_population()
        children = steady_state_children(population, toolbox, 4, 1.0, 3)
        for child in children:
            child.fitness.values = (100.0, 0.0)
        replaced = steady_state_replace(population, children, 0.20, 4)
        # Each takes the worst remaining tree: fitness 0, 1, 2, 3 in turn.
        self.assertEqual([index for index, _ in replaced], [0, 1, 2, 3])


class DiversityTests(unittest.TestCase):
    def setUp(self):
        self.primitive_set = build_primitive_set()

    def tree(self, expression):
        from deap import creator, gp
        return creator.BoardGraphIndividual(
            gp.PrimitiveTree.from_string(expression, self.primitive_set)
        )

    def test_states_are_fixed_and_leave_the_random_stream_alone(self):
        random.seed(11)
        expected = random.random()
        random.seed(11)
        trainer.diversity_states.cache_clear()
        first = trainer.diversity_states()
        self.assertEqual(random.random(), expected)
        trainer.diversity_states.cache_clear()
        self.assertEqual(first, trainer.diversity_states())
        self.assertGreaterEqual(len(first), 200)
        self.assertTrue(all(len(observations) >= 2 for observations, _ in first))

    def test_copies_and_rewrites_of_one_strategy_have_no_diversity(self):
        clones = [self.tree("sqrt(pellet_here)") for _ in range(6)]
        clones.append(self.tree("abs(pellet_here)"))
        diversity, behaviours = trainer.population_diversity(clones, self.primitive_set)
        self.assertEqual(diversity, 0.0)
        self.assertEqual(behaviours, 1)

    def test_different_strategies_are_counted_and_spread(self):
        mixed = [self.tree(name) for name in (
            "pellet_here", "neg(pellet_here)", "danger_ghost_distance",
            "neg(pellet_distance)", "is_reverse",
        )]
        diversity, behaviours = trainer.population_diversity(mixed, self.primitive_set)
        self.assertGreaterEqual(behaviours, 4)
        self.assertGreater(diversity, 0.3)
        self.assertLess(diversity, 1.0)
        half = [self.tree("pellet_here") for _ in range(5)] + [self.tree("neg(pellet_here)") for _ in range(5)]
        lower, two = trainer.population_diversity(half, self.primitive_set)
        self.assertEqual(two, 2)
        self.assertLess(lower, diversity + 0.4)

    def test_a_bred_child_does_not_inherit_its_parents_signature(self):
        population, toolbox = scored_population()
        trainer.population_diversity(population, self.primitive_set)
        child, = steady_state_children(population, toolbox, 1, 1.0, 3)
        self.assertIsNone(child.move_signature)


class BoardCacheTests(unittest.TestCase):
    SETTINGS = dict(
        parsimony=1.0, stagnation_penalty=20.0, stagnation_grace_steps=0,
        stagnation_termination_steps=50, random_tie_break=False,
    )

    def setUp(self):
        random.seed(5)
        self.primitive_set = build_primitive_set()
        toolbox = build_toolbox(self.primitive_set, 2, 4, 10, 0.5, 0.8)
        self.population = toolbox.population(n=5)
        self.toolbox = toolbox
        self.boards = [(seed, trainer.generate_board(seed)) for seed in (901, 902, 903, 904)]
        self.games = []
        original = trainer.evaluate_population_on_boards

        def counting(trees, boards, *args, **kwargs):
            self.games.append((len(trees), [seed for seed, _ in boards]))
            return original(trees, boards, *args, **kwargs)

        trainer.evaluate_population_on_boards = counting
        self.addCleanup(setattr, trainer, "evaluate_population_on_boards", original)
        self.original = original

    def cached(self, trees, boards):
        return trainer.evaluate_population_with_board_cache(
            trees, boards, 150, self.primitive_set, **self.SETTINGS
        )

    def test_window_slides_with_one_new_game_per_tree(self):
        first = self.cached(self.population, self.boards[:3])
        self.assertEqual(self.games, [(5, [901]), (5, [902]), (5, [903])])
        self.games.clear()
        again = self.cached(self.population, self.boards[:3])
        self.assertEqual(self.games, [])  # nothing is replayed
        self.assertEqual(first, again)
        slid = self.cached(self.population, self.boards[1:])
        self.assertEqual(self.games, [(5, [904])])
        self.assertEqual(set(self.population[0].board_scores), {902, 903, 904})
        # Same numbers as scoring every tree on the whole window from scratch.
        fresh = [self.toolbox.clone(tree) for tree in self.population]
        expected = self.original(fresh, self.boards[1:], 150, self.primitive_set, **self.SETTINGS)
        for got, want in zip(slid[0], expected[0]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(slid[3], expected[3])
        self.assertEqual(slid[1], expected[1])

    def test_a_child_plays_the_whole_window_and_only_that(self):
        self.cached(self.population, self.boards[:3])
        children = steady_state_children(self.population, self.toolbox, 2, 1.0, 3)
        self.assertTrue(all(child.board_scores == {} for child in children))
        self.games.clear()
        self.cached(children, self.boards[:3])
        self.assertEqual(self.games, [(2, [901]), (2, [902]), (2, [903])])
        self.assertEqual(set(children[0].board_scores), {901, 902, 903})


if __name__ == "__main__":
    unittest.main()
