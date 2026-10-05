"""Scaled stagnation, late-pellet weighting and endgame training boards."""
import unittest

from GeneticAlgo.train_board_graph import (
    ENDGAME_START,
    EvaluationRules,
    build_primitive_set,
    endgame_board,
    endgame_style_for,
    evaluate_population_lockstep,
    is_endgame_slot,
    pellet_value,
    stagnation_limit,
)
from deap import creator, gp
from PacmanGame import PacmanGame
from PacmanGame.generate_tetris_board import generate_board
from PacmanGame.render_board import PELLET, POWER_PELLET


def count_pellets(board):
    return sum(bool(value & (PELLET | POWER_PELLET)) for _, _, value in board.iter_tiles())


class PelletValueTests(unittest.TestCase):
    def test_linear_by_default(self):
        for ratio in (0.0, 0.37, 0.8, 1.0):
            self.assertEqual(pellet_value(ratio), ratio)

    def test_last_fifth_is_worth_the_first_four_fifths(self):
        self.assertAlmostEqual(pellet_value(0.4, 0.5), 0.25)
        self.assertAlmostEqual(pellet_value(0.8, 0.5), 0.5)
        self.assertAlmostEqual(pellet_value(0.9, 0.5), 0.75)
        self.assertEqual(pellet_value(1.0, 0.5), 1.0)
        self.assertEqual(pellet_value(0.0, 0.5), 0.0)

    def test_share_of_one_fifth_is_linear(self):
        for ratio in (0.1, 0.5, 0.85):
            self.assertAlmostEqual(pellet_value(ratio, 0.2), ratio)


class StagnationLimitTests(unittest.TestCase):
    def test_fixed_ignores_the_board(self):
        self.assertEqual(stagnation_limit(50, 3, 300), 50)

    def test_inverse_grows_as_the_board_empties(self):
        self.assertEqual(stagnation_limit(50, 300, 300, "inverse"), 50)
        self.assertEqual(stagnation_limit(50, 150, 300, "inverse"), 100)
        self.assertEqual(stagnation_limit(50, 30, 300, "inverse"), 500)


class EndgameBoardTests(unittest.TestCase):
    def setUp(self):
        self.full = generate_board(1234)
        self.endgame = endgame_board(self.full)

    def test_three_quarters_of_the_pellets_are_gone(self):
        total = count_pellets(self.full)
        eaten = int(total * ENDGAME_START)
        self.assertEqual(count_pellets(self.endgame), total - eaten)
        self.assertEqual(self.endgame.pellets_remaining, total - eaten)
        self.assertEqual(self.endgame.pellets_eaten, eaten)
        self.assertEqual(count_pellets(self.full), self.full.pellets_remaining)

    def test_deterministic_and_leaves_the_full_board_alone(self):
        self.assertEqual(endgame_board(self.full), self.endgame)
        self.assertEqual(self.full, generate_board(1234))

    def test_every_ghost_is_released_at_the_start(self):
        game = PacmanGame(self.endgame)
        self.assertNotIn("house", game._ghost_state.values())
        self.assertIn("house", PacmanGame(self.full)._ghost_state.values())

    def test_score_counts_only_pellets_eaten_from_the_start(self):
        primitive_set = build_primitive_set()
        tree = creator.BoardGraphIndividual(
            gp.PrimitiveTree.from_string(primitive_set.arguments[0], primitive_set)
        )
        _, _, ratios, _ = evaluate_population_lockstep(
            [tree], self.endgame, 1, primitive_set
        )
        self.assertLessEqual(ratios[0], 1 / self.endgame.pellets_remaining)

    def test_scaled_stagnation_sees_the_whole_board(self):
        total = self.endgame.pellets_eaten + self.endgame.pellets_remaining
        limit = stagnation_limit(50, self.endgame.pellets_remaining, total, "inverse")
        self.assertAlmostEqual(limit, 50 * total / self.endgame.pellets_remaining)
        self.assertGreater(limit, 190)


class IslandEndgameTests(unittest.TestCase):
    def test_same_pellet_count_as_a_sweep_and_deterministic(self):
        for seed in (1234, 1235, 1236):
            full = generate_board(seed)
            islands = endgame_board(full, style="islands", variant=seed)
            self.assertEqual(islands.pellets_remaining, endgame_board(full).pellets_remaining)
            self.assertEqual(count_pellets(islands), islands.pellets_remaining)
            self.assertEqual(islands, endgame_board(full, style="islands", variant=seed))

    def test_pellets_are_left_in_a_few_connected_patches(self):
        full = generate_board(1234)
        board = endgame_board(full, style="islands", variant=1234)
        pellets = {
            (x, y) for x, y, value in board.iter_tiles() if value & (PELLET | POWER_PELLET)
        }
        patches, seen = 0, set()
        for start in pellets:
            if start in seen:
                continue
            patches += 1
            stack = [start]
            seen.add(start)
            while stack:
                x, y = stack.pop()
                for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    point = (x + step[0], y + step[1])
                    if point in pellets and point not in seen:
                        seen.add(point)
                        stack.append(point)
        self.assertLessEqual(patches, 4)

    def test_unknown_style_is_rejected(self):
        with self.assertRaises(ValueError):
            endgame_board(generate_board(1234), style="random")

    def test_endgame_boards_cycle_through_the_styles(self):
        styles = [endgame_style_for(index, 8, 3, ("sweep", "islands")) for index in range(48)]
        used = [style for style in styles if style]
        self.assertEqual(used, ["sweep", "islands"] * 9)
        self.assertEqual([bool(style) for style in styles],
                         [is_endgame_slot(index, 8, 3) for index in range(48)])


class EndgameSlotTests(unittest.TestCase):
    def test_every_window_has_the_requested_number(self):
        for boards, endgames in ((8, 3), (3, 1), (8, 0), (8, 8), (9, 3)):
            flags = [is_endgame_slot(index, boards, endgames) for index in range(60)]
            for start in range(60 - boards):
                self.assertEqual(sum(flags[start:start + boards]), endgames)


class RulesTests(unittest.TestCase):
    def test_late_pellet_share_changes_only_the_pellet_score(self):
        primitive_set = build_primitive_set()
        board = generate_board(77)
        tree = creator.BoardGraphIndividual(
            gp.PrimitiveTree.from_string(primitive_set.arguments[0], primitive_set)
        )
        plain, _, ratios, labels = evaluate_population_lockstep([tree], board, 200, primitive_set)
        curved, _, curved_ratios, curved_labels = evaluate_population_lockstep(
            [tree], board, 200, primitive_set, rules=EvaluationRules(late_pellet_share=0.5)
        )
        self.assertEqual(ratios, curved_ratios)
        self.assertEqual(labels, curved_labels)
        if 0 < ratios[0] < 1:
            self.assertLess(curved[0], plain[0])


if __name__ == "__main__":
    unittest.main()
