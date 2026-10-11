"""The two-sided match behind versus.sh (no window)."""
import unittest

from deap import creator, gp

from GeneticAlgo.train_board_graph import build_primitive_set
from GeneticAlgo.versus import RETRO_PALETTE, VersusMatch, maze_color


def simple_tree():
    primitive_set = build_primitive_set()
    return creator.BoardGraphIndividual(
        gp.PrimitiveTree.from_string(primitive_set.arguments[0], primitive_set)
    )


class VersusMatchTests(unittest.TestCase):
    def test_both_sides_start_on_the_same_board(self):
        match = VersusMatch(simple_tree(), seed=5)
        self.assertEqual(match.human.board, match.tree_game.board)
        self.assertEqual(match.seed, 5)

    def test_one_key_press_steps_both_sides(self):
        match = VersusMatch(simple_tree(), seed=5)
        match.step("left")
        match.step(None)
        self.assertEqual(match.steps, {"human": 2, "tree": 2})
        self.assertIn("Score ", match.status("human"))
        self.assertIn("Score ", match.status("tree"))

    def test_a_finished_side_stops_while_the_other_continues(self):
        match = VersusMatch(simple_tree(), seed=5)
        for _ in range(2000):
            if match.over:
                break
            match.step(None)
        self.assertTrue(match.over)
        self.assertIn(match.result(), ("Draw", "PLAYER wins", "AI wins"))
        steps = dict(match.steps)
        match.step("up")
        self.assertEqual(match.steps, steps)

    def test_new_board_resets_and_changes_the_layout(self):
        match = VersusMatch(simple_tree(), seed=5)
        first = match.human.board.tiles
        match.step("left")
        match.new_board(6)
        self.assertEqual(match.steps, {"human": 0, "tree": 0})
        self.assertNotEqual(match.human.board.tiles, first)
        match.new_board()
        self.assertNotEqual(match.seed, 6)


class ScoringTests(unittest.TestCase):
    def test_dots_score_ten_and_energizers_fifty(self):
        match = VersusMatch(simple_tree(), seed=5)
        for _ in range(60):
            match.step(None)
        for side, game in (("human", match.human), ("tree", match.tree_game)):
            eaten = match.eaten(game)
            self.assertGreaterEqual(match.score[side], 10 * eaten)
            self.assertEqual(match.score[side] % 10, 0)

    def test_ghost_chain_doubles_and_resets(self):
        match = VersusMatch(simple_tree(), seed=5)

        class FakeBoard:
            score = 0
            def iter_tiles(self):
                return iter(())

        class FakeGame:
            board = FakeBoard()
            def step(self, move):
                self.board.score += self.gain

        game = FakeGame()
        for gain, expected in ((100, 200), (200, 200 + 400 + 800), (100, 3000), (100, 4600)):
            game.gain = gain
            match._scored_step("human", game, "left")
            self.assertEqual(match.score["human"], expected)

    def test_a_clear_beats_a_higher_score(self):
        match = VersusMatch(simple_tree(), seed=5)
        match.score = {"human": 100, "tree": 9000}
        match.human.won = True
        self.assertEqual(match.result(), "PLAYER wins")
        match.human.won = False
        self.assertEqual(match.result(), "AI wins")
        match.score["human"] = 9000
        self.assertEqual(match.result(), "Draw")


class MazeColorTests(unittest.TestCase):
    def test_sixteen_colours_and_none_is_blue(self):
        self.assertEqual(len(set(RETRO_PALETTE)), 16)
        for red, green, blue in RETRO_PALETTE:
            self.assertFalse(blue > 1.2 * max(red, green) and blue > 150)

    def test_a_maze_keeps_its_colour_and_mazes_differ(self):
        self.assertEqual(maze_color(7), maze_color(7))
        self.assertGreater(len({maze_color(seed) for seed in range(200)}), 12)
        self.assertEqual(VersusMatch(simple_tree(), seed=7).wall_color, maze_color(7))


if __name__ == "__main__":
    unittest.main()
