"""Tests for the shared-tree move features and primitives."""

import unittest

from deap import creator, gp

from GeneticAlgo.gp_core import bounded_divide, if_lte
from GeneticAlgo.train_board_graph import (
    build_primitive_set,
    evaluate_population_lockstep,
)
from PacmanGame import Board, MOVE_FEATURE_NAMES, PacmanGame
from PacmanGame.board_graph import GHOST_NEAR_RADIUS, VISIT_WINDOW
from PacmanGame.render_board import PELLET, WALL

# A loop (rows 1-3) with a two-tile dead end hanging below (3, 3).
LAYOUT = (
    "#######",
    "#.....#",
    "#.###.#",
    "#.....#",
    "###.###",
    "###.###",
    "#######",
)


def make_board(player=(3, 3), ghosts=((1, 1), (5, 1), (1, 3), (5, 3)), **changes):
    tiles = [[WALL if c == "#" else PELLET for c in row] for row in LAYOUT]
    inky, blinky, pinky, clyde = ghosts
    board = Board(
        tiles=tiles, player_pos=player, player_powerup=False,
        player_powerup_turns=0, game_turn=0, score=0, game_over=False,
        inky_pos=inky, blinky_pos=blinky, pinky_pos=pinky, clyde_pos=clyde,
        pellets_remaining=sum(c == "." for row in LAYOUT for c in row),
    )
    return board.copy(**changes) if changes else board


class MoveFeatureTests(unittest.TestCase):
    def test_features_are_complete_and_bounded(self) -> None:
        observations = PacmanGame(make_board()).get_move_observations()
        self.assertEqual(set(observations), {"down", "left", "right"})
        for features in observations.values():
            self.assertEqual(tuple(features), MOVE_FEATURE_NAMES)
            for name, value in features.items():
                low = -1.0 if name == "pellet_center_direction" else 0.0
                self.assertGreaterEqual(value, low, name)
                self.assertLessEqual(value, 1.0, name)

    def test_dead_end_depth_and_escape_options(self) -> None:
        game = PacmanGame(make_board())
        graph = game._observation_provider.static
        scale = graph.distance_scale
        down = game.get_move_observations()["down"]
        self.assertAlmostEqual(down["dead_end_depth"], 2 / scale)
        self.assertEqual(down["escape_options"], 0.0)
        left = game.get_move_observations()["left"]
        self.assertEqual(left["dead_end_depth"], 0.0)
        # Leaving the dead end is not "entering a pocket".
        self.assertEqual(graph.pocket_depths[((3, 4), "up")], 0)
        self.assertEqual(graph.pocket_depths[((3, 4), "down")], 1)

    def test_multi_ghost_features(self) -> None:
        ghosts = ((1, 3), (5, 3), (1, 1), (5, 1))
        game = PacmanGame(make_board(ghosts=ghosts))
        features = game.get_move_observations()
        scale = game._observation_provider.distance_scale
        left = features["left"]  # candidate (2, 3)
        self.assertAlmostEqual(left["danger_ghost_distance"], 1 / scale)
        self.assertAlmostEqual(left["second_danger_ghost_distance"], 3 / scale)
        self.assertEqual(
            left["danger_ghosts_nearby"],
            sum(d <= GHOST_NEAR_RADIUS for d in (1, 3, 3, 5)) / 4,
        )
        self.assertEqual(left["edible_ghost_distance"], 1.0)
        self.assertEqual(left["frightened_time"], 0.0)

    def test_frightened_swaps_danger_for_edible(self) -> None:
        board = make_board(player_powerup=True, player_powerup_turns=20)
        left = PacmanGame(board).get_move_observations()["left"]
        self.assertEqual(left["danger_ghost_distance"], 1.0)
        self.assertEqual(left["second_danger_ghost_distance"], 1.0)
        self.assertEqual(left["danger_ghosts_nearby"], 0.0)
        self.assertLess(left["edible_ghost_distance"], 1.0)
        self.assertEqual(left["frightened_time"], 0.5)

    def test_recently_visited_marks_tiles_just_left(self) -> None:
        game = PacmanGame(make_board())
        self.assertEqual(game.get_move_observations()["left"]["recently_visited"], 0.0)
        game.step("left")
        observations = game.get_move_observations()
        self.assertEqual(observations["right"]["recently_visited"], 1.0)
        self.assertEqual(observations["left"]["recently_visited"], 0.0)
        # Recency fades linearly with age and vanishes after VISIT_WINDOW.
        provider = game._observation_provider
        provider.player_moved((1, 1))
        provider.player_moved((2, 1))
        self.assertEqual(provider._recently_visited((3, 3)), (VISIT_WINDOW - 2) / VISIT_WINDOW)
        for _ in range(VISIT_WINDOW):
            provider.player_moved((5, 1))
        self.assertEqual(provider._recently_visited((3, 3)), 0.0)
        game.reset()
        self.assertEqual(game.get_move_observations()["left"]["recently_visited"], 0.0)

    def test_primitives(self) -> None:
        self.assertEqual(if_lte(1, 2, 10, 20), 10)
        self.assertEqual(if_lte(3, 2, 10, 20), 20)
        # Closer (smaller) distances must never look worse.
        self.assertGreater(bounded_divide(1, 0.0), bounded_divide(1, 0.05))
        self.assertEqual(bounded_divide(1e9, 1), 1_000.0)
        pset = build_primitive_set()
        self.assertIn("if_lte", pset.mapping)
        self.assertNotIn("tan", pset.mapping)

    def test_revisit_penalty_charges_dithering(self) -> None:
        pset = build_primitive_set()
        # Prefer the reverse move every step: pure back-and-forth dithering.
        dither = creator.BoardGraphIndividual(
            gp.PrimitiveTree.from_string("is_reverse", pset)
        )
        board = make_board(ghosts=((1, 1), (2, 1), (4, 1), (5, 1)))
        common = dict(stagnation_penalty=0.0, stagnation_termination_steps=1_000)
        free, *_ = evaluate_population_lockstep(
            [dither], board, 10, pset, revisit_penalty=0.0, **common
        )
        charged, *_ = evaluate_population_lockstep(
            [dither], board, 10, pset, revisit_penalty=7.0, **common
        )
        self.assertLess(charged[0], free[0])
        self.assertEqual((free[0] - charged[0]) % 7.0, 0.0)


if __name__ == "__main__":
    unittest.main()
