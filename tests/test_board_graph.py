"""Tests for graph-backed controller observations."""

from pathlib import Path
import unittest

from PacmanGame import PacmanGame, precompute_board_graph, read_board
from PacmanGame.render_board import WALL


BOARD_PATH = Path(__file__).parents[1] / "PacmanGame" / "test_board.txt"


class BoardGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = PacmanGame(read_board(BOARD_PATH))

    def test_observations_are_legal_complete_and_read_only(self) -> None:
        before = self.game.observation()
        observations = self.game.get_move_observations()

        self.assertEqual(
            set(observations),
            {move for move, legal in self.game.legal_moves().items() if legal},
        )
        self.assertEqual(before, self.game.observation())
        expected_features = {
            "pellet_here", "pellet_distance", "power_pellet_distance",
            "danger_ghost_distance", "edible_ghost_distance", "is_reverse",
            "is_forward", "candidate_degree", "pellets_ahead",
            "steps_to_junction", "pellet_center_direction",
        }
        for features in observations.values():
            self.assertEqual(set(features), expected_features)
            self.assertTrue(all(
                value >= 0
                for name, value in features.items()
                if name != "pellet_center_direction"
            ))
            self.assertGreaterEqual(features["pellet_center_direction"], -1.0)
            self.assertLessEqual(features["pellet_center_direction"], 1.0)
            self.assertLessEqual(features["candidate_degree"], 1.0)
            self.assertLessEqual(features["pellets_ahead"], 1.0)
            for name in (
                "pellet_distance", "power_pellet_distance",
                "danger_ghost_distance", "edible_ghost_distance",
                "steps_to_junction",
            ):
                self.assertLessEqual(features[name], 1.0)

    def test_static_graph_and_distance_maps_cannot_be_mutated(self) -> None:
        point = self.game.board.player_pos
        with self.assertRaises(TypeError):
            self.game.get_tile_neighbors()[point]["bad"] = point
        with self.assertRaises(TypeError):
            self.game.get_distance_map("pellet")[point] = 99

    def test_consumption_invalidates_only_the_pellet_map(self) -> None:
        pellet_map = self.game.get_distance_map("pellet")
        power_map = self.game.get_distance_map("power_pellet")
        destination = self.game.get_tile_neighbors()[self.game.board.player_pos]["left"]
        self.assertEqual(pellet_map[destination], 0)

        self.game.step("left")

        new_pellet_map = self.game.get_distance_map("pellet")
        self.assertIsNot(new_pellet_map, pellet_map)
        self.assertGreater(new_pellet_map[destination], 0)
        self.assertIs(self.game.get_distance_map("power_pellet"), power_map)

    def test_ghost_map_is_reused_until_ghost_state_changes(self) -> None:
        first = self.game.get_distance_map("danger_ghost")
        self.assertIs(first, self.game.get_distance_map("danger_ghost"))
        self.game.step("left")
        self.assertIsNot(first, self.game.get_distance_map("danger_ghost"))

    def test_reset_restores_consumed_targets(self) -> None:
        destination = self.game.get_tile_neighbors()[self.game.board.player_pos]["left"]
        self.game.step("left")
        self.assertGreater(self.game.get_distance_map("pellet")[destination], 0)
        self.game.reset()
        self.assertEqual(self.game.get_distance_map("pellet")[destination], 0)

    def test_static_precomputation_is_shared_between_games(self) -> None:
        board = read_board(BOARD_PATH)
        graph = precompute_board_graph(board)
        games = [PacmanGame(board.copy(), board_graph=graph) for _ in range(10)]
        self.assertTrue(
            all(game._observation_provider.static is graph for game in games)
        )
        self.assertIs(graph, precompute_board_graph(board.copy()))

    def test_rejects_graph_from_different_same_sized_layout(self) -> None:
        board = read_board(BOARD_PATH)
        graph = precompute_board_graph(board)
        changed = board.copy()
        x, y = changed.player_pos
        changed.tiles[y][x - 1] |= WALL
        with self.assertRaisesRegex(ValueError, "topology"):
            PacmanGame(changed, board_graph=graph)

    def test_board_derived_observations_stay_in_sync(self) -> None:
        game = PacmanGame(BOARD_PATH)
        self.assertEqual(game.board.legal_moves, game.legal_moves())
        destination = game.get_tile_neighbors()[game.board.player_pos]["left"]
        self.assertEqual(game.get_distance_map("pellet")[destination], 0)
        pellets_before = game.board.pellets_remaining

        game.step("left")

        self.assertEqual(game.board.legal_moves, game.legal_moves())
        self.assertEqual(game.board.pellets_remaining, pellets_before - 1)
        self.assertEqual(game.board.pellets_eaten, 1)
        self.assertGreater(game.get_distance_map("pellet")[destination], 0)

    def test_all_pairs_distance_matches_collectible_bfs(self) -> None:
        graph = self.game._observation_provider.static
        pellet_map = self.game.get_distance_map("pellet")
        pellets = self.game._observation_provider._pellets
        for point in graph.points:
            expected = min(
                (graph.distance(point, pellet) for pellet in pellets),
                default=graph.unreachable_distance,
            )
            self.assertEqual(pellet_map[point], expected)


if __name__ == "__main__":
    unittest.main()
