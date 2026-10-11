"""Tests for the evasion move features (approaching, pincer, safe territory)."""

import unittest

from PacmanGame import EVASION_FEATURE_NAMES, MOVE_FEATURE_NAMES
from PacmanGame.board_graph import ObservationProvider

from tests.test_move_features import make_board

# Only pinky (1, 3) and clyde (5, 3) are dangerous; inky and blinky are
# frightened. Pac-Man stands at (3, 3) with moves left, right and down.
TWO_GHOSTS = dict(inky_scared=True, blinky_scared=True, pinky_scared=False, clyde_scared=False)


def features(ghost_directions=None, **changes):
    board = make_board(**{**TWO_GHOSTS, **changes})
    return ObservationProvider(board).for_legal_moves(board, "left", ghost_directions)


class EvasionFeatureTests(unittest.TestCase):
    def test_names_are_appended(self):
        self.assertEqual(MOVE_FEATURE_NAMES[19:22], EVASION_FEATURE_NAMES)

    def test_approaching_follows_heading(self):
        # Pinky heading right comes towards (2, 3); heading left (into the
        # corner) it can only turn up, away from it.
        towards = features({"pinky_pos": "right", "clyde_pos": "right"})
        away = features({"pinky_pos": "left", "clyde_pos": "right"})
        self.assertEqual(towards["left"]["danger_ghost_approaching"], 1.0)
        self.assertEqual(away["left"]["danger_ghost_approaching"], -1.0)

    def test_no_dangerous_ghost(self):
        result = features(pinky_scared=True, clyde_scared=True)
        for move in result.values():
            self.assertEqual(move["danger_ghost_approaching"], 0.0)
            self.assertEqual(move["pincer"], 0.0)
            self.assertEqual(move["safe_territory"], 1.0)

    def test_pincer_needs_a_ghost_closing_from_behind(self):
        # Moving left heads at pinky; clyde behind is closing (heading left)
        # or leaving (heading right, so it can only turn up).
        closing = features({"pinky_pos": "right", "clyde_pos": "left"})
        leaving = features({"pinky_pos": "right", "clyde_pos": "right"})
        self.assertEqual(closing["left"]["pincer"], 1.0)
        self.assertEqual(leaving["left"]["pincer"], 0.0)
        # The dead end below has both ghosts behind, none ahead.
        self.assertEqual(closing["down"]["pincer"], 0.0)

    def test_safe_territory(self):
        result = features()
        # Next to pinky, (2, 3) is reached at the same time as by the ghost.
        self.assertEqual(result["left"]["safe_territory"], 0.0)
        # The dead end: (3, 4) and (3, 5) are Pac-Man's, out of 14 tiles.
        self.assertAlmostEqual(result["down"]["safe_territory"], 2 / 14)
        for move in result.values():
            self.assertGreaterEqual(move["safe_territory"], 0.0)
            self.assertLessEqual(move["safe_territory"], 1.0)


if __name__ == "__main__":
    unittest.main()
