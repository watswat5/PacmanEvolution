"""Regression tests for the Tetris-stacking Pac-Man board generator."""

import unittest

from PacmanGame.generate_tetris_board import (
    GHOST_HOUSE_INTERIOR,
    HEIGHT,
    WIDTH,
    _valid,
    _wide_corridors,
    generate_board,
)
from PacmanGame.render_board import PELLET, POWER_PELLET


class TetrisBoardGeneratorTests(unittest.TestCase):
    def test_seed_is_deterministic_and_changes_layout(self) -> None:
        first = generate_board(7)
        self.assertEqual(first.tiles, generate_board(7).tiles)
        self.assertNotEqual(first.tiles, generate_board(8).tiles)

    def test_generated_boards_are_playable_and_well_formed(self) -> None:
        for seed in range(10):
            with self.subTest(seed=seed):
                board = generate_board(seed)
                self.assertEqual(board.shape, (HEIGHT, WIDTH))
                self.assertTrue(_valid(board.tiles))
                self.assertEqual(_wide_corridors(board.tiles), [])
                actual_pellets = sum(
                    bool(value & (PELLET | POWER_PELLET))
                    for row in board.tiles
                    for value in row
                )
                self.assertEqual(board.pellets_remaining, actual_pellets)

    def test_ghost_house_contains_no_pellets(self) -> None:
        board = generate_board(0)
        self.assertTrue(all(
            not board.tiles[y][x] & (PELLET | POWER_PELLET)
            for x, y in GHOST_HOUSE_INTERIOR
        ))


if __name__ == "__main__":
    unittest.main()
