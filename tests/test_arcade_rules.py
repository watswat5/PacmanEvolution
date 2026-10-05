"""Tests for the arcade (level 1) ghost timing and behaviour."""

from pathlib import Path
import tempfile
import unittest

from PacmanGame import Board, PacmanGame, read_board, write_board
from PacmanGame.pacman import GHOST_KEYS, GHOST_SCORE, POWERUP_DURATION
from PacmanGame.render_board import PELLET, POWER_PELLET, WALL

BOARD_PATH = Path(__file__).parents[1] / "PacmanGame" / "test_board.txt"

# A loop with a two-tile dead end; an energizer sits at (2, 3).
LAYOUT = (
    "#######",
    "#.....#",
    "#.###.#",
    "#.o...#",
    "###.###",
    "###.###",
    "#######",
)


def small_board(ghosts):
    values = {"#": WALL, ".": PELLET, "o": POWER_PELLET}
    tiles = [[values[c] for c in row] for row in LAYOUT]
    inky, blinky, pinky, clyde = ghosts
    return Board(
        tiles=tiles, player_pos=(3, 3), player_powerup=False,
        player_powerup_turns=0, game_turn=0, score=0, game_over=False,
        inky_pos=inky, blinky_pos=blinky, pinky_pos=pinky, clyde_pos=clyde,
        pellets_remaining=sum(c in ".o" for row in LAYOUT for c in row),
    )


class ArcadeRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = PacmanGame(read_board(BOARD_PATH))

    def test_ghosts_are_slower_than_pacman(self) -> None:
        # Level 1: ghosts cover 75% to Pac-Man's 80%, so 15 tiles per 16 turns.
        moves, previous = 0, self.game.board.blinky_pos
        for _ in range(64):
            self.game.step("stay")
            moves += self.game.board.blinky_pos != previous
            previous = self.game.board.blinky_pos
        self.assertEqual(moves, 60)

    def test_scatter_comes_first_then_chase(self) -> None:
        self.assertEqual(self.game.ghost_mode, "scatter")
        for _ in range(50):
            self.game.step("stay")
        self.assertEqual(self.game.ghost_mode, "scatter")
        # Scattering Blinky heads for the top-right corner, away from Pac-Man.
        self.assertGreater(self.game.board.blinky_pos[0], 20)
        self.assertLess(self.game.board.blinky_pos[1], 6)
        for _ in range(6):
            self.game.step("stay")
        self.assertEqual(self.game.ghost_mode, "chase")

    def test_ghosts_leave_the_house_in_order(self) -> None:
        states = self.game._ghost_state
        self.assertEqual(states["blinky_pos"], "active")
        self.assertEqual(states["inky_pos"], "house")
        for _ in range(6):
            self.game.step("stay")
        self.assertEqual(states["pinky_pos"], "active")
        for _ in range(14):
            self.game.step("stay")
        # No dots eaten: Inky and Clyde wait for the four-second idle timer.
        self.assertEqual(states["inky_pos"], "house")
        self.assertEqual(states["clyde_pos"], "house")
        for _ in range(20):
            self.game.step("stay")
        self.assertNotEqual(states["inky_pos"], "house")
        self.assertEqual(states["clyde_pos"], "house")

    def test_energizer_frightens_and_eaten_ghost_returns_dangerous(self) -> None:
        game = PacmanGame(small_board(((1, 3), (5, 1), (1, 1), (5, 3))))
        game.step("left")  # eat the energizer at (2, 3)
        self.assertTrue(game.board.player_powerup)
        self.assertEqual(game.board.player_powerup_turns, POWERUP_DURATION)
        self.assertTrue(all(game._ghost_frightened.values()))
        self.assertEqual(set(game.board.ghost_scared.values()), {True})
        score = game.board.score
        eaten = [
            key for key in GHOST_KEYS if getattr(game.board, key) in {(1, 3), (2, 3)}
        ]
        self.assertEqual(eaten, ["inky_pos"])
        if game.board.inky_pos == (1, 3):
            game.step("left")
        self.assertFalse(game.game_over)
        self.assertGreaterEqual(game.board.score, score + GHOST_SCORE)
        self.assertEqual(game._ghost_state["inky_pos"], "house")
        self.assertFalse(game._ghost_frightened["inky_pos"])
        # The board itself records that only the eaten ghost is no longer scared.
        self.assertEqual(
            game.board.ghost_scared,
            {"inky": False, "blinky": True, "pinky": True, "clyde": True},
        )
        self.assertTrue(game.board.player_powerup)
        self.assertEqual(game.get_distance_map("danger_ghost")[game.board.inky_pos], 0)

    def test_scared_status_round_trips_through_a_board_file(self) -> None:
        board = read_board(BOARD_PATH)
        self.assertEqual(set(board.ghost_scared.values()), {False})
        powered = board.copy(
            player_powerup=True, player_powerup_turns=10, blinky_scared=False
        )
        # Unspecified ghosts follow player_powerup; explicit values win.
        self.assertEqual(
            powered.ghost_scared,
            {"inky": True, "blinky": False, "pinky": True, "clyde": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.txt"
            write_board(powered, path)
            loaded = read_board(path)
        self.assertEqual(loaded.ghost_scared, powered.ghost_scared)
        self.assertIs(loaded.blinky_scared, False)
        game = PacmanGame(loaded)
        self.assertFalse(game._ghost_frightened["blinky_pos"])
        self.assertTrue(game._ghost_frightened["pinky_pos"])

    def test_fright_wears_off(self) -> None:
        board = small_board(((1, 1), (5, 1), (3, 1), (5, 3)))
        game = PacmanGame(board.copy(player_powerup=True, player_powerup_turns=3))
        self.assertTrue(all(game._ghost_frightened.values()))
        for _ in range(4):
            game.step("stay")
        self.assertFalse(game.board.player_powerup)
        self.assertFalse(any(game._ghost_frightened.values()))


if __name__ == "__main__":
    unittest.main()
