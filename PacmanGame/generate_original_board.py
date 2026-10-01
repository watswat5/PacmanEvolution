#!/usr/bin/env python3
"""Write the original arcade Pac-Man maze in the project's board-state format.

The maze uses the same 28x31 grid and fixed positions as
``generate_tetris_board`` (ghost gate at (13, 12)-(14, 12), ghost house
interior x=11..16, y=13..15, Pac-Man at (13, 23)). That keeps it a drop-in
validation board for ``train_board_graph`` and the tests.

Legend: ``#`` wall, ``X`` solid filler outside the maze (also a wall), ``.``
pellet, ``o`` power pellet, ``-`` ghost gate, and a space for an open tile
with no pellet. Row 14 is the wrap-around tunnel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .board import Board
    from .generate_tetris_board import (
        BLINKY_POS, CLYDE_POS, HEIGHT, INKY_POS, PINKY_POS, PLAYER_POS, WIDTH,
    )
    from .render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, write_board
except ImportError:  # Direct execution from PacmanGame/
    from board import Board
    from generate_tetris_board import (
        BLINKY_POS, CLYDE_POS, HEIGHT, INKY_POS, PINKY_POS, PLAYER_POS, WIDTH,
    )
    from render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, write_board


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "test_board.txt"

ORIGINAL_LAYOUT = (
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "XXXXX#.##### ## #####.#XXXXX",
    "XXXXX#.##          ##.#XXXXX",
    "XXXXX#.## ###--### ##.#XXXXX",
    "######.## #      # ##.######",
    "      .   #      #   .      ",
    "######.## #      # ##.######",
    "XXXXX#.## ######## ##.#XXXXX",
    "XXXXX#.##          ##.#XXXXX",
    "XXXXX#.## ######## ##.#XXXXX",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#o..##.......  .......##..o#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
)

TILE_VALUES = {
    "#": WALL,
    "X": WALL,
    "-": GHOST_GATE,
    ".": PELLET,
    "o": POWER_PELLET,
    " ": 0,
}


def original_board() -> Board:
    """Return the original maze as a fresh starting state."""
    if len(ORIGINAL_LAYOUT) != HEIGHT or any(len(row) != WIDTH for row in ORIGINAL_LAYOUT):
        raise AssertionError("original layout must be 28x31")
    tiles = [[TILE_VALUES[char] for char in row] for row in ORIGINAL_LAYOUT]
    pellets = sum(bool(v & (PELLET | POWER_PELLET)) for row in tiles for v in row)
    return Board(
        tiles=tiles,
        player_pos=PLAYER_POS,
        player_powerup=False,
        player_powerup_turns=0,
        game_turn=0,
        score=0,
        game_over=False,
        inky_pos=INKY_POS,
        blinky_pos=BLINKY_POS,
        pinky_pos=PINKY_POS,
        clyde_pos=CLYDE_POS,
        player_left=True,
        pellets_remaining=pellets,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    board = original_board()
    # Route through the game once so the saved legal-move flags are correct.
    try:
        from .pacman import PacmanGame
    except ImportError:
        from pacman import PacmanGame
    PacmanGame(board).save(args.output)
    print(f"Wrote {args.output} ({board.pellets_remaining} pellets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
