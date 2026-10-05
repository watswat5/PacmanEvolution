"""Shared board-state type used by the game, renderer, and training code."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator


Point = tuple[int, int]


@dataclass(frozen=True)
class Board:
    """Complete serializable state for one Pac-Man timestep.

    The dataclass is frozen so fields cannot be reassigned accidentally. Use
    ``copy()`` when a script needs an independently mutable tile grid.
    """

    tiles: list[list[int]]
    player_pos: Point
    player_powerup: bool
    player_powerup_turns: int
    game_turn: int
    score: int
    game_over: bool
    inky_pos: Point
    blinky_pos: Point
    pinky_pos: Point
    clyde_pos: Point
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    player_up: bool = False
    player_down: bool = False
    player_left: bool = True
    player_right: bool = False
    pellets_eaten: int = 0
    pellets_remaining: int = 0
    # Per-ghost frightened ("scared") status. ``None`` means unspecified, in
    # which case the ghost is scared exactly when ``player_powerup`` is set;
    # the game always stores explicit values, because a ghost that was eaten
    # and has returned is dangerous while the others are still scared.
    inky_scared: bool | None = None
    blinky_scared: bool | None = None
    pinky_scared: bool | None = None
    clyde_scared: bool | None = None

    @property
    def width(self) -> int:
        return len(self.tiles[0]) if self.tiles else 0

    @property
    def height(self) -> int:
        return len(self.tiles)

    @property
    def shape(self) -> tuple[int, int]:
        """Return board shape as (height, width)."""
        return self.height, self.width

    @property
    def ghost_positions(self) -> dict[str, Point]:
        return {
            "inky": self.inky_pos,
            "blinky": self.blinky_pos,
            "pinky": self.pinky_pos,
            "clyde": self.clyde_pos,
        }

    @property
    def ghost_scared(self) -> dict[str, bool]:
        """Whether each ghost can currently be eaten, with ``None`` resolved."""
        return {
            name: self.player_powerup if value is None else value
            for name, value in (
                ("inky", self.inky_scared),
                ("blinky", self.blinky_scared),
                ("pinky", self.pinky_scared),
                ("clyde", self.clyde_scared),
            )
        }

    @property
    def legal_moves(self) -> dict[str, bool]:
        return {
            "up": self.up,
            "down": self.down,
            "left": self.left,
            "right": self.right,
        }

    def iter_tiles(self) -> Iterator[tuple[int, int, int]]:
        """Yield (x, y, value) for every board tile."""
        for y, row in enumerate(self.tiles):
            for x, value in enumerate(row):
                yield x, y, value

    def copy(self, **changes: object) -> Board:
        """Return an independent state copy, optionally replacing fields."""
        replacement_tiles = changes.pop(
            "tiles", [row.copy() for row in self.tiles]
        )
        return replace(self, tiles=replacement_tiles, **changes)


# Descriptive and backward-compatible aliases.
BoardState = Board
GameBoard = Board


__all__ = ("Board", "BoardState", "GameBoard", "Point")
