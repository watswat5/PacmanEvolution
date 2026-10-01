#!/usr/bin/env python3
"""Generate Pac-Man boards with Shaun LeBron's Tetris-stacking model.

The source algorithm fills a 5x9 half-map from left to right with connected
polyominoes, expands that simple model to tile resolution, and mirrors it.
This port emits this project's standard 28x31 bit-flag board-state format.

Reference: https://github.com/shaunlebron/pacman-mazegen/tree/gh-pages/tetris
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .board import Board
    from .render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, write_board
except ImportError:  # Direct execution from PacmanGame/
    from board import Board
    from render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, write_board


SCRIPT_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT = 28, 31
MODEL_WIDTH, MODEL_HEIGHT = 5, 9

PLAYER_POS = (13, 23)
INKY_POS = (12, 14)
BLINKY_POS = (13, 11)
PINKY_POS = (13, 14)
CLYDE_POS = (15, 14)
SPAWNS = (PLAYER_POS, INKY_POS, BLINKY_POS, PINKY_POS, CLYDE_POS)
GATE_TILES = {(13, 12), (14, 12)}
GHOST_HOUSE_INTERIOR = {(x, y) for y in range(13, 16) for x in range(11, 17)}
NO_PELLETS = {*SPAWNS, *GATE_TILES, *GHOST_HOUSE_INTERIOR}

UP, RIGHT, DOWN, LEFT = range(4)
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class Cell:
    x: int
    y: int
    group: int = -1
    connections: set[int] = field(default_factory=set)


class TetrisHalfMap:
    """Low-resolution half-map made from Tetris-like wall pieces."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cells = [
            Cell(x, y)
            for y in range(MODEL_HEIGHT)
            for x in range(MODEL_WIDTH)
        ]

    def cell(self, x: int, y: int) -> Cell | None:
        if 0 <= x < MODEL_WIDTH and 0 <= y < MODEL_HEIGHT:
            return self.cells[x + y * MODEL_WIDTH]
        return None

    def _connect(self, a: Cell, b: Cell, direction: int) -> None:
        a.connections.add(direction)
        b.connections.add((direction + 2) % 4)

    def _open_directions(self, cell: Cell, previous: int | None, size: int) -> list[int]:
        result: list[int] = []
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            # The reference prevents straight triominoes and grows only into
            # cells whose left neighbor is already occupied.
            if size == 2 and previous is not None and direction in (previous, (previous + 2) % 4):
                continue
            other = self.cell(cell.x + dx, cell.y + dy)
            if other is None or other.group >= 0:
                continue
            left = self.cell(other.x - 1, other.y)
            if left is not None and left.group < 0:
                continue
            result.append(direction)
        return result

    def generate(self) -> None:
        group = 0
        stop_probabilities = (0.0, 0.0, 0.10, 0.50, 0.75, 1.0)
        while True:
            unfilled_x = next(
                (x for x in range(MODEL_WIDTH)
                 if any(self.cell(x, y).group < 0 for y in range(MODEL_HEIGHT))),
                None,
            )
            if unfilled_x is None:
                return
            starts = [
                self.cell(unfilled_x, y) for y in range(MODEL_HEIGHT)
                if self.cell(unfilled_x, y).group < 0
            ]
            current = first = self.rng.choice(starts)
            current.group = group
            size = 1
            previous: int | None = None

            # Edge pieces in the reference may remain singletons. Elsewhere,
            # grow a compact piece of two to five cells.
            if current.x < MODEL_WIDTH - 1:
                while size < 5:
                    choices = self._open_directions(current, previous, size)
                    if not choices:
                        break
                    direction = self.rng.choice(choices)
                    dx, dy = DIRECTIONS[direction]
                    other = self.cell(current.x + dx, current.y + dy)
                    assert other is not None
                    self._connect(current, other, direction)
                    other.group = group
                    current = other
                    previous = direction
                    size += 1
                    if first.x == 0 and size == 3:
                        break
                    if size >= 2 and self.rng.random() <= stop_probabilities[size]:
                        break
            group += 1

    def render(self) -> list[list[int]]:
        """Direct port of mapgen.js ``getTiles`` path/wall expansion."""
        subrows = MODEL_HEIGHT * 3 + 1 + 3  # 31
        subcols = MODEL_WIDTH * 3 - 1 + 2  # 16 (half plus margins)
        midcols = subcols - 2
        chars = [["_" for _ in range(WIDTH)] for _ in range(subrows)]
        tile_cells: list[list[Cell | None]] = [
            [None for _ in range(subcols)] for _ in range(subrows)
        ]

        def full_xs(x: int) -> tuple[int, int]:
            shifted = x - 2
            return midcols + shifted, midcols - 1 - shifted

        def set_tile(x: int, y: int, value: str) -> None:
            if 0 <= x < subcols and 0 <= y < subrows:
                first, second = full_xs(x)
                chars[y][first] = chars[y][second] = value

        def get_tile(x: int, y: int) -> str | None:
            if not (0 <= x < subcols and 0 <= y < subrows):
                return None
            return chars[y][full_xs(x)[0]]

        def get_cell(x: int, y: int) -> Cell | None:
            if not (0 <= x < subcols and 0 <= y < subrows):
                return None
            return tile_cells[y][x]

        # The reference expands one row from 3 to 4 tiles and contracts one
        # column from 3 to 2. Choosing the same row/column across the half-map
        # is the simplest legal form of its resize-coordinate transform.
        for cell in self.cells:
            final_x = cell.x * 3
            final_y = cell.y * 3
            final_w = 2 if cell.x == MODEL_WIDTH - 1 else 3
            final_h = 4 if cell.y == 0 else 3
            if cell.y > 0:
                final_y += 1
            for y in range(final_y + 1, final_y + 1 + final_h):
                for x in range(final_x, final_x + final_w):
                    tile_cells[y][x] = cell

        # Literal translation of the source boundary-to-path pass.
        for y in range(subrows):
            for x in range(subcols):
                cell = get_cell(x, y)
                left = get_cell(x - 1, y)
                above = get_cell(x, y - 1)
                if cell is not None:
                    if (
                        (left is not None and cell.group != left.group)
                        or (above is not None and cell.group != above.group)
                        or (above is None and UP not in cell.connections)
                    ):
                        set_tile(x, y, ".")
                elif (
                    (left is not None and (
                        RIGHT not in left.connections or get_tile(x - 1, y) == "."
                    ))
                    or (above is not None and (
                        DOWN not in above.connections or get_tile(x, y - 1) == "."
                    ))
                ):
                    set_tile(x, y, ".")
                if (
                    get_tile(x - 1, y) == "."
                    and get_tile(x, y - 1) == "."
                    and get_tile(x - 1, y - 1) == "_"
                ):
                    set_tile(x, y, ".")

        # Extend one eligible path through both horizontal edges as a tunnel.
        tunnel_candidates = [
            cell for cell in self.cells
            if cell.x == MODEL_WIDTH - 1
            and 1 < cell.y < MODEL_HEIGHT - 2
            and UP not in cell.connections
        ]
        if not tunnel_candidates:
            raise ValueError("low-resolution model has no tunnel candidate")
        tunnel = self.rng.choice(tunnel_candidates)
        tunnel_y = tunnel.y * 3 + (2 if tunnel.y > 0 else 1)
        set_tile(subcols - 1, tunnel_y, ".")
        set_tile(subcols - 2, tunnel_y, ".")

        # Any blank sharing a vertex with a path becomes a wall, as in JS.
        for y in range(subrows):
            for x in range(subcols):
                if get_tile(x, y) == ".":
                    continue
                if any(
                    get_tile(x + dx, y + dy) == "."
                    for dx, dy in (
                        (-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (1, -1), (1, 1), (-1, 1),
                    )
                ):
                    set_tile(x, y, "|")

        return [
            [0 if char == "." else WALL for char in row]
            for row in chars
        ]


def _apply_ghost_house(tiles: list[list[int]]) -> None:
    template = ("WWWGGWWW", "W......W", "W......W", "W......W", "WWWWWWWW")
    for dy, row in enumerate(template):
        for dx, char in enumerate(row):
            tiles[12 + dy][10 + dx] = {"W": WALL, "G": GHOST_GATE, ".": 0}[char]


def _walkable(tiles: list[list[int]]) -> set[tuple[int, int]]:
    return {
        (x, y) for y, row in enumerate(tiles) for x, value in enumerate(row)
        if not value & (WALL | GHOST_GATE)
    }


def _reachable(tiles: list[list[int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    allowed = _walkable(tiles)
    seen = {start} if start in allowed else set()
    queue = deque(seen)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = (x + dx) % WIDTH, y + dy
            point = (nx, ny)
            if 0 <= ny < HEIGHT and point in allowed and point not in seen:
                seen.add(point)
                queue.append(point)
    return seen


def _valid(tiles: list[list[int]]) -> bool:
    allowed = _walkable(tiles)
    return (
        all(point in allowed for point in SPAWNS)
        and _reachable(tiles, PLAYER_POS) == allowed - GHOST_HOUSE_INTERIOR
        and any((0, y) in allowed and (WIDTH - 1, y) in allowed for y in range(HEIGHT))
        and not _wide_corridors(tiles)
    )


def _wide_corridors(tiles: list[list[int]]) -> list[tuple[int, int]]:
    """Return top-left coordinates of illegal 2x2 player-open regions."""
    violations = []
    for y in range(HEIGHT - 1):
        for x in range(WIDTH):
            right = (x + 1) % WIDTH
            square = {(x, y), (right, y), (x, y + 1), (right, y + 1)}
            if square <= GHOST_HOUSE_INTERIOR:
                continue
            if all(not tiles[py][px] & (WALL | GHOST_GATE) for px, py in square):
                violations.append((x, y))
    return violations


def _decorate(tiles: list[list[int]]) -> None:
    _apply_ghost_house(tiles)
    for x, y in SPAWNS:
        tiles[y][x] = 0
    for y, row in enumerate(tiles):
        for x, value in enumerate(row):
            if not value & (WALL | GHOST_GATE) and (x, y) not in NO_PELLETS:
                tiles[y][x] |= PELLET
    for x, y in ((1, 3), (26, 3), (1, 23), (26, 23)):
        candidates = sorted(
            _walkable(tiles),
            key=lambda point: abs(point[0] - x) + abs(point[1] - y),
        )
        if candidates:
            px, py = candidates[0]
            if tiles[py][px] & PELLET:
                tiles[py][px] = (tiles[py][px] & ~PELLET) | POWER_PELLET


def generate_board(seed: int = 0, max_attempts: int = 500) -> Board:
    """Return a deterministic, fully connected Tetris-stacking board."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    seeds = random.Random(seed)
    for _ in range(max_attempts):
        model = TetrisHalfMap(random.Random(seeds.getrandbits(64)))
        model.generate()
        tiles = model.render()
        _decorate(tiles)
        if _valid(tiles):
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
    raise RuntimeError(f"unable to generate a valid board in {max_attempts} attempts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "tetris_board.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    board = generate_board(args.seed, args.max_attempts)
    write_board(board, args.output)
    print(f"wrote {board.width}x{board.height} Tetris maze to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
