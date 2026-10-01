#!/usr/bin/env python3
"""Generate a new Pac-Man board layout using the random-fill maze algorithm.

Algorithm source inspiration:
https://github.com/shaunlebron/pacman-mazegen (randomfill/walls.py)
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

try:
    from .board import Board
    from .render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, write_board
except ImportError:  # Direct execution: python PacmanGame/generate_maze_layout.py
    from board import Board
    from render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, write_board


SCRIPT_DIR = Path(__file__).resolve().parent

WIDTH = 28
HEIGHT = 31
HALF_WIDTH = 16

PLAYER_POS = (13, 23)
INKY_POS = (12, 14)
BLINKY_POS = (13, 11)
PINKY_POS = (13, 14)
CLYDE_POS = (15, 14)

GATE_TILES = {(13, 12), (14, 12)}
NO_PELLET_TILES = {
    PLAYER_POS,
    INKY_POS,
    BLINKY_POS,
    PINKY_POS,
    CLYDE_POS,
    *GATE_TILES,
}

# Keep the ghost house stable so spawn points remain valid.
GHOST_HOUSE_TEMPLATE = [
    "WWWGGWWW",
    "W......W",
    "W......W",
    "W......W",
    "WWWWWWWW",
]


def _format_map_str(tiles: str, separator: str) -> str:
    return separator.join(line.strip() for line in tiles.splitlines() if line.strip())


@dataclass
class RandomFillHalfMap:
    width: int
    height: int
    tiles: list[str]
    rng: random.Random

    @classmethod
    def from_template(cls, width: int, height: int, tile_str: str, rng: random.Random) -> "RandomFillHalfMap":
        tiles = list(_format_map_str(tile_str, ""))
        if len(tiles) != width * height:
            raise ValueError("half-map template size mismatch")
        return cls(width=width, height=height, tiles=tiles, rng=rng)

    def _xy_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _get_tile(self, x: int, y: int) -> str | None:
        if not self._xy_valid(x, y):
            return None
        return self.tiles[x + y * self.width]

    def _set_wall_tile(self, x: int, y: int) -> None:
        if self._xy_valid(x, y):
            self.tiles[x + y * self.width] = "|"

    def _is_wall_block_filled(self, x: int, y: int) -> bool:
        for dy in range(1, 3):
            for dx in range(1, 3):
                if self._get_tile(x + dx, y + dy) != "|":
                    return False
        return True

    def _add_wall_block(self, x: int, y: int) -> None:
        self._set_wall_tile(x + 1, y + 1)
        self._set_wall_tile(x + 2, y + 1)
        self._set_wall_tile(x + 1, y + 2)
        self._set_wall_tile(x + 2, y + 2)

    def _can_new_block_fit(self, x: int, y: int) -> bool:
        if not (self._xy_valid(x, y) and self._xy_valid(x + 3, y + 3)):
            return False
        for y0 in range(y, y + 4):
            for x0 in range(x, x + 4):
                if self._get_tile(x0, y0) != ".":
                    return False
        return True

    def _update_pos_list(self) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self._can_new_block_fit(x, y):
                    positions.append((x, y))
        return positions

    def _add_connection(
        self,
        pos_list: set[tuple[int, int]],
        connections: dict[tuple[int, int], list[tuple[int, int]]],
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> None:
        def connect(x0: int, y0: int) -> None:
            src = (x, y)
            dest = (x0, y0)
            if dest not in pos_list:
                return
            connections.setdefault(dest, []).append(src)

        if (x, y) not in pos_list:
            return
        connect(x + dx, y + dy)
        connect(x + 2 * dx, y + 2 * dy)
        if (x - dy, y - dx) not in pos_list:
            connect(x + dx - dy, y + dy - dx)
        if (x + dy, y + dx) not in pos_list:
            connect(x + dx + dy, y + dy + dx)
        if (x + dx - dy, y + dy - dx) not in pos_list:
            connect(x + 2 * dx - dy, y + 2 * dy - dx)
        if (x + dx + dy, y + dy + dx) not in pos_list:
            connect(x + 2 * dx + dy, y + 2 * dy + dx)

    def _update_connections(
        self,
        pos_list: list[tuple[int, int]],
    ) -> dict[tuple[int, int], list[tuple[int, int]]]:
        positions = set(pos_list)
        connections: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in positions:
                    continue
                if any(self._get_tile(x - 1, y + y0) == "|" for y0 in range(4)):
                    self._add_connection(positions, connections, x, y, 1, 0)
                if any(self._get_tile(x + 4, y + y0) == "|" for y0 in range(4)):
                    self._add_connection(positions, connections, x, y, -1, 0)
                if any(self._get_tile(x + x0, y - 1) == "|" for x0 in range(4)):
                    self._add_connection(positions, connections, x, y, 0, 1)
                if any(self._get_tile(x + x0, y + 4) == "|" for x0 in range(4)):
                    self._add_connection(positions, connections, x, y, 0, -1)
        return connections

    def _expand_wall(
        self,
        x: int,
        y: int,
        connections: dict[tuple[int, int], list[tuple[int, int]]],
    ) -> int:
        visited: set[tuple[int, int]] = set()

        def expand(x0: int, y0: int) -> int:
            src = (x0, y0)
            if src in visited:
                return 0
            visited.add(src)
            count = 0
            for target_x, target_y in connections.get(src, []):
                if not self._is_wall_block_filled(target_x, target_y):
                    count += 1
                    self._add_wall_block(target_x, target_y)
                count += expand(target_x, target_y)
            return count

        return expand(x, y)

    def add_wall_obstacle(self, extend: bool = True) -> bool:
        pos_list = self._update_pos_list()
        if not pos_list:
            return False
        connections = self._update_connections(pos_list)

        x, y = self.rng.choice(pos_list)
        self._add_wall_block(x, y)

        count = self._expand_wall(x, y, connections)

        if extend:
            max_blocks = 4
            turn = False
            turn_blocks = max_blocks
            if self.rng.random() <= 0.35:
                turn_blocks = 4
                max_blocks += turn_blocks

            dx, dy = self.rng.choice(((0, -1), (0, 1), (1, 0), (-1, 0)))
            original_direction = (dx, dy)
            positions = set(pos_list)

            i = 0
            while count < max_blocks:
                x0 = x + dx * i
                y0 = y + dy * i
                if ((not turn and count >= turn_blocks) or (x0, y0) not in positions):
                    turn = True
                    dx, dy = -dy, dx
                    i = 1
                    if (dx, dy) == original_direction:
                        break
                    continue

                if not self._is_wall_block_filled(x0, y0):
                    self._add_wall_block(x0, y0)
                    # Recompute dependencies after each placement to match the
                    # original fill behavior in dynamic neighborhoods.
                    pos_list = self._update_pos_list()
                    connections = self._update_connections(pos_list)
                    count += 1 + self._expand_wall(x0, y0, connections)
                    positions = set(pos_list)
                i += 1

        return True

    def reflected_ascii_rows(self) -> list[str]:
        rows: list[str] = []
        for y in range(self.height):
            line = "".join(self.tiles[y * self.width : (y + 1) * self.width])
            left = line[:14]
            rows.append(left + left[::-1])
        return rows


def _default_half_template() -> str:
    return """
        ||||||||||||||||
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |.........||||||
        |.........||||||
        |.........||||||
        |.........||||||
        |.........||||||
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        |...............
        ||||||||||||||||
    """


def _apply_ghost_house_template(tiles: list[list[int]]) -> None:
    top = 12
    left = 10
    for dy, row in enumerate(GHOST_HOUSE_TEMPLATE):
        for dx, char in enumerate(row):
            x = left + dx
            y = top + dy
            if char == "W":
                tiles[y][x] = WALL
            elif char == "G":
                tiles[y][x] = GHOST_GATE
            else:
                tiles[y][x] = 0


def _nearest_walkable(
    tiles: list[list[int]],
    origin: tuple[int, int],
    max_radius: int = 4,
) -> tuple[int, int] | None:
    ox, oy = origin
    for radius in range(max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = ox + dx
                y = oy + dy
                if 0 <= y < HEIGHT and 0 <= x < WIDTH and not (tiles[y][x] & (WALL | GHOST_GATE)):
                    return (x, y)
    return None


def _place_power_pellets(tiles: list[list[int]]) -> None:
    anchors = [(1, 3), (26, 3), (1, 23), (26, 23)]
    for anchor in anchors:
        point = _nearest_walkable(tiles, anchor)
        if point is None:
            continue
        x, y = point
        if tiles[y][x] & PELLET:
            tiles[y][x] = (tiles[y][x] & ~PELLET) | POWER_PELLET


def _carve_spawn_if_needed(tiles: list[list[int]], point: tuple[int, int]) -> None:
    x, y = point
    if tiles[y][x] & WALL:
        tiles[y][x] = 0


def _walkable_tiles(tiles: list[list[int]]) -> set[tuple[int, int]]:
    return {
        (x, y)
        for y in range(HEIGHT)
        for x in range(WIDTH)
        if not (tiles[y][x] & (WALL | GHOST_GATE))
    }


def _reachable_from(tiles: list[list[int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    walkable = _walkable_tiles(tiles)
    if start not in walkable:
        return set()
    seen: set[tuple[int, int]] = {start}
    pending: deque[tuple[int, int]] = deque([start])
    while pending:
        x, y = pending.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if dy == 0:
                nx %= WIDTH
            if 0 <= ny < HEIGHT and (nx, ny) in walkable and (nx, ny) not in seen:
                seen.add((nx, ny))
                pending.append((nx, ny))
    return seen


def _is_valid_maze(tiles: list[list[int]]) -> bool:
    walkable = _walkable_tiles(tiles)
    if not walkable:
        return False

    # Require every corridor tile to be reachable from the player.
    player_region = _reachable_from(tiles, PLAYER_POS)
    if player_region != walkable:
        return False

    # Ensure all spawn points are in the connected region.
    for point in (PLAYER_POS, INKY_POS, BLINKY_POS, PINKY_POS, CLYDE_POS):
        if point not in player_region:
            return False

    # Ensure at least one row acts as a wrap tunnel (left and right edge open).
    if not any((0, y) in walkable and (WIDTH - 1, y) in walkable for y in range(HEIGHT)):
        return False

    return True


def _generate_tiles(rng: random.Random) -> list[list[int]]:
    half_map = RandomFillHalfMap.from_template(
        width=HALF_WIDTH,
        height=HEIGHT,
        tile_str=_default_half_template(),
        rng=rng,
    )
    while half_map.add_wall_obstacle(extend=True):
        pass
    rows = half_map.reflected_ascii_rows()

    tiles = [[WALL if char == "|" else 0 for char in row] for row in rows]

    _apply_ghost_house_template(tiles)

    for point in (PLAYER_POS, INKY_POS, BLINKY_POS, PINKY_POS, CLYDE_POS):
        _carve_spawn_if_needed(tiles, point)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            if tiles[y][x] & (WALL | GHOST_GATE):
                continue
            if (x, y) in NO_PELLET_TILES:
                continue
            tiles[y][x] |= PELLET

    _place_power_pellets(tiles)

    return tiles


def generate_board(seed: int, max_attempts: int = 200) -> Board:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    rng = random.Random(seed)
    tiles: list[list[int]] | None = None
    for _ in range(max_attempts):
        candidate_rng = random.Random(rng.getrandbits(64))
        candidate_tiles = _generate_tiles(candidate_rng)
        if _is_valid_maze(candidate_tiles):
            tiles = candidate_tiles
            break

    if tiles is None:
        raise RuntimeError(
            f"unable to generate a valid maze in {max_attempts} attempts; "
            "try increasing --max-attempts or using a different --seed"
        )

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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="max generation retries to satisfy validity constraints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "generated_board.txt",
        help="output board-state path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    board = generate_board(args.seed, max_attempts=args.max_attempts)
    write_board(board, args.output)
    print(f"wrote maze board to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())