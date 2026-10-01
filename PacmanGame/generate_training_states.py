#!/usr/bin/env python3
"""Generate unique pellet layouts by walking through a Pac-Man board."""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

from render_board import (
    GHOST_GATE,
    PELLET,
    PLAYER_LOCATION,
    POWER_PELLET,
    WALL,
    GameBoard,
    read_board,
    write_board,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECTIBLES = PELLET | POWER_PELLET
DIRECTIONS = ((0, -1), (-1, 0), (0, 1), (1, 0))


def layout_key(tiles: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Return a hashable key containing only board data."""
    return tuple(tuple(row) for row in tiles)


def next_output_index(directory: Path) -> int:
    highest = -1
    for path in directory.glob("state_*.txt"):
        prefix, separator, number = path.stem.rpartition("_")
        if separator and prefix == "state" and number.isdigit():
            highest = max(highest, int(number))
    return highest + 1


def load_existing_layouts(directory: Path) -> set[tuple[tuple[int, ...], ...]]:
    layouts: set[tuple[tuple[int, ...], ...]] = set()
    for path in sorted(directory.glob("state_*.txt")):
        try:
            layouts.add(layout_key(read_board(path).tiles))
        except (OSError, ValueError) as exc:
            print(f"warning: ignoring unreadable state {path}: {exc}", file=sys.stderr)
    return layouts


def neighbors(
    point: tuple[int, int], tiles: list[list[int]]
) -> list[tuple[int, int]]:
    width = len(tiles[0])
    height = len(tiles)
    result: list[tuple[int, int]] = []
    for dx, dy in DIRECTIONS:
        x = point[0] + dx
        y = point[1] + dy
        if dy == 0:
            x %= width
        if 0 <= x < width and 0 <= y < height:
            if not tiles[y][x] & (WALL | GHOST_GATE):
                result.append((x, y))
    return result


def path_to_target(
    start: tuple[int, int],
    target: tuple[int, int],
    tiles: list[list[int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Find a shortest legal path, randomizing ties between equal routes."""
    pending = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while pending:
        point = pending.popleft()
        if point == target:
            break
        options = neighbors(point, tiles)
        rng.shuffle(options)
        for candidate in options:
            if candidate not in previous:
                previous[candidate] = point
                pending.append(candidate)

    if target not in previous:
        return []
    path: list[tuple[int, int]] = []
    point = target
    while point != start:
        path.append(point)
        parent = previous[point]
        assert parent is not None
        point = parent
    path.reverse()
    return path


def placeholder_state(source: GameBoard, tiles: list[list[int]]) -> GameBoard:
    """Build a serializable state whose headers are fixed placeholders."""
    return replace(
        source,
        tiles=[row.copy() for row in tiles],
        player_powerup=False,
        player_powerup_turns=0,
        game_turn=0,
        score=0,
        game_over=False,
    )


def generate(args: argparse.Namespace) -> int:
    source = read_board(args.board)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = load_existing_layouts(args.output_dir)
    output_index = next_output_index(args.output_dir)
    rng = random.Random(args.seed)
    generated = 0
    stagnant_episodes = 0

    def save_if_new(tiles: list[list[int]]) -> bool:
        nonlocal generated, output_index
        key = layout_key(tiles)
        if key in seen:
            return False
        output = args.output_dir / f"state_{output_index:06d}.txt"
        while output.exists():
            output_index += 1
            output = args.output_dir / f"state_{output_index:06d}.txt"
        write_board(placeholder_state(source, tiles), output)
        seen.add(key)
        generated += 1
        output_index += 1
        return True

    for episode in range(1, args.max_episodes + 1):
        if generated >= args.count:
            break
        tiles = [row.copy() for row in source.tiles]
        # Entity locations belong in the placeholder header, not board_data.
        for row in tiles:
            for index, value in enumerate(row):
                row[index] = value & ~PLAYER_LOCATION

        walker = source.player_pos
        added_this_episode = int(save_if_new(tiles))
        remaining = {
            (x, y)
            for y, row in enumerate(tiles)
            for x, value in enumerate(row)
            if value & COLLECTIBLES
        }

        while remaining and generated < args.count:
            target = rng.choice(sorted(remaining))
            path = path_to_target(walker, target, tiles, rng)
            if not path:
                remaining.remove(target)
                continue
            for walker in path:
                x, y = walker
                if tiles[y][x] & COLLECTIBLES:
                    tiles[y][x] &= ~COLLECTIBLES
                    remaining.discard(walker)
                    added_this_episode += int(save_if_new(tiles))
                    if generated >= args.count:
                        break

        if added_this_episode == 0:
            stagnant_episodes += 1
            if stagnant_episodes >= 25:
                break
        else:
            stagnant_episodes = 0

        if episode % 10 == 0 or generated >= args.count:
            print(f"episodes={episode} new_states={generated}/{args.count}")

    print(
        f"Generated {generated} unique layouts in {args.output_dir}; "
        f"next file index is {output_index}."
    )
    if generated < args.count:
        print(
            "warning: stopped before reaching the requested count; try a different "
            "--seed or increase --max-episodes",
            file=sys.stderr,
        )
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board",
        type=Path,
        default=SCRIPT_DIR / "test_board.txt",
        help="source board (default: test_board.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "training_states",
        help="training-state destination",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3000,
        help="number of new unique layouts to generate (default: 3000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max-episodes", type=int, default=1000)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
