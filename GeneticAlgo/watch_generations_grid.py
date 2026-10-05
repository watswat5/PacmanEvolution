#!/usr/bin/env python3
"""Render every generation's champion playing at once, as a grid video.

Each cell is one ``generation_NNNN_best.pkl`` from a run directory, all playing
the same board from the same start, in generation order (left to right, top to
bottom). A cell freezes and is tinted when its game ends:

    red     Pac-Man died
    green   the board was cleared
    amber   cut off for stagnation (no pellet for --stagnation-steps steps)

Cells still playing at --max-steps stay untinted. The final frame is also saved
as a PNG next to the video.

    python -m GeneticAlgo.watch_generations_grid runs/<run directory>
    python -m GeneticAlgo.watch_generations_grid runs/<run> --every 2 --fps 20

The games are simulated in parallel worker processes; drawing uses flat colored
blocks rather than the sprites, because a cell is only a few pixels per tile.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from PacmanGame import PacmanGame, read_board
from PacmanGame.render_board import PELLET, POWER_PELLET, WALL

from .train_board_graph import VALIDATION_BOARD_PATH, build_primitive_set, choose_move
from .watch_individual import open_video_writer

GHOSTS = ("blinky", "pinky", "inky", "clyde")
COLORS = {
    "floor": (0, 0, 0),
    "wall": (33, 33, 150),
    "pellet": (255, 214, 190),
    "player": (255, 235, 0),
    "blinky": (255, 40, 40),
    "pinky": (255, 150, 220),
    "inky": (0, 230, 230),
    "clyde": (255, 160, 40),
    "scared": (60, 90, 255),
    "gap": (24, 24, 24),
}
TINTS = {
    "died": (215, 35, 35),
    "cleared": (35, 180, 70),
    "stagnated": (225, 165, 30),
}
TINT_STRENGTH = 0.55
GAP = 2  # pixels between cells
HOLD_SECONDS = 3  # how long the final frame stays on screen


def play(task: tuple) -> dict:
    """Play one champion and return what the renderer needs for every step."""
    path, board_path, max_steps, stagnation_steps, seed = task
    with open(path, "rb") as handle:
        tree = pickle.load(handle)
    primitive_set = build_primitive_set()
    game = PacmanGame(read_board(Path(board_path)))
    rng = random.Random(seed)
    start_pellets = game.board.pellets_remaining

    def snapshot(eaten) -> tuple:
        board = game.board
        scared = board.ghost_scared
        return (
            board.player_pos,
            tuple(getattr(board, f"{name}_pos") for name in GHOSTS),
            tuple(scared[name] for name in GHOSTS),
            eaten,
        )

    steps = [snapshot(None)]
    scoreless = 0
    outcome = "running"
    while len(steps) <= max_steps:
        move, _ = choose_move(
            tree, game.get_move_observations(), primitive_set, game.player_direction, rng
        )
        eaten_before = game.board.pellets_eaten
        game.step(move or "stay")
        ate = game.board.pellets_eaten > eaten_before
        scoreless = 0 if ate else scoreless + 1
        steps.append(snapshot(game.board.player_pos if ate else None))
        if game.won:
            outcome = "cleared"
        elif game.game_over:
            outcome = "died"
        elif stagnation_steps and scoreless >= stagnation_steps:
            outcome = "stagnated"
        if outcome != "running":
            break
    return {
        "steps": steps,
        "outcome": outcome,
        "pellets": game.board.pellets_eaten / max(1, start_pellets),
    }


def block(image: np.ndarray, x: int, y: int, tile: int, color, inset: int = 0) -> None:
    image[y * tile + inset:(y + 1) * tile - inset, x * tile + inset:(x + 1) * tile - inset] = color


def maze_image(board, tile: int) -> np.ndarray:
    """Walls and every pellet of the starting board, at ``tile`` pixels per tile."""
    image = np.zeros((board.height * tile, board.width * tile, 3), dtype=np.uint8)
    pellet_inset = max(1, (tile - max(1, tile // 3)) // 2)
    for x, y, value in board.iter_tiles():
        if value & WALL:
            block(image, x, y, tile, COLORS["wall"])
        elif value & POWER_PELLET:
            block(image, x, y, tile, COLORS["pellet"], inset=max(0, tile // 6))
        elif value & PELLET:
            block(image, x, y, tile, COLORS["pellet"], inset=pellet_inset)
    return image


def draw_label(cell: np.ndarray, text: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(cell)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((1, 0), text)
    draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill=(0, 0, 0))
    draw.text((1, 0), text, fill=(255, 255, 255))
    cell[:] = np.asarray(image)


def generation_files(run: Path, every: int, limit: int | None) -> list[tuple[int, Path]]:
    found = []
    for path in run.glob("generation_*_best.pkl"):
        match = re.fullmatch(r"generation_(\d+)_best\.pkl", path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    found = found[::every]
    return found[:limit] if limit else found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="run directory with generation_NNNN_best.pkl files")
    parser.add_argument("--video", type=Path, help="output file (default: <run>/generation_grid.mp4)")
    parser.add_argument("--board", type=Path, default=VALIDATION_BOARD_PATH,
                        help="board every champion plays (default: the original maze)")
    parser.add_argument("--columns", type=int, help="cells per row (default: a square grid)")
    parser.add_argument("--every", type=int, default=1, help="use every Nth generation")
    parser.add_argument("--limit", type=int, help="use at most this many generations")
    parser.add_argument("--tile-size", type=int, default=6, help="pixels per maze tile (default: 6)")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stagnation-steps", type=int, default=50,
                        help="end a game after this many steps without a pellet; 0 disables (default: 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="simulation processes (default: all CPUs)")
    args = parser.parse_args()
    if args.every < 1 or args.tile_size < 3 or args.max_steps < 1:
        parser.error("--every and --max-steps must be positive and --tile-size at least 3")
    return args


def main() -> int:
    args = parse_args()
    files = generation_files(args.run, args.every, args.limit)
    if not files:
        raise SystemExit(f"No generation_NNNN_best.pkl files in {args.run}")
    video = args.video or args.run / "generation_grid.mp4"
    workers = args.workers or len(os.sched_getaffinity(0))
    tasks = [
        (str(path), str(args.board), args.max_steps, args.stagnation_steps, args.seed)
        for _, path in files
    ]
    print(f"Simulating {len(files)} champions on {args.board.name} with {workers} workers")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            games = list(executor.map(play, tasks, chunksize=1))
    else:
        games = [play(task) for task in tasks]

    board = read_board(args.board)
    tile = args.tile_size
    cell_h, cell_w = board.height * tile, board.width * tile
    columns = args.columns or math.ceil(math.sqrt(len(files)))
    rows = math.ceil(len(files) / columns)
    width = columns * cell_w + (columns + 1) * GAP
    height = rows * cell_h + (rows + 1) * GAP
    width += width % 2  # H.264 needs even dimensions
    height += height % 2

    # ``base`` holds every cell's maze with its remaining pellets; entities are
    # drawn on a copy each frame.
    base = np.full((height, width, 3), COLORS["gap"], dtype=np.uint8)
    maze = maze_image(board, tile)
    origins = []
    for index, (generation, _) in enumerate(files):
        row, column = divmod(index, columns)
        y0, x0 = GAP + row * (cell_h + GAP), GAP + column * (cell_w + GAP)
        origins.append((y0, x0))
        cell = base[y0:y0 + cell_h, x0:x0 + cell_w]
        cell[:] = maze
        draw_label(cell, str(generation))

    def draw_entities(target: np.ndarray, origin, step) -> None:
        y0, x0 = origin
        cell = target[y0:y0 + cell_h, x0:x0 + cell_w]
        player, ghosts, scared, _ = step
        for name, (x, y), is_scared in zip(GHOSTS, ghosts, scared):
            block(cell, x, y, tile, COLORS["scared" if is_scared else name])
        block(cell, player[0], player[1], tile, COLORS["player"])

    longest = max(len(game["steps"]) for game in games)
    writer = open_video_writer(video, width, height, args.fps)
    assert writer.stdin is not None
    frame = base
    try:
        for index in range(longest):
            for (generation, _), game, origin in zip(files, games, origins):
                steps = game["steps"]
                y0, x0 = origin
                cell = base[y0:y0 + cell_h, x0:x0 + cell_w]
                if index < len(steps) and steps[index][3] is not None:
                    x, y = steps[index][3]
                    block(cell, x, y, tile, COLORS["floor"])
                if index == len(steps) - 1 and game["outcome"] in TINTS:
                    # Freeze the finished cell with its last positions, tinted.
                    draw_entities(base, origin, steps[-1])
                    tint = np.array(TINTS[game["outcome"]], dtype=np.float32)
                    cell[:] = (cell * (1 - TINT_STRENGTH) + tint * TINT_STRENGTH).astype(np.uint8)
                    draw_label(cell, f"{generation}  {game['pellets']:.0%}")
            frame = base.copy()
            for game, origin in zip(games, origins):
                steps = game["steps"]
                if index < len(steps) - 1 or game["outcome"] not in TINTS:
                    draw_entities(frame, origin, steps[min(index, len(steps) - 1)])
            writer.stdin.write(frame.tobytes())
        for _ in range(int(HOLD_SECONDS * args.fps)):
            writer.stdin.write(frame.tobytes())
    finally:
        if not writer.stdin.closed:
            writer.stdin.close()
    if writer.wait():
        raise SystemExit("ffmpeg failed while writing the video")

    from PIL import Image

    poster = video.with_suffix(".png")
    Image.fromarray(frame).save(poster)
    counts = {name: sum(game["outcome"] == name for game in games)
              for name in ("cleared", "died", "stagnated", "running")}
    print(
        f"Saved {longest} frames, {columns}x{rows} cells, {width}x{height} px to {video}\n"
        f"Final frame: {poster}\n"
        f"Outcomes: {counts['cleared']} cleared, {counts['died']} died, "
        f"{counts['stagnated']} stagnated, {counts['running']} still playing at the step limit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
