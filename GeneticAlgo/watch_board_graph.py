#!/usr/bin/env python3
import argparse, pickle, random
from pathlib import Path
from PacmanGame import PacmanGame, read_board
from .train_board_graph import build_primitive_set, choose_move
from .watch_individual import open_video_writer, rendered_frame, write_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("individual", type=Path)
    parser.add_argument(
        "--board",
        type=Path,
        required=True,
        help="board-state file paired with the saved individual",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-tie-break", action="store_true")
    args = parser.parse_args()
    with args.individual.open("rb") as handle:
        individual = pickle.load(handle)
    pset = build_primitive_set()
    game, rng = PacmanGame(read_board(args.board)), random.Random(args.seed)
    frame = rendered_frame(game, args.scale)
    writer = open_video_writer(args.video, frame.width, frame.height, args.fps)
    assert writer.stdin is not None
    steps = 0
    try:
        write_frame(writer.stdin, frame)
        while not (game.game_over or game.won) and steps < args.max_steps:
            move, _ = choose_move(individual, game.get_move_observations(), pset, game.player_direction, rng, random_tie_break=args.random_tie_break)
            game.step(move or "stay"); steps += 1
            write_frame(writer.stdin, rendered_frame(game, args.scale))
    finally:
        if not writer.stdin.closed: writer.stdin.close()
    code = writer.wait()
    if code: raise SystemExit(f"ffmpeg exited with status {code}")
    print(f"Saved {steps + 1} frames ({steps} steps) to {args.video} at {args.fps:g} fps")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
