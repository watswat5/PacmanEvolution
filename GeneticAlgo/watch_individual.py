#!/usr/bin/env python3
"""Watch or record a saved GP individual playing Pac-Man."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path
from typing import BinaryIO

from PacmanGame import PacmanGame, read_board, render_board

from .gp_core import build_primitive_set, evaluate_individual_move, load_individual
from .train_pacman import (
    DEFAULT_INITIAL_BOARD,
    SUPPORTED_LATENT_DIMS,
    EncoderBundle,
    encode_board,
    terminal_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("individual", type=Path, help="saved DEAP individual (.pkl)")
    parser.add_argument(
        "--board",
        type=Path,
        default=DEFAULT_INITIAL_BOARD,
        help="initial board state (default: PacmanGame/test_board.txt)",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=16, choices=SUPPORTED_LATENT_DIMS
    )
    parser.add_argument("--fps", type=float, default=10.0, help="moves per second")
    parser.add_argument("--scale", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening a window (requires --video)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        metavar="PATH",
        help="save all frames as a video; implies --headless",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-tie-break",
        action="store_true",
        help="randomly choose among equally scored legal actions",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.headless and args.video is None:
        parser.error("--headless requires --video PATH")
    return args


def rendered_frame(game: PacmanGame, scale: int):
    image = render_board(game.board).convert("RGB")
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale), resample=0
        )
    return image


def open_video_writer(
    path: Path, width: int, height: int, fps: float
) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ]
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise SystemExit("Video recording requires ffmpeg on PATH") from exc


def write_frame(stream: BinaryIO, image: object) -> None:
    stream.write(image.tobytes())


def main() -> int:
    args = parse_args()
    individual = load_individual(args.individual)
    encoder_bundle = EncoderBundle.load(args.latent_dim)
    game = PacmanGame(read_board(args.board))
    terminal_names = [
        "player_x", "player_y", "powerup_timer",
        "player_up", "player_down", "player_left", "player_right",
        "inky_x", "inky_y", "blinky_x", "blinky_y",
        "pinky_x", "pinky_y", "clyde_x", "clyde_y",
        "up", "down", "left", "right",
    ] + [f"z{index}" for index in range(args.latent_dim)]
    function_set = build_primitive_set(terminal_names)
    rng = random.Random(args.seed)
    frame_delay_ms = max(1, round(1_000 / args.fps))

    if args.video is not None:
        initial_frame = rendered_frame(game, args.scale)
        writer = open_video_writer(
            args.video, initial_frame.width, initial_frame.height, args.fps
        )
        assert writer.stdin is not None
        steps = 0
        try:
            write_frame(writer.stdin, initial_frame)
            while not (game.game_over or game.won) and steps < args.max_steps:
                latent = encode_board(
                    game.board, encoder_bundle.encoder, encoder_bundle.metadata
                )
                move, _ = evaluate_individual_move(
                    individual, terminal_values(game, latent), game.legal_moves(),
                    function_set, rng, random_tie_break=args.random_tie_break,
                )
                game.step(move or "stay")
                steps += 1
                write_frame(writer.stdin, rendered_frame(game, args.scale))
        except BrokenPipeError as exc:
            writer.wait()
            raise SystemExit("ffmpeg failed while writing the video") from exc
        finally:
            if not writer.stdin.closed:
                writer.stdin.close()
        return_code = writer.wait()
        if return_code != 0:
            raise SystemExit(f"ffmpeg exited with status {return_code}")
        print(
            f"Saved {steps + 1} frames ({steps} steps) to {args.video} "
            f"at {args.fps:g} fps"
        )
        return 0

    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError as exc:
        raise SystemExit(f"Playback requires Tkinter and Pillow: {exc}") from exc

    root = tk.Tk()
    root.title(f"Pac-Man GP Playback — {args.individual.name}")
    image_label = tk.Label(root, background="#202428")
    image_label.pack()
    status_label = tk.Label(
        root,
        background="#202428",
        foreground="white",
        pady=6,
        justify="left",
    )
    status_label.pack(fill="x")

    state = {"photo": None, "paused": False, "steps": 0, "move": "starting"}

    def redraw(scores: dict[str, float] | None = None) -> None:
        image = rendered_frame(game, args.scale)
        state["photo"] = ImageTk.PhotoImage(image)
        image_label.configure(image=state["photo"])

        if game.won:
            outcome = "YOU WIN"
        elif game.game_over:
            outcome = "GAME OVER"
        elif state["steps"] >= args.max_steps:
            outcome = "STEP LIMIT"
        elif state["paused"]:
            outcome = "PAUSED"
        else:
            outcome = "RUNNING"
        score_text = ""
        if scores is not None:
            score_text = " • " + " ".join(
                f"{action[0].upper()}={value:.2f}"
                for action, value in scores.items()
            )
        status_label.configure(
            text=(
                f"{outcome} • Score {game.board.score} • Turn {game.turn} "
                f"• Move {state['move']}{score_text}\n"
                "Space: pause/resume • Esc: quit"
            )
        )

    def advance() -> None:
        if not state["paused"] and not (game.game_over or game.won):
            if state["steps"] < args.max_steps:
                latent = encode_board(
                    game.board, encoder_bundle.encoder, encoder_bundle.metadata
                )
                move, scores = evaluate_individual_move(
                    individual,
                    terminal_values(game, latent),
                    game.legal_moves(),
                    function_set,
                    rng,
                    random_tie_break=args.random_tie_break,
                )
                state["move"] = move or "stay"
                game.step(move or "stay")
                state["steps"] += 1
                redraw(scores)
        if not (game.game_over or game.won) and state["steps"] < args.max_steps:
            root.after(frame_delay_ms, advance)

    def on_key(event: object) -> None:
        keysym = getattr(event, "keysym", "")
        if keysym == "Escape":
            root.destroy()
        elif keysym == "space":
            state["paused"] = not state["paused"]
            redraw()

    redraw()
    root.bind("<Key>", on_key)
    root.focus_force()
    root.after(frame_delay_ms, advance)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
