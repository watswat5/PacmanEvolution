#!/usr/bin/env python3
"""Batch-encode Pac-Man board states with a trained autoencoder encoder."""

from __future__ import annotations

import argparse
import importlib.util
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PACMAN_DIR = PROJECT_DIR / "PacmanGame"
sys.path.insert(0, str(PACMAN_DIR))


def load_render_board_module():
    module_path = PACMAN_DIR / "render_board.py"
    spec = importlib.util.spec_from_file_location("render_board", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load render_board from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_board = load_render_board_module()
PELLET = render_board.PELLET
POWER_PELLET = render_board.POWER_PELLET
WALL = render_board.WALL
read_board = render_board.read_board

COLLECTIBLE = PELLET | POWER_PELLET
DEFAULT_MODEL_DIR = SCRIPT_DIR / "training_32"


def find_board_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.glob("state_*.txt")))
        else:
            files.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(files):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_metadata(model_dir: Path) -> dict[str, np.ndarray]:
    metadata_path = model_dir / "board_metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        required = {"base_tiles", "collectible_positions", "collectible_values", "board_shape"}
        missing = required - set(metadata.files)
        if missing:
            raise ValueError(f"{metadata_path} is missing keys: {', '.join(sorted(missing))}")
        return {name: metadata[name] for name in metadata.files}


def encode_board(path: Path, metadata: dict[str, np.ndarray]) -> np.ndarray:
    board = read_board(path)
    tiles = np.asarray(board.tiles, dtype=np.uint8)
    base_tiles = np.asarray(metadata["base_tiles"], dtype=np.uint8)
    board_shape = tuple(int(value) for value in np.asarray(metadata["board_shape"]).tolist())
    if tiles.shape != board_shape:
        raise ValueError(f"{path} has shape {tiles.shape}; expected {board_shape}")
    if tiles.shape != base_tiles.shape:
        raise ValueError(f"{path} has shape {tiles.shape}; expected {base_tiles.shape}")

    static_mask = np.uint8(~COLLECTIBLE & 0xFF)
    if not np.array_equal(tiles & static_mask, base_tiles & static_mask):
        raise ValueError(f"{path} does not match the trained board layout")

    positions = np.asarray(metadata["collectible_positions"], dtype=np.int64)
    values = np.asarray(metadata["collectible_values"], dtype=np.uint8)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("collectible_positions must have shape (N, 2)")
    if len(positions) != len(values):
        raise ValueError("collectible_positions and collectible_values must have the same length")

    bits = (tiles[positions[:, 0], positions[:, 1]] & values) == values
    return bits.astype(np.float32)


def extract_passthrough(board) -> dict[str, object]:
    return {
        "player_pos": board.player_pos,
        "powerup_timer": board.player_powerup_turns,
        "game_over": board.game_over,
        "inky_pos": board.inky_pos,
        "blinky_pos": board.blinky_pos,
        "pinky_pos": board.pinky_pos,
        "clyde_pos": board.clyde_pos,
        "up": board.up,
        "down": board.down,
        "left": board.left,
        "right": board.right,
    }


def encode_files(files: list[Path], model_dir: Path, batch_size: int) -> tuple[list[Path], np.ndarray]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            f"TensorFlow is not installed. Run: python3 -m pip install -r {SCRIPT_DIR / 'requirements.txt'}"
        ) from exc

    metadata = load_metadata(model_dir)
    encoder_path = model_dir / "encoder.keras"
    if not encoder_path.exists():
        raise FileNotFoundError(f"missing encoder model: {encoder_path}")

    encoder = tf.keras.models.load_model(encoder_path)
    expected_input = int(encoder.input_shape[-1])
    expected_output = int(encoder.output_shape[-1])

    samples: list[np.ndarray] = []
    ordered_files: list[Path] = []
    passthrough_rows: list[dict[str, object]] = []
    for path in files:
        board = read_board(path)
        samples.append(encode_board(path, metadata))
        ordered_files.append(path)
        passthrough_rows.append(extract_passthrough(board))

    if not samples:
        raise ValueError("no board files were provided")

    sample_width = samples[0].shape[0]
    if sample_width != expected_input:
        raise ValueError(
            f"encoder expects {expected_input} collectible inputs, but metadata produced {sample_width}"
        )

    encoded_batches: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        batch = np.stack(samples[start : start + batch_size], axis=0)
        encoded = encoder.predict(batch, batch_size=batch_size, verbose=0)
        if encoded.shape[1] != expected_output:
            raise ValueError(
                f"encoder output width {encoded.shape[1]} does not match expected {expected_output}"
            )
        encoded_batches.append(encoded.astype(np.float32, copy=False))

    return ordered_files, passthrough_rows, np.concatenate(encoded_batches, axis=0)


def write_output(
    output_path: Path,
    files: list[Path],
    passthrough_rows: list[dict[str, object]],
    latents: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        fieldnames = [
            "file",
            "player_pos_x",
            "player_pos_y",
            "powerup_timer",
            "game_over",
            "inky_pos_x",
            "inky_pos_y",
            "blinky_pos_x",
            "blinky_pos_y",
            "pinky_pos_x",
            "pinky_pos_y",
            "clyde_pos_x",
            "clyde_pos_y",
            "up",
            "down",
            "left",
            "right",
        ] + [f"latent_{index}" for index in range(latents.shape[1])]
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for path, passthrough, vector in zip(files, passthrough_rows, latents):
                row = {
                    "file": str(path),
                    "player_pos_x": passthrough["player_pos"][0],
                    "player_pos_y": passthrough["player_pos"][1],
                    "powerup_timer": passthrough["powerup_timer"],
                    "game_over": int(bool(passthrough["game_over"])),
                    "inky_pos_x": passthrough["inky_pos"][0],
                    "inky_pos_y": passthrough["inky_pos"][1],
                    "blinky_pos_x": passthrough["blinky_pos"][0],
                    "blinky_pos_y": passthrough["blinky_pos"][1],
                    "pinky_pos_x": passthrough["pinky_pos"][0],
                    "pinky_pos_y": passthrough["pinky_pos"][1],
                    "clyde_pos_x": passthrough["clyde_pos"][0],
                    "clyde_pos_y": passthrough["clyde_pos"][1],
                    "up": int(bool(passthrough["up"])),
                    "down": int(bool(passthrough["down"])),
                    "left": int(bool(passthrough["left"])),
                    "right": int(bool(passthrough["right"])),
                }
                row.update({f"latent_{index}": float(value) for index, value in enumerate(vector)})
                writer.writerow(row)
        return

    np.savez_compressed(
        output_path,
        files=np.asarray([str(path) for path in files]),
        player_pos=np.asarray([row["player_pos"] for row in passthrough_rows], dtype=np.int16),
        powerup_timer=np.asarray([row["powerup_timer"] for row in passthrough_rows], dtype=np.int32),
        game_over=np.asarray([row["game_over"] for row in passthrough_rows], dtype=np.bool_),
        inky_pos=np.asarray([row["inky_pos"] for row in passthrough_rows], dtype=np.int16),
        blinky_pos=np.asarray([row["blinky_pos"] for row in passthrough_rows], dtype=np.int16),
        pinky_pos=np.asarray([row["pinky_pos"] for row in passthrough_rows], dtype=np.int16),
        clyde_pos=np.asarray([row["clyde_pos"] for row in passthrough_rows], dtype=np.int16),
        up=np.asarray([row["up"] for row in passthrough_rows], dtype=np.bool_),
        down=np.asarray([row["down"] for row in passthrough_rows], dtype=np.bool_),
        left=np.asarray([row["left"] for row in passthrough_rows], dtype=np.bool_),
        right=np.asarray([row["right"] for row in passthrough_rows], dtype=np.bool_),
        latents=latents.astype(np.float32, copy=False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="board files or directories containing state_*.txt files",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="directory containing encoder.keras and board_metadata.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "encoded_boards.npz",
        help="output file (.npz or .csv)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    files = find_board_files(args.paths)
    if not files:
        raise SystemExit("no board files found")
    ordered_files, passthrough_rows, latents = encode_files(files, args.model_dir, args.batch_size)
    write_output(args.output, ordered_files, passthrough_rows, latents)
    print(
        f"Encoded {len(ordered_files)} boards to {args.output} "
        f"with latent width {latents.shape[1]}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
