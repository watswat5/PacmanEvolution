#!/usr/bin/env python3
"""Render an input board next to its encoded and decoded representations."""

from __future__ import annotations

import argparse
import importlib.util
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
GHOST_GATE = render_board.GHOST_GATE
PELLET = render_board.PELLET
POWER_PELLET = render_board.POWER_PELLET
WALL = render_board.WALL
read_board = render_board.read_board

COLLECTIBLE = PELLET | POWER_PELLET
MODEL_DIRS = {
    "training": SCRIPT_DIR / "training",
    "training_32": SCRIPT_DIR / "training_32",
    "training_8": SCRIPT_DIR / "training_8",
    "training_4": SCRIPT_DIR / "training_4",
}


def load_metadata(model_dir: Path) -> dict[str, np.ndarray]:
    metadata_path = model_dir / "board_metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    with np.load(metadata_path, allow_pickle=False) as metadata:
        required = {
            "base_tiles",
            "collectible_positions",
            "collectible_values",
            "board_shape",
        }
        missing = required - set(metadata.files)
        if missing:
            raise ValueError(f"{metadata_path} is missing keys: {', '.join(sorted(missing))}")
        return {name: metadata[name] for name in metadata.files}


def board_to_sample(board_path: Path, metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    board = read_board(board_path)
    tiles = np.asarray(board.tiles, dtype=np.uint8)

    board_shape = tuple(int(value) for value in np.asarray(metadata["board_shape"]).tolist())
    if tiles.shape != board_shape:
        raise ValueError(f"{board_path} has shape {tiles.shape}; expected {board_shape}")

    base_tiles = np.asarray(metadata["base_tiles"], dtype=np.uint8)
    static_mask = np.uint8(~COLLECTIBLE & 0xFF)
    if not np.array_equal(tiles & static_mask, base_tiles & static_mask):
        raise ValueError(f"{board_path} does not match the trained board layout")

    positions = np.asarray(metadata["collectible_positions"], dtype=np.int64)
    values = np.asarray(metadata["collectible_values"], dtype=np.uint8)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("collectible_positions must have shape (N, 2)")
    if len(positions) != len(values):
        raise ValueError("collectible_positions and collectible_values must have the same length")

    bits = (tiles[positions[:, 0], positions[:, 1]] & values) == values
    return tiles, bits.astype(np.float32)


def rebuild_board(bits: np.ndarray, metadata: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    base_tiles = np.asarray(metadata["base_tiles"], dtype=np.uint8)
    positions = np.asarray(metadata["collectible_positions"], dtype=np.int64)
    values = np.asarray(metadata["collectible_values"], dtype=np.uint8)

    board = base_tiles.copy()
    for present, (row, column), value in zip(bits >= threshold, positions, values):
        if present:
            board[row, column] |= value
    return board


def to_display_values(board: np.ndarray) -> np.ndarray:
    display = np.zeros_like(board, dtype=np.uint8)
    display[board & WALL != 0] = 1
    display[board & PELLET != 0] = 2
    display[board & POWER_PELLET != 0] = 3
    display[board & GHOST_GATE != 0] = 4
    return display


def render_comparison(
    output_path: Path,
    original_board: np.ndarray,
    latent: np.ndarray,
    decoded_board: np.ndarray,
    model_name: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    board_palette = ListedColormap(["black", "#1010ff", "#ffb7ae", "#ff7f9f", "#ffb7ff"])

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), squeeze=False)
    ax_original, ax_latent, ax_decoded = axes[0]

    ax_original.imshow(to_display_values(original_board), cmap=board_palette, vmin=0, vmax=4)
    ax_original.set_title("Original Board")
    ax_original.set_xticks([])
    ax_original.set_yticks([])

    latent_image = latent.reshape(1, -1)
    heatmap = ax_latent.imshow(latent_image, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax_latent.set_title(f"Encoded Latent ({latent.size})")
    ax_latent.set_yticks([])
    ax_latent.set_xlabel("Latent Index")
    figure.colorbar(heatmap, ax=ax_latent, fraction=0.046, pad=0.04)

    ax_decoded.imshow(to_display_values(decoded_board), cmap=board_palette, vmin=0, vmax=4)
    ax_decoded.set_title("Decoded Board")
    ax_decoded.set_xticks([])
    ax_decoded.set_yticks([])

    figure.suptitle(f"Autoencoder Visualization ({model_name})", fontsize=12)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path, help="input board-state file")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_DIRS.keys()),
        default="training_32",
        help="trained model bundle to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "encoding_visualization.png",
        help="PNG output path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decode threshold override (default: threshold from board_metadata.npz or 0.5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = MODEL_DIRS[args.model]

    if not model_dir.exists():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            f"TensorFlow is not installed. Run: python3 -m pip install -r {SCRIPT_DIR / 'requirements.txt'}"
        ) from exc

    encoder_path = model_dir / "encoder.keras"
    autoencoder_path = model_dir / "autoencoder.keras"
    if not encoder_path.exists():
        raise FileNotFoundError(f"missing encoder model: {encoder_path}")
    if not autoencoder_path.exists():
        raise FileNotFoundError(f"missing autoencoder model: {autoencoder_path}")

    metadata = load_metadata(model_dir)
    threshold = (
        float(np.asarray(metadata.get("threshold", np.asarray(0.5))).item())
        if args.threshold is None
        else args.threshold
    )

    original_board, sample = board_to_sample(args.board, metadata)
    sample_batch = np.expand_dims(sample, axis=0)

    encoder = tf.keras.models.load_model(encoder_path)
    autoencoder = tf.keras.models.load_model(autoencoder_path)

    latent_batch = encoder.predict(sample_batch, verbose=0)
    decoded_sample_batch = autoencoder.predict(sample_batch, verbose=0)

    latent = latent_batch[0].astype(np.float32, copy=False)
    decoded_board = rebuild_board(decoded_sample_batch[0], metadata, threshold)

    render_comparison(args.output, original_board, latent, decoded_board, args.model)
    print(f"Saved visualization to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
