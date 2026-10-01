#!/usr/bin/env python3
"""Train a TensorFlow/Keras autoencoder on Pac-Man board-state files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PACMAN_DIR = PROJECT_DIR / "PacmanGame"
sys.path.insert(0, str(PACMAN_DIR))

from render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL, read_board  # noqa: E402

COLLECTIBLE = PELLET | POWER_PELLET


@dataclass
class Dataset:
    samples: np.ndarray
    files: list[Path]
    base_tiles: np.ndarray
    positions: np.ndarray
    values: np.ndarray


def load_dataset(directory: Path) -> Dataset:
    """Load board_data and encode only variable collectible occupancy."""
    files = sorted(directory.glob("state_*.txt"))
    if not files:
        raise ValueError(f"no state_*.txt files found in {directory}")

    boards: list[np.ndarray] = []
    shape = None
    for path in files:
        tiles = np.asarray(read_board(path).tiles, dtype=np.uint8)
        if shape is None:
            shape = tiles.shape
        elif tiles.shape != shape:
            raise ValueError(f"{path} has shape {tiles.shape}; expected {shape}")
        boards.append(tiles)

    stacked = np.stack(boards)
    static = stacked & np.uint8(~COLLECTIBLE & 0xFF)
    if not np.all(static == static[0]):
        raise ValueError("wall, floor, or gate layout changes between input files")

    collectible_bits = stacked & COLLECTIBLE
    possible = np.bitwise_or.reduce(collectible_bits, axis=0)
    rows, columns = np.nonzero(possible)
    values = possible[rows, columns]
    if len(rows) == 0:
        raise ValueError("input states contain no pellets or power pellets")
    if np.any((values != PELLET) & (values != POWER_PELLET)):
        raise ValueError("a collectible position changes type between input files")

    samples = (
        collectible_bits[:, rows, columns] == values[np.newaxis, :]
    ).astype(np.float32)
    return Dataset(
        samples,
        files,
        static[0],
        np.column_stack((rows, columns)).astype(np.int16),
        values.astype(np.uint8),
    )


def split_dataset(samples, validation_fraction, test_fraction, seed):
    order = np.random.default_rng(seed).permutation(len(samples))
    test_count = max(1, round(len(samples) * test_fraction))
    validation_count = max(1, round(len(samples) * validation_fraction))
    training_count = len(samples) - validation_count - test_count
    if training_count < 1:
        raise ValueError("dataset is too small for the requested splits")
    return (
        samples[order[:training_count]],
        samples[order[training_count : training_count + validation_count]],
        samples[order[training_count + validation_count :]],
    )


def build_models(tf, input_dim: int, hidden_dim: int, latent_dim: int):
    inputs = tf.keras.Input(shape=(input_dim,), name="collectible_occupancy")
    hidden = tf.keras.layers.Dense(hidden_dim, activation="relu", name="encoder_hidden")(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation="sigmoid", name="latent")(hidden)
    decoded = tf.keras.layers.Dense(hidden_dim, activation="relu", name="decoder_hidden")(latent)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(decoded)
    return (
        tf.keras.Model(inputs, outputs, name="pacman_autoencoder"),
        tf.keras.Model(inputs, latent, name="pacman_encoder"),
    )


def rebuild_board(bits: np.ndarray, dataset: Dataset) -> np.ndarray:
    board = dataset.base_tiles.copy()
    for present, (row, column), value in zip(bits, dataset.positions, dataset.values):
        if present:
            board[row, column] |= value
    return board


def save_reconstructions(path, predictions, samples, dataset):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    count = min(6, len(samples))
    palette = ListedColormap(["black", "#1010ff", "#ffb7ae", "#ff7f9f", "#ffb7ff"])

    def display_values(bits):
        board = rebuild_board(bits, dataset)
        display = np.zeros_like(board)
        display[board & WALL != 0] = 1
        display[board & PELLET != 0] = 2
        display[board & POWER_PELLET != 0] = 3
        display[board & GHOST_GATE != 0] = 4
        return display

    figure, axes = plt.subplots(count, 2, figsize=(5, count * 2.3), squeeze=False)
    for index in range(count):
        axes[index, 0].imshow(display_values(samples[index] >= 0.5), cmap=palette, vmin=0, vmax=4)
        axes[index, 1].imshow(display_values(predictions[index] >= 0.5), cmap=palette, vmin=0, vmax=4)
        axes[index, 0].set_ylabel(f"sample {index}")
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Reconstruction")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(args) -> int:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            f"TensorFlow is not installed. Run: python3 -m pip install -r {SCRIPT_DIR / 'requirements.txt'}"
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    dataset = load_dataset(args.input_dir)
    training, validation, testing = split_dataset(
        dataset.samples, args.validation_fraction, args.test_fraction, args.seed
    )
    autoencoder, encoder = build_models(
        tf, dataset.samples.shape[1], args.hidden_dim, args.latent_dim
    )
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bit_accuracy", threshold=0.5)],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_autoencoder.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, min_delta=args.min_delta,
            restore_best_weights=True, verbose=1,
        ),
    ]
    print(
        f"Loaded {len(dataset.samples)} boards with {dataset.samples.shape[1]} collectible "
        f"inputs ({len(training)} train, {len(validation)} validation, {len(testing)} test)."
    )
    history = autoencoder.fit(
        training, training, validation_data=(validation, validation),
        epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
        callbacks=callbacks, verbose=2,
    )

    autoencoder.save(args.output_dir / "autoencoder.keras")
    encoder.save(args.output_dir / "encoder.keras")
    np.savez_compressed(
        args.output_dir / "board_metadata.npz",
        base_tiles=dataset.base_tiles,
        collectible_positions=dataset.positions,
        collectible_values=dataset.values,
        board_shape=np.asarray(dataset.base_tiles.shape, dtype=np.int16),
        threshold=np.asarray(0.5, dtype=np.float32),
    )

    predictions = autoencoder.predict(testing, batch_size=args.batch_size, verbose=0)
    evaluation = autoencoder.evaluate(
        testing, testing, batch_size=args.batch_size, return_dict=True, verbose=0
    )
    evaluation["exact_board_accuracy"] = float(
        np.mean(np.all((predictions >= 0.5) == (testing >= 0.5), axis=1))
    )
    save_reconstructions(args.output_dir / "reconstructions.png", predictions, testing, dataset)

    rows = [
        {"epoch": index + 1} | {key: float(values[index]) for key, values in history.history.items()}
        for index in range(len(history.history["loss"]))
    ]
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_files": len(dataset.files),
        "board_shape": list(dataset.base_tiles.shape),
        "collectible_inputs": int(dataset.samples.shape[1]),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "test_samples": len(testing),
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "epochs_completed": len(rows),
        "test": {key: float(value) for key, value in evaluation.items()},
        "seed": args.seed,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Saved TensorFlow models and reports to {args.output_dir}. "
        f"Test loss={evaluation['loss']:.5f}, bit accuracy={evaluation['bit_accuracy']:.3%}, "
        f"exact boards={evaluation['exact_board_accuracy']:.3%}."
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PACMAN_DIR / "training_states")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: training_<latent-dim>)",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, choices=(4, 8, 16, 32), default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.hidden_dim < 1 or args.latent_dim < 1:
        parser.error("--hidden-dim and --latent-dim must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1:
        parser.error("validation and test fractions must each be between 0 and 1")
    if args.validation_fraction + args.test_fraction >= 1:
        parser.error("validation and test fractions must sum to less than 1")
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / f"training_{args.latent_dim}"
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(train(parse_args()))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
