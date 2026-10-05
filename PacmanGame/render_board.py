#!/usr/bin/env python3
"""Render a bit-flag Pac-Man board CSV as a PNG image."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

# Pillow is only needed to draw boards. Importing it lazily keeps the game,
# board files, and GP training usable without it.
if TYPE_CHECKING:
    from PIL import Image

try:
    from .board import Board, BoardState, GameBoard
except ImportError:  # Direct execution: python PacmanGame/render_board.py
    from board import Board, BoardState, GameBoard


WALL = 1 << 0
PELLET = 1 << 1
POWER_PELLET = 1 << 2
GHOST_GATE = 1 << 3
PLAYER_LOCATION = 1 << 4
KNOWN_FLAGS = WALL | PELLET | POWER_PELLET | GHOST_GATE | PLAYER_LOCATION

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPHICS_DIR = SCRIPT_DIR / "graphics"
POSITION_KEYS = ("player_pos", "inky_pos", "blinky_pos", "pinky_pos", "clyde_pos")
MOVE_KEYS = ("up", "down", "left", "right")
PLAYER_DIRECTION_KEYS = tuple(f"player_{key}" for key in MOVE_KEYS)
SCARED_KEYS = ("inky_scared", "blinky_scared", "pinky_scared", "clyde_scared")


def _parse_position(path: Path, key: str, value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", value)
    if not match:
        raise ValueError(f"{path}: {key} must use the format (x, y)")
    return int(match.group(1)), int(match.group(2))


def _parse_bool(path: Path, key: str, value: str) -> bool:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"{path}: {key} must be true or false")
    return lowered == "true"


def _player_move_flags(board: GameBoard) -> dict[str, bool]:
    width = len(board.tiles[0])
    height = len(board.tiles)
    x, y = board.player_pos
    blocked = WALL | GHOST_GATE
    options = {
        "up": (x, y - 1),
        "down": (x, y + 1),
        "left": ((x - 1) % width, y),
        "right": ((x + 1) % width, y),
    }
    result: dict[str, bool] = {}
    for key, (next_x, next_y) in options.items():
        if 0 <= next_y < height:
            result[key] = not (board.tiles[next_y][next_x] & blocked)
        else:
            result[key] = False
    return result


def read_board(path: Path) -> GameBoard:
    """Read and validate entity headers followed by a rectangular board_data CSV."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        board_start = next(
            index for index, line in enumerate(lines) if line.strip().startswith("board_data")
        )
    except StopIteration as exc:
        raise ValueError(f"{path}: missing board_data = [...]") from exc

    settings: dict[str, str] = {}
    for line_number, line in enumerate(lines[:board_start], start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"{path}: malformed setting on row {line_number}")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if key in settings:
            raise ValueError(f"{path}: duplicate setting {key}")
        settings[key] = value

    required_settings = set(POSITION_KEYS) | {"player_powerup"}
    optional_settings = {
        "player_powerup_turns", "game_turn", "score", "game_over",
        "pellets_eaten", "pellets_remaining",
        *MOVE_KEYS, *PLAYER_DIRECTION_KEYS, *SCARED_KEYS,
    }
    missing = required_settings - settings.keys()
    unknown = settings.keys() - required_settings - optional_settings
    if missing:
        raise ValueError(f"{path}: missing setting(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path}: unknown setting(s): {', '.join(sorted(unknown))}")

    positions = {
        key: _parse_position(path, key, settings[key]) for key in POSITION_KEYS
    }
    powerup_text = settings["player_powerup"].lower()
    if powerup_text not in {"true", "false"}:
        raise ValueError(f"{path}: player_powerup must be true or false")
    try:
        powerup_turns = int(settings.get("player_powerup_turns", "1" if powerup_text == "true" else "0"))
    except ValueError as exc:
        raise ValueError(f"{path}: player_powerup_turns must be a non-negative integer") from exc
    if powerup_turns < 0:
        raise ValueError(f"{path}: player_powerup_turns must be a non-negative integer")
    if (powerup_turns > 0) != (powerup_text == "true"):
        raise ValueError(
            f"{path}: player_powerup must be true exactly when player_powerup_turns is positive"
        )
    # Per-ghost scared flags are optional; a missing one follows player_powerup.
    scared: dict[str, bool | None] = {}
    for key in SCARED_KEYS:
        text = settings.get(key)
        if text is None:
            scared[key] = None
            continue
        if text.lower() not in {"true", "false"}:
            raise ValueError(f"{path}: {key} must be true or false")
        scared[key] = text.lower() == "true"
        if scared[key] and powerup_text != "true":
            raise ValueError(f"{path}: {key} requires player_powerup to be true")
    try:
        game_turn = int(settings.get("game_turn", "0"))
    except ValueError as exc:
        raise ValueError(f"{path}: game_turn must be a non-negative integer") from exc
    if game_turn < 0:
        raise ValueError(f"{path}: game_turn must be a non-negative integer")
    try:
        score = int(settings.get("score", "0"))
    except ValueError as exc:
        raise ValueError(f"{path}: score must be a non-negative integer") from exc
    if score < 0:
        raise ValueError(f"{path}: score must be a non-negative integer")
    game_over_text = settings.get("game_over", "false").lower()
    if game_over_text not in {"true", "false"}:
        raise ValueError(f"{path}: game_over must be true or false")
    game_over = game_over_text == "true"

    provided_moves = {
        key: _parse_bool(path, key, settings[key])
        for key in MOVE_KEYS
        if key in settings
    }
    provided_direction = {
        key: _parse_bool(path, key, settings[key])
        for key in PLAYER_DIRECTION_KEYS
        if key in settings
    }
    if provided_direction and set(provided_direction) != set(PLAYER_DIRECTION_KEYS):
        missing_directions = set(PLAYER_DIRECTION_KEYS) - set(provided_direction)
        raise ValueError(
            f"{path}: missing player direction setting(s): "
            f"{', '.join(sorted(missing_directions))}"
        )
    if provided_direction and sum(provided_direction.values()) != 1:
        raise ValueError(f"{path}: exactly one player direction must be true")

    board_lines = lines[board_start:]
    first = board_lines[0].split("=", 1)
    if len(first) != 2 or first[0].strip() != "board_data":
        raise ValueError(f"{path}: expected board_data = [")
    opening = first[1].strip()
    if not opening.startswith("["):
        raise ValueError(f"{path}: board_data must begin with [")
    board_lines[0] = opening[1:]

    while board_lines and not board_lines[-1].strip():
        board_lines.pop()
    if not board_lines or not board_lines[-1].rstrip().endswith("]"):
        raise ValueError(f"{path}: board_data must end with ]")
    board_lines[-1] = board_lines[-1].rstrip()[:-1]

    rows: list[list[int]] = []
    for row_offset, raw_row in enumerate(csv.reader(board_lines), start=board_start + 1):
        if not raw_row or all(not cell.strip() for cell in raw_row):
            continue
        try:
            row = [int(cell.strip()) for cell in raw_row]
        except ValueError as exc:
            raise ValueError(
                f"{path}: board row {row_offset} contains a non-integer value"
            ) from exc
        if any(value < 0 for value in row):
            raise ValueError(f"{path}: board row {row_offset} contains a negative value")
        if rows and len(row) != len(rows[0]):
            raise ValueError(
                f"{path}: board row {row_offset} has {len(row)} columns; "
                f"expected {len(rows[0])}"
            )
        invalid = [value for value in row if value & ~KNOWN_FLAGS]
        if invalid:
            raise ValueError(
                f"{path}: board row {row_offset} contains unknown flag bits: "
                + ", ".join(map(str, invalid))
            )
        rows.append(row)

    if not rows:
        raise ValueError(f"{path}: board is empty")

    height = len(rows)
    width = len(rows[0])
    for key, (x, y) in positions.items():
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{path}: {key} {(x, y)} is outside the {width}x{height} board")
        if rows[y][x] & WALL:
            raise ValueError(f"{path}: {key} {(x, y)} is inside a wall")

    # Ghosts may share a tile, as in the arcade game.
    ghost_positions = [positions[key] for key in POSITION_KEYS if key != "player_pos"]
    player_collisions = sum(
        point == positions["player_pos"] for point in ghost_positions
    )
    if player_collisions and not game_over:
        raise ValueError(f"{path}: player and ghost may overlap only when game_over is true")
    if game_over and not player_collisions:
        raise ValueError(
            f"{path}: game_over state must have a ghost overlapping the player"
        )

    actual_pellets_remaining = sum(
        bool(value & (PELLET | POWER_PELLET)) for row in rows for value in row
    )
    try:
        pellets_eaten = int(settings.get("pellets_eaten", "0"))
        pellets_remaining = int(
            settings.get("pellets_remaining", str(actual_pellets_remaining))
        )
    except ValueError as exc:
        raise ValueError(
            f"{path}: pellet counters must be non-negative integers"
        ) from exc
    if pellets_eaten < 0 or pellets_remaining < 0:
        raise ValueError(f"{path}: pellet counters must be non-negative integers")
    if pellets_remaining != actual_pellets_remaining:
        raise ValueError(
            f"{path}: pellets_remaining is {pellets_remaining}, "
            f"but the board contains {actual_pellets_remaining}"
        )

    board = GameBoard(
        tiles=rows,
        player_powerup=powerup_text == "true",
        player_powerup_turns=powerup_turns,
        game_turn=game_turn,
        score=score,
        game_over=game_over,
        pellets_eaten=pellets_eaten,
        pellets_remaining=pellets_remaining,
        **provided_direction,
        **positions,
        **scared,
    )
    computed_moves = _player_move_flags(board)
    for key, value in provided_moves.items():
        if computed_moves[key] != value:
            raise ValueError(f"{path}: {key} does not match the current board state")
    return replace(board, **computed_moves)


def write_board(board: GameBoard, path: Path) -> None:
    """Save a GameBoard in the same reusable state format accepted by read_board."""
    move_flags = _player_move_flags(board)
    lines = [
        f"player_pos = {board.player_pos}",
        f"player_powerup = {str(board.player_powerup).lower()}",
        f"player_powerup_turns = {board.player_powerup_turns}",
        f"game_turn = {board.game_turn}",
        f"score = {board.score}",
        f"pellets_eaten = {board.pellets_eaten}",
        f"pellets_remaining = {board.pellets_remaining}",
        f"game_over = {str(board.game_over).lower()}",
        f"inky_pos = {board.inky_pos}",
        f"blinky_pos = {board.blinky_pos}",
        f"pinky_pos = {board.pinky_pos}",
        f"clyde_pos = {board.clyde_pos}",
        *(
            f"{name}_scared = {str(value).lower()}"
            for name, value in board.ghost_scared.items()
        ),
        f"up = {str(move_flags['up']).lower()}",
        f"down = {str(move_flags['down']).lower()}",
        f"left = {str(move_flags['left']).lower()}",
        f"right = {str(move_flags['right']).lower()}",
        f"player_up = {str(board.player_up).lower()}",
        f"player_down = {str(board.player_down).lower()}",
        f"player_left = {str(board.player_left).lower()}",
        f"player_right = {str(board.player_right).lower()}",
        "",
    ]
    for row_index, row in enumerate(board.tiles):
        prefix = "board_data = [" if row_index == 0 else ""
        suffix = "]" if row_index == len(board.tiles) - 1 else ""
        lines.append(prefix + ",".join(map(str, row)) + suffix)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_sprite(name: str, tile_size: tuple[int, int]) -> Image.Image:
    from PIL import Image

    path = GRAPHICS_DIR / name
    sprite = Image.open(path).convert("RGBA")
    if sprite.size != tile_size:
        raise ValueError(
            f"{path} is {sprite.width}x{sprite.height}; expected "
            f"{tile_size[0]}x{tile_size[1]}"
        )
    return sprite


def render_board(board: GameBoard) -> Image.Image:
    from PIL import Image, ImageDraw

    """Compose a board image, always placing floor below every other layer."""
    floor = Image.open(GRAPHICS_DIR / "floor.png").convert("RGBA")
    tile_width, tile_height = floor.size
    tile_size = floor.size

    sprites = {
        WALL: load_sprite("wall.png", tile_size),
        PELLET: load_sprite("orb.png", tile_size),
        POWER_PELLET: load_sprite("power_orb.png", tile_size),
    }

    entity_sprites = {
        "player_pos": load_sprite("pacman.png", tile_size),
        "inky_pos": load_sprite("inky.png", tile_size),
        "blinky_pos": load_sprite("blinky.png", tile_size),
        "pinky_pos": load_sprite("pinky.png", tile_size),
        "clyde_pos": load_sprite("clyde.png", tile_size),
    }
    scared_ghost = load_sprite("scared_ghost.png", tile_size)

    width = len(board.tiles[0]) * tile_width
    height = len(board.tiles) * tile_height
    image = Image.new("RGBA", (width, height))

    # Each pass is a layer. The floor pass deliberately covers every cell,
    # including walls, so transparent sprite pixels never reveal empty canvas.
    for row_index, row in enumerate(board.tiles):
        for column_index, _ in enumerate(row):
            image.alpha_composite(
                floor, (column_index * tile_width, row_index * tile_height)
            )

    # Walls are the first foreground layer.
    for row_index, row in enumerate(board.tiles):
        for column_index, value in enumerate(row):
            if value & WALL:
                image.alpha_composite(
                    sprites[WALL],
                    (column_index * tile_width, row_index * tile_height),
                )

    # The sheet has no gate sprite, so draw the classic centered pink gate.
    gate_layer = Image.new("RGBA", image.size)
    gate_draw = ImageDraw.Draw(gate_layer)
    for row_index, row in enumerate(board.tiles):
        for column_index, value in enumerate(row):
            if value & GHOST_GATE:
                left = column_index * tile_width
                top = row_index * tile_height
                gate_draw.rectangle(
                    (left, top + tile_height // 2 - 1, left + tile_width - 1,
                     top + tile_height // 2),
                    fill=(255, 183, 255, 255),
                )
    image.alpha_composite(gate_layer)

    for flag in (PELLET, POWER_PELLET):
        sprite = sprites[flag]
        for row_index, row in enumerate(board.tiles):
            for column_index, value in enumerate(row):
                if value & flag:
                    image.alpha_composite(
                        sprite,
                        (column_index * tile_width, row_index * tile_height),
                    )

    # Entities are independent of tile flags and always render on top.
    ghost_scared = board.ghost_scared
    for key in POSITION_KEYS:
        x, y = getattr(board, key)
        sprite = (
            scared_ghost
            if key != "player_pos" and ghost_scared[key.removesuffix("_pos")]
            else entity_sprites[key]
        )
        image.alpha_composite(sprite, (x * tile_width, y * tile_height))

    # player_powerup is retained in GameBoard for game state. There is currently
    # no separate powered-up Pac-Man sprite, so it does not alter this still image.
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Pac-Man board-state file to a PNG."
    )
    parser.add_argument("board", type=Path, help="input board-state file")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="output PNG (default: input filename with a .png suffix)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.board.with_suffix(".png")
    board = read_board(args.board)
    rendered = render_board(board)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output, "PNG")
    print(f"Saved {rendered.width}x{rendered.height} board to {output}")


if __name__ == "__main__":
    main()
