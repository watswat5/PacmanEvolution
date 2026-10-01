#!/usr/bin/env python3
"""Playable and headless Pac-Man state-transition engine.

Interactive mode uses Tkinter. Headless mode reads one board-state file, applies
one player action plus one deterministic ghost turn, and writes the next state.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .board_graph import ObservationProvider, StaticBoardGraph
    from .render_board import (
        GHOST_GATE, PELLET, PLAYER_LOCATION, POWER_PELLET, WALL,
        GameBoard, read_board, render_board, write_board,
    )
except ImportError:  # Direct execution: python PacmanGame/pacman.py
    from board_graph import ObservationProvider, StaticBoardGraph
    from render_board import (
        GHOST_GATE, PELLET, PLAYER_LOCATION, POWER_PELLET, WALL,
        GameBoard, read_board, render_board, write_board,
    )


Point = tuple[int, int]
ACTIONS = ("up", "down", "left", "right")
DIRECTIONS: dict[str, Point] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "stay": (0, 0),
}
REVERSE_DIRECTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GHOST_KEYS = ("blinky_pos", "pinky_pos", "inky_pos", "clyde_pos")
POWERUP_DURATION = 40
PELLET_SCORE = 1_000
POWER_PELLET_SCORE = 1_000
GHOST_SCORE = 100
SNAPSHOT_PATTERN = "state_*.txt"


@dataclass(frozen=True)
class StepResult:
    """Result returned after one environment timestep."""

    observation: dict[str, Any]
    reward: int
    terminated: bool
    won: bool
    game_over: bool


class PacmanGame:
    """Reusable deterministic Pac-Man environment for play or training.

    ``board`` may be a parsed GameBoard or the path to a board-state file.
    Integer actions use 0=up, 1=down, 2=left, and 3=right.
    """

    def __init__(
        self,
        board: GameBoard | str | Path,
        *,
        board_graph: StaticBoardGraph | None = None,
    ):
        if isinstance(board, (str, Path)):
            board = read_board(Path(board))
        self._initial_board = self._copy_board(board)
        self.board = self._copy_board(board)
        self.player_direction = self._saved_player_direction(board)
        self.game_over = board.game_over
        self.won = self._board_is_complete()
        self.turn = board.game_turn
        self._ghost_homes = {key: getattr(board, key) for key in GHOST_KEYS}
        self._ghost_directions: dict[str, str | None] = {
            key: None for key in GHOST_KEYS
        }
        self._gate_positions = self._find_gate_positions()
        self._ghost_house = self._find_ghost_house()
        self._observation_provider = ObservationProvider(
            self.board, board_graph, powerup_duration=POWERUP_DURATION
        )
        self._sync_board_observations()

    @staticmethod
    def _copy_board(board: GameBoard) -> GameBoard:
        return replace(board, tiles=[row.copy() for row in board.tiles])

    @staticmethod
    def _saved_player_direction(board: GameBoard) -> str:
        directions = {
            action: getattr(board, f"player_{action}") for action in ACTIONS
        }
        return next(action for action, active in directions.items() if active)

    def _board_is_complete(self) -> bool:
        return not any(
            value & (PELLET | POWER_PELLET)
            for row in self.board.tiles
            for value in row
        )

    def reset(self) -> dict[str, Any]:
        """Restore the initial state and return its observation."""
        self.board = self._copy_board(self._initial_board)
        self.player_direction = self._saved_player_direction(self.board)
        self.game_over = self.board.game_over
        self.won = self._board_is_complete()
        self.turn = self.board.game_turn
        self._ghost_homes = {
            key: getattr(self.board, key) for key in GHOST_KEYS
        }
        self._ghost_directions = {key: None for key in GHOST_KEYS}
        self._gate_positions = self._find_gate_positions()
        self._ghost_house = self._find_ghost_house()
        self._observation_provider.reset(self.board)
        self._sync_board_observations()
        return self.observation()

    def _sync_board_observations(self) -> None:
        """Keep serialized, derived fields consistent with the active layout."""
        self.board = replace(self.board, **self.legal_moves())

    def get_move_observations(self) -> dict[str, dict[str, float]]:
        """Return read-only graph features for every currently legal move.

        Feature distances start at each candidate adjacent tile, so a target on
        that tile has distance zero.  Calling this method never advances or
        otherwise mutates the game state.
        """
        return self._observation_provider.for_legal_moves(
            self.board, self.player_direction
        )

    def get_distance_map(self, target: str) -> Mapping[Point, int]:
        """Return cached tile distances for a named dynamic target category."""
        return self._observation_provider.distance_map(target, self.board)

    def get_tile_neighbors(self) -> Mapping[Point, Mapping[str, Point]]:
        """Return the precomputed player-valid tile adjacency mapping."""
        return self._observation_provider.neighbors

    def legal_moves(self) -> dict[str, bool]:
        """Return exact movement availability for the four controller actions."""
        return {
            action: self._can_enter(
                self._destination(self.board.player_pos, action), ghost=False
            )
            for action in ACTIONS
        }

    def observation(self) -> dict[str, Any]:
        """Return a copy-safe state suitable for a controller or encoder."""
        return {
            "board": tuple(tuple(row) for row in self.board.tiles),
            "player_pos": self.board.player_pos,
            "powerup_timer": self.board.player_powerup_turns,
            "inky_pos": self.board.inky_pos,
            "blinky_pos": self.board.blinky_pos,
            "pinky_pos": self.board.pinky_pos,
            "clyde_pos": self.board.clyde_pos,
            "legal_moves": self.legal_moves(),
            "score": self.board.score,
            "turn": self.turn,
            "player_direction": self.player_direction,
        }

    def save(self, path: str | Path) -> None:
        """Save the current state in the standard board-state format."""
        write_board(self.board, Path(path))

    def render(self):
        """Return the current frame as a Pillow Image without opening a window."""
        return render_board(self.board)

    @property
    def width(self) -> int:
        return len(self.board.tiles[0])

    @property
    def height(self) -> int:
        return len(self.board.tiles)

    def _find_gate_positions(self) -> list[Point]:
        return [
            (x, y)
            for y, row in enumerate(self.board.tiles)
            for x, value in enumerate(row)
            if value & GHOST_GATE
        ]

    def _find_ghost_house(self) -> set[Point]:
        """Find the enclosed area immediately below the ghost gate."""
        if not self._gate_positions:
            return set()
        gate_x, gate_y = self._gate_positions[len(self._gate_positions) // 2]
        seed = (gate_x, gate_y + 1)
        if not self._in_bounds(seed) or self._tile(seed) & (WALL | GHOST_GATE):
            return set()

        found = {seed}
        pending = deque([seed])
        while pending:
            point = pending.popleft()
            for direction in ("up", "left", "right", "down"):
                neighbor = self._destination(point, direction, allow_wrap=False)
                if (
                    neighbor is not None
                    and neighbor not in found
                    and not self._tile(neighbor) & (WALL | GHOST_GATE)
                ):
                    found.add(neighbor)
                    pending.append(neighbor)
        return found

    def _in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def _tile(self, point: Point) -> int:
        x, y = point
        return self.board.tiles[y][x]

    def _destination(
        self, point: Point, direction: str, *, allow_wrap: bool = True
    ) -> Point | None:
        dx, dy = DIRECTIONS[direction]
        x, y = point[0] + dx, point[1] + dy
        if allow_wrap and dy == 0:
            x %= self.width
        candidate = (x, y)
        return candidate if self._in_bounds(candidate) else None

    def _can_enter(self, point: Point | None, *, ghost: bool) -> bool:
        if point is None:
            return False
        blocked = WALL if ghost else WALL | GHOST_GATE
        return not self._tile(point) & blocked

    def _set_position(self, key: str, point: Point) -> None:
        self.board = replace(self.board, **{key: point})

    def _move_player(self, direction: str) -> None:
        if direction != "stay":
            self.player_direction = direction
            self.board = replace(
                self.board,
                **{
                    f"player_{action}": action == direction
                    for action in ACTIONS
                },
            )
        old_position = self.board.player_pos
        destination = self._destination(old_position, direction)
        if not self._can_enter(destination, ghost=False):
            destination = old_position

        ghosts = {getattr(self.board, key): key for key in GHOST_KEYS}
        if destination in ghosts:
            if self.board.player_powerup:
                self._send_ghost_home(ghosts[destination])
            else:
                self._set_position("player_pos", destination)
                self.game_over = True
                self.board = replace(self.board, game_over=True)
                return

        self._set_position("player_pos", destination)
        x, y = destination
        tile = self.board.tiles[y][x]
        score = self.board.score
        if tile & PELLET:
            score += PELLET_SCORE
        if tile & POWER_PELLET:
            score += POWER_PELLET_SCORE
        powerup_turns = self.board.player_powerup_turns
        if tile & POWER_PELLET:
            powerup_turns = POWERUP_DURATION
        pellets_consumed = int(bool(tile & (PELLET | POWER_PELLET)))
        self.board.tiles[y][x] = tile & ~(PELLET | POWER_PELLET)
        self._observation_provider.pellet_consumed(
            destination,
            ordinary=bool(tile & PELLET),
            power=bool(tile & POWER_PELLET),
        )
        self.board = replace(
            self.board,
            player_powerup=powerup_turns > 0,
            player_powerup_turns=powerup_turns,
            score=score,
            pellets_eaten=self.board.pellets_eaten + pellets_consumed,
            pellets_remaining=max(0, self.board.pellets_remaining - pellets_consumed),
        )

        # If a board uses bit 16 as a location marker, keep that marker current.
        if any(value & PLAYER_LOCATION for row in self.board.tiles for value in row):
            for row in self.board.tiles:
                for index, value in enumerate(row):
                    row[index] = value & ~PLAYER_LOCATION
            self.board.tiles[y][x] |= PLAYER_LOCATION

    def _send_ghost_home(self, key: str) -> None:
        self.board = replace(self.board, score=self.board.score + GHOST_SCORE)
        self._ghost_directions[key] = None
        occupied = {
            getattr(self.board, other)
            for other in GHOST_KEYS
            if other != key
        } | {self.board.player_pos}
        for point in sorted(self._ghost_house, key=lambda p: (p[1], p[0])):
            if point not in occupied:
                self._set_position(key, point)
                return
        preferred = self._ghost_homes[key]
        if preferred not in occupied:
            self._set_position(key, preferred)

    @staticmethod
    def _distance(first: Point, second: Point) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _ahead_of_player(self, distance: int) -> Point:
        dx, dy = DIRECTIONS[self.player_direction]
        x = min(max(self.board.player_pos[0] + dx * distance, 0), self.width - 1)
        y = min(max(self.board.player_pos[1] + dy * distance, 0), self.height - 1)
        return x, y

    def _ghost_target(self, key: str, position: Point) -> Point:
        if position in self._ghost_house and self._gate_positions:
            gate = min(self._gate_positions, key=lambda p: self._distance(position, p))
            return gate[0], max(0, gate[1] - 1)
        if self._tile(position) & GHOST_GATE:
            # Finish crossing the gate before enabling normal targeting. Without
            # this forced step, a target below the house can pull a ghost back in.
            return position[0], max(0, position[1] - 1)
        if key == "blinky_pos":
            return self.board.player_pos
        if key == "pinky_pos":
            return self._ahead_of_player(4)
        if key == "inky_pos":
            ahead = self._ahead_of_player(2)
            blinky = self.board.blinky_pos
            return (
                min(max(ahead[0] * 2 - blinky[0], 0), self.width - 1),
                min(max(ahead[1] * 2 - blinky[1], 0), self.height - 1),
            )
        if self._distance(position, self.board.player_pos) > 8:
            return self.board.player_pos
        return 0, self.height - 1

    def _move_ghosts(self) -> None:
        occupied = {getattr(self.board, key) for key in GHOST_KEYS}
        direction_order = ("up", "left", "down", "right")

        for key in GHOST_KEYS:
            position = getattr(self.board, key)
            occupied.remove(position)
            target = self._ghost_target(key, position)
            forced_exit = (
                position in self._ghost_house
                or bool(self._tile(position) & GHOST_GATE)
            )
            options: list[tuple[int, int, Point, str]] = []
            for order, direction in enumerate(direction_order):
                destination = self._destination(position, direction)
                # A normal ghost may leave through the gate, but it cannot turn
                # around and re-enter the house after reaching the maze.
                entering_gate_from_outside = (
                    destination is not None
                    and self._tile(destination) & GHOST_GATE
                    and position not in self._ghost_house
                    and not self._tile(position) & GHOST_GATE
                )
                if entering_gate_from_outside:
                    continue
                if self._can_enter(destination, ghost=True) and destination not in occupied:
                    assert destination is not None
                    distance = self._distance(destination, target)
                    if self.board.player_powerup and not forced_exit:
                        distance = -self._distance(destination, self.board.player_pos)
                    options.append((distance, order, destination, direction))

            # Arcade ghosts retain their heading and do not reverse at an
            # intersection. Reversal is allowed only when it is the sole way
            # out, such as at a dead end. A ghost leaving home has no prior
            # heading, so all exits are considered on its first step.
            previous_direction = self._ghost_directions[key]
            if previous_direction is not None:
                reverse = REVERSE_DIRECTION[previous_direction]
                non_reverse_options = [
                    option for option in options if option[3] != reverse
                ]
                if non_reverse_options:
                    options = non_reverse_options

            if options:
                _, _, destination, direction = min(options)
                self._ghost_directions[key] = direction
            else:
                destination = position
            if destination == self.board.player_pos:
                if self.board.player_powerup:
                    self._send_ghost_home(key)
                    destination = getattr(self.board, key)
                else:
                    self._set_position(key, destination)
                    self.game_over = True
                    self.board = replace(self.board, game_over=True)
            else:
                self._set_position(key, destination)
            occupied.add(destination)
            if self.game_over:
                break

    def step(self, direction: str | int) -> StepResult:
        """Apply one action and return an environment-style transition result."""
        if isinstance(direction, int):
            if not 0 <= direction < len(ACTIONS):
                raise ValueError(f"action index must be from 0 to {len(ACTIONS) - 1}")
            direction = ACTIONS[direction]
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown move {direction!r}")
        previous_score = self.board.score
        if self.game_over or self.won:
            return StepResult(
                self.observation(), 0, True, self.won, self.game_over
            )
        if self.board.player_powerup_turns > 0:
            remaining = self.board.player_powerup_turns - 1
            self.board = replace(
                self.board,
                player_powerup=remaining > 0,
                player_powerup_turns=remaining,
            )
        self._move_player(direction)
        self._observation_provider.player_moved(self.board.player_pos)
        next_turn = self.turn + 1
        ghosts_move_this_turn = (
            not self.board.player_powerup or next_turn % 2 == 0
        )
        if not self.game_over and ghosts_move_this_turn:
            self._move_ghosts()
        self.turn = next_turn
        self.board = replace(self.board, game_turn=self.turn)
        self._sync_board_observations()
        self.won = self._board_is_complete()
        return StepResult(
            observation=self.observation(),
            reward=self.board.score - previous_score,
            terminated=self.game_over or self.won,
            won=self.won,
            game_over=self.game_over,
        )

    def run_episode(
        self,
        policy: Callable[[dict[str, Any]], str | int],
        max_steps: int = 10_000,
        *,
        reset: bool = True,
    ) -> dict[str, Any]:
        """Run a controller until termination or the timestep limit."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        observation = self.reset() if reset else self.observation()
        steps = 0
        while not (self.game_over or self.won) and steps < max_steps:
            result = self.step(policy(observation))
            observation = result.observation
            steps += 1
        return {
            "score": self.board.score,
            "steps": steps,
            "won": self.won,
            "game_over": self.game_over,
            "truncated": not (self.game_over or self.won),
            "observation": observation,
        }


def run_step(args: argparse.Namespace) -> int:
    game = PacmanGame(read_board(args.input))
    game.step(args.move)
    write_board(game.board, args.output)
    if args.render is not None:
        args.render.parent.mkdir(parents=True, exist_ok=True)
        render_board(game.board).save(args.render, "PNG")
    status = "game-over" if game.game_over else "won" if game.won else "running"
    print(f"Saved turn {game.turn} ({status}) to {args.output}")
    return 2 if game.game_over else 0


def next_snapshot_index(directory: Path) -> int:
    """Return one past the highest numbered training state in a directory."""
    highest = -1
    for path in directory.glob(SNAPSHOT_PATTERN):
        stem = path.stem
        prefix, separator, number = stem.rpartition("_")
        if separator and prefix == "state" and number.isdigit():
            highest = max(highest, int(number))
    return highest + 1


def run_interactive(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError as exc:
        raise SystemExit(f"Interactive mode requires Tkinter: {exc}") from exc

    game = PacmanGame(read_board(args.board))
    root = tk.Tk()
    root.title("Pac-Man")
    image_label = tk.Label(root, background="#202428")
    image_label.pack()
    status_label = tk.Label(
        root,
        text="Arrow keys or WASD to move • Esc to quit",
        background="#202428",
        foreground="white",
        pady=6,
    )
    status_label.pack(fill="x")
    photo = None
    snapshot_index = next_snapshot_index(args.save_dir) if args.save_dir else 0

    def save_snapshot() -> None:
        nonlocal snapshot_index
        if args.save_dir is not None:
            output = args.save_dir / f"state_{snapshot_index:06d}.txt"
            while output.exists():
                snapshot_index += 1
                output = args.save_dir / f"state_{snapshot_index:06d}.txt"
            write_board(game.board, output)
            snapshot_index += 1

    def redraw() -> None:
        nonlocal photo
        image = render_board(game.board)
        if args.scale != 1:
            image = image.resize(
                (image.width * args.scale, image.height * args.scale),
                resample=0,
            )
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        if game.game_over:
            status_label.configure(
                text=f"Score {game.board.score} • GAME OVER • Esc to quit"
            )
        elif game.won:
            status_label.configure(
                text=f"Score {game.board.score} • YOU WIN • Esc to quit"
            )
        else:
            power = " • POWERED" if game.board.player_powerup else ""
            status_label.configure(
                text=f"Score {game.board.score} • Turn {game.turn}{power} • Arrow keys or WASD"
            )

    key_moves = {
        "Up": "up", "w": "up", "W": "up",
        "Down": "down", "s": "down", "S": "down",
        "Left": "left", "a": "left", "A": "left",
        "Right": "right", "d": "right", "D": "right",
    }

    def on_key(event: object) -> None:
        keysym = getattr(event, "keysym", "")
        if keysym == "Escape":
            root.destroy()
            return
        move = key_moves.get(keysym)
        if move is not None and not (game.game_over or game.won):
            game.step(move)
            save_snapshot()
            redraw()

    args.save_dir.mkdir(parents=True, exist_ok=True) if args.save_dir else None
    save_snapshot()
    redraw()
    root.bind("<Key>", on_key)
    root.focus_force()
    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="play in an interactive window")
    play.add_argument("board", type=Path, help="initial board-state file")
    play.add_argument(
        "--save-dir",
        type=Path,
        help="save the initial state and every subsequent turn in this directory",
    )
    play.add_argument("--scale", type=int, default=1, choices=range(1, 5))
    play.set_defaults(handler=run_interactive)

    step = subparsers.add_parser("step", help="advance a board by one turn")
    step.add_argument("input", type=Path, help="input board-state file")
    step.add_argument("move", choices=tuple(DIRECTIONS), help="player action")
    step.add_argument("output", type=Path, help="output board-state file")
    step.add_argument("--render", type=Path, help="also save the resulting PNG")
    step.set_defaults(handler=run_step)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
