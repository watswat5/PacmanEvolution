#!/usr/bin/env python3
"""Playable and headless Pac-Man state-transition engine.

Interactive mode uses Tkinter. Headless mode reads one board-state file, applies
one player action plus one deterministic ghost turn, and writes the next state.

Ghost behaviour and relative speeds follow level 1 of the arcade game on a
one-tile-per-turn grid: see the "Arcade rules" constants below.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from functools import lru_cache
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .board_graph import ObservationProvider, StaticBoardGraph, _layout_key
    from .render_board import (
        GHOST_GATE, PELLET, PLAYER_LOCATION, POWER_PELLET, WALL,
        GameBoard, read_board, render_board, write_board,
    )
except ImportError:  # Direct execution: python PacmanGame/pacman.py
    from board_graph import ObservationProvider, StaticBoardGraph, _layout_key
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
# --- Arcade rules (level 1) --------------------------------------------------
# One turn is one Pac-Man tile. Its length in 60 Hz frames depends on his speed
# and on what he ate; each ghost then advances by its own speed over that time.
FRAMES_PER_SECOND = 60
# Tiles per frame at "100%" speed (75.76 pixels per second, 8 pixels per tile).
FULL_SPEED_TILES_PER_FRAME = 75.75757625 / FRAMES_PER_SECOND / 8
PACMAN_SPEED = 0.80
PACMAN_FRIGHT_SPEED = 0.90
DOT_PAUSE_FRAMES = 1
ENERGIZER_PAUSE_FRAMES = 3
GHOST_SPEED = 0.75
GHOST_FRIGHT_SPEED = 0.50
GHOST_TUNNEL_SPEED = 0.40
# Blinky's "Cruise Elroy" stages as (dots left, speed), fastest first.
ELROY_STAGES = ((10, 0.85), (20, 0.80))
# Approximation: the arcade does not document how fast eaten ghosts' eyes move.
EYES_SPEED = 2.0
FRIGHT_FRAMES = 6 * FRAMES_PER_SECOND
# Alternating scatter/chase durations in seconds; chase is permanent afterwards.
SCATTER_CHASE_SECONDS = (7, 20, 7, 20, 5, 20, 5)
MODE_BOUNDARIES = tuple(
    sum(SCATTER_CHASE_SECONDS[: index + 1]) * FRAMES_PER_SECOND
    for index in range(len(SCATTER_CHASE_SECONDS))
)
# Dots Pac-Man must eat while each ghost is next in line to leave the house.
HOUSE_DOT_LIMITS = {"pinky_pos": 0, "inky_pos": 30, "clyde_pos": 60}
HOUSE_RELEASE_ORDER = ("blinky_pos", "pinky_pos", "inky_pos", "clyde_pos")
# The next ghost is released anyway after this long without a dot being eaten.
HOUSE_IDLE_FRAMES = 4 * FRAMES_PER_SECOND
GHOST_DIRECTION_ORDER = ("up", "left", "down", "right")
# Original-maze tiles where chasing or scattering ghosts may not turn up.
ORIGINAL_NO_UP_TILES = frozenset({(12, 11), (15, 11), (12, 23), (15, 23)})
# Frames per Pac-Man tile while frightened, and the fright length in turns.
FRIGHT_TURN_FRAMES = 1.0 / (FULL_SPEED_TILES_PER_FRAME * PACMAN_FRIGHT_SPEED)
POWERUP_DURATION = math.ceil(FRIGHT_FRAMES / FRIGHT_TURN_FRAMES)
PELLET_SCORE = 1_000
POWER_PELLET_SCORE = 1_000
GHOST_SCORE = 100
SNAPSHOT_PATTERN = "state_*.txt"


@lru_cache(maxsize=1)
def _original_layout():
    """Wall layout of the original arcade maze, for its maze-specific rules."""
    try:
        from .generate_original_board import original_board
    except ImportError:
        from generate_original_board import original_board
    return _layout_key(original_board())


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
        self._gate_positions = self._find_gate_positions()
        self._ghost_house = self._find_ghost_house()
        self._observation_provider = ObservationProvider(
            self.board, board_graph, powerup_duration=POWERUP_DURATION
        )
        self._reset_dynamics()
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
        self._gate_positions = self._find_gate_positions()
        self._ghost_house = self._find_ghost_house()
        self._observation_provider.reset(self.board)
        self._reset_dynamics()
        self._sync_board_observations()
        return self.observation()

    def _sync_board_observations(self) -> None:
        """Keep serialized, derived fields consistent with the active layout."""
        self.board = replace(
            self.board,
            **self.legal_moves(),
            **{
                f"{key.removesuffix('_pos')}_scared": self._ghost_frightened[key]
                for key in GHOST_KEYS
            },
        )

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

    def _reset_dynamics(self) -> None:
        """Initialise the arcade timing and per-ghost state for a new game."""
        static = self._observation_provider.static
        self._ghost_directions = {key: None for key in GHOST_KEYS}
        self._ghost_state = {}
        for key in GHOST_KEYS:
            position = getattr(self.board, key)
            if position in self._ghost_house:
                self._ghost_state[key] = "house"
            elif self._tile(position) & GHOST_GATE:
                self._ghost_state[key] = "leaving"
            else:
                self._ghost_state[key] = "active"
        self._ghost_progress = {key: 0.0 for key in GHOST_KEYS}
        self._ghost_reverse = {key: False for key in GHOST_KEYS}
        self._respawn_frames = {key: 0.0 for key in GHOST_KEYS}
        # A board saved partway through a game (an endgame training board)
        # has already had its dots counted, so nobody is held in the house.
        self._house_dots = {key: self.board.pellets_eaten for key in GHOST_KEYS}
        self._dot_idle_frames = 0.0
        self._mode_frames = 0.0
        self._mode_index = 0
        self._fright_frames = self.board.player_powerup_turns * FRIGHT_TURN_FRAMES
        scared = self.board.ghost_scared
        self._ghost_frightened = {
            key: self._fright_frames > 0 and scared[key.removesuffix("_pos")]
            for key in GHOST_KEYS
        }
        self._rng = random.Random(0)
        self._step_frames = 0.0
        self._ate_dot = False
        # Tiles of a wrap-around row from each edge to the first junction.
        tunnel: set[Point] = set()
        for y in range(self.height):
            if (0, y) in static.neighbors and (self.width - 1, y) in static.neighbors:
                for xs in (range(self.width), range(self.width - 1, -1, -1)):
                    for x in xs:
                        if static.degrees.get((x, y)) != 2:
                            break
                        tunnel.add((x, y))
        self._tunnel_tiles = frozenset(tunnel)
        self._no_up_tiles = (
            ORIGINAL_NO_UP_TILES if static.layout == _original_layout() else frozenset()
        )
        self._exit_tiles = frozenset(
            (x, y - 1) for x, y in self._gate_positions if y > 0
        )
        self._release_ghosts()

    @property
    def ghost_mode(self) -> str:
        """Current global ghost mode: ``scatter`` or ``chase``."""
        scatter = self._mode_index < len(MODE_BOUNDARIES) and self._mode_index % 2 == 0
        return "scatter" if scatter else "chase"

    def _release_ghosts(self) -> None:
        """Let waiting ghosts leave the house once their dot limit is met."""
        for key in GHOST_KEYS:
            if self._ghost_state[key] != "house" or self._respawn_frames[key] > 0:
                continue
            if self._house_dots[key] < HOUSE_DOT_LIMITS.get(key, 0):
                # Only the most-preferred waiting ghost counts dots, so the
                # ghosts behind it keep waiting too.
                break
            self._start_leaving(key)

    def _start_leaving(self, key: str) -> None:
        in_house = getattr(self.board, key) in self._ghost_house
        self._ghost_state[key] = "leaving" if in_house else "active"
        self._ghost_progress[key] = 0.0

    def _preferred_waiting_ghost(self) -> str | None:
        for key in HOUSE_RELEASE_ORDER:
            if self._ghost_state[key] == "house" and self._respawn_frames[key] <= 0:
                return key
        return None

    def _frighten_ghosts(self) -> None:
        self._fright_frames = float(FRIGHT_FRAMES)
        for key in GHOST_KEYS:
            if self._respawn_frames[key] > 0:
                continue  # eaten and still on its way back to the house
            self._ghost_frightened[key] = True
            if self._ghost_state[key] == "active":
                self._ghost_reverse[key] = True

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

        was_frightened = self._fright_frames > 0
        self._set_position("player_pos", destination)
        if self._resolve_collisions():
            return
        x, y = destination
        tile = self.board.tiles[y][x]
        score = self.board.score
        if tile & PELLET:
            score += PELLET_SCORE
        if tile & POWER_PELLET:
            score += POWER_PELLET_SCORE
        pellets_consumed = int(bool(tile & (PELLET | POWER_PELLET)))
        self.board.tiles[y][x] = tile & ~(PELLET | POWER_PELLET)
        self._observation_provider.pellet_consumed(
            destination,
            ordinary=bool(tile & PELLET),
            power=bool(tile & POWER_PELLET),
        )
        self.board = replace(
            self.board,
            score=score,
            pellets_eaten=self.board.pellets_eaten + pellets_consumed,
            pellets_remaining=max(0, self.board.pellets_remaining - pellets_consumed),
        )
        # Arcade timing: a tile takes 8 pixels at Pac-Man's current speed, and
        # eating freezes him for one frame per dot or three per energizer.
        speed = PACMAN_FRIGHT_SPEED if was_frightened else PACMAN_SPEED
        self._step_frames = 1.0 / (FULL_SPEED_TILES_PER_FRAME * speed)
        self._ate_dot = bool(pellets_consumed)
        if tile & POWER_PELLET:
            self._step_frames += ENERGIZER_PAUSE_FRAMES
            self._frighten_ghosts()
        elif tile & PELLET:
            self._step_frames += DOT_PAUSE_FRAMES

        # If a board uses bit 16 as a location marker, keep that marker current.
        if any(value & PLAYER_LOCATION for row in self.board.tiles for value in row):
            for row in self.board.tiles:
                for index, value in enumerate(row):
                    row[index] = value & ~PLAYER_LOCATION
            self.board.tiles[y][x] |= PLAYER_LOCATION

    def _resolve_collisions(self) -> bool:
        """Eat or be killed by every ghost on Pac-Man's tile. True if he died."""
        for key in GHOST_KEYS:
            if (
                self._ghost_state[key] == "house"
                or getattr(self.board, key) != self.board.player_pos
            ):
                continue
            if self._ghost_frightened[key]:
                self._send_ghost_home(key)
            else:
                self.game_over = True
                self.board = replace(self.board, game_over=True)
                return True
        return False

    def _send_ghost_home(self, key: str) -> None:
        """Score an eaten ghost and park it in the house until its eyes return.

        The arcade's eyes travel back to the house and cannot interact with
        Pac-Man, so the ghost is moved at once and held for the travel time.
        """
        static = self._observation_provider.static
        eaten_at = getattr(self.board, key)
        self.board = replace(self.board, score=self.board.score + GHOST_SCORE)
        self._ghost_directions[key] = None
        self._ghost_frightened[key] = False
        self._ghost_reverse[key] = False
        self._ghost_progress[key] = 0.0
        self._ghost_state[key] = "house"
        home = self._ghost_homes[key]
        if self._ghost_house and home not in self._ghost_house:
            home = min(
                self._ghost_house,
                key=lambda p: (
                    min(self._distance_squared(p, g) for g in self._gate_positions),
                    p[1],
                    p[0],
                ),
            )
        distance = min(
            (static.distance(eaten_at, tile) for tile in self._exit_tiles),
            default=static.distance(eaten_at, home),
        )
        if distance >= static.unreachable_distance:
            distance = 0
        self._respawn_frames[key] = max(
            1.0, distance / (FULL_SPEED_TILES_PER_FRAME * EYES_SPEED)
        )
        self._set_position(key, home)

    @staticmethod
    def _distance_squared(first: Point, second: Point) -> int:
        return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2

    def _ahead_of_player(self, distance: int) -> Point:
        """Tile ``distance`` ahead of Pac-Man, with the arcade's "up" bug.

        When Pac-Man faces up the arcade also offsets the point the same
        distance to the left. Targets may lie outside the maze.
        """
        dx, dy = DIRECTIONS[self.player_direction]
        x = self.board.player_pos[0] + dx * distance
        y = self.board.player_pos[1] + dy * distance
        if self.player_direction == "up":
            x -= distance
        return x, y

    def _scatter_target(self, key: str) -> Point:
        return {
            "blinky_pos": (self.width - 3, -3),
            "pinky_pos": (2, -3),
            "inky_pos": (self.width - 1, self.height + 1),
            "clyde_pos": (0, self.height + 1),
        }[key]

    def _elroy_speed(self) -> float | None:
        """Blinky's "Cruise Elroy" speed once few dots remain, else None."""
        for dots_left, speed in ELROY_STAGES:
            if self.board.pellets_remaining <= dots_left:
                return speed
        return None

    def _ghost_target(self, key: str, position: Point) -> Point:
        if self._ghost_state[key] == "leaving":
            gate = min(
                self._gate_positions,
                key=lambda p: self._distance_squared(position, p),
                default=position,
            )
            return gate[0], gate[1] - 1
        elroy = key == "blinky_pos" and self._elroy_speed() is not None
        if self.ghost_mode == "scatter" and not elroy:
            return self._scatter_target(key)
        if key == "blinky_pos":
            return self.board.player_pos
        if key == "pinky_pos":
            return self._ahead_of_player(4)
        if key == "inky_pos":
            ahead = self._ahead_of_player(2)
            blinky = self.board.blinky_pos
            return ahead[0] * 2 - blinky[0], ahead[1] * 2 - blinky[1]
        if self._distance_squared(position, self.board.player_pos) >= 64:
            return self.board.player_pos
        return self._scatter_target(key)

    def _ghost_speed(self, key: str) -> float:
        if getattr(self.board, key) in self._tunnel_tiles:
            return GHOST_TUNNEL_SPEED
        if self._ghost_frightened[key]:
            return GHOST_FRIGHT_SPEED
        if key == "blinky_pos":
            return self._elroy_speed() or GHOST_SPEED
        return GHOST_SPEED

    def _step_ghost(self, key: str) -> None:
        """Move one ghost one tile using the arcade's turning rules."""
        position = getattr(self.board, key)
        leaving = self._ghost_state[key] == "leaving"
        frightened = self._ghost_frightened[key] and not leaving
        options: dict[str, Point] = {}
        for direction in GHOST_DIRECTION_ORDER:
            destination = self._destination(position, direction)
            if not self._can_enter(destination, ghost=True):
                continue
            assert destination is not None
            if not leaving and (
                self._tile(destination) & GHOST_GATE
                or destination in self._ghost_house
            ):
                continue  # only a ghost leaving the house may use the gate
            if (
                direction == "up"
                and not frightened
                and not leaving
                and position in self._no_up_tiles
            ):
                continue
            options[direction] = destination
        if not options:
            return

        previous = self._ghost_directions[key]
        direction = None
        if self._ghost_reverse[key]:
            self._ghost_reverse[key] = False
            if previous is not None and REVERSE_DIRECTION[previous] in options:
                direction = REVERSE_DIRECTION[previous]
        if direction is None:
            # Ghosts never reverse on their own unless it is the only way out.
            if previous is not None and len(options) > 1:
                options.pop(REVERSE_DIRECTION[previous], None)
            if frightened:
                # A random first choice, then the fixed order until one is open.
                start = self._rng.randrange(len(GHOST_DIRECTION_ORDER))
                order = GHOST_DIRECTION_ORDER[start:] + GHOST_DIRECTION_ORDER[:start]
                direction = next(d for d in order if d in options)
            else:
                target = self._ghost_target(key, position)
                direction = min(
                    options,
                    key=lambda d: (
                        self._distance_squared(options[d], target),
                        GHOST_DIRECTION_ORDER.index(d),
                    ),
                )
        destination = options[direction]
        self._ghost_directions[key] = direction
        self._set_position(key, destination)
        if leaving and destination in self._exit_tiles:
            # Ghosts leave the house heading left unless told to reverse.
            self._ghost_state[key] = "active"
            self._ghost_directions[key] = "left"

    def _advance_time(self) -> None:
        """Run the fright, scatter/chase and house-release clocks for one step."""
        frames = self._step_frames
        for key in GHOST_KEYS:
            if self._respawn_frames[key] > 0:
                self._respawn_frames[key] = max(0.0, self._respawn_frames[key] - frames)
        if self._fright_frames > 0:
            # The scatter/chase clock is paused while ghosts are frightened.
            if self._fright_frames < FRIGHT_FRAMES or not self._ate_dot:
                self._fright_frames = max(0.0, self._fright_frames - frames)
            if self._fright_frames == 0:
                self._ghost_frightened = {key: False for key in GHOST_KEYS}
        else:
            self._mode_frames += frames
            while (
                self._mode_index < len(MODE_BOUNDARIES)
                and self._mode_frames >= MODE_BOUNDARIES[self._mode_index]
            ):
                self._mode_index += 1
                for key in GHOST_KEYS:
                    if self._ghost_state[key] == "active":
                        self._ghost_reverse[key] = True
        if self._ate_dot:
            self._dot_idle_frames = 0.0
            waiting = self._preferred_waiting_ghost()
            if waiting is not None:
                self._house_dots[waiting] += 1
        else:
            self._dot_idle_frames += frames
            if self._dot_idle_frames >= HOUSE_IDLE_FRAMES:
                self._dot_idle_frames = 0.0
                waiting = self._preferred_waiting_ghost()
                if waiting is not None:
                    self._start_leaving(waiting)
        self._release_ghosts()
        turns = math.ceil(self._fright_frames / FRIGHT_TURN_FRAMES)
        self.board = replace(
            self.board, player_powerup=turns > 0, player_powerup_turns=turns
        )

    def _move_ghosts(self) -> None:
        """Advance every ghost by the distance it covers during this step."""
        for key in GHOST_KEYS:
            if self._ghost_state[key] == "house":
                continue
            self._ghost_progress[key] += (
                self._step_frames * FULL_SPEED_TILES_PER_FRAME * self._ghost_speed(key)
            )
            while self._ghost_progress[key] >= 1.0:
                self._ghost_progress[key] -= 1.0
                self._step_ghost(key)
                if self._resolve_collisions():
                    return
                if self._ghost_state[key] == "house":
                    break  # it was eaten

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
        self._move_player(direction)
        self._observation_provider.player_moved(self.board.player_pos)
        if not self.game_over:
            self._advance_time()
            self._move_ghosts()
        self.turn += 1
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
