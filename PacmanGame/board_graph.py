"""Shared static graph data and dynamic observations for Pac-Man controllers."""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Iterable, Mapping

try:
    from .board import GameBoard, Point
    from .render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL
except ImportError:
    from board import GameBoard, Point
    from render_board import GHOST_GATE, PELLET, POWER_PELLET, WALL

ACTIONS = ("up", "down", "left", "right")
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
REVERSE_DIRECTION = {"up": "down", "down": "up", "left": "right", "right": "left"}
GHOST_KEYS = ("blinky_pos", "pinky_pos", "inky_pos", "clyde_pos")
LayoutKey = tuple[tuple[bool, ...], ...]


class StaticBoardGraph:
    """Immutable topology shared by every game on the same maze."""

    def __init__(self, blocked: LayoutKey):
        self.layout = blocked
        self.height = len(blocked)
        self.width = len(blocked[0]) if blocked else 0
        raw = self._build_neighbors(blocked)
        self.neighbors: Mapping[Point, Mapping[str, Point]] = MappingProxyType({
            point: MappingProxyType(adjacent) for point, adjacent in raw.items()
        })
        self.points = tuple(self.neighbors)
        self.point_ids = MappingProxyType({
            point: index for index, point in enumerate(self.points)
        })
        self.degrees = MappingProxyType({
            point: len(adjacent) for point, adjacent in self.neighbors.items()
        })
        self.unreachable_distance = len(self.points) + 1
        self._neighbor_ids = tuple(
            tuple(self.point_ids[p] for p in self.neighbors[point].values())
            for point in self.points
        )
        self.distances = self._build_all_pairs_distances()
        longest_reachable = max(
            (
                distance
                for row in self.distances
                for distance in row
                if distance < self.unreachable_distance
            ),
            default=0,
        )
        # Reachable distances occupy [0, 1); exactly 1 means unreachable.
        self.distance_scale = float(longest_reachable + 1)
        self.corridors = MappingProxyType(self._build_corridors())

    def _destination(self, point: Point, direction: str) -> Point | None:
        dx, dy = DIRECTIONS[direction]
        x, y = point[0] + dx, point[1] + dy
        if dy == 0:
            x %= self.width
        return (x, y) if 0 <= x < self.width and 0 <= y < self.height else None

    def _build_neighbors(self, blocked: LayoutKey) -> dict[Point, dict[str, Point]]:
        result = {}
        for y, row in enumerate(blocked):
            for x, is_blocked in enumerate(row):
                if is_blocked:
                    continue
                point = (x, y)
                adjacent = {}
                for action in ACTIONS:
                    destination = self._destination(point, action)
                    if destination is not None and not blocked[destination[1]][destination[0]]:
                        adjacent[action] = destination
                result[point] = adjacent
        return result

    def _build_all_pairs_distances(self) -> tuple[tuple[int, ...], ...]:
        rows = []
        for source in range(len(self.points)):
            distances = [self.unreachable_distance] * len(self.points)
            distances[source] = 0
            queue = deque([source])
            while queue:
                tile_id = queue.popleft()
                for other_id in self._neighbor_ids[tile_id]:
                    if distances[other_id] == self.unreachable_distance:
                        distances[other_id] = distances[tile_id] + 1
                        queue.append(other_id)
            rows.append(tuple(distances))
        return tuple(rows)

    def _build_corridors(self) -> dict[tuple[Point, str], tuple[Point, ...]]:
        result = {}
        for origin, adjacent in self.neighbors.items():
            for action, candidate in adjacent.items():
                path = [candidate]
                previous, current = origin, candidate
                while self.degrees[current] == 2:
                    onward = [p for p in self.neighbors[current].values() if p != previous]
                    if len(onward) != 1:
                        break
                    previous, current = current, onward[0]
                    path.append(current)
                result[(origin, action)] = tuple(path)
        return result

    def distance(self, first: Point, second: Point) -> int:
        first_id = self.point_ids.get(first)
        second_id = self.point_ids.get(second)
        if first_id is None or second_id is None:
            return self.unreachable_distance
        return self.distances[first_id][second_id]


def _layout_key(board: GameBoard) -> LayoutKey:
    blocked = WALL | GHOST_GATE
    return tuple(tuple(bool(value & blocked) for value in row) for row in board.tiles)


@lru_cache(maxsize=None)
def _precompute_layout(layout: LayoutKey) -> StaticBoardGraph:
    return StaticBoardGraph(layout)


def precompute_board_graph(board: GameBoard) -> StaticBoardGraph:
    """Return the process-wide shared precomputed graph for a maze."""
    return _precompute_layout(_layout_key(board))


class ObservationProvider:
    """Per-game dynamic state backed by a shared precomputed graph."""

    def __init__(self, board: GameBoard, static_graph: StaticBoardGraph | None = None):
        self.static = static_graph or precompute_board_graph(board)
        if (self.static.width, self.static.height) != (board.width, board.height):
            raise ValueError("static board graph dimensions do not match board")
        if self.static.layout != _layout_key(board):
            raise ValueError("static board graph topology does not match board layout")
        self.neighbors = self.static.neighbors
        self.degrees = self.static.degrees
        self.unreachable_distance = self.static.unreachable_distance
        self.distance_scale = self.static.distance_scale
        self._pellets: set[Point] = set()
        self._power_pellets: set[Point] = set()
        self._collectibles: set[Point] = set()
        self._collectible_sum_x = 0
        self._collectible_sum_y = 0
        self._distance_cache: dict[str, Mapping[Point, int]] = {}
        self._ghost_signature = None
        self.reset(board)

    def reset(self, board: GameBoard) -> None:
        if self.static.layout != _layout_key(board):
            raise ValueError("cannot reset observations with a different board layout")
        self._pellets = {(x, y) for x, y, value in board.iter_tiles() if value & PELLET}
        self._power_pellets = {
            (x, y) for x, y, value in board.iter_tiles() if value & POWER_PELLET
        }
        self._collectibles = self._pellets | self._power_pellets
        self._collectible_sum_x = sum(x for x, _ in self._collectibles)
        self._collectible_sum_y = sum(y for _, y in self._collectibles)
        self._distance_cache.clear()
        self._ghost_signature = None

    def pellet_consumed(self, point: Point, *, ordinary: bool, power: bool) -> None:
        was_collectible = point in self._collectibles
        if ordinary and point in self._pellets:
            self._pellets.remove(point)
            self._distance_cache.pop("pellet", None)
        if power and point in self._power_pellets:
            self._power_pellets.remove(point)
            self._distance_cache.pop("power_pellet", None)
        if was_collectible and point not in self._pellets and point not in self._power_pellets:
            self._collectibles.remove(point)
            self._collectible_sum_x -= point[0]
            self._collectible_sum_y -= point[1]

    def _pellet_center_alignment(self, origin: Point, action: str) -> float:
        """Return how closely an action points at the collectible center of mass."""
        count = len(self._collectibles)
        if not count:
            return 0.0
        dx = self._collectible_sum_x / count - origin[0]
        dy = self._collectible_sum_y / count - origin[1]
        magnitude = math.hypot(dx, dy)
        if magnitude == 0.0:
            return 0.0
        move_x, move_y = DIRECTIONS[action]
        return (move_x * dx + move_y * dy) / magnitude

    def _sync_ghosts(self, board: GameBoard) -> tuple[Point, ...]:
        positions = tuple(getattr(board, key) for key in GHOST_KEYS)
        signature = (board.player_powerup, positions)
        if signature != self._ghost_signature:
            self._ghost_signature = signature
            self._distance_cache.pop("danger_ghost", None)
            self._distance_cache.pop("edible_ghost", None)
        return positions

    def _distance_map(self, sources: Iterable[Point]) -> dict[Point, int]:
        distances = {point: self.unreachable_distance for point in self.static.points}
        queue = deque()
        for point in sources:
            if point in distances and distances[point] != 0:
                distances[point] = 0
                queue.append(point)
        while queue:
            point = queue.popleft()
            for other in self.neighbors[point].values():
                if distances[other] == self.unreachable_distance:
                    distances[other] = distances[point] + 1
                    queue.append(other)
        return distances

    def _ghost_distance_map(self, positions: tuple[Point, ...]) -> dict[Point, int]:
        return {
            point: min(
                (self.static.distance(point, ghost) for ghost in positions),
                default=self.unreachable_distance,
            )
            for point in self.static.points
        }

    def distance_map(self, target: str, board: GameBoard) -> Mapping[Point, int]:
        ghosts = self._sync_ghosts(board)
        if target not in self._distance_cache:
            if target == "pellet":
                distances = self._distance_map(self._pellets)
            elif target == "power_pellet":
                distances = self._distance_map(self._power_pellets)
            elif target in {"danger_ghost", "edible_ghost"}:
                wanted = (target == "edible_ghost") == board.player_powerup
                distances = self._ghost_distance_map(ghosts if wanted else ())
            else:
                raise ValueError(f"unknown distance-map target {target!r}")
            self._distance_cache[target] = MappingProxyType(distances)
        return self._distance_cache[target]

    def _nearest_ghost(self, candidate: Point, ghosts: tuple[Point, ...]) -> int:
        return min(
            (self.static.distance(candidate, ghost) for ghost in ghosts),
            default=self.unreachable_distance,
        )

    def _normalized_distance(self, distance: int) -> float:
        if distance >= self.unreachable_distance:
            return 1.0
        return distance / self.distance_scale

    def for_legal_moves(
        self, board: GameBoard, last_direction: str
    ) -> dict[str, dict[str, float]]:
        ghosts = self._sync_ghosts(board)
        pellet_distances = self.distance_map("pellet", board)
        power_distances = self.distance_map("power_pellet", board)
        adjacent = self.neighbors.get(board.player_pos, {})
        result = {}
        for action in ACTIONS:
            candidate = adjacent.get(action)
            if candidate is None:
                continue
            corridor = self.static.corridors[(board.player_pos, action)]
            ghost_distance = self._nearest_ghost(candidate, ghosts)
            result[action] = {
                "pellet_here": float(candidate in self._pellets),
                "pellet_distance": self._normalized_distance(
                    pellet_distances[candidate]
                ),
                "power_pellet_distance": self._normalized_distance(
                    power_distances[candidate]
                ),
                "danger_ghost_distance": self._normalized_distance(
                    self.unreachable_distance if board.player_powerup else ghost_distance
                ),
                "edible_ghost_distance": self._normalized_distance(
                    ghost_distance if board.player_powerup else self.unreachable_distance
                ),
                "is_reverse": float(action == REVERSE_DIRECTION.get(last_direction)),
                "is_forward": float(action == last_direction),
                "candidate_degree": self.degrees[candidate] / 4.0,
                "pellets_ahead": (
                    sum(p in self._pellets for p in corridor) / len(corridor)
                ),
                "steps_to_junction": (len(corridor) - 1) / self.distance_scale,
                "pellet_center_direction": self._pellet_center_alignment(
                    board.player_pos, action
                ),
            }
        return result
