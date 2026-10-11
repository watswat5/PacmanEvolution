"""Shared static graph data and dynamic observations for Pac-Man controllers."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping as MappingBase
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
DEFAULT_POWERUP_DURATION = 52
# Tile-path radius used by ``danger_ghosts_nearby``.
GHOST_NEAR_RADIUS = 8
# Pac-Man steps that ``recently_visited`` remembers (also the trainer's
# revisit-penalty window).
VISIT_WINDOW = 12

# Every value is bounded: distances are normalized to [0, 1) with exactly 1.0
# meaning "no such target / unreachable", flags and fractions lie in [0, 1],
# and ``pellet_center_direction``, ``junction_ghost_margin`` and
# ``danger_ghost_approaching`` lie in [-1, 1].
# Append new names at the end so previously pickled trees keep working.
MOVE_FEATURE_NAMES = (
    "pellet_here",
    "pellet_distance",
    "power_pellet_distance",
    "danger_ghost_distance",
    "second_danger_ghost_distance",
    "danger_ghosts_nearby",
    "edible_ghost_distance",
    "frightened_time",
    "is_reverse",
    "is_forward",
    "candidate_degree",
    "pellets_ahead",
    "steps_to_junction",
    "escape_options",
    "dead_end_depth",
    "recently_visited",
    "pellet_center_direction",
    "junction_ghost_margin",
    "safe_escape_options",
    # Evasion features (2026-10-09). Training leaves them out unless asked
    # (train_board_graph.py --add-features); see EVASION_FEATURE_NAMES.
    "danger_ghost_approaching",
    "pincer",
    "safe_territory",
    # Game phase (2026-10-10): share of the board's pellets still uneaten.
    "pellets_left",
)
# The original 19 features, the trainer's default terminals.
BASE_MOVE_FEATURE_NAMES = MOVE_FEATURE_NAMES[:19]
EVASION_FEATURE_NAMES = MOVE_FEATURE_NAMES[19:22]
# Features training can add with --add-features.
OPTIONAL_FEATURE_NAMES = MOVE_FEATURE_NAMES[19:]
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
        # Continuations available where each move's corridor ends, excluding
        # the corridor just travelled (0 at a dead end, up to 3).
        self.corridor_exits = MappingProxyType({
            key: max(0, self.degrees[path[-1]] - 1)
            for key, path in self.corridors.items()
        })
        self.pocket_depths = MappingProxyType(self._build_pocket_depths())
        # For each move: the corridors that continue from where its corridor
        # ends (excluding the way back), as (far junction id, length) pairs.
        self.corridor_continuations = MappingProxyType(
            self._build_corridor_continuations()
        )

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

    def _build_corridor_continuations(
        self,
    ) -> dict[tuple[Point, str], tuple[tuple[int, int], ...]]:
        result = {}
        for (origin, action), path in self.corridors.items():
            junction = path[-1]
            came_from = path[-2] if len(path) > 1 else origin
            result[(origin, action)] = tuple(
                (self.point_ids[onward[-1]], len(onward))
                for onward_action, first in self.neighbors[junction].items()
                if first != came_from
                for onward in (self.corridors[(junction, onward_action)],)
            )
        return result

    def _build_pocket_depths(self) -> dict[tuple[Point, str], int]:
        """Depth of the dead-end pocket each move enters, or 0 if none.

        A move from ``origin`` to ``candidate`` enters a pocket when the edge
        between them is a bridge (without it, ``candidate`` cannot reach
        ``origin``, so the only way out is back the way Pac-Man came) *and* the
        candidate's side is the smaller side. The size check stops a move that
        leaves a dead end from being labelled as entering the rest of the maze.
        The depth is the farthest pocket tile counted from ``origin`` (a one-tile
        dead end has depth 1).
        """
        result = {}
        for origin, adjacent in self.neighbors.items():
            origin_id = self.point_ids[origin]
            component_size = sum(
                d < self.unreachable_distance for d in self.distances[origin_id]
            )
            for action, candidate in adjacent.items():
                start = self.point_ids[candidate]
                seen = {start: 0}
                queue = deque([start])
                escaped = False
                while queue and not escaped:
                    tile = queue.popleft()
                    for other in self._neighbor_ids[tile]:
                        if tile == start and other == origin_id:
                            continue  # the removed edge
                        if other == origin_id:
                            escaped = True
                            break
                        if other not in seen:
                            seen[other] = seen[tile] + 1
                            queue.append(other)
                is_pocket = not escaped and 2 * len(seen) < component_size
                result[(origin, action)] = max(seen.values()) + 1 if is_pocket else 0
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


# Bounded: each graph is about 1.8 MB, and training sees new boards every
# generation, so an unbounded cache grew by hundreds of megabytes per worker
# over a run. 32 covers the boards in use at once (training boards, the 10
# tournament boards and the validation board); older layouts are recomputed
# in about 0.05 s if they ever come back.
STATIC_GRAPH_CACHE_SIZE = 32


@lru_cache(maxsize=STATIC_GRAPH_CACHE_SIZE)
def _precompute_layout(layout: LayoutKey) -> StaticBoardGraph:
    return StaticBoardGraph(layout)


def precompute_board_graph(board: GameBoard) -> StaticBoardGraph:
    """Return the process-wide shared precomputed graph for a maze."""
    return _precompute_layout(_layout_key(board))


class NearestDistanceMap(MappingBase):
    """Read-only map from each tile to its path distance to the nearest source.

    Values come from the precomputed all-pairs table, worked out only for the
    tiles that are actually looked up (usually the three or four tiles next to
    Pac-Man), instead of a breadth-first search over the whole maze every time
    a pellet is eaten. Same numbers as that search.
    """

    __slots__ = ("_static", "_sources", "_cache")

    def __init__(self, static: StaticBoardGraph, sources: Iterable[Point]):
        self._static = static
        ids = static.point_ids
        self._sources = tuple({ids[point] for point in sources if point in ids})
        self._cache: dict[Point, int] = {}

    def __getitem__(self, point: Point) -> int:
        cached = self._cache.get(point)
        if cached is None:
            if not self._sources:
                cached = self._static.unreachable_distance
            else:
                row = self._static.distances[self._static.point_ids[point]]
                cached = min(map(row.__getitem__, self._sources))
            self._cache[point] = cached
        return cached

    def __iter__(self):
        return iter(self._static.points)

    def __len__(self) -> int:
        return len(self._static.points)


class ObservationProvider:
    """Per-game dynamic state backed by a shared precomputed graph."""

    def __init__(
        self,
        board: GameBoard,
        static_graph: StaticBoardGraph | None = None,
        *,
        powerup_duration: int = DEFAULT_POWERUP_DURATION,
    ):
        self.static = static_graph or precompute_board_graph(board)
        if (self.static.width, self.static.height) != (board.width, board.height):
            raise ValueError("static board graph dimensions do not match board")
        if self.static.layout != _layout_key(board):
            raise ValueError("static board graph topology does not match board layout")
        self.neighbors = self.static.neighbors
        self.degrees = self.static.degrees
        self.unreachable_distance = self.static.unreachable_distance
        self.distance_scale = self.static.distance_scale
        self.powerup_duration = max(1, powerup_duration)
        self._visit_clock = 0
        self._last_visit: dict[Point, int] = {}
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
        self._visit_clock = 0
        self._last_visit = {board.player_pos: 0}

    def player_moved(self, point: Point) -> None:
        """Record Pac-Man's tile after one game step (including standing still)."""
        self._visit_clock += 1
        self._last_visit[point] = self._visit_clock

    def visit_age(self, point: Point) -> int | None:
        """Steps since Pac-Man last stood on ``point``, or None if never."""
        last = self._last_visit.get(point)
        return None if last is None else self._visit_clock - last

    def _recently_visited(self, point: Point) -> float:
        """1.0 for the tile just left, fading linearly to 0 after VISIT_WINDOW."""
        age = self.visit_age(point)
        if age is None or age > VISIT_WINDOW:
            return 0.0
        return (VISIT_WINDOW - max(age, 1) + 1) / VISIT_WINDOW

    @property
    def collectibles_left(self) -> int:
        """Pellets and power pellets still on the board."""
        return len(self._collectibles)

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

    def _sync_ghosts(
        self, board: GameBoard
    ) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
        """Return the (dangerous, edible) ghost positions for this state.

        Ghosts are scared one by one (``Board.ghost_scared``): one that was
        eaten and has returned is dangerous while the others are still edible.
        """
        scared = board.ghost_scared
        danger = tuple(
            getattr(board, key) for key in GHOST_KEYS
            if not scared[key.removesuffix("_pos")]
        )
        edible = tuple(
            getattr(board, key) for key in GHOST_KEYS
            if scared[key.removesuffix("_pos")]
        )
        roles = (danger, edible)
        if roles != self._ghost_signature:
            self._ghost_signature = roles
            self._distance_cache.pop("danger_ghost", None)
            self._distance_cache.pop("edible_ghost", None)
        return roles

    def _distance_map(self, sources: Iterable[Point]) -> Mapping[Point, int]:
        return NearestDistanceMap(self.static, sources)

    def _ghost_distance_map(self, positions: tuple[Point, ...]) -> dict[Point, int]:
        return {
            point: min(
                (self.static.distance(point, ghost) for ghost in positions),
                default=self.unreachable_distance,
            )
            for point in self.static.points
        }

    def distance_map(self, target: str, board: GameBoard) -> Mapping[Point, int]:
        danger, edible = self._sync_ghosts(board)
        if target not in self._distance_cache:
            if target == "pellet":
                distances = self._distance_map(self._pellets)
            elif target == "power_pellet":
                distances = self._distance_map(self._power_pellets)
            elif target in {"danger_ghost", "edible_ghost"}:
                distances = self._ghost_distance_map(
                    edible if target == "edible_ghost" else danger
                )
            else:
                raise ValueError(f"unknown distance-map target {target!r}")
            self._distance_cache[target] = (
                distances if isinstance(distances, NearestDistanceMap)
                else MappingProxyType(distances)
            )
        return self._distance_cache[target]

    def _ghost_distances(self, candidate: Point, ghosts: tuple[Point, ...]) -> list[int]:
        return sorted(self.static.distance(candidate, ghost) for ghost in ghosts)

    def _junction_safety(
        self, key: tuple[Point, str], ghosts: tuple[Point, ...]
    ) -> tuple[float, float]:
        """Race the dangerous ghosts to the end of a move's corridor.

        Returns ``(junction_ghost_margin, safe_escape_options)``. The margin is
        the nearest ghost's path distance to the corridor's end minus
        Pac-Man's, normalized and clamped to [-1, 1]; zero or less means a
        ghost gets there first (a tie loses because Pac-Man moves first and the
        ghost then steps onto him). The second value counts the onward
        corridors whose far junction Pac-Man also reaches strictly first,
        divided by 3. Path distances ignore that ghosts cannot reverse, so both
        values err towards caution.
        """
        static = self.static
        corridor = static.corridors[key]
        continuations = static.corridor_continuations[key]
        ghost_rows = [
            static.distances[ghost_id]
            for ghost_id in (static.point_ids.get(ghost) for ghost in ghosts)
            if ghost_id is not None
        ]
        if not ghost_rows:
            return 1.0, len(continuations) / 3.0
        steps = len(corridor)
        junction_id = static.point_ids[corridor[-1]]
        margin = min(row[junction_id] for row in ghost_rows) - steps
        safe = 0
        if margin > 0:
            safe = sum(
                min(row[far_id] for row in ghost_rows) > steps + length
                for far_id, length in continuations
            )
        return max(-1.0, min(1.0, margin / self.distance_scale)), safe / 3.0

    def _normalized_distance(self, distance: int) -> float:
        if distance >= self.unreachable_distance:
            return 1.0
        return distance / self.distance_scale

    def _approaching(
        self, ghost: Point, heading: str | None, target: Point
    ) -> bool | None:
        """Whether a ghost's next step can bring it closer to ``target``.

        Ghosts never reverse on their own, so only the moves other than
        straight back count (all moves at a dead end, or for a ghost with no
        heading yet). A ghost on the target counts as approaching. None when
        the ghost is off Pac-Man's graph (in the house or on the gate) or
        cannot reach the target.
        """
        static = self.static
        ghost_id = static.point_ids.get(ghost)
        if ghost_id is None:
            return None
        target_id = static.point_ids[target]
        row_distance = static.distances[ghost_id][target_id]
        if row_distance >= self.unreachable_distance:
            return None
        if row_distance == 0:
            return True
        adjacent = static.neighbors[ghost]
        back = REVERSE_DIRECTION.get(heading) if heading else None
        options = [point for action, point in adjacent.items() if action != back]
        if not options:
            options = list(adjacent.values())
        if not options:
            return None
        return min(
            static.distances[static.point_ids[point]][target_id] for point in options
        ) < row_distance

    def _danger_ghost_approaching(
        self, candidate: Point, danger: list[tuple[Point, str | None]]
    ) -> float:
        """+1 if the dangerous ghost nearest the candidate tile is heading
        towards it, -1 if heading away, 0 with no reachable dangerous ghost."""
        nearest = None
        nearest_distance = self.unreachable_distance
        for position, heading in danger:
            distance = self.static.distance(candidate, position)
            if distance < nearest_distance:
                nearest, nearest_distance = (position, heading), distance
        if nearest is None:
            return 0.0
        approaching = self._approaching(nearest[0], nearest[1], candidate)
        if approaching is None:
            return 0.0
        return 1.0 if approaching else -1.0

    def _pincer(
        self, origin: Point, candidate: Point, danger: list[tuple[Point, str | None]]
    ) -> float:
        """1 when the move heads towards a dangerous ghost while another closes
        in from behind, both within GHOST_NEAR_RADIUS of the candidate tile.

        A ghost is ahead when the move brings Pac-Man closer to it, behind when
        the move takes him further away and the ghost is heading towards him.
        """
        ahead = behind = False
        for position, heading in danger:
            to_candidate = self.static.distance(candidate, position)
            if to_candidate > GHOST_NEAR_RADIUS:
                continue
            to_origin = self.static.distance(origin, position)
            if to_candidate < to_origin:
                ahead = True
            elif to_candidate > to_origin and self._approaching(position, heading, origin):
                behind = True
        return float(ahead and behind)

    def _safe_territory(self, candidate: Point, ghost_distances: Mapping[Point, int]) -> float:
        """Share of the maze's tiles Pac-Man reaches strictly before every
        dangerous ghost, moving onto the candidate tile first.

        Pac-Man reaches a tile at 1 + its path length from the candidate,
        counting only paths through tiles he also reaches first; a ghost
        reaches it at its own path length. Ties go to the ghost, and speeds
        and the ghosts' no-reverse rule are ignored, so this errs towards
        caution.

        The "only through tiles he reaches first" condition never removes a
        tile: if 1 + d(candidate, x) < g(x), every tile y on a shortest path
        to x also has 1 + d(candidate, y) < g(y), because g(y) >= g(x) -
        d(y, x). So this counts tiles with 1 + d(candidate, x) < g(x) from the
        all-pairs distances instead of searching (same result, much faster).
        """
        row = self.static.distances[self.static.point_ids[candidate]]
        points = self.static.points
        safe = sum(
            1 + distance < ghost_distances[point]
            for distance, point in zip(row, points)
        )
        return safe / len(points)

    def for_legal_moves(
        self,
        board: GameBoard,
        last_direction: str,
        ghost_directions: Mapping[str, str | None] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Features of every legal move. ``ghost_directions`` (by GHOST_KEYS
        key) gives the ghosts' current headings for the evasion features;
        without it every ghost is treated as having no heading yet."""
        danger_ghosts, edible_ghosts = self._sync_ghosts(board)
        scared = board.ghost_scared
        headings = ghost_directions or {}
        danger_headed = [
            (getattr(board, key), headings.get(key)) for key in GHOST_KEYS
            if not scared[key.removesuffix("_pos")]
        ]
        ghost_distances = self.distance_map("danger_ghost", board)
        pellet_distances = self.distance_map("pellet", board)
        power_distances = self.distance_map("power_pellet", board)
        frightened_time = min(1.0, board.player_powerup_turns / self.powerup_duration)
        # Same for every move: pellets (and power pellets) still on the board
        # as a share of the whole board, counting an endgame's head start.
        pellet_total = board.pellets_eaten + board.pellets_remaining
        pellets_left = board.pellets_remaining / pellet_total if pellet_total else 0.0
        unreachable = self.unreachable_distance
        adjacent = self.neighbors.get(board.player_pos, {})
        result = {}
        for action in ACTIONS:
            candidate = adjacent.get(action)
            if candidate is None:
                continue
            key = (board.player_pos, action)
            corridor = self.static.corridors[key]
            danger = self._ghost_distances(candidate, danger_ghosts)
            edible = self._ghost_distances(candidate, edible_ghosts)
            junction_margin, safe_exits = self._junction_safety(key, danger_ghosts)
            result[action] = {
                "pellet_here": float(candidate in self._pellets),
                "pellet_distance": self._normalized_distance(
                    pellet_distances[candidate]
                ),
                "power_pellet_distance": self._normalized_distance(
                    power_distances[candidate]
                ),
                "danger_ghost_distance": self._normalized_distance(
                    danger[0] if danger else unreachable
                ),
                "second_danger_ghost_distance": self._normalized_distance(
                    danger[1] if len(danger) > 1 else unreachable
                ),
                "danger_ghosts_nearby": (
                    sum(d <= GHOST_NEAR_RADIUS for d in danger) / len(GHOST_KEYS)
                ),
                "edible_ghost_distance": self._normalized_distance(
                    edible[0] if edible else unreachable
                ),
                "frightened_time": frightened_time,
                "is_reverse": float(action == REVERSE_DIRECTION.get(last_direction)),
                "is_forward": float(action == last_direction),
                "candidate_degree": self.degrees[candidate] / 4.0,
                "pellets_ahead": (
                    sum(p in self._pellets for p in corridor) / len(corridor)
                ),
                # A corridor that loops back round can outrun the
                # shortest-path scale, so clamp to keep the feature in [0, 1].
                "steps_to_junction": min(
                    1.0, (len(corridor) - 1) / self.distance_scale
                ),
                "escape_options": self.static.corridor_exits[key] / 3.0,
                "dead_end_depth": self._normalized_distance(
                    self.static.pocket_depths[key]
                ),
                "recently_visited": self._recently_visited(candidate),
                "pellet_center_direction": self._pellet_center_alignment(
                    board.player_pos, action
                ),
                "junction_ghost_margin": junction_margin,
                "safe_escape_options": safe_exits,
                "danger_ghost_approaching": self._danger_ghost_approaching(
                    candidate, danger_headed
                ),
                "pincer": self._pincer(board.player_pos, candidate, danger_headed),
                "safe_territory": self._safe_territory(candidate, ghost_distances),
                "pellets_left": pellets_left,
            }
        return result
