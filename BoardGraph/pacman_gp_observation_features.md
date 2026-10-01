# Pac-Man GP observation and feature plan

## Goal and contract

Keep the existing tile-by-tile game. On **every Pac-Man step**, ask the game for its legal moves and read-only observations. Evaluate **one shared GP scoring tree** on each legal candidate, choose the highest score, move one tile, and repeat. The graph and pathfinding code supply facts; the GP learns how to value them.

```python
observations = game.get_move_observations()  # {direction: feature_dict}
scores = {move: score_tree(features) for move, features in observations.items()}
move = choose_highest_with_tie_policy(scores, game.last_direction)
game.step(move)
```

An observation is computed from the *current* game state and a hypothetical adjacent tile. It must not move entities or consume pellets. Decide and document whether ghosts move before or after Pac-Man in `game.step`; compute observations after the previous step has fully resolved.

## The two representations

1. **Tile adjacency graph:** one vertex per traversable tile, one unit-cost link per permitted neighboring move, including wrap tunnels. Use this as the ground truth for exact distances. Ordinary BFS, not Dijkstra, handles unit-cost links.
2. **Optional corridor graph:** vertices at junctions, dead ends, corners where needed, portals, and other decision landmarks; edges are ordered corridors. Each edge stores its **length** and ordered tile IDs. Dynamic pellet count is a separate property, not the edge's distance weight. This graph supports corridor-level features; do not replace the existing movement system with it.

If corridor edges have different lengths, use Dijkstra for shortest *weighted* junction-to-junction routes. For an initial implementation, the tile graph alone is sufficient and simpler to validate.

## Recommended initial feature set

All numbers below describe the **candidate move** unless labeled global. Compute them for each *legal* adjacent tile. The GP should see the same terminal names for every candidate, regardless of compass direction.

| Feature | Meaning | Method | Reuse/update |
| --- | --- | --- | --- |
| `pellet_here` | 1 if the candidate tile has an ordinary pellet, else 0 | Tile lookup | Direct lookup each step |
| `pellet_distance` | Shortest distance from candidate tile to any remaining ordinary pellet | Multi-source BFS seeded by pellet tiles | Rebuild after an ordinary pellet disappears; lookup otherwise |
| `power_pellet_distance` | Same, for power pellets | Separate multi-source BFS | Rebuild after a power pellet disappears |
| `danger_ghost_distance` | Shortest tile-path distance from candidate tile to a non-frightened, threatening ghost | Multi-source BFS seeded by dangerous ghost positions | Rebuild after ghosts move or change mode |
| `edible_ghost_distance` | Shortest tile-path distance to a currently edible ghost | Separate multi-source BFS | Rebuild after ghosts move or change mode |
| `is_reverse` | 1 if candidate reverses Pac-Man's previous move | Direction comparison | Direct computation |
| `is_forward` | 1 if candidate continues previous move | Direction comparison | Direct computation |
| `candidate_degree` | Number of legal outgoing moves from candidate tile (0–4) | Static neighbor count | Precompute once |
| `pellets_ahead` | Remaining pellets along the candidate-facing corridor segment up to its next junction/dead end | Ordered corridor-tile scan or maintained per-segment counts | Update on pellet consumption; see direction caveat below |
| `steps_to_junction` | Steps along that segment until its next decision point | Static corridor lookup | Precompute once |

Start with these **10 features**, then add features only when a concrete failure mode motivates them. Including both `pellet_here` and `pellet_distance` matters: a pellet on the next tile should be distinguishable from one several tiles away.

### Useful second-stage features

| Feature | Meaning | Method | Reuse/update |
| --- | --- | --- | --- |
| `pellets_within_radius` | Count of pellets within, say, 6–10 path steps of candidate tile | Precomputed radius-neighborhood tile IDs + dynamic pellet bitset/count; or bounded BFS | Update when pellets disappear; lookup/count per candidate |
| `pellets_remaining_fraction` (global) | Remaining/initial ordinary pellets | Game counter | O(1) |
| `frightened_time_fraction` (global) | Remaining power-up duration / maximum | Game timer | O(1) |
| `power_pellet_ahead` | Whether the candidate-facing corridor portion contains a power pellet | Ordered segment lookup | Update when eaten |
| `ghost_ahead_distance` | Distance to a threatening ghost *along the entered corridor*, if any | Compare ghost edge/offset to candidate-facing segment | Update after ghost motion |
| `escape_options` | Available continuations at the next junction, excluding the edge just traversed | Corridor topology | Precompute once; adjust if temporary restrictions exist |
| `dead_end_depth` | Travel distance from candidate to a dead end if this direction enters one | Corridor graph/topology | Precompute once; optionally incorporate ghost risk dynamically |
| `fruit_distance` | Path distance to an active fruit | BFS from fruit tile or static all-pairs tile distances | Update when fruit appears/moves/disappears |

Avoid raw absolute `up/down/left/right` features in the first experiment: the shared tree should generalize across rotations. For ghost behavior, mode and time remaining often matter more than ghost identity. Treat ghosts in the house or otherwise not threatening according to your game's actual rules.

## Precise distance semantics

From a candidate `next_tile`, `pellet_distance = 0` when that tile has a pellet. If you want distance **from Pac-Man**, add 1 for the candidate step. Use one convention everywhere and document it.

An unrestricted `pellet_distance` lookup can produce an unintuitive result: after stepping toward an empty dead end, the shortest path to a pellet might immediately return through Pac-Man's old tile. That is still a correct **distance from the hypothetical next tile**, but it is *not* “pellets ahead.” Keep these separate:

- `pellet_distance`: nearest pellet by any legal path from the candidate tile; reversal after the first step is allowed.
- `pellets_ahead`: only pellets encountered by continuing away from Pac-Man along the candidate-facing segment (stop at its next junction/dead end). When currently inside a corridor, count only the portion on that side of Pac-Man, not the whole edge.
- If you truly need **distance to a pellet without immediately returning through the old tile**, run a small BFS from the candidate tile with the old tile temporarily blocked, or use a directed/corridor-aware query. Do not relabel the unrestricted map as first-direction-constrained distance.

For no remaining target or an unreachable target, use an explicit `target_exists` flag plus a bounded sentinel distance (for example, `board_walkable_tile_count + 1`), or a documented bounded proximity of 0. Avoid handing `inf` to arithmetic GP primitives.

## What to precompute and what to cache

| Scope | Data | Strategy |
| --- | --- | --- |
| Board load, once | Tile neighbor lists, wrap links, tile IDs, degrees | Precompute static arrays |
| Board load, once | Corridor membership, ordered corridor tiles, next junction, corridor length, dead-end information | Precompute if using corridor features |
| Board load, optional | All-pairs tile distances or radius neighborhoods | Only if profiling shows value; a small fixed maze may make these affordable |
| After a pellet is consumed | Ordinary-pellet distance map | Rebuild with **one multi-source BFS from all remaining pellets**, not one BFS for each candidate |
| After a power pellet is consumed | Power-pellet distance map | Rebuild with its own multi-source BFS |
| After ghost positions or modes change | Dangerous and edible ghost distance maps | Rebuild with separate multi-source BFS runs from eligible ghost tiles |
| Every Pac-Man decision | Candidate-tile features | O(1) distance-array lookups plus cheap segment/count queries; share maps across candidates and GP individuals evaluating the same state |

**Recommendation:** start with cached maps and invalidation flags (`pellets_version`, `power_pellets_version`, `ghosts_version`, `ghost_modes_version`). Build a map lazily on first request after its inputs change. For a maze with only hundreds of tiles, a few BFS traversals per changed state are cheap relative to running whole GP populations; profile before adding incremental distance-map maintenance.

Do **not** recompute BFS inside each GP primitive occurrence. A tree may mention a distance terminal several times, and there can be multiple candidate moves and many individuals. Compute each required map once per state version, then give the tree plain numeric feature values. Static lookup arrays are especially useful; caching an entire feature dictionary by only `next_tile` is unsafe because pellets and ghosts move.

### Multi-source BFS skeleton

```python
from collections import deque

def distance_map(neighbors, source_tiles, unreachable):
    distances = [unreachable] * len(neighbors)
    queue = deque()
    for tile in source_tiles:
        if distances[tile] != 0:
            distances[tile] = 0
            queue.append(tile)
    while queue:
        tile = queue.popleft()
        for other in neighbors[tile]:
            if distances[other] == unreachable:
                distances[other] = distances[tile] + 1
                queue.append(other)
    return distances
```

`neighbors` must encode the actual movement graph: walls, one-way restrictions if any, and wrap/tunnel connections. Multi-source BFS is `O(V + E)` for `V` walkable tiles and `E` tile connections, regardless of how many source pellets or ghosts you start with. If movement rules depend on entity type, maintain the appropriate adjacency or eligibility rules for each query.

## Suggested game-facing API

```python
class ObservationProvider:
    def __init__(self, static_board):
        self.neighbors = build_tile_adjacency(static_board)
        self.corridors = build_corridors(static_board)  # optional
        # Maintain versioned lazy caches for dynamic distance maps.

    def for_legal_moves(self, state) -> dict[str, dict[str, float]]:
        # Read-only: map each currently legal move to its feature dictionary.
        ...
```

The game owns authoritative state: traversability, pellet locations, entity positions/modes, collision and scoring. The observation provider reads it and handles cached derived data. The GP sees only candidate feature dictionaries. Make `for_legal_moves` return no illegal candidates rather than asking GP to discover wall collisions.

## GP selection and evaluation details

- One DEAP `PrimitiveTree` represents one move-scoring expression. Evaluate it once per legal candidate each tile step; choose the greatest finite result.
- With a single tree, `len(individual)` is its node count for parsimony. If you retain a wrapper such as a list of trees, use `sum(len(t) for t in individual)` instead.
- Add `is_reverse` and prefer continuing forward on equal scores, but do not ban reversals: escaping a ghost may require one.
- Use safe arithmetic (`protected_div`, bounded `sqrt`, bounded inputs) and handle NaN/nonfinite scores deterministically.
- Compare controllers on the **same evaluation seeds and starting conditions**. Track score, pellets cleared, deaths, reversals, and repeated-tile loops in addition to tree size.

## Implementation order and checks

1. Build and test tile adjacency. Verify walls, tunnels, and that BFS distances agree with hand-counted routes.
2. Add cached ordinary-pellet and ghost distance maps. Check that consuming a pellet and moving a ghost invalidate only the relevant maps.
3. Add `for_legal_moves` and the first seven simple features in the table. Print observations at a corridor, junction, and dead end.
4. Add corridor segmentation and the final three initial features if useful; check counts *ahead* in both directions when Pac-Man is mid-corridor.
5. Replace four directional action trees with one shared edge/move-scoring tree, leaving the game stepping unchanged. Run a random-tree smoke test, then GP training.
6. Profile a complete evaluation generation. Add optional features and optimized caching only when measurement or specific failure cases justify them.

The simplest reliable baseline is **tile adjacency + multi-source BFS maps + one shared scoring tree**. Corridor graph features can be layered in without changing movement or the controller interface.
