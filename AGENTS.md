# AGENTS.md

Notes for anyone (human or agent) working in this repo. Keep this file current
when you change features, primitives, or fitness.

## What this is

CSCI 4560 project: evolve a genetic-programming controller for Pac-Man with
DEAP. There are two controller lines:

| Line | Entry point | Observation | Individual |
| --- | --- | --- | --- |
| **Board graph (current focus)** | `GeneticAlgo/train_board_graph.py` | Per-move graph features (`PacmanGame/board_graph.py`) | **One shared scoring tree**, run on every legal move; the highest score wins |
| Autoencoder (comparison) | `GeneticAlgo/train_pacman.py` | 16-D latent from `Autoencoder/` | Four trees (UP/DOWN/LEFT/RIGHT) |

The design notes are in `GeneticAlgo/strategy.md` (autoencoder line) and
`BoardGraph/pacman_gp_observation_features.md` (feature plan).

## Layout

- `PacmanGame/pacman.py`: deterministic game (`PacmanGame.step`). Pac-Man moves first, then the ghosts (ghosts move every other turn while Pac-Man is powered up).
- `PacmanGame/board_graph.py`:
  - `StaticBoardGraph`: per-layout precompute, cached with `lru_cache`. It holds adjacency, all-pairs distances, corridors, `corridor_exits` and `pocket_depths`.
  - `ObservationProvider`: per-game dynamic state, including pellet sets, cached BFS maps and Pac-Man's visit history.
  - `MOVE_FEATURE_NAMES` is the **single source of truth** for terminal names and order.
- `GeneticAlgo/gp_core.py`: shared GP helpers. `build_primitive_set` is used by the 4-tree line. `build_move_primitive_set` is used by the shared tree.
- `GeneticAlgo/train_board_graph.py`: shared-tree trainer. Its `build_primitive_set()` wraps `build_move_primitive_set(FEATURE_NAMES)`. Always use this wrapper so training, `watch_board_graph.py`, `plot_board_graph.py` and the tests agree.
- `tests/`: run with `python -m unittest discover -s tests` from the project root.

## Shared-tree observation contract

`game.get_move_observations()` returns `{action: {feature: float}}` for
**legal moves only**. It is read-only. All values are bounded:

- Distances are path distances from the *candidate* tile, divided by `distance_scale` (longest shortest path + 1). They lie in [0, 1), and **exactly 1.0 means "no such target / unreachable"**.
- Flags and fractions lie in [0, 1]. `pellet_center_direction` lies in [-1, 1].

| Feature | Meaning |
| --- | --- |
| `pellet_here` | Candidate tile has an ordinary pellet |
| `pellet_distance`, `power_pellet_distance` | Nearest remaining (power) pellet |
| `danger_ghost_distance` | Nearest ghost while **not** powered (1.0 while powered) |
| `second_danger_ghost_distance` | Second-nearest dangerous ghost. This catches pinches |
| `danger_ghosts_nearby` | Dangerous ghosts within `GHOST_NEAR_RADIUS` (8) path steps, divided by 4 |
| `edible_ghost_distance` | Nearest ghost while powered (1.0 otherwise) |
| `frightened_time` | `player_powerup_turns / POWERUP_DURATION` |
| `is_reverse`, `is_forward` | Move relative to the last direction |
| `candidate_degree` | Exits from the candidate tile, divided by 4 |
| `pellets_ahead` | Pellet fraction of the corridor this move enters |
| `steps_to_junction` | Corridor length to the next decision tile (clamped to 1) |
| `escape_options` | Continuations at the end of that corridor, excluding the way back, divided by 3. It is 0 at a dead end |
| `dead_end_depth` | If the move crosses a bridge into the *smaller* side of the maze (a pocket with one way out), the pocket's depth; otherwise 0 |
| `recently_visited` | 1.0 for the tile just left, fading linearly to 0 after `VISIT_WINDOW` (12) steps |
| `pellet_center_direction` | Cosine between the move and the direction to the collectibles' center of mass |

Visit history is updated in `PacmanGame.step` via
`ObservationProvider.player_moved`. `reset` clears it.

## Shared-tree primitive set (`build_move_primitive_set`)

The primitives are `add sub mul div min max abs neg sqrt if_lte`. The
constants are `-1, 0, 0.5, 1, 2`.

- `div` is `bounded_divide`. It floors the denominator's magnitude at 1e-3 and clamps the result to ±1000. Unlike `protected_divide`, `1/d` keeps growing as `d → 0`, so an adjacent pellet never scores below a distant one.
- `if_lte(a, b, then, else)` lets a tree switch modes (e.g. flee vs. chase).
- `sin/cos/tan/log/dist` were dropped for this line. They are still present in `build_primitive_set` for the 4-tree line.

## Fitness (board-graph trainer)

```
100000 * pellet completion ratio
- 10000 if died / terminated early
- stagnation_penalty per non-scoring step   (--stagnation-penalty, default 100)
- revisit_penalty per non-scoring move onto a tile visited in the last
  VISIT_WINDOW steps                         (--revisit-penalty, default 50)
```

The second fitness value is `-parsimony * node_count`, used only as a
tie-break.

## Gotchas

- Dependencies: everything is in the root `requirements.txt` (deap, numpy, pillow, matplotlib, tensorflow). System packages are ffmpeg (videos) and python3-tk (interactive play). Pillow is imported lazily in `render_board.py`, so the game and training import without it.
- **Pickled individuals from before the feature/primitive change cannot be replayed.** Their terminals and primitives no longer exist in the primitive set. Retrain.
- The validation board is `PacmanGame/test_board.txt`. It is the original arcade maze: 28x31, 240 pellets plus 4 power pellets, a wrap tunnel on row 14, and the same spawn/gate/house coordinates as the tetris generator. Regenerate it with `python -m PacmanGame.generate_original_board`. The layout lives in `ORIGINAL_LAYOUT`.
- `plot_board_graph.py` colors terminals with `tab20`. Keep `MOVE_FEATURE_NAMES` at 20 entries or fewer, or extend the colormap.
- On the generated "tetris" boards, `dead_end_depth` is often 0 everywhere because many layouts have no dead ends. That is expected.
- Static precompute is about O(V²) per layout (all-pairs BFS plus one BFS per directed edge for pockets). That is about 0.05 s per board at about 370 tiles.

## Change log

- **2026-10-01: shared-tree refinements**
  - Added the features `second_danger_ghost_distance`, `danger_ghosts_nearby`, `frightened_time`, `escape_options`, `dead_end_depth` and `recently_visited`.
  - `steps_to_junction` is now clamped to 1.
  - Added `build_move_primitive_set` with `if_lte` and `bounded_divide`.
  - Added `--revisit-penalty`.
  - Added `tests/test_move_features.py`.
  - Made the Pillow import in `render_board.py` lazy and made the root `requirements.txt` list every dependency.
  - Added `PacmanGame/generate_original_board.py` and regenerated the missing `test_board.txt` (original maze).
  - In a smoke run (pop 64, 8 gens) a generation was about 20% slower than before. The run was too small to say anything about solution quality.
