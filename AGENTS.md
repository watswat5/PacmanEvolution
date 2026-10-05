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

`EXPERIMENTS.md` is the experiment log: every run by phase, and the fitness
function each phase used. **Add to it when the fitness formula or the game
rules change**, because fitness values are not comparable across those changes.

## Layout

- `PacmanGame/pacman.py`: deterministic game (`PacmanGame.step`) following level 1 of the arcade game. See "Arcade rules" below.
- `PacmanGame/board_graph.py`:
  - `StaticBoardGraph`: per-layout precompute, cached with `lru_cache`. It holds adjacency, all-pairs distances, corridors, `corridor_exits` and `pocket_depths`.
  - `ObservationProvider`: per-game dynamic state, including pellet sets, cached BFS maps and Pac-Man's visit history.
  - `MOVE_FEATURE_NAMES` is the **single source of truth** for terminal names and order.
- `GeneticAlgo/gp_core.py`: shared GP helpers. `build_primitive_set` is used by the 4-tree line. `build_move_primitive_set` is used by the shared tree.
- `GeneticAlgo/train_board_graph.py`: shared-tree trainer. Its `build_primitive_set()` wraps `build_move_primitive_set(FEATURE_NAMES)`. Always use this wrapper so training, `watch_board_graph.py`, `plot_board_graph.py` and the tests agree.
- `tests/`: run with `python -m unittest discover -s tests` from the project root.

## Arcade rules (`PacmanGame/pacman.py`)

The game follows **level 1 of the arcade Pac-Man** as closely as a
one-tile-per-turn grid allows. One turn is one Pac-Man tile. Its length in
60 Hz frames depends on his speed and what he ate, and each ghost then advances
by its own speed over that time (a per-ghost fractional accumulator). All the
numbers are constants at the top of `pacman.py`.

| Rule | Value |
| --- | --- |
| Pac-Man speed | 80% (90% while ghosts are frightened); pauses 1 frame per dot and 3 per energizer, so ghosts gain on him while he eats |
| Ghost speed | 75%, so about 15 tiles per 16 Pac-Man turns; 50% frightened; 40% in the side tunnel |
| Scatter / chase | Scatter 7 s, chase 20 s, scatter 7, chase 20, scatter 5, chase 20, scatter 5, then chase for good. About 53 turns of scatter and 152 of chase. The clock pauses while ghosts are frightened. Every switch makes active ghosts reverse |
| Targets | Blinky: Pac-Man. Pinky: 4 tiles ahead. Inky: 2 tiles ahead, mirrored through Blinky. Clyde: Pac-Man when 8 or more tiles away, else his corner. Facing up also shifts the "ahead" point left by the same amount (the arcade overflow bug). Scatter corners are top-right, top-left, bottom-right, bottom-left |
| Turning | Ghosts pick the open tile closest to the target by straight-line distance, ties broken up, left, down, right. They never reverse unless forced or signalled |
| Frightened | 6 s (`POWERUP_DURATION` = 52 turns). Ghosts reverse, then turn pseudo-randomly. Each ghost is frightened individually: an eaten ghost comes back dangerous while the others are still edible |
| Eaten ghosts | Moved to the house at once and held for the time its eyes would take to travel back (eyes speed is an approximation), then leave again |
| House release | Blinky starts outside, Pinky leaves at once, Inky after 30 dots, Clyde after a further 60. The next ghost is released anyway after 4 s without a dot. Ghosts leave heading left |
| Cruise Elroy | Blinky runs at 80% with 20 dots left and 85% with 10, and keeps chasing during scatter |
| Original maze only | Ghosts may not turn up at the four arcade "no-up" tiles |

Ghosts may share a tile. `PacmanGame.ghost_mode` reports scatter or chase.

`Board` records which ghosts are frightened in `inky_scared`, `blinky_scared`,
`pinky_scared` and `clyde_scared`; use `Board.ghost_scared` to read them. A
value of `None` (boards built by hand or files without those keys) means the
ghost follows `player_powerup`. The game always writes explicit values, and
`write_board` / `read_board` / `render_board` and the observation features all
use them.

Not modelled: lives (one life per game), levels beyond 1, fruit, arcade point
values, cornering, and the eyes' journey as a visible sprite. Timers and
per-ghost state live in `PacmanGame`, not in `Board`, so a game loaded from a
saved board file restarts its scatter/chase clock and house release.

The game is still deterministic: frightened ghosts use a generator seeded the
same way every game.

## Shared-tree observation contract

`game.get_move_observations()` returns `{action: {feature: float}}` for
**legal moves only**. It is read-only. All values are bounded:

- Distances are path distances from the *candidate* tile, divided by `distance_scale` (longest shortest path + 1). They lie in [0, 1), and **exactly 1.0 means "no such target / unreachable"**.
- Flags and fractions lie in [0, 1]. `pellet_center_direction` and `junction_ghost_margin` lie in [-1, 1].

| Feature | Meaning |
| --- | --- |
| `pellet_here` | Candidate tile has an ordinary pellet |
| `pellet_distance`, `power_pellet_distance` | Nearest remaining (power) pellet |
| `danger_ghost_distance` | Nearest ghost that can kill Pac-Man (1.0 if none). Frightened ghosts and ghosts waiting in the house do not count; a ghost that was eaten and has come back does |
| `second_danger_ghost_distance` | Second-nearest dangerous ghost. This catches pinches |
| `danger_ghosts_nearby` | Dangerous ghosts within `GHOST_NEAR_RADIUS` (8) path steps, divided by 4 |
| `edible_ghost_distance` | Nearest frightened ghost (1.0 if none) |
| `frightened_time` | `player_powerup_turns / POWERUP_DURATION` |
| `is_reverse`, `is_forward` | Move relative to the last direction |
| `candidate_degree` | Exits from the candidate tile, divided by 4 |
| `pellets_ahead` | Pellet fraction of the corridor this move enters |
| `steps_to_junction` | Corridor length to the next decision tile (clamped to 1) |
| `escape_options` | Continuations at the end of that corridor, excluding the way back, divided by 3. It is 0 at a dead end |
| `dead_end_depth` | If the move crosses a bridge into the *smaller* side of the maze (a pocket with one way out), the pocket's depth; otherwise 0 |
| `recently_visited` | 1.0 for the tile just left, fading linearly to 0 after `VISIT_WINDOW` (12) steps |
| `pellet_center_direction` | Cosine between the move and the direction to the collectibles' center of mass |
| `junction_ghost_margin` | Race to the end of the corridor this move enters: nearest dangerous ghost's path distance to that junction minus Pac-Man's, divided by `distance_scale` and clamped to [-1, 1]. **Zero or less means a ghost gets there first** (a tie loses, because Pac-Man moves first and the ghost then steps onto him). 1.0 when no dangerous ghost can reach it |
| `safe_escape_options` | Ghost-aware `escape_options`: onward corridors from that junction whose *far* junction Pac-Man also reaches strictly before every dangerous ghost, divided by 3. It is 0 whenever `junction_ghost_margin` is not positive, and equals `escape_options` when no ghost is dangerous |

The two junction features use path distances, which ignore that ghosts cannot
reverse, that they are slightly slower than Pac-Man, and that ghosts still in
the house are unreachable (the gate is a wall in the graph). They therefore err
towards caution for ghosts in the maze and see nothing of ghosts that have not
left the house. No feature tells the tree whether ghosts are scattering. They look only one junction
past the current corridor. **New features must be appended to the end of
`MOVE_FEATURE_NAMES`** so previously pickled trees keep working.

Visit history is updated in `PacmanGame.step` via
`ObservationProvider.player_moved`. `reset` clears it.

## Shared-tree primitive set (`build_move_primitive_set`)

The primitives are `add sub mul div min max abs neg sqrt if_lte`. The
constants are `-1, 0, 0.5, 1, 2`.

- `div` is `bounded_divide`. It floors the denominator's magnitude at 1e-3 and clamps the result to ±1000. Unlike `protected_divide`, `1/d` keeps growing as `d → 0`, so an adjacent pellet never scores below a distant one.
- `if_lte(a, b, then, else)` lets a tree switch modes (e.g. flee vs. chase).
- `sin/cos/tan/log/dist` were dropped for this line. They are still present in `build_primitive_set` for the 4-tree line.

## Tree repair

**Off by default; turn it on with `--repair`.** When on, every new tree (the
initial population and every crossover or mutation offspring) passes through
`gp_core.repair_individual`, which collapses trivial chains: `abs(abs x)`, `abs(neg x)`, `neg(neg x)`, `sqrt(abs x)`, `x * 1`,
`x + 0`, `x - x`, `x * 0`, `min(x, x)`, sign shuffles such as `a + neg b`,
`if_lte` with constant or identical arguments, and constant-only expressions
whose value is one of the set's own constants. The full list is in
`simplify_tree`'s docstring.

- **It never changes what a tree computes** (for finite inputs), so fitness is unaffected apart from the smaller size cost. `tests/test_tree_repair.py` checks this.
- `div` is deliberately not simplified, because its denominator floor and +/-1000 clamp make `x / 1` and `x / x` inexact.
- Saved trees from earlier runs are not altered; a run resumed with `--repair` repairs offspring from then on.

## Fitness (board-graph trainer)

```
100000 * pellet completion ratio
+ 100000 * clear_bonus if the board is cleared (--clear-bonus, default 0.30)
- if died or cut off for stagnation (--stagnation-termination-steps):
    death_penalty                              (--death-penalty, default 10000)
    + death_pellet_penalty * 100000 * (1 - pellet completion ratio)
                                               (--death-pellet-penalty, default 0.25)
- stagnation_penalty per non-scoring step   (--stagnation-penalty, default 100)
- revisit_penalty per non-scoring move onto a tile visited in the last
  VISIT_WINDOW steps                         (--revisit-penalty, default 50)
- size_penalty * nodes^2 / 400               (--size-penalty, default 10)
```

The second fitness value is `-parsimony * node_count`, used only as a
tie-break. **`--parsimony` therefore does nothing to limit tree size**: averaged
fitnesses almost never tie exactly. `--size-penalty` is the bloat control. Its per-node rate is proportional to
tree size (`size_cost`): it is the flag's value at 400 nodes
(`SIZE_PENALTY_REFERENCE_NODES`) and scales down below that, so small trees are
not pushed to collapse early. One pellet is worth about 400 fitness. At the
default of 10: 50 nodes pay 62, 100 pay 250, 200 pay 1000, 400 pay 4000 and 800
pay 16000.

A plain timeout pays neither death term. With the defaults, dying at 60% of the
pellets scores 60000 - 10000 - 10000 = 40000, below a tree that is still alive
at 45% (45000).

Optional rules, all off by default (`EvaluationRules`; the defaults reproduce
the formula above exactly, which was checked by rerunning a seeded run and a
resumed run and comparing `history.csv` byte for byte):

- `--stagnation-scaling inverse`: the stagnation cutoff becomes
  `--stagnation-termination-steps / (pellets left / pellets on the whole board)`
  (`stagnation_limit`). With 50 steps: 50 on a full board, 100 at half, 200 at
  25% left, 500 at 10% left. The per-step and revisit costs are unchanged, so
  waiting still costs fitness; and the step limit is unchanged, so a long wait
  can now end as a timeout, which pays no death term.
- `--late-pellet-share S`: the pellet completion ratio is replaced by
  `pellet_value(ratio, S)` in both the pellet term and the death term. It is two
  straight lines: the first 80% of the pellets earn `1 - S` in total and the last
  20% (`LATE_PELLET_FRACTION`) earn `S`. `S = 0.5` makes the last 20% worth as
  much as the first 80% (a pellet is then worth about 4 times more late than
  early); `S = 0.2` is the linear scale. A clear is still 100000 + bonus.
- Both rules also apply to that run's tournament and validation games, so the
  `10-board` and `val` numbers of runs with different rules are not comparable.
  Compare such runs on fresh boards under one rule (`--stagnation-scaling` on
  `summarize_experiments.py`, `plot_experiment_comparison.py` and `playoff.py`).

## Training boards and champion selection

- `--boards-per-generation N` scores every tree on N shared random boards and averages. `--boards-replaced-per-generation R` keeps the newest N - R boards from the previous generation and swaps in R new ones (default: replace all N). The run scripts use N = 8, R = 2; override with `BOARD_GRAPH_BOARDS_PER_GENERATION` and `BOARD_GRAPH_BOARDS_REPLACED`.
- Each generation, the `--champion-candidates` (default 10) best *distinct* trees by training fitness are scored on the 10 fixed tournament boards. The best of those is the generation champion: it is saved as `generation_NNNN_best.pkl`, run on the validation board, and tracked for `best_overall.pkl`. The `seed=` in the log line is the oldest board still in rotation.
- The run scripts use `--timestep-limit 1000`. At 500, near-clears were running out of steps.
- `--endgame-boards E` makes E of the N training boards endgame starts (default 0); the other N - E are full games. `--endgame-start` (default 0.75) is the fraction of pellets already eaten. Tournament, validation, fresh-board and playoff games are always full games.
  - `endgame_board()` builds one from the same layout seed with a ghost-free walker that starts at Pac-Man's start and always walks to the nearest pellet. Pac-Man starts where the walker stopped, usually next to the remaining pellets. Both styles are deterministic and leave the same number of pellets.
  - `--endgame-styles` (default `sweep`; the endgame experiment uses `sweep,islands`) lists the styles, and successive endgame boards take them in turn (`endgame_style_for`). `sweep`: the walker eats until 75% are gone, leaving whole corridors, mostly in one or two connected regions. `islands`: two to four connected patches near maze corners are set aside first (`_island_pellets`; the layout seed picks which corners, and the patches take turns growing outwards from the pellet nearest each corner) and the walker eats everything else, going around the patches where it can.
  - The pellet ratio is relative to the pellets on the board at the start, so an endgame scores 0 where it starts and the same 100000 + bonus as a full game when cleared. The death term is relative too.
  - `pellets_eaten` on the board records the head start. `PacmanGame` seeds its ghost-house dot counters from it, so all four ghosts leave the house at once, and inverse stagnation scaling still sees the whole board (200 steps at the start of an endgame). Cruise Elroy depends only on pellets left, so it behaves as in a real endgame.
  - Approximations: the scatter/chase clock starts from 0 (a real endgame would be in a later chase period) and ghosts start at home. A sweep leaves about 0.5 energizers per board and islands about 2.9 (averages over 30 boards), because the energizers sit near the corners.
  - `is_endgame_slot()` decides which boards in the run's board sequence are endgames; the pattern repeats every N boards, so the rotating set always holds exactly E. The checkpoint stores each board's style (`board_endgame`, `""` for a full game).
  - The training `C/D/S/T` counts and the training fitness then mix both kinds of game.

## Checkpoints and the overnight script

- The trainer writes `checkpoint.pkl` in the output directory after every generation (population, random state, history, best tree, training-board seeds, tournament seeds). `--resume` continues from it, or starts a new run if there is none. A resumed run reproduces the uninterrupted run exactly; this was checked for both evolution methods by killing a run mid-way.
- Resuming refuses to continue if seed, population size, evolution method, board settings or step limit differ from the checkpoint. `--generations` may be raised to extend a run.
- A finished run writes `summary.txt`. Resuming a finished run changes nothing and does not redo the plots or video.
- `run_board_graph_overnight.sh` runs several seeds one after another (`BOARD_GRAPH_SEEDS`, default `42 43 44 45 46`; `BOARD_GRAPH_GENERATIONS`, default 400, raised from 200 on 2026-10-03), each in `runs/board_graph_overnight_*/seed_N/`, sharing one `training.log`. After a crash or reboot run `BOARD_GRAPH_RESUME=runs/<directory> ./run_board_graph_overnight.sh`: finished seeds are skipped and the interrupted one continues. Ctrl-C stops the current seed cleanly and skips the rest.
- The overnight script uses cellular evolution on a 20x20 grid (`BOARD_GRAPH_POPULATION`, default 400), `--stagnation-penalty 20` (`BOARD_GRAPH_STAGNATION_PENALTY`) and 30% mutation; the other run scripts still use 100 and 10%. Overnight runs made with the earlier global-method version of the script cannot be resumed with it.

## Island model

`run_board_graph_islands.sh` runs several populations (one per seed, default
`42 43 44 45 46`) as islands. Each island is an ordinary cellular run with the
overnight settings in `runs/board_graph_islands_*/seed_<seed>/`. The script
advances every island `BOARD_GRAPH_MIGRATION_INTERVAL` generations (default 20),
then runs `migrate_islands.py`, and repeats up to `BOARD_GRAPH_GENERATIONS`
(default 300).

- **Migration:** each island sends `BOARD_GRAPH_MIGRANTS` (default 1) randomly chosen trees to every other island, where they replace the worst trees by that generation's fitness. Migrants are picked before any arrivals, arrive unscored, and are scored with everyone else at the start of the next generation. With 5 islands that is 4 arrivals per island per round (1% of 400).
- **How it is done:** migration edits the islands' `checkpoint.pkl` files between `--resume` calls; the trainer itself knows nothing about islands. `island_state.json` records finished rounds so a round is never repeated, and `migrations.csv` logs every move.
- **Shared tournament boards:** the script passes `--tournament-seed 424242`, so all islands score champions on the same 10 boards. Those boards differ from the ones independent runs use (which derive from each run's seed), so compare island and independent runs on fresh boards, not on the 10-board score.
- **Resume:** `BOARD_GRAPH_RESUME=runs/<directory> ./run_board_graph_islands.sh`. Migration refuses to run unless every island is at the same generation.
- **Compiled charts:** every `BOARD_GRAPH_GRAPH_INTERVAL` generations (default 100) and at the end, `plot_experiment_comparison.py <run> --islands` writes one chart per metric to `<run>/comparison_gen_NNN/`, with a line per island, their mean in black, and dotted lines at the migration rounds. Each snapshot keeps its own folder. The same command works by hand at any time, and on an independent multi-seed run such as the overnight folder.
- Compute equals running the seeds independently. Only the last stretch renders videos.
- Tested end to end at toy size (3 islands, 16 trees, 6 generations) including stop and resume; not yet run at full size.

## Comparing runs with each other

`compare_runs.py LABEL=DIR [LABEL=DIR ...] --through N --output DIR` compares
multi-seed runs over generations 0 to N-1, using only the seeds they share: one
line per run (the mean of its seeds) per metric, plus summary dot plots that
show every seed. The first run is the reference, drawn in black. Tournament
score charts are left out unless `--tournament` is given, because island runs
use different tournament boards from independent runs. `--fresh-boards N` also
plays each seed's best champion from that span on N fresh boards.

    python compare_runs.py "separate seeds=runs/<overnight run>" "islands=runs/<island run>" \
        --through 100 --fresh-boards 30 --output runs/<island run>/comparison_vs_separate_gen_100

## Playoff: choosing the most general tree

`best_overall.pkl` is only the tree with the highest score on a run's 10 fixed
tournament boards, out of thousands of candidates, and it is never re-tested.
It is therefore partly a lucky fit to those boards. `playoff.py` picks a winner
fairly after the fact:

    python playoff.py runs/<run or folder of runs> [more runs ...] --top 10 --boards 100

- From every run it takes the `--top` best distinct generation champions by tournament score (the same tree is often champion for many generations).
- All candidates play the same `--boards` fresh boards (seeds from 2,000,000,000, above any training or tournament seed). Ranking is by mean fitness at the overnight step cost; `--rank-by pellets` or `cleared` changes it.
- The winner then plays `--final-boards` separate boards (seeds from 2,100,000,000). **Quote that result**, not its score on the selection boards.
- Output: `playoff.csv` (every candidate, ranked, with its run, generation, size and outcomes) and `playoff_winner.pkl`. It also prints where each run's saved `best_overall` placed.
- Passing an independent run and an island run together compares the two methods on the same boards.

## Comparison experiments

`run_board_graph_conditions.sh` runs a list of conditions for a list of seeds
and prints a comparison table. Each condition is the baseline with one setting
changed. The baseline is the overnight configuration scaled down to fit many
runs in a night: cellular 15x15 (225 trees), 100 generations, 8 boards with 2
rotating, step cost 20, 30% mutation, repair off, no video.

| Condition | Change from the baseline |
| --- | --- |
| `baseline` | none |
| `repair` | `--repair` |
| `mutation10` | 10% mutation, 90% crossover |
| `step100` | `--stagnation-penalty 100` |
| `small_inserts` | `--mutation-min-depth 0` (mutation may insert a single node) |
| `global` | global over-selection instead of the cellular grid |
| `grid8x8` | 64 trees |
| `no_junction` | `--exclude-features junction_ghost_margin,safe_escape_options` |

- `run_board_graph_repair_comparison.sh` runs `baseline` and `repair`; `run_board_graph_ablation.sh` runs all eight. Both use 3 seeds (`BOARD_GRAPH_SEEDS`, default `101 102 103`) and write to `runs/experiment_ablation/<condition>/seed_<seed>/` (`BOARD_GRAPH_EXPERIMENT_DIRECTORY`), so the baseline is only run once between them. Do not run the two at the same time.
- The same seed gives every condition the same training and tournament boards. Seeds are the outer loop, so each pass leaves every condition with one more finished seed.
- Rerunning the same command resumes: finished runs are skipped. Ctrl-C stops cleanly.
- `plot_experiment_comparison.py <directory> --seed N` writes one chart per metric to `<directory>/comparison_seed_N/`, each with one line per condition: best, average and worst training fitness, champion fitness and pellets on the tournament boards, cleared/died/stagnated rates, average and best tree depth, average and best node count. Noisy per-generation metrics are a 5-generation rolling mean (`--smooth`). A condition keeps the same colour in every chart. `--seed all` (the default) plots the mean over seeds into `comparison_all_seeds/` and adds `summary_*.png` dot plots showing every seed's end result beside the condition mean; `--fresh-boards 30` adds the fresh-board summaries, cached in `fresh_boards.csv`.
- `summarize_experiments.py <directory> [--fresh-boards N]` prints the table (and writes `summary.csv`) at any time, including mid-run. With `--fresh-boards`, each run's best tree also plays N random boards no run was trained or selected on.
- Estimated time, from the overnight run's timings scaled by population: about 42 minutes per 225-tree cellular run, so about 4 hours for the repair comparison and about 14 hours for the full ablation. Untested at full size.
- The arcade-rules change cannot be ablated this way; the old game rules are no longer in the code.

### Endgame experiment (`run_board_graph_endgame_experiment.sh`)

Asks which single change raises the clear rate, after the playoff showed every
champion stuck at 80-88% of pellets and about 6 clears per 100 boards. Same
engine as above, but the baseline is the full overnight configuration with the
depth limit raised from 16 to 20 (`BOARD_GRAPH_MAX_DEPTH`): cellular 20x20 (400
trees), 200 generations, seeds `42 43 44`, directory `runs/experiment_endgame`.

| Condition | Change from the baseline |
| --- | --- |
| `baseline` | none (depth limit 20) |
| `scaled_stagnation` | `--stagnation-scaling inverse` |
| `late_pellets` | `--late-pellet-share 0.5` |
| `endgame` | 5 full boards + 3 endgame boards (`BOARD_GRAPH_FULL_BOARDS`, `BOARD_GRAPH_ENDGAME_BOARDS`), still 8 in total with 2 rotating; endgames alternate between `sweep` and `islands` (`BOARD_GRAPH_ENDGAME_STYLES`) |

- Depth 20 is the default of this experiment, not a condition: all four runs per seed use it. `runs/experiment_endgame/baseline_depth16/seed_42..44` is a reference only: the first 200 generations of the overnight run `board_graph_overnight_20261002_012238_GTc622` (depth limit 16), copied with `truncate_run.py SOURCE DESTINATION --through 200` (history, generation champions, and `best_overall.pkl` set to the best tournament champion of those generations). Those runs used the same seeds and generation count, hence the same training layouts and tournament boards, so `baseline` against `baseline_depth16` shows the effect of the depth limit alone. The copies have no checkpoint and are never trained. The other run scripts (overnight, islands, ablation) still default to depth 16.
- Compare conditions on fresh boards under one stagnation rule: `python summarize_experiments.py runs/experiment_endgame --fresh-boards 30 --stagnation-scaling inverse` (the script's final table does this), `python plot_experiment_comparison.py runs/experiment_endgame --fresh-boards 30 --stagnation-scaling inverse` (cached in `fresh_boards_inverse.csv`), and `python playoff.py runs/experiment_endgame --stagnation-scaling inverse`. The training-fitness and 10-board charts are not comparable across these conditions: `late_pellets` changes the scale, `scaled_stagnation` changes the rule the tournament is played under, and `endgame` mixes endgame games into the training numbers.
- Estimated time: about 2.8 hours per run from the overnight timings, so about 34 hours for the 12 runs. All of them may be slower than that because of the deeper trees, and `scaled_stagnation` because of longer games. Untested at full size.

## Attaching to a running run

`./attach_board_graph_run.sh [runs/<directory>]` hooks back into a run that is
already going in the background, for example after its terminal was closed. It
follows the log; Ctrl-C stops training cleanly (same as in the launcher) and `q`
stops following without stopping training. `--stop` stops a run without
following and `--status` reports whether it is running. With no directory it
picks the run whose log changed most recently.

It finds the run through the `BOARD_GRAPH_DETACHED` environment the launchers
set, so it works for every `run_board_graph_*.sh` script and needs no PID file.
It must be run on the machine itself (a terminal or SSH); a sandboxed shell
cannot see or signal the training processes.

## Watching one tree on the original maze

`./play_individual.sh FILE.pkl [more.pkl ...]` plays a saved tree
(`generation_NNNN_best.pkl`, `best_overall.pkl`, `playoff_winner.pkl`) on the
original maze, prints how the game ended and the pellets eaten, saves
`<name>_original_maze.mp4` next to the file and opens it with `xdg-open`. No
stagnation cutoff; the game ends on death, a clear or the step limit
(`PLAY_MAX_STEPS`, default 1000). Also `PLAY_FPS`, `PLAY_SCALE`, `PLAY_OPEN=0`.

For drag and drop there is `Play Pac-Man individual.desktop`, which runs the
script on the dropped file in a terminal. It holds the absolute path of the
project folder, so edit its `Exec=` line if the folder moves. Whether a file
manager accepts drops on a launcher depends on the desktop; dragging the file
into a terminal after typing `./play_individual.sh ` works everywhere. The
launcher has not been tested on the real desktop.

`GeneticAlgo.watch_board_graph` now also prints a `Result:` line.

## Generation grid video

`python -m GeneticAlgo.watch_generations_grid runs/<run directory>` renders
every `generation_NNNN_best.pkl` in a run playing the same board at once, one
cell per generation in order (a 10x10 grid for 100 generations). A cell freezes
and is tinted when its game ends: red for a death, green for a clear, amber for
a stagnation cutoff. Each cell is labelled with its generation and, once
finished, the share of pellets eaten.

- Output: `<run>/generation_grid.mp4` and the final frame as `generation_grid.png` (`--video` changes the path).
- Default board is the original maze; `--board` picks another. `--every N` and `--limit N` thin out long runs, `--columns` sets the grid width, `--tile-size` the pixels per maze tile (default 6), `--stagnation-steps 0` disables the amber cutoff.
- Games are simulated in parallel worker processes and drawn as flat colored blocks, not sprites. 100 champions at 1,000 steps took about 20 seconds on 2 cores.
- It is a standalone script; the trainer does not call it.

## Gotchas

- **Runs from before the arcade-rules change (2026-10-02) used a different game**: ghosts at Pac-Man's speed, always chasing, half speed while powered. Their trees still load, but results are not comparable.
- The autoencoder line still normalizes the power-up timer by 40 (`train_pacman.py`); the fright now lasts 52 turns.
- Dependencies: everything is in the root `requirements.txt` (deap, numpy, pillow, matplotlib, tensorflow). System packages are ffmpeg (videos) and python3-tk (interactive play). Pillow is imported lazily in `render_board.py`, so the game and training import without it.
- **Pickled individuals from before the 2026-10-01 feature/primitive change cannot be replayed.** Their terminals and primitives no longer exist in the primitive set. Retrain. Trees from after it still load, because later features were appended.
- Fitness values are not comparable across the 2026-10-01 fitness changes (clear bonus, stagnation penalty, scaled death penalty). Compare pellet ratios and C/D/S/T instead.
- The validation board is `PacmanGame/test_board.txt`. It is the original arcade maze: 28x31, 240 pellets plus 4 power pellets, a wrap tunnel on row 14, and the same spawn/gate/house coordinates as the tetris generator. Regenerate it with `python -m PacmanGame.generate_original_board`. The layout lives in `ORIGINAL_LAYOUT`.
- `plot_board_graph.py` colors terminals with `tab20`. Keep `MOVE_FEATURE_NAMES` at 20 entries or fewer, or extend the colormap. There are 19 now.
- On the generated "tetris" boards, `dead_end_depth` is often 0 everywhere because many layouts have no dead ends. That is expected.
- Static precompute is about O(V²) per layout (all-pairs BFS plus one BFS per directed edge for pockets). That is about 0.05 s per board at about 370 tiles. Each graph is about 1.8 MB, so the per-process cache is capped at 32 layouts (`STATIC_GRAPH_CACHE_SIZE`). Raise it if a run uses more than about 20 training boards at once.

## Change log

- **2026-10-05: `play_individual.sh`, drag-and-drop launcher, playoff `--through`**
  - `play_individual.sh` and `Play Pac-Man individual.desktop` (see "Watching one tree on the original maze"). `watch_board_graph` prints the outcome and pellets eaten.
  - `playoff.py --through N` limits candidates to generations 0 .. N-1, for runs of different lengths.
  - Separate seeds 43-46 were extended to 300 generations (seed 42 has 400). Islands against separate seeds at generation 300, mean of seeds 42-46: average training fitness 72,167 against 73,618; training clear rate 10.0% against 11.6%; deaths 72.9% against 65.2%; each seed's best champion on 30 fresh boards 85.5% of pellets against 84.4%, 1.8 against 3.6 boards cleared. The island lead in the earlier playoff was mostly the extra 100 generations. Charts: `runs/board_graph_islands_20261003_161020_3yflrj/comparison_vs_separate_gen_300/`.
  - The playoff winner (separate seed 46, generation 158) dies after 110 steps with 37.7% of the pellets on the original maze, although it averages 84% on random boards.

- **2026-10-04: endgame experiment prepared (scaled stagnation, late-pellet weighting, endgame boards)**
  - Why: in the playoff of 100 champions from the separate-seed and island runs, every tree landed at 80-88% of pellets with about 6 clears and 70-77 deaths per 100 boards, regardless of generations or migration. At a 6% clear rate a tree clears zero or one of its 8 training boards, so selection cannot tell a better clearer from a lucky one; and the 50-step stagnation cutoff scores waiting for an opening near the end the same as dying.
  - Trainer: `--stagnation-scaling inverse`, `--late-pellet-share`, `--endgame-boards`, `--endgame-start`, `--endgame-styles` (see "Fitness" and "Training boards"). All default to the previous behaviour. Checkpoints record the new settings and refuse to resume with different ones; checkpoints from before the change resume as before.
  - `PacmanGame._reset_dynamics` seeds the ghost-house dot counters from `board.pellets_eaten`. Boards that start with 0 eaten (every board until now) behave exactly as before.
  - `summarize_experiments.py`, `plot_experiment_comparison.py` and `playoff.py` take `--stagnation-scaling`. New `truncate_run.py`.
  - `run_board_graph_conditions.sh` gains the conditions `scaled_stagnation`, `late_pellets`, `endgame`, and the settings `BOARD_GRAPH_MAX_DEPTH` and `BOARD_GRAPH_SUMMARY_FLAGS`; new wrapper `run_board_graph_endgame_experiment.sh`.
  - Tests: `tests/test_endgame_rules.py` (47 tests pass in total). A seeded 3-generation run and a resume of an old checkpoint give byte-identical `history.csv` before and after the change. Each new condition was smoke-run for 3 generations at 16 trees only; nothing has been trained with these settings yet.
  - Known loose ends: under scaled stagnation a tree can loiter until the 1000-step limit and avoid the death term (it still pays 20 per step and 50 per revisit); the experiment's three seeds can only show a large effect.
- **2026-10-04: playoff of both runs; island run finished 300 generations**
  - Islands, best champion on the shared 10 tournament boards: 83,828 / 91,324 / 86,669 / 86,437 / 83,357 (seeds 42-46). Four of the five were set after generation 270. Mean of islands per 50 generations, 150-199 / 200-249 / 250-299: average training fitness 67,992 / 70,240 / 71,420; champion tournament fitness 70,567 / 72,300 / 75,468. Training clear rate reached 8-10% in the last 50 generations.
  - Separate seeds continued: seed 42 reached 82,976 by generation 265 and was flat to 400; seed 43's last new best (81,813) was at generation 241.
  - `python playoff.py <separate run> <island run> --top 10 --boards 100` (results in `runs/playoff.csv`): winner is separate seed 46, generation 158, 109 nodes. On the 100 selection boards 88.9% of pellets and 10 clears; on 100 further fresh boards 84.4%, 6 cleared, 60 died, 34 stagnated, fitness 68,686 (the figure to quote).
  - Island candidates averaged 66,413 playoff fitness against 63,850 for separate seeds, with 32 of the top 50; each seed's best ranged 68,298-73,394 (islands) against 59,527-75,460 (separate). Pellets 82.2% against 80.2%; clears per 100 boards 5.9 against 5.8. Confounded by run length: islands all ran 300 generations, separate seeds 44-46 only 200 and seed 43 272 at the time.
  - The separate run's candidates scored 16,600 lower on the 100 playoff boards than on their 10 tournament boards on average, and every separate seed's `best_overall.pkl` ranked 52nd or worse of 100. Do not quote tournament scores as generalisation.

- **2026-10-04: islands against separate seeds at generation 200**
  - Still no measurable difference after ten migration rounds. Generations 180-199, mean of seeds 42-46: average training fitness 68,068 (islands) against 66,677 (separate); best training fitness 85,577 against 86,141; training clear rate 5.5% against 7.8%. Each seed's best champion on 30 fresh boards: 81.2% of pellets against 81.6%, 1.2 against 1.6 boards cleared.
  - The islands remain more uniform (average training fitness 67,000-70,400 across islands against 57,500-72,800 across separate seeds).
  - Migrants leave only a small trace: island champions at generation 199 share about 3% of their 6-node sequences with each other on average (separate seeds share none), so the islands have not merged into one family.
  - Charts: `runs/board_graph_islands_20261003_161020_3yflrj/comparison_vs_separate_gen_200/`.
- **2026-10-04: island script depth cap is now a setting (still 16)**
  - `run_board_graph_islands.sh` takes `BOARD_GRAPH_MAX_DEPTH` (default 16). A cap of 20 was set briefly and reverted before any run used it, so the island run `board_graph_islands_20261003_161020_3yflrj` has had a cap of 16 throughout and stays comparable with the separate-seed run. Testing 20 is planned as a separate run.
- **2026-10-03: islands against separate seeds at generation 100**
  - Added `compare_runs.py`. For seeds 42-46 the island run and the independent run are identical for the first 20 generations (checked), then diverge at the first migration.
  - Through generation 100 (five migration rounds) there is no measurable difference: average training fitness over generations 80-99 is 54,328 for islands against 55,014 for separate seeds, and each seed's best champion ate 70.0% against 70.6% of pellets on 30 fresh boards. The islands' seeds are closer together (66-73% against 62-81%). Charts are in `runs/board_graph_islands_20261003_161020_3yflrj/comparison_vs_separate_gen_100/`.
- **2026-10-03: playoff script**
  - Added `playoff.py`; see "Playoff: choosing the most general tree". Tested on the overnight run at toy size only (2 candidates per seed, 6 boards).
- **2026-10-03: island run defaults to 300 generations with charts every 100**
  - Why: seed 42 of the independent run plateaued at about generation 200 (best champion 80,976 at generation 192, 82,976 at 265, nothing better through 400), so 400 generations is not worth the time.
  - `run_board_graph_islands.sh` now stops at 300 and compiles all-island charts at 100, 200 and 300; `plot_experiment_comparison.py` gained `--islands`.
- **2026-10-03: island model**
  - Added `run_board_graph_islands.sh`, `migrate_islands.py` and the trainer flag `--tournament-seed`; see "Island model". Planned use: rerun seeds 42-46 from generation 0 as islands and compare with the independent 400-generation run of the same seeds.
- **2026-10-03: overnight script runs 400 generations**
  - `run_board_graph_overnight.sh` now defaults to 400 generations per seed, because the 200-generation seeds were still improving. Resuming `runs/board_graph_overnight_20261002_012238_GTc622` with `BOARD_GRAPH_RESUME` continues each seed from generation 200.
- **2026-10-03: ablation complete (3 seeds); the seed-101 reading was wrong**
  - Full tables and findings are in `EXPERIMENTS.md`, Phase 7. The only clear effect is cellular over global selection. 10% mutation, which led by 30,000 after one seed, ties the 30% baseline over three (75,973 against 75,933); the junction features, repair, small inserts and the 8x8 grid show no established effect either. Seeds within one condition differ by 10,000 to 31,000.
  - The earlier entry below ("ablation seed 101 results recorded") is superseded. The run scripts stay at 30% mutation and cellular.
  - `plot_experiment_comparison.py` gained `--seed all` mean charts and per-seed summary dot plots.
- **2026-10-03: fixed memory growth in the evaluation workers**
  - Symptom: each worker grew from about 50 MB to about 0.5 GB over a 100-generation run (about 0.9 GB projected for 200 generations), times 16 workers. This is the likely cause of the machine hanging after the long overnight run.
  - Cause 1: the static board-graph cache was unbounded and every generation brings new layouts at about 1.8 MB each. Now capped at 32 layouts.
  - Cause 2: compiled trees were cached with the primitive-set object as part of the key; workers get a new copy with every task, so the cache never hit across tasks and kept those copies alive. Now keyed by expression and argument names.
  - Measured after the fix: flat at about 90 MB after 120 boards (was 141 MB after 60 and still climbing). Results are unchanged; only caching differs.
- **2026-10-03: ablation seed 101 results recorded**
  - Full table and findings are in `EXPERIMENTS.md`, Phase 7. In short, for one seed: the cellular grid clearly beats global selection; 10% mutation beat the 30% baseline by about 30,000 on the tournament boards but by much less on fresh boards (79% of pellets against 64%); the junction features, repair, small inserts and the 8x8 grid showed no clear effect.
  - The run scripts still use 30% mutation. Do not change the default until seeds 102 and 103 confirm it.
- **2026-10-02: attach script**
  - Added `attach_board_graph_run.sh` to follow, stop or check a run that is already in the background; see "Attaching to a running run".
- **2026-10-02: generation grid video**
  - Added `GeneticAlgo/watch_generations_grid.py`; see "Generation grid video".
- **2026-10-02: repair off by default; comparison experiment scripts**
  - `--repair` replaces `--no-repair`; repair is now opt-in until the comparison shows whether it helps.
  - Added `--mutation-min-depth` and `--exclude-features`, so the ablation conditions can be expressed as flags.
  - Added `run_board_graph_conditions.sh`, `run_board_graph_repair_comparison.sh`, `run_board_graph_ablation.sh` and `summarize_experiments.py`; see "Comparison experiments".
- **2026-10-02: tree repair step**
  - Why: the overnight populations were full of trivial chains (`abs(abs x)`, `abs(neg x)`, chains of `mul` by 1) that add depth and nodes without changing behaviour, while 12-26% of trees sat at the depth limit.
  - Added `simplify_tree` / `repair_individual` in `gp_core.py`, applied to the initial population and to every offspring when `--repair` is given (off by default).
  - On the four overnight final populations (1,600 trees) repair cut mean size by 5-17% and the number of trees at depth 16 from 328 to 185, with identical outputs on every tested input. Its effect on evolution (it removes neutral code that crossover could otherwise hit) is untested.
- **2026-10-02: final video uses the original maze**
  - `best_overall.mp4` is now always rendered on `PacmanGame/test_board.txt` (the original arcade layout) instead of the first tournament board. `best_overall_board.txt` is no longer written.
- **2026-10-02: larger mutation subtrees; overnight script is cellular**
  - Why: in the first overnight attempt (global, 400 trees, 30% mutation) average size fell from 30 nodes to about 7 by generation 4 and stayed near 10, with no trend in the 10-board score over 35 generations. Mutation used `genGrow` with a minimum depth of 0, which returned a single node 72% of the time (mean 2.1 nodes).
  - Mutation (`mutate_within_depth`) now inserts a subtree of depth `MUTATION_MIN_DEPTH` (1) up to the room left under `--max-evolved-depth` at the mutation point, so the smallest insert is 2 nodes and a mutation is never rejected for depth. A node already at the limit becomes a terminal.
  - `run_board_graph_overnight.sh` now uses cellular evolution (20x20 grid) for its per-cell elitism. Cellular evaluates parents and children each generation, so a generation costs about twice as much as global at the same population.
- **2026-10-02: checkpoints and a multi-seed overnight script**
  - Why: a 64-tree, 50-generation run plateaued at generation 7 on a tree that uses no ghost feature (30 deaths in 30 fresh boards, 61% pellets), and every earlier run plateaued within about 15 generations. Each non-scoring step cost 100, so a tree that dodged a ghost paid for it.
  - Added `--resume` and per-generation `checkpoint.pkl`; see "Checkpoints and the overnight script".
  - `run_board_graph_overnight.sh` now runs 5 seeds of 200 generations instead of one run of 1000, with the non-scoring step penalty cut to 20 and mutation raised to 30%. Whether these help is untested.
- **2026-10-02: Ctrl-C in the run scripts stops training cleanly**
  - The `run_board_graph_*.sh` launchers forward the first Ctrl-C to the detached trainer as an interrupt, then keep following the log while it writes the final plots and video. A second Ctrl-C stops following. Closing the terminal still leaves training running.
  - `train_board_graph.main` restores Python's SIGINT handler, because a shell starts background jobs with SIGINT ignored.
- **2026-10-02: arcade level-1 rules**
  - Why: with ghosts at Pac-Man's speed and no scatter phases, a ghost one step behind could never be shaken off. The previous champion spent 13% of its decisions in that state and died in 25 of 30 games.
  - Rewrote ghost movement and timing in `pacman.py` (see "Arcade rules"): relative speeds, scatter/chase clock with reversals, per-ghost fright with pseudo-random turns, eaten-ghost return delay, staggered house release, Cruise Elroy, arcade targeting. Ghosts may now overlap, and `read_board` accepts that.
  - `Board` gained per-ghost `*_scared` fields, saved in board files and used by the renderer and the danger/edible features; `POWERUP_DURATION` is now 52.
  - Added `--size-penalty` (default 10 per node at 400 nodes, quadratic in tree size, subtracted from the main fitness) after the previous run bloated to 350-400 nodes with `--parsimony 1.0`. Run scripts set it via `BOARD_GRAPH_SIZE_PENALTY`.
  - Added `tests/test_arcade_rules.py`.
  - The previous champion (trained on the old rules) scores about the same on 30 random boards under the new rules: 25 deaths, 5 stagnations, 64% mean pellets. Nothing has been retrained on the new rules yet.
- **2026-10-01: junction-safety features and noise/survival changes**
  - Why: the run-`board_graph_20261001_214554` champion died in 29 of 30 fresh random boards (56% mean pellets, no clears). A 6-move look-ahead veto on the same tree cut deaths to 15 of 30 and raised pellets to 78.5%, with 12 games timing out at 90-99% under the 500-step limit. The same tree ranged from 14% to 90% pellets across boards, so 3 new boards per generation was mostly noise.
  - Added the features `junction_ghost_margin` and `safe_escape_options` (`StaticBoardGraph.corridor_continuations`, `ObservationProvider._junction_safety`).
  - Added `--boards-replaced-per-generation` (rotating training boards); run scripts now use 8 boards, 2 replaced, and smaller populations to keep evaluations per generation roughly level (short 100 -> 64, long 800 -> 300, overnight 1000 -> 400, large 4000 -> 1500).
  - Run scripts now use `--timestep-limit 1000`.
  - Added `--death-penalty` and `--death-pellet-penalty` (default 0.25): death and stagnation now also cost a quarter of the value of the uneaten pellets.
  - Added `--champion-candidates` (default 10): the generation champion is the best of the top distinct training trees on the 10 tournament boards, instead of the single best training tree.
  - Verified with the unit tests and 4-generation smoke runs only. No full training run has been done with these changes, so their effect on clearing is untested.
- **2026-10-01: clear bonus, stagnation penalty and C/D/S/T outcomes**
  - `--clear-bonus` (default 0.30) adds that fraction of the full-clear score when every pellet is eaten, so a clear is worth 130000.
  - A stagnation cutoff now receives the death penalty on top of its per-step costs. A plain timeout still does not.
  - Outcomes are logged as C/D/S/T (cleared, died, stagnated, timed out); `history.csv` gains `stagnated` and `validation_stagnated` columns.
- **2026-10-01: random-board champion tournament and outcome fitness**
  - Each generation champion is evaluated on the same 10 seeded random boards; `best_overall.pkl` tracks the champion with the best average tournament fitness.
  - Death receives the death penalty; timeout does not, so timeout is preferred to death at equal pellet completion and stagnation cost.
- **2026-10-01: shared-tree refinements**
  - Added the features `second_danger_ghost_distance`, `danger_ghosts_nearby`, `frightened_time`, `escape_options`, `dead_end_depth` and `recently_visited`.
  - `steps_to_junction` is now clamped to 1.
  - Added `build_move_primitive_set` with `if_lte` and `bounded_divide`.
  - Added `--revisit-penalty`.
  - Added `tests/test_move_features.py`.
  - Made the Pillow import in `render_board.py` lazy and made the root `requirements.txt` list every dependency.
  - Added `PacmanGame/generate_original_board.py` and regenerated the missing `test_board.txt` (original maze).
  - In a smoke run (pop 64, 8 gens) a generation was about 20% slower than before. The run was too small to say anything about solution quality.
