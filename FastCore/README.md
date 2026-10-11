# FastCore: the training games in C++

Everything in this folder is an add-on. The original trainers, run scripts,
game and renderer are untouched and still work exactly as before; FastCore
copies of the run scripts play the same runs with the training games in C++.

## What moved to C++, and what did not

`pacman_core.cpp` is a port of the arcade game (`PacmanGame/pacman.py`), the
board graph and the 23 move features (`PacmanGame/board_graph.py`; the three
evasion features only for trees that use them), tree
evaluation, and the per-game loop and fitness of
`GeneticAlgo.train_board_graph.evaluate_population_lockstep` (stagnation
cutoff and scaling, revisit cost, late-pellet weighting, clear bonus, death
terms, size cost). It plays a whole population on one board on several
threads.

Python still does everything else: DEAP evolution, board generation,
checkpoints, history, champion selection, the diversity measurement, and all
watching and rendering (`play_individual.sh`, `versus.sh`, the videos at the
end of a run), which keep using the Python game.

`backend.py` encodes trees for the engine and, through `install()`, swaps the
C++ version into the unchanged trainer module, so the trainers' own code
paths (tournaments, steady state, the board cache, the original-maze
minimiser) all use it. With `--random-tie-break` it falls back to Python.
Encoded trees are cached by node identity, and the original-maze check
compares layout keys instead of building a Python board graph per call
(together these cut per-generation overhead by about a third). Saving
(history, checkpoint, layouts) is about 2% of a generation.

## Build (once, and after changing `pacman_core.cpp`)

On the PC, in the project folder:

```
GeneticAlgo/.venv/bin/python -m pip install pybind11
FastCore/build.sh
```

It needs g++ with C++17 and the Python headers (on Ubuntu, the `python3-dev`
package for the venv's Python version). The module is built for the
interpreter that builds it: `FastCore/_pacman_core.cpython-312-...so`. Run
`python -m FastCore.check` and `python -m FastCore.check --features`
afterwards. Rebuild after every change to `pacman_core.cpp` (the evasion
features of 2026-10-09 need it); an older build refuses trees that use a
feature it does not have.

## Run

Every training script has a FastCore copy with the same settings and
environment variables:

```
FastCore/run_board_graph_steady.sh
FastCore/run_board_graph_overnight.sh
FastCore/run_board_graph_endgame_experiment.sh
FastCore/run_original_board.sh
...
```

`BOARD_GRAPH_THREADS` sets the thread count (default: all CPUs); `--workers N`
on a direct call means N threads. Direct calls:
`python -m FastCore.train_board_graph <trainer arguments>` and
`python -m FastCore.train_original_board <arguments>`. Run folders, files and
checkpoints are the same as the originals', and a run can be resumed by either
version: a cellular run started in Python and finished in C++ gives the same
history byte for byte as one that never switched.

The copies are generated: after editing an original script, run
`python FastCore/sync_scripts.py`. `attach_board_graph_run.sh` works for the
copies as for the originals.

## How close is it to the Python game?

The goal was the same rules, close enough; in practice it matches exactly so
far. `python -m FastCore.check` plays seeded random trees and saved champions
on random boards and the original maze with both engines (half the boards
with scaled stagnation and late-pellet weighting). On 693 games: same outcome
693/693, same pellets 693/693, identical fitness 693/693. Seeded steady-state,
cellular and original-maze runs gave byte-identical histories.
Speed details (2026-10-10): `if_lte` evaluates only the branch it takes, and
the nearest-pellet features scan only the pellets still on the board.
`python -m FastCore.check --features` steps both engines together and compares
every legal move and every feature at every step: 188/188 games (39,875
steps) matched, largest gap 2.2e-16.

Where it could differ: `math.hypot` is `std::hypot` (one feature,
`pellet_center_direction`, could differ in the last bit), and the corridor
walk stops after one lap on a maze ring with no junction, where the Python
version loops forever. Frightened ghosts use an exact port of Python's
`random.Random(0)`. Compiled with `-ffp-contract=off` so `a*b+c` rounds as in
Python.

## Speed

In the agent's sandbox (2 CPUs), the check's games ran about 100x faster on
one thread and 150x on two. A whole run speeds up less, because evolution,
tournaments, files and the diversity measurement stay in Python: a 36-tree
cellular generation took about 1.5 s against about 6.6 s.

## Keeping the two in step

The C++ engine is a copy of the rules. Any change to `pacman.py`,
`board_graph.py`, the primitive set or the fitness in
`train_board_graph.py` must be made in `pacman_core.cpp` too, then rebuilt
and checked with `python -m FastCore.check`, which fails (exit 1) if any
game's outcome differs.
