#!/usr/bin/env bash
# Island run with the evasion features and less noisy selection.
#
# run_board_graph_islands.sh with the overnight script's settings (800 trees
# per island on a 20x40 grid, depth 24, 90% crossover, top fraction 0.30, all
# three evasion terminals, pellets_left and the
# two endgame cues) and:
#   - 16 training boards per generation, 4 replaced each generation (the
#     overnight run uses 5 and 2), so a tree's fitness rests on 16 games;
#   - migration every 100 generations;
#   - 1% immigration: 8 of each island's 800 trees arrive per round, 2 from
#     each of the 4 other islands, picked uniformly at random and replacing
#     the island's worst trees.
# Five islands (seeds 42-46), 2000 generations each, trained at the same time
# (BOARD_GRAPH_PARALLEL_ISLANDS=1; each gets an equal share of the CPUs).
#
# Every variable of run_board_graph_islands.sh still applies, e.g.
# BOARD_GRAPH_GENERATIONS=3000 or BOARD_GRAPH_RESUME=runs/<directory>.
set -euo pipefail
export BOARD_GRAPH_SEEDS=${BOARD_GRAPH_SEEDS:-"42 43 44 45 46"}
export BOARD_GRAPH_GENERATIONS=${BOARD_GRAPH_GENERATIONS:-2000}
export BOARD_GRAPH_MIGRATION_INTERVAL=${BOARD_GRAPH_MIGRATION_INTERVAL:-100}
export BOARD_GRAPH_GRAPH_INTERVAL=${BOARD_GRAPH_GRAPH_INTERVAL:-500}
export BOARD_GRAPH_IMMIGRATION=${BOARD_GRAPH_IMMIGRATION:-0.01}
export BOARD_GRAPH_POPULATION=${BOARD_GRAPH_POPULATION:-800}
export BOARD_GRAPH_BOARDS_PER_GENERATION=${BOARD_GRAPH_BOARDS_PER_GENERATION:-16}
export BOARD_GRAPH_BOARDS_REPLACED=${BOARD_GRAPH_BOARDS_REPLACED:-4}
export BOARD_GRAPH_MAX_DEPTH=${BOARD_GRAPH_MAX_DEPTH:-24}
export BOARD_GRAPH_CROSSOVER=${BOARD_GRAPH_CROSSOVER:-0.90}
export BOARD_GRAPH_MUTATION=${BOARD_GRAPH_MUTATION:-0.10}
export BOARD_GRAPH_TOP_FRACTION=${BOARD_GRAPH_TOP_FRACTION:-0.30}
export BOARD_GRAPH_SAVE_INTERVAL=${BOARD_GRAPH_SAVE_INTERVAL:-100}
export BOARD_GRAPH_ADD_FEATURES=${BOARD_GRAPH_ADD_FEATURES-all,pellets_left,safe_pellet_distance,ghost_path_conflict}
export BOARD_GRAPH_PARALLEL_ISLANDS=${BOARD_GRAPH_PARALLEL_ISLANDS:-1}
exec "$(dirname "$0")/run_board_graph_islands.sh"
