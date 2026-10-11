#!/usr/bin/env bash
# FastCore copy of run_board_graph_evasion_experiment.sh, made by FastCore/sync_scripts.py: the same run
# with its training games played in C++ (FastCore/README.md). Do not edit;
# edit the original and rerun sync_scripts.py. Build first with
# FastCore/build.sh. Threads: BOARD_GRAPH_THREADS (default: all CPUs).
# Evasion experiment: do the evasion features help Pac-Man stop dying?
# The baseline is the endgame experiment's baseline (cellular 20x20 grid,
# depth limit 20, the original 19 terminals). Each other condition adds
# terminals and changes nothing else:
#   baseline        the original 19 features
#   approaching     + danger_ghost_approaching: +1 if the dangerous ghost nearest
#                   the candidate tile can step closer to it (ghosts do not
#                   reverse), -1 if heading away, 0 with none
#   pincer          + pincer: 1 if the move heads at a dangerous ghost while
#                   another closes in from behind, both within 8 tiles
#   safe_territory  + safe_territory: share of the maze Pac-Man reaches strictly
#                   before every dangerous ghost after taking the move
#   evasion         + all three
#
# Seeds 42 43 44 for 1000 generations. Seeds are the outer loop: seed 42
# finishes all five conditions before seed 43 starts. Rerun the same command
# to continue after a stop or a crash. Compare with plot_experiment_comparison.py
# and plot_cdst.py (death rate is the number to watch).
#
# See run_board_graph_conditions.sh for the other settings.
set -euo pipefail
export BOARD_GRAPH_CONDITIONS=${BOARD_GRAPH_CONDITIONS:-"baseline approaching pincer safe_territory evasion"}
export BOARD_GRAPH_SEEDS=${BOARD_GRAPH_SEEDS:-"42 43 44"}
export BOARD_GRAPH_GENERATIONS=${BOARD_GRAPH_GENERATIONS:-1000}
export BOARD_GRAPH_POPULATION=${BOARD_GRAPH_POPULATION:-400}
export BOARD_GRAPH_MAX_DEPTH=${BOARD_GRAPH_MAX_DEPTH:-20}
export BOARD_GRAPH_EXPERIMENT_DIRECTORY=${BOARD_GRAPH_EXPERIMENT_DIRECTORY:-runs/experiment_evasion}
exec "$(dirname "$0")/run_board_graph_conditions.sh"
