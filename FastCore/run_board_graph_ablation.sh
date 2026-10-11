#!/usr/bin/env bash
# FastCore copy of run_board_graph_ablation.sh, made by FastCore/sync_scripts.py: the same run
# with its training games played in C++ (FastCore/README.md). Do not edit;
# edit the original and rerun sync_scripts.py. Build first with
# FastCore/build.sh. Threads: BOARD_GRAPH_THREADS (default: all CPUs).
# One-setting-at-a-time ablation: the baseline plus seven variants, 3 seeds each.
#   repair         tree repair on
#   mutation10     10% mutation / 90% crossover instead of 30% / 70%
#   step100        100 per non-scoring step instead of 20
#   small_inserts  mutation may insert a single node (minimum depth 0)
#   global         global over-selection instead of the cellular grid
#   grid8x8        64 trees instead of 225
#   no_junction    without junction_ghost_margin and safe_escape_options
# Shares its directory with run_board_graph_repair_comparison.sh, so runs that
# script already finished are skipped.
# See run_board_graph_conditions.sh for the settings and environment variables.
set -euo pipefail
export BOARD_GRAPH_CONDITIONS=${BOARD_GRAPH_CONDITIONS:-"baseline repair mutation10 step100 small_inserts global grid8x8 no_junction"}
exec "$(dirname "$0")/run_board_graph_conditions.sh"
