#!/usr/bin/env bash
# FastCore copy of run_board_graph_repair_comparison.sh, made by FastCore/sync_scripts.py: the same run
# with its training games played in C++ (FastCore/README.md). Do not edit;
# edit the original and rerun sync_scripts.py. Build first with
# FastCore/build.sh. Threads: BOARD_GRAPH_THREADS (default: all CPUs).
# Tree repair off (baseline) against repair on, 3 seeds each.
# See run_board_graph_conditions.sh for the settings and environment variables.
set -euo pipefail
export BOARD_GRAPH_CONDITIONS=${BOARD_GRAPH_CONDITIONS:-"baseline repair"}
exec "$(dirname "$0")/run_board_graph_conditions.sh"
