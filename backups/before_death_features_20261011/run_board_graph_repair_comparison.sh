#!/usr/bin/env bash
# Tree repair off (baseline) against repair on, 3 seeds each.
# See run_board_graph_conditions.sh for the settings and environment variables.
set -euo pipefail
export BOARD_GRAPH_CONDITIONS=${BOARD_GRAPH_CONDITIONS:-"baseline repair"}
exec "$(dirname "$0")/run_board_graph_conditions.sh"
