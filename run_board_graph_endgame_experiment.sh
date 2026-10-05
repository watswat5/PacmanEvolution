#!/usr/bin/env bash
# Endgame experiment: does any one of three changes raise the clear rate?
# The baseline is the overnight configuration (cellular 20x20 grid) with the
# depth limit raised from 16 to 20. Each other condition is that baseline with
# ONE setting changed:
#   baseline           nothing else changed
#   scaled_stagnation  stagnation cutoff of 50 steps divided by the fraction of
#                      pellets still uneaten (50 on a full board, 200 at 25% left)
#   late_pellets       the last 20% of the pellets is worth as much as the first 80%
#   endgame            5 full games + 3 endgame boards per generation; an endgame
#                      starts with 75% of the pellets eaten and scores 0 there
#                      (alternately the leftovers of a nearest-pellet sweep and
#                      separate patches near corners)
#
# Seeds 42 43 44 for 200 generations, the seeds of the overnight run, so every
# run here uses that run's training and tournament boards. Its first 200
# generations (depth limit 16) are in runs/experiment_endgame/baseline_depth16
# as a reference, copied with truncate_run.py; comparing that with "baseline"
# shows what the depth limit alone does.
#
# Seeds are the outer loop: seed 42 finishes all four conditions before seed 43
# starts. Rerun the same command to continue after a stop or a crash.
#
#   BOARD_GRAPH_FULL_BOARDS=5 BOARD_GRAPH_ENDGAME_BOARDS=3   boards of each kind
#                                                            in the endgame condition
# See run_board_graph_conditions.sh for the other settings.
set -euo pipefail
export BOARD_GRAPH_CONDITIONS=${BOARD_GRAPH_CONDITIONS:-"baseline scaled_stagnation late_pellets endgame"}
export BOARD_GRAPH_SEEDS=${BOARD_GRAPH_SEEDS:-"42 43 44"}
export BOARD_GRAPH_GENERATIONS=${BOARD_GRAPH_GENERATIONS:-200}
export BOARD_GRAPH_POPULATION=${BOARD_GRAPH_POPULATION:-400}
export BOARD_GRAPH_MAX_DEPTH=${BOARD_GRAPH_MAX_DEPTH:-20}
export BOARD_GRAPH_EXPERIMENT_DIRECTORY=${BOARD_GRAPH_EXPERIMENT_DIRECTORY:-runs/experiment_endgame}
export BOARD_GRAPH_FULL_BOARDS=${BOARD_GRAPH_FULL_BOARDS:-5}
export BOARD_GRAPH_ENDGAME_BOARDS=${BOARD_GRAPH_ENDGAME_BOARDS:-3}
# Judge every condition by the same, more forgiving stagnation rule.
export BOARD_GRAPH_SUMMARY_FLAGS=${BOARD_GRAPH_SUMMARY_FLAGS:-"--stagnation-scaling inverse"}
exec "$(dirname "$0")/run_board_graph_conditions.sh"
