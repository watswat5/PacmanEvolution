#!/usr/bin/env bash
# Play Pac-Man side by side against a saved tree, one step per key press.
#
#   ./versus.sh path/to/individual.pkl [--seed N] [--scale 1-4]
#
# With no file it uses runs/playoff_winner.pkl. Each game is a new random
# board, the same for both sides. Arrows or WASD move, Space keeps going,
# N starts a new board, Esc quits. Needs a display, so run it at the PC.
set -euo pipefail
individual=${1:-runs/playoff_winner.pkl}
if [[ $# -gt 0 ]]; then shift; fi
individual=$(realpath -- "$individual")
cd "$(dirname "$(realpath -- "$0")")"
source GeneticAlgo/.venv/bin/activate
exec python -m GeneticAlgo.versus "$individual" "$@"
