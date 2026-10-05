#!/usr/bin/env bash
# Watch a saved tree play the original Pac-Man maze.
#
#   ./play_individual.sh path/to/individual.pkl [more.pkl ...]
#
# Drag and drop: drop a .pkl file onto "Play Pac-Man individual.desktop" in
# this folder, or type "./play_individual.sh " in a terminal and drag the file
# into the terminal window to paste its path.
#
# For each file it prints how the game ended, saves the game as
# <name>_original_maze.mp4 next to the .pkl, and opens the video. Works on any
# generation_NNNN_best.pkl, best_overall.pkl or playoff_winner.pkl from the
# board-graph trainer. The game runs until Pac-Man dies, clears the board, or
# reaches the step limit; there is no stagnation cutoff here.
#
#   PLAY_FPS=10          video frames per second
#   PLAY_SCALE=2         video size multiplier (1-4)
#   PLAY_MAX_STEPS=1000  step limit
#   PLAY_OPEN=1          set to 0 to save the video without opening it
set -uo pipefail

# Absolute paths first: the script changes folder below.
files=()
for file in "$@"; do
  files+=("$(realpath -- "$file")")
done
cd "$(dirname "$(realpath -- "$0")")"

finish() {
  # Keep the window open when started by drag and drop.
  if [[ -n "${PLAY_WAIT:-}" ]]; then
    read -r -p "Press Enter to close. " _
  fi
  exit "$1"
}

if (( ${#files[@]} == 0 )); then
  echo "Usage: $0 individual.pkl [more.pkl ...]" >&2
  finish 1
fi
# shellcheck disable=SC1091
source GeneticAlgo/.venv/bin/activate || finish 1

status=0
for file in "${files[@]}"; do
  if [[ ! -f "$file" || "$file" != *.pkl ]]; then
    echo "Not a .pkl file: $file" >&2
    status=1
    continue
  fi
  video="${file%.pkl}_original_maze.mp4"
  echo "== $file"
  if python -m GeneticAlgo.watch_board_graph "$file" \
      --board PacmanGame/test_board.txt \
      --video "$video" \
      --fps "${PLAY_FPS:-10}" \
      --scale "${PLAY_SCALE:-2}" \
      --max-steps "${PLAY_MAX_STEPS:-1000}"; then
    if [[ "${PLAY_OPEN:-1}" == 1 ]] && command -v xdg-open >/dev/null; then
      xdg-open "$video" >/dev/null 2>&1 &
    fi
  else
    echo "Could not play $file" >&2
    status=1
  fi
done
finish "$status"
