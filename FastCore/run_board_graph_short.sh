#!/usr/bin/env bash
# FastCore copy of run_board_graph_short.sh, made by FastCore/sync_scripts.py: the same run
# with its training games played in C++ (FastCore/README.md). Do not edit;
# edit the original and rerun sync_scripts.py. Build first with
# FastCore/build.sh. Threads: BOARD_GRAPH_THREADS (default: all CPUs).
set -euo pipefail

script="$(realpath "$0")"
cd "$(dirname "$0")/.."

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p runs
  run_directory=$(mktemp -d "runs/board_graph_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$script" </dev/null >"$log_path" 2>&1 &
  training_pid=$!
  printf 'Run artifacts: %s\nFollowing progress in %s\nCtrl-C stops training early and still writes the final plots and video.\nPID: %s\n' \
    "$run_directory" "$log_path" "$training_pid"
  # Training runs in its own session, so the terminal's Ctrl-C only reaches
  # this launcher. Pass the first one on as an interrupt, which makes the
  # trainer finish with its plots and video; a second Ctrl-C stops following.
  # The log follower runs in the background, where the shell makes it ignore
  # Ctrl-C, so no log lines are lost; it exits when training does.
  tail --pid="$training_pid" -n +1 -f "$log_path" &
  tail_pid=$!
  interrupted=0
  forward_interrupt() {
    if (( interrupted )); then
      printf '\nStopped following. PID %s may still be writing its final artifacts.\n' "$training_pid"
      kill "$tail_pid" 2>/dev/null || true
      exit 130
    fi
    interrupted=1
    printf '\nInterrupting training (PID %s); waiting for the final plots and video. Ctrl-C again to stop following.\n' "$training_pid"
    kill -INT "$training_pid" 2>/dev/null || true
  }
  trap forward_interrupt INT
  while kill -0 "$tail_pid" 2>/dev/null; do
    wait "$tail_pid" || true
  done
  exit 0
fi

# BOARD_GRAPH_ADD_FEATURES: terminals added to the original 19. Default
# "all,pellets_left,safe_pellet_distance,ghost_path_conflict" (every optional feature); "" for
# the original 19 only. A resumed run must use the value it was started with.
# BOARD_GRAPH_VENV=GeneticAlgo/.venv-pypy trains under PyPy (see check_pypy.sh).
source "${BOARD_GRAPH_VENV:-GeneticAlgo/.venv}/bin/activate"

exec python -u -m FastCore.train_board_graph \
  --evolution-method cellular \
  --population-size 64 \
  --generations 50 \
  --add-features "${BOARD_GRAPH_ADD_FEATURES-all,pellets_left,safe_pellet_distance,ghost_path_conflict,blinky_distance,elroy,safe_power_pellet_distance,onward_safe_exits}" \
  --boards-per-generation "${BOARD_GRAPH_BOARDS_PER_GENERATION:-8}" \
  --boards-replaced-per-generation "${BOARD_GRAPH_BOARDS_REPLACED:-2}" \
  --timestep-limit 1000 \
  --stagnation-penalty 20 \
  --stagnation-grace-steps 0 \
  --stagnation-termination-steps 50 \
  --min-initial-depth 4 \
  --max-initial-depth 8 \
  --max-evolved-depth 10 \
  --crossover-probability 0.75 \
  --mutation-probability 0.25 \
  --top-fraction 0.32 \
  --top-probability 0.80 \
  --size-penalty "${BOARD_GRAPH_SIZE_PENALTY:-10}" \
  --parsimony 1.0 \
  --seed 42 \
  --output-directory "$BOARD_GRAPH_OUTPUT_DIRECTORY"
