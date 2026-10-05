#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p runs
  run_directory=$(mktemp -d "runs/board_graph_large_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$0" </dev/null >"$log_path" 2>&1 &
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

source GeneticAlgo/.venv/bin/activate
exec python -u -m GeneticAlgo.train_board_graph \
  --evolution-method global \
  --cellular-tournament-size 3 \
  --population-size 1500 \
  --generations 25 \
  --boards-per-generation "${BOARD_GRAPH_BOARDS_PER_GENERATION:-8}" \
  --boards-replaced-per-generation "${BOARD_GRAPH_BOARDS_REPLACED:-2}" \
  --timestep-limit 1000 \
  --stagnation-penalty 100.0 \
  --stagnation-grace-steps 0 \
  --stagnation-termination-steps 50 \
  --min-initial-depth 2 \
  --max-initial-depth 8 \
  --max-evolved-depth 16 \
  --crossover-probability 0.90 \
  --mutation-probability 0.10 \
  --top-fraction 0.50 \
  --top-probability 0.80 \
  --size-penalty "${BOARD_GRAPH_SIZE_PENALTY:-10}" \
  --parsimony 1.0 \
  --seed 42 \
  --output-directory "$BOARD_GRAPH_OUTPUT_DIRECTORY"
