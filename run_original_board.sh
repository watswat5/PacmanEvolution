#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Original-maze minimiser: find the smallest tree that clears the original
# Pac-Man maze (GeneticAlgo/train_original_board.py). A clear beats any
# non-clear; among clearing trees, fewer nodes wins. When a seed finishes, or on
# Ctrl-C, its smallest clearing tree is pruned further and saved as
# smallest_clear_pruned.pkl with a video.
#
#   BOARD_GRAPH_SEEDS="42"                seeds to run, in order
#   BOARD_GRAPH_GENERATIONS=500           generations per seed
#   BOARD_GRAPH_POPULATION=400            population (20x20 grid)
#   ORIGINAL_START_FROM="path ..."        saved trees or run folders; trees in
#                                         them that already clear the maze join
#                                         the starting population
#   ORIGINAL_TARGET_NODES=                stop a seed once a clear is this small
#   BOARD_GRAPH_RESUME=runs/<directory>   continue an earlier run of this script
#
# To prune one saved tree without training:
#   python -m GeneticAlgo.train_original_board --prune-only path/to/tree.pkl
seeds=${BOARD_GRAPH_SEEDS:-"42"}

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p runs
  if [[ -n "${BOARD_GRAPH_RESUME:-}" ]]; then
    run_directory=${BOARD_GRAPH_RESUME%/}
    if [[ ! -d "$run_directory" ]]; then
      printf 'BOARD_GRAPH_RESUME: %s is not a directory\n' "$run_directory" >&2
      exit 1
    fi
  else
    run_directory=$(mktemp -d "runs/original_board_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  fi
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$0" </dev/null >>"$log_path" 2>&1 &
  training_pid=$!
  printf 'Run artifacts: %s (one folder per seed: %s)\nFollowing progress in %s\nCtrl-C stops training early and still prunes and saves the current seed'"'"'s smallest clearing tree.\nResume later with: BOARD_GRAPH_RESUME=%s %s\nPID: %s\n' \
    "$run_directory" "$seeds" "$log_path" "$run_directory" "$0" "$training_pid"
  # Training runs in its own session, so the terminal's Ctrl-C only reaches
  # this launcher. The first one asks the seed loop to stop (USR1), which
  # interrupts the running trainer so it still prunes and saves its smallest tree; a
  # second Ctrl-C stops following.
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
    printf '\nInterrupting training (PID %s); waiting for the final pruning and video. Ctrl-C again to stop following.\n' "$training_pid"
    kill -USR1 "$training_pid" 2>/dev/null || true
  }
  trap forward_interrupt INT
  while kill -0 "$tail_pid" 2>/dev/null; do
    wait "$tail_pid" || true
  done
  exit 0
fi

# BOARD_GRAPH_VENV=GeneticAlgo/.venv-pypy trains under PyPy (see check_pypy.sh).
source "${BOARD_GRAPH_VENV:-GeneticAlgo/.venv}/bin/activate"

# Detached seed loop. This shell ignores Ctrl-C (it was started in the
# background), so the launcher signals USR1 instead. The trainer runs as a
# background child so that the trap can fire while it is running.
stop_requested=0
trainer_pid=
request_stop() {
  stop_requested=1
  if [[ -n "$trainer_pid" ]]; then
    kill -INT "$trainer_pid" 2>/dev/null || true
  fi
}
trap request_stop USR1 TERM

for seed in $seeds; do
  if (( stop_requested )); then
    break
  fi
  printf '\n=== Seed %s ===\n' "$seed"
  # shellcheck disable=SC2086  # the start-from paths are meant to split
  python -u -m GeneticAlgo.train_original_board \
    --resume \
    --population-size "${BOARD_GRAPH_POPULATION:-400}" \
    --generations "${BOARD_GRAPH_GENERATIONS:-500}" \
    --max-evolved-depth "${BOARD_GRAPH_MAX_DEPTH:-20}" \
    ${ORIGINAL_TARGET_NODES:+--target-nodes "$ORIGINAL_TARGET_NODES"} \
    ${ORIGINAL_START_FROM:+--start-from $ORIGINAL_START_FROM} \
    --seed "$seed" \
    --output-directory "$BOARD_GRAPH_OUTPUT_DIRECTORY/seed_$seed" &
  trainer_pid=$!
  while kill -0 "$trainer_pid" 2>/dev/null; do
    wait "$trainer_pid" || true
  done
  trainer_pid=
done

printf '\n=== Smallest clearing tree per seed ===\n'
for seed in $seeds; do
  summary_path="$BOARD_GRAPH_OUTPUT_DIRECTORY/seed_$seed/summary.txt"
  if [[ -f "$summary_path" ]]; then
    printf 'seed %s: %s\n' "$seed" "$(grep -m1 -e '^After pruning' -e '^No tree' "$summary_path" || head -n 1 "$summary_path")"
  else
    printf 'seed %s: not finished\n' "$seed"
  fi
done
