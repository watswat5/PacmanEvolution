#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Overnight run: several independent seeds, one after another. Each seed is a
# fresh population, which explores more than one very long run because a run
# commits to one family of trees early.
#
# Evolution is cellular (a 20x20 toroidal grid by default): a child replaces
# its parent only if it scores higher on the same boards, so good trees are
# never lost. BOARD_GRAPH_POPULATION changes the grid size.
#
#   BOARD_GRAPH_SEEDS="42 43 44 45 46"   seeds to run, in order
#   BOARD_GRAPH_GENERATIONS=400          generations per seed
#   BOARD_GRAPH_RESUME=runs/<directory>  continue an earlier overnight run
#
# Every seed saves a checkpoint after each generation, so after a crash or a
# reboot, rerun with BOARD_GRAPH_RESUME set: finished seeds are skipped and the
# interrupted one continues where it stopped.
seeds=${BOARD_GRAPH_SEEDS:-"42 43 44 45 46"}

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p runs
  if [[ -n "${BOARD_GRAPH_RESUME:-}" ]]; then
    run_directory=${BOARD_GRAPH_RESUME%/}
    if [[ ! -d "$run_directory" ]]; then
      printf 'BOARD_GRAPH_RESUME: %s is not a directory\n' "$run_directory" >&2
      exit 1
    fi
  else
    run_directory=$(mktemp -d "runs/board_graph_overnight_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  fi
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$0" </dev/null >>"$log_path" 2>&1 &
  training_pid=$!
  printf 'Run artifacts: %s (one folder per seed: %s)\nFollowing progress in %s\nCtrl-C stops training early and still writes the current seed'"'"'s plots and video.\nResume later with: BOARD_GRAPH_RESUME=%s %s\nPID: %s\n' \
    "$run_directory" "$seeds" "$log_path" "$run_directory" "$0" "$training_pid"
  # Training runs in its own session, so the terminal's Ctrl-C only reaches
  # this launcher. The first one asks the seed loop to stop (USR1), which
  # interrupts the running trainer so it still writes its plots and video; a
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
    printf '\nInterrupting training (PID %s); waiting for the final plots and video. Ctrl-C again to stop following.\n' "$training_pid"
    kill -USR1 "$training_pid" 2>/dev/null || true
  }
  trap forward_interrupt INT
  while kill -0 "$tail_pid" 2>/dev/null; do
    wait "$tail_pid" || true
  done
  exit 0
fi

source GeneticAlgo/.venv/bin/activate

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
  python -u -m GeneticAlgo.train_board_graph \
    --resume \
    --evolution-method cellular \
    --cellular-tournament-size 3 \
    --population-size "${BOARD_GRAPH_POPULATION:-400}" \
    --generations "${BOARD_GRAPH_GENERATIONS:-400}" \
    --boards-per-generation "${BOARD_GRAPH_BOARDS_PER_GENERATION:-8}" \
    --boards-replaced-per-generation "${BOARD_GRAPH_BOARDS_REPLACED:-2}" \
    --timestep-limit 1000 \
    --stagnation-penalty "${BOARD_GRAPH_STAGNATION_PENALTY:-20}" \
    --stagnation-grace-steps 0 \
    --stagnation-termination-steps 50 \
    --min-initial-depth 2 \
    --max-initial-depth 8 \
    --max-evolved-depth 16 \
    --crossover-probability 0.70 \
    --mutation-probability 0.30 \
    --top-fraction 0.50 \
    --top-probability 0.80 \
    --size-penalty "${BOARD_GRAPH_SIZE_PENALTY:-10}" \
    --parsimony 1.0 \
    --seed "$seed" \
    --output-directory "$BOARD_GRAPH_OUTPUT_DIRECTORY/seed_$seed" &
  trainer_pid=$!
  while kill -0 "$trainer_pid" 2>/dev/null; do
    wait "$trainer_pid" || true
  done
  trainer_pid=
done

printf '\n=== Best 10-board fitness per seed ===\n'
for seed in $seeds; do
  summary_path="$BOARD_GRAPH_OUTPUT_DIRECTORY/seed_$seed/summary.txt"
  if [[ -f "$summary_path" ]]; then
    printf 'seed %s: %s\n' "$seed" "$(head -n 1 "$summary_path")"
  else
    printf 'seed %s: not finished\n' "$seed"
  fi
done
