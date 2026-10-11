#!/usr/bin/env bash
# FastCore copy of run_board_graph_steady.sh, made by FastCore/sync_scripts.py: the same run
# with its training games played in C++ (FastCore/README.md). Do not edit;
# edit the original and rerun sync_scripts.py. Build first with
# FastCore/build.sh. Threads: BOARD_GRAPH_THREADS (default: all CPUs).
set -euo pipefail

script="$(realpath "$0")"
cd "$(dirname "$0")/.."

# Steady-state run. One generation is one step:
#   1. one new random board joins a rolling window of the last
#      BOARD_GRAPH_STEADY_WINDOW boards and the oldest drops out. Every tree in
#      the population plays the new board only; its fitness is the average
#      over the boards in the window, using the results it already has;
#   2. BOARD_GRAPH_STEADY_CHILDREN children are bred, each from two parents
#      chosen by tournament, and play every board in the window, so they are
#      scored on exactly the same games as everyone else;
#   3. for each child, three trees are drawn at random from the worst 20% of
#      the population, and the child replaces the worst of the three if its
#      average is at least as high (a tie goes to the child).
# There is no champion tournament and no validation game. The generation's
# best is the highest window average, and best_overall.pkl is the highest
# window average any tree has had.
#
#   BOARD_GRAPH_SEEDS="42"               seeds to run, in order
#   BOARD_GRAPH_GENERATIONS=20000        steps per seed
#   BOARD_GRAPH_STEADY_WINDOW=10         boards in the rolling window
#   BOARD_GRAPH_STEADY_CHILDREN=1        children per step
#   BOARD_GRAPH_POPULATION=400           population size
#   BOARD_GRAPH_STEADY_REPLACE_ON_TIE=1  a tie goes to the child (0: the
#                                        child must be strictly better)
#   BOARD_GRAPH_STEADY_REJECT_DUPLICATES=0
#                                        1 drops a child that plays the window
#                                        exactly like an existing tree
#   BOARD_GRAPH_STEADY_DUEL_BOARDS=1     above the window size, the child and
#                                        its three opponents also play extra
#                                        boards before the decision
#   BOARD_GRAPH_FULL_MIN_DEPTH=4         smallest depth of the "full" half of
#                                        the starting trees
#   BOARD_GRAPH_TERMINAL_PROBABILITY=0.4 chance a node in the "grow" half is a
#                                        terminal (60/40 functions/terminals)
#   BOARD_GRAPH_ADD_FEATURES=all,pellets_left,safe_pellet_distance,ghost_path_conflict
#                                        added terminals ("" for the original 19)
#   BOARD_GRAPH_SAVE_INTERVAL=10         keep the step's best tree and newest
#                                        board file for every N-th step only
#   BOARD_GRAPH_RESUME=runs/<directory>  continue an earlier steady-state run
#
# A step costs one game per tree in the population plus a window of games per
# child. The first step plays the whole window for everyone.
#
# Every seed saves a checkpoint after each step, so after a crash or a reboot,
# rerun with BOARD_GRAPH_RESUME set. The window and duel settings must match
# the run being resumed (the duplicate and tie settings may change); runs from before the rolling window used
# BOARD_GRAPH_STEADY_WINDOW=1.
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
    run_directory=$(mktemp -d "runs/board_graph_steady_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  fi
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$script" </dev/null >>"$log_path" 2>&1 &
  training_pid=$!
  printf 'Run artifacts: %s (one folder per seed: %s)\nFollowing progress in %s\nCtrl-C stops training early and still writes the current seed'"'"'s plots and video.\nResume later with: BOARD_GRAPH_RESUME=%s %s\nPID: %s\n' \
    "$run_directory" "$seeds" "$log_path" "$run_directory" "$script" "$training_pid"
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
  python -u -m FastCore.train_board_graph \
    --resume \
    --evolution-method steady \
    --steady-children "${BOARD_GRAPH_STEADY_CHILDREN:-1}" \
    --steady-tournament-size 3 \
    --steady-worst-fraction 0.20 \
    --steady-replacement-pool 3 \
    $([[ "${BOARD_GRAPH_STEADY_REJECT_DUPLICATES:-0}" == 1 ]] && echo --steady-reject-duplicates) \
    $([[ "${BOARD_GRAPH_STEADY_REPLACE_ON_TIE:-1}" == 1 ]] && echo --steady-replace-on-tie) \
    --save-interval "${BOARD_GRAPH_SAVE_INTERVAL:-10}" \
    --steady-duel-boards "${BOARD_GRAPH_STEADY_DUEL_BOARDS:-1}" \
    --population-size "${BOARD_GRAPH_POPULATION:-400}" \
    --generations "${BOARD_GRAPH_GENERATIONS:-20000}" \
    --add-features "${BOARD_GRAPH_ADD_FEATURES-all,pellets_left,safe_pellet_distance,ghost_path_conflict,blinky_distance,elroy,safe_power_pellet_distance,onward_safe_exits}" \
    --boards-per-generation "${BOARD_GRAPH_STEADY_WINDOW:-10}" \
    --boards-replaced-per-generation 1 \
    --timestep-limit 1000 \
    --stagnation-penalty "${BOARD_GRAPH_STAGNATION_PENALTY:-20}" \
    --stagnation-grace-steps 0 \
    --stagnation-termination-steps 50 \
    --min-initial-depth 2 \
    --max-initial-depth 8 \
    --full-min-initial-depth "${BOARD_GRAPH_FULL_MIN_DEPTH:-4}" \
    --grow-terminal-probability "${BOARD_GRAPH_TERMINAL_PROBABILITY:-0.4}" \
    --max-evolved-depth "${BOARD_GRAPH_MAX_DEPTH:-20}" \
    --crossover-probability 0.70 \
    --mutation-probability 0.30 \
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
