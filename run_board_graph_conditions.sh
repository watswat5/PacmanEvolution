#!/usr/bin/env bash
# Shared engine for comparison experiments: runs every listed condition for
# every seed, one after another, and prints a comparison table at the end.
# Use the wrappers run_board_graph_repair_comparison.sh,
# run_board_graph_ablation.sh and run_board_graph_endgame_experiment.sh, or
# set BOARD_GRAPH_CONDITIONS yourself.
#
# Each condition is the baseline below with ONE setting changed, so a
# difference from the baseline can be attributed to that setting. The same
# seed gives every condition the same training and tournament boards.
#
#   BOARD_GRAPH_CONDITIONS="baseline repair"    conditions to run (see the case below)
#   BOARD_GRAPH_SEEDS="101 102 103"             seeds per condition
#   BOARD_GRAPH_GENERATIONS=100                 generations per run
#   BOARD_GRAPH_POPULATION=225                  baseline population (15x15 grid)
#   BOARD_GRAPH_EXPERIMENT_DIRECTORY=runs/experiment_ablation
#   BOARD_GRAPH_FULL_BOARDS=5 BOARD_GRAPH_ENDGAME_BOARDS=3
#                                               training boards of each kind in
#                                               the "endgame" condition
#   BOARD_GRAPH_ENDGAME_STYLES=sweep,islands    how its endgame boards are built
#   BOARD_GRAPH_MAX_DEPTH=16                    baseline depth limit
#   BOARD_GRAPH_SUMMARY_FLAGS=""                extra flags for the final table
#
# Results go to <directory>/<condition>/seed_<seed>/. Runs are resumable:
# rerunning the same command skips finished runs and continues an interrupted
# one, and the two wrappers share the directory, so the baseline is only run
# once. Seeds are the outer loop: after the first pass every condition has one
# finished seed. Ctrl-C stops the current run cleanly and skips the rest.
set -euo pipefail

cd "$(dirname "$0")"

conditions=${BOARD_GRAPH_CONDITIONS:-"baseline"}
seeds=${BOARD_GRAPH_SEEDS:-"101 102 103"}
run_directory=${BOARD_GRAPH_EXPERIMENT_DIRECTORY:-runs/experiment_ablation}
run_directory=${run_directory%/}

# Extra trainer flags for a condition; they override the baseline flags.
condition_flags() {
  case "$1" in
    baseline)      echo "" ;;
    repair)        echo "--repair" ;;
    mutation10)    echo "--crossover-probability 0.90 --mutation-probability 0.10" ;;
    step100)       echo "--stagnation-penalty 100" ;;
    small_inserts) echo "--mutation-min-depth 0" ;;
    global)        echo "--evolution-method global" ;;
    grid8x8)       echo "--population-size 64" ;;
    no_junction)   echo "--exclude-features junction_ghost_margin,safe_escape_options" ;;
    # Endgame experiment (run_board_graph_endgame_experiment.sh).
    scaled_stagnation) echo "--stagnation-scaling inverse" ;;
    late_pellets)      echo "--late-pellet-share 0.5" ;;
    endgame)
      echo "--boards-per-generation $(( ${BOARD_GRAPH_FULL_BOARDS:-5} + ${BOARD_GRAPH_ENDGAME_BOARDS:-3} ))" \
        "--endgame-boards ${BOARD_GRAPH_ENDGAME_BOARDS:-3}" \
        "--endgame-styles ${BOARD_GRAPH_ENDGAME_STYLES:-sweep,islands}" ;;
    *) return 1 ;;
  esac
}

for condition in $conditions; do
  if ! condition_flags "$condition" >/dev/null; then
    printf 'Unknown condition: %s\n' "$condition" >&2
    exit 1
  fi
done

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p "$run_directory"
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_CONDITIONS="$conditions" \
    BOARD_GRAPH_SEEDS="$seeds" BOARD_GRAPH_EXPERIMENT_DIRECTORY="$run_directory" \
    bash "$0" </dev/null >>"$log_path" 2>&1 &
  training_pid=$!
  printf 'Experiment directory: %s\nConditions: %s\nSeeds: %s\nFollowing progress in %s\nCtrl-C stops the current run cleanly; rerun the same command to continue.\nPID: %s\n' \
    "$run_directory" "$conditions" "$seeds" "$log_path" "$training_pid"
  # Training runs in its own session, so the terminal's Ctrl-C only reaches
  # this launcher. The first one asks the loop to stop (USR1); a second stops
  # following. The log follower runs in the background, where the shell makes
  # it ignore Ctrl-C, so no log lines are lost; it exits when training does.
  tail --pid="$training_pid" -n 0 -f "$log_path" &
  tail_pid=$!
  interrupted=0
  forward_interrupt() {
    if (( interrupted )); then
      printf '\nStopped following. PID %s may still be writing its final artifacts.\n' "$training_pid"
      kill "$tail_pid" 2>/dev/null || true
      exit 130
    fi
    interrupted=1
    printf '\nInterrupting training (PID %s); waiting for it to finish writing. Ctrl-C again to stop following.\n' "$training_pid"
    kill -USR1 "$training_pid" 2>/dev/null || true
  }
  trap forward_interrupt INT
  while kill -0 "$tail_pid" 2>/dev/null; do
    wait "$tail_pid" || true
  done
  exit 0
fi

source GeneticAlgo/.venv/bin/activate

# Detached loop. This shell ignores Ctrl-C (it was started in the background),
# so the launcher signals USR1 instead. The trainer runs as a background child
# so that the trap can fire while it is running.
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
  for condition in $conditions; do
    if (( stop_requested )); then
      break 2
    fi
    printf '\n=== %s, seed %s (%s) ===\n' "$condition" "$seed" "$(date '+%Y-%m-%d %H:%M')"
    # shellcheck disable=SC2046  # the condition's flags are meant to split
    python -u -m GeneticAlgo.train_board_graph \
      --resume \
      --no-video \
      --evolution-method cellular \
      --cellular-tournament-size 3 \
      --population-size "${BOARD_GRAPH_POPULATION:-225}" \
      --generations "${BOARD_GRAPH_GENERATIONS:-100}" \
      --boards-per-generation 8 \
      --boards-replaced-per-generation 2 \
      --timestep-limit 1000 \
      --stagnation-penalty 20 \
      --stagnation-grace-steps 0 \
      --stagnation-termination-steps 50 \
      --min-initial-depth 2 \
      --max-initial-depth 8 \
      --max-evolved-depth "${BOARD_GRAPH_MAX_DEPTH:-16}" \
      --crossover-probability 0.70 \
      --mutation-probability 0.30 \
      --top-fraction 0.50 \
      --top-probability 0.80 \
      --size-penalty 10 \
      --parsimony 1.0 \
      --seed "$seed" \
      --output-directory "$run_directory/$condition/seed_$seed" \
      $(condition_flags "$condition") &
    trainer_pid=$!
    while kill -0 "$trainer_pid" 2>/dev/null; do
      wait "$trainer_pid" || true
    done
    trainer_pid=
  done
done

printf '\n=== Comparison (%s) ===\n' "$(date '+%Y-%m-%d %H:%M')"
# shellcheck disable=SC2086  # the extra flags are meant to split
python summarize_experiments.py "$run_directory" --fresh-boards 30 ${BOARD_GRAPH_SUMMARY_FLAGS:-} || true
