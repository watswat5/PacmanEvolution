#!/usr/bin/env bash
# Island model: several populations (one per seed) evolve separately and
# exchange trees every few generations.
#
# Each island is an ordinary cellular run with the overnight settings. After
# every BOARD_GRAPH_MIGRATION_INTERVAL generations, each island sends
# BOARD_GRAPH_MIGRANTS randomly chosen trees to every other island, where they
# replace the worst trees (migrate_islands.py). All islands share the same 10
# tournament boards, so their champions are directly comparable.
#
#   BOARD_GRAPH_SEEDS="42 43 44 45 46"   one island per seed
#   BOARD_GRAPH_GENERATIONS=300          total generations per island
#   BOARD_GRAPH_GRAPH_INTERVAL=100       compile comparison charts this often
#   BOARD_GRAPH_MIGRATION_INTERVAL=20    generations between migrations
#   BOARD_GRAPH_MIGRANTS=1               trees sent to each other island
#   BOARD_GRAPH_POPULATION=400           trees per island (20x20 grid)
#   BOARD_GRAPH_MAX_DEPTH=16             depth cap for evolved trees
#   BOARD_GRAPH_RESUME=runs/<directory>  continue an earlier island run
#
# Every BOARD_GRAPH_GRAPH_INTERVAL generations (and at the end) the script
# writes one chart per metric, with a line per island and their mean, to
# <run>/comparison_gen_NNN/.
#
# The islands run one after another within each interval, so the total compute
# equals running the seeds independently. Everything is resumable: after a
# stop or crash, rerun with BOARD_GRAPH_RESUME set and it continues from the
# last completed generation without repeating a migration. Ctrl-C stops
# cleanly.
set -euo pipefail

cd "$(dirname "$0")"

seeds=${BOARD_GRAPH_SEEDS:-"42 43 44 45 46"}
generations=${BOARD_GRAPH_GENERATIONS:-300}
graph_interval=${BOARD_GRAPH_GRAPH_INTERVAL:-100}
interval=${BOARD_GRAPH_MIGRATION_INTERVAL:-20}
migrants=${BOARD_GRAPH_MIGRANTS:-1}

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p runs
  if [[ -n "${BOARD_GRAPH_RESUME:-}" ]]; then
    run_directory=${BOARD_GRAPH_RESUME%/}
    if [[ ! -d "$run_directory" ]]; then
      printf 'BOARD_GRAPH_RESUME: %s is not a directory\n' "$run_directory" >&2
      exit 1
    fi
  else
    run_directory=$(mktemp -d "runs/board_graph_islands_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  fi
  log_path="$run_directory/training.log"
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$0" </dev/null >>"$log_path" 2>&1 &
  training_pid=$!
  printf 'Run artifacts: %s (one folder per island: %s)\nMigration: %s tree(s) to each other island every %s generations, %s generations in total\nFollowing progress in %s\nCtrl-C stops training cleanly.\nResume later with: BOARD_GRAPH_RESUME=%s %s\nPID: %s\n' \
    "$run_directory" "$seeds" "$migrants" "$interval" "$generations" "$log_path" "$run_directory" "$0" "$training_pid"
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
run_directory=$BOARD_GRAPH_OUTPUT_DIRECTORY

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

target=0
while (( target < generations )); do
  target=$(( target + interval ))
  if (( target > generations )); then
    target=$generations
  fi
  # Only the last stretch renders the videos.
  video_flag=--no-video
  if (( target == generations )); then
    video_flag=
  fi
  for seed in $seeds; do
    if (( stop_requested )); then
      break 2
    fi
    printf '\n=== Island %s, to generation %s (%s) ===\n' "$seed" "$target" "$(date '+%Y-%m-%d %H:%M')"
    python -u -m GeneticAlgo.train_board_graph \
      --resume \
      $video_flag \
      --tournament-seed "${BOARD_GRAPH_TOURNAMENT_SEED:-424242}" \
      --evolution-method cellular \
      --cellular-tournament-size 3 \
      --population-size "${BOARD_GRAPH_POPULATION:-400}" \
      --generations "$target" \
      --boards-per-generation "${BOARD_GRAPH_BOARDS_PER_GENERATION:-8}" \
      --boards-replaced-per-generation "${BOARD_GRAPH_BOARDS_REPLACED:-2}" \
      --timestep-limit 1000 \
      --stagnation-penalty "${BOARD_GRAPH_STAGNATION_PENALTY:-20}" \
      --stagnation-grace-steps 0 \
      --stagnation-termination-steps 50 \
      --min-initial-depth 2 \
      --max-initial-depth 8 \
      --max-evolved-depth "${BOARD_GRAPH_MAX_DEPTH:-16}" \
      --crossover-probability 0.70 \
      --mutation-probability 0.30 \
      --top-fraction 0.50 \
      --top-probability 0.80 \
      --size-penalty "${BOARD_GRAPH_SIZE_PENALTY:-10}" \
      --parsimony 1.0 \
      --seed "$seed" \
      --output-directory "$run_directory/seed_$seed" &
    trainer_pid=$!
    while kill -0 "$trainer_pid" 2>/dev/null; do
      wait "$trainer_pid" || true
    done
    trainer_pid=
  done
  if (( stop_requested )); then
    break
  fi
  # Compile the all-islands charts at each checkpoint, kept in their own
  # folder so earlier snapshots are not overwritten.
  if (( target % graph_interval == 0 || target == generations )); then
    printf '\n=== Comparison charts at generation %s ===\n' "$target"
    python plot_experiment_comparison.py "$run_directory" --islands \
      --output "$run_directory/comparison_gen_$(printf '%03d' "$target")" || true
  fi
  if (( target < generations )); then
    printf '\n=== Migration after generation %s ===\n' "$target"
    if ! python migrate_islands.py "$run_directory" --generation "$target" --migrants "$migrants"; then
      printf 'Migration failed; stopping so the islands stay in step.\n'
      exit 1
    fi
  fi
done

printf '\n=== Best 10-board fitness per island (same boards for all) ===\n'
for seed in $seeds; do
  summary_path="$run_directory/seed_$seed/summary.txt"
  if [[ -f "$summary_path" ]]; then
    printf 'island %s: %s\n' "$seed" "$(head -n 1 "$summary_path")"
  else
    printf 'island %s: not finished\n' "$seed"
  fi
done
