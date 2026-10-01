#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${BOARD_GRAPH_DETACHED:-}" != 1 ]]; then
  mkdir -p runs
  run_directory=$(mktemp -d "runs/board_graph_$(date +%Y%m%d_%H%M%S)_XXXXXX")
  nohup setsid env BOARD_GRAPH_DETACHED=1 BOARD_GRAPH_OUTPUT_DIRECTORY="$run_directory" \
    bash "$0" </dev/null >"$run_directory/training.log" 2>&1 &
  printf 'Run artifacts: %s\nLog: %s/training.log\nPID: %s\n' \
    "$run_directory" "$run_directory" "$!"
  exit 0
fi

source GeneticAlgo/.venv/bin/activate
exec python -u -m GeneticAlgo.train_board_graph \
  --evolution-method global \
  --cellular-tournament-size 3 \
  --population-size 1000 \
  --generations 1000 \
  --boards-per-generation "${BOARD_GRAPH_BOARDS_PER_GENERATION:-3}" \
  --timestep-limit 800 \
  --stagnation-penalty 100.0 \
  --stagnation-grace-steps 0 \
  --stagnation-termination-steps 50 \
  --min-initial-depth 2 \
  --max-initial-depth 8\
  --max-evolved-depth 16 \
  --crossover-probability 0.90 \
  --mutation-probability 0.10 \
  --top-fraction 0.50 \
  --top-probability 0.80 \
  --parsimony 1.0 \
  --seed 42 \
  --output-directory "$BOARD_GRAPH_OUTPUT_DIRECTORY"
