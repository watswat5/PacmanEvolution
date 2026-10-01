#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source GeneticAlgo/.venv/bin/activate

output="${1:-long_best_500_steps.mp4}"
fps="${2:-30}"

python -m GeneticAlgo.watch_individual \
  checkpoints_long/run_0010/best_overall.pkl \
  --video "$output" \
  --max-steps 500 \
  --fps "$fps"
