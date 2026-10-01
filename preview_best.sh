#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source GeneticAlgo/.venv/bin/activate

individual="${1:-checkpoints_long/run_0010/best_overall.pkl}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "$individual" ]]; then
  echo "Best-individual file not found: $individual" >&2
  echo "Usage: $0 [individual.pkl] [playback options]" >&2
  exit 1
fi

python -m GeneticAlgo.watch_individual \
  "$individual" \
  --fps 10 \
  --max-steps 1000 \
  "$@"
