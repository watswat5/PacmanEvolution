#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source GeneticAlgo/.venv/bin/activate

run_number=1
while true; do
  run_name=$(printf "run_%04d" "$run_number")
  run_directory="checkpoints_long/$run_name"
  if [[ ! -e "$run_directory" ]]; then
    break
  fi
  run_number=$((run_number + 1))
done
mkdir -p "$run_directory"

arguments=(
  PacmanGame/training_states/
  --latent-dim 32
  --population-size 1000
  --generations 100
  --timestep-limit 100
  --full-game-timestep-limit 1000
  --sampled-state-count 6
  --full-game-weight 0.5
  --stagnation-penalty 100.0
  --stagnation-grace-steps 0
  --stagnation-termination-steps 50
  --min-initial-depth 3
  --max-initial-depth 6
  --max-evolved-depth 10
  --crossover-probability 0.90
  --mutation-probability 0.10
  --top-fraction 0.32
  --top-probability 0.80
  --parsimony 0.1
  --worker-count 8
  --seed 42
  --checkpoint-directory "$run_directory"
  --log-path "$run_directory/history.csv"
)

{
  printf "run_name=%s\n" "$run_name"
  printf "created_at=%s\n" "$(date --iso-8601=seconds)"
  printf "command=python -m GeneticAlgo.train_pacman"
  printf " %q" "${arguments[@]}"
  printf "\n"
} > "$run_directory/run_settings.txt"

printf "Starting %s; outputs will be saved in %s\n" "$run_name" "$run_directory"
python -m GeneticAlgo.train_pacman "${arguments[@]}"
