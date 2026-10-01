#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source GeneticAlgo/.venv/bin/activate

python -m GeneticAlgo.train_pacman \
  PacmanGame/training_states/ \
  --latent-dim 16 \
  --population-size 500 \
  --generations 100 \
  --timestep-limit 150 \
  --full-game-timestep-limit 500 \
  --sampled-state-count 3 \
  --full-game-weight 0.8 \
  --stagnation-penalty 100.0 \
  --stagnation-grace-steps 0 \
  --stagnation-termination-steps 50 \
  --min-initial-depth 3 \
  --max-initial-depth 6 \
  --max-evolved-depth 14 \
  --crossover-probability 0.90 \
  --mutation-probability 0.10 \
  --top-fraction 0.32 \
  --top-probability 0.80 \
  --parsimony 0.01 \
  --worker-count 8 \
  --seed 42 \
  --checkpoint-directory checkpoints_short \
  --log-path gp_short.csv
