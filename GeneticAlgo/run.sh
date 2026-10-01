python -m GeneticAlgo.train_pacman \
  PacmanGame/training_states/state_003121.txt \
  --latent-dim 16 \
  --population-size 200 \
  --generations 10 \
  --timestep-limit 250 \
  --min-initial-depth 4 \
  --max-initial-depth 8 \
  --max-evolved-depth 16 \
  --crossover-probability 0.97 \
  --mutation-probability 0.03 \
  --top-fraction 0.32 \
  --top-probability 0.8 \
  --parsimony 0.01 \
  --worker-count 16 \
  --seed 42 \
  --checkpoint-directory checkpoints_test \
  --log-path gp_test.csv
