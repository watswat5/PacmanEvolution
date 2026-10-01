# Pac-Man Genetic Programming Architecture

## 1. Project Goal

The goal is to evolve a Genetic Programming (GP) controller capable of playing Pac-Man.

The controller receives a compressed representation of the complete visible Pac-Man board and chooses one of four actions:

* Up
* Down
* Left
* Right

The GP agent has the same basic control authority as a human player: it can observe the game state and choose a direction, but it cannot directly modify the board, ghosts, pellets, or other game state.

---

# 2. High-Level Architecture

```text
Full Pac-Man Board
        ↓
Pretrained Autoencoder Encoder
        ↓
16-Dimensional Latent Vector
        ↓
Genetic Programming Controller
        ↓
┌────────┬────────┬────────┬────────┐
│   UP   │  DOWN  │  LEFT  │ RIGHT  │
│  Tree  │  Tree  │  Tree  │  Tree  │
└────────┴────────┴────────┴────────┘
        ↓
4 Action Scores
        ↓
Mask Illegal Moves
        ↓
Argmax
        ↓
Selected Direction
        ↓
Pac-Man Simulator
        ↓
Updated Full Board
        ↓
Re-encode
        ↓
Repeat
```

---

# 3. Board Representation

The Pac-Man simulator maintains the complete game state.

This includes information such as:

* Wall locations
* Pellet locations
* Power pellet locations
* Fruit
* Pac-Man position
* Individual ghost positions
* Ghost states
* Frightened state/timer
* Score
* Other necessary game timers or state

The GP controller does **not** directly receive this complete representation.

Instead:

```text
full board → encoder → 16 latent values
```

The 16-dimensional representation acts as the GP controller's observation of the board.

---

# 4. Autoencoder

## Purpose

The autoencoder reduces the dimensionality of the complete Pac-Man board.

During autoencoder training:

```text
Full Board
    ↓
Encoder
    ↓
16-D Latent Vector
    ↓
Decoder
    ↓
Reconstructed Board
```

The autoencoder is trained before GP evolution begins.

After training:

* Freeze the encoder weights.
* Keep the decoder for debugging and visualization.
* Use only the encoder during GP fitness evaluation.

The GP training loop therefore uses:

```text
board → encoder → latent vector
```

and does not normally use the decoder.

## Decoder Use

The decoder can be used to inspect what information survives in the 16-dimensional representation:

```text
Actual Board

vs.

Board → Encoder → Decoder → Reconstructed Board
```

This can help identify whether important information such as ghosts, pellets, or Pac-Man's position is being lost.

Actual gameplay rendering should use the simulator's real board state rather than the decoder output.

---

# 5. GP Individual Representation

Each individual contains four expression trees:

```text
Individual
├── UP tree
├── DOWN tree
├── LEFT tree
└── RIGHT tree
```

Each tree receives the same 16 latent values:

```text
z0, z1, z2, ... z15
```

Each tree outputs one real-valued score.

Example:

```text
UP    = z3 - z7
DOWN  = z2 * z11
LEFT  = max(z4, z8)
RIGHT = z1 / (abs(z9) + 1)
```

After evaluation:

```text
UP     = 0.42
DOWN   = -0.18
LEFT   = 1.37
RIGHT  = 0.91
```

The legal action with the highest score is selected.

---

# 6. Function and Terminal Sets

## Terminals

Initial terminals:

```text
z0
z1
...
z15
```

Optionally include ephemeral random constants.

Example:

```text
0.25
-0.8
1.4
```

## Functions

Start with a small function set:

```text
+
-
*
protected_divide
min
max
abs
```

More complex operators can be added later if necessary:

```text
sin
cos
conditional
comparisons
```

Keeping the initial function set small reduces the search space.

---

# 7. Action Selection

GP evaluates all four action trees:

```text
latent vector
      ↓
┌─────┬─────┬─────┬─────┐
│ UP  │DOWN │LEFT │RIGHT│
└─────┴─────┴─────┴─────┘
      ↓
four scores
```

The simulator determines which actions are legal.

Illegal actions are masked:

```text
UP      4.2
DOWN    ILLEGAL → -∞
LEFT    2.8
RIGHT   ILLEGAL → -∞
```

Then:

```text
action = argmax(legal action scores)
```

In this example:

```text
UP
```

is selected.

This prevents evolution from wasting generations learning that Pac-Man cannot move through walls.

---

# 8. Population Initialization

Use **ramped half-and-half initialization**.

The initial population contains trees generated using both:

* Full method
* Grow method

across several initial depths.

Example initial depth range:

```text
2–5
```

This creates structural diversity in the initial population.

Each individual receives four independently generated trees:

```text
Individual
├── random UP tree
├── random DOWN tree
├── random LEFT tree
└── random RIGHT tree
```

---

# 9. Fitness Evaluation

Each individual is evaluated through short Pac-Man rollouts rather than necessarily playing a complete game.

For each starting state:

```text
Starting Board
      ↓
Encoder
      ↓
16-D State
      ↓
GP Controller
      ↓
Action
      ↓
Simulator advances
      ↓
New Board
      ↓
Encoder
      ↓
Repeat for T steps
```

After `T` steps, calculate fitness.

A simple initial fitness function is:

```text
fitness = score_end - score_start
```

Possible later additions include:

```text
fitness =
    points gained
    - death penalty
    - tree-size penalty
```

Avoid heavily engineered reward functions initially. Pac-Man's actual game score should remain the primary objective.

---

# 10. Starting States

Do not evaluate each individual on a different random state.

Instead, every individual within a generation should receive the same set of starting states.

Example:

```text
Generation 20

Starting states:
A
B
C
D
E
```

Every individual plays:

```text
A for T steps
B for T steps
C for T steps
D for T steps
E for T steps
```

Fitness:

```text
fitness =
    average(score gained across all starting states)
```

This makes comparisons between individuals fair.

Starting states should preferably come from valid game trajectories rather than arbitrary random placement of objects.

A state database can be generated by:

* Random gameplay
* Simple heuristic agents
* Previous evolved agents
* Recorded full games

---

# 11. Selection

Use tournament selection.

Example:

```text
Tournament size = 3–5
```

For each parent:

1. Randomly select `k` individuals.
2. Compare their fitness.
3. Select the highest-fitness individual.

Smaller tournament sizes preserve more diversity.

Larger tournament sizes increase selection pressure.

---

# 12. Crossover

Each parent contains four trees.

```text
Parent 1
├── U1
├── D1
├── L1
└── R1

Parent 2
├── U2
├── D2
├── L2
└── R2
```

For a crossover event:

1. Randomly choose one action.
2. Select a random subtree from that action tree in Parent 1.
3. Select a random subtree from the matching action tree in Parent 2.
4. Swap the selected subtrees.

Example: LEFT selected.

```text
L1 ↔ L2
```

The children become:

```text
Child 1
├── U1
├── D1
├── L1'
└── R1

Child 2
├── U2
├── D2
├── L2'
└── R2
```

Only one matching action tree undergoes subtree crossover per event.

The other three trees remain unchanged.

This reduces the destructiveness of crossover.

---

# 13. Mutation

After crossover, each child has some probability of mutation.

Initial proposed mutation probability:

```text
~5–10%
```

A very low rate such as `2%` can be tested, but slightly higher mutation is likely useful for maintaining structural diversity.

Only one randomly selected action tree is mutated.

## Initial Mutation Type: Point Mutation

Example:

```text
+ → -
```

or:

```text
z3 → z11
```

or:

```text
max → min
```

Operators should normally be replaced with operators of compatible arity.

Later, subtree mutation can be introduced:

```text
random subtree
      ↓
replace with newly generated subtree
```

This allows evolution to introduce new structures that crossover alone cannot create.

---

# 14. Elitism

Preserve a small percentage of the best individuals unchanged.

Example:

```text
Elitism = 1–5%
```

This prevents the current best solution from being destroyed by crossover or mutation.

---

# 15. Tree Bloat Control

GP trees tend to grow larger over time even when the additional structure provides little benefit.

Use a combination of a hard size limit and parsimony pressure.

## Hard Limit

Example:

```text
Maximum depth = 10
```

or:

```text
Maximum nodes per tree = 100
```

Invalid offspring can be rejected or replaced by the parent.

## Parsimony Pressure

Example:

```text
adjustedFitness =
    gameFitness - λ(totalNodes)
```

where:

```text
totalNodes =
    nodes(UP)
  + nodes(DOWN)
  + nodes(LEFT)
  + nodes(RIGHT)
```

Start with a small value for `λ`.

Tree size should act as a secondary objective rather than overpowering actual gameplay ability.

---

# 16. Discrete Pac-Man Simulation

Use Pac-Man movement as the simulation clock:

```text
1 simulation step =
Pac-Man attempts/makes one tile movement
```

Pac-Man therefore moves approximately:

```text
1 tile / step
```

Ghost speeds are represented by how frequently they receive movement opportunities.

Example:

```text
Pac-Man:
move every step

Normal ghost:
move 4 out of every 5 steps

Frightened ghost:
move every 2 steps
```

An accumulator can represent non-integer ratios without fractional tile movement.

Example:

```text
normal ghost speed = 4/5

progress += 4

if progress >= 5:
    move ghost
    progress -= 5
```

---

# 17. Generation Loop

Complete evolutionary loop:

```text
TRAIN AUTOENCODER
        ↓
freeze encoder
        ↓
initialize GP population
using ramped half-and-half
        ↓
┌───────────────────────────────────┐
│                                   │
│ Select shared starting states     │
│             ↓                     │
│ Evaluate population               │
│             ↓                     │
│ Run T-step Pac-Man rollouts       │
│             ↓                     │
│ Calculate fitness                 │
│             ↓                     │
│ Preserve elites                   │
│             ↓                     │
│ Tournament selection              │
│             ↓                     │
│ Subtree crossover                 │
│             ↓                     │
│ Mutation                          │
│             ↓                     │
│ Build next population             │
│                                   │
└───────────────↺───────────────────┘
```

---

# 18. Fitness Evaluation Pseudocode

```python
for generation in range(NUM_GENERATIONS):

    starting_states = sample_starting_states()

    for individual in population:

        rollout_scores = []

        for start_state in starting_states:

            game = copy(start_state)
            start_score = game.score

            for step in range(ROLLOUT_LENGTH):

                board = game.get_board()

                latent = encoder(board)

                scores = [
                    individual.up(latent),
                    individual.down(latent),
                    individual.left(latent),
                    individual.right(latent)
                ]

                legal = game.get_legal_moves()

                mask_illegal(scores, legal)

                action = argmax(scores)

                game.step(action)

                if game.is_dead():
                    break

            rollout_scores.append(
                game.score - start_score
            )

        individual.fitness = mean(rollout_scores)
```

Parsimony can then be applied:

```python
individual.fitness -= (
    PARSIMONY *
    individual.total_node_count()
)
```

---

# 19. Reproduction Pseudocode

```python
next_population = preserve_elites(population)

while len(next_population) < POPULATION_SIZE:

    parent1 = tournament_select(population)
    parent2 = tournament_select(population)

    child1 = copy(parent1)
    child2 = copy(parent2)

    action_tree = random_choice(
        ["up", "down", "left", "right"]
    )

    subtree_crossover(
        child1[action_tree],
        child2[action_tree]
    )

    if random() < MUTATION_RATE:
        mutate_random_tree(child1)

    if random() < MUTATION_RATE:
        mutate_random_tree(child2)

    enforce_tree_limits(child1)
    enforce_tree_limits(child2)

    next_population.append(child1)
    next_population.append(child2)

population = next_population
```

---

# 20. Performance Strategy

The first implementation should prioritize correctness over GPU optimization.

Initial approach:

```text
CPU:
Pac-Man simulation
GP representation
selection
crossover
mutation

GPU:
batched encoder inference
```

GP trees can initially be evaluated directly on the CPU.

If GP evaluation becomes a bottleneck, trees can later be compiled into postfix/stack-based bytecode.

Example:

```text
Tree:

      +
     / \
   z3   *
       / \
      z7 z9
```

becomes:

```text
PUSH z3
PUSH z7
PUSH z9
MUL
ADD
```

This representation is easier to batch and potentially execute on the GPU.

Do not optimize this until profiling shows GP evaluation is actually a major bottleneck.

---

# 21. Validation

Short rollouts are used for evolutionary fitness, but the best individuals should periodically be evaluated on complete games.

Example:

```text
Every 10 generations:

take best 5 individuals
        ↓
run complete Pac-Man games
        ↓
record:
- score
- survival time
- pellets eaten
- ghosts eaten
- tree size
```

This checks whether improvement on short training rollouts transfers to actual gameplay.

---

# 22. Metrics to Record

For every generation, record:

```text
Best fitness
Average fitness
Median fitness

Best full-game score
Average tree size
Maximum tree size

Average rollout survival
Deaths per rollout

Generation runtime
Encoder runtime
GP evaluation runtime
Simulation runtime
```

These measurements will make it possible to identify convergence, overfitting, GP bloat, and computational bottlenecks.

---

# 23. Initial Hyperparameters

Reasonable starting values:

```text
Latent dimensions:       16

Population size:         500–2000

Tournament size:         3–5

Initial tree depth:      2–5

Maximum tree depth:      ~10

Crossover:
one matching action tree
per parent pair

Mutation probability:    5–10%

Elitism:                 1–5%

Starting states:         5–20

Rollout length:          50–200 steps

Parsimony pressure:      small initially
```

These values are starting points and should be treated as experimental parameters.

---

# 24. Experimental Comparisons

Once the baseline system works, useful experiments include:

## Autoencoder vs. Full Board

Compare:

```text
16-dimensional latent input
```

against:

```text
full uncompressed board input
```

Measure:

* Convergence speed
* Final gameplay score
* Runtime
* Tree size
* Number of generations required

## Mutation Rates

Compare:

```text
2%
5%
10%
20%
```

## Tournament Sizes

Compare:

```text
2
3
5
7
```

## Rollout Lengths

Compare:

```text
25
50
100
200
```

## Number of Starting States

Compare robustness as more states are used for fitness evaluation.

## Parsimony Pressure

Measure the tradeoff between:

```text
game performance
vs.
tree complexity
```

---

# 25. Final Intended System

```text
                  PAC-MAN
                     │
                     ↓
              Full Board State
                     │
                     ↓
             Frozen Autoencoder
                  Encoder
                     │
                     ↓
              16-Dimensional
               Latent State
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ↓             ↓             ↓
    UP Tree      DOWN Tree      LEFT Tree ...
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                Action Scores
                     │
                     ↓
              Legal Move Mask
                     │
                     ↓
                   Argmax
                     │
                     ↓
                Pac-Man Move
                     │
                     ↓
             Simulator Updates
                     │
                     ↓
              New Board State
                     │
                     └──────↺
```

The evolutionary system therefore optimizes **decision-making logic**, while the pretrained autoencoder handles dimensionality reduction and the Pac-Man simulator enforces the actual rules of the environment.

