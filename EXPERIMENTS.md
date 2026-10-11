# Experiment log

History of the Pac-Man GP experiments through 9 Oct 2026, compiled from the run
folders, CSV histories, run scripts, git history and `AGENTS.md`.

**Fitness numbers are not comparable across phases.** The fitness formula and
the game rules both changed. Pellet percentages are the closest thing to a
common measure. Dates are local time (run-folder names).

## Fitness functions by phase

Each entry says how sure the formula is: **from code** (read from git or the
working copy), **inferred** (the code from that date is not in git; deduced from
run scripts, CSV columns and value ranges), or **unknown**.

Common terms:

- *ratio*: pellets eaten in the episode / pellets on the board at its start.
- *step cost*: a charge for every step that scores nothing, after a grace
  period (always 0 steps in the run scripts). An episode is cut off after 50
  such steps in a row ("stagnation").
- *score*: the game's own score. Since at least 1 Oct it is 1,000 per pellet,
  1,000 per power pellet, 100 per ghost.

### Phase 1: autoencoder line, `train_pacman.py` (24-27 Sep)

| Runs | Formula | Confidence |
| --- | --- | --- |
| `gp_test`, `gp_overnight` (24-25 Sep) | Score gained in a 250-step rollout, with a parsimony term of 0.01 per node. No stagnation flags in `GeneticAlgo/run.sh`. Best values near 500 imply a smaller per-pellet score than today's 1,000. | inferred; exact formula unknown |
| `gp_long` (25 Sep) | Best value 1,810; same small score scale. `run_long.sh` was edited after this run, so its flags may not match. | unknown |
| `gp_short` (27 Sep) | Per episode: `score gained - 10,000 if died or cut off before the step limit - 100 per non-scoring step - parsimony * total nodes`. Final fitness: `0.8 * full game from the initial board + 0.2 * mean over 3 sampled mid-game states`. | from code (the committed `train_pacman.py` and `run_short.sh`); assumes the code did not change between 27 Sep and the first commit |

Design intent in `GeneticAlgo/strategy.md`: start from `score_end - score_start`
and add a death penalty and a size penalty later.

### Phase 2: board-graph line, first runs (28-29 Sep)

The code from these dates is not in git.

| Runs | Formula | Confidence |
| --- | --- | --- |
| 28 Sep 11:36 to 14:05 (12 runs) | Raw game score. Best values of 120,000 to 297,000 and no outcome columns in `history.csv`. | inferred |
| 28 Sep 17:28 onward, and 29 Sep | `100,000 * ratio - 10,000 if died or ended early - 100 per non-scoring step`. Values at or below about 83,000 and cleared/died/timed_out columns appear. From 29 Sep 12:23 the fitness is the average over 3 boards. | inferred (matches the first commit) |

### Phase 3: 17 features, cellular, tournament (1 Oct, day)

| Runs | Formula | Confidence |
| --- | --- | --- |
| Commit `5055109` (13:48) | `100,000 * ratio - 10,000 if died or ended before the step limit without winning - step cost (100)`. A stagnation cutoff therefore paid the death penalty. | from code |
| Commit `13f5844` (16:32); run `164446` | Same, plus `50` per non-scoring move onto a tile visited in the last 12 steps (revisit penalty). | from code |
| Uncommitted edit, about 16:45-16:58; runs `165819`, `large_183442`, `214554` | Death penalty changed to **actual deaths only**: `100,000 * ratio - 10,000 if died - step cost - revisit cost`. A stagnation cutoff or timeout no longer paid the 10,000. The second fitness value `-parsimony * nodes` is a tie-break only. | from code (working copy at the start of the 1 Oct evening session); which runs it applies to is inferred from the first appearance of the 10-board column |

### Phase 4: fitness and noise changes (1 Oct, night); run `235623`

```
100,000 * ratio
+ 30,000 if the board is cleared                     (--clear-bonus 0.30)
- if died or cut off for stagnation:
    10,000 + 0.25 * 100,000 * (1 - ratio)            (--death-penalty, --death-pellet-penalty)
- 100 per non-scoring step                           (--stagnation-penalty)
- 50 per non-scoring revisit                         (--revisit-penalty)
```

Averaged over 8 boards, 2 of them replaced each generation. A plain timeout
pays neither death term. Maximum 130,000. Confidence: from code.

### Phase 5: arcade rules (2 Oct, early); short runs `003926`, `004410`, `005424`

Phase 4 formula plus a size cost in the main fitness:

```
- size_penalty * nodes^2 / 400                       (--size-penalty 10)
```

So 50 nodes cost 62, 100 cost 250, 200 cost 1,000, 400 cost 4,000. The size cost
was briefly linear (`10 * nodes`) before becoming quadratic; which of the three
short runs, if any, used the linear form is not known. The game itself changed
in this phase (arcade ghost speeds and scatter), so the same formula measures a
different game. Confidence: from code, except the linear/quadratic timing.

### Phase 6: overnight runs (2 Oct); `overnight_*`

Phase 5 formula with the non-scoring step cost cut from 100 to **20**
(`--stagnation-penalty 20`, overnight script only). Everything else unchanged:
revisit cost 50, death term `10,000 + 25,000 * (1 - ratio)`, clear bonus 30,000,
quadratic size cost. Confidence: from code. This is the current formula; it is
also in the "Fitness" section of `AGENTS.md`.

### Phase 8: longer runs, islands, playoff (3-4 Oct)

Phase 6 formula, unchanged. Confidence: from code.

### Phase 9: endgame experiment (prepared 4 Oct, not yet run)

Phase 6 formula with one of these changed per condition. Confidence: from code.
Every run also has its depth limit raised from 16 to 20, which does not change
the formula.

- `scaled_stagnation`: the stagnation cutoff is `50 / (pellets left / pellets on
  the whole board)` steps without a pellet: 50 on a full board, 100 at half,
  200 at 25% left, 500 at 10% left. Step cost (20) and revisit cost (50) still
  apply to every such step. The reasoning: the cutoff was added to limit compute
  and stop infinite loops, before ghosts had a scatter mode that breaks loops.
- `late_pellets`: *ratio* is replaced, in the pellet term and in the death term,
  by a two-line curve in which the first 80% of the pellets earn 0.5 and the
  last 20% earn 0.5. So `0.4 -> 0.25`, `0.8 -> 0.5`, `0.9 -> 0.75`, `1.0 -> 1.0`.
  A pellet in the last fifth is worth about 4 times one in the first four
  fifths. A clear is still 100,000 + 30,000.
- `endgame`: no formula change, but 3 of the 8 training boards start with 75%
  of the pellets already eaten. *ratio* was always "pellets eaten in the episode
  / pellets on the board at its start", so an endgame scores 0 at its start and
  100,000 + 30,000 when cleared, the same as a full game; the remaining 25% of
  a board counts as much as a whole board.

Tournament and validation games use the run's own rules, so 10-board fitness
is not comparable between these conditions. Compare on fresh boards.

### Phase 10: evasion experiment (prepared 9 Oct, not yet run)

Fitness formula unchanged from Phase 9's baseline (depth limit 20). What
changes is what the tree can see: one or all of three new terminals
(`danger_ghost_approaching`, `pincer`, `safe_territory`; definitions in
AGENTS.md, "Shared-tree observation contract"). Training and tournament fitness
are therefore comparable between all conditions. Confidence: from code.

### What changed, at a glance

| Term | Phase 2 (late) | Phase 3 (final) | Phase 4 | Phase 5 | Phase 6 |
| --- | --- | --- | --- | --- | --- |
| Pellets | 100,000 * ratio | same | same | same | same |
| Clear bonus | none | none | 30,000 | 30,000 | 30,000 |
| Death | 10,000 | 10,000 | 10,000 + 25,000 * (1 - ratio) | same | same |
| Stagnation cutoff | as death | no penalty | as death | as death | as death |
| Non-scoring step | 100 | 100 | 100 | 100 | 20 |
| Revisit | none | 50 | 50 | 50 | 50 |
| Tree size | none | tie-break only | tie-break only | 10 * nodes^2 / 400 | same |
| Boards averaged | 1, then 3 | 3 | 8 (2 rotate) | 8 | 8 |
| Game rules | old | old | old | arcade | arcade |

"Old" game rules: ghosts at Pac-Man's speed, always chasing, half speed while
Pac-Man is powered up.

## Runs by phase

### Phase 1: autoencoder line (24-27 Sep)

Board -> pretrained autoencoder latent vector -> four GP trees (up/down/left/right).
Autoencoders were trained at latent sizes 4, 8, 16 and 32.

| Run | Date | Setup | Result (its own scale) |
| --- | --- | --- | --- |
| `gp_test.csv`, `checkpoints_test` | 24-25 Sep | latent 16, pop 200, 10 generations | best 499 |
| `gp_overnight.csv`, `checkpoints_overnight` | 25 Sep | 9 generations | best 504 |
| `gp_long.csv`, `checkpoints_long` | 25 Sep | 100 generations (latent 32, pop 1000 per the current script) | best 1,810; average tree 22 nodes |
| `gp_short.csv`, `checkpoints_short` | 27 Sep | latent 16, pop 500, 100 generations | best 114,093; video `long_best_500_steps.mp4` |

These runs were not re-analysed in October; only the CSV numbers are known.

### Phase 2: board-graph line begins (28-29 Sep)

One shared scoring tree run on every legal move, 11 hand-built graph features.

- 28 Sep midday: about 12 short runs, about 100 trees, one board. No clears.
- 28 Sep evening: 800 games per generation; validation board added (best validation pellets 74-96%).
- 29 Sep: 3 boards per generation, global selection, population 800-1000. Best fitness 50,000-62,000, validation 86-97%. One run bloated to 300 nodes; a 429-generation run ended at 11 nodes with at most 2 clears per generation.

### Phase 3: 17 features, cellular, tournament (1 Oct, day)

| Run | Setup | Best 10-board | Notes |
| --- | --- | --- | --- |
| `20261001_164446` | global, 100 gens | n/a | collapsed to 4 nodes |
| `20261001_165819` | global, 100 gens | 56,768 (74% pellets) | |
| `large_20261001_183442` | global, pop 4000, 25 gens | 44,278 (58%) | |
| `20261001_214554` | cellular 25x32, 67 gens | 58,937 (75%) | 30 fresh boards: 56% pellets, 29 deaths, 0 clears |

### Phase 4: fitness and noise changes (1 Oct, night)

Also added: `junction_ghost_margin`, `safe_escape_options` (19 features),
1,000-step limit, champion chosen on the tournament boards.

- `20261001_235623` (cellular 15x20, 41 gens): best 10-board 49,674; trees bloated to about 360 nodes; 30 fresh boards: 64% pellets, 25 deaths.

### Phase 5: arcade rules (2 Oct, early)

Game rewritten to arcade level-1 rules; per-ghost scared flags in `Board`.

- Three short runs (cellular 8x8, 50 gens): best 10-board 35,748 / 41,837 / 56,575. The 41,837 run plateaued at generation 7 on a tree that uses no ghost feature; 30 deaths in 30 fresh boards, 61% pellets.

### Phase 6: overnight runs (2 Oct)

Also changed: mutation 10% -> 30% with larger inserts, checkpoints and resume, multi-seed script.

| Run | Setup | Result |
| --- | --- | --- |
| `overnight_011037` seed 42 | global, pop 400, single-node mutation inserts | stopped at gen 35; trees shrank to about 3-10 nodes; best 10-board 42,887 |
| `overnight_011921` seed 42 | cellular 20x20 | stopped at gen 9 |
| `overnight_012238` seed 42 | cellular 20x20, 200 gens, mutation inserts depth 1-4 | best 10-board 80,976 (89%); fresh boards 87% pellets, 0 clears |
| `overnight_012238` seed 43 | same, inserts up to the depth room | 79,883 (90%); fresh boards 75%, 2 clears |
| `overnight_012238` seed 44 | same | 73,533 (91%); fresh boards 83%, 0 clears |
| `overnight_012238` seed 45 | same | 87,852; not yet tested on fresh boards |
| `overnight_012238` seed 46 | same | 91,463; not yet tested on fresh boards |

All five seeds finished 200 generations. Mean best 10-board fitness 82,741 (range 73,533 to 91,463; maximum possible 130,000).

"Fresh boards" means the same 30 random boards (seeds 9000-9029) at 1,000 steps.
The Phase 3 and 4 champions were tested under the old game rules.

### Phase 7: one-setting-at-a-time ablation (2-3 Oct); `runs/experiment_ablation`

Each condition is the baseline with one setting changed. Baseline: the Phase 6
configuration scaled down (cellular 15x15 = 225 trees, 100 generations, 8 boards
with 2 rotating, step cost 20, 30% mutation, repair off). The same seed gives
every condition the same training and tournament boards. Fitness formula: Phase
6, except `step100`, which charges 100 per non-scoring step.

**Status: complete.** 8 conditions x 3 seeds (101, 102, 103), all 100 generations.

| Condition | Change | Best 10-board, per seed | Mean | vs baseline | Fresh boards: pellets, per seed | Mean | Fresh clears (of 90) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mutation10` | 10% mutation (90% crossover) | 93,590 / 72,113 / 62,215 | 75,973 | +39 | 79% / 77% / 67% | 74.1% | 0 |
| `baseline` | none | 63,935 / 79,078 / 84,786 | 75,933 | 0 | 64% / 81% / 80% | 75.2% | 5 |
| `no_junction` | without the two junction features | 69,042 / 79,327 / 73,280 | 73,883 | -2,050 | 80% / 73% / 63% | 72.0% | 1 |
| `repair` | tree repair on | 61,081 / 77,243 / 72,167 | 70,164 | -5,769 | 72% / 82% / 64% | 72.4% | 1 |
| `small_inserts` | mutation may insert one node | 71,155 / 64,387 / 66,798 | 67,447 | -8,487 | 75% / 55% / 71% | 67.1% | 1 |
| `grid8x8` | 64 trees | 65,006 / 75,497 / 59,216 | 66,573 | -9,360 | 64% / 74% / 60% | 66.2% | 1 |
| `step100` | step cost 100 | 60,812 / 60,926 / 57,827 | 59,855 | not comparable | 71% / 69% / 73% | 70.9% | 3 |
| `global` | global over-selection | 45,677 / 46,029 / 40,584 | 44,097 | -31,836 | 51% / 53% / 51% | 52.0% | 0 |

Columns: "Best 10-board" is the best generation champion's fitness on the 10
fixed tournament boards. "Fresh boards" is that run's `best_overall.pkl` on 30
random boards (seeds 9000-9029) that no run trained or was selected on; the
per-run values are cached in `fresh_boards.csv`.

Other per-seed measures (seeds 101 / 102 / 103):

| Condition | Training clear rate, last 20 gens | Avg nodes, final generation | Minutes per run |
| --- | --- | --- | --- |
| `baseline` | 1.9% / 0.9% / 4.7% | 69 / 106 / 61 | 42 / 141 / 41 |
| `mutation10` | 10.7% / 0.4% / 2.9% | 85 / 103 / 64 | 41 / 43 / 40 |
| `no_junction` | 0.0% / 4.8% / 0.0% | 44 / 80 / 100 | 37 / 41 / 45 |
| `repair` | 0.0% / 2.1% / 3.7% | 52 / 65 / 55 | 37 / 39 / 37 |
| `small_inserts` | 1.0% / 0.0% / 1.5% | 34 / 74 / 49 | 35 / 42 / 37 |
| `grid8x8` | 0.1% / 0.3% / 0.1% | 71 / 56 / 61 | 13 / 11 / 14 |
| `step100` | 2.8% / 5.3% / 0.0% | 35 / 64 / 37 | 41 / 42 / 34 |
| `global` | 0.0% / 0.0% / 0.0% | 60 / 6 / 42 | 15 / 12 / 13 |

(The 141 minutes for baseline seed 102 includes the hours it sat stopped
between the two sessions.)

Findings from three seeds:

1. **Cellular beats global. This is the one clear effect.** `global` is last
   on every measure in every seed: 44,097 against 75,933 on the tournament
   boards and 52% against 75% of pellets on fresh boards, and its three seeds
   do not overlap any other condition's. The global method keeps no tree
   unchanged between generations; the cellular grid keeps each cell's best.
   Caveat: cellular scores parents and children, so it uses about twice the
   game evaluations per generation.
2. **Mutation rate showed no effect.** `mutation10` and the baseline have the
   same mean (75,973 and 75,933) and overlapping seeds. The 30,000 lead that
   `mutation10` had after seed 101 was seed luck: it scored 93,590 / 72,113 /
   62,215 where the baseline scored 63,935 / 79,078 / 84,786. On fresh boards
   the baseline is level or slightly ahead (75.2% against 74.1%).
3. **The junction features showed no effect.** `no_junction` is within 2,050
   of the baseline on the tournament boards and 3 points below on fresh boards,
   inside the seed spread.
4. **Nothing beat the baseline.** `repair`, `small_inserts` and `grid8x8` are
   5,800 to 9,400 lower on the tournament boards and 3 to 9 points lower on
   fresh boards. The baseline's own seeds span 21,000 and 16 points, so none of
   these differences is established with three seeds. `small_inserts` and
   `grid8x8` are the most consistently lower.
5. **`step100` is not worse on play.** Its fitness is on a different scale. On
   fresh boards it ate 70.9% against 75.2%, with 3 clears against 5, and its
   seeds are the most consistent of any condition (69% to 73%).
6. **No condition clears reliably.** 12 clears in 720 fresh-board games; the
   baseline has the most (5 of 90).
7. **Seed-to-seed variation is the dominant effect.** Within a condition, seeds
   differ by 10,000 to 31,000 on the tournament boards. With three seeds, only
   an effect the size of cellular-versus-global can be detected.

What this means for the Phase 6 story: of the changes made before the
overnight run, this ablation confirms the cellular method and finds no
measurable effect from the mutation rate, mutation insert size, or junction
features at this scale. The step cost (100 against 20) and the arcade rules
remain the untested or inconclusive candidates for what made the difference.

Artifacts in `runs/experiment_ablation/`: `summary.csv`, `fresh_boards.csv`,
`comparison_all_seeds/` (14 mean-of-seeds line charts and 8 `summary_*.png`
dot plots showing every seed), `comparison_seed_101/`, `_102/`, `_103/`
(per-seed line charts), and `generation_grid.mp4` / `.png` in each
`<condition>/seed_101/` folder.

Not tested: combinations of settings, mutation rates other than 10% and 30%,
and the arcade-rules change (the old game rules are no longer in the code).

### Phase 8: longer runs, islands, playoff (3-4 Oct)

**Separate seeds continued** (`overnight_012238`, same settings). Seed 42 ran to
400 generations: best 10-board 82,976, reached by generation 265 and flat after;
average training fitness stayed near 71,500 from generation 200. Seed 43: last
new best 81,813 at generation 241. Seeds 43-46 were being extended to 300
generations when this was written.

**Island model** (`board_graph_islands_20261003_161020`): the same five seeds
and settings; every 20 generations each island sends one uniformly random tree
to every other island, where it replaces the worst tree. All islands share 10
tournament boards. 300 generations.

| Island | Best 10-board | Set at generation |
| --- | --- | --- |
| 42 | 83,828 | 164 |
| 43 | 91,324 | 273 |
| 44 | 86,669 | 298 |
| 45 | 86,437 | 294 |
| 46 | 83,357 | 298 |

- Against the separate seeds at generations 100 and 200: no measurable
  difference (fresh-board pellets 70.0% against 70.6% at 100, 81.2% against
  81.6% at 200). Islands are more uniform across seeds.
- Between 200 and 300 the islands kept improving slowly (mean champion
  tournament fitness 70,567 -> 72,300 -> 75,468 per 50 generations; training
  clear rate up to 8-10%), while separate seed 42 was flat. Only one or two
  separate seeds had run that long, so this is weak evidence.

**Playoff** (`runs/playoff.csv`): each run's 10 best distinct champions by
tournament score, 100 candidates, all on the same 100 unseen boards.

| | Separate seeds | Islands |
| --- | --- | --- |
| Mean playoff fitness | 63,850 | 66,413 |
| Candidates in the top 50 | 18 | 32 |
| Best candidate per seed, range | 59,527 - 75,460 | 68,298 - 73,394 |
| Mean pellets | 80.2% | 82.2% |
| Clears per 100 boards | 5.8 | 5.9 |
| Died / stagnated per 100 | 70 / 24 | 77 / 17 |

- Winner: separate seed 46, generation 158, 109 nodes. On 100 further fresh
  boards: **84.4% of pellets, 6 cleared, 60 died, 34 stagnated** (fitness
  68,686). This is the unbiased figure; on the selection boards it scored 88.9%
  and 10 clears.
- Tournament scores overstate: the separate candidates averaged 16,600 lower on
  the 100 playoff boards than on their 10 tournament boards, and every separate
  seed's saved best-overall tree ranked 52nd or worse.
- The island advantage is confounded with run length (300 generations for every
  island; 200 for separate seeds 44-46 at the time).
- The main result: no setting tried so far moves the clear rate. Every candidate
  eats 80-88% of the pellets and clears about 6 boards in 100.

### Phase 9: endgame experiment (prepared 4 Oct, not yet run); `runs/experiment_endgame`

Question: does any one of three changes raise the clear rate? Two suspected
causes of the cap: (1) with a 6% clear rate, a tree clears zero or one of its 8
training boards, so selection cannot distinguish a better clearer from a lucky
one, and fitness is in effect "pellets eaten before dying"; (2) the 50-step
stagnation cutoff scores waiting for an opening late in the game like a death.

Design: the Phase 6 configuration (cellular 20x20, 8 boards with 2 rotating,
step cost 20, 30% mutation) with the depth limit raised from 16 to 20, for 200
generations, seeds 42, 43, 44. Depth 20 is the new default for this phase, not a
condition. Each condition changes one more setting. Each seed runs all four
runs before the next seed starts.

| Condition | Change |
| --- | --- |
| `baseline` | none (depth 20) |
| `scaled_stagnation` | stagnation cutoff 50 / fraction of pellets left |
| `late_pellets` | last 20% of pellets worth as much as the first 80% |
| `endgame` | 5 full games + 3 endgame boards (75% pre-eaten) per generation |
| `baseline_depth16` | reference only: the first 200 generations of the Phase 6 run, same seeds and boards, depth 16 (copied, not retrained) |

Endgame boards are made from the same layouts by a deterministic, ghost-free
walker that always eats the nearest pellet; Pac-Man starts where it stopped.
Two styles alternate. *Sweep*: the walker eats until 75% are gone, leaving whole
corridors in one or two connected regions and about 0.5 energizers. *Islands*:
two to four connected patches near maze corners are set aside first and the
walker eats everything else, leaving separate patches and about 2.9 energizers.
All four ghosts are released at once. Approximations: the scatter/chase clock
starts at 0 and ghosts start at home.

How it will be judged: each run's champion(s) on fresh full-game boards, all
under the scaled stagnation rule, by pellets, clears and deaths
(`summarize_experiments.py --fresh-boards 30 --stagnation-scaling inverse`,
and the playoff with the same flag). Training and tournament fitness are not
comparable across these conditions.

Limits known in advance:

- Three seeds can only show a large effect (seed spread was 10,000-31,000 in
  Phase 7).
- `endgame` keeps 8 boards in total, so it also has 5 full games per generation
  where the others have 8.
- Under scaled stagnation a tree can loiter to the 1,000-step limit and avoid
  the death term; it still pays the step and revisit costs.
- `baseline` against `baseline_depth16` gives the effect of the depth limit
  alone. The depth-16 reference was trained before the code changes; the
  defaults were verified to give byte-identical results, but it was not
  literally rerun.
- Estimated 34 hours or more for the 12 runs; untested at full size.

Results: none yet.

### Phase 10: evasion experiment (prepared 9 Oct, not yet run); `runs/experiment_evasion`

Why: the 50,000-generation overnight run (`board_graph_overnight_20261009_001834_Y6qOm1`,
FastCore, 800 trees, depth 24) levelled off within the first few thousand
generations. Over the whole run, 60-65% of its training games ended in death,
about 22% in a clear and about 15% in stagnation (`plot_cdst.py`); its
generation champions died on about half of their validation boards. Dying is
the main failure, so this phase gives the trees cues a human player uses to
evade:

- `danger_ghost_approaching`: is the nearest dangerous ghost coming towards the
  candidate tile or moving away (ghosts cannot reverse)?
- `pincer`: does the move head at one ghost while another closes from behind?
- `safe_territory`: how much of the maze can Pac-Man reach before any ghost
  after this move? It generalises `junction_ghost_margin`, which looks one
  junction ahead.

Design: baseline (the original 19 terminals) plus one condition per feature
and one with all three; nothing else changes. Cellular 20x20, depth 20, 1000
generations, seeds 42 43 44, seeds as the outer loop
(`FastCore/run_board_graph_evasion_experiment.sh`).

How it will be judged: each run's champion on 30 fresh boards, by deaths first,
then clears and pellets (`summarize_experiments.py --fresh-boards 30`), plus
`champion_curves.py` over generations.

Limits known in advance:

- Three seeds can only show a large effect.
- More terminals make the search space larger; a feature can help a human and
  still slow evolution down at 1000 generations.
- The features use path distances and ignore ghost speeds, so
  `safe_territory` and `pincer` err towards caution.
- Speed: after the 9 Oct fix, a tree using `safe_territory` plays about 15%
  slower; the other two features cost nothing measurable.

**First result (one seed, not the five-condition experiment).** The overnight
script with all three features on (`board_graph_overnight_20261009_154225_r1Vtux`, seed 42) against the 50k
baseline run `board_graph_overnight_20261009_001834_Y6qOm1`: same seed, boards
and settings (800 trees, 20x40 grid, depth 24), only the terminals differ.
Both runs plateau around generation 1000.

| | Baseline | Evasion |
| --- | --- | --- |
| Population training games, generations 1000-2783: cleared / died / stagnated | 15.7 / 66.6 / 17.6% | 26.9 / 59.4 / 13.7% |
| Average training fitness, same generations | 78,000 | 85,900 |
| Best 10-board so far (by generation 2783) | 111,243 | 116,068 |

Generation champions on 100 fresh boards (seeds 9000-9099, C++ engine):

| Generation | Baseline: pellets, cleared, died | Evasion: pellets, cleared, died |
| --- | --- | --- |
| 999 | 84.8%, 9, 77 | 88.0%, 25, 63 |
| 1999 | 89.5%, 9, 73 | 89.3%, 29, 57 |
| 2699 | 88.7%, 8, 77 | 90.0%, 23, 58 |

- The features roughly triple the clear rate (8-9 to 23-29 per 100 fresh
  boards) and cut deaths by 15-20 per 100; pellets barely change.
- Every evasion champion uses `pincer` and `safe_territory`; none uses
  `danger_ghost_approaching`.
- The plateau is not removed: the evasion run levels off at the same point,
  just higher.
- One seed only; the five-condition experiment (three seeds, each feature
  alone) is still to run.

## Open points

- Reliable clearing is not reached: the playoff winner clears 6 of 100 fresh boards, and no run or setting so far has moved that rate (Phase 8). Phase 9 tests three candidate changes. Deaths are the main failure (60-65% of training games over a 50,000-generation run). Phase 10's evasion features about tripled fresh-board clears in one seed (8-9 to 23-29 per 100), but the run still plateaus near generation 1000.
- Separate seeds plateau at about generation 200-265; islands were still improving slowly at 300. Not yet compared at equal generations.
- The ablation is underpowered for small effects: three seeds per condition against a seed spread of 10,000 to 31,000. Settling the mid-table conditions would take roughly 8 to 10 seeds each.
- The tournament score overstates generalisation. Fresh-board results should be the headline comparison in any write-up.
- Run scripts are unchanged (30% mutation, cellular); the ablation gives no reason to change them.
- Git has two commits, both from the afternoon of 1 Oct. Everything from Phase 4 onward is uncommitted, and the code for Phases 1-2 was never committed, which is why some early formulas above are only inferred.
