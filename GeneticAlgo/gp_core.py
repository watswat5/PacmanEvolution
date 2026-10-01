from __future__ import annotations

import copy
import math
import operator
import pickle
import random
from functools import lru_cache, partial
from pathlib import Path
from typing import Iterable, Sequence

from deap import base, creator, gp, tools

ACTIONS = ("up", "down", "left", "right")
ACTION_TREE_SWAP_PROBABILITY = 0.10


def _ensure_deap_types() -> None:
    if not hasattr(creator, "PacmanFitnessMax"):
        creator.create("PacmanFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "PacmanIndividual"):
        creator.create("PacmanIndividual", list, fitness=creator.PacmanFitnessMax)


_ensure_deap_types()
Individual = creator.PacmanIndividual


def protected_divide(a: float, b: float) -> float:
    return a if abs(b) < 1e-9 else a / b


def protected_log(a: float) -> float:
    return 0.0 if a <= 0 else math.log(a)


def protected_sqrt(a: float) -> float:
    return math.sqrt(abs(a))


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def safe_number(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-1e9, min(1e9, value))


def build_primitive_set(
    terminal_names: Sequence[str], constants: Iterable[float] = (-1, 0, 1)
) -> gp.PrimitiveSet:
    """Build the DEAP primitive set shared by the four action trees."""
    pset = gp.PrimitiveSet("PACMAN", len(terminal_names))
    pset.renameArguments(
        **{f"ARG{index}": name for index, name in enumerate(terminal_names)}
    )
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(protected_divide, 2, name="div")
    pset.addPrimitive(min, 2, name="min")
    pset.addPrimitive(max, 2, name="max")
    pset.addPrimitive(abs, 1, name="abs")
    pset.addPrimitive(operator.neg, 1, name="neg")
    pset.addPrimitive(protected_log, 1, name="log")
    pset.addPrimitive(math.sin, 1, name="sin")
    pset.addPrimitive(math.cos, 1, name="cos")
    pset.addPrimitive(math.tan, 1, name="tan")
    pset.addPrimitive(protected_sqrt, 1, name="sqrt")
    pset.addPrimitive(distance, 4, name="dist")
    for index, value in enumerate(constants):
        pset.addTerminal(float(value), name=f"constant_{index}")
    return pset


MOVE_DIVIDE_LIMIT = 1_000.0
MOVE_DIVIDE_EPSILON = 1e-3


def bounded_divide(a: float, b: float) -> float:
    """Division that stays monotonic near zero and never leaves +/-1000.

    The older ``protected_divide`` returns ``a`` when ``b`` is 0, so with
    normalized distances ``1 / pellet_distance`` would rank an adjacent pellet
    (distance 0) *below* one several tiles away. Flooring the denominator's
    magnitude keeps "closer means larger" intact.
    """
    if abs(b) < MOVE_DIVIDE_EPSILON:
        b = MOVE_DIVIDE_EPSILON if b >= 0 else -MOVE_DIVIDE_EPSILON
    return max(-MOVE_DIVIDE_LIMIT, min(MOVE_DIVIDE_LIMIT, a / b))


def if_lte(a: float, b: float, then: float, otherwise: float) -> float:
    """Return ``then`` when ``a <= b``, else ``otherwise`` (lets trees switch modes)."""
    return then if a <= b else otherwise


def build_move_primitive_set(
    terminal_names: Sequence[str],
    constants: Iterable[float] = (-1, 0, 0.5, 1, 2),
) -> gp.PrimitiveSet:
    """Primitive set for the shared move-scoring tree.

    Every terminal is a bounded per-move feature, so this set leaves out
    primitives that only made sense for raw coordinates (``dist``) or that turn
    bounded inputs into noise (``sin``/``cos``/``tan``/``log``). ``if_lte``
    adds explicit branching, for example "flee unless frightened".
    """
    pset = gp.PrimitiveSet("PACMAN_MOVE", len(terminal_names))
    pset.renameArguments(
        **{f"ARG{index}": name for index, name in enumerate(terminal_names)}
    )
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(bounded_divide, 2, name="div")
    pset.addPrimitive(min, 2, name="min")
    pset.addPrimitive(max, 2, name="max")
    pset.addPrimitive(abs, 1, name="abs")
    pset.addPrimitive(operator.neg, 1, name="neg")
    pset.addPrimitive(protected_sqrt, 1, name="sqrt")
    pset.addPrimitive(if_lte, 4, name="if_lte")
    for index, value in enumerate(constants):
        pset.addTerminal(float(value), name=f"constant_{index}")
    return pset


def build_toolbox(
    primitive_set: gp.PrimitiveSet,
    min_depth: int,
    max_depth: int,
    max_evolved_depth: int,
    top_fraction: float,
    top_probability: float,
) -> base.Toolbox:
    """Register DEAP initialization, selection, and variation operators."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be greater than 0 and at most 1")
    if not 0.0 <= top_probability <= 1.0:
        raise ValueError("top_probability must be between 0 and 1")
    toolbox = base.Toolbox()
    toolbox.register(
        "expression", gp.genHalfAndHalf, pset=primitive_set,
        min_=min_depth - 1, max_=max_depth - 1,
    )
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expression)
    toolbox.register("individual", tools.initRepeat, Individual, toolbox.tree, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)
    toolbox.register(
        "select",
        select_over,
        top_fraction=top_fraction,
        top_probability=top_probability,
    )

    def mate(first: Individual, second: Individual):
        action_index = random.randrange(len(ACTIONS))
        first[action_index], second[action_index] = gp.cxOnePoint(
            first[action_index], second[action_index]
        )
        return first, second

    def mutate(individual: Individual):
        if random.random() < ACTION_TREE_SWAP_PROBABILITY:
            first, second = random.sample(range(len(ACTIONS)), 2)
            individual[first], individual[second] = (
                individual[second],
                individual[first],
            )
            return individual,

        action_index = random.randrange(len(ACTIONS))
        expression = partial(
            gp.genGrow, pset=primitive_set, min_=0, max_=max_evolved_depth - 1
        )
        individual[action_index], = gp.mutUniform(
            individual[action_index], expr=expression, pset=primitive_set
        )
        return individual,

    height_limit = gp.staticLimit(
        key=lambda individual: max(tree.height for tree in individual),
        max_value=max_evolved_depth - 1,
    )
    toolbox.register("mate", height_limit(mate))
    toolbox.register("mutate", height_limit(mutate))
    return toolbox


def select_over(
    individuals: Sequence[Individual],
    k: int,
    top_fraction: float = 0.32,
    top_probability: float = 0.8,
) -> list[Individual]:
    """DEAP-compatible over-selection from ranked top and bottom groups."""
    if not individuals:
        raise ValueError("cannot select from an empty population")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be greater than 0 and at most 1")
    if not 0.0 <= top_probability <= 1.0:
        raise ValueError("top_probability must be between 0 and 1")
    ranked = tools.selBest(individuals, len(individuals))
    top_count = max(1, math.ceil(len(ranked) * top_fraction))
    top, bottom = ranked[:top_count], ranked[top_count:]
    selected = []
    for _ in range(k):
        group = top if not bottom or random.random() < top_probability else bottom
        selected.extend(tools.selRandom(group, 1))
    return selected


def action_tree(individual: Individual, action: str) -> gp.PrimitiveTree:
    return individual[ACTIONS.index(action)]


def tree_depth(tree: gp.PrimitiveTree) -> int:
    return tree.height + 1


def tree_size(tree: gp.PrimitiveTree) -> int:
    return len(tree)


def individual_node_count(individual: Individual) -> int:
    return sum(len(tree) for tree in individual)


def individual_depth(individual: Individual) -> int:
    return max(tree_depth(tree) for tree in individual)


def fitness_value(individual: Individual) -> float | None:
    return individual.fitness.values[0] if individual.fitness.valid else None


def set_fitness(individual: Individual, value: float) -> None:
    individual.fitness.values = (float(value),)


def save_individual(individual: Individual, path: str | Path) -> None:
    """Pickle the DEAP individual directly instead of maintaining a parser."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(individual, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_individual(path: str | Path) -> Individual:
    with Path(path).open("rb") as handle:
        individual = pickle.load(handle)
    if (
        not isinstance(individual, list)
        or len(individual) != len(ACTIONS)
        or not all(isinstance(tree, gp.PrimitiveTree) for tree in individual)
        or not hasattr(individual, "fitness")
    ):
        raise ValueError("file does not contain a four-tree DEAP Pac-Man individual")
    return individual


@lru_cache(maxsize=8192)
def _compile_tree(expression: str, primitive_set: gp.PrimitiveSet):
    """Cache DEAP compilation while a tree is repeatedly used in a rollout."""
    return gp.compile(expr=expression, pset=primitive_set)


def evaluate_individual_move(
    individual: Individual,
    terminal_values: dict[str, float],
    legal_moves: dict[str, bool],
    primitive_set: gp.PrimitiveSet,
    rng: random.Random,
    random_tie_break: bool = False,
) -> tuple[str | None, dict[str, float]]:
    arguments = [terminal_values.get(name, 0.0) for name in primitive_set.arguments]
    scores: dict[str, float] = {}
    for action, tree in zip(ACTIONS, individual):
        try:
            policy = _compile_tree(str(tree), primitive_set)
            scores[action] = safe_number(float(policy(*arguments)))
        except (ArithmeticError, OverflowError, TypeError, ValueError):
            scores[action] = 0.0
    legal_actions = [action for action in ACTIONS if legal_moves.get(action, False)]
    if not legal_actions:
        return None, scores
    best_score = max(scores[action] for action in legal_actions)
    candidates = [action for action in legal_actions if scores[action] == best_score]
    if random_tie_break and len(candidates) > 1:
        return rng.choice(candidates), scores
    return min(candidates, key=ACTIONS.index), scores
