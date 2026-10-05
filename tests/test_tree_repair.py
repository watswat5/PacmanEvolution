"""The repair step must shrink trees without changing what they compute."""

import random
import unittest

from deap import gp

from GeneticAlgo.gp_core import repair_individual, safe_number, simplify_tree
from GeneticAlgo.train_board_graph import build_primitive_set, build_toolbox


def text(expression: str, pset: gp.PrimitiveSet) -> str:
    tree = gp.PrimitiveTree.from_string(expression, pset)
    return str(gp.PrimitiveTree(simplify_tree(tree, pset)))


class TreeRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pset = build_primitive_set()

    def test_trivial_chains_collapse(self) -> None:
        # Constants are -1, 0, 0.5, 1, 2 as constant_0 .. constant_4.
        cases = {
            "abs(abs(pellet_here))": "abs(pellet_here)",
            "abs(neg(pellet_here))": "abs(pellet_here)",
            "neg(neg(neg(pellet_here)))": "neg(pellet_here)",
            "sqrt(abs(neg(pellet_here)))": "sqrt(pellet_here)",
            "mul(mul(pellet_here, constant_3), constant_3)": "pellet_here",
            "add(pellet_here, neg(neg(constant_1)))": "pellet_here",
            "sub(is_forward, is_forward)": "constant_1",
            "mul(constant_4, constant_2)": "constant_3",
            "mul(neg(pellet_here), neg(is_forward))": "mul(pellet_here, is_forward)",
            "if_lte(constant_0, constant_3, pellet_here, is_forward)": "pellet_here",
            "max(abs(abs(is_reverse)), abs(is_reverse))": "abs(is_reverse)",
        }
        for expression, expected in cases.items():
            self.assertEqual(text(expression, self.pset), expected, expression)

    def test_inexact_rewrites_are_left_alone(self) -> None:
        # div floors its denominator and clamps to +/-1000.
        for expression in (
            "div(pellet_here, constant_3)",
            "div(pellet_here, pellet_here)",
            "add(constant_4, constant_4)",  # 4 is not one of the constants
        ):
            self.assertEqual(text(expression, self.pset), expression)

    def test_repair_preserves_values_on_random_trees(self) -> None:
        random.seed(7)
        toolbox = build_toolbox(self.pset, 2, 8, 16, 0.5, 0.8, repair=False)
        shrunk = 0
        for _ in range(300):
            tree = toolbox.individual()
            for _ in range(3):
                tree, = toolbox.mutate(tree)
            original = gp.compile(str(tree), self.pset)
            size = len(tree)
            repair_individual(tree, self.pset)
            repaired = gp.compile(str(tree), self.pset)
            shrunk += len(tree) < size
            self.assertLessEqual(len(tree), size)
            for _ in range(8):
                values = [random.choice((0.0, 1.0, random.uniform(-1, 1)))
                          for _ in self.pset.arguments]
                self.assertEqual(
                    safe_number(float(original(*values))),
                    safe_number(float(repaired(*values))),
                )
        self.assertGreater(shrunk, 0)


if __name__ == "__main__":
    unittest.main()
