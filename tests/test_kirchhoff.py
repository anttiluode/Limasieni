import sys
import unittest
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kirchhoff import (two_port_attention, softmax_rows, single_graph_attention,
                       entropy_production, max_principle_violation)
from physarum import check_gradient


class TestKirchhoff(unittest.TestCase):
    def setUp(self):
        self.rng = onp.random.default_rng(0)
        self.n = 8

    def test_softmax_is_grounded_current_division(self):
        S = self.rng.normal(0, 1.5, (self.n, self.n))
        A = two_port_attention(onp.exp(S), onp.full(self.n, 1e9))
        self.assertLess(onp.abs(A - softmax_rows(S)).max(), 1e-7)

    def test_rows_conserve_current(self):
        C = self.rng.uniform(0.1, 2, (self.n, self.n))
        for A in (two_port_attention(C, onp.full(self.n, 3.0)),
                  single_graph_attention(C + C.T, self.rng.uniform(.2, 2, self.n))):
            self.assertLess(onp.abs(A.sum(1) - 1).max(), 1e-12)
            self.assertGreaterEqual(A.min(), -1e-12)

    def test_reciprocal_network_obeys_detailed_balance_and_max_principle(self):
        W = self.rng.uniform(0, 1, (self.n, self.n)); W = W + W.T
        onp.fill_diagonal(W, 0)
        g = self.rng.uniform(.2, 2, self.n)
        A = single_graph_attention(W, g)
        self.assertLess(onp.abs(g[:, None] * A - (g[:, None] * A).T).max(), 1e-12)
        self.assertLess(entropy_production(A), 1e-10)
        self.assertEqual(max_principle_violation(A), 0.0)

    def test_permutation_breaks_max_principle(self):
        P = onp.eye(self.n)[::-1] * 0.93 + 0.01        # soft mirror, reversible
        self.assertEqual(max_principle_violation(P), 1.0)

    def test_local_rule_is_the_gradient(self):
        cos, rel = check_gradient()
        self.assertGreater(cos, 0.9999)
        self.assertLess(rel, 0.05)


if __name__ == "__main__":
    unittest.main()
