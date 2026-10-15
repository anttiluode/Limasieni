import sys
import unittest
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer import make_batch, init_params, attention


class TestTasks(unittest.TestCase):
    def test_recall_labels_point_at_the_queried_pair(self):
        X, Y = make_batch("recall", 50, onp.random.default_rng(0))
        for x, y in zip(X, Y):
            keys = (x[:7] - 12) // 6
            vals = (x[:7] - 12) % 6
            self.assertEqual(len(set(keys)), 7)                 # keys are distinct
            self.assertTrue(onp.all(y[:7] == -1))               # only the query is scored
            self.assertEqual(y[7], vals[list(keys).index(x[7])])

    def test_window4_tube_conducts_best_between_matching_biases(self):
        p = init_params(onp.random.default_rng(0), d=16, hidden=32, vocab=84)
        row = onp.array([12, 13, 14, 15, 16, 17, 18, 12])        # positions 0 and 7 share a token
        p["theta"] = onp.zeros((8, 8))
        A = onp.asarray(attention(p, p["E"][row] + p["P"], "window4", row))
        self.assertAlmostEqual(A.sum(1).max(), 1.0, places=9)
        # from position 7 the matching token (pos 0) and itself carry the most current
        self.assertEqual(set(onp.argsort(A[7])[-2:]), {0, 7})


if __name__ == "__main__":
    unittest.main()
