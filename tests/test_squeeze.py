import sys
import unittest
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recall_circuit as rc
import g10_squeeze as g


class TestSqueeze(unittest.TestCase):
    def test_importing_g10_leaves_the_g9_circuit_alone(self):
        self.assertEqual(rc.KEYS, 12)

    def test_twins_never_share_a_sequence(self):
        X, Y = g.make_batch("twin", 200, onp.random.default_rng(0))
        for x, y in zip(X, Y):
            keys = (x[:7] - g.KEYS) // 6
            self.assertEqual(sorted(keys // 2), list(range(7)))      # one member of each twin pair
            self.assertEqual(y, (x[:7] - g.KEYS)[list(keys).index(x[7])] % 6)

    def test_arbitrary_keys_do_cooccur_with_their_twin(self):
        X, _ = g.make_batch("arbitrary", 200, onp.random.default_rng(1))
        keys = (X[:, :7] - g.KEYS) // 6
        both = sum(len(set(k // 2)) < 7 for k in keys)
        self.assertGreater(both, 150)

    def test_identical_pins_are_merged_distant_pins_are_not(self):
        grid = onp.array([(a, b) for a in onp.linspace(-1, 1, 4) for b in onp.linspace(-1, 1, 4)])
        pins = grid[:g.KEYS].copy()                  # 2 pins, spacing 0.67 >> window 0.25
        pins[1] = pins[0]
        groups = g.components(g.key_resistance(pins, 2), 0.5)
        self.assertIn([0, 1], groups)
        self.assertEqual(len(groups), g.KEYS - 1)

    def test_short_run_restores_key_count(self):
        params, traj = g.run("twin", 1, 0, steps=2, rec=1)
        self.assertEqual(rc.KEYS, 12)
        self.assertEqual(traj.shape, (3, g.KEYS, 1))


if __name__ == "__main__":
    unittest.main()
