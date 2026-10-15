import sys
import unittest
from pathlib import Path

import numpy as onp
from autograd import grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall_circuit import init, local_grads, exact_loss, tubes, bus_currents
from transformer import make_batch


class TestRecallCircuit(unittest.TestCase):
    def test_local_rule_equals_exact_gradient_for_every_parameter(self):
        rng = onp.random.default_rng(0)
        P = init(rng)
        P["theta"] = rng.normal(0, .5, (8, 8))
        X, Y = make_batch("recall", 8, onp.random.default_rng(1))
        gl, _, _ = local_grads(P, X, Y[:, -1])
        ge = grad(exact_loss)(P, X, Y[:, -1])
        for k in P:
            a, b = gl[k].ravel(), onp.asarray(ge[k]).ravel()
            self.assertGreater(a @ b / (onp.linalg.norm(a) * onp.linalg.norm(b)), 0.9999, k)

    def test_bus_currents_conserve_the_injected_unit(self):
        P = init(onp.random.default_rng(3))
        X, _ = make_batch("recall", 5, onp.random.default_rng(4))
        for x in X:
            logC, _ = tubes(P, x)
            self.assertAlmostEqual(bus_currents(onp.exp(logC), x).sum(), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
