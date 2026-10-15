"""Growing a two-port Physarum network until it carries the teacher's attention.

Network: query port i (food source) -- tube C_ij -- key port j -- sink g_j -- ground.

Local learning rule (equilibrium propagation / coupled learning for a resistor
network, Scellier & Bengio 2017; Stern et al. 2021):

  free phase   : inject one unit of 'food' at query port i, let the flow settle.
  nudged phase : same, plus a weak pull (strength beta) on every key sink toward
                 the teacher's uptake A_T[i, j].
  update       : each tube compares its OWN pressure drop in the two phases
                     dlogC_e = lr * [ (dv_e^free)^2 - (dv_e^nudged)^2 ] / (2 beta)
                 (summed over the n food sources).  Nothing global is read.

For small beta this is exact gradient descent (in log-conductance) on
    loss = 1/2 sum_ij (uptake_ij - A_T_ij)^2.
check_gradient() verifies that numerically.
"""
from __future__ import annotations

import numpy as onp
import autograd.numpy as np
from autograd import grad

from kirchhoff import two_port_system, two_port_attention


def potentials(C, g, A_T=None, beta=0.0):
    """Free (beta=0) or nudged node potentials; column i = food at query i."""
    n = C.shape[0]
    M = two_port_system(C, g)
    rhs = onp.concatenate([onp.eye(n), onp.zeros((n, n))], axis=0)
    if beta:
        M = M + onp.diag(onp.concatenate([onp.zeros(n), beta * g ** 2]))
        rhs = rhs + onp.concatenate([onp.zeros((n, n)), beta * (g[:, None] * A_T.T)], axis=0)
    return onp.linalg.solve(M, rhs)


def tube_drops(V, n):
    """dv[i, a, b] = pressure drop across tube (query a -> key b) when food is at query i."""
    q, k = V[:n, :], V[n:, :]          # (n nodes, n injections)
    return q.T[:, :, None] - k.T[:, None, :]


def local_update(C, g, A_T, beta=1e-3):
    n = C.shape[0]
    dF = tube_drops(potentials(C, g), n)
    dN = tube_drops(potentials(C, g, A_T, beta), n)
    # per-tube contrast, summed over food sources; multiply by C for log-space step
    return C * ((dF ** 2 - dN ** 2).sum(axis=0) / (2 * beta))


def loss(logC, g, A_T):
    A = two_port_attention(np.exp(logC), g)
    return 0.5 * np.sum((A - A_T) ** 2)


exact_grad = grad(loss)


def check_gradient(n=8, seed=0):
    rng = onp.random.default_rng(seed)
    logC = rng.normal(0, 0.5, (n, n))
    g = onp.full(n, 20.0)
    A_T = rng.dirichlet(onp.ones(n), size=n)
    local = local_update(onp.exp(logC), g, A_T)       # = -dLoss/dlogC (ascent direction)
    true = -exact_grad(logC, g, A_T)
    cos = float((local * true).sum() / (onp.linalg.norm(local) * onp.linalg.norm(true)))
    rel = float(onp.linalg.norm(local - true) / onp.linalg.norm(true))
    return cos, rel


def grow(A_T, steps=600, lr=4.0, g=20.0, beta=1e-3, record_every=10, rule="local"):
    """Start from a uniform mesh and grow. Returns history of conductances and errors."""
    n = A_T.shape[0]
    gv = onp.full(n, g)
    logC = onp.zeros((n, n))
    hist = {"C": [], "kl": [], "mse": [], "step": []}
    for t in range(steps + 1):
        C = onp.exp(logC)
        if t % record_every == 0:
            A = onp.asarray(two_port_attention(C, gv))
            hist["C"].append(C.copy())
            hist["mse"].append(float(0.5 * ((A - A_T) ** 2).sum()))
            hist["kl"].append(float(onp.mean(onp.sum(A_T * onp.log((A_T + 1e-12) / (A + 1e-12)), axis=1))))
            hist["step"].append(t)
        if t == steps:
            break
        if rule == "local":
            step = local_update(C, gv, A_T, beta)
        else:
            step = -exact_grad(logC, gv, A_T)
        logC = onp.clip(logC + lr * step, -12, 12)
    return hist
