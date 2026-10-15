"""G2/G3: how well can each Kirchhoff layout reproduce the teacher's attention?

For every held-out input we take the teacher's attention matrix A_T and fit the
network's free parameters (tube conductances and sink conductances) by exact
gradient (L-BFGS) to minimise KL(A_T || A_net).  This is the CEILING: the best
any learning rule could ever do with that physical layout.

Layouts:
  single  : one node per token, reciprocal tubes    (classic Physarum graph)
  valves  : one node per token, one-way tubes
  two_port: query port + key port per token, finite sinks

Diagnostics on the teacher matrix itself (no fitting involved):
  sigma   : entropy production (0 iff detailed balance holds)
  maxprin : fraction of rows that 'look away from themselves' after reweighting
            by the stationary distribution (forbidden for reciprocal networks)
"""
import json
import pickle
from pathlib import Path
import numpy as onp
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize

from kirchhoff import (single_graph_attention, two_port_attention, entropy_production,
                       max_principle_violation, kl_rows)
from transformer import TASKS, N

n = N
iu = onp.triu_indices(n, 1)
off = ~onp.eye(n, dtype=bool)


def unpack(theta, layout):
    if layout == "single":
        w = np.exp(theta[: len(iu[0])])
        W = np.zeros((n, n))
        # build symmetric matrix from upper-triangle vector (autograd-friendly)
        U = onp.zeros((len(iu[0]), n, n))
        U[onp.arange(len(iu[0])), iu[0], iu[1]] = 1
        W = np.einsum("e,eij->ij", w, U)
        W = W + W.T
        g = np.exp(theta[len(iu[0]):])
        return single_graph_attention(W, g)
    if layout == "valves":
        m = int(off.sum())
        w = np.exp(theta[:m])
        U = onp.zeros((m, n, n))
        r, c = onp.where(off)
        U[onp.arange(m), r, c] = 1
        W = np.einsum("e,eij->ij", w, U)
        g = np.exp(theta[m:])
        return single_graph_attention(W, g)
    if layout == "two_port":
        C = np.exp(theta[: n * n]).reshape(n, n)
        g = np.exp(theta[n * n:]) * 20.0
        return two_port_attention(C, g)
    raise ValueError(layout)


def n_params(layout):
    return {"single": len(iu[0]) + n, "valves": int(off.sum()) + n, "two_port": n * n + n}[layout]


def fit(A_T, layout, restarts=3, seed=0):
    rng = onp.random.default_rng(seed)
    f = lambda th: kl_obj(th, A_T, layout)
    gf = grad(lambda th: kl_obj(th, A_T, layout))
    best = None
    for r in range(restarts):
        th0 = rng.normal(0, 0.5, n_params(layout))
        res = minimize(f, th0, jac=gf, method="L-BFGS-B", bounds=[(-12, 12)] * len(th0),
                       options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    return best.fun, onp.asarray(unpack(best.x, layout))


def kl_obj(theta, A_T, layout):
    A = unpack(theta, layout)
    return np.mean(np.sum(A_T * (np.log(A_T + 1e-12) - np.log(A + 1e-12)), axis=1))


if __name__ == "__main__":
    report = {}
    for task in TASKS:
        with open(f"results/teacher_{task}.pkl", "rb") as fh:
            T = pickle.load(fh)
        attn = T["attn"][:24]
        rows = {"sigma": [], "maxprin": [], "single": [], "valves": [], "two_port": []}
        examples = {}
        for i, A_T in enumerate(attn):
            rows["sigma"].append(entropy_production(A_T))
            rows["maxprin"].append(max_principle_violation(A_T))
            for layout in ("single", "valves", "two_port"):
                kl, A_fit = fit(A_T, layout, seed=i)
                rows[layout].append(kl)
                if i == 0:
                    examples[layout] = A_fit.tolist()
        report[task] = {k: {"mean": float(onp.mean(v)), "median": float(onp.median(v))} for k, v in rows.items()}
        report[task]["example_teacher"] = attn[0].tolist()
        report[task]["example_fits"] = examples
        print(task, {k: round(report[task][k]["mean"], 4) for k in rows}, flush=True)
    Path("results/g2_layouts.json").write_text(json.dumps(report, indent=1))
