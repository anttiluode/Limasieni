"""A tiny one-head encoder transformer in numpy + autograd.

Three toy tasks, chosen because each one stresses a different property of the
attention matrix it needs (see README):

  mirror : y_i = x_{n-1-i}            attend to the reflected position
  shift  : y_i = x_{i-1 mod n}        attend to the previous position (a directed cycle)
  dup    : y_i = 1 if x_i occurs elsewhere in the sequence else 0
                                      attend to tokens equal to yourself

The attention can be either ordinary softmax or a Kirchhoff network
(see kirchhoff.py) so the same harness trains both.
"""
from __future__ import annotations

import autograd.numpy as np
from autograd import grad
import numpy as onp

from kirchhoff import softmax_rows, single_graph_attention, two_port_attention

N, V = 8, 8
TASKS = ("mirror", "shift", "dup")


def make_batch(task, size, rng):
    X = rng.integers(0, V, size=(size, N))
    if task == "mirror":
        Y = X[:, ::-1]
    elif task == "shift":
        Y = onp.roll(X, 1, axis=1)
    elif task == "dup":
        counts = (X[:, :, None] == X[:, None, :]).sum(axis=2)
        Y = (counts >= 2).astype(int)
    else:
        raise ValueError(task)
    return X, Y


def init_params(rng, d=32, hidden=64):
    s = lambda *shape: rng.normal(0, 1 / onp.sqrt(shape[0]), size=shape)
    return {"E": s(V, d), "P": s(N, d), "Wq": s(d, d), "Wk": s(d, d),
            "Wv": s(d, d), "Wo": s(d, d), "W1": s(d, hidden), "b1": onp.zeros(hidden),
            "W2": s(hidden, d), "Wout": s(d, V), "g": onp.zeros(N)}


def attention(params, x, kind):
    """x: (n, d). Returns attention matrix (n, n)."""
    d = x.shape[1]
    q, k = x @ params["Wq"], x @ params["Wk"]
    S = q @ k.T / onp.sqrt(d)
    if kind == "softmax":
        return softmax_rows(S)
    if kind == "two_port":                       # query ports -> key ports, finite sinks
        C = np.exp(S - np.max(S))
        g = np.exp(params["g"]) * 20.0
        return two_port_attention(C, g)
    if kind == "single":                         # one node per token, reciprocal tubes
        W = np.exp((S + S.T) / 2 - np.max(S))
        W = W * (1 - np.eye(x.shape[0]))
        return single_graph_attention(W, np.exp(params["g"]))
    if kind == "valves":                         # one node per token, one-way tubes
        W = np.exp(S - np.max(S)) * (1 - np.eye(x.shape[0]))
        return single_graph_attention(W, np.exp(params["g"]))
    raise ValueError(kind)


def forward(params, X, kind="softmax", return_attn=False):
    logits, attns = [], []
    for row in X:                                 # tiny batches; clarity over speed
        x = params["E"][row] + params["P"]
        A = attention(params, x, kind)
        z = x + (A @ (x @ params["Wv"])) @ params["Wo"]
        m = z + np.maximum(0, z @ params["W1"] + params["b1"]) @ params["W2"]
        logits.append(m @ params["Wout"])
        attns.append(A)
    L = np.stack(logits)
    return (L, attns) if return_attn else L


def loss(params, X, Y, kind):
    L = forward(params, X, kind)
    L = L - np.max(L, axis=2, keepdims=True)
    logp = L - np.log(np.sum(np.exp(L), axis=2, keepdims=True))
    onehot = onp.eye(V)[Y]
    return -np.mean(np.sum(logp * onehot, axis=2))


def accuracy(params, X, Y, kind="softmax"):
    L = forward(params, X, kind)
    return float(onp.mean(onp.argmax(L, axis=2) == Y))


def train(task, kind="softmax", steps=1500, batch=64, lr=3e-3, seed=0, log=None):
    rng = onp.random.default_rng(seed)
    params = init_params(rng)
    g = grad(loss)
    m = {k: onp.zeros_like(v) for k, v in params.items()}
    v = {k: onp.zeros_like(v) for k, v in params.items()}
    b1, b2 = 0.9, 0.999
    for t in range(1, steps + 1):
        X, Y = make_batch(task, batch, rng)
        G = g(params, X, Y, kind)
        for key in params:
            m[key] = b1 * m[key] + (1 - b1) * G[key]
            v[key] = b2 * v[key] + (1 - b2) * G[key] ** 2
            mh, vh = m[key] / (1 - b1 ** t), v[key] / (1 - b2 ** t)
            params[key] = params[key] - lr * mh / (onp.sqrt(vh) + 1e-8)
        if log and t % log == 0:
            Xt, Yt = make_batch(task, 256, onp.random.default_rng(10_000 + t))
            print(f"  {task:6s} {kind:8s} step {t:5d}  loss {float(loss(params, X, Y, kind)):.3f}"
                  f"  acc {accuracy(params, Xt, Yt, kind):.3f}", flush=True)
    return params
