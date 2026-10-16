"""G9: a recall machine that is ONLY a circuit, trained by a local rule.

No embeddings, no Wv, no MLP, no backprop.

Circuit for one sequence (8 tokens, recall task from transformer.py):
  * every token owns a query port and a key port (two-port layout);
  * tube (query i -> key j) is a series chain of 4 window elements. Element k
    conducts best when pin k of token x_i and pin k of token x_j sit at the same
    bias voltage:   C_ij = exp(theta_ij) / sum_k exp((b[x_i,k] - b[x_j,k])^2 / w^2)
  * each key port drains through sink g into a VALUE BUS: pair tokens feed the bus
    of their value (0..5); the query token itself feeds a 'null' bus (6);
  * one unit of current is injected at the query port (position 7). The answer is the
    value bus that receives the most current.

Learning (centred equilibrium propagation for a linear resistor network):
  free    : inject, let the circuit settle.
  nudged  : same, plus a weak pull (+beta and -beta) of the bus currents toward the
            target (all current into the correct value bus, none into the others).
  tube    : dL/dC_e = [ dv_e(+beta)^2 - dv_e(-beta)^2 ] / (4 beta)   <- its own drops only
  pins    : each window element converts its tube's signal into a nudge for the two
            pins it touches (chain rule through its own I-V curve). Pins belonging to
            the same token value share one register (weight sharing, like a
            convolution kernel shared across positions) - that is the one non-spatially-
            local step, and it is stated as such.

Given the biases, the circuit is LINEAR in the signal, so the free/nudged contrast is
exact EP, not an approximation of a nonlinear one.
"""
from __future__ import annotations

import numpy as onp

N, PINS, G, NBUS = 8, 4, 20.0, 7
KEYS = 12          # key tokens are 0..KEYS-1; pair tokens KEYS + 6*key + value (G10 sets 14)
QUERY = N - 1


def bus_of(x):
    c = onp.full(N, NBUS - 1)
    m = x >= KEYS
    c[m] = (x[m] - KEYS) % 6
    return c


def init(rng, vocab=84, w0=4.0, pins=PINS, scale=1.0):
    return {"theta": onp.zeros((N, N)), "bias": rng.normal(0, scale, (vocab, pins)),
            "logw": onp.array([onp.log(w0)])}


def tubes(params, x, content=True):
    """log conductances (N,N) and the pieces needed for pin gradients."""
    if not content:
        return onp.clip(params["theta"], -20, 6), None
    b = params["bias"][x]
    w2 = onp.exp(2 * params["logw"][0])
    d = b[:, None, :] - b[None, :, :]                    # query i minus key j
    z = d ** 2 / w2
    mx = z.max(2, keepdims=True)
    S = onp.exp(z - mx)
    p = S / S.sum(2, keepdims=True)                      # share of each element in the series resistance
    logR = mx[..., 0] + onp.log(S.sum(2))
    return onp.clip(params["theta"] - logR, -20, 6), (d, z, p, w2)


def system(C):
    top = onp.concatenate([onp.diag(C.sum(1)), -C], 1)
    bot = onp.concatenate([-C.T, onp.diag(C.sum(0) + G)], 1)
    return onp.concatenate([top, bot], 0)


def bus_matrix(x):
    P = onp.zeros((NBUS, 2 * N))
    P[bus_of(x), N + onp.arange(N)] = G                 # bus current = sum of its key sink currents
    return P


def solve(C, x, y=None, beta=0.0):
    M = system(C)
    rhs = onp.zeros(2 * N); rhs[QUERY] = 1.0
    if beta:
        P = bus_matrix(x)
        target = onp.zeros(NBUS); target[y] = 1.0
        M = M + beta * P.T @ P
        rhs = rhs + beta * P.T @ target
    return onp.linalg.solve(M, rhs)


def bus_currents(C, x):
    return bus_matrix(x) @ solve(C, x)


def drops(v):
    return v[:N, None] - v[None, N:]                    # (query i, key j)


def local_grads(params, X, Y, beta=1e-3, content=True):
    """Per-tube free/nudged contrast, routed to theta, pins and width. Returns grads, loss, acc."""
    g = {k: onp.zeros_like(v) for k, v in params.items()}
    loss, correct = 0.0, 0
    for x, y in zip(X, Y):
        logC, aux = tubes(params, x, content)
        C = onp.exp(logC)
        I = bus_currents(C, x)
        t = onp.zeros(NBUS); t[y] = 1.0
        loss += 0.5 * ((I - t) ** 2).sum()
        correct += int(onp.argmax(I[:6]) == y)
        dp, dm = drops(solve(C, x, y, beta)), drops(solve(C, x, y, -beta))
        dL_dlogC = C * (dp ** 2 - dm ** 2) / (4 * beta)
        dL_dlogC[(logC <= -20) | (logC >= 6)] = 0.0
        g["theta"] += dL_dlogC
        if content:
            d, z, p, w2 = aux
            k = dL_dlogC[..., None] * p * 2 * d / w2      # dlogC/db_query = -p*2d/w2, key side = +
            onp.add.at(g["bias"], x, -k.sum(1))           # each query port's pins
            onp.add.at(g["bias"], x, k.sum(0))            # each key port's pins
            g["logw"][0] += (dL_dlogC * 2 * (p * z).sum(2)).sum()
    n = len(X)
    return {k: v / n for k, v in g.items()}, loss / n, correct / n


def exact_loss(params, X, Y, content=True):
    """Same circuit and loss in autograd, for checking the local rule."""
    import autograd.numpy as anp
    tot = 0.0
    for x, y in zip(X, Y):
        if content:
            b = params["bias"][x]
            w2 = anp.exp(2 * params["logw"][0])
            z = (b[:, None, :] - b[None, :, :]) ** 2 / w2
            mx = anp.max(z, axis=2, keepdims=True)
            logC = params["theta"] - (mx[..., 0] + anp.log(anp.sum(anp.exp(z - mx), axis=2)))
        else:
            logC = params["theta"]
        C = anp.exp(logC)
        top = anp.concatenate([anp.diag(anp.sum(C, 1)), -C], 1)
        bot = anp.concatenate([-C.T, anp.diag(anp.sum(C, 0) + G)], 1)
        M = anp.concatenate([top, bot], 0)
        rhs = onp.zeros(2 * N); rhs[QUERY] = 1.0
        I = bus_matrix(x) @ anp.linalg.solve(M, rhs)
        t = onp.zeros(NBUS); t[y] = 1.0
        tot = tot + 0.5 * anp.sum((I - t) ** 2)
    return tot / len(X)


def accuracy(params, X, Y, content=True):
    hits = 0
    for x, y in zip(X, Y):
        logC, _ = tubes(params, x, content)
        hits += int(onp.argmax(bus_currents(onp.exp(logC), x)[:6]) == y)
    return hits / len(X)
