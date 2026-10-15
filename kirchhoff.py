"""Kirchhoff networks that produce attention matrices.

Two physical layouts are implemented.

1. single_graph_attention(W, g)
   One node per token. Tokens are both the place current is injected (food
   source) and the place it leaves (sink conductance g_j to ground).  For a
   symmetric (reciprocal) tube network W this is the electrical current
   divider; for a non-symmetric W it is an absorbing random walk ("valves").
       A = (L + G)^-1 G,   L = diag(W 1) - W,   G = diag(g)
   Row i = where a unit of current injected at token i leaves the network.

2. two_port_attention(C, g)
   Two nodes per token: a query port (source, no sink) and a key port (sink g).
   Tubes C_ij connect query i to key j.  In the grounded limit g -> inf this is
   exactly current division:  A_ij = C_ij / sum_k C_ik.  With C = exp(q.k/sqrt d)
   that IS softmax attention.

Everything is plain numpy so it can be wrapped by autograd.
"""
from __future__ import annotations

import autograd.numpy as np


def laplacian(W):
    """Out-degree Laplacian. Symmetric W -> ordinary graph Laplacian."""
    return np.diag(np.sum(W, axis=1)) - W


def single_graph_attention(W, g):
    n = W.shape[0]
    M = laplacian(W) + np.diag(g)
    return np.linalg.solve(M, np.diag(g))


def two_port_system(C, g):
    """2n x 2n conductance matrix: nodes 0..n-1 query ports, n..2n-1 key ports."""
    n = C.shape[0]
    top = np.concatenate([np.diag(np.sum(C, axis=1)), -C], axis=1)
    bot = np.concatenate([-C.T, np.diag(np.sum(C, axis=0) + g)], axis=1)
    return np.concatenate([top, bot], axis=0)


def two_port_potentials(C, g):
    """Column i = node potentials for a unit current injected at query port i."""
    n = C.shape[0]
    M = two_port_system(C, g)
    rhs = np.concatenate([np.eye(n), np.zeros((n, n))], axis=0)
    return np.linalg.solve(M, rhs)


def two_port_attention(C, g):
    n = C.shape[0]
    V = two_port_potentials(C, g)
    return (V[n:, :] * g[:, None]).T     # A_ij = g_j * v_{k_j} for injection at q_i


def softmax_rows(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


# ---------------------------------------------------------------- diagnostics
def stationary(A, iters=5000):
    p = np.ones(A.shape[0]) / A.shape[0]
    for _ in range(iters):
        p = p @ A
    return p / p.sum()


def entropy_production(A, eps=1e-12):
    """sigma = sum pi_i A_ij log(pi_i A_ij / pi_j A_ji). Zero iff A is reversible.

    A reciprocal Kirchhoff network can only produce reversible matrices
    (detailed balance g_i A_ij = g_j A_ji), so sigma > 0 is an obstruction.
    """
    A = np.asarray(A)
    pi = stationary(A)
    F = pi[:, None] * A + eps
    return float(np.sum(F * np.log(F / F.T)))


def max_principle_violation(A, eps=1e-12):
    """Fraction of rows whose largest entry of K = A diag(1/pi) is off the diagonal.

    For a reciprocal network K = (L+G)^-1 and the discrete maximum principle
    forces K_ii >= K_ij: the potential is highest where current is injected,
    so current leaves preferentially near its source.  Any reversible A that
    'looks away from itself' (relative to pi) cannot be a reciprocal network.
    """
    A = np.asarray(A)
    pi = stationary(A)
    K = A / (pi[None, :] + eps)
    return float(np.mean(np.argmax(K, axis=1) != np.arange(A.shape[0])))


def kl_rows(P, Q, eps=1e-12):
    P = np.asarray(P) + eps
    Q = np.asarray(Q) + eps
    return float(np.mean(np.sum(P * np.log(P / Q), axis=1)))
