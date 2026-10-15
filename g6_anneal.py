"""G6: what does UNSUPERVISED Physarum do to a transformer's attention?

On a grounded two-port network, food at query i sends flux Q_ij = A_ij through
tube (i, j).  Tero's adaptation  dC/dt = Q^mu - C  then gives:
  mu < 1 : loops win, attention flattens toward uniform
  mu = 1 : every row-normalised C is a fixed point (attention unchanged)
  mu > 1 : tree regime, winner takes all (hard argmax attention)
We start from the teacher's own attention and put the result back into the teacher.
"""
import json
import pickle
import numpy as onp
from g4_grow import forward_with, target_of


def anneal(A, mu, steps=400, rate=0.1):
    C = A.copy()
    for _ in range(steps):
        Q = C / C.sum(1, keepdims=True)
        C += rate * (Q ** mu - C)
    return C / C.sum(1, keepdims=True)


if __name__ == "__main__":
    report = {}
    for task in ("mirror", "shift", "dup"):
        T = pickle.load(open(f"results/teacher_{task}.pkl", "rb"))
        report[task] = {}
        for mu in (0.7, 1.0, 1.5):
            accs, rowmax = [], []
            for A, x in zip(T["attn"][:100], T["X"][:100]):
                Q = anneal(A, mu)
                accs.append(float(onp.mean(onp.argmax(forward_with(T["params"], x, Q), 1) == target_of(task, x))))
                rowmax.append(float(Q.max(1).mean()))
            report[task][str(mu)] = {"accuracy": float(onp.mean(accs)), "mean_row_max": float(onp.mean(rowmax))}
        print(task, report[task])
    json.dump(report, open("results/g6_anneal.json", "w"), indent=1)
