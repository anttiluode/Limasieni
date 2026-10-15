"""G4: grow a two-port Physarum network from a uniform mesh until it carries the
teacher transformer's attention, using only the local free/nudged pressure rule.

Then the decisive check: put the GROWN matrix back into the transformer in place
of its own softmax attention and see whether the transformer still answers
correctly.  If it does, the slime network is doing the transformer's routing.
"""
import json
import pickle
from pathlib import Path
import numpy as onp

from physarum import grow, check_gradient
from kirchhoff import two_port_attention
from transformer import TASKS


def forward_with(params, row, A):
    x = params["E"][row] + params["P"]
    z = x + (A @ (x @ params["Wv"])) @ params["Wo"]
    m = z + onp.maximum(0, z @ params["W1"] + params["b1"]) @ params["W2"]
    return m @ params["Wout"]


def target_of(task, row):
    if task == "mirror":
        return row[::-1]
    if task == "shift":
        return onp.roll(row, 1)
    counts = (row[:, None] == row[None, :]).sum(1)
    return (counts >= 2).astype(int)


if __name__ == "__main__":
    cos, rel = check_gradient()
    print(f"local rule vs exact gradient: cosine {cos:.6f}, relative error {rel:.4f}")
    report = {"gradient_check": {"cosine": cos, "relative_error": rel}}
    histories = {}
    for task in TASKS:
        with open(f"results/teacher_{task}.pkl", "rb") as fh:
            T = pickle.load(fh)
        P, X, attn = T["params"], T["X"], T["attn"]
        rows = []
        for i in range(12):
            A_T = attn[i]
            h_loc = grow(A_T, steps=1500, lr=20.0, record_every=25, rule="local")
            h_ex = grow(A_T, steps=1500, lr=20.0, record_every=1500, rule="exact")
            C = h_loc["C"][-1]
            A_P = onp.asarray(two_port_attention(C, onp.full(8, 20.0)))
            y = target_of(task, X[i])
            acc_teacher = float(onp.mean(onp.argmax(forward_with(P, X[i], A_T), 1) == y))
            acc_grown = float(onp.mean(onp.argmax(forward_with(P, X[i], A_P), 1) == y))
            acc_uniform = float(onp.mean(onp.argmax(forward_with(P, X[i], onp.full((8, 8), 1 / 8)), 1) == y))
            pruned = float(onp.mean(C < 0.01 * C.max()))
            rows.append({"kl_local": h_loc["kl"][-1], "kl_exact": h_ex["kl"][-1],
                         "acc_teacher_attn": acc_teacher, "acc_grown_attn": acc_grown,
                         "acc_uniform_attn": acc_uniform, "tubes_pruned_below_1pct": pruned})
            if i == 0:
                histories[task] = {"X": X[i].tolist(), "A_T": A_T.tolist(),
                                   "steps": h_loc["step"], "kl": h_loc["kl"],
                                   "C": [c.tolist() for c in h_loc["C"]]}
        agg = {k: float(onp.mean([r[k] for r in rows])) for k in rows[0]}
        report[task] = {"mean": agg, "per_input": rows}
        print(task, {k: round(v, 4) for k, v in agg.items()}, flush=True)
    Path("results/g4_grow.json").write_text(json.dumps(report, indent=1))
    Path("results/growth_histories.json").write_text(json.dumps(histories))
