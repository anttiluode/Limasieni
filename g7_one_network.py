"""G7: ONE grown network for ALL inputs.

G4 grew a fresh network per input sequence. Here a single two-port network is
grown once, with the same local free/nudged rule, against the teacher's attention
on 100 training sequences (the per-tube update is summed over sequences as well
as food sources, so it stays local). The grown network is then frozen and put
into the teacher transformer for 512 NEW sequences it never saw.

A static network has one fixed current split, so it can only carry the part of a
head that does not depend on token content. G7 measures that part:
  content_share = mean over inputs of ||A_input - mean A||^2 / ||A_input||^2
"""
import json
import pickle
from pathlib import Path
import numpy as onp

from physarum import local_update
from kirchhoff import two_port_attention
from transformer import TASKS, make_batch, forward
from g4_grow import forward_with, target_of

N, G = 8, 20.0


def grow_shared(targets, steps=1500, lr=20.0, beta=1e-3):
    logC = onp.zeros((N, N))
    g = onp.full(N, G)
    for _ in range(steps):
        C = onp.exp(logC)
        step = sum(local_update(C, g, A_T, beta) for A_T in targets) / len(targets)
        logC = onp.clip(logC + lr * step, -12, 12)
    return onp.exp(logC)


if __name__ == "__main__":
    report = {}
    for task in TASKS:
        T = pickle.load(open(f"results/teacher_{task}.pkl", "rb"))
        P = T["params"]
        train_A = T["attn"][:100]
        C = grow_shared(train_A)
        A_static = onp.asarray(two_port_attention(C, onp.full(N, G)))

        Xt, Yt = make_batch(task, 512, onp.random.default_rng(4242))
        _, attn_test = forward(P, Xt, "softmax", return_attn=True)
        attn_test = onp.array([onp.asarray(a) for a in attn_test])
        acc_static = float(onp.mean([onp.mean(onp.argmax(forward_with(P, x, A_static), 1) == y)
                                     for x, y in zip(Xt, Yt)]))
        acc_teacher = float(onp.mean([onp.mean(onp.argmax(forward_with(P, x, a), 1) == y)
                                      for x, y, a in zip(Xt, Yt, attn_test)]))
        mean_A = attn_test.mean(0)
        content_share = float(onp.mean(((attn_test - mean_A) ** 2).sum((1, 2)) / (attn_test ** 2).sum((1, 2))))
        kl = float(onp.mean([onp.mean(onp.sum(a * onp.log((a + 1e-12) / (A_static + 1e-12)), 1)) for a in attn_test]))
        report[task] = {"acc_teacher_own_attention": acc_teacher,
                        "acc_one_static_grown_network": acc_static,
                        "kl_teacher_vs_static": kl,
                        "content_share_of_head": content_share,
                        "static_matrix": A_static.tolist()}
        print(task, {k: round(v, 4) for k, v in report[task].items() if k != "static_matrix"}, flush=True)
    Path("results/g7_one_network.json").write_text(json.dumps(report, indent=1))
