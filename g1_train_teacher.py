"""G1: train the softmax teacher transformer on each task, save weights + attention."""
import pickle
from pathlib import Path
import numpy as onp
from transformer import TASKS, train, make_batch, accuracy, forward

OUT = Path("results")
OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    summary = {}
    for task in TASKS:
        params = train(task, "softmax", steps=1200, log=400, seed=1)
        Xt, Yt = make_batch(task, 512, onp.random.default_rng(777))
        acc = accuracy(params, Xt, Yt)
        _, attns = forward(params, Xt[:200], "softmax", return_attn=True)
        attns = onp.array([onp.asarray(a) for a in attns])
        with open(OUT / f"teacher_{task}.pkl", "wb") as f:
            pickle.dump({"params": {k: onp.asarray(v) for k, v in params.items()},
                         "X": Xt[:200], "attn": attns, "acc": acc}, f)
        summary[task] = acc
        print(f"G1 {task}: held-out accuracy {acc:.4f}", flush=True)
    print(summary)
