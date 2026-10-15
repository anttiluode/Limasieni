"""G9: train the pure-circuit recall machine (recall_circuit.py) with the LOCAL rule.

Arms (same circuit, same optimiser, same batches):
  local   : window tubes, updates from free/nudged tube contrasts only
  exact   : window tubes, updates from the exact autograd gradient (reference)
  static  : positional tubes only (no bias dependence), local rule; content-free control

usage: python g9_local_recall.py ARM [SEED] [STEPS]
"""
import json
import sys
from pathlib import Path
import numpy as onp
from autograd import grad

from recall_circuit import init, local_grads, exact_loss, accuracy
from transformer import make_batch

exact_grad = grad(exact_loss)


def run(arm, seed=1, steps=3000, batch=32, lr=0.05, log=500):
    rng = onp.random.default_rng(seed)
    params = init(rng)
    content = arm != "static"
    m = {k: onp.zeros_like(v) for k, v in params.items()}
    s = {k: onp.zeros_like(v) for k, v in params.items()}
    Xv, Yv = make_batch("recall", 512, onp.random.default_rng(777))
    Yv = Yv[:, -1]
    curve = []
    for t in range(1, steps + 1):
        X, Y = make_batch("recall", batch, rng)
        Y = Y[:, -1]
        if arm == "exact":
            g = {k: onp.asarray(v) for k, v in exact_grad(params, X, Y).items()}
        else:
            g, _, _ = local_grads(params, X, Y, content=content)
        for k in params:                                   # Adam: per-parameter state only
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - 0.9 ** t)) / (onp.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
        if t % log == 0:
            acc = accuracy(params, Xv, Yv, content)
            curve.append((t, acc))
            print(f"  {arm:6s} seed {seed} step {t:5d}  held-out acc {acc:.4f}", flush=True)
    return params, curve


if __name__ == "__main__":
    arm = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    params, curve = run(arm, seed, steps)
    path = Path("results/g9_local_recall.json")
    report = json.loads(path.read_text()) if path.exists() else {}
    report[f"{arm}_seed{seed}"] = {"curve": curve, "final_heldout_acc": curve[-1][1], "steps": steps}
    path.write_text(json.dumps(report, indent=1))
    if arm == "local":
        import pickle
        pickle.dump(params, open(f"results/g9_local_params_seed{seed}.pkl", "wb"))
