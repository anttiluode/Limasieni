"""G5: replace softmax by a Kirchhoff network INSIDE the transformer and train end to end.

G2 asks whether a layout can copy the teacher's matrix.  This asks a different
question: can the layout still solve the task, perhaps with a different matrix?
(The MLP sees the residual stream, so it can subtract a token's own contribution.)

usage: python g5_end_to_end.py KIND [STEPS] [NSEEDS] [TASK ...]
"""
import json
import os
import sys
from pathlib import Path
import numpy as onp
from transformer import TASKS, train, make_batch, accuracy

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "single"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    first = int(os.environ.get("SEED_START", "1"))
    seeds = range(first, first + (int(sys.argv[3]) if len(sys.argv) > 3 else 2))
    tasks = sys.argv[4:] or TASKS
    path = Path(f"results/g5_{kind}.json")
    report = json.loads(path.read_text()) if path.exists() else {}
    for task in tasks:
        accs = []
        for seed in seeds:
            p = train(task, kind, steps=steps, seed=seed, log=300)
            Xt, Yt = make_batch(task, 512, onp.random.default_rng(777))
            accs.append(accuracy(p, Xt, Yt, kind))
        old = report.get(task, {})
        if old.get("steps") == steps and first > 1:          # append extra seeds to the record
            accs = old["accuracy_per_seed"] + accs
        report[task] = {"accuracy_per_seed": accs, "steps": steps}
        print(f"G5 {kind:8s} {task:6s} held-out acc per seed {accs}", flush=True)
        path.write_text(json.dumps(report, indent=1))
