"""G10: squeeze test on the G9 recall circuit (prompted by Goodfire's "Neural Networks Are
Already Clean", which separates two accounts of mixed units: superposition-style packing
of unrelated features versus recursive generalization along related ones).

The circuit has no input features; tokens are bare ids. So "related" can only come from
task statistics. Two structures with the same 14 keys, 6 values, 7 pairs + 1 query:

  arbitrary : the 7 keys of a sequence are drawn at random, so every two keys co-occur
  twin      : keys come in 7 twin pairs {2a, 2a+1}; a sequence holds exactly one member
              of each pair, so twins NEVER co-occur and confusing them costs nothing

Squeeze: every token gets P bias pins clipped to [-1, 1], and the window width is FIXED
at w = 0.25 (chosen by a short seed-1 sweep over 0.15 / 0.25 / 0.35 / 0.5 at P = 1: at 0.15
one pin still held 14 arbitrary keys at 0.99, at 0.25 it dropped to 0.86 while twin
structure kept 0.97).

Training is G9's local free/nudged rule (Adam, or plain gradient descent), pins start tiny (scale 0.01) so every
key starts in one cluster and must split. Key-token pins are recorded during training.

usage: python g10_squeeze.py STRUCTURE PINS SEED [INIT_SCALE] [OPT LR STEPS]
       OPT = adam (default, lr 0.02, 3000 steps) or gd (plain gradient descent)
"""
import json
import sys
from functools import wraps
from pathlib import Path
import numpy as onp

import recall_circuit as rc

KEYS, VALS, VOCAB = 14, 6, 14 + 14 * 6


def uses_14_keys(fn):
    """Run fn with the circuit's key count set to 14, restoring it afterwards, so importing
    this module never changes G9's 12-key circuit in the same process."""
    @wraps(fn)
    def inner(*a, **k):
        old, rc.KEYS = rc.KEYS, KEYS
        try:
            return fn(*a, **k)
        finally:
            rc.KEYS = old
    return inner
W = 0.25
TWIN = [(2 * a, 2 * a + 1) for a in range(7)]


def make_batch(structure, size, rng):
    X = onp.zeros((size, rc.N), dtype=int)
    Y = onp.zeros(size, dtype=int)
    for r in range(size):
        if structure == "twin":
            keys = 2 * onp.arange(7) + rng.integers(0, 2, 7)
            rng.shuffle(keys)
        else:
            keys = rng.permutation(KEYS)[:7]
        vals = rng.integers(0, VALS, 7)
        X[r, :7] = KEYS + 6 * keys + vals
        pick = rng.integers(0, 7)
        X[r, 7] = keys[pick]
        Y[r] = vals[pick]
    return X, Y


def key_resistance(pins, pin_count):
    """log series resistance between every two KEY tokens, 0 = identical."""
    d2 = ((pins[:, None, :] - pins[None, :, :]) ** 2) / W ** 2
    return onp.log(onp.exp(d2).sum(2)) - onp.log(pin_count)


def components(R, thr):
    """connected components of the 'merged' graph (R < thr) among keys."""
    lab = list(range(KEYS))
    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for a in range(KEYS):
        for b in range(a + 1, KEYS):
            if R[a, b] < thr:
                lab[find(a)] = find(b)
    groups = {}
    for i in range(KEYS):
        groups.setdefault(find(i), []).append(i)
    return sorted(sorted(g) for g in groups.values())


def merge_stats(R, thr=1.0):
    twin = [R[a, b] < thr for a, b in TWIN]
    twin_set = set(TWIN)
    other = [R[a, b] < thr for a in range(KEYS) for b in range(a + 1, KEYS) if (a, b) not in twin_set]
    return float(onp.mean(twin)), float(onp.mean(other))


@uses_14_keys
def run(structure, pins, seed, init_scale=0.01, steps=3000, batch=32, lr=0.02, rec=25, opt="adam"):
    rng = onp.random.default_rng(seed)
    params = rc.init(rng, vocab=VOCAB, w0=W, pins=pins, scale=init_scale)
    m = {k: onp.zeros_like(v) for k, v in params.items()}
    s = {k: onp.zeros_like(v) for k, v in params.items()}
    traj = [params["bias"][:KEYS].copy()]
    for t in range(1, steps + 1):
        X, Y = make_batch(structure, batch, rng)
        g, _, _ = rc.local_grads(params, X, Y)
        g["logw"][:] = 0.0                                   # width is fixed: the squeeze
        if opt == "gd":                                      # plain gradient descent: step size
            for k in params:                                 # follows gradient size (no normalising)
                params[k] = params[k] - lr * g[k]
            params["bias"] = onp.clip(params["bias"], -1, 1)
            if t % rec == 0:
                traj.append(params["bias"][:KEYS].copy())
            continue
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - 0.9 ** t)) / (onp.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
        params["bias"] = onp.clip(params["bias"], -1, 1)     # bounded pin voltages
        if t % rec == 0:
            traj.append(params["bias"][:KEYS].copy())
    return params, onp.array(traj)


@uses_14_keys
def analyse(params, traj, structure, pins, thr=1.0):
    test = {}
    for st in ("twin", "arbitrary"):
        X, Y = make_batch(st, 512, onp.random.default_rng(777))
        test[st] = rc.accuracy(params, X, Y)
    R = key_resistance(params["bias"][:KEYS], pins)
    twin_m, other_m = merge_stats(R, thr)
    counts, merges_after, prev = [], 0, None
    for P in traj:
        comps = components(key_resistance(P, pins), thr)
        counts.append(len(comps))
        if prev is not None:                                  # a key pair split earlier but now rejoined?
            prev_of = {k: i for i, g in enumerate(prev) for k in g}
            for g in comps:
                if len({prev_of[k] for k in g}) > 1:
                    merges_after += 1
        prev = comps
    return {"acc_on_own_structure": test[structure], "acc_on_twin": test["twin"],
            "acc_on_arbitrary": test["arbitrary"],
            "twin_pairs_merged": twin_m, "other_pairs_merged": other_m,
            "final_groups": components(R, thr), "group_count_curve": counts,
            "rejoin_events": merges_after}


if __name__ == "__main__":
    structure, pins, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    init_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    opt = sys.argv[5] if len(sys.argv) > 5 else "adam"
    lr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
    steps = int(sys.argv[7]) if len(sys.argv) > 7 else 3000
    params, traj = run(structure, pins, seed, init_scale, steps=steps, lr=lr, opt=opt)
    res = analyse(params, traj, structure, pins)
    name = f"{structure}_p{pins}_s{seed}" + ("" if init_scale == 0.01 else f"_init{init_scale}") \
        + ("" if opt == "adam" else f"_{opt}")
    out = Path("results/g10")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))     # one file per run: safe in parallel
    onp.save(out / f"traj_{name}.npy", traj)
    print(name, {k: v for k, v in res.items() if k not in ("group_count_curve", "final_groups")},
          "groups:", len(res["final_groups"]), flush=True)
