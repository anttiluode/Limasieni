"""Aggregate G10 runs (results/g10/*.json) into results/g10_summary.json and draw the figure."""
import json
from pathlib import Path
import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from g10_squeeze import TWIN, KEYS

runs = {p.stem: json.loads(p.read_text()) for p in sorted(Path("results/g10").glob("*.json"))}


def arm(structure, pins, init="", opt=""):
    keys = [k for k in runs if k.startswith(f"{structure}_p{pins}_s")
            and ("_init" in k) == bool(init) and ("_gd" in k) == bool(opt)
            and (not init or init in k)]
    return [runs[k] for k in keys]


summary = {}
ARMS = {"twin_p1": ("twin", 1, "", ""), "twin_p2": ("twin", 2, "", ""), "twin_p4": ("twin", 4, "", ""),
        "arbitrary_p1": ("arbitrary", 1, "", ""), "arbitrary_p2": ("arbitrary", 2, "", ""),
        "arbitrary_p4": ("arbitrary", 4, "", ""),
        "twin_p1_scattered": ("twin", 1, "_init0.5", ""), "twin_p4_scattered": ("twin", 4, "_init0.5", ""),
        "twin_p1_gd": ("twin", 1, "", "gd"), "twin_p4_gd": ("twin", 4, "", "gd"),
        "arbitrary_p1_gd": ("arbitrary", 1, "", "gd"), "arbitrary_p4_gd": ("arbitrary", 4, "", "gd")}
for name, (st, P, init, opt) in ARMS.items():
    rs = arm(st, P, init, opt)
    if not rs:
        continue
    f = lambda key: [r[key] for r in rs]
    summary[name] = {
        "seeds": len(rs),
        "acc_own": f("acc_on_own_structure"), "acc_on_arbitrary": f("acc_on_arbitrary"),
        "twin_pairs_merged": f("twin_pairs_merged"), "other_pairs_merged": f("other_pairs_merged"),
        "twin_merges_total": int(round(sum(r["twin_pairs_merged"] * 7 for r in rs))),
        "other_merges_total": int(round(sum(r["other_pairs_merged"] * 84 for r in rs))),
        "final_group_counts": [len(r["final_groups"]) for r in rs],
        "rejoin_events": f("rejoin_events"),
    }
Path("results/g10_summary.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(f"{k:20s} acc_own {onp.mean(v['acc_own']):.3f} {[round(a, 3) for a in v['acc_own']]}  "
          f"on_arbitrary {onp.mean(v['acc_on_arbitrary']):.3f}  twin merges {v['twin_merges_total']}/{7 * v['seeds']}  "
          f"other merges {v['other_merges_total']}/{84 * v['seeds']}  groups {v['final_group_counts']}  "
          f"rejoins {v['rejoin_events']}")

fig, axs = plt.subplots(2, 3, figsize=(18, 9.5), layout="constrained")
ax = axs[0, 0]
for st, c in (("twin", "tab:green"), ("arbitrary", "tab:red")):
    ms = [onp.mean(summary[f"{st}_p{P}"]["acc_own"]) for P in (1, 2, 4)]
    for i, P in enumerate((1, 2, 4)):
        ax.scatter([i] * 4, summary[f"{st}_p{P}"]["acc_own"], color=c, alpha=.45, s=16)
    ax.plot(range(3), ms, "-o", color=c, label=f"{st} keys (Adam)")
    gd = [onp.mean(summary[f"{st}_p{P}_gd"]["acc_own"]) for P in (1, 4)]
    ax.plot([0, 2], gd, ":s", color=c, ms=7, label=f"{st} keys (plain descent)")
ax.plot(range(3), [onp.mean(summary[f"twin_p{P}"]["acc_on_arbitrary"]) for P in (1, 2, 4)], "--o",
        color="tab:green", alpha=.5, label="twin-trained, tested on arbitrary")
ax.set_xticks(range(3), ["1 pin", "2 pins", "4 pins"])
ax.set(title="held-out recall vs pins (window fixed at 0.25)", ylim=(0.5, 1.02), ylabel="accuracy")
ax.legend(fontsize=8, loc="lower right")

ax = axs[0, 1]
names = [("twin_p1", "twin\n1 pin\nAdam"), ("twin_p1_gd", "twin\n1 pin\nplain"),
         ("twin_p1_scattered", "twin\n1 pin\nscattered"), ("arbitrary_p1_gd", "arbitrary\n1 pin\nplain"),
         ("twin_p4", "twin\n4 pins\nAdam"), ("twin_p4_gd", "twin\n4 pins\nplain")]
for i, (n, label) in enumerate(names):
    v = summary[n]
    ax.bar(i - .18, onp.mean(v["twin_pairs_merged"]), .36, color="tab:purple", label="twin pairs" if i == 0 else None)
    ax.bar(i + .18, onp.mean(v["other_pairs_merged"]), .36, color="0.6", label="all other pairs" if i == 0 else None)
    ax.scatter([i - .18] * v["seeds"], v["twin_pairs_merged"], color="k", s=9, zorder=3)
    ax.scatter([i + .18] * v["seeds"], v["other_pairs_merged"], color="k", s=9, zorder=3)
ax.set_xticks(range(len(names)), [l for _, l in names], fontsize=8)
ax.set(title="which keys share a position (log-resistance < 1)", ylabel="fraction of pairs merged")
ax.legend(fontsize=8)

ax = axs[0, 2]
for name, c, ls in (("twin_p1_s{}_gd", "tab:green", "-"), ("arbitrary_p1_s{}_gd", "tab:red", "-"),
                    ("twin_p4_s{}_gd", "tab:green", ":"), ("arbitrary_p4_s{}_gd", "tab:red", ":")):
    for sd in (1, 2, 3, 4):
        curve = runs[name.format(sd)]["group_count_curve"][:41]
        ax.plot(onp.arange(len(curve)) * 25, curve, ls, color=c, alpha=.6,
                label=(name.split("_s")[0].replace("_p", ", ").replace("1", "1 pin").replace("4", "4 pins")
                       if sd == 1 else None))
ax.set(title="number of separated key groups, plain descent (4 seeds each)", xlabel="training step",
       ylabel="key groups")
ax.legend(fontsize=8)

cmap = plt.get_cmap("tab10")
for ax, name, title in ((axs[1, 0], "twin_p1_s3_gd", "1-pin key positions, twin keys (plain descent, seed 3)"),
                        (axs[1, 1], "arbitrary_p1_s1_gd", "same, arbitrary keys (seed 1)"),
                        (axs[1, 2], "twin_p1_s1", "twin keys with Adam (seed 1)")):
    tr = onp.load(f"results/g10/traj_{name}.npy")[:, :, 0]
    steps = onp.arange(len(tr)) * 25
    for a, (k1, k2) in enumerate(TWIN):
        ax.plot(steps, tr[:, k1], color=cmap(a), lw=1.5)
        ax.plot(steps, tr[:, k2], color=cmap(a), lw=1.5, ls="--")
    ax.set(title=title, xlabel="training step", ylabel="pin voltage", ylim=(-1.05, 1.05))
    ax.set_xscale("symlog", linthresh=200)
axs[1, 0].text(0.02, 0.02, "colour = twin pair (solid/dashed = the two twins)", transform=axs[1, 0].transAxes, fontsize=8)
axs[1, 1].text(0.02, 0.02, "same colouring; here twins carry no meaning", transform=axs[1, 1].transAxes, fontsize=8)
fig.savefig("results/g10_squeeze.png", dpi=110)
print("saved results/g10_squeeze.png")
