"""Figure for G9: accuracy of each arm, the learned tube map, and value-bus currents."""
import json
import pickle
import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recall_circuit import tubes, bus_currents
from transformer import make_batch

r = json.load(open("results/g9_local_recall.json"))
P = pickle.load(open("results/g9_local_params_seed3.pkl", "rb"))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")

names = ["local_seed1", "local_seed2", "local_seed3", "exact_seed1", "static_seed1", "static_seed2"]
vals = [r[k]["final_heldout_acc"] for k in names]
cols = ["tab:green"] * 3 + ["tab:gray"] + ["tab:red"] * 2
axs[0].bar(range(len(names)), vals, color=cols)
axs[0].axhline(1 / 6, ls="--", c="k", lw=1); axs[0].text(5.4, 1 / 6 + .02, "chance", ha="right")
axs[0].set_xticks(range(len(names)), [n.replace("_seed", "\nseed ") for n in names])
axs[0].set(ylim=(0, 1.05), title="held-out recall (512 sequences)")

b, w = P["bias"], float(onp.exp(P["logw"][0]))
q, pairs = b[:12], b[12:]
logR = onp.log(onp.exp(((q[:, None, :] - pairs[None, :, :]) ** 2) / w ** 2).sum(2))
im = axs[1].imshow(-logR, aspect="auto", cmap="viridis")
for k in range(12):
    axs[1].add_patch(plt.Rectangle((6 * k - .5, k - .5), 6, 1, fill=False, ec="w", lw=1))
axs[1].set(title="learned tube conductance (log), local rule, seed 3",
           xlabel="pair token (grouped by key)", ylabel="query key")
fig.colorbar(im, ax=axs[1], shrink=.8)

X, Y = make_batch("recall", 4, onp.random.default_rng(2026))
for r_, (x, y) in enumerate(zip(X, Y[:, -1])):
    logC, _ = tubes(P, x)
    I = bus_currents(onp.exp(logC), x)
    axs[2].bar(onp.arange(7) + r_ * 8, I, color=["tab:green" if c == y else "0.6" for c in range(6)] + ["tab:red"])
axs[2].set_xticks([3 + 8 * i for i in range(4)], [f"seq {i}" for i in range(4)])
axs[2].set(title="bus currents (green = correct value, red = null)",
           ylabel="fraction of injected current")
fig.savefig("results/g9_local_recall.png", dpi=120)
print("saved results/g9_local_recall.png")
