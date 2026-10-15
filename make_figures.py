"""Summary figure: teacher attention vs best single-graph Physarum vs grown two-port Physarum."""
import json
import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kirchhoff import two_port_attention

g2 = json.load(open("results/g2_layouts.json"))
hist = json.load(open("results/growth_histories.json"))
fig, axs = plt.subplots(3, 4, figsize=(13, 9.5), layout="constrained")
for r, task in enumerate(("mirror", "shift", "dup")):
    A_T = onp.array(g2[task]["example_teacher"])
    single = onp.array(g2[task]["example_fits"]["single"])
    C = onp.array(hist[task]["C"][-1])
    grown = onp.asarray(two_port_attention(C, onp.full(8, 20.0)))
    for c, (M, name) in enumerate(((A_T, "transformer (teacher)"),
                                   (single, "best ONE-node-per-token net"),
                                   (grown, "grown TWO-port net (local rule)"))):
        axs[r, c].imshow(M, vmin=0, vmax=1, cmap="magma")
        axs[r, c].set_title(f"{task}: {name}", fontsize=9)
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    axs[r, 3].semilogy(hist[task]["steps"], hist[task]["kl"], lw=2)
    axs[r, 3].set_title(f"{task}: KL during growth", fontsize=9)
    axs[r, 3].set_xlabel("step")
fig.savefig("results/summary.png", dpi=120)
print("saved results/summary.png")
