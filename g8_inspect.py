"""G8 inspection: train the window4 transformer on recall (seed 1, same as the gate run),
save it, and draw what the medium learned.

Panel 1: attention from the query position for a few held-out sequences
         (the tube to the matching pair should carry the current).
Panel 2: series resistance between every query-key token and every pair token,
         grouped by the pair's key. Low resistance should sit on the diagonal blocks.
Panel 3: the learned positional base conductances theta.
"""
import pickle
import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer import train, make_batch, forward, accuracy

if __name__ == "__main__":
    P = train("recall", "window4", steps=2500, seed=1, log=500)
    P = {k: onp.asarray(v) for k, v in P.items()}
    pickle.dump(P, open("results/window4_recall_params.pkl", "wb"))
    Xt, Yt = make_batch("recall", 512, onp.random.default_rng(777))
    print("held-out recall accuracy:", accuracy(P, Xt, Yt, "window4"))

    _, attn = forward(P, Xt[:6], "window4", return_attn=True)
    w = float(onp.exp(P["logw4"][0]))
    b = P["bias4"]
    q_tokens = onp.arange(12)
    pair_tokens = 12 + onp.arange(72)                      # ordered by key, then value
    d2 = ((b[q_tokens][:, None, :] - b[pair_tokens][None, :, :]) ** 2) / w ** 2
    logR = onp.log(onp.exp(d2).sum(2))
    match = onp.array([[logR[k, 6 * k:6 * k + 6].mean() for k in range(12)]])
    other = onp.array([[onp.delete(logR[k], onp.arange(6 * k, 6 * k + 6)).min() for k in range(12)]])
    print("mean log-resistance query->its own pairs:", float(match.mean()),
          " lowest log-resistance query->any other pair:", float(other.mean()))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    rows = onp.array([onp.asarray(a)[-1] for a in attn])
    axs[0].imshow(rows, cmap="magma", vmin=0, vmax=1, aspect="auto")
    for r, x in enumerate(Xt[:6]):
        tgt = [p for p in range(7) if (x[p] - 12) // 6 == x[7]][0]
        axs[0].add_patch(plt.Rectangle((tgt - .5, r - .5), 1, 1, fill=False, ec="cyan", lw=2))
    axs[0].set(title="current from the query port (cyan = matching pair)",
               xlabel="key port (position)", ylabel="held-out sequence")
    im = axs[1].imshow(-logR, aspect="auto", cmap="viridis")
    for k in range(12):
        axs[1].add_patch(plt.Rectangle((6 * k - .5, k - .5), 6, 1, fill=False, ec="w", lw=1))
    axs[1].set(title="tube conductance (log) query-key token -> pair token",
               xlabel="pair token (grouped by key)", ylabel="query key")
    fig.colorbar(im, ax=axs[1], shrink=.8)
    axs[2].imshow(P["theta"], cmap="RdBu_r")
    axs[2].set(title="learned positional base conductance theta", xlabel="key port", ylabel="query port")
    fig.savefig("results/g8_window4.png", dpi=120)
    print("saved results/g8_window4.png")
