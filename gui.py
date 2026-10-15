"""Limasieni GUI: watch a slime-mold network grow a transformer's attention matrix.

    python gui.py

Top row of dots   : query ports (food sources), one per token.
Bottom row of dots: key ports, each draining to ground through a sink.
Lines             : tubes. Thickness = conductance. Colour = flow when food is
                    injected at the selected query (click a top dot to choose).

Grow       : run the local free/nudged rule live from the current network.
Reset mesh : back to a uniform mesh (every tube equal).
New input  : another held-out sequence (teacher attention changes).
Task       : mirror / shift / dup teacher transformer.

Requires results/teacher_*.pkl (run g1_train_teacher.py first).
"""
import pickle
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.collections import LineCollection

from physarum import local_update, potentials, tube_drops
from kirchhoff import two_port_attention

N, G = 8, 20.0
teachers = {t: pickle.load(open(f"results/teacher_{t}.pkl", "rb")) for t in ("mirror", "shift", "dup")}
state = {"task": "mirror", "idx": 0, "C": onp.ones((N, N)), "step": 0, "kl": [],
         "running": False, "query": 0}


def A_T():
    return teachers[state["task"]]["attn"][state["idx"]]


def tokens():
    return teachers[state["task"]]["X"][state["idx"]]


def kl(A, B):
    return float(onp.mean(onp.sum(A * onp.log((A + 1e-12) / (B + 1e-12)), axis=1)))


fig = plt.figure(figsize=(14, 7.5))
fig.canvas.manager.set_window_title("Limasieni - a slime mold grows a transformer's attention")
ax_net = fig.add_axes([0.03, 0.12, 0.42, 0.78])
ax_T = fig.add_axes([0.50, 0.55, 0.20, 0.35])
ax_P = fig.add_axes([0.50, 0.12, 0.20, 0.35])
ax_kl = fig.add_axes([0.76, 0.55, 0.22, 0.35])
ax_txt = fig.add_axes([0.76, 0.12, 0.22, 0.35]); ax_txt.axis("off")

qx, kx = onp.arange(N), onp.arange(N)
segs = [[(qx[i], 1.0), (kx[j], 0.0)] for i in range(N) for j in range(N)]
lc = LineCollection(segs, cmap="viridis")
ax_net.add_collection(lc)
qdots = ax_net.scatter(qx, onp.ones(N), s=260, c="tab:orange", zorder=3, picker=True)
ax_net.scatter(kx, onp.zeros(N), s=260, c="tab:blue", zorder=3)
for j in range(N):
    ax_net.plot([kx[j], kx[j]], [0, -0.25], color="0.5", lw=1)
    ax_net.plot([kx[j] - 0.15, kx[j] + 0.15], [-0.25, -0.25], color="0.5", lw=2)
qlabels = [ax_net.text(qx[i], 1.12, "", ha="center", fontsize=11) for i in range(N)]
klabels = [ax_net.text(kx[j], -0.42, "", ha="center", fontsize=11) for j in range(N)]
ax_net.set_xlim(-0.7, N - 0.3); ax_net.set_ylim(-0.6, 1.3); ax_net.axis("off")
title_net = ax_net.set_title("")
imT = ax_T.imshow(onp.zeros((N, N)), vmin=0, vmax=1, cmap="magma")
ax_T.set_title("Transformer attention (teacher)"); ax_T.set_ylabel("query"); ax_T.set_xticks([])
imP = ax_P.imshow(onp.zeros((N, N)), vmin=0, vmax=1, cmap="magma")
ax_P.set_title("Physarum current split"); ax_P.set_xlabel("key"); ax_P.set_ylabel("query")
kl_line, = ax_kl.plot([], [], lw=2)
ax_kl.set_yscale("log"); ax_kl.set_title("KL(teacher || slime)"); ax_kl.set_xlabel("growth step")
info = ax_txt.text(0, 1, "", va="top", family="monospace", fontsize=10)


def redraw():
    C, AT = state["C"], A_T()
    A = onp.asarray(two_port_attention(C, onp.full(N, G)))
    q = state["query"]
    drops = tube_drops(potentials(C, onp.full(N, G)), N)[q]          # food at query q
    flow = onp.abs(C * drops).ravel()
    w = onp.log10(C.ravel() / C.max() + 1e-6)
    lc.set_linewidths(0.3 + 7.0 * onp.clip((w + 3) / 3, 0, 1))
    lc.set_array(flow / (flow.max() + 1e-12))
    sizes = onp.full(N, 260); sizes[q] = 520
    qdots.set_sizes(sizes)
    tk = tokens()
    for i in range(N):
        qlabels[i].set_text(f"q{i}\n{tk[i]}")
        klabels[i].set_text(f"k{i}:{tk[i]}")
    imT.set_data(AT); imP.set_data(A)
    if state["kl"]:
        s, v = zip(*state["kl"])
        kl_line.set_data(s, v); ax_kl.relim(); ax_kl.autoscale_view()
    pruned = onp.mean(C < 0.01 * C.max())
    title_net.set_text(f"task: {state['task']}   input #{state['idx']}   step {state['step']}"
                       f"   (food injected at q{q}; click a top dot)")
    info.set_text(f"KL now      : {kl(AT, A):.4f}\n"
                  f"tubes < 1%  : {pruned:.0%}\n"
                  f"row sums    : {A.sum(1).min():.4f}-{A.sum(1).max():.4f}\n\n"
                  "Rule (each tube, locally):\n"
                  " dlogC = lr*[dv_free^2 - dv_nudged^2]/2b\n"
                  " free  : food at one query\n"
                  " nudged: same + weak pull of\n"
                  "         key sinks toward the\n"
                  "         teacher's uptake")
    fig.canvas.draw_idle()


def tick(_=None):
    if not state["running"]:
        return
    for _ in range(10):
        C = state["C"]
        logC = onp.clip(onp.log(C) + 20.0 * local_update(C, onp.full(N, G), A_T()), -12, 12)
        state["C"] = onp.exp(logC)
        state["step"] += 1
    A = onp.asarray(two_port_attention(state["C"], onp.full(N, G)))
    state["kl"].append((state["step"], kl(A_T(), A)))
    redraw()
    if state["step"] >= 1500:
        state["running"] = False


timer = fig.canvas.new_timer(interval=40)
timer.add_callback(tick)
timer.start()


def on_grow(_):
    state["running"] = not state["running"]
    b_grow.label.set_text("Pause" if state["running"] else "Grow")


def on_reset(_):
    state.update(C=onp.ones((N, N)), step=0, kl=[], running=False)
    b_grow.label.set_text("Grow"); redraw()


def on_new(_):
    state["idx"] = (state["idx"] + 1) % len(teachers[state["task"]]["X"])
    on_reset(None)


def on_task(label):
    state["task"] = label
    on_reset(None)


def on_pick(event):
    if event.artist is qdots and len(event.ind):
        state["query"] = int(event.ind[0]); redraw()


b_grow = Button(fig.add_axes([0.03, 0.02, 0.09, 0.05]), "Grow"); b_grow.on_clicked(on_grow)
b_reset = Button(fig.add_axes([0.13, 0.02, 0.09, 0.05]), "Reset mesh"); b_reset.on_clicked(on_reset)
b_new = Button(fig.add_axes([0.23, 0.02, 0.09, 0.05]), "New input"); b_new.on_clicked(on_new)
radio = RadioButtons(fig.add_axes([0.34, 0.005, 0.10, 0.10]), ("mirror", "shift", "dup"))
radio.on_clicked(on_task)
fig.canvas.mpl_connect("pick_event", on_pick)
redraw()
plt.show()
