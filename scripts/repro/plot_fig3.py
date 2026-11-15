"""Overlay our InforMARL training curves on the digitized Figure 3 curves."""
import csv
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ours_csv, paper_json, out_png = sys.argv[1:4]
SMOOTH = 5  # moving average over 5 logged points (25 updates), centred

# palette: default categorical slots 1 and 2 (dataviz references/palette.md)
C_OURS, C_PAPER = "#2a78d6", "#eb6834"
INK, INK2, GRID, AXIS = "#1f1f1e", "#5f5e58", "#e1e0d9", "#c3c2b7"

runs = defaultdict(lambda: defaultdict(list))
for name, step, rew in csv.reader(open(ours_csv)):
    n = int(name.split("_n")[1].split("_")[0])
    runs[n][name].append((int(step), float(rew)))
paper = {int(k): np.array(v) for k, v in json.load(open(paper_json)).items()}


def smooth(y, k):
    if k <= 1:
        return y
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
fig.patch.set_facecolor("white")
summary = {}
for ax, n in zip(axes, (3, 7, 10)):
    curves = []
    for name, pts in sorted(runs[n].items()):
        pts = np.array(sorted(pts))
        curves.append(pts)
    L = min(len(c) for c in curves)
    steps = curves[0][:L, 0]
    Y = np.stack([smooth(c[:L, 1], SMOOTH) for c in curves])
    mu, sd = Y.mean(0), Y.std(0)
    pp = paper[n]
    pp = pp[np.argsort(pp[:, 0])]
    ax.fill_between(steps, mu - sd, mu + sd, color=C_OURS, alpha=0.18, linewidth=0)
    ax.plot(steps, mu, color=C_OURS, lw=2, label=f"Ours (mean ± std, {len(curves)} seeds)")
    ax.plot(pp[:, 0], pp[:, 1], color=C_PAPER, lw=2, label="Paper, Figure 3 (digitized mean)")
    ax.set_title(f"{n} agents", color=INK, fontsize=11)
    ax.set_xlabel("Training steps", color=INK2)
    if n == 3:
        ax.set_ylabel("Average episode reward (team)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    # numbers: ours vs paper at fractions of the paper's x-range
    xmax = pp[-1, 0]
    rows = []
    for f in (0.25, 0.5, 0.75, 1.0):
        x = f * xmax
        p = float(np.interp(x, pp[:, 0], pp[:, 1]))
        o = float(np.interp(x, steps, mu))
        osd = float(np.interp(x, steps, sd))
        rows.append({"steps": x, "paper": p, "ours": o, "ours_sd": osd})
    # steps to reach 90% of the paper's final reward (measured from its start)
    target = pp[0, 1] + 0.9 * (pp[-1, 1] - pp[0, 1])
    def first_cross(xs, ys):
        i = np.argmax(ys >= target)
        return float(xs[i]) if ys[i] >= target else None
    summary[n] = {"rows": rows, "target_90pct": float(target),
                  "steps_to_target_paper": first_cross(pp[:, 0], pp[:, 1]),
                  "steps_to_target_ours": first_cross(steps, mu),
                  "ours_final": float(mu[-1]), "ours_final_steps": float(steps[-1])}
    if steps[-1] > xmax * 1.02:
        ax.axvline(xmax, color=AXIS, lw=1, ls=":")
        ax.text(xmax, ax.get_ylim()[0], " paper ends", color=INK2, fontsize=7, va="bottom")
axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
fig.suptitle("InforMARL training curves, Target task: our reproduction vs Nayak et al. (2023), Figure 3",
             color=INK, fontsize=11)
fig.text(0.5, -0.02, f"Ours smoothed with a centred {SMOOTH}-point moving average (logged every 5 updates). "
         "Paper curve digitized from the PDF; its std band (5 seeds) is not reproduced.",
         ha="center", color=INK2, fontsize=8)
fig.tight_layout()
fig.savefig(out_png, bbox_inches="tight", facecolor="white")
json.dump(summary, open(out_png.replace(".png", ".json"), "w"), indent=1)
for n, s in summary.items():
    print(f"n={n}: " + "  ".join(f"{r['steps']/1e6:.2f}M paper {r['paper']:.0f} ours {r['ours']:.0f}±{r['ours_sd']:.0f}"
                                  for r in s["rows"]))
    print(f"      90% of paper's gain ({s['target_90pct']:.0f}): paper at {s['steps_to_target_paper']/1e6:.2f}M, "
          f"ours at {s['steps_to_target_ours']/1e6 if s['steps_to_target_ours'] else float('nan'):.2f}M; "
          f"ours final {s['ours_final']:.0f} at {s['ours_final_steps']/1e6:.2f}M")
