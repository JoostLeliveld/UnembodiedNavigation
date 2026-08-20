#!/usr/bin/env python3
"""Where the iteration levels are, and what is not a textbook Kalman filter.

A reader's map of the notebook's three estimators, drawn from the code itself:

  left    section 3, `kalman_filter` + `rts_smoother`. One sweep over time. This is the
          textbook recursion; the four places it departs are marked.
  middle  section 5, `learn_R`. THREE nested levels: a per-step recursion inside a
          forward-and-backward sweep inside a variational loop of 12 iterations.
  right   section 7, `offset_state_filter`. ONE sweep, no outer loop at all -- the
          offsets are in the state, so they converge along TIME instead.

Every label is taken from `pp4_for_camera_localisation.py`; the two learning schemes
differ in where their iteration lives, which is the thing the diagram exists to show.

Outputs -> logs/studies/filter_notebook/explainer/
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
OUT = REPO / "logs/studies/filter_notebook/explainer"

INK = "#1A1A1A"
STEP = "#0072B2"        # the per-step recursion
SWEEP = "#009E73"       # a pass over the whole drive
OUTER = "#6A3D9A"       # the variational loop
DIFF = "#D55E00"        # departures from the textbook filter
MUTED = "#6B6B6B"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9.5,
        "axes.titlesize": 11.5, "text.color": INK, "legend.frameon": False,
    })


def box(ax, x, y, w, h, text, *, colour=INK, face="white", fontsize=8.6, lw=1.3,
        weight="normal", style="round,pad=0.012", ha="center", zorder=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                                edgecolor=colour, facecolor=face, alpha=alpha,
                                zorder=zorder, mutation_aspect=0.55))
    ax.text(x + (w / 2 if ha == "center" else 0.018), y + h / 2, text, ha=ha,
            va="center", fontsize=fontsize, color=INK, weight=weight, zorder=zorder + 1)


def arrow(ax, start, end, *, colour=INK, lw=1.4, style="-|>", rad=0.0, zorder=4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, lw=lw, color=colour,
                                 shrinkA=1.5, shrinkB=1.5, mutation_scale=11,
                                 connectionstyle=f"arc3,rad={rad}", zorder=zorder))


def frame(ax, title, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 1.045, title, ha="center", va="bottom", fontsize=11.5, weight="bold")
    ax.text(0.5, 1.008, subtitle, ha="center", va="bottom", fontsize=8.8, color=MUTED)


# ------------------------------------------------------------------- the panels

def panel_plain_filter(ax):
    frame(ax, "3.  the filter itself",
          "`kalman_filter` + `rts_smoother` — one sweep, no iteration")

    box(ax, 0.04, 0.30, 0.92, 0.575, "", colour=SWEEP, lw=1.6, face="#F2FBF7")
    ax.text(0.075, 0.845, "for every odometry step  $k = 0 \\ldots T$", fontsize=9,
            color=SWEEP, weight="bold", zorder=5)

    box(ax, 0.10, 0.715, 0.80, 0.10,
        "predict:   $m \\leftarrow m + u_k$,    $P \\leftarrow P + Q_k$", colour=STEP)
    box(ax, 0.10, 0.575, 0.80, 0.10,
        "did a camera fire?   $v = y_k - m$,   $S = P + R_c$", colour=STEP)
    box(ax, 0.10, 0.435, 0.80, 0.10,
        "gate:  reject if  $v^{\\top} S^{-1} v > 5.991$", colour=DIFF)
    box(ax, 0.10, 0.325, 0.80, 0.075,
        "update:  $K = P S^{-1}$,   $m \\leftarrow m + Kv$", colour=STEP)
    for y0, y1 in ((0.715, 0.675), (0.575, 0.535), (0.435, 0.400)):
        arrow(ax, (0.5, y0), (0.5, y1), colour=STEP)

    arrow(ax, (0.5, 0.30), (0.5, 0.245), colour=SWEEP)
    box(ax, 0.04, 0.145, 0.92, 0.10,
        "then one backward pass:  the RTS smoother\n"
        "$G_k = P_k (P^-_{k+1})^{-1}$ — later cameras improve earlier steps",
        colour=SWEEP, face="#F2FBF7", fontsize=8.4)

    ax.text(0.5, 0.075, "iteration levels here:  ONE  (over time)", ha="center",
            fontsize=9.2, color=SWEEP, weight="bold")
    ax.text(0.5, 0.015,
            "$R_c$ is asserted, from commissioning on other runs.\n"
            "Nothing about the model is estimated from this drive.",
            ha="center", va="bottom", fontsize=8.2, color=MUTED)


def panel_learn_r(ax):
    frame(ax, "5.  learning $R$   (variational inference)",
          "`learn_R` — the only place with an outer loop")

    box(ax, 0.02, 0.145, 0.96, 0.775, "", colour=OUTER, lw=2.0, face="#F7F3FB")
    ax.text(0.055, 0.885, "repeat 12 times   (`iterations=12`)", fontsize=9.2,
            color=OUTER, weight="bold", zorder=5)

    box(ax, 0.045, 0.545, 0.845, 0.315, "", colour=SWEEP, lw=1.5, face="#F2FBF7")
    ax.text(0.070, 0.815, "the $x$ step — hold $R$ fixed, re-estimate the path",
            fontsize=8.8, color=SWEEP, weight="bold", zorder=5)
    box(ax, 0.075, 0.685, 0.79, 0.095,
        "forward Kalman pass over $k = 0 \\ldots T$   ← the whole left panel",
        colour=STEP, fontsize=8.3)
    box(ax, 0.075, 0.575, 0.79, 0.085,
        "backward RTS smoothing pass over $k = T \\ldots 0$", colour=STEP, fontsize=8.3)
    arrow(ax, (0.47, 0.685), (0.47, 0.662), colour=STEP)

    arrow(ax, (0.30, 0.545), (0.30, 0.495), colour=OUTER, lw=1.7)
    ax.text(0.315, 0.520, "the smoothed path  $m^s_k,\\; P^s_k$", fontsize=8.0,
            color=OUTER, va="center", zorder=6)

    box(ax, 0.045, 0.245, 0.845, 0.25, "", colour=OUTER, lw=1.5, face="white")
    ax.text(0.070, 0.457, "the $R$ step — hold the path fixed, re-estimate $R$",
            fontsize=8.8, color=OUTER, weight="bold", zorder=5)
    ax.text(0.48, 0.395,
            "$\\Psi_c^{+} = \\Psi + \\sum_{k\\,\\in\\,c}"
            "\\left[(y_k - m^s_k)(y_k - m^s_k)^{\\top} + P^s_k\\right]$",
            ha="center", va="center", fontsize=9.0, zorder=5)
    ax.text(0.48, 0.330,
            "$\\nu_c^{+} = \\nu\\; +\\;$ how many observations camera $c$ gave",
            ha="center", va="center", fontsize=8.8, zorder=5)
    ax.text(0.48, 0.278,
            "closed form, no inner iteration — one shot per camera",
            ha="center", va="center", fontsize=8.2, color=MUTED, zorder=5)

    arrow(ax, (0.890, 0.370), (0.945, 0.370), colour=OUTER, lw=1.7, style="-")
    arrow(ax, (0.945, 0.370), (0.945, 0.733), colour=OUTER, lw=1.7, style="-")
    arrow(ax, (0.945, 0.733), (0.867, 0.733), colour=OUTER, lw=1.7)
    ax.text(0.958, 0.552, "$\\bar{R}_c = \\Psi^{+}_c / \\nu^{+}_c$", rotation=90,
            ha="center", va="center", fontsize=8.6, color=OUTER)

    ax.text(0.5, 0.075, "iteration levels here:  THREE", ha="center", fontsize=9.2,
            color=OUTER, weight="bold")
    ax.text(0.5, 0.012,
            "per-step recursion  ⊂  forward+backward sweep  ⊂  variational loop.\n"
            "$\\bar{R}$ is the inverse of the expected PRECISION, not the expected "
            "covariance. $Q$ is never touched.",
            ha="center", va="bottom", fontsize=8.2, color=MUTED)


def panel_offset(ax):
    frame(ax, "7.  learning the offset",
          "`offset_state_filter` — no outer loop at all")

    box(ax, 0.04, 0.755, 0.92, 0.16,
        "the state grows from 2 numbers to 10\n"
        "$z = [\\,x,\\; b^{(A)},\\; b^{(B)},\\; b^{(C)},\\; b^{(D)}\\,]$",
        colour=DIFF, face="#FDF3EC", fontsize=8.6)

    box(ax, 0.04, 0.275, 0.92, 0.44, "", colour=SWEEP, lw=1.6, face="#F2FBF7")
    ax.text(0.075, 0.685, "for every odometry step  $k = 0 \\ldots T$", fontsize=9,
            color=SWEEP, weight="bold", zorder=5)
    box(ax, 0.10, 0.555, 0.80, 0.095,
        "predict: only the position moves\n(the offsets sit still, or drift slowly)",
        colour=STEP, fontsize=8.2)
    box(ax, 0.10, 0.415, 0.80, 0.105,
        "camera $c$ fired, so\n$H_c$ picks out position PLUS that camera's own offset",
        colour=DIFF, fontsize=8.2)
    box(ax, 0.10, 0.300, 0.80, 0.085,
        "one ordinary Kalman update — all 10 numbers move", colour=STEP, fontsize=8.3)
    arrow(ax, (0.5, 0.555), (0.5, 0.522), colour=STEP)
    arrow(ax, (0.5, 0.415), (0.5, 0.388), colour=STEP)

    ax.text(0.5, 0.215, "iteration levels here:  ONE  (over time)", ha="center",
            fontsize=9.2, color=SWEEP, weight="bold")
    ax.text(0.5, 0.115,
            "The offsets converge along the DRIVE, not along iterations: each one only\n"
            "moves while its own camera can see the robot. Nothing is re-run.",
            ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.text(0.5, 0.012,
            "Catch: adding a constant to every $b^{(c)}$ and subtracting it from $x$\n"
            "changes no prediction, so the COMMON part of the offsets is invisible.",
            ha="center", va="bottom", fontsize=8.2, color=DIFF)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    figure = plt.figure(figsize=(15.0, 7.4))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.28, 1.0], wspace=0.10,
                               left=0.012, right=0.988, top=0.825, bottom=0.075)
    panel_plain_filter(figure.add_subplot(grid[0, 0]))
    panel_learn_r(figure.add_subplot(grid[0, 1]))
    panel_offset(figure.add_subplot(grid[0, 2]))

    figure.suptitle("Three estimators in the notebook, and where each one iterates",
                    fontsize=15, y=0.985)
    figure.text(0.5, 0.932,
                "Orange marks every place the code departs from a textbook Kalman "
                "filter. Green is a sweep over the whole drive; blue is the per-step "
                "recursion; purple is the only genuine outer loop in the notebook.",
                ha="center", fontsize=9, color=MUTED)
    figure.text(0.5, 0.018,
                "Read from pp4_for_camera_localisation.py: `kalman_filter` (§3), "
                "`learn_R` (§5), `offset_state_filter` (§7). Two ways to learn a "
                "parameter — batch and iterative for $R$, recursive and in-state for the "
                "offsets — which is why only one of them has an iteration count.",
                ha="center", fontsize=8.3, color=MUTED)

    for extension in ("png", "pdf"):
        figure.savefig(OUT / f"fig_where_the_loops_are.{extension}",
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  wrote {OUT}/fig_where_the_loops_are.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
