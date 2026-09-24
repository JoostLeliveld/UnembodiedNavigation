#!/usr/bin/env python3
"""Every campaign run at a glance, and what removing the task camera cost each model.

(a) One square per run: rows are tasks, column groups are models, and within a group the
    three seeds with all cameras (left) and under dropout of the task camera (right). Filled in
    the model colour: success. Failures are empty cells with a glyph: x the footprint left
    the driveable region, o the run failed without leaving it.
(b) Removal minus intact, matched on (task, seed): success rate, mean belief error and mean
    belief one-sigma, with 95 % bootstrap intervals from logs/thesis/analysis/summary.json.

    python3 figures/make_campaign_outcomes.py
"""
from __future__ import annotations

import csv
import json

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import paper as P

SEEDS = ("91500", "91501", "91502")


def cell_style(r):
    """Success is filled in the model colour; a failure is an empty cell with its glyph."""
    if r["success"] == "1":
        return P.MODEL_COLOUR[r["model"]], None
    if r["collision"] == "1":
        return "white", ("x", P.COLLISION)
    return "white", ("o", P.STUCK)


def main():
    with (P.THESIS / "analysis/runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((P.THESIS / "analysis/summary.json").read_text())
    tasks = [t["name"] for t in P.tasks()]
    fig = P.plt.figure(figsize=(P.TEXT, 2.55))
    ga = fig.add_axes([0.0, 0.08, 0.56, 0.80])
    s, gap_seed, gap_state, gap_model = 1.0, 0.18, 0.55, 1.1
    x0 = {}
    x = 0.0
    for model in P.MODELS:
        for state in ("intact", "removal"):
            x0[(model, state)] = x
            x += 3 * s + 2 * gap_seed + (gap_state if state == "intact" else gap_model)
    width = x - gap_model
    for i, task in enumerate(tasks):
        y = (len(tasks) - 1 - i) * (s + 0.45)
        for r in rows:
            if r["task"] != task:
                continue
            fc, glyph = cell_style(r)
            xi = x0[(r["model"], r["state"])] + SEEDS.index(r["seed"]) * (s + gap_seed)
            ga.add_patch(Rectangle((xi, y), s, s, fc=fc, ec=P.MODEL_COLOUR[r["model"]] if glyph else "white",
                                   lw=0.7))
            if glyph:
                ga.plot(xi + s / 2, y + s / 2, glyph[0], color=glyph[1], mfc="none", ms=5, mew=1.3)
        ga.text(-0.35, y + s / 2, P.TASK_LABEL[task], ha="right", va="center", fontsize=7)
        removed = next(t["removed"] for t in P.tasks() if t["name"] == task)
        ga.text(width + 0.25, y + s / 2, removed[-1], ha="left", va="center", fontsize=7, color=P.MUTED)
    top = len(tasks) * (s + 0.45)
    ga.text(width + 0.25, top - 0.2, "dropped", ha="left", va="bottom", fontsize=6.5, color=P.MUTED)
    for model in P.MODELS:
        xa, xb = x0[(model, "intact")], x0[(model, "removal")] + 3 * s + 2 * gap_seed
        ga.text((xa + xb) / 2, top + 1.05, P.MODEL_LABEL[model], ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=P.MODEL_COLOUR[model])
        for state in ("intact", "removal"):
            xs = x0[(model, state)]
            n = sum(r["success"] == "1" for r in rows if r["model"] == model and r["state"] == state)
            ga.text(xs + 1.5 * s + gap_seed, top - 0.2, f"{'all' if state == 'intact' else 'dropout'}\n{n}/15",
                    ha="center", va="bottom", fontsize=6.5, linespacing=1.0)
    ga.set_xlim(-0.2, width + 1.4); ga.set_ylim(-0.2, top + 1.75)
    ga.set_aspect("equal"); ga.axis("off")
    handles = [Line2D([], [], marker="s", ls="none", ms=6, mfc=P.MODEL_COLOUR["spatial"], mec="none"),
               Line2D([], [], marker="x", ls="none", ms=5, color=P.COLLISION, mew=1.3),
               Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec=P.STUCK, mew=1.3)]
    ga.legend(handles, ["success (model colour)", "failed, left the driveable region", "failed, stayed inside"],
              loc="upper center", bbox_to_anchor=(0.52, 0.0), ncol=4, fontsize=6.3, handlelength=0.9,
              columnspacing=0.8, handletextpad=0.35)
    ga.text(-0.02, 0.97, "(a)", transform=ga.transAxes, fontweight="bold")

    # (b) matched differences
    metrics = (("success", "success rate", 100.0, "pp"),
               ("belief_error_m", "belief error", 100.0, "cm"),
               ("belief_sigma_major_m", "belief one-sigma", 100.0, "cm"))
    left = 0.64
    w = (0.995 - left - 0.02 * (len(metrics) - 1)) / len(metrics)
    for k, (key, label, scale, unit) in enumerate(metrics):
        ax = fig.add_axes([left + k * (w + 0.02), 0.20, w, 0.62])
        for j, model in enumerate(P.MODELS):
            d = summary["removal_minus_intact"][model][key]
            m, (lo, hi) = d["mean"] * scale, (v * scale for v in d["ci95"])
            ax.plot([lo, hi], [2 - j, 2 - j], color=P.MODEL_COLOUR[model], lw=1.6, solid_capstyle="round")
            ax.plot(m, 2 - j, "o", ms=4.5, color=P.MODEL_COLOUR[model], mec="white", mew=0.5, zorder=3)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.set_ylim(-0.6, 2.6); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(label, pad=3)
        ax.set_xlabel(f"dropout $-$ all ({unit})", fontsize=6.8, labelpad=1)
        ax.tick_params(axis="x", labelsize=6.5)
        if k == 0:
            for j, model in enumerate(P.MODELS):
                ax.text(-0.06, (2 - j + 0.6) / 3.2, P.MODEL_LABEL[model], transform=ax.transAxes, ha="right",
                        va="center", fontsize=7, color=P.MODEL_COLOUR[model], fontweight="bold")
            ax.text(-0.62, 1.13, "(b)", transform=ax.transAxes, fontweight="bold")
    P.save(fig, "campaign_outcomes")


if __name__ == "__main__":
    main()
