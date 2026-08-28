#!/usr/bin/env python3
"""Does the fusion result hold on a different corridor, and over twice the distance?

    python3 experiments/fusion_on_fixed_routes/routes_compare.py

Four routes, the same six arms on each:
  network_traverse  the mixed corridor            (0 to 4 cameras)
  overlap_rich      SAME endpoints, best-covered  (92% of the way with two or more)
  overlap_sparse    SAME endpoints, worst-covered (45%, and 18% with none at all)
  long_traverse     twice the distance, out and back

rich and sparse are the control: same start, same goal, same length to within a metre. If an
arm's advantage survives both of those AND the long run, it is a property of the rule. If it
moves with the corridor, it is a property of the corridor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "deck_figures"))
sys.path.insert(0, str(HERE.parent))
import style as D                                        # noqa: E402
from score import FOLDER, TASKS, story_dir               # noqa: E402
from compare import COLOUR, LABEL                        # noqa: E402

OUT = D.REPO / "logs/studies/fusion_on_fixed_routes/routes_compare"
SHORT = {"fusion_network_traverse": "mixed\n30.6 m",
         "fusion_overlap_rich": "well covered\n31.4 m",
         "fusion_overlap_sparse": "poorly covered\n30.8 m",
         "fusion_long_traverse": "long run\n63.3 m"}


def gather():
    out = {}
    for task in TASKS:
        for arm in FOLDER:
            path = story_dir(task, arm) / "numbers.json"
            if path.exists():
                out[(task, arm)] = json.loads(path.read_text())
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = gather()
    tasks = [t for t in TASKS if any(k[0] == t for k in data)]
    arms = [a for a in FOLDER if any(k[1] == a for k in data)]
    if not tasks:
        raise SystemExit("no scored routes yet")

    metrics = [
        ("honesty", lambda n: n["honesty"]["truth_inside_stated_95pct_ellipse"] * 100,
         "truth inside its own 95% ellipse (%)", "higher is better"),
        ("median", lambda n: n["belief_error_cm"]["median"],
         "median belief error (cm)", "lower is better"),
        ("worse", lambda n: (n.get("fusion", {}).get("worse_than_best_available_camera") or 0) * 100,
         "answer worse than the best camera it had (%)", "lower is better"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 6.6),
                             constrained_layout=True)
    x = np.arange(len(tasks))
    for ax, (_key, get, ylabel, hint) in zip(axes, metrics):
        for arm in arms:
            ys = [get(data[(t, arm)]) if (t, arm) in data else np.nan for t in tasks]
            ax.plot(x, ys, "-o", color=COLOUR[arm], lw=2.2, ms=9, label=arm)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(t, t) for t in tasks], fontsize=11.5)
        ax.set_ylabel(ylabel, fontsize=12.5)
        ax.grid(True, color="#eeede8", lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(hint, loc="left", fontsize=13, color=D.INK2)
    if "honesty" in metrics[0][0]:
        axes[0].axhline(95, color=D.INK, lw=1.4, ls=(0, (5, 3)))
        axes[0].text(0.02, 96.5, "honest is 95%", fontsize=11.5, color=D.INK)
    axes[0].legend(frameon=False, fontsize=11.5, ncol=3, loc="lower left")
    fig.suptitle("Does it hold on another corridor, and over twice the distance?",
                 x=0.004, ha="left", fontsize=20, color=D.INK)
    fig.text(0.004, -0.035,
             "The same six arms on four routes. 'Well covered' and 'poorly covered' share the "
             "mixed route's start, goal and length and differ only in how much of the way has "
             "two or more cameras (92% against 45%), so the pair separates the rule from the "
             "corridor.\n"
             "One drive per arm per route, seed 0. Lines that keep their order across all four "
             "are properties of the rule; lines that cross are properties of the route.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "01_does_it_hold.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"{'route':26s} " + " ".join(f"{a:>6s}" for a in arms))
    for label, get, _y, _h in metrics:
        print(f"-- {label}")
        for t in tasks:
            row = " ".join(f"{get(data[(t, a)]):6.1f}" if (t, a) in data else "     -"
                           for a in arms)
            print(f"   {t:24s} {row}")
    print(f"wrote {OUT.relative_to(D.REPO)}/01_does_it_hold.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
