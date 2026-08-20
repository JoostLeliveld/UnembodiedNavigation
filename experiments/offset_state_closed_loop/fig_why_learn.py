#!/usr/bin/env python3
"""Step 1: why does anything about the cameras need to be LEARNED?

Both panels read the SAME canonical source -- the balanced set-pose dataset scored
under the current floor-plane IPM path
(``logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json``, 1844
detections, four cardinal yaws, one scoring code). Nothing here comes from the
retired v2 driving captures. See ``docs/localization_metrics.md``.

  left   the four cameras are equally accurate ON AVERAGE (64.6-68.1 mm mean
         error, a 5 % spread) but their systematic error points in DIFFERENT
         DIRECTIONS: sideways bias runs +18.8 mm on camera C to -16.0 mm on
         camera D. One pooled number cannot represent a disagreement in sign.

  right  the obvious fix -- measure each camera once at commissioning and ship the
         constants -- was tried and made things WORSE. On both cameras it was
         fitted for, the correction inverted the very bias it existed to remove.

Together: not settable as one number, and not fixable by a measured constant.

CORRECTION 2026-08-10: an earlier version of this figure plotted the v2-era
per-camera bias BUDGET (7/12/77/33 mm) as "how far off this camera reads" and
claimed an 11x spread between cameras. That conflated a signed component measured
under a retired pipeline on confounded driving routes with total measurement error,
and it is the specific mix-up ``docs/localization_metrics.md`` exists to stop.
Under the current path there is no 11x spread and camera C is not unusually bad.

Outputs -> logs/studies/offset_state_closed_loop/why_learn/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
OUT = REPO / "logs/studies/offset_state_closed_loop/why_learn"
E7 = (REPO / "logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json")

RAW_ARM = "raw IPM (floor, no correction)"
FITTED_ARM = "v4 (contact_z=0.0)"
CAMERAS = ["A", "B", "C", "D"]

INK = "#1A1A1A"
POS = "#0072B2"      # leans one way
NEG = "#D55E00"      # leans the other way
GREY = "#8A8A8A"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    })


def load():
    payload = json.loads(E7.read_text())
    arms = payload["arms"]
    raw, fitted = arms[RAW_ARM]["per_camera"], arms[FITTED_ARM]["per_camera"]
    out = {}
    for c in CAMERAS:
        key = f"camera_{c}"
        out[c] = {
            "mean_mm": 1000.0 * raw[key]["mean_m"],
            "lateral_mm": 1000.0 * raw[key]["lateral_bias_m"],
            "lateral_fixed_mm": 1000.0 * fitted[key]["lateral_bias_m"],
            "n": int(raw[key]["n"]),
        }
    return out, payload


def panel_directions(ax, data) -> None:
    positions = np.arange(len(CAMERAS))
    lateral = [data[c]["lateral_mm"] for c in CAMERAS]
    colors = [POS if v > 0 else NEG for v in lateral]

    ax.bar(positions, lateral, 0.58, color=colors)
    ax.axhline(0.0, color=INK, lw=1.2)
    for x, v in zip(positions, lateral):
        ax.text(x, v + (2.0 if v > 0 else -2.0), f"{v:+.1f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=11,
                fontweight="bold", color=INK)

    means = [data[c]["mean_mm"] for c in CAMERAS]
    ax.set_xticks(positions,
                  [f"camera {c}\n{data[c]['mean_mm']:.1f} mm total" for c in CAMERAS],
                  fontsize=9)
    ax.set_ylabel("sideways error, with its sign (mm)\nzero is correct")
    ax.set_ylim(-27, 27)
    ax.set_title("All four cameras are about equally accurate\n"
                 f"({min(means):.1f}–{max(means):.1f} mm) — but they lean opposite ways",
                 fontweight="bold", fontsize=10.5)
    ax.annotate("", xy=(2, 18.8), xytext=(3, -16.0),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4))
    ax.text(2.5, 3.0, "C and D disagree\nby 35 mm", ha="center", fontsize=9.5,
            color=GREY, style="italic")


def panel_transfer(ax, data) -> None:
    positions = np.arange(len(CAMERAS))
    width = 0.36
    before = [data[c]["lateral_mm"] for c in CAMERAS]
    after = [data[c]["lateral_fixed_mm"] for c in CAMERAS]
    fitted_for = [abs(a - b) > 1e-6 for a, b in zip(before, after)]

    ax.bar(positions - width / 2, before, width, color=GREY,
           label="before: no correction at all")
    ax.bar(positions + width / 2, after, width, color=NEG,
           label="after: the correction we measured and shipped")
    ax.axhline(0.0, color=INK, lw=1.2)

    for x, (b, a, was_fit) in enumerate(zip(before, after, fitted_for)):
        if not was_fit:
            ax.text(x, max(b, a) + 4.0, "left alone", ha="center", va="bottom",
                    fontsize=8.5, color="#555555", style="italic")
            continue
        ax.annotate("", xy=(x + width / 2, a), xytext=(x - width / 2, b),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.6,
                                    connectionstyle="arc3,rad=0.30"))
        for value, xpos, colour in ((b, x - width / 2, "#555555"),
                                    (a, x + width / 2, NEG)):
            ax.text(xpos, value + (4.5 if value > 0 else -4.5), f"{value:+.1f}",
                    ha="center", va="bottom" if value > 0 else "top",
                    fontsize=9.5, fontweight="bold", color=colour)

    ax.set_xticks(positions, [f"camera {c}" for c in CAMERAS])
    ax.set_ylabel("sideways error, with its sign (mm)\nzero is correct")
    ax.set_ylim(-78, 46)
    ax.set_title("So we measured that lean once and shipped the numbers.\n"
                 "On both cameras it was fitted for, it made things WORSE",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    data, payload = load()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    panel_directions(ax, data)
    panel_transfer(ax2, data)

    total = sum(data[c]["n"] for c in CAMERAS)
    fig.suptitle("Why the cameras cannot simply be set up correctly once",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.925,
             f"Both panels: the same {total} detections, balanced set-pose dataset, "
             "four robot headings, current floor-plane projection. Scored against "
             "commanded ground truth, offline.",
             ha="center", va="top", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.872))

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_w1_why_learn.{ext}", bbox_inches="tight")
    plt.close(fig)

    for c in CAMERAS:
        d = data[c]
        print(f"  camera {c}: n={d['n']:4d}  mean {d['mean_mm']:5.1f} mm  "
              f"lateral {d['lateral_mm']:+6.1f} -> {d['lateral_fixed_mm']:+6.1f} mm")
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
