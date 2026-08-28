"""One figure: what each camera's reading is worth, and where its error comes from.

    python3 experiments/learned_measurement_covariance/fig_per_camera_error.py

Reads the drives directly (nothing cached), writes
``logs/studies/learned_measurement_covariance/01_what_a_camera_reading_is_worth.png``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/learned_measurement_covariance"))
sys.path.insert(0, str(REPO / "experiments/deck_figures"))
sys.path.insert(0, str(REPO / "src/unav_common"))
import per_camera_error as P                                   # noqa: E402
import style as D                                              # noqa: E402

OUT = REPO / "logs/studies/learned_measurement_covariance"
CAMS = "ABCDE"


def implied_px(rows):
    """The detector noise a set of misses implies, in pixels."""

    return P.SIGMA_PX * math.sqrt(np.median(P._nees(rows)) / P.CHI2_2_MEDIAN)


def main():
    rows = P._height_ratios([r for c, a in P.FIT + P.HELD_OUT for r in P.readings(c, a)])
    n_drives = len(P.FIT) + len(P.HELD_OUT)

    fig = plt.figure(figsize=(17.0, 9.6))
    grid = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.32], hspace=0.42, wspace=0.55,
                            left=0.05, right=0.985, top=0.845, bottom=0.075)

    # ---- top row: one panel per camera, its misses in its own frame -------------------
    lim = 9.0
    for i, cam in enumerate(CAMS):
        ax = fig.add_subplot(grid[0, i])
        sub = [r for r in rows if r["camera"] == cam]
        along = np.array([r["along"] for r in sub]) * 100
        across = np.array([r["across"] for r in sub]) * 100
        colour = D.CAM_COLOUR[cam]
        ax.axhline(0, color="#d5d4cf", lw=0.9, zorder=1)
        ax.axvline(0, color="#d5d4cf", lw=0.9, zorder=1)
        ax.scatter(np.clip(along, -lim, lim), np.clip(across, -lim, lim), s=2.0,
                   color=colour, alpha=0.11, linewidths=0, zorder=2)
        claim_along = np.median([math.sqrt(r["ray"] @ r["cov"] @ r["ray"]) for r in sub]) * 100
        claim_across = np.median([
            math.sqrt(np.array([-r["ray"][1], r["ray"][0]]) @ r["cov"]
                      @ np.array([-r["ray"][1], r["ray"][0]])) for r in sub]) * 100
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(claim_along * np.cos(theta), claim_across * np.sin(theta),
                color=D.INK, lw=1.8, ls="--", zorder=4)
        ax.plot([along.mean()], [across.mean()], marker="o", ms=10, color=colour,
                mec="white", mew=1.8, zorder=5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"camera {cam}\n{len(sub):,} readings, median "
                     f"{np.median([r['range_m'] for r in sub]):.0f} m\n"
                     f"leans {along.mean():+.1f} cm outward",
                     fontsize=9.5, color=D.INK, pad=5)
        ax.tick_params(labelsize=8.5)
        if i == 0:
            ax.set_ylabel("cm across the\nline of sight", fontsize=9.5, color=D.INK2)
            ax.set_xlabel("cm along the line of sight\n(right = read as too far away)",
                          fontsize=9.5, color=D.INK2)

    legend = fig.add_subplot(grid[0, 5])
    legend.axis("off")
    legend.plot([0.04, 0.24], [0.80, 0.80], color=D.INK, lw=1.8, ls="--")
    legend.text(0.30, 0.80, "what the reading claims\nit is worth (1$\\sigma$)", fontsize=10,
                color=D.INK, va="center")
    legend.plot([0.14], [0.55], marker="o", ms=10, color=D.MUTED, mec="white", mew=1.8)
    legend.text(0.30, 0.55, "its average miss\n— the lean", fontsize=10, color=D.INK,
                va="center")
    legend.scatter([0.09, 0.14, 0.19], [0.32, 0.29, 0.34], s=5, color=D.MUTED, alpha=0.55)
    legend.text(0.30, 0.31, "one reading", fontsize=10, color=D.INK, va="center")
    legend.text(0.04, 0.10, "Each cloud is drawn in that camera's\nown frame, so the five are "
                            "comparable\ndespite pointing different ways.",
                fontsize=9.5, color=D.INK2, va="center")
    legend.set_xlim(0, 1); legend.set_ylim(0, 1)

    # ---- bottom left: where the spread comes from --------------------------------------
    ax = fig.add_subplot(grid[1, 0:2])
    bins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 26)]
    centres, implied, errors, claims = [], [], [], []
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["range_m"] < hi]
        if len(sub) < 30:
            continue
        centres.append(0.5 * (lo + hi))
        implied.append(implied_px(sub))
        errors.append(np.median([np.linalg.norm(r["error"]) for r in sub]) * 100)
        claims.append(np.median([math.sqrt(np.trace(r["cov"]) / 2) for r in sub]) * 100)
    ax.axvspan(16, 26, color=D.BAD, alpha=0.08, zorder=0)
    ax.axhline(P.SIGMA_PX, color=D.GOOD, lw=2.4, zorder=3)
    ax.annotate(f"one commissioned number,\n{P.SIGMA_PX:.2f} px, used by every camera",
                xy=(7.6, P.SIGMA_PX), xytext=(1.2, 1.20), fontsize=9.5, color=D.GOOD,
                va="top", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=D.GOOD, lw=1.6, shrinkA=2,
                                shrinkB=2))
    ax.plot(centres, implied, marker="o", ms=8, color=D.INK, lw=2.4, zorder=4,
            mec="white", mew=1.5)
    for cam in CAMS:
        sub = [r for r in rows if r["camera"] == cam]
        rng = float(np.median([r["range_m"] for r in sub]))
        ax.plot([rng], [implied_px(sub)], marker="D", ms=10, color=D.CAM_COLOUR[cam],
                mec="white", mew=1.5, zorder=5)
        ax.annotate(cam, xy=(rng, implied_px(sub)), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=11,
                    color=D.CAM_COLOUR[cam], fontweight="bold")
    ax.text(21, 1.24, "beyond 16 m", fontsize=10.5, color=D.BAD, ha="center")
    ax.set_ylim(0, 1.38); ax.set_xlim(0, 26)
    ax.set_xlabel("how far the robot was from the camera (m)", fontsize=10.5)
    ax.set_ylabel("detector noise the misses imply (pixels)", fontsize=10.5)
    ax.set_title("The spread is ONE pixel noise, magnified by each\ncamera's geometry — "
                 "nothing is per-camera", fontsize=12.5, color=D.INK, loc="left", pad=8)
    for centre, value, err in zip(centres, implied, errors):
        above = value > 0.85
        ax.annotate(f"{err:.1f} cm", xy=(centre, value),
                    xytext=(14 if above else 0, 12 if above else -19),
                    textcoords="offset points", ha="left" if above else "center",
                    fontsize=9, color=D.INK2)
    ax.text(0.5, 0.13, "the label under each point is the miss\nin centimetres at that range",
            fontsize=9.5, color=D.INK2, va="bottom")

    # ---- bottom middle: where the lean comes from --------------------------------------
    ax = fig.add_subplot(grid[1, 2:4])
    bands = [(0.85, 0.92), (0.92, 0.96), (0.96, 0.99), (0.99, 1.02), (1.02, 3.0)]
    labels = ["85–92%\nbadly cut", "92–96%\ncut", "96–99%\nslightly cut",
              "99–102%\nall there", "over 102%\ntoo tall"]
    values, counts = [], []
    for lo, hi in bands:
        sub = [r for r in rows if lo <= r["height_ratio"] < hi]
        values.append(float(np.mean([r["along"] for r in sub]) * 100) if sub else 0.0)
        counts.append(len(sub))
    colours = [D.BAD, D.BAD, "#f0a882", D.GOOD, "#9fbfd8"]
    bars = ax.bar(range(len(values)), values, color=colours, width=0.66)
    for j, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, values[j] + 0.4,
                f"{values[j]:+.1f} cm\n{counts[j]:,} readings", ha="center", fontsize=9,
                color=D.INK)
    ax.axhline(0, color=D.INK, lw=1.2)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(-1.5, 17.5)
    ax.set_ylabel("how far outward the reading leans (cm)", fontsize=10.5)
    ax.set_xlabel("how much of the robot the detector's box covered", fontsize=10.5)
    ax.set_title("The lean is boxes with the bottom cut off\nby an obstruction — and only "
                 "those", fontsize=12.5, color=D.INK, loc="left", pad=8)

    # ---- bottom right: what a tighter rule costs ---------------------------------------
    ax = fig.add_subplot(grid[1, 4:])
    rules = [("keep\neverything", lambda r: True),
             ("drop boxes\nunder 96%", lambda r: r["height_ratio"] >= 0.96),
             ("drop boxes\nunder 99%", lambda r: r["height_ratio"] >= 0.99),
             ("drop sightings\nbeyond 16 m", lambda r: r["range_m"] < 16.0)]
    kept, worst = [], []
    for _, keep in rules:
        sub = [r for r in rows if keep(r)]
        kept.append(100.0 * len(sub) / len(rows))
        worst.append(max(abs(float(np.mean([r["along"] for r in sub if r["camera"] == c]) * 100))
                         for c in CAMS if any(r["camera"] == c for r in sub)))
    x = np.arange(len(rules))
    ax.bar(x - 0.19, kept, width=0.36, color=D.GOOD, label="readings kept (%)")
    ax.bar(x + 0.19, worst, width=0.36, color=D.BAD, label="worst camera's lean (cm)")
    for j in range(len(rules)):
        ax.text(x[j] - 0.19, kept[j] + 2.0, f"{kept[j]:.0f}%", ha="center", fontsize=9.5,
                color=D.INK)
        ax.text(x[j] + 0.19, worst[j] + 2.0, f"{worst[j]:.1f} cm", ha="center", fontsize=9.5,
                color=D.INK)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rules], fontsize=9.5)
    ax.set_ylim(0, 122)
    ax.legend(fontsize=9.5, frameon=False, ncol=1, loc="upper right")
    ax.set_ylabel("percent   /   centimetres", fontsize=10.5)
    ax.set_title("A range limit removes the lean for 14% of\nthe sightings; a height rule "
                 "costs far more", fontsize=12.5, color=D.INK, loc="left", pad=8)

    fig.suptitle("What one camera's reading is worth: the spread is geometry, the lean is "
                 "obstruction", fontsize=19.5, color=D.INK, x=0.05, ha="left", y=0.965)
    fig.text(0.05, 0.905,
             f"{len(rows):,} admitted readings from {n_drives} closed-loop drives in "
             f"warehouse_v2, each scored against the true pose at the instant its own camera "
             f"saw the robot.  Every camera states the same commissioned {P.SIGMA_PX:.2f} px "
             f"of detector noise pushed through its own imaging geometry — nothing is fitted "
             f"per camera.",
             fontsize=11.5, color=D.INK2, ha="left", va="top")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "01_what_a_camera_reading_is_worth.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
