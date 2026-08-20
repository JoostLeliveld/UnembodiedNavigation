#!/usr/bin/env python3
"""One comparison sheet across the current world and the five candidates.

Small multiples rather than one chart with several y-scales: each measure has
its own units, and putting them on a shared axis would be a lie about scale.
One hue for the candidates, a hatched neutral for the current world so the
baseline never reads as a sixth option.
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wh_v2_mpl")

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

import coverage as cov                               # noqa: E402
from baseline import load_current                    # noqa: E402
from layouts import load_all                         # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

SERIES = "#2a78d6"
BASE = "#8d8b86"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd6"


def collect():
    rows = []
    base = load_current()
    rb = cov.analyse(base, derive_lanes=True)
    rows.append(("current", base, rb, True))
    for L in load_all():
        rows.append((L.key, L, cov.analyse(L), False))
    return rows


def mean_crossing_angle(res):
    """Cell-weighted mean of the pairwise crossing angle over every overlapped
    cell. Higher = the overlap that exists is nearer head-on, which is the
    geometry measured to be worth 2.2-2.7x a count-matched perpendicular pair."""
    a = res["angles"]
    if not a:
        return 0.0
    n = sum(r[1] for r in a)
    return sum(r[1] * r[2] for r in a) / n


PANELS = [
    ("Drivable floor",           lambda r: r["area"],                 "m²",
     "how much room the robot actually has"),
    ("Seen by ≥1 camera",        lambda r: r["covA1"] * 100,          "% of drivable floor",
     "fill state A; higher is more of the floor observable at all"),
    ("Seen by ≥2 cameras",       lambda r: r["covA2"] * 100,          "% of drivable floor",
     "fill state A; this is where fusion is even possible"),
    ("Restock lever (≥2 cams)",  lambda r: r["flip2_frac"] * 100,     "% of cells that flip",
     "map identical in A and B; higher = harder to fake from the floor plan"),
    ("Mean crossing angle",      mean_crossing_angle,                 "degrees between views",
     "cell-weighted over all overlapped cells; 180° = head-on, which pays 2.2–2.7×"),
    ("Map captured by rectangles", lambda r: r["map_fidelity"] * 100, "% of reachable free space",
     "cost of the shape: what the axis-aligned lane format cannot express"),
    ("Ground resolution",        lambda r: r["px_per_m_median"],      "px per metre, median",
     "at covered cells, best camera; falls off with range and grazing angle"),
]


def main():
    rows = collect()
    labels = [f"{k}\n{L.title.split('(')[0].strip()}" if k != "current" else "current\nwarehouse_full_4cam"
              for k, L, _r, _b in rows]
    is_base = [b for *_x, b in rows]

    fig, axes = plt.subplots(3, 3, figsize=(15.4, 12.2), dpi=170)
    fig.patch.set_facecolor("#fcfcfb")
    order = list(range(len(rows)))[::-1]              # first row at the top

    for ax, (title, fn, unit, sub) in zip(axes.ravel(), PANELS):
        vals = [fn(r) for *_x, r, _b in [(k, L, r, b) for k, L, r, b in rows]]
        y = np.arange(len(rows))
        for i in order:
            ax.barh(y[i], vals[i], height=0.62,
                    color=BASE if is_base[i] else SERIES,
                    hatch="///" if is_base[i] else None,
                    edgecolor="#fcfcfb", linewidth=2.0, zorder=3)
            ax.text(vals[i] + max(vals) * 0.022, y[i], f"{vals[i]:.0f}" if max(vals) > 20
                    else f"{vals[i]:.1f}", va="center", ha="left",
                    fontsize=8.2, color=INK, weight="bold", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.4, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.22)
        ax.text(0, 1.155, title, transform=ax.transAxes, fontsize=10.5,
                weight="bold", color=INK, va="bottom")
        ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=7.2, color=INK2, va="bottom")
        ax.set_xlabel(unit, fontsize=7.6, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", labelsize=7.2, colors=INK2, length=0)
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor("#fcfcfb")

    for ax in axes.ravel()[len(PANELS):]:
        ax.set_visible(False)
    fig.suptitle("Five candidate warehouses against the one in the repo today",
                 fontsize=15, weight="bold", color=INK, x=0.008, ha="left", y=0.990)
    fig.text(0.008, 0.960,
             "Hatched grey is the current world, not a candidate. Every number is a ray-cast "
             "against the measured mesh geometry at a 0.10 m grid, marker at 0.35 m, "
             "1280×720 / 90° cameras.",
             fontsize=8.2, color=INK2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.935], h_pad=5.0, w_pad=2.6)
    p = OUT / "comparison.png"
    fig.savefig(p, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # text table for the README
    hdr = (f"{'':<26}{'floor':>8}{'>=1cam':>8}{'>=2cam':>8}{'flip1':>7}{'flip2':>7}"
           f"{'angle':>7}{'mapfid':>8}{'px/m':>7}{'cams':>6}")
    lines = [hdr, "-" * len(hdr)]
    for k, L, r, b in rows:
        nm = "current world" if b else f"{k} {L.title}"
        lines.append(f"{nm:<26}{r['area']:8.0f}{r['covA1']*100:8.1f}{r['covA2']*100:8.1f}"
                     f"{r['flip_frac']*100:7.1f}{r['flip2_frac']*100:7.1f}"
                     f"{mean_crossing_angle(r):7.0f}{r['map_fidelity']*100:8.1f}{r['px_per_m_median']:7.0f}{len(L.cameras):6d}")
    table = "\n".join(lines)
    print(table)
    (OUT.parent / "comparison_table.txt").write_text(table + "\n")
    return p


if __name__ == "__main__":
    print("wrote", main())
