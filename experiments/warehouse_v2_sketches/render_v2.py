#!/usr/bin/env python3
"""Draw warehouse_v2: the plan, what the cameras see, and what the stock change does."""
from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wh_v2_mpl")

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.cm import ScalarMappable             # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

import coverage as cov                               # noqa: E402
from layouts import HALL_X, HALL_Y, NOGO_MARGIN      # noqa: E402
from warehouse_v2 import C_PEAK, C_ROWS, DOCK_DOORS, build  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2 = "#0b0b0b", "#52514e"
FLOOR, WALL, LANE = "#eceae5", "#3f4a55", "#a9dceb"
# one sequential ramp, light to dark, on the thing that matters: how tall it is
HEIGHT = LinearSegmentedColormap.from_list(
    "stock", ["#f6ecd4", "#e0c081", "#c2913c", "#96661d", "#5f3d0c"])
HNORM = Normalize(0.8, 5.4)


def is_box(o):
    return o.zone.startswith("C")


def shell(ax, small=False):
    ax.set_facecolor("#fcfcfb")
    ax.add_patch(Rectangle((HALL_X[0], HALL_Y[0]), 24, 20, facecolor=FLOOR,
                           edgecolor=WALL, lw=2.2, zorder=0))
    for x in DOCK_DOORS:
        ax.add_patch(Rectangle((x - 1.4, HALL_Y[0] - 0.17), 2.8, 0.34,
                               facecolor="#5d6a75", edgecolor=WALL, lw=0.6, zorder=6))
    ax.set_xlim(HALL_X[0] - 1.0, HALL_X[1] + 1.0)
    ax.set_ylim(HALL_Y[0] - 1.0, HALL_Y[1] + 1.0)
    ax.set_aspect("equal")
    ax.set_xticks(range(-12, 13, 4)); ax.set_yticks(range(-8, 9, 4))
    ax.tick_params(labelsize=6.5 if small else 8, colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def zones(ax, faded=False):
    for z in W.zones:
        ax.add_patch(Rectangle((z.xmin - NOGO_MARGIN, z.ymin - NOGO_MARGIN),
                               z.sx + 2 * NOGO_MARGIN, z.sy + 2 * NOGO_MARGIN,
                               facecolor="none", edgecolor="#1f8a4c", lw=0.8,
                               ls=(0, (4, 3)), alpha=0.35 if faded else 0.85, zorder=3))
        ax.add_patch(Rectangle((z.xmin, z.ymin), z.sx, z.sy, facecolor="#d5dade",
                               edgecolor="#828d96", lw=0.7,
                               alpha=0.3 if faded else 0.65, zorder=3.4))


def contents(ax, state, labels=False):
    for o in (W.fill_a if state == "A" else W.fill_b):
        if o.zone in ("DOCK_OFFICE",):
            face, edge, hatch = "#c9c3b8", "#6d6459", None
        else:
            face = HEIGHT(HNORM(o.h))
            edge, hatch = ("#2f4858", "///") if is_box(o) else ("#5a3f12", None)
        ax.add_patch(Rectangle((o.cx - o.sx / 2, o.cy - o.sy / 2), o.sx, o.sy,
                               facecolor=face, edgecolor=edge, lw=0.9,
                               hatch=hatch, zorder=4))
        if labels and o.sx * o.sy > 1.9:
            ax.text(o.cx, o.cy, f"{o.h:.1f}", ha="center", va="center",
                    rotation=90 if o.sy > o.sx else 0, fontsize=5.2, weight="bold",
                    color="#ffffff" if o.h > 2.6 else "#241a08", zorder=6)


def lanes(ax):
    for L in W.lanes:
        ax.add_patch(Rectangle((L.xmin, L.ymin), L.xmax - L.xmin, L.ymax - L.ymin,
                               facecolor=LANE, edgecolor="none", alpha=0.6, zorder=2))


def cameras(ax, arrows=True):
    for c in W.cameras:
        yaw = math.radians(c.yaw_deg)
        if arrows:
            ax.annotate("", xy=(c.x + 2.2 * math.cos(yaw), c.y + 2.2 * math.sin(yaw)),
                        xytext=(c.x, c.y), zorder=8,
                        arrowprops=dict(arrowstyle="-|>", color=c.colour, lw=1.8))
        r = 0.60 if arrows else 0.44
        ax.add_patch(Circle((c.x, c.y), r, facecolor=c.colour, edgecolor="#ffffff",
                            lw=1.6, zorder=9))
        ax.add_patch(Circle((c.x, c.y), r, facecolor="none", edgecolor="#10161c",
                            lw=0.9, zorder=9.1))
        ax.text(c.x, c.y, c.name, ha="center", va="center",
                fontsize=8.5 if arrows else 6.5, color="white", weight="bold", zorder=10)


def cov_panel(ax, res, state):
    n = np.where(res["drive"], res[state]["n_visible"].astype(float), np.nan)
    ax.imshow(n, origin="lower", extent=[*HALL_X, *HALL_Y],
              cmap=ListedColormap(["#c0392b", "#f0a04b", "#f4e08a",
                                   "#8fce8f", "#2e8b57", "#14532d"]),
              norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5], 6),
              interpolation="nearest", zorder=2)


def main():
    global W
    W = build()
    res = cov.analyse(W)

    fig = plt.figure(figsize=(17.6, 9.4), dpi=170)
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 3, width_ratios=[2.0, 1.0, 1.0], hspace=0.24, wspace=0.10)

    ax = fig.add_subplot(gs[:, 0])
    shell(ax); lanes(ax); zones(ax); contents(ax, "A", labels=True); cameras(ax)
    ax.set_title("Plan at peak stock: two racking sections and one block-stack section",
                 fontsize=11.5, weight="bold", color=INK, pad=10)
    ax.set_xlabel("metres east of hall centre", fontsize=8, color=INK2, labelpad=2)
    ax.set_ylabel("metres north of hall centre", fontsize=8, color=INK2)

    for x, y, t, rot, size in [
        (-5.6, 8.35, "SECTION A · racking, north–south, 11.75 m runs", 0, 7.6),
        (5.7, 2.80, "SECTION B · racking turned 90°, shorter, lower", 0, 7.6),
        (6.6, -6.15, "SECTION C · two long floor-storage rows, 9.55 m box aisle", 0, 7.6),
        (-5.6, -6.15, "INBOUND DOCK APRON", 0, 7.6),
        (0.62, 5.00, "artery", 90, 6.6),
    ]:
        ax.text(x, y, t, ha="center", va="center", rotation=rot, fontsize=size,
                weight="bold", color="#5b6771", zorder=7)
    for name, y0, y1 in C_ROWS:                 # label each stack with its height
        for k, nh in enumerate(C_PEAK[name]):
            if nh < 1:
                continue
            cx = 1.80 + 0.06 + (9.55 - 0.12) / len(C_PEAK[name]) * (k + 0.5)
            ax.text(cx, 0.5 * (y0 + y1) - 1.55, f"{nh} high", ha="center", va="center",
                    fontsize=6.6, color="#ffffff", weight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.20", fc="#2f4858", ec="none"))

    ramp = ScalarMappable(HNORM, HEIGHT)
    cb = fig.colorbar(ramp, ax=ax, orientation="horizontal", fraction=0.030,
                      pad=0.085, aspect=42)
    cb.set_label("how tall the stored goods are [m] — the one thing the map does not say",
                 fontsize=7.4, color=INK2, labelpad=4)
    cb.ax.tick_params(labelsize=6.8, colors=INK2, length=2)
    cb.outline.set_visible(False)

    ax.legend([Patch(facecolor="#c2913c", edgecolor="#5a3f12"),
               Patch(facecolor="#c2913c", edgecolor="#2f4858", hatch="///"),
               Patch(facecolor=LANE, alpha=0.6, edgecolor="none"),
               Patch(facecolor="#d5dade", edgecolor="#828d96"),
               Line2D([0], [0], color="#1f8a4c", ls=(0, (4, 3)), lw=1.2)],
              ["pallet racking", "boxes stacked on the floor",
               "drivable lane (in the map)", "storage zone (in the map)",
               "0.32 m keep-out envelope"],
              loc="upper center", bbox_to_anchor=(0.5, -0.245), ncol=3,
              fontsize=7.2, frameon=False, handlelength=1.6, columnspacing=1.8)

    for state, cell, when in (("A", gs[0, 1], "peak season: racks and floor full"),
                              ("B", gs[1, 1], "after the post-peak ship-out")):
        a = fig.add_subplot(cell)
        shell(a, small=True); cov_panel(a, res, state); zones(a, faded=True)
        contents(a, state); cameras(a, arrows=False)
        a.set_title(f"{when}\n{res[f'cov{state}1']*100:.0f}% of the lanes seen by a "
                    f"camera, {res[f'cov{state}2']*100:.0f}% by two",
                    fontsize=8.5, weight="bold", color=INK, pad=6)

    a = fig.add_subplot(gs[0, 2])
    shell(a, small=True)
    dA, dB = res["A"]["n_visible"] >= 2, res["B"]["n_visible"] >= 2
    flip = np.where(res["drive"], np.where(dA & ~dB, -1.0, np.where(dB & ~dA, 1.0, 0.0)), np.nan)
    a.imshow(flip, origin="lower", extent=[*HALL_X, *HALL_Y],
             cmap=ListedColormap(["#c0392b", "#e7ecef", "#2a78d6"]),
             norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3), interpolation="nearest", zorder=2)
    zones(a, faded=True); cameras(a, arrows=False)
    a.set_title(f"what the stock change does, map held identical\n"
                f"{res['flip2_frac']*100:.0f}% of the lanes change two-camera cover",
                fontsize=8.5, weight="bold", color=INK, pad=6)
    a.text(0.5, -0.10, "red = lost the second camera,  blue = gained one",
           transform=a.transAxes, ha="center", fontsize=6.8, color=INK2)

    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    ang = "\n".join(f"     {p:<5} {n:6d} cells  {d:5.0f}°" for p, n, d in res["angles"][:6])
    txt = f"""BUILDING
     hall             24.0 x 20.0 m, walls 9.0 m
     dock apron       4.95 m deep, doors at x = -8.5, -1.5, +5.5
     main artery      2.35 m, north-south, between the halves

  SECTION A  racking, north-south
     3 back-to-back pairs, 1.90 m deep, 1.80 m aisles
     runs 11.75 m = 3 native ShelfD/E modules, top beam 4.2 m

  SECTION B  racking, turned 90 degrees, lower
     2 pairs, runs 7.83 m = 2 modules, top beam 2.6 m
     reached from the artery and the 1.80 m east cross aisle

  SECTION C  boxes on the floor, no racking
     the same pallet, stacked three different ways
     C1  one high,   1.06 m, dense grid, 2.30 m bay
     C2  two high,   2.12 m, rows with gaps, 2.10 m bay
     C3  three high, 3.17 m, against the east wall

WHAT THE ROBOT GETS
     drivable floor   {res['area']:.0f} m2 in {res['n_lanes']} lane rectangles
     lanes capture    {res['map_fidelity']*100:.1f}% of reachable free space
     lane in keepout  {res['envelope_hit']} cells
     ground res       {res['px_per_m_median']:.0f} px per metre, median

WHAT THE CAMERAS GET
     seen by 1+       {res['covA1']*100:.0f}% peak  ->  {res['covB1']*100:.0f}% after
     seen by 2+       {res['covA2']*100:.0f}% peak  ->  {res['covB2']*100:.0f}% after
     stock flips      {res['flip_frac']*100:.0f}% of lanes for one camera,
                      {res['flip2_frac']*100:.0f}% for two -- map unchanged

OVERLAP: cells both see, and the angle between them
{ang}

CAMERAS
""" + "\n".join(f"     {c.name}  ({c.x:+6.2f}, {c.y:+6.2f}, {c.z:.1f} m)  {c.mount}"
                for c in W.cameras)
    a.text(0, 1.0, txt, va="top", ha="left", fontsize=6.3, family="DejaVu Sans Mono",
           color=INK, transform=a.transAxes)

    fig.suptitle("warehouse_v2: three storage sections, none of them a copy of another — "
                 "and the drivable map is the same in both stock states",
                 fontsize=14, weight="bold", color=INK, y=0.985)
    p = OUT / "warehouse_v2.png"
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"drivable {res['area']:.0f} m2 | seen 1+ {res['covA1']*100:.0f}/{res['covB1']*100:.0f}% "
          f"| 2+ {res['covA2']*100:.0f}/{res['covB2']*100:.0f}% | flip1 {res['flip_frac']*100:.0f}% "
          f"| flip2 {res['flip2_frac']*100:.0f}% | mapfid {res['map_fidelity']*100:.0f}% "
          f"| lanes {res['n_lanes']} -> {p.name}")
    return p


if __name__ == "__main__":
    main()
