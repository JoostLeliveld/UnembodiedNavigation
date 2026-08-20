#!/usr/bin/env python3
"""Render the candidate layouts: top-down plan + measured coverage.

Every rectangle drawn is a measured mesh footprint from mesh_library, every
coverage number is a ray-cast against the same geometry, and the A/B panels use
identical footprints so the drivable map is provably the same in both.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wh_v2_mpl")

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

import coverage as cov                              # noqa: E402
from layouts import HALL_X, HALL_Y, MARKER_Z, NOGO_MARGIN, load_all  # noqa: E402
from mesh_library import colour as mesh_colour      # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

HEIGHT_EDGE = {  # outline weight communicates the stock level at a glance
    "low": ("#7f8c8d", 0.9),
    "std": ("#5b4222", 1.3),
    "tall": ("#b34700", 1.9),
    "high": ("#8e1b1b", 2.4),
    "bulk": ("#3b0d63", 3.0),
}


def band(h):
    if h >= 5.0:
        return "bulk"
    if h >= 3.5:
        return "high"
    if h >= 2.4:
        return "tall"
    if h >= 1.6:
        return "std"
    return "low"


def draw_shell(ax):
    ax.add_patch(Rectangle((HALL_X[0], HALL_Y[0]), 24, 20, facecolor="#e8e4dd",
                           edgecolor="#3f4a55", lw=2.0, zorder=0))
    ax.set_xlim(HALL_X[0] - 1.1, HALL_X[1] + 1.1)
    ax.set_ylim(HALL_Y[0] - 1.1, HALL_Y[1] + 1.1)
    ax.set_aspect("equal")
    ax.set_xticks(range(-12, 13, 4))
    ax.set_yticks(range(-10, 11, 4))
    ax.grid(True, color="white", lw=0.9, ls=(0, (2, 5)), zorder=1)
    ax.tick_params(labelsize=7)


def draw_zones(ax, layout, faded=False):
    """The declared map: identical in every fill state."""
    for z in layout.zones:
        ax.add_patch(Rectangle((z.xmin - NOGO_MARGIN, z.ymin - NOGO_MARGIN),
                               z.sx + 2 * NOGO_MARGIN, z.sy + 2 * NOGO_MARGIN,
                               facecolor="none", edgecolor="#1f8a4c", lw=0.7,
                               ls=(0, (3, 2)), alpha=0.5 if faded else 0.85, zorder=3))
        ax.add_patch(Rectangle((z.xmin, z.ymin), z.sx, z.sy,
                               facecolor="#cfd6dc", edgecolor="#5d6a75", lw=0.7,
                               alpha=0.30 if faded else 0.55, zorder=3.5))


def draw_fill(ax, layout, state="A", label_heights=True):
    """What is actually in the bays in this fill state."""
    for o in (layout.fill_a if state == "A" else layout.fill_b):
        ec, lw = HEIGHT_EDGE[band(o.h)]
        ax.add_patch(Polygon(o.corners(), closed=True,
                             facecolor=mesh_colour(o.mesh), edgecolor=ec, lw=lw,
                             alpha=0.95, zorder=4))
        if label_heights and (o.sx * o.sy) > 1.6:
            ax.text(o.cx, o.cy, f"{o.h:.1f}", ha="center", va="center", fontsize=4.4,
                    color="#16202b", rotation=math.degrees(o.yaw),
                    zorder=6, weight="bold")


def draw_lanes(ax, layout):
    for L in layout.lanes:
        ax.add_patch(Rectangle((L.xmin, L.ymin), L.xmax - L.xmin, L.ymax - L.ymin,
                               facecolor="#35b8d4", edgecolor="none",
                               alpha=0.30, zorder=2.2))


def draw_cameras(ax, layout, reach=14.1):
    """Wedge reach 14.1 m is the real one: 6.1 m mount, 90 deg HFOV on 1280x720
    gives a 58.7 deg vertical FOV, so a 52.7 deg down-tilt puts the far edge at
    6.1/tan(23.4 deg) = 14.1 m and the near edge at 0.85 m."""
    for c in layout.cameras:
        half = math.degrees(math.atan(math.tan(cov.FOV_H / 2)))
        vfov = 2 * math.degrees(math.atan(math.tan(cov.FOV_H / 2) * cov.IMG_H / cov.IMG_W))
        far_ang = max(math.radians(c.pitch_deg - vfov / 2), math.radians(2.0))
        near_ang = math.radians(min(c.pitch_deg + vfov / 2, 88.0))
        r_far = min(c.z / math.tan(far_ang), 26.0)
        r_near = c.z / math.tan(near_ang)
        yaw = math.radians(c.yaw_deg)
        pts = []
        for r in (r_near, r_far):
            arc = [(c.x + r * math.cos(yaw + a), c.y + r * math.sin(yaw + a))
                   for a in np.linspace(-math.radians(half), math.radians(half), 12)]
            pts.append(arc)
        poly = pts[0] + pts[1][::-1]
        ax.add_patch(Polygon(poly, closed=True, facecolor=c.colour, alpha=0.045,
                             lw=0.0, zorder=1.6))
        ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=c.colour,
                             alpha=0.40, lw=0.9, zorder=5))
        ax.add_patch(Circle((c.x, c.y), 0.42, facecolor=c.colour, edgecolor="#10161c",
                            lw=1.1, zorder=8))
        ax.text(c.x, c.y, c.name, ha="center", va="center", fontsize=6.5,
                color="white", weight="bold", zorder=9)
        lx = c.x + 1.5 * math.cos(yaw)
        ly = c.y + 1.5 * math.sin(yaw)
        ax.annotate("", xy=(lx, ly), xytext=(c.x, c.y), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color="#10161c", lw=1.2))


def coverage_panel(ax, res, state):
    xs, ys = res["xs"], res["ys"]
    n = res[state]["n_visible"].astype(float)
    n = np.where(res["drive"], n, np.nan)
    cmap = ListedColormap(["#c0392b", "#e8983a", "#f2df6b", "#7fc97f", "#2e8b57", "#14532d"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    ax.imshow(n, origin="lower", extent=[HALL_X[0], HALL_X[1], HALL_Y[0], HALL_Y[1]],
              cmap=cmap, norm=norm, interpolation="nearest", zorder=2)
    return cmap, norm


def render(res):
    L = res["layout"]
    fig = plt.figure(figsize=(18.2, 9.6), dpi=170)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.05, 1.0, 1.0], height_ratios=[1.0, 1.0],
                          hspace=0.16, wspace=0.13)

    ax = fig.add_subplot(gs[:, 0])
    draw_shell(ax); draw_lanes(ax, L); draw_zones(ax, L); draw_fill(ax, L, "A"); draw_cameras(ax, L)
    ax.set_title(f"{L.key} - {L.title}: plan, stock state A", fontsize=11, weight="bold", pad=8)
    ax.set_xlabel("x [m]", fontsize=8); ax.set_ylabel("y [m]", fontsize=8)

    for k, (state, gsi) in enumerate([("A", gs[0, 1]), ("B", gs[1, 1])]):
        a = fig.add_subplot(gsi)
        draw_shell(a)
        coverage_panel(a, res, state)
        draw_zones(a, L, faded=True); draw_fill(a, L, state, label_heights=False)
        for c in L.cameras:
            a.add_patch(Circle((c.x, c.y), 0.42, facecolor=c.colour, edgecolor="#10161c",
                               lw=1.0, zorder=8))
        c1 = res[f"cov{state}1"]; c2 = res[f"cov{state}2"]
        a.set_title(f"stock state {state}: {c1*100:.0f}% seen by >=1 cam, "
                    f"{c2*100:.0f}% by >=2", fontsize=8.5, weight="bold", pad=5)

    # flip map: what the drivable map cannot tell you
    a = fig.add_subplot(gs[0, 2])
    draw_shell(a)
    dA = res["A"]["n_visible"] >= 1
    dB = res["B"]["n_visible"] >= 1
    flip = np.where(res["drive"], np.where(dA & ~dB, -1.0, np.where(dB & ~dA, 1.0, 0.0)), np.nan)
    a.imshow(flip, origin="lower", extent=[HALL_X[0], HALL_X[1], HALL_Y[0], HALL_Y[1]],
             cmap=ListedColormap(["#b1372f", "#dfe6ea", "#2f6fb5"]),
             norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3), interpolation="nearest", zorder=2)
    draw_zones(a, L, faded=True)
    a.set_title(f"A->B coverage flips: {res['flip_frac']*100:.1f}% of drivable cells\n"
                f"(red lost, blue gained; the map is identical in both)",
                fontsize=8, weight="bold", pad=5)

    # text panel
    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    ang = res["angles"][:5]
    lines = [
        f"IDEA   {L.idea}", "",
        f"TRADE-OFF   {L.tradeoff}", "",
        f"drivable area          {res['area']:.1f} m2  ({res['n_drive']} cells)",
        f"cameras                {len(L.cameras)}",
        f"reachable free space   {res['reach_area']:.1f} m2",
        f"lanes (derived rects)  {res['n_lanes']}  covering {res['map_fidelity']*100:.1f}% of it",
        f"lane inside envelope   {res['envelope_hit']} cells",
        f"coverage A / B (>=1)   {res['covA1']*100:.1f}% / {res['covB1']*100:.1f}%",
        f"coverage A / B (>=2)   {res['covA2']*100:.1f}% / {res['covB2']*100:.1f}%",
        f"median ground res      {res['px_per_m_median']:.0f} px per m at covered cells",
        f"restock flips >=1 cam  {res['flip_cells']} cells ({res['flip_frac']*100:.1f}%)",
        f"restock flips >=2 cam  {res['flip2_frac']*100:.1f}% of drivable cells",
        f"camera-cell pairs lost {res['pair_delta']}",
        "", "overlap pairs (cells, median crossing angle):",
    ] + [f"   {p:<7} {n:6d} cells   {d:5.0f} deg" for p, n, d in ang] + ["", "notes:"] \
      + [f"   - {t}" for t in L.notes]
    a.text(0.0, 1.0, "\n".join(_wrap(lines, 74)), va="top", ha="left", fontsize=6.4,
           family="DejaVu Sans Mono", transform=a.transAxes)

    fig.suptitle(f"warehouse v2 sketch {L.key}: {L.title}", fontsize=14, weight="bold", y=0.975)
    p = OUT / f"sketch_{L.key}_{L.title.lower().replace(' ', '_')}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _wrap(lines, w):
    out = []
    for ln in lines:
        while len(ln) > w:
            cut = ln.rfind(" ", 0, w)
            cut = cut if cut > 20 else w
            out.append(ln[:cut]); ln = "      " + ln[cut:].lstrip()
        out.append(ln)
    return out


def main():
    rows = []
    for L in load_all():
        res = cov.analyse(L)
        p = render(res)
        rows.append((L, res, p))
        print(f"{L.key} {L.title:<16} area={res['area']:6.1f} m2  "
              f"cov1 A/B={res['covA1']*100:5.1f}/{res['covB1']*100:5.1f}%  "
              f"cov2 A/B={res['covA2']*100:5.1f}/{res['covB2']*100:5.1f}%  "
              f"flip1={res['flip_frac']*100:5.1f}%  flip2={res['flip2_frac']*100:5.1f}%  "
              f"map_fid={res['map_fidelity']*100:5.1f}%  -> {p.name}")
    return rows


if __name__ == "__main__":
    main()
