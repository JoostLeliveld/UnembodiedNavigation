"""Facility-plan drawing primitives — make a warehouse plan read like a drawing.

The layout figures in this study looked like coloured rectangles because they were
coloured rectangles: every element rendered at the same visual weight, zones absent,
a coverage heatmap laid over the geometry so the building disappeared underneath it,
and no drawing furniture at all (no scale, no legend, no dimensions).

This module fixes the rendering side. It provides architectural symbols rather than
patches, a deliberate visual hierarchy, and the annotation furniture a plan needs.
Layout content is supplied by the caller; nothing here knows about any specific
facility.

Visual hierarchy, heaviest to lightest — this is the thing that was missing:

    1. STRUCTURE     walls (poche), columns          near-black, heavy line
    2. STORAGE       racking, block stacks           mid grey, symbol not blob
    3. EQUIPMENT     machines, conveyors, stations   desaturated blue-grey
    4. ROUTE         robot mission                   saturated accent, dashed
    5. ZONES         functional areas                very pale tint, behind all
    6. FURNITURE     scale bar, north, dimensions    thin, black, outside the plan

Rule of thumb enforced throughout: a plan is read by SHAPE first and colour second.
Every element type is distinguishable in greyscale.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle, Wedge

# --------------------------------------------------------------------- palette
# Okabe-Ito derived where colour carries meaning; greys carry structure so the
# drawing survives greyscale printing.
INK = "#1a1a1a"
WALL_FILL = "#3f3f3f"
COLUMN = "#1a1a1a"
RACK_EDGE = "#4a4a4a"
RACK_FILL = "#c9c9c9"
BLOCK_FILL = "#a8927a"          # block-stacked goods: tan, hatched
MACHINE_FILL = "#7f96a8"        # equipment: desaturated blue-grey
STATION_FILL = "#0072B2"
CHARGER_FILL = "#009E73"
DOCK = "#D55E00"
ROUTE = "#7B3294"
CAM_INHERITED = "#C1121F"
CAM_RETROFIT = "#1B7837"
ZONE_TINTS = {
    "inbound": "#FDE0C5", "outbound": "#CFE8F3", "storage": "#EDEDED",
    "pick": "#DFF0D8", "pack": "#E7DDEC", "charge": "#D8F0E6",
    "qc": "#FFF3C4", "maintenance": "#E8E0D8", "walkway": "#F2F2F2",
    "staging": "#FBE3E8", "default": "#F0F0F0",
}


def style_axes(ax, x0, y0, x1, y1, *, margin=1.6):
    """Plan-drawing axes: equal aspect, no ticks, no frame, generous margin."""
    ax.set_xlim(x0 - margin, x1 + margin)
    ax.set_ylim(y0 - margin, y1 + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


# ------------------------------------------------------------------- structure


def wall_poche(ax, x0, y0, x1, y1, *, thickness=0.45, z=10):
    """Walls as filled poche — the standard architectural convention.

    A single-line rectangle reads as a plot border; a filled band reads as a
    building. Drawn as four rectangles so openings can be punched over the top.
    """
    t = thickness
    for rect in (
        (x0 - t, y0 - t, (x1 - x0) + 2 * t, t),          # south
        (x0 - t, y1, (x1 - x0) + 2 * t, t),              # north
        (x0 - t, y0, t, y1 - y0),                        # west
        (x1, y0, t, y1 - y0),                            # east
    ):
        ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor=WALL_FILL,
                               edgecolor=INK, lw=0.8, zorder=z))


def opening(ax, x0, y0, x1, y1, *, thickness=0.45, z=11):
    """Punch a hole in the wall poche (a doorway, roller shutter, or pass-through)."""
    ax.add_patch(Rectangle((x0, y0), max(x1 - x0, thickness), max(y1 - y0, thickness),
                           facecolor="white", edgecolor="none", zorder=z))


def dock_door(ax, x0, x1, y, *, thickness=0.45, inward=1.0, label=None, z=12):
    """Dock door: opening in the poche plus a leaf swing and an approach arrow."""
    opening(ax, x0, y - thickness, x1, y, thickness=thickness, z=z)
    ax.plot([x0, x1], [y, y], lw=3.4, color=DOCK, solid_capstyle="butt", zorder=z + 1)
    mid = 0.5 * (x0 + x1)
    ax.add_patch(FancyArrow(mid, y + 0.15 * inward, 0, 0.85 * inward, width=0.06,
                            head_width=0.34, head_length=0.34, length_includes_head=True,
                            facecolor=DOCK, edgecolor="none", alpha=0.75, zorder=z + 1))
    if label:
        ax.annotate(label, xy=(mid, y - 0.75), ha="center", va="top", fontsize=6.2,
                    color=DOCK, fontweight="bold", zorder=z + 2)


def column(ax, x, y, *, size=0.4, tag=None, z=13):
    """Building column: small solid square, optionally tagged with its grid ref."""
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size,
                           facecolor=COLUMN, edgecolor="none", zorder=z))
    if tag:
        ax.annotate(tag, xy=(x, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=5.2, color=INK, alpha=0.65, zorder=z)


def column_grid(ax, xs, ys, *, size=0.4, tags=True, z=13):
    """A regular structural column grid, labelled A1/A2/... like a real drawing."""
    letters = "ABCDEFGHJKLMN"
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            column(ax, x, y, size=size,
                   tag=f"{letters[i % len(letters)]}{j + 1}" if tags else None, z=z)


# --------------------------------------------------------------------- storage


def rack_run(ax, x, y, sx, sy, *, bays=None, tall=False, label=None, z=6):
    """Selective racking: outline with bay divisions, NOT a solid blob.

    Bay lines are what makes racking read as racking on a plan. ``tall`` marks a
    double-stacked run, which is what actually occludes a 6.1 m camera.
    """
    fill = "#9a9a9a" if tall else RACK_FILL
    ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=fill,
                           edgecolor=RACK_EDGE, lw=0.9, zorder=z))
    along_y = sy >= sx
    length = sy if along_y else sx
    if bays is None:
        bays = max(1, int(round(length / 2.7)))          # ~2.7 m bay, standard beam
    for k in range(1, bays):
        t = k / bays
        if along_y:
            yy = y - sy / 2 + t * sy
            ax.plot([x - sx / 2, x + sx / 2], [yy, yy], lw=0.5, color=RACK_EDGE,
                    zorder=z + 1)
        else:
            xx = x - sx / 2 + t * sx
            ax.plot([xx, xx], [y - sy / 2, y + sy / 2], lw=0.5, color=RACK_EDGE,
                    zorder=z + 1)
    if tall:      # corner ticks mark the double stack without adding colour
        for cx in (x - sx / 2, x + sx / 2):
            for cy in (y - sy / 2, y + sy / 2):
                ax.plot([cx], [cy], marker="s", ms=2.0, color=INK, zorder=z + 2)
    if label:
        ax.annotate(label, xy=(x, y), ha="center", va="center", fontsize=5.6,
                    color=INK, rotation=90 if along_y else 0, alpha=0.8, zorder=z + 2)


def block_stack(ax, x, y, sx, sy, *, label=None, z=6):
    """Bulk floor-stacked goods: hatched, no bay lines, fully opaque to a camera."""
    ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=BLOCK_FILL,
                           edgecolor="#6b5a45", lw=0.9, hatch="xx", zorder=z))
    if label:
        ax.annotate(label, xy=(x, y), ha="center", va="center", fontsize=5.6,
                    color="#3a2f22", zorder=z + 2)


def cage(ax, x, y, sx, sy, *, label=None, z=6):
    """Mesh-panel security cage: see-through to a camera, solid to a robot.

    Drawn open with a cross-hatch and a heavy edge, so it is not mistaken for
    racking — the distinction matters here because the cage does NOT occlude.
    """
    ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor="none",
                           edgecolor="#4a4a4a", lw=1.4, hatch="++", zorder=z))
    if label:
        ax.annotate(label, xy=(x, y), ha="center", va="center", fontsize=5.6,
                    color=INK, zorder=z + 2)


def mezzanine(ax, x0, y0, x1, y1, *, label="MEZZANINE", z=5):
    """Raised structure: dashed outline with a light diagonal hatch."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=INK, lw=1.1, ls=(0, (5, 3)), hatch="//",
                           alpha=0.55, zorder=z))
    ax.annotate(label, xy=(x0 + 0.2, y1 - 0.18), ha="left", va="top", fontsize=6.0,
                color=INK, fontweight="bold", zorder=22,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="#cfcfcf", lw=0.5, alpha=0.92))


# ------------------------------------------------------------------- equipment


def machine(ax, x, y, sx, sy, *, label=None, kind="box", z=7):
    """Fixed plant: distinct fill plus a symbol glyph so it is not another rectangle."""
    ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=MACHINE_FILL,
                           edgecolor="#41566b", lw=1.0, zorder=z))
    if kind == "wrapper":                                  # turntable
        ax.add_patch(Circle((x, y), min(sx, sy) * 0.33, facecolor="none",
                            edgecolor="#22333f", lw=1.0, zorder=z + 1))
    elif kind == "scale":
        ax.plot([x - sx / 2, x + sx / 2], [y, y], lw=0.8, color="#22333f", zorder=z + 1)
        ax.plot([x, x], [y - sy / 2, y + sy / 2], lw=0.8, color="#22333f", zorder=z + 1)
    elif kind == "charger":
        ax.plot([x - sx * 0.18, x + sx * 0.05, x - sx * 0.05, x + sx * 0.18],
                [y + sy * 0.22, y, y, y - sy * 0.22], lw=1.3, color="#22333f",
                zorder=z + 1)
    if label:
        ax.annotate(label, xy=(x, y - sy / 2 - 0.18), ha="center", va="top",
                    fontsize=5.8, color="#22333f", zorder=z + 2)


def conveyor(ax, points, *, width=0.7, label=None, z=7):
    """Conveyor: twin rails with roller ticks — reads as a conveyor, not a wall."""
    pts = np.asarray(points, dtype=float)
    for i in range(len(pts) - 1):
        p, q = pts[i], pts[i + 1]
        d = q - p
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        n = np.array([-d[1], d[0]]) / length * (width / 2)
        for sign in (1, -1):
            a, b = p + sign * n, q + sign * n
            ax.plot([a[0], b[0]], [a[1], b[1]], lw=1.1, color="#41566b", zorder=z)
        for t in np.linspace(0.06, 0.94, max(2, int(length / 0.55))):
            a, b = p + d * t + n, p + d * t - n
            ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.45, color="#7f96a8", zorder=z)
    if label:
        mid = pts[len(pts) // 2]
        ax.annotate(label, xy=(mid[0], mid[1]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=5.8,
                    color="#22333f", zorder=z + 2)


def station(ax, x, y, sx, sy, *, label=None, z=8):
    """Pick/pack workstation with an operator standing position."""
    ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=STATION_FILL,
                           edgecolor=INK, lw=0.8, alpha=0.85, zorder=z))
    ax.add_patch(Circle((x, y), 0.22, facecolor="white", edgecolor=INK, lw=0.7,
                        zorder=z + 1))
    if label:
        ax.annotate(label, xy=(x, y - sy / 2 - 0.18), ha="center", va="top",
                    fontsize=5.8, color=INK, zorder=z + 2)


def guardrail(ax, points, *, z=9):
    """Safety guardrail / bollard line: heavy dashes with end posts."""
    pts = np.asarray(points, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], lw=2.0, color="#E6B800", zorder=z,
            solid_capstyle="round", ls=(0, (4, 2)))
    for p in (pts[0], pts[-1]):
        ax.plot([p[0]], [p[1]], marker="o", ms=3.0, color="#8a6d00", zorder=z + 1)


def walkway(ax, points, *, width=1.2, z=2):
    """Pedestrian walkway: hatched band, drawn low so traffic reads over it."""
    pts = np.asarray(points, dtype=float)
    for i in range(len(pts) - 1):
        p, q = pts[i], pts[i + 1]
        d = q - p
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        n = np.array([-d[1], d[0]]) / length * (width / 2)
        poly = np.array([p + n, q + n, q - n, p - n])
        ax.add_patch(Polygon(poly, closed=True, facecolor="#FFFFFF",
                             edgecolor="#bdbdbd", lw=0.5, hatch="//", alpha=0.6,
                             zorder=z))


# ----------------------------------------------------------------------- zones


def zone(ax, x0, y0, x1, y1, *, name, kind="default", label_xy=None, z=1):
    """Functional zone: pale tint plus a dashed boundary and a set-back label.

    Zones go BEHIND everything (zorder 1) and stay very pale — they orient the
    reader without competing with the geometry.
    """
    tint = ZONE_TINTS.get(kind, ZONE_TINTS["default"])
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=tint,
                           edgecolor="#b0b0b0", lw=0.7, ls=(0, (4, 3)), zorder=z))
    lx, ly = label_xy if label_xy else (x0 + 0.22, y1 - 0.22)
    ha = "center" if label_xy else "left"
    # White bbox and a high zorder: a zone label that disappears under a rack run
    # tells the reader nothing, and every earlier version lost several of them.
    ax.annotate(name.upper(), xy=(lx, ly), ha=ha, va="top", fontsize=6.6,
                color="#5a5a5a", fontweight="bold", zorder=22,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="#cfcfcf", lw=0.5, alpha=0.92))


# --------------------------------------------------------------------- cameras


def camera(ax, x, y, yaw_deg, *, reach=11.0, fov_deg=90.0, inherited=True,
           label=None, show_fov=True, label_offset=(0, -13), clip_to=None, z=20):
    """Camera with its floor footprint as a translucent wedge.

    The wedge is the honest way to show a camera on a plan: it communicates
    direction, reach and overlap at a glance, which a dot and a stick do not.
    """
    colour = CAM_INHERITED if inherited else CAM_RETROFIT
    if show_fov:
        wedge = Wedge((x, y), reach, yaw_deg - fov_deg / 2, yaw_deg + fov_deg / 2,
                      facecolor=colour, alpha=0.07, edgecolor=colour, lw=0.4,
                      zorder=3)
        ax.add_patch(wedge)
        if clip_to is not None:
            # Unclipped wedges spill outside the shell and swamp the drawing.
            cx0, cy0, cx1, cy1 = clip_to
            wedge.set_clip_path(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0,
                                          transform=ax.transData))
    ax.add_patch(Circle((x, y), 0.30, facecolor=colour, edgecolor="white", lw=1.0,
                        zorder=z))
    ax.plot([x, x + 1.5 * math.cos(math.radians(yaw_deg))],
            [y, y + 1.5 * math.sin(math.radians(yaw_deg))],
            lw=1.6, color=colour, zorder=z)
    if label:
        ax.annotate(label, xy=(x, y), xytext=label_offset, textcoords="offset points",
                    ha="center", fontsize=6.0, fontweight="bold", color=colour,
                    zorder=z + 3,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                              edgecolor=colour, lw=0.5, alpha=0.95))


# ----------------------------------------------------------------------- route


def route(ax, points, *, label=None, z=15):
    """Robot mission route: dashed accent line with numbered waypoints."""
    pts = np.asarray(points, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], lw=2.0, ls=(0, (7, 3)), color=ROUTE, zorder=z,
            alpha=0.9)
    for i, p in enumerate(pts):
        ax.add_patch(Circle((p[0], p[1]), 0.30, facecolor="white", edgecolor=ROUTE,
                            lw=1.4, zorder=z + 1))
        ax.annotate(str(i + 1), xy=(p[0], p[1]), ha="center", va="center",
                    fontsize=5.6, color=ROUTE, fontweight="bold", zorder=z + 2)
    if label:
        ax.annotate(label, xy=(pts[0][0], pts[0][1]), xytext=(10, 10),
                    textcoords="offset points", fontsize=6.4, color=ROUTE,
                    fontweight="bold", zorder=z + 2)


# ------------------------------------------------------------------- furniture


def scale_bar(ax, x, y, *, length=5.0, height=0.22, z=25):
    """Alternating black/white scale bar with end labels."""
    segments = 5
    step = length / segments
    for i in range(segments):
        ax.add_patch(Rectangle((x + i * step, y), step, height,
                               facecolor=INK if i % 2 == 0 else "white",
                               edgecolor=INK, lw=0.6, zorder=z))
    ax.annotate("0", xy=(x, y - 0.12), ha="center", va="top", fontsize=5.8, zorder=z)
    ax.annotate(f"{length:.0f} m", xy=(x + length, y - 0.12), ha="center", va="top",
                fontsize=5.8, zorder=z)


def north_arrow(ax, x, y, *, size=1.1, z=25):
    ax.add_patch(Polygon([[x, y + size], [x - size * 0.32, y - size * 0.28],
                          [x, y - size * 0.08], [x + size * 0.32, y - size * 0.28]],
                         closed=True, facecolor=INK, edgecolor=INK, zorder=z))
    ax.annotate("N", xy=(x, y + size + 0.12), ha="center", va="bottom", fontsize=7.0,
                fontweight="bold", zorder=z)


def dimension(ax, p0, p1, *, label=None, offset=0.55, z=24):
    """Dimension line with witness ticks — proves the drawing is to scale."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.hypot(*d))
    if length < 1e-9:
        return
    n = np.array([-d[1], d[0]]) / length * offset
    a, b = p0 + n, p1 + n
    ax.annotate("", xy=tuple(a), xytext=tuple(b),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7), zorder=z)
    for p, q in ((p0, a), (p1, b)):
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=0.5, color=INK, alpha=0.7, zorder=z)
    mid = 0.5 * (a + b)
    ax.annotate(label or f"{length:.2f} m", xy=tuple(mid), ha="center", va="center",
                fontsize=5.8, color=INK, zorder=z + 1,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none"))


def legend_panel(ax, entries, *, loc="upper left", title="LEGEND", fontsize=6.4):
    """Proxy-artist legend so symbols match what is actually drawn."""
    handles = []
    for kind, text in entries:
        if kind == "rack":
            handles.append(Rectangle((0, 0), 1, 1, facecolor=RACK_FILL,
                                     edgecolor=RACK_EDGE, label=text))
        elif kind == "rack_tall":
            handles.append(Rectangle((0, 0), 1, 1, facecolor="#9a9a9a",
                                     edgecolor=INK, label=text))
        elif kind == "block":
            handles.append(Rectangle((0, 0), 1, 1, facecolor=BLOCK_FILL,
                                     edgecolor="#6b5a45", hatch="xx", label=text))
        elif kind == "machine":
            handles.append(Rectangle((0, 0), 1, 1, facecolor=MACHINE_FILL,
                                     edgecolor="#41566b", label=text))
        elif kind == "station":
            handles.append(Rectangle((0, 0), 1, 1, facecolor=STATION_FILL,
                                     edgecolor=INK, label=text))
        elif kind == "wall":
            handles.append(Rectangle((0, 0), 1, 1, facecolor=WALL_FILL,
                                     edgecolor=INK, label=text))
        elif kind == "cam_inherited":
            handles.append(Line2D([], [], marker="o", ls="none", color=CAM_INHERITED,
                                  ms=6, label=text))
        elif kind == "cam_retrofit":
            handles.append(Line2D([], [], marker="o", ls="none", color=CAM_RETROFIT,
                                  ms=6, label=text))
        elif kind == "route":
            handles.append(Line2D([], [], color=ROUTE, lw=2, ls=(0, (7, 3)),
                                  label=text))
        elif kind == "dock":
            handles.append(Line2D([], [], color=DOCK, lw=3, label=text))
        elif kind == "walkway":
            handles.append(Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="#9a9a9a", hatch="///", label=text))
        elif kind == "guardrail":
            handles.append(Line2D([], [], color="#E6B800", lw=2, ls=(0, (4, 2)),
                                  label=text))
        else:
            handles.append(Rectangle((0, 0), 1, 1, facecolor="#dddddd",
                                     edgecolor=INK, label=text))
    leg = ax.legend(handles=handles, loc=loc, fontsize=fontsize, title=title,
                    framealpha=0.94, borderpad=0.7, labelspacing=0.55,
                    handlelength=1.5)
    leg.get_title().set_fontsize(fontsize + 0.6)
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(30)
    return leg


def title_block(fig, *, project, drawing, revision, notes=None):
    """Title block along the figure foot, like a real drawing sheet."""
    text = f"PROJECT: {project}     DRAWING: {drawing}     REV: {revision}"
    if notes:
        text += f"\n{notes}"
    fig.text(0.01, 0.012, text, fontsize=6.6, color=INK, va="bottom", ha="left",
             family="monospace")
