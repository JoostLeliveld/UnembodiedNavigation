"""WHY 2: how large the gap is, and how it changes as the robot turns."""
import sys, math; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collections, numpy as np, matplotlib.pyplot as plt
from _common import CAMS, OUT, gap_vector, index
import style as D

L, W = 0.80, 0.55
PLACES = [("camera_E", 2.0062, -6.1222), ("camera_A", -5.35, -1.61), ("camera_D", 5.35, 4.19)]
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), constrained_layout=True)
ax = axes[0]
for (cam_id, px, py), col in zip(PLACES, (D.BAD, "#c98500", D.OLD)):
    cam = CAMS[cam_id]
    pts = np.array([gap_vector(cam, px, py, t) for t in np.arange(0, 2 * math.pi, math.radians(2))])
    d = math.hypot(px - cam.cam_pos[0], py - cam.cam_pos[1])
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=4.0, solid_capstyle="round",
            label=f"camera {cam_id[-1]}, {d:.0f} m away")
for t, c in ((0.0, "#c6c5bd"), (math.pi / 2, "#e2e1da")):
    cs, sn = math.cos(t), math.sin(t)
    corners = np.array([[-L/2,-W/2],[L/2,-W/2],[L/2,W/2],[-L/2,W/2]]) @ np.array([[cs,sn],[-sn,cs]])
    ax.add_patch(plt.Polygon(corners * 100, closed=True, facecolor=c, edgecolor="#9a998f",
                             lw=1.3, zorder=1))
ax.plot(0, 0, "*", ms=26, color=D.GOOD, mec="white", mew=1.6, zorder=8)
ax.set_aspect("equal"); ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_xlabel("centimetres east", fontsize=12.5); ax.set_ylabel("centimetres north", fontsize=12.5)
ax.legend(frameon=False, fontsize=12, loc="upper left")
ax.set_title("Where the measurement lands, over a full turn", loc="left", fontsize=16.5, color=D.INK, pad=30)
ax.text(0, 1.005, "grey = the robot's own footprint at two headings, for scale.  Each loop\n"
        "is one place: the gap always points at that place's camera.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
ax = axes[1]
by = collections.defaultdict(list)
for r in index():
    cam = CAMS.get(r["camera_id"])
    g = gap_vector(cam, float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"]))
    if g is None: continue
    by[(r["camera_id"], round(float(r["robot_x"]), 3), round(float(r["robot_y"]), 3))].append(np.linalg.norm(g))
sw = np.array([max(v) - min(v) for v in by.values() if len(v) >= 4])
mg = np.array([np.mean(v) for v in by.values() if len(v) >= 4])
ax.scatter(mg, sw, s=14, color=D.ROBOT, alpha=0.32, lw=0)
ax.axhline(np.median(sw), color=D.BAD, lw=2.6)
ax.text(mg.max() * 0.99, np.median(sw) + 0.35, f"typical swing: {np.median(sw):.0f} cm",
        ha="right", fontsize=13.5, color=D.BAD, fontweight="bold")
ax.set_xlabel("average gap at that place (cm)", fontsize=12.5)
ax.set_ylabel("how much the gap moves as the robot turns (cm)", fontsize=12.5)
ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("And it is not a fixed offset you could just subtract", loc="left", fontsize=16.5, color=D.INK, pad=30)
ax.text(0, 1.005, f"one dot per camera per place, {len(mg)} of them.  A 0.80 x 0.55 m robot shows\n"
        f"between 27.5 and 48.5 cm of itself along the viewing ray, depending which way it faces.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.savefig(OUT / "02_how_big_and_how_it_moves.png", dpi=175, bbox_inches="tight")
print(f"02: gap {mg.min():.0f}-{mg.max():.0f} cm, swing median {np.median(sw):.1f} cm")
