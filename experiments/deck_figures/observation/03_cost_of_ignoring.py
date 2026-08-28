"""WHY 3: what it costs to treat the box bottom-centre as the robot."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collections, numpy as np, matplotlib.pyplot as plt
from _common import CAMS, OUT, gap_vector, index
import style as D

cells = collections.defaultdict(list)
mags = []
for r in index():
    cam = CAMS.get(r["camera_id"])
    x, y, yaw = float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])
    g = gap_vector(cam, x, y, yaw)
    if g is None:
        continue
    cells[(round(x / 1.6) * 1.6, round(y / 1.6) * 1.6)].append(g)
    mags.append(np.linalg.norm(g))
mags = np.array(mags)
fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
D.draw_warehouse(ax, D.layout())
G = np.array([(k[0], k[1], *np.mean(v, axis=0)) for k, v in cells.items() if len(v) >= 4])
m = np.hypot(G[:, 2], G[:, 3])
q = ax.quiver(G[:, 0], G[:, 1], G[:, 2], G[:, 3], m, cmap=D.SUPPORT, clim=(0, m.max()),
              angles="xy", scale_units="xy", scale=13.0, width=0.006, zorder=6)
cb = fig.colorbar(q, ax=ax, shrink=0.72, pad=0.015)
cb.set_label("how far off, in centimetres", color=D.INK2, fontsize=12)
cb.outline.set_edgecolor("#d5d4cf")
ax.set_title("Every arrow points at its own camera", loc="left", fontsize=17, color=D.INK, pad=30)
ax.text(0, 1.005, "if you treat the bottom of the box as the robot, this is where you think it is",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
ax = axes[1]
ax.hist(mags, bins=44, color=D.BAD, alpha=0.85, edgecolor="white", lw=0.6)
ax.axvline(np.median(mags), color=D.INK, lw=2.6)
ax.annotate(f"typical: {np.median(mags):.0f} cm", xy=(np.median(mags), 0), xytext=(0.62, 0.80),
            textcoords="axes fraction", fontsize=15, color=D.INK, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=D.INK, lw=2.2))
ax.axvline(2.2, color=D.GOOD, lw=2.6, ls=(0, (5, 4)))
ax.text(3.2, ax.get_ylim()[1] * 0.55, "the detector's own\nscatter: 2.2 cm",
        color=D.GOOD, fontsize=12.5, fontweight="bold")
ax.set_xlabel("how far the bottom-centre lands from the robot (cm)", fontsize=13)
ax.set_ylabel("number of sightings", fontsize=13)
ax.grid(True, axis="y", color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("Fourteen times the detector's own error", loc="left", fontsize=17, color=D.INK, pad=30)
ax.text(0, 1.005, f"{len(mags)} sightings.  This is not noise to average away — it is the same "
        f"way\nevery time, and it is by far the largest error in the chain.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.suptitle("The cost of assuming the box bottom-centre is the robot",
             x=0.006, ha="left", fontsize=19, color=D.INK)
fig.savefig(OUT / "03_cost_of_ignoring.png", dpi=175, bbox_inches="tight")
print(f"03: median {np.median(mags):.1f} cm, p90 {np.percentile(mags,90):.1f}, max {mags.max():.1f}")
