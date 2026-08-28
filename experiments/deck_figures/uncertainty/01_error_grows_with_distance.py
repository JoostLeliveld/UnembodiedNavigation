"""WHY 1: a sighting is not equally good everywhere."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib.pyplot as plt
from _common import rows, OUT, BANDS
import style as D

R, cal, _ = rows()
fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
xs, med, p90, ns = [], [], [], []
for lo, hi in BANDS:
    s = [r for r in R if lo <= r["range_m"] < hi]
    if len(s) < 30: continue
    e = np.linalg.norm(np.array([r["ground_cm"] for r in s]), axis=1)
    xs.append((lo + hi) / 2); med.append(np.median(e)); p90.append(np.percentile(e, 90)); ns.append(len(s))
ax.fill_between(xs, 0, p90, color=D.ROBOT, alpha=0.13, lw=0, label="9 sightings in 10 land inside this")
ax.plot(xs, med, color=D.ROBOT, lw=3.4, marker="o", ms=10, label="the typical sighting")
for x, m, n in zip(xs, med, ns):
    ax.annotate(f"{m:.1f} cm", (x, m), textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=13, color=D.ROBOT, fontweight="bold")
ax.set_xlabel("distance from camera to robot (m)", fontsize=13)
ax.set_ylabel("how far the sighting lands from the robot (cm)", fontsize=13)
ax.set_ylim(bottom=0); ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="upper left")
ax.set_title("A sighting close to a camera is worth five of one far away",
             loc="left", fontsize=17, color=D.INK, pad=30)
ax.text(0, 1.005, f"{len(R)} sightings, all five cameras.  The error grows about fivefold "
        f"across the building — so one number for\nhow much to trust a camera cannot be right "
        f"in both places.", transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.savefig(OUT / "01_error_grows_with_distance.png", dpi=180, bbox_inches="tight")
print("01:", " ".join(f"{x:.0f}m={m:.1f}cm" for x, m in zip(xs, med)))
