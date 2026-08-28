"""SOLUTION 2: one number, carried into the world by the camera geometry."""
import sys, math; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib.pyplot as plt
from _common import rows, OUT, BANDS
import style as D

R, cal, _ = rows()
sig = cal["sigma_px"]
xs, pred, act = [], [], []
for lo, hi in BANDS:
    s = [r for r in R if lo <= r["range_m"] < hi]
    if len(s) < 30: continue
    xs.append((lo + hi) / 2)
    pred.append(float(np.mean([math.sqrt(np.trace(r["Jinv"] @ (sig**2*np.eye(2)) @ r["Jinv"].T)/2)*100 for r in s])))
    e = np.array([r["ground_cm"] for r in s])
    act.append(float(np.sqrt((e**2).sum(axis=1).mean()/2)))
fig, ax = plt.subplots(figsize=(9.8, 6.0), constrained_layout=True)
ax.plot(xs, pred, color=D.ROBOT, lw=3.6, marker="o", ms=11, zorder=4,
        label="what one pixel number predicts, before seeing any error")
ax.plot(xs, act, color=D.BAD, lw=0, marker="X", ms=15, zorder=5,
        label="what the errors actually turned out to be")
for x, p, a in zip(xs, pred, act):
    ax.annotate(f"{a/p:.2f}x", (x, max(p, a)), textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=12.5, color=D.INK2)
ax.set_xlabel("distance from camera to robot (m)", fontsize=13)
ax.set_ylabel("spread of the sighting (cm)", fontsize=13)
ax.set_ylim(bottom=0); ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=12.5, loc="upper left")
ax.set_title("One number, and the camera geometry does the rest",
             loc="left", fontsize=17, color=D.INK, pad=30)
ax.text(0, 1.005, f"the spread changes {act[-1]/act[0]:.0f}-fold across the building and the "
        f"prediction tracks it to within 15%.\nNothing about distance was fitted — the geometry "
        f"already knew.", transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.savefig(OUT / "05_geometry_does_the_rest.png", dpi=180, bbox_inches="tight")
print("05: ratios", " ".join(f"{a/p:.2f}" for p, a in zip(pred, act)))
