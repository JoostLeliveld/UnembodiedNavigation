"""SOLUTION 3: the fix needs a heading, and that is what it costs."""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib.pyplot as plt
from _common import OUT
import style as D

G = json.loads((D.REPO / "logs/studies/measurement_commissioning/heading_gate.json").read_text())
T = G["table"]; sig = G["sigma_px"]
deg = [t["heading_error_deg"] for t in T]
px = [t["pixel_shift_median_px"] for t in T]
cm = [t["absorbed_position_error_median_cm"] for t in T]
cm90 = [t["absorbed_position_error_p90_cm"] for t in T]
fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.0), constrained_layout=True)
ax = axes[0]
ax.plot(deg, px, color=D.ROBOT, lw=3.4, marker="o", ms=10, zorder=4)
ax.axhline(sig, color=D.BAD, lw=2.6, ls=(0, (5, 4)))
ax.text(deg[0], sig * 1.06, f"the detector's own noise, {sig:.2f} px", color=D.BAD,
        fontsize=12.5, fontweight="bold", va="bottom")
be = G["break_even_heading_error_deg"]
ax.axvline(be, color=D.INK, lw=1.8, ls=(0, (2, 3)))
ax.annotate(f"below {be:.0f}°, heading error is\nquieter than the noise already there",
            xy=(be, sig), xytext=(1.4, 0.30), fontsize=12, color=D.INK,
            arrowprops=dict(arrowstyle="-|>", color=D.INK, lw=1.8))
ax.set_xlabel("how wrong the robot thinks its heading is (degrees)", fontsize=12.5)
ax.set_ylabel("how far that moves the predicted pixel", fontsize=12.5)
ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True); ax.set_ylim(0, 1.15)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("The catch: it barely shows in the picture", loc="left", fontsize=16.5, color=D.INK, pad=30)
ax.text(0, 1.005, "which is why the usability check cannot catch a heading error — it does not\n"
        "look like a disagreement", transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
ax = axes[1]
ax.fill_between(deg, cm, cm90, color=D.BAD, alpha=0.16, lw=0, label="the worst tenth of sightings")
ax.plot(deg, cm, color=D.BAD, lw=3.4, marker="o", ms=10, label="the typical sighting")
ax.axhline(2.24, color=D.GOOD, lw=2.6, ls=(0, (5, 4)))
ax.text(deg[-1], 2.36, "the detector's own scatter, 2.2 cm", ha="right", color=D.GOOD,
        fontsize=12.5, fontweight="bold")
for d, c in zip(deg, cm):
    if d in (3.0, 10.0):
        ax.annotate(f"{c:.1f} cm", (d, c), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=12.5, color=D.BAD, fontweight="bold")
ax.set_xlabel("how wrong the robot thinks its heading is (degrees)", fontsize=12.5)
ax.set_ylabel("position error the filter silently absorbs (cm)", fontsize=12.5)
ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="upper left")
ax.set_title("But it lands in the answer anyway", loc="left", fontsize=16.5, color=D.INK, pad=30)
ax.text(0, 1.005, "the camera update is allowed to move position and not heading, so whatever\n"
        "the heading gets wrong is explained by moving the robot instead",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.suptitle("What the fix costs: it needs a heading, and the heading comes from odometry",
             x=0.006, ha="left", fontsize=19, color=D.INK)
fig.savefig(OUT / "06_the_price_is_heading.png", dpi=175, bbox_inches="tight")
print(f"06: break-even {be:.0f} deg")
