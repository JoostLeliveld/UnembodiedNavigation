"""Slide 3: where camera sightings are actually lost."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D
import json, csv, collections

OUT = D.REPO / "logs/studies/deck_figures/availability"; OUT.mkdir(parents=True, exist_ok=True)
cal = json.loads((D.REPO / "logs/studies/measurement_commissioning/calibration.json").read_text())["calibration"]
STAGES = [("the robot is somewhere a camera\ncould in principle watch", 11585, None),
          ("that camera actually has\na clear line of sight", 5407, "blocked by racks and goods"),
          ("the detector finds the robot", 4797, "too small, too far, too hidden"),
          ("the sighting survives the\nadmission checks", 3351, "shape does not match what\nthe robot should look like there")]
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.75, 1]})

ax = axes[0]
top = STAGES[0][1]
for i, (label, n, lost) in enumerate(STAGES):
    frac = n / top
    y = -i
    ax.barh(y, frac, height=0.56, color=D.SUPPORT(0.25 + 0.72 * frac),
            edgecolor="white", lw=2)
    ax.text(frac + 0.012, y, f"{n:,}".replace(",", " ") + f"   {frac*100:.0f}%",
            va="center", ha="left", fontsize=15, color=D.INK, fontweight="bold")
    ax.text(-0.015, y, label, va="center", ha="right", fontsize=12.5, color=D.INK)
    if lost:
        prev = STAGES[i - 1][1]
        ax.annotate(f"− {prev-n:,}".replace(",", " ") + f"   {lost}",
                    xy=(frac, y + 0.42), xytext=(frac, y + 0.42),
                    fontsize=11, color=D.BAD, va="bottom", ha="left")
ax.set_xlim(-0.62, 1.30); ax.set_ylim(-3.75, 0.85)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Only 29% of the chances to see the robot\nbecome a usable position measurement",
             loc="left", fontsize=18, color=D.INK, pad=12)

ax = axes[1]
reasons = {"the box is the wrong width": cal["gate_failures"]["wrong_width"],
           "the robot's feet are hidden": cal["gate_failures"]["bottom_hidden"],
           "the box is too short": cal["gate_failures"]["too_short"],
           "the robot is cut off by\nthe edge of the picture": cal["gate_failures"]["touches_frame_edge"]}
ks = list(reasons)[::-1]; vs = [reasons[k] for k in ks]
ax.barh(np.arange(len(ks)), vs, height=0.62, color=D.BAD, edgecolor="white", lw=2)
for i, v in enumerate(vs):
    ax.text(v + 22, i, f"{v:,}".replace(",", " "), va="center", fontsize=13, color=D.INK)
ax.set_yticks(np.arange(len(ks))); ax.set_yticklabels(ks, fontsize=12)
ax.set_xticks([]); ax.set_xlim(0, max(vs) * 1.26)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Why a detection is rejected\none sighting can fail more than one check",
             loc="left", fontsize=15, color=D.INK, pad=12)
fig.text(0.012, 0.015,
         "2 316 robot placements x 5 cameras in the peak-stock warehouse.  Every check compares the "
         "detection against what the robot should look like there, so all of them work without ground truth.",
         fontsize=11.5, color=D.INK2)
fig.savefig(OUT / "02_where_sightings_are_lost.png", dpi=190, bbox_inches="tight")
print("wrote 02_where_sightings_are_lost.png")
