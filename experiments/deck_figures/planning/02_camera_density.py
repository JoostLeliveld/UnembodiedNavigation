"""Slide 13: the shape of the result the paper is trying to earn -- with predicted numbers."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D

OUT = D.REPO / "logs/studies/deck_figures/planning"; OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("/tmp/claude-1000/-home-joostleliveld-Thesis/aef77627-7fa8-4158-b9c3-48026fb1db68/scratchpad/camdensity.json")
data = json.loads(SRC.read_text())
ORDER = ["1 camera", "2 cameras", "3 cameras", "5 cameras"]
n = [1, 2, 3, 5]
short = np.array([[data[t][k]["short_blind"] for t in data] for k in ORDER])
best = np.array([[data[t][k]["best_blind"] for t in data] for k in ORDER])
det = np.array([[data[t][k]["detour"] * 100 for t in data] for k in ORDER])

fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.25, 1]})

ax = axes[0]
for arr, col, lab, mk in ((short, D.BAD, "shortest route", "o"),
                          (best, D.GOOD, "route chosen for camera support", "s")):
    med = np.median(arr, axis=1)
    ax.fill_between(n, np.percentile(arr, 25, axis=1), np.percentile(arr, 75, axis=1),
                    color=col, alpha=0.16, lw=0)
    ax.plot(n, med, color=col, lw=3.4, marker=mk, ms=11, label=lab, zorder=5)
    for x, y in zip(n, med):
        ax.annotate(f"{y:.1f} m", (x, y), textcoords="offset points", xytext=(0, 13 if col == D.BAD else -22),
                    ha="center", fontsize=12, color=col, fontweight="bold")
ax.set_xticks(n); ax.set_xticklabels(["1", "2", "3", "5"], fontsize=13)
ax.set_xlabel("cameras installed in the warehouse", fontsize=13)
ax.set_ylabel("longest stretch driven with\nno usable camera sighting  (m)", fontsize=13)
ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=12.5, frameon=False, loc="upper right")
ax.set_ylim(bottom=0)
ax.set_title("Thinner camera coverage punishes the shortest route,\n"
             "and barely touches a support-aware one",
             loc="left", fontsize=17, color=D.INK, pad=10)

ax = axes[1]
med = np.median(det, axis=1)
ax.bar(range(len(n)), med, color=D.ROBOT, width=0.58, edgecolor="white", lw=2)
for i, v in enumerate(med):
    ax.text(i, v + 0.09, f"{v:.1f}%", ha="center", fontsize=13, color=D.INK, fontweight="bold")
ax.set_xticks(range(len(n))); ax.set_xticklabels(["1", "2", "3", "5"], fontsize=13)
ax.set_xlabel("cameras installed", fontsize=13)
ax.set_ylabel("extra distance travelled (%)", fontsize=13)
ax.grid(True, axis="y", color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_ylim(0, max(med) * 1.5)
ax.set_title("What that costs\nthe detour stays small however sparse the cameras",
             loc="left", fontsize=17, color=D.INK, pad=10)

fig.text(0.012, -0.045,
         f"Median over {len(data)} start-goal tasks; shaded band is the middle half.  "
         "Predicted from the measured availability of each camera, not yet from driven runs.  "
         "Camera subsets are the best-covering choice at each count.",
         fontsize=11.5, color=D.INK2)
fig.savefig(OUT / "02_camera_density.png", dpi=185, bbox_inches="tight")
print("wrote 02_camera_density.png")
for i, k in enumerate(ORDER):
    print(f"  {k:10s} shortest {np.median(short[i]):5.1f} m   support-aware {np.median(best[i]):5.1f} m   detour {np.median(det[i]):4.1f}%")
