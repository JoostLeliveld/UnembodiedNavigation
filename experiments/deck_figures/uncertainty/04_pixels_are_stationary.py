"""SOLUTION 1: the detector's error is the same size everywhere, in pixels."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib.pyplot as plt
from _common import rows, OUT, BANDS
import style as D

R, cal, _ = rows()
fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), constrained_layout=True)
ax = axes[0]
xs, gnd, pxl = [], [], []
for lo, hi in BANDS:
    s = [r for r in R if lo <= r["range_m"] < hi]
    if len(s) < 30: continue
    xs.append((lo + hi) / 2)
    gnd.append(float(np.linalg.norm(np.array([r["ground_cm"] for r in s]), axis=1).std()))
    pxl.append(float(np.array([r["px"] for r in s]).std(axis=0).mean()))
ax.plot(xs, np.array(gnd) / gnd[0], color=D.BAD, lw=3.4, marker="o", ms=10,
        label="measured in centimetres, on the floor")
ax.plot(xs, np.array(pxl) / pxl[0], color=D.GOOD, lw=3.4, marker="s", ms=10,
        label="measured in pixels, in the image")
ax.axhline(1.0, color=D.MUTED, lw=1.4)
ax.set_xlabel("distance from camera to robot (m)", fontsize=13)
ax.set_ylabel("how much the error grows,\nrelative to close range", fontsize=13)
ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True); ax.set_ylim(0, 5.4)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="upper left")
ax.set_title("The same error, measured two ways", loc="left", fontsize=16, color=D.INK, pad=30)
ax.text(0, 1.005, "on the floor it grows fivefold across the building.  In pixels it does not\n"
        "move — because it was never a property of where the robot was standing.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
ax = axes[1]
side = [float(np.array([r["px"] for r in R if lo <= r["range_m"] < hi])[:, 0].std())
        for lo, hi in BANDS if len([r for r in R if lo <= r["range_m"] < hi]) >= 30]
vert = [float(np.array([r["px"] for r in R if lo <= r["range_m"] < hi])[:, 1].std())
        for lo, hi in BANDS if len([r for r in R if lo <= r["range_m"] < hi]) >= 30]
w = 0.36; ix = np.arange(len(xs))
ax.bar(ix - w/2, side, w, color=D.ROBOT, edgecolor="white", lw=2, label="sideways")
ax.bar(ix + w/2, vert, w, color="#8fbce8", edgecolor="white", lw=2, label="up and down")
ax.axhline(cal["sigma_px"], color=D.GOOD, lw=2.6, ls=(0, (5, 4)))
ax.text(len(xs) - 0.4, cal["sigma_px"] + 0.035, f"the one number we keep: {cal['sigma_px']:.2f} px",
        ha="right", fontsize=12, color=D.GOOD, fontweight="bold")
ax.set_xticks(ix); ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS][:len(xs)], fontsize=11.5)
ax.set_xlabel("distance from camera to robot (m)", fontsize=13)
ax.set_ylabel("spread of the detector's error (pixels)", fontsize=13)
ax.set_ylim(0, 1.25); ax.grid(True, axis="y", color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=2)
ax.set_title("So measure it once, in pixels", loc="left", fontsize=16, color=D.INK, pad=30)
ax.text(0, 1.005, "the sideways spread varies by a tenth of a pixel across the whole range;\n"
        "one number covers it", transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.savefig(OUT / "04_pixels_are_stationary.png", dpi=180, bbox_inches="tight")
print("04: ground growth", round(gnd[-1]/gnd[0],2), "x   pixel growth", round(pxl[-1]/pxl[0],2), "x")
