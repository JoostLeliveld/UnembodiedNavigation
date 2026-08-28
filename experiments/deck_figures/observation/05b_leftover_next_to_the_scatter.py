"""Slide 9: the bias correction -- what it moves, and how small it is next to the scatter."""
import csv, json, sys, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models  # noqa: E402
from observation import jacobian  # noqa: E402

DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
OUT = D.REPO / "logs/studies/deck_figures/observation"; OUT.mkdir(parents=True, exist_ok=True)
cal = json.loads((D.REPO / "logs/studies/measurement_commissioning/calibration.json").read_text())["calibration"]
W = np.array([cal["coefficients_du"], cal["coefficients_dv"]]).T
cams = camera_models(DATA)
raw, cor = [], []
for r in csv.DictReader(open(D.REPO / "logs/studies/measurement_commissioning/sightings.csv")):
    x, y, yaw, rng = float(r["x"]), float(r["y"]), float(r["yaw"]), float(r["range_m"])
    Ji = np.linalg.inv(jacobian(cams[r["camera"]], x, y, yaw))
    d = np.array([float(r["du_px"]), float(r["dv_px"])])
    raw.append((Ji @ d) * 100.0)
    cor.append((Ji @ (d - np.array([1.0, rng, rng ** 2]) @ W)) * 100.0)
raw, cor = np.array(raw), np.array(cor)
mr, mc = raw.mean(axis=0), cor.mean(axis=0)
sr = raw.std(axis=0) / math.sqrt(len(raw)); sc = cor.std(axis=0) / math.sqrt(len(cor))

fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1]})

ax = axes[0]
ax.scatter(raw[:, 0], raw[:, 1], s=8, color=D.INK, alpha=0.15, lw=0, zorder=2)
ax.plot(0, 0, "+", ms=20, mew=3.4, color=D.GOOD, zorder=6)
ring = cal["after"]["random_spread_cm"]
ax.add_patch(Circle((0, 0), ring, facecolor="none", edgecolor=D.ROBOT, lw=2.6,
                    ls=(0, (5, 4)), zorder=5))
ax.annotate(f"a single sighting typically\nlands within {ring:.1f} cm",
            xy=(ring * 0.71, ring * 0.71), xytext=(4.6, 6.2), fontsize=12.5, color=D.ROBOT,
            arrowprops=dict(arrowstyle="-|>", color=D.ROBOT, lw=2.0, shrinkA=2, shrinkB=4))
ax.annotate("the systematic part is a\nfifth of that, and never\naverages away", xy=(0, 0),
            xytext=(3.9, -6.6), fontsize=12.5, color=D.BAD, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=D.BAD, lw=2.0, shrinkA=2, shrinkB=22))
ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal")
ax.set_xlabel("error east–west (cm)", fontsize=12.5); ax.set_ylabel("error north–south (cm)", fontsize=12.5)
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title(f"Every one of the {len(raw)} sightings\nmost of the error is random and averages away",
             loc="left", fontsize=15.5, color=D.INK, pad=8)

ax = axes[1]
ax.axhline(0, color=D.MUTED, lw=1); ax.axvline(0, color=D.MUTED, lw=1)
ax.plot(0, 0, "+", ms=22, mew=3.6, color=D.GOOD, zorder=8)
ax.annotate("", xy=(mc[0], mc[1]), xytext=(mr[0], mr[1]), zorder=6,
            arrowprops=dict(arrowstyle="-|>", color=D.INK, lw=2.4, shrinkA=9, shrinkB=9))
for m, s, col, lab in ((mr, sr, D.BAD, "before"), (mc, sc, D.GOOD, "after")):
    ax.add_patch(Circle((m[0], m[1]), 1.96 * np.linalg.norm(s), facecolor=col, alpha=0.28,
                        edgecolor=col, lw=2.4, zorder=7))
    off = (-0.02, -0.21) if lab == "before" else (0.30, -0.16)
    ax.annotate(f"{lab}\n{np.linalg.norm(m):.2f} cm", xy=(m[0], m[1]),
                xytext=(m[0] + off[0], m[1] + off[1]),
                fontsize=14, color=col, fontweight="bold", ha="center", va="center", zorder=9)
ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_aspect("equal")
ax.set_xlabel("error east–west (cm)", fontsize=12.5)
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Zoomed in on the average\nwhere the average sighting lands, before and after",
             loc="left", fontsize=15.5, color=D.INK, pad=8)

fig.suptitle("A small repeatable offset is removed once, during commissioning",
             x=0.006, ha="left", fontsize=18.5, color=D.INK)
fig.text(0.006, -0.028,
         "Green cross = the robot's true position.  Shaded discs are the margin of error on the average itself.  "
         "The offset is only a fifth of the random scatter — but unlike the scatter it never averages away, "
         "so a robot that looks a thousand times still carries it.  Note the right panel is 12x zoomed.",
         fontsize=11.5, color=D.INK2)
fig.savefig(OUT / "05b_leftover_next_to_the_scatter.png", dpi=175, bbox_inches="tight")
print(f"wrote fig_offset.png   before {np.linalg.norm(mr):.3f} cm -> after {np.linalg.norm(mc):.3f} cm")
