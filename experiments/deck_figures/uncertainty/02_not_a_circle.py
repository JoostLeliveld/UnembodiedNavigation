"""Slide 8: what the measurement uncertainty means, and why it is an ellipse."""
import csv, json, sys, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models  # noqa: E402
from observation import jacobian  # noqa: E402

DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
OUT = D.REPO / "logs/studies/deck_figures/uncertainty"; OUT.mkdir(parents=True, exist_ok=True)
cal = json.loads((D.REPO / "logs/studies/measurement_commissioning/calibration.json").read_text())["calibration"]
SIG = cal["sigma_px"]; W = np.array([cal["coefficients_du"], cal["coefficients_dv"]]).T
cams = camera_models(DATA)
CAM = "camera_E"

rows = []
for r in csv.DictReader(open(D.REPO / "logs/studies/measurement_commissioning/sightings.csv")):
    if r["camera"] != CAM: continue
    x, y, yaw, rng = float(r["x"]), float(r["y"]), float(r["yaw"]), float(r["range_m"])
    b = np.array([1.0, rng, rng ** 2]) @ W
    J = jacobian(cams[CAM], x, y, yaw); Ji = np.linalg.inv(J)
    e = (Ji @ (np.array([float(r["du_px"]), float(r["dv_px"])]) - b)) * 100.0
    eraw = (Ji @ np.array([float(r["du_px"]), float(r["dv_px"])])) * 100.0
    rows.append((rng, x, y, yaw, e, Ji, eraw))

near = [r for r in rows if 4.0 <= r[0] < 7.0]
far = [r for r in rows if 17.0 <= r[0] < 21.0]
print(f"{CAM}: {len(near)} sightings at 4-7 m, {len(far)} at 17-21 m")

fig, axes = plt.subplots(1, 2, figsize=(11.6, 7.0), constrained_layout=True, sharex=True, sharey=True)
for ax, sub, title in ((axes[0], near, "Close to the camera  (4–7 m away)"),
                       (axes[1], far, "Far from the camera  (17–21 m away)")):
    E = np.array([s[4] for s in sub])
    Eraw = np.array([s[6] for s in sub])
    med = np.median([s[0] for s in sub])
    # the ellipse the model states, at the median pose in this group
    s0 = sub[len(sub) // 2]
    R = s0[5] @ (SIG ** 2 * np.eye(2)) @ s0[5].T * 1e4
    vals, vecs = np.linalg.eigh(R)
    ang = math.degrees(math.atan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
    k = math.sqrt(5.991)                      # 95% for two dimensions
    ax.add_patch(Ellipse((0, 0), 2 * k * math.sqrt(max(vals)), 2 * k * math.sqrt(min(vals)),
                         angle=ang, facecolor=D.ROBOT, alpha=0.13, edgecolor=D.ROBOT,
                         lw=2.6, zorder=2))
    ax.scatter(E[:, 0], E[:, 1], s=17, color=D.INK, alpha=0.42, lw=0, zorder=3)
    cx, cy = cams[CAM].cam_pos[0], cams[CAM].cam_pos[1]
    d = np.array([s0[1] - cx, s0[2] - cy]); d = d / np.linalg.norm(d) * 9
    ax.annotate("", xy=(-d[0], -d[1]), xytext=(0, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=D.BAD))
    ax.text(-d[0] * 1.14, -d[1] * 1.14, "towards\nthe camera", color=D.BAD, fontsize=11.5,
            ha="center", va="center", zorder=6)
    ax.plot(0, 0, "+", ms=18, mew=3, color=D.GOOD, zorder=7)
    mr = Eraw.mean(axis=0)
    ax.plot(mr[0], mr[1], "o", ms=11, color=D.BAD, mec="white", mew=1.8, zorder=8)
    ax.annotate(f"where this cloud sat\nbefore commissioning\n({np.linalg.norm(mr):.1f} cm off)",
                xy=(mr[0], mr[1]), xytext=(mr[0] * 0.55 - 2.0, -10.4), fontsize=10.5,
                color=D.BAD, ha="center", zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=D.BAD, lw=1.8, shrinkA=2, shrinkB=8))
    ax.set_title(f"{title}\n{len(sub)} sightings · stated uncertainty "
                 f"{math.sqrt(np.trace(R)/2):.1f} cm", loc="left", fontsize=15, color=D.INK, pad=8)
    ax.set_xlabel("error east–west (cm)", fontsize=12.5)
    ax.set_aspect("equal"); ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("error north–south (cm)", fontsize=12.5)
axes[0].set_xlim(-13, 13); axes[0].set_ylim(-13.5, 13)
fig.suptitle("A sighting is not equally good everywhere,\nand its error is a stretched ellipse rather than a circle",
             x=0.006, ha="left", fontsize=17, color=D.INK)
fig.text(0.006, -0.115,
         "Green cross = the robot's true position; each dot is one sighting after the commissioned correction.\n"
         "The clouds sit on zero by construction, so read this figure for the ellipse's SIZE and SHAPE, not its centre.\n"
         "Both come from the camera geometry plus one pixel-noise number, fitted once for the whole warehouse.",
         fontsize=12, color=D.INK2)
fig.savefig(OUT / "02_not_a_circle.png", dpi=175, bbox_inches="tight")
print("wrote 02_not_a_circle.png")
for nm, sub in (("near", near), ("far", far)):
    E = np.array([s[4] for s in sub])
    print(f"  {nm}: actual spread {E.std(axis=0)[0]:.1f} x {E.std(axis=0)[1]:.1f} cm, "
          f"median |error| {np.median(np.linalg.norm(E,axis=1)):.1f} cm")
