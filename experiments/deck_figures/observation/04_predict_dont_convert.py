"""SOLUTION 1: predict the box instead of converting the pixel."""
import sys, math; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv, numpy as np, cv2, matplotlib.pyplot as plt
from _common import CAMS, DATA, OUT, gap_vector, index
import style as D, sources as src
from observation import h, predicted_box  # noqa: E402
sys.path.insert(0, str(D.REPO / "src/unav_common"))
from unav_common.robot_hull import VISUAL_HULL  # noqa: E402

CAM = "camera_E"
rows = [r for r in index() if r["image"] and r["camera_id"] == CAM]
r = min(rows, key=lambda q: abs(float(q["camera_range_m"]) - 9.0))
cam = CAMS[CAM]
x, y, yaw = float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])
im = cv2.imread(str(DATA / CAM / r["image"]))[:, :, ::-1]
box = predicted_box(cam, x, y, yaw)
bu, bv = h(cam, x, y, yaw)
half = 95
x0 = int(0.5 * (box[0] + box[2]) - half); y0 = int(box[3] - 1.3 * half)
x1, y1 = int(x0 + 2 * half), int(y0 + 1.85 * half)

fig, axes = plt.subplots(1, 4, figsize=(18.6, 5.6), constrained_layout=True)
c, s = math.cos(yaw), math.sin(yaw)
world = VISUAL_HULL @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + np.array([x, y, 0.0])
incam = (world - cam.cam_pos) @ cam.R.T
ahead = incam[:, 2] > 1e-6
uv = (cam.K @ incam[ahead].T).T
uv = uv[:, :2] / uv[:, 2:3]

ax = axes[0]
ax.imshow(im[y0:y1, x0:x1])
ax.scatter(uv[:, 0] - x0, uv[:, 1] - y0, s=5, color=D.ROBOT, alpha=0.55, lw=0, zorder=4)
ax.set_title("1.  project the robot's own shape", loc="left", fontsize=14.5, color=D.INK, pad=6)
ax.set_xlabel("its outline, put at a candidate pose", fontsize=12.5, color=D.INK2, labelpad=8)

ax = axes[1]
ax.imshow(im[y0:y1, x0:x1])
ax.add_patch(plt.Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor=D.ROBOT, lw=3.0, zorder=4))
ax.plot(bu - x0, bv - y0, "o", ms=13, color=D.ROBOT, mec="white", mew=2, zorder=6)
ax.set_title("2.  box it, take the bottom-centre", loc="left", fontsize=14.5, color=D.INK, pad=6)
ax.set_xlabel("exactly what the detector reports", fontsize=12.5, color=D.INK2, labelpad=8)

ax = axes[2]
det = None
for q in csv.DictReader(open(DATA / CAM / "detector_readings_halfopen_detect_20260825.csv")):
    if q["image"] == r["image"] and q["detected"] == "1":
        det = (float(q["x0"]), float(q["y0"]), float(q["x1"]), float(q["y1"])); break
ax.imshow(im[y0:y1, x0:x1])
ax.add_patch(plt.Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
                           fill=False, edgecolor=D.ROBOT, lw=2.6, ls=(0, (5, 3)), zorder=4))
if det:
    ax.add_patch(plt.Rectangle((det[0] - x0, det[1] - y0), det[2] - det[0], det[3] - det[1],
                               fill=False, edgecolor=D.GOOD, lw=3.0, zorder=5))
    ax.plot(0.5 * (det[0] + det[2]) - x0, det[3] - y0, "o", ms=13, color=D.GOOD,
            mec="white", mew=2, zorder=6)
ax.set_title("3.  compare with what YOLO drew", loc="left", fontsize=14.5, color=D.INK, pad=6)
ax.set_xlabel("the disagreement, in pixels", fontsize=12.5, color=D.GOOD, labelpad=8)
for ax in axes[:3]:
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_edgecolor("#d5d4cf")

# --- 4: what the disagreement is worth on the floor -------------------------------
ax = axes[3]
from observation import jacobian  # noqa: E402
J = jacobian(cam, x, y, yaw)
Ji = np.linalg.inv(J)
dz = (np.array([0.5 * (det[0] + det[2]), det[3]]) - np.array([bu, bv])) if det else np.array([2.0, 2.0])
shift = (Ji @ dz) * 100.0
ax.axhline(0, color="#e8e7e2", lw=1); ax.axvline(0, color="#e8e7e2", lw=1)
ax.plot(0, 0, "o", ms=15, color=D.ROBOT, mec="white", mew=2, zorder=6)
ax.annotate("", xy=(shift[0], shift[1]), xytext=(0, 0), zorder=5,
            arrowprops=dict(arrowstyle="-|>", color=D.GOOD, lw=3.4, shrinkA=9, shrinkB=6))
ax.plot(shift[0], shift[1], "*", ms=24, color=D.GOOD, mec="white", mew=1.5, zorder=7)
ax.text(0, -0.55, "where the robot\nthought it was", ha="center", va="top",
        fontsize=12, color=D.ROBOT, fontweight="bold")
ax.text(shift[0], shift[1] + 0.45, "where it moves to", ha="center", va="bottom",
        fontsize=12, color=D.GOOD, fontweight="bold")
lim = max(1.6, np.abs(shift).max() * 2.1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.grid(True, color="#f0efea", lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("centimetres", fontsize=12.5, labelpad=8)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
scale = np.linalg.norm(Ji[:, 1]) * 100
ax.set_title("4.  move the guess by what it is worth", loc="left", fontsize=14.5, color=D.INK, pad=6)
gap = np.linalg.norm(gap_vector(cam, x, y, yaw))
fig.suptitle("Do not convert the pixel into a position — predict the pixel from a position",
             x=0.006, ha="left", fontsize=19, color=D.INK)
fig.text(0.006, -0.10,
         f"The measured pixel is never turned into a position -- that would need the heading, which one box cannot give.\n"
         f"What IS turned into centimetres is the DISAGREEMENT between the two boxes, and only that.  Here one pixel of\n"
         f"disagreement is worth about {1/max(np.linalg.norm(J[:, 0]), 1e-9)*100:.1f} cm on the floor, so the guess moves "
         f"{np.linalg.norm(shift):.1f} cm and the loop repeats.\n"
         f"The {gap:.0f} cm gap never appears anywhere: it is present on both sides and cancels.",
         fontsize=12.5, color=D.INK2)
fig.savefig(OUT / "04_predict_dont_convert.png", dpi=175, bbox_inches="tight")
print(f"04: gap at this sighting {gap:.1f} cm")
