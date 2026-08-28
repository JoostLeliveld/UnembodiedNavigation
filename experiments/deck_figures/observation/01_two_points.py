"""WHY 1: the box bottom-centre and the robot are not the same point."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2, matplotlib.pyplot as plt
from _common import CAMS, DATA, OUT, gap_vector, index
import style as D
from observation import h, predicted_box  # noqa: E402

PICKS = [("camera_E", 5.0), ("camera_A", 10.0), ("camera_D", 16.0)]
rows = [r for r in index() if r["image"]]
fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.2), constrained_layout=True)
for ax, (cam_id, want) in zip(axes, PICKS):
    cand = [r for r in rows if r["camera_id"] == cam_id]
    r = min(cand, key=lambda q: abs(float(q["camera_range_m"]) - want))
    cam = CAMS[cam_id]
    x, y, yaw = float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])
    im = cv2.imread(str(DATA / cam_id / r["image"]))[:, :, ::-1]
    box = predicted_box(cam, x, y, yaw)
    bu, bv = h(cam, x, y, yaw)
    cu, cvv = cam.world_to_pixel(x, y, 0.0)[:2]
    half = max(70.0, (box[2] - box[0]) * 1.9)
    x0 = int(np.clip(0.5 * (box[0] + box[2]) - half, 0, im.shape[1] - 2 * half))
    y0 = int(np.clip(box[3] - 1.35 * half, 0, im.shape[0] - 1.9 * half))
    x1, y1 = int(x0 + 2 * half), int(y0 + 1.9 * half)
    ax.imshow(im[y0:y1, x0:x1])
    ax.add_patch(plt.Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
                               fill=False, edgecolor="white", lw=2.4, zorder=4))
    ax.annotate("", xy=(cu - x0, cvv - y0), xytext=(bu - x0, bv - y0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=D.INK, lw=2.6, shrinkA=7, shrinkB=9))
    ax.plot(bu - x0, bv - y0, "o", ms=12, color=D.BAD, mec="white", mew=2, zorder=6)
    ax.plot(cu - x0, cvv - y0, "*", ms=21, color=D.GOOD, mec="white", mew=1.4, zorder=6)
    gap = np.linalg.norm(gap_vector(cam, x, y, yaw))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_edgecolor("#d5d4cf")
    ax.set_title(f"camera {cam_id[-1]}, {float(r['camera_range_m']):.0f} m away",
                 loc="left", fontsize=14.5, color=D.INK, pad=6)
    ax.set_xlabel(f"the two points are {gap:.0f} cm apart on the floor",
                  fontsize=13.5, color=D.BAD, labelpad=9)
fig.suptitle("The bottom of the box is the robot's nearest corner, not its middle",
             x=0.006, ha="left", fontsize=19, color=D.INK)
fig.text(0.006, -0.035, "Orange dot = the bottom-centre of the detector's box.  Green star = "
         "where the robot actually is.  The box's bottom edge is\nwhichever part of the robot "
         "is closest to the camera, and its horizontal middle is the middle of its widest part.  "
         "Neither is the robot.", fontsize=12.5, color=D.INK2)
fig.savefig(OUT / "01_two_points.png", dpi=175, bbox_inches="tight")
print("01: ok")
