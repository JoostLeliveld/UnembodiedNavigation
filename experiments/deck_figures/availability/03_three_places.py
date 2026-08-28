"""Slide 7: what 'availability' means, in three real camera frames."""
import csv, sys, collections
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models  # noqa: E402
from observation import predicted_box  # noqa: E402

CAM = "camera_E"
DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
OUT = D.REPO / "logs/studies/deck_figures/availability"; OUT.mkdir(parents=True, exist_ok=True)
cams = camera_models(DATA)
det = {}
for r in csv.DictReader(open(DATA / CAM / "detector_readings_halfopen_detect_20260825.csv")):
    det[r["image"]] = r

per = collections.defaultdict(list)
for r in csv.DictReader(open(D.REPO / "logs/studies/measurement_commissioning/availability.csv")):
    if r["camera"] != CAM: continue
    per[(float(r["x"]), float(r["y"]))].append((float(r["yaw"]), int(r["line_of_sight"]),
                                                int(r["usable"]), r["image"]))

EXAMPLES = [((2.0062499999999996, -6.122222222222222), "The robot is in the open"),
            ((-3.34375, -4.188888888888888), "A rack hides part of it"),
            ((5.349999999999998, -1.6111111111111107), "It is almost entirely behind goods")]

fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.4), constrained_layout=True)
for ax, (pos, caption) in zip(axes, EXAMPLES):
    rows = sorted(per[pos])
    usable = sum(r[2] for r in rows)
    yaw, _los, _u, img = next(r for r in rows if r[3])
    im = cv2.imread(str(DATA / CAM / img))[:, :, ::-1]
    box = predicted_box(cams[CAM], pos[0], pos[1], yaw)
    cu, cv_ = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    half = 150.0        # the same crop everywhere, so the panels compare
    x0, x1 = int(max(cu - half, 0)), int(min(cu + half, im.shape[1]))
    y0, y1 = int(max(cv_ - half, 0)), int(min(cv_ + half, im.shape[0]))
    ax.imshow(im[y0:y1, x0:x1])
    ax.add_patch(plt.Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
                               fill=False, edgecolor="white", lw=2.6, ls=(0, (4, 3)), zorder=4))
    ax.annotate("where the robot is", xy=((box[0] + box[2]) / 2 - x0, box[3] - y0),
                xytext=((box[0] + box[2]) / 2 - x0, box[3] - y0 + 62), ha="center",
                fontsize=11.5, color="white", zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0, shrinkA=2, shrinkB=3))
    d = det.get(img)
    if d and d["detected"] == "1":
        bx = [float(d["x0"]) - x0, float(d["y0"]) - y0,
              float(d["x1"]) - float(d["x0"]), float(d["y1"]) - float(d["y0"])]
        ax.add_patch(plt.Rectangle(bx[:2], bx[2], bx[3], fill=False,
                                   edgecolor=D.GOOD if usable > 0 else D.BAD, lw=3.2, zorder=6))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#d5d4cf")
    col = D.GOOD if usable >= 5 else (D.BAD if usable == 0 else "#c98500")
    ax.set_title(caption, loc="left", fontsize=15, color=D.INK, pad=8)
    ax.set_xlabel(f"usable from this camera at {usable} of the 6 headings tried",
                  fontsize=13.5, color=col, labelpad=10)
fig.suptitle("The same camera, three places on the floor", x=0.006, ha="left",
             fontsize=19, color=D.INK)
fig.text(0.006, -0.035,
         "Camera E, three floor positions about 10 m away.  The dashed white box is where the robot actually is; "
         "the coloured box is what the detector returned.  Repeat this everywhere and the result is the availability map.",
         fontsize=12, color=D.INK2)
fig.savefig(OUT / "03_three_places.png", dpi=170, bbox_inches="tight")
print("wrote 03_three_places.png")
for pos, cap in EXAMPLES:
    rows = per[pos]
    print(f"  {cap}: usable {sum(r[2] for r in rows)}/6, line of sight {sum(r[1] for r in rows)}/6")
