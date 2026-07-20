#!/usr/bin/env python3
"""Verify tracking + bounding boxes on real frames, on a solid footing.

For a LOW- and a HIGH-error frame:
  - run YOLO live on that exact frame  -> DETECTED box (red), guaranteed to match the image;
  - project the robot's 3D model (cylinder r=0.09, h=0.19) at the truth interpolated
    to the frame's CAPTURE time -> EXPECTED box (green dashed);
  - project rack R3 footprint (cyan) for spatial reference;
  - metric = | pixel_to_world(detected box-bottom) - truth@capture |.

Params come from load_profile(warehouse_aws) (no hardcoding). world_to_pixel does
full 3D projection (z respected).
"""
import csv, glob, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

ROOT = "/home/joostleliveld/Thesis/UnembodiedNavigation"
RUN = sorted(glob.glob(f"{ROOT}/logs/visibility_comparison/aws_f31b1_v4_b5c2_debug/b5_a4_apron_to_a2_low/C2/seed*/experiment_*"))[0]
FRAMES = "/tmp/b5_c2_debug_frames"


def f(r, k):
    try:
        v = float(r[k]); return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def main():
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
    from ultralytics import YOLO
    _p, intr, _w, cp = load_profile(f"{ROOT}/src/experiments/config/world_profiles.yaml", "warehouse_aws.world.sdf")
    cam = ObliqueCameraModel(cam_pos=[cp[0], cp[1], cp[2]],
                             look_at=compute_look_at_from_pose([cp[0], cp[1], cp[2]], cp[3], cp[4], cp[5]),
                             img_width=intr["img_width"], img_height=intr["img_height"], fov_h_rad=intr["fov_h_rad"])

    def w2p(x, y, z=0.0):
        u, v, _ = cam.world_to_pixel(x, y, z); return u, v

    exp = list(csv.DictReader(open(f"{RUN}/experiment.csv")))
    tr = np.array([[f(r, "stamp"), f(r, "truth_x"), f(r, "truth_y")] for r in exp
                   if math.isfinite(f(r, "stamp")) and math.isfinite(f(r, "truth_x"))])

    def truth_at(t):
        return float(np.interp(t, tr[:, 0], tr[:, 1])), float(np.interp(t, tr[:, 0], tr[:, 2]))

    per = [r for r in csv.DictReader(open(f"{RUN}/perception.csv"))
           if f(r, "yolo_detected_after_threshold") >= 0.5 and math.isfinite(f(r, "localization_error_calibrated_m"))]
    frames = {round(float(os.path.splitext(os.path.basename(p))[0].split("_")[-1]), 3): p
              for p in glob.glob(f"{FRAMES}/raw/*.png")}
    fk = sorted(frames)

    def fr_at(st):
        k = min(fk, key=lambda j: abs(j - st))
        return (frames[k], k) if abs(k - st) < 0.2 else (None, None)

    def expected_box(tx, ty):
        pts = [w2p(tx + 0.09 * math.cos(a), ty + 0.09 * math.sin(a), z)
               for a in np.linspace(0, 2 * math.pi, 16) for z in (0.0, 0.19)]
        us = [p[0] for p in pts]; vs = [p[1] for p in pts]
        return min(us), min(vs), max(us), max(vs)

    model = YOLO(f"{ROOT}/logs/perception_models/warehouse_yolo_detector_v1/model.pt")
    per.sort(key=lambda r: f(r, "localization_error_calibrated_m"))
    picks = [("LOW-error (median)", per[len(per) // 2]), ("HIGH-error (worst)", per[-1])]
    r3 = [(-0.225, -0.8), (0.325, -0.8), (0.325, 1.25), (-0.225, 1.25), (-0.225, -0.8)]

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (lab, r) in zip(axs, picks):
        st = f(r, "diag_stamp")
        path, cap = fr_at(st)
        ax.imshow(mpimg.imread(path))
        tx, ty = truth_at(cap)
        res = model.predict(path, imgsz=960, conf=0.05, verbose=False, device=0)[0]
        derr = float("nan")
        if res.boxes is not None and len(res.boxes):
            b = res.boxes.xyxy.cpu().numpy(); c = res.boxes.conf.cpu().numpy(); i = int(c.argmax())
            dx0, dy0, dx1, dy1 = b[i]
            ax.add_patch(Rectangle((dx0, dy0), dx1 - dx0, dy1 - dy0, fill=False, edgecolor="red", lw=2.5))
            wx, wy = cam.pixel_to_world((dx0 + dx1) / 2, dy1)
            derr = math.hypot(wx - tx, wy - ty)
        ex0, ey0, ex1, ey1 = expected_box(tx, ty)
        ax.add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=False, edgecolor="lime", lw=2.5, ls="--"))
        px = [w2p(x, y, 0) for x, y in r3]
        ax.plot([p[0] for p in px], [p[1] for p in px], "-", color="cyan", lw=1.2)
        cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
        ax.set_xlim(cx - 130, cx + 130); ax.set_ylim(cy + 105, cy - 105)
        ax.set_title(f"{lab}\ncapture-truth ({tx:.2f},{ty:.2f}) | live-YOLO box-bottom->world err {derr:.2f} m | logged {f(r,'localization_error_calibrated_m'):.2f} m",
                     fontsize=10)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color="red", lw=2.5, label="DETECTED box (live YOLO on this frame)"),
                        Line2D([0], [0], color="lime", lw=2.5, ls="--", label="EXPECTED box (robot 3D model @ capture truth)"),
                        Line2D([0], [0], color="cyan", lw=1.2, label="rack R3 footprint (z=0)")],
               loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig("/tmp/verify_boxes.png", dpi=95, bbox_inches="tight")
    print("wrote /tmp/verify_boxes.png")


if __name__ == "__main__":
    main()
