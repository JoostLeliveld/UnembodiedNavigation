#!/usr/bin/env python3
"""Validate a YOLO detector's LOCALIZATION accuracy by image region.

Motivation: the visibility-aware experiment localizes the robot by projecting the
detected bbox-bottom-centre through the ground-plane homography. What matters for
the EKF is therefore not detection *rate* but the *world* position error of that
projected point. v3 (trained+inferred at imgsz 640) localizes well in the central
image band (~3 cm) but degrades badly at the grazing top-edge periphery where the
goal sits (~0.5 m, with ~25% misses), because the native-1280 robot shrinks to
~7 px at imgsz 640. This script quantifies that, per region, so v3 and v4 can be
compared apples-to-apples (each at its own training imgsz).

For every validation image it:
  - parses the GT segmentation polygon -> GT bbox (and its bottom-centre pixel),
  - runs the model, takes the highest-confidence box,
  - projects BOTH the predicted and GT bbox-bottom-centre to the ground plane,
  - reports the world distance between them (detector-attributable error) and
    recall, binned into: goal-pocket, top-edge, centre, and all.

Comparing predicted-vs-GT box-bottom (rather than vs the true robot pose) isolates
the detector's contribution: both share the identical projection geometry, so the
0.07 m bbox-bottom-vs-true-contact offset cancels.

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/perception/validate_detector_localization.py \
    --model logs/perception_models/warehouse_yolo_detector_v1/model.pt --imgsz 960
"""
import argparse, glob, math, os
import numpy as np

# Camera pose/intrinsics for warehouse_aws (SDF external_camera + world_profiles).
CAM_POS = [0.0, -5.5, 4.8]
CAM_RPY = [0.0, 0.92, 1.5708]
IMG_W, IMG_H, FOV_H = 1280, 720, 1.5708

# Image-space regions (pixels). The goal pocket is where Task A's goal projects and
# where v3 fails; top-edge = the grazing far field; centre = the well-localized band.
REGIONS = {
    "goal-pocket": lambda cu, cv: 620 <= cu <= 820 and 100 <= cv <= 210,
    "top-edge": lambda cu, cv: cv < 200,
    "centre": lambda cu, cv: 200 <= cv < 400,
    "all": lambda cu, cv: True,
}


def gt_bbox(label_line):
    p = label_line.split()
    pts = np.array(list(map(float, p[1:])), float).reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def stats(vals):
    if not vals:
        return "n=0"
    s = sorted(vals)
    return f"n={len(s):3d} med={s[len(s)//2]:.3f} p90={s[int(len(s)*0.9)]:.3f} max={max(s):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--dataset", default="logs/perception_datasets/warehouse_yolo_dataset_v1")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    from ultralytics import YOLO
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import compute_look_at_from_pose

    look_at = compute_look_at_from_pose(CAM_POS, *CAM_RPY)
    cam = ObliqueCameraModel(cam_pos=CAM_POS, look_at=look_at,
                             img_width=IMG_W, img_height=IMG_H, fov_h_rad=FOV_H)
    model = YOLO(args.model)

    label_files = sorted(glob.glob(f"{args.dataset}/labels/{args.split}/*.txt"))
    # region -> dict(world_err=[], detected=int, total=int)
    acc = {r: {"err": [], "det": 0, "tot": 0} for r in REGIONS}

    for lf in label_files:
        line = open(lf).readline()
        if len(line.split()) < 5:
            continue
        x0, y0, x1, y1 = gt_bbox(line)
        gcu, gcv = (x0 + x1) / 2 * IMG_W, (y0 + y1) / 2 * IMG_H        # GT centre (for binning)
        gbu, gbv = (x0 + x1) / 2 * IMG_W, y1 * IMG_H                    # GT bottom-centre
        gwx, gwy = cam.pixel_to_world(gbu, gbv)
        in_regions = [r for r, fn in REGIONS.items() if fn(gcu, gcv)]
        for r in in_regions:
            acc[r]["tot"] += 1

        base = os.path.splitext(os.path.basename(lf))[0]
        imgs = glob.glob(f"{args.dataset}/images/{args.split}/{base}.*")
        if not imgs:
            continue
        res = model.predict(imgs[0], imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        b = res.boxes.xyxy.cpu().numpy()
        c = res.boxes.conf.cpu().numpy()
        i = int(c.argmax())
        px0, py0, px1, py1 = b[i]
        pbu, pbv = (px0 + px1) / 2, py1                                  # predicted bottom-centre
        pwx, pwy = cam.pixel_to_world(pbu, pbv)
        werr = math.hypot(pwx - gwx, pwy - gwy)
        for r in in_regions:
            acc[r]["det"] += 1
            acc[r]["err"].append(werr)

    print(f"\nmodel={args.model}")
    print(f"imgsz={args.imgsz}  split={args.split}  conf={args.conf}")
    print(f"{'region':12s}  recall      world-localization error (m)")
    for r in ("goal-pocket", "top-edge", "centre", "all"):
        d = acc[r]
        rec = d["det"] / d["tot"] if d["tot"] else float("nan")
        print(f"{r:12s}  {d['det']:3d}/{d['tot']:<3d} ({100*rec:4.0f}%)  {stats(d['err'])}")


if __name__ == "__main__":
    main()
