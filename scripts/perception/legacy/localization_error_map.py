#!/usr/bin/env python3
"""Spatial localization-error map for two detector checkpoints.

For every dataset image we know the robot's GT box (segmentation label). We map
its bottom-centre to a world (x,y) via the ground homography = the position. We
run each detector, map ITS box-bottom to world, and the distance to the GT-box
world is the detector-attributable localization error at that position (the
homography cancels, so this isolates the detector). We then paint two heatmaps
of that error over the workspace, so you can see WHERE each model is accurate vs
poor (and a third panel = candidate improvement over reference).

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/perception/localization_error_map.py
"""
import csv, glob, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DS = "logs/perception_datasets/warehouse_yolo_dataset_v1"
REFERENCE_MODEL = "logs/perception_models/warehouse_yolo_detector_reference/model.pt"
CANDIDATE_MODEL = "logs/perception_models/warehouse_yolo_detector_v1/model.pt"
GPZ = "paper_artifacts/gp/aws_gp_v3/yolo_score_raw_gp.npz"
W, H = 1280, 720
OUT = "/tmp/localization_error_map.png"


def main():
    from ultralytics import YOLO
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import compute_look_at_from_pose
    cam = ObliqueCameraModel(cam_pos=[0, -5.5, 4.8],
                             look_at=compute_look_at_from_pose([0, -5.5, 4.8], 0, 0.92, 1.5708),
                             img_width=W, img_height=H, fov_h_rad=1.5708)

    def box_bottom_world(x0, y0, x1, y1):  # normalized box -> bottom-centre -> world
        u = (x0 + x1) / 2 * W
        v = y1 * H
        return cam.pixel_to_world(u, v)

    imgs = sorted(glob.glob(f"{DS}/images/train/*.*") + glob.glob(f"{DS}/images/val/*.*"))
    # GT box-bottom world per image (the position + reference)
    gt = {}
    for lf in glob.glob(f"{DS}/labels/train/*.txt") + glob.glob(f"{DS}/labels/val/*.txt"):
        line = open(lf).readline()
        p = line.split()
        if len(p) < 5:
            continue
        pts = np.array(list(map(float, p[1:])), float).reshape(-1, 2)
        x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        gt[os.path.splitext(os.path.basename(lf))[0]] = (box_bottom_world(x0, y0, x1, y1),
                                                         ((x0 + x1) / 2 * W, y1 * H))

    rows = []  # (gx, gy, err_reference, err_candidate, det_reference, det_candidate)
    m_ref, m_candidate = YOLO(REFERENCE_MODEL), YOLO(CANDIDATE_MODEL)
    for i, img in enumerate(imgs):
        base = os.path.splitext(os.path.basename(img))[0]
        if base not in gt:
            continue
        (gx, gy), _ = gt[base]
        rec = [gx, gy, math.nan, math.nan, 0, 0]
        for j, (m, sz) in enumerate(((m_ref, 960), (m_candidate, 960))):
            r = m.predict(img, imgsz=sz, conf=0.05, verbose=False, device=0)[0]
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                c = r.boxes.conf.cpu().numpy()
                k = int(c.argmax())
                dx0, dy0, dx1, dy1 = b[k]
                dw = cam.pixel_to_world((dx0 + dx1) / 2, dy1)
                rec[2 + j] = math.hypot(dw[0] - gx, dw[1] - gy)
                rec[4 + j] = 1
        rows.append(rec)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(imgs)}")
    R = np.array(rows, float)
    print(f"images scored: {len(R)}")

    # GP background + racks
    gp = np.load(GPZ, allow_pickle=True)
    import json
    racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
             for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]

    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (col, title) in zip(axs, [(2, "reference detector"), (3, "candidate detector"),
                                       (None, "improvement (reference err - candidate err)")]):
        for (x0, x1, y0, y1) in racks:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.8", edgecolor="0.4", lw=0.5, zorder=0))
        det = R[(R[:, 4] >= 0.5) & (R[:, 5] >= 0.5)]  # detected by both, for fair compare
        if col is None:
            c = det[:, 2] - det[:, 3]
            sc = ax.scatter(det[:, 0], det[:, 1], c=c, cmap="RdBu", vmin=-0.3, vmax=0.3, s=14, zorder=2)
            fig.colorbar(sc, ax=ax, label="error reduced by candidate (m)  [blue=candidate better]")
        else:
            c = det[:, col]
            sc = ax.scatter(det[:, 0], det[:, 1], c=c, cmap="inferno_r", vmin=0, vmax=0.6, s=14, zorder=2)
            fig.colorbar(sc, ax=ax, label="localization error (m)")
            med = np.nanmedian(c)
            ax.set_title(f"{title}\nmedian {med:.3f} m, >0.3m: {100*np.mean(c>0.3):.0f}%", fontsize=11)
            continue
        ax.set_title(title, fontsize=11)
    for ax in axs:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
        ax.set_xlim(-5.7, 4.0); ax.set_ylim(-3.0, 5.0)
    # mark the camera + the task goals
    for ax in axs:
        ax.scatter([0], [-5.5], marker="^", s=80, c="cyan", edgecolor="k", zorder=5, clip_on=False)
    fig.suptitle("Detector localization error by position (box-bottom -> world; homography cancels)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=85, bbox_inches="tight")
    print(f"wrote {OUT}")
    # region summary: west (x<-0.5) vs centre/east
    det = R[(R[:, 4] >= 0.5) & (R[:, 5] >= 0.5)]
    for lab, mask in (("west x<-0.5", det[:, 0] < -0.5), ("centre/east x>=-0.5", det[:, 0] >= -0.5)):
        d = det[mask]
        if len(d):
            print(f"  {lab}: n={len(d)}  reference med={np.median(d[:,2]):.3f}  candidate med={np.median(d[:,3]):.3f}")


if __name__ == "__main__":
    main()
