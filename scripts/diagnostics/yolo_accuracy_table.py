#!/usr/bin/env python3
"""Stage 2 sanity check — YOLO static-pose accuracy table.

Joins a perception_targets.csv (detected vs oracle pixel at known poses) and
reports detection rate, pixel error, and BEV (world) error decomposed into
systematic bias vs noise. Back-projects both detected and oracle pixels through
the world's ObliqueCameraModel so the homography bias is separated from YOLO
pixel noise.

This is the controlled-grid analog of the on-the-fly decomposition we get from a
driving run; it isolates WHERE the localization error comes from.

Usage:
    python3 scripts/diagnostics/yolo_accuracy_table.py \
        --targets logs/visibility_comparison/<dir>/perception_targets.csv \
        --world warehouse_aws.world.sdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ("src/experiments", "src/unav_common"):
    sys.path.insert(0, str(REPO_ROOT / rel))

from experiments.core.world_profiles import load_profile, compute_look_at_from_pose  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

WORLD_PROFILES = REPO_ROOT / "src/experiments/config/world_profiles.yaml"


def build_camera(world: str) -> ObliqueCameraModel:
    profile, intrinsics, world_path, camera_pose = load_profile(str(WORLD_PROFILES), world)
    cam_pos = camera_pose[:3]
    look_at = compute_look_at_from_pose(cam_pos, camera_pose[3], camera_pose[4], camera_pose[5])
    return ObliqueCameraModel(
        cam_pos=cam_pos, look_at=look_at,
        img_width=int(intrinsics["img_width"]), img_height=int(intrinsics["img_height"]),
        fov_h_rad=float(intrinsics["fov_h_rad"]),
    )


def bev(camera, u, v):
    xy = camera.pixel_to_world(float(u), float(v))
    return (np.nan, np.nan) if xy is None else (xy[0], xy[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--y-calib", type=float, default=0.0,
                    help="y calibration offset applied to detected BEV (state-node value)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.targets)
    camera = build_camera(args.world)
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "logs/diagnostics/stage2_yolo" / Path(args.targets).parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    det = df[df["yolo_detected_after_threshold"] > 0.5].copy()
    det_rate = len(det) / max(n, 1)
    print(f"World: {args.world}   samples={n}   detection rate={det_rate:.1%}")

    if "oracle_visible" in df.columns:
        vis = df[df["oracle_visible"] > 0.5]
        if len(vis):
            dr_vis = (vis["yolo_detected_after_threshold"] > 0.5).mean()
            print(f"  detection rate where oracle-visible: {dr_vis:.1%} (n={len(vis)})")

    # Pixel error (detected vs oracle), detected & oracle-visible only
    m = det.copy()
    if "oracle_visible" in m.columns:
        m = m[m["oracle_visible"] > 0.5]
    m = m.dropna(subset=["yolo_bottom_u", "yolo_bottom_v", "oracle_bottom_u", "oracle_bottom_v"])
    pix_err = np.hypot(m["yolo_bottom_u"] - m["oracle_bottom_u"],
                       m["yolo_bottom_v"] - m["oracle_bottom_v"])
    print(f"\nPIXEL error (detected vs oracle), n={len(m)}:")
    print(f"  mean={pix_err.mean():.1f}px  p90={np.percentile(pix_err,90):.1f}px  max={pix_err.max():.1f}px")

    # BEV error: back-project both, compare detected-BEV to truth (x,y)
    det_xy = np.array([bev(camera, u, v) for u, v in zip(m["yolo_bottom_u"], m["yolo_bottom_v"])])
    ora_xy = np.array([bev(camera, u, v) for u, v in zip(m["oracle_bottom_u"], m["oracle_bottom_v"])])
    det_xy[:, 1] += args.y_calib  # state-node y calibration
    truth_xy = m[["x", "y"]].values

    det_err = np.hypot(det_xy[:, 0] - truth_xy[:, 0], det_xy[:, 1] - truth_xy[:, 1])
    ora_err = np.hypot(ora_xy[:, 0] - truth_xy[:, 0], ora_xy[:, 1] - truth_xy[:, 1])
    dbx = det_xy[:, 0] - truth_xy[:, 0]
    dby = det_xy[:, 1] - truth_xy[:, 1]
    obx = ora_xy[:, 0] - truth_xy[:, 0]
    oby = ora_xy[:, 1] - truth_xy[:, 1]

    print(f"\nBEV error (back-projected, y_calib={args.y_calib}):")
    print(f"  DETECTED vs truth: mean={np.nanmean(det_err):.3f}m p90={np.nanpercentile(det_err,90):.3f}m")
    print(f"     bias dx={np.nanmean(dbx):+.3f} dy={np.nanmean(dby):+.3f}  std dx={np.nanstd(dbx):.3f} dy={np.nanstd(dby):.3f}")
    print(f"  ORACLE-pixel vs truth (homography-only error, no YOLO noise):")
    print(f"     mean={np.nanmean(ora_err):.3f}m  bias dx={np.nanmean(obx):+.3f} dy={np.nanmean(oby):+.3f}")
    print(f"  => YOLO-pixel-noise contribution ≈ {np.nanmean(det_err) - np.nanmean(ora_err):+.3f}m on top of homography bias")

    # Heatmap of detected-BEV error over the floor
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), facecolor="white")
    sc = axes[0].scatter(truth_xy[:, 0], truth_xy[:, 1], c=det_err, cmap="viridis", s=30, vmin=0, vmax=0.6)
    plt.colorbar(sc, ax=axes[0], label="detected BEV err [m]")
    axes[0].set_title("Detected-BEV error over floor"); axes[0].set_aspect("equal", "box")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")

    axes[1].quiver(truth_xy[:, 0], truth_xy[:, 1], dbx, dby, det_err,
                   cmap="viridis", angles="xy", scale_units="xy", scale=1.0)
    axes[1].set_title("BEV error vectors (detected - truth)"); axes[1].set_aspect("equal", "box")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")

    axes[2].hist(det_err[~np.isnan(det_err)], bins=30, alpha=0.7, label="detected")
    axes[2].hist(ora_err[~np.isnan(ora_err)], bins=30, alpha=0.7, label="oracle-pixel")
    axes[2].axvline(0.25, color="r", ls="--", label="0.25 m"); axes[2].legend()
    axes[2].set_title("BEV error distribution"); axes[2].set_xlabel("BEV error [m]")

    png = out_dir / "yolo_accuracy_table.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {png}")


if __name__ == "__main__":
    main()
