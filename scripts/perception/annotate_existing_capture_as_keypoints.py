#!/usr/bin/env python3
"""Generate YOLO pose labels from an existing segmentation capture dataset.

Takes the ground-truth (x, y, theta) poses from samples.csv and projects
front/rear keypoints analytically — no new Gazebo capture needed.

Usage:
    python3 scripts/perception/annotate_existing_capture_as_keypoints.py \
        --capture logs/visibility_comparison/current_capture \
        --out     logs/keypoint_dataset_occ_light_v1 \
        --world   warehouse_occ_light.world.sdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
from perception.core.pose_keypoints import DEFAULT_MARKER_GEOMETRY, KPT_VIS_VISIBLE, KPT_VIS_OCCLUDED
from unav_common.camera_model import ObliqueCameraModel


def _project_world_point(camera: ObliqueCameraModel, pt: np.ndarray):
    cam_pt = camera.R @ (pt - camera.t)
    if cam_pt[2] <= 0.0:
        return None
    pixel_h = camera.K @ cam_pt
    return np.array([pixel_h[0] / pixel_h[2], pixel_h[1] / pixel_h[2]], dtype=float)


def _marker_world_pos(x, y, yaw, robot_z, offset_x):
    c, s = math.cos(yaw), math.sin(yaw)
    geom = DEFAULT_MARKER_GEOMETRY
    wz = robot_z + geom.base_joint_z + geom.marker_z_in_base_link
    return np.array([x + c * offset_x, y + s * offset_x, wz], dtype=float)


def _project_aabb(camera, x, y, yaw, robot_z, box_l=0.16, box_w=0.16, box_h=0.20, img_w=1280, img_h=720):
    hx, hy, hz = box_l / 2, box_w / 2, box_h
    local = np.array([[-hx,-hy,0],[-hx,hy,0],[hx,-hy,0],[hx,hy,0],
                      [-hx,-hy,hz],[-hx,hy,hz],[hx,-hy,hz],[hx,hy,hz]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    world = (rot @ local.T).T
    world[:, 0] += x; world[:, 1] += y
    pts = []
    for corner in world:
        uv = _project_world_point(camera, corner)
        if uv is None:
            return None
        pts.append(uv)
    pts = np.array(pts, dtype=float)
    x0 = np.clip(pts[:,0].min(), 0, img_w-1)
    y0 = np.clip(pts[:,1].min(), 0, img_h-1)
    x1 = np.clip(pts[:,0].max(), 0, img_w-1)
    y1 = np.clip(pts[:,1].max(), 0, img_h-1)
    if x1 <= x0 or y1 <= y0:
        return None
    min_px = min(x1-x0, y1-y0)
    return (x0, y0, x1, y1), min_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True, help='Path to existing segmentation capture dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--world', default='warehouse_occ_light.world.sdf')
    ap.add_argument('--world-profiles', default=str(REPO_ROOT / 'src/experiments/config/world_profiles.yaml'))
    ap.add_argument('--robot-z', type=float, default=0.05)
    ap.add_argument('--min-bbox-px', type=float, default=10.0)
    ap.add_argument('--val-fraction', type=float, default=0.20)
    ap.add_argument('--preview-count', type=int, default=20)
    args = ap.parse_args()

    capture_dir = Path(args.capture)
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f'Output already exists: {out_dir}')

    # Load camera model
    profile, intrinsics, world_path, camera_pose = load_profile(args.world_profiles, args.world)
    cam_pos = camera_pose[:3]
    look_at = compute_look_at_from_pose(cam_pos, camera_pose[3], camera_pose[4], camera_pose[5])
    img_w = int(intrinsics['img_width'])
    img_h = int(intrinsics['img_height'])
    camera = ObliqueCameraModel(
        cam_pos=cam_pos, look_at=look_at, img_width=img_w, img_height=img_h,
        fov_h_rad=float(intrinsics['fov_h_rad']),
    )

    # Load poses
    samples_csv = capture_dir / 'samples.csv'
    rows = list(csv.DictReader(open(samples_csv)))
    print(f'Loaded {len(rows)} poses from {samples_csv}')

    records = []
    skipped = 0
    for row in rows:
        x, y, yaw = float(row['x']), float(row['y']), float(row['theta'])
        img_rel = row['image_path']  # e.g. images/000022_xy0005_h02.jpg
        img_path = capture_dir / img_rel

        if not img_path.exists():
            skipped += 1
            continue

        # Project bounding box
        result = _project_aabb(camera, x, y, yaw, args.robot_z, img_w=img_w, img_h=img_h)
        if result is None or result[1] < args.min_bbox_px:
            skipped += 1
            continue
        (bx0, by0, bx1, by1), _ = result

        # Project front keypoint (red body)
        front_world = _marker_world_pos(x, y, yaw, args.robot_z, DEFAULT_MARKER_GEOMETRY.front_x)
        rear_world  = _marker_world_pos(x, y, yaw, args.robot_z, DEFAULT_MARKER_GEOMETRY.rear_x)

        front_uv = _project_world_point(camera, front_world)
        rear_uv  = _project_world_point(camera, rear_world)

        front_vis = KPT_VIS_VISIBLE if (front_uv is not None and
                    0 <= front_uv[0] < img_w and 0 <= front_uv[1] < img_h) else KPT_VIS_OCCLUDED
        rear_vis  = KPT_VIS_VISIBLE if (rear_uv is not None and
                    0 <= rear_uv[0] < img_w and 0 <= rear_uv[1] < img_h) else KPT_VIS_OCCLUDED

        if front_vis == KPT_VIS_OCCLUDED and rear_vis == KPT_VIS_OCCLUDED:
            skipped += 1
            continue

        if front_uv is None: front_uv = np.array([0.0, 0.0])
        if rear_uv is None:  rear_uv  = np.array([0.0, 0.0])

        # YOLO format: class cx cy w h | kpt_x kpt_y kpt_vis (normalized)
        cx = ((bx0 + bx1) / 2) / img_w
        cy = ((by0 + by1) / 2) / img_h
        bw = (bx1 - bx0) / img_w
        bh = (by1 - by0) / img_h

        records.append({
            'img_path': img_path,
            'img_rel': img_rel,
            'label': f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                     f"{front_uv[0]/img_w:.6f} {front_uv[1]/img_h:.6f} {float(front_vis):.1f} "
                     f"{rear_uv[0]/img_w:.6f} {rear_uv[1]/img_h:.6f} {float(rear_vis):.1f}",
            'separation_px': float(np.hypot(front_uv[0]-rear_uv[0], front_uv[1]-rear_uv[1])),
            'x': x, 'y': y,
        })

    print(f'Annotated: {len(records)}, skipped: {skipped}')

    # Train/val split — spatial block split
    random.seed(42)
    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_fraction))
    val_set = set(i for i in range(0, len(records), max(1, len(records)//n_val))[:n_val])

    # Write dataset
    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True)
        (out_dir / 'labels' / split).mkdir(parents=True)

    previews_written = 0
    (out_dir / 'previews').mkdir()

    for i, rec in enumerate(records):
        split = 'val' if i in val_set else 'train'
        stem = rec['img_path'].stem
        dst_img = out_dir / 'images' / split / rec['img_path'].name
        dst_lbl = out_dir / 'labels' / split / (stem + '.txt')
        shutil.copy2(rec['img_path'], dst_img)
        dst_lbl.write_text(rec['label'] + '\n')

        # Write preview
        if previews_written < args.preview_count:
            img = cv2.imread(str(rec['img_path']))
            if img is not None:
                lbl = rec['label'].split()
                # Parse keypoints back
                fu, fv = float(lbl[5])*img_w, float(lbl[6])*img_h
                ru, rv = float(lbl[8])*img_w, float(lbl[9])*img_h
                cv2.circle(img, (int(fu), int(fv)), 5, (0, 0, 255), -1)   # red = front
                cv2.circle(img, (int(ru), int(rv)), 5, (255, 0, 0), -1)   # blue = rear
                cx_px = int(float(lbl[1]) * img_w)
                cy_px = int(float(lbl[2]) * img_h)
                bw_px = int(float(lbl[3]) * img_w)
                bh_px = int(float(lbl[4]) * img_h)
                cv2.rectangle(img, (cx_px-bw_px//2, cy_px-bh_px//2),
                              (cx_px+bw_px//2, cy_px+bh_px//2), (0, 255, 0), 1)
                cv2.putText(img, f"sep={rec['separation_px']:.1f}px",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imwrite(str(out_dir / 'previews' / f'{i:04d}_{stem}.jpg'), img)
                previews_written += 1

    # data.yaml
    (out_dir / 'data.yaml').write_text(yaml.dump({
        'path': str(out_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'robot'},
        'kpt_shape': [2, 3],
        'flip_idx': [1, 0],  # front/rear swap when horizontally flipped
    }))

    sep_vals = [r['separation_px'] for r in records]
    print(f'Train: {len(records)-n_val}  Val: {n_val}')
    print(f'Keypoint separation — mean: {np.mean(sep_vals):.1f}px  min: {np.min(sep_vals):.1f}px  max: {np.max(sep_vals):.1f}px')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
