#!/usr/bin/env python3
"""Capture simulator frames and auto-label visible-robot YOLO segmentation masks."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in ('scripts', 'src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from capture_yolo_dataset import YoloDatasetCapture, _default_arg_path, _sample_positions
from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
from unav_common.camera_model import ObliqueCameraModel


def _clip_bbox(bbox, img_w: int, img_h: int, pad_px: float):
    if bbox is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = int(max(math.floor(x0 - pad_px), 0))
    y0 = int(max(math.floor(y0 - pad_px), 0))
    x1 = int(min(math.ceil(x1 + pad_px), img_w - 1))
    y1 = int(min(math.ceil(y1 + pad_px), img_h - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _red_mask(crop_bgr: np.ndarray, *, morph_kernel_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel_size = max(int(morph_kernel_px), 1)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _polygon_area(pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _largest_visible_robot_polygon(
    image_bgr: np.ndarray,
    projected_bbox,
    *,
    roi_pad_px: float,
    morph_kernel_px: int,
    min_mask_area_px: float,
    polygon_epsilon_frac: float,
    max_polygon_points: int,
):
    img_h, img_w = image_bgr.shape[:2]
    roi = _clip_bbox(projected_bbox, img_w, img_h, roi_pad_px)
    if roi is None:
        return None, None, 0.0, 0
    x0, y0, x1, y1 = roi
    crop = image_bgr[y0:y1 + 1, x0:x1 + 1]
    mask = _red_mask(crop, morph_kernel_px=morph_kernel_px)
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, roi, 0.0, 0

    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    if area_px < float(min_mask_area_px):
        return None, roi, area_px, int(np.count_nonzero(mask))

    arc_len = float(cv2.arcLength(contour, closed=True))
    eps = max(float(polygon_epsilon_frac) * arc_len, 0.5)
    approx = cv2.approxPolyDP(contour, eps, closed=True)
    if approx.shape[0] < 3:
        approx = cv2.convexHull(contour)

    for _ in range(12):
        if int(max_polygon_points) <= 0 or approx.shape[0] <= int(max_polygon_points):
            break
        eps *= 1.35
        approx = cv2.approxPolyDP(contour, eps, closed=True)

    pts = approx.reshape(-1, 2).astype(float)
    pts[:, 0] += float(x0)
    pts[:, 1] += float(y0)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(img_w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(img_h - 1))

    if pts.shape[0] < 3 or _polygon_area(pts) < float(min_mask_area_px):
        return None, roi, area_px, int(np.count_nonzero(mask))
    return pts, roi, area_px, int(np.count_nonzero(mask))


def _write_seg_label(path: Path, polygon, img_w: int, img_h: int) -> None:
    if polygon is None:
        path.write_text('', encoding='utf-8')
        return
    values = ['0']
    for x, y in np.asarray(polygon, dtype=float):
        values.append(f'{np.clip(x / float(img_w), 0.0, 1.0):.8f}')
        values.append(f'{np.clip(y / float(img_h), 0.0, 1.0):.8f}')
    path.write_text(' '.join(values) + '\n', encoding='utf-8')


def _write_preview(path: Path, image_bgr: np.ndarray, projected_bbox, polygon, roi, mask_area_px: float) -> None:
    preview = image_bgr.copy()
    if projected_bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in projected_bbox]
        cv2.rectangle(preview, (x0, y0), (x1, y1), (80, 80, 80), 1)
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 160, 0), 1)
    if polygon is not None:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        preview = cv2.addWeighted(overlay, 0.28, preview, 0.72, 0.0)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        label = f'robot mask area={mask_area_px:.0f}'
    else:
        label = f'empty mask area={mask_area_px:.0f}'
    cv2.putText(preview, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture auto-labeled YOLO segmentation dataset from Gazebo.')
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=_default_arg_path('src/experiments/config/world_profiles.yaml'))
    parser.add_argument('--output-root', default=_default_arg_path('logs/yolo_seg_datasets'))
    parser.add_argument('--sample-nx', type=int, default=15)
    parser.add_argument('--sample-ny', type=int, default=15)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--wall-margin-m', type=float, default=0.45)
    parser.add_argument('--yaw-rad', type=float, default=0.0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--robot-half-x', type=float, default=0.11)
    parser.add_argument('--robot-half-y', type=float, default=0.11)
    parser.add_argument('--robot-height-m', type=float, default=0.20)
    parser.add_argument('--bbox-pad-px', type=float, default=8.0)
    parser.add_argument('--seg-roi-pad-px', type=float, default=8.0)
    parser.add_argument('--seg-morph-kernel-px', type=int, default=3)
    parser.add_argument('--min-mask-area-px', type=float, default=20.0)
    parser.add_argument('--polygon-epsilon-frac', type=float, default=0.015)
    parser.add_argument('--max-polygon-points', type=int, default=64)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ready-timeout-s', type=float, default=20.0)
    parser.add_argument('--settle-s', type=float, default=0.25)
    parser.add_argument('--image-timeout-s', type=float, default=1.0)
    parser.add_argument('--service-timeout-s', type=float, default=20.0)
    parser.add_argument('--preview-count', type=int, default=80)
    args = parser.parse_args()

    profile, intrinsics, _world_path, camera_pose = load_profile(args.world_profiles, args.world)
    vis = dict(profile.get('visibility_defaults') or {})
    cam_pos = [camera_pose[0], camera_pose[1], camera_pose[2]]
    look_at = compute_look_at_from_pose(cam_pos, camera_pose[3], camera_pose[4], camera_pose[5])
    camera_model = ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=int(intrinsics['img_width']),
        img_height=int(intrinsics['img_height']),
        fov_h_rad=float(intrinsics['fov_h_rad']),
    )

    out_dir = Path(args.output_root).expanduser().resolve() / f'yolo_seg_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    positions = _sample_positions(
        vis,
        sample_nx=int(args.sample_nx),
        sample_ny=int(args.sample_ny),
        wall_margin_m=float(args.wall_margin_m),
        sample_limit=int(args.sample_limit),
    )

    rclpy.init()
    node = YoloDatasetCapture(
        world_name=str(profile['world_name']),
        camera_model=camera_model,
        robot_z=float(args.robot_z),
        yaw_rad=float(args.yaw_rad),
        service_timeout_s=float(args.service_timeout_s),
    )
    rows = []
    try:
        node.wait_for_ready(float(args.ready_timeout_s))
        sample_idx = 0
        for repeat_idx in range(max(int(args.repeats), 1)):
            for x, y in positions:
                last_stamp = node.latest_stamp
                node.set_robot_pose(x, y)
                settle_until = time.monotonic() + max(float(args.settle_s), 0.0)
                while rclpy.ok() and time.monotonic() < settle_until:
                    rclpy.spin_once(node, timeout_sec=0.05)
                image_bgr, stamp = node.wait_for_image_after(last_stamp, float(args.image_timeout_s))
                projected_bbox = node.project_robot_bbox(
                    x,
                    y,
                    half_x=float(args.robot_half_x),
                    half_y=float(args.robot_half_y),
                    height_m=float(args.robot_height_m),
                    pad_px=float(args.bbox_pad_px),
                )
                polygon, roi, mask_area_px, visible_red_px = _largest_visible_robot_polygon(
                    image_bgr,
                    projected_bbox,
                    roi_pad_px=float(args.seg_roi_pad_px),
                    morph_kernel_px=int(args.seg_morph_kernel_px),
                    min_mask_area_px=float(args.min_mask_area_px),
                    polygon_epsilon_frac=float(args.polygon_epsilon_frac),
                    max_polygon_points=int(args.max_polygon_points),
                )
                split = 'val' if float(rng.random()) < float(args.val_fraction) else 'train'
                stem = f'{sample_idx:06d}'
                image_path = out_dir / 'images' / split / f'{stem}.jpg'
                label_path = out_dir / 'labels' / split / f'{stem}.txt'
                cv2.imwrite(str(image_path), image_bgr)
                _write_seg_label(label_path, polygon, image_bgr.shape[1], image_bgr.shape[0])
                preview_path = ''
                if sample_idx < max(int(args.preview_count), 0):
                    preview = out_dir / 'previews' / f'{stem}_{split}.jpg'
                    _write_preview(preview, image_bgr, projected_bbox, polygon, roi, mask_area_px)
                    preview_path = str(preview.relative_to(out_dir))
                rows.append({
                    'sample_idx': sample_idx,
                    'repeat_idx': repeat_idx,
                    'split': split,
                    'x': float(x),
                    'y': float(y),
                    'yaw': float(args.yaw_rad),
                    'stamp': float(stamp),
                    'image': str(image_path.relative_to(out_dir)),
                    'label': str(label_path.relative_to(out_dir)),
                    'projected_bbox': list(projected_bbox) if projected_bbox is not None else None,
                    'roi': list(roi) if roi is not None else None,
                    'has_mask': polygon is not None,
                    'polygon_points': int(len(polygon)) if polygon is not None else 0,
                    'mask_area_px': float(mask_area_px),
                    'visible_red_px': int(visible_red_px),
                    'preview': preview_path,
                })
                sample_idx += 1
                if sample_idx % 25 == 0:
                    node.get_logger().info(f'Captured {sample_idx} YOLO segmentation dataset frames')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    data_yaml = {
        'path': str(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'robot'},
        'task': 'segment',
    }
    (out_dir / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8')
    (out_dir / 'capture_manifest.json').write_text(json.dumps({
        'world': args.world,
        'world_name': profile['world_name'],
        'n_frames': len(rows),
        'n_masks': sum(1 for row in rows if row['has_mask']),
        'camera': {
            'cam_pos': cam_pos,
            'look_at': look_at,
            'img_width': int(intrinsics['img_width']),
            'img_height': int(intrinsics['img_height']),
            'fov_h_rad': float(intrinsics['fov_h_rad']),
        },
        'label_model': {
            'method': 'projected_robot_bbox_roi_plus_visible_red_mask',
            'robot_half_x': float(args.robot_half_x),
            'robot_half_y': float(args.robot_half_y),
            'robot_height_m': float(args.robot_height_m),
            'bbox_pad_px': float(args.bbox_pad_px),
            'seg_roi_pad_px': float(args.seg_roi_pad_px),
            'seg_morph_kernel_px': int(args.seg_morph_kernel_px),
            'min_mask_area_px': float(args.min_mask_area_px),
            'polygon_epsilon_frac': float(args.polygon_epsilon_frac),
            'max_polygon_points': int(args.max_polygon_points),
        },
        'samples': rows,
    }, indent=2), encoding='utf-8')
    print(f'Wrote YOLO segmentation dataset to {out_dir}')
    print(f'Wrote {sum(1 for row in rows if row["has_mask"])} non-empty mask labels and {sum(1 for row in rows if not row["has_mask"])} empty labels')
    print(f'Train with: yolo segment train model=yolo11n-seg.pt data={out_dir / "data.yaml"} imgsz=640')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
