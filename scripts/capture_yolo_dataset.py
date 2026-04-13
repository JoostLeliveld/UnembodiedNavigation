#!/usr/bin/env python3
"""Capture simulator frames and auto-label a one-class YOLO robot dataset."""

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
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in ('src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
from perception.core.ros_image import image_msg_to_bgr8
from unav_common.camera_model import ObliqueCameraModel


def _default_arg_path(relative: str) -> str:
    return str((REPO_ROOT / relative).resolve())


def _sample_positions(vis: dict, *, sample_nx: int, sample_ny: int, wall_margin_m: float, sample_limit: int):
    xmin = float(vis.get('visibility_map_min_x', -6.0)) + float(max(wall_margin_m, 0.0))
    xmax = float(vis.get('visibility_map_max_x', 6.0)) - float(max(wall_margin_m, 0.0))
    ymin = float(vis.get('visibility_map_min_y', -6.0)) + float(max(wall_margin_m, 0.0))
    ymax = float(vis.get('visibility_map_max_y', 6.0)) - float(max(wall_margin_m, 0.0))
    xs = np.linspace(xmin, xmax, max(int(sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(sample_ny), 1))
    positions = []
    for row_idx, y in enumerate(ys):
        row_xs = xs if row_idx % 2 == 0 else xs[::-1]
        for x in row_xs:
            positions.append((float(x), float(y)))
    if int(sample_limit) > 0:
        positions = positions[: int(sample_limit)]
    return positions


class YoloDatasetCapture(Node):
    def __init__(self, *, world_name: str, camera_model: ObliqueCameraModel, robot_z: float, yaw_rad: float, service_timeout_s: float):
        super().__init__('yolo_dataset_capture')
        self.world_name = str(world_name)
        self.camera_model = camera_model
        self.robot_z = float(robot_z)
        self.yaw_rad = float(yaw_rad)
        self.service_timeout_s = float(service_timeout_s)
        self.latest_image = None
        self.latest_stamp = math.nan
        self.create_subscription(Image, '/external_camera/image_raw', self._image_cb, 10)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)

    def _image_cb(self, msg: Image) -> None:
        try:
            self.latest_image = image_msg_to_bgr8(msg)
        except ValueError as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}')
            return
        self.latest_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _yaw_to_quaternion(yaw: float):
        half = 0.5 * float(yaw)
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(float(timeout_s), 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if service_ready and self.latest_image is not None:
                return
        missing = []
        if not service_ready:
            missing.append(self.service_name)
        if self.latest_image is None:
            missing.append('/external_camera/image_raw')
        raise RuntimeError(f'Timed out waiting for {", ".join(missing) or "capture interfaces"}')

    def set_robot_pose(self, x: float, y: float) -> None:
        req = SetEntityPose.Request()
        req.entity = Entity(name='turtlebot3')
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.robot_z)
        qx, qy, qz, qw = self._yaw_to_quaternion(self.yaw_rad)
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw
        future = self.client.call_async(req)
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - start > self.service_timeout_s:
                raise RuntimeError(
                    f'Timed out waiting for {self.service_name}. This usually means the Gazebo robot model '
                    "was not spawned yet; check that the launch terminal reached 'OK creation of entity' "
                    "and clean up stale ign/gz processes if ros_gz_sim create is stuck."
                )
        result = future.result()
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(f'Failed to set robot pose to ({x:.3f}, {y:.3f})')

    def wait_for_image_after(self, min_stamp: float, timeout_s: float):
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_image is not None and (not math.isfinite(min_stamp) or self.latest_stamp >= min_stamp):
                return np.asarray(self.latest_image).copy(), float(self.latest_stamp)
        if self.latest_image is None:
            raise RuntimeError('No camera image received')
        return np.asarray(self.latest_image).copy(), float(self.latest_stamp)

    def project_robot_bbox(
        self,
        x: float,
        y: float,
        *,
        half_x: float,
        half_y: float,
        height_m: float,
        pad_px: float,
    ):
        yaw = float(self.yaw_rad)
        c, s = math.cos(yaw), math.sin(yaw)
        corners = []
        for lx in (-half_x, half_x):
            for ly in (-half_y, half_y):
                wx = float(x) + c * lx - s * ly
                wy = float(y) + s * lx + c * ly
                for z in (0.0, float(height_m)):
                    corners.append((wx, wy, z))

        pixels = []
        for wx, wy, wz in corners:
            world_3d = np.array([wx, wy, wz], dtype=float)
            cam_pt = self.camera_model.R @ (world_3d - self.camera_model.cam_pos)
            if cam_pt[2] <= 1e-6:
                continue
            pix = self.camera_model.K @ cam_pt
            pixels.append((float(pix[0] / pix[2]), float(pix[1] / pix[2])))
        if not pixels:
            return None
        arr = np.asarray(pixels, dtype=float)
        xmin = float(np.min(arr[:, 0]) - pad_px)
        ymin = float(np.min(arr[:, 1]) - pad_px)
        xmax = float(np.max(arr[:, 0]) + pad_px)
        ymax = float(np.max(arr[:, 1]) + pad_px)
        w = float(self.camera_model.img_width)
        h = float(self.camera_model.img_height)
        xmin = float(np.clip(xmin, 0.0, w - 1.0))
        xmax = float(np.clip(xmax, 0.0, w - 1.0))
        ymin = float(np.clip(ymin, 0.0, h - 1.0))
        ymax = float(np.clip(ymax, 0.0, h - 1.0))
        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax


def _write_label(path: Path, bbox, img_w: int, img_h: int) -> None:
    if bbox is None:
        path.write_text('', encoding='utf-8')
        return
    xmin, ymin, xmax, ymax = bbox
    cx = 0.5 * (xmin + xmax) / float(img_w)
    cy = 0.5 * (ymin + ymax) / float(img_h)
    bw = (xmax - xmin) / float(img_w)
    bh = (ymax - ymin) / float(img_h)
    path.write_text(f'0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n', encoding='utf-8')


def _write_preview(path: Path, image_bgr: np.ndarray, bbox) -> None:
    preview = np.asarray(image_bgr).copy()
    if bbox is not None:
        xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
        cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(
            preview,
            'robot',
            (xmin, max(ymin - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def _red_pixels_in_bbox(image_bgr: np.ndarray, bbox) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_bgr.shape[1] - 1)
    y1 = min(y1, image_bgr.shape[0] - 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = image_bgr[y0:y1 + 1, x0:x1 + 1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    return int(np.count_nonzero(mask))


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture auto-labeled YOLO robot dataset from Gazebo.')
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=_default_arg_path('src/experiments/config/world_profiles.yaml'))
    parser.add_argument('--output-root', default=_default_arg_path('logs/yolo_datasets'))
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
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ready-timeout-s', type=float, default=20.0)
    parser.add_argument('--settle-s', type=float, default=0.25)
    parser.add_argument('--image-timeout-s', type=float, default=1.0)
    parser.add_argument('--service-timeout-s', type=float, default=20.0)
    parser.add_argument(
        '--min-label-red-px',
        type=float,
        default=0.0,
        help='Optional capture-time quality gate; >0 blanks projected boxes with too little visible red robot evidence.',
    )
    parser.add_argument('--preview-count', type=int, default=40, help='Number of captured frames with bbox overlays to save for manual spot-checking.')
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

    out_dir = Path(args.output_root).expanduser().resolve() / f'yolo_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
                bbox = node.project_robot_bbox(
                    x,
                    y,
                    half_x=float(args.robot_half_x),
                    half_y=float(args.robot_half_y),
                    height_m=float(args.robot_height_m),
                    pad_px=float(args.bbox_pad_px),
                )
                raw_projected_bbox = list(bbox) if bbox is not None else None
                visible_red_px = _red_pixels_in_bbox(image_bgr, bbox)
                if bbox is not None and visible_red_px < max(float(args.min_label_red_px), 0.0):
                    bbox = None
                split = 'val' if float(rng.random()) < float(args.val_fraction) else 'train'
                stem = f'{sample_idx:06d}'
                image_path = out_dir / 'images' / split / f'{stem}.jpg'
                label_path = out_dir / 'labels' / split / f'{stem}.txt'
                cv2.imwrite(str(image_path), image_bgr)
                _write_label(label_path, bbox, image_bgr.shape[1], image_bgr.shape[0])
                preview_path = ''
                if sample_idx < max(int(args.preview_count), 0):
                    preview = out_dir / 'previews' / f'{stem}_{split}.jpg'
                    _write_preview(preview, image_bgr, bbox)
                    preview_path = str(preview.relative_to(out_dir))
                rows.append({
                    'sample_idx': sample_idx,
                    'repeat_idx': repeat_idx,
                    'split': split,
                    'x': x,
                    'y': y,
                    'yaw': float(args.yaw_rad),
                    'stamp': stamp,
                    'image': str(image_path.relative_to(out_dir)),
                    'label': str(label_path.relative_to(out_dir)),
                    'bbox': list(bbox) if bbox is not None else None,
                    'raw_projected_bbox': raw_projected_bbox,
                    'visible_red_px': int(visible_red_px),
                    'preview': preview_path,
                })
                sample_idx += 1
                if sample_idx % 25 == 0:
                    node.get_logger().info(f'Captured {sample_idx} YOLO dataset frames')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    data_yaml = {
        'path': str(out_dir),
        'task': 'detect',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'robot'},
    }
    (out_dir / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8')
    (out_dir / 'capture_manifest.json').write_text(json.dumps({
        'world': args.world,
        'world_name': profile['world_name'],
        'n_frames': len(rows),
        'camera': {
            'cam_pos': cam_pos,
            'look_at': look_at,
            'img_width': int(intrinsics['img_width']),
            'img_height': int(intrinsics['img_height']),
            'fov_h_rad': float(intrinsics['fov_h_rad']),
        },
        'label_model': {
            'robot_half_x': float(args.robot_half_x),
            'robot_half_y': float(args.robot_half_y),
            'robot_height_m': float(args.robot_height_m),
            'bbox_pad_px': float(args.bbox_pad_px),
            'min_label_red_px': float(args.min_label_red_px),
        },
        'samples': rows,
    }, indent=2), encoding='utf-8')
    print(f'Wrote YOLO dataset to {out_dir}')
    print(
        'Next supervisor-aligned step: '
        f'python3 scripts/convert_yolo_labels_to_seg_by_sam.py {out_dir} --sam-model mobile_sam.pt'
    )
    print(f'Optional bbox-only baseline: yolo detect train model=yolo11n.pt data={out_dir / "data.yaml"} imgsz=640')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
