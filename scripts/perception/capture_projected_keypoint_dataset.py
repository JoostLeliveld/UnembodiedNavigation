#!/usr/bin/env python3
"""Capture a YOLO pose-keypoint dataset by teleporting the robot.

Front/rear marker positions in the robot frame are projected analytically with
the existing oblique camera model. The robot URDF must be launched with
`pose_markers:=true` so the markers are physically present in the rendered
camera image — projection is only ground-truth, not a substitute for rendering.

Mirrors the structure of capture_projected_bbox_dataset.py to keep behaviour and
sampling bounds identical.
"""

from __future__ import annotations

import argparse
import csv
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
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rclpy.node import Node
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.world_profiles import compute_look_at_from_pose, load_profile
from dataset_split_utils import assign_splits, build_pose_records, evenly_spaced_yaws
from perception.core.pose_keypoints import (
    DEFAULT_MARKER_GEOMETRY,
    KEYPOINT_FRONT,
    KEYPOINT_REAR,
    KPT_VIS_OCCLUDED,
    KPT_VIS_VISIBLE,
    NUM_KEYPOINTS,
)
from perception.core.ros_image import image_msg_to_bgr8
from unav_common.camera_model import ObliqueCameraModel


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _project_world_point(camera: ObliqueCameraModel, xyz: np.ndarray) -> np.ndarray | None:
    cam_pt = camera.R @ (np.asarray(xyz, dtype=float) - camera.cam_pos)
    if cam_pt[2] <= 1e-6:
        return None
    pixel_h = camera.K @ cam_pt
    return np.asarray([pixel_h[0] / pixel_h[2], pixel_h[1] / pixel_h[2]], dtype=float)


def _marker_world_position(
    *, x: float, y: float, yaw: float, robot_z: float, offset_x: float, offset_z: float,
) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    wx = float(x) + c * float(offset_x)
    wy = float(y) + s * float(offset_x)
    wz = float(robot_z) + DEFAULT_MARKER_GEOMETRY.base_joint_z + float(offset_z)
    return np.asarray([wx, wy, wz], dtype=float)


def _project_robot_aabb(
    camera: ObliqueCameraModel,
    *,
    x: float,
    y: float,
    yaw: float,
    box_length: float,
    box_width: float,
    box_height: float,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    hx, hy = 0.5 * float(box_length), 0.5 * float(box_width)
    hz = float(box_height)
    local = np.asarray([
        [-hx, -hy, 0.0], [-hx, hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0],
        [-hx, -hy, hz], [-hx, hy, hz], [hx, -hy, hz], [hx, hy, hz],
    ], dtype=float)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    world = (rot @ local.T).T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    pts = []
    for corner in world:
        uv = _project_world_point(camera, corner)
        if uv is None:
            return None
        pts.append(uv)
    pts = np.asarray(pts, dtype=float)
    x0 = float(np.clip(np.min(pts[:, 0]), 0.0, img_w - 1.0))
    y0 = float(np.clip(np.min(pts[:, 1]), 0.0, img_h - 1.0))
    x1 = float(np.clip(np.max(pts[:, 0]), 0.0, img_w - 1.0))
    y1 = float(np.clip(np.max(pts[:, 1]), 0.0, img_h - 1.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=float)


def _kpt_inside(uv: np.ndarray, img_w: int, img_h: int, margin: float = 1.0) -> bool:
    return bool(margin <= uv[0] <= img_w - 1.0 - margin and margin <= uv[1] <= img_h - 1.0 - margin)


def _build_pose_label_line(
    *, bbox: np.ndarray, front_uv: np.ndarray, rear_uv: np.ndarray,
    front_visible: bool, rear_visible: bool,
    img_w: int, img_h: int,
) -> str:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx = 0.5 * (x0 + x1) / float(img_w)
    cy = 0.5 * (y0 + y1) / float(img_h)
    bw = (x1 - x0) / float(img_w)
    bh = (y1 - y0) / float(img_h)
    f_vis = KPT_VIS_VISIBLE if front_visible else KPT_VIS_OCCLUDED
    r_vis = KPT_VIS_VISIBLE if rear_visible else KPT_VIS_OCCLUDED
    fx = float(front_uv[0]) / float(img_w)
    fy = float(front_uv[1]) / float(img_h)
    rx = float(rear_uv[0]) / float(img_w)
    ry = float(rear_uv[1]) / float(img_h)
    return (
        f'0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} '
        f'{fx:.8f} {fy:.8f} {f_vis:.0f} {rx:.8f} {ry:.8f} {r_vis:.0f}\n'
    )


class TeleportKeypointCapture(Node):
    def __init__(
        self, *, world_name: str, image_topic: str,
        robot_z: float, settle_s: float, image_timeout_s: float,
    ):
        super().__init__('capture_projected_keypoint_dataset')
        self.world_name = str(world_name)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.image_timeout_s = float(image_timeout_s)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.image_count = 0
        self.latest_image = None
        self.create_subscription(Image, str(image_topic), self._image_cb, 10)

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = image_msg_to_bgr8(msg)
        self.image_count += 1

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(float(timeout_s), 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if self.latest_image is not None and service_ready:
                return
        raise RuntimeError(
            f'Timed out waiting for {self.service_name} and first camera image. Launch the simulator first.'
        )

    def _set_robot_pose(self, *, x: float, y: float, yaw: float) -> None:
        req = SetEntityPose.Request()
        req.entity = Entity(name='turtlebot3')
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.robot_z)
        half = 0.5 * float(yaw)
        req.pose.orientation.z = math.sin(half)
        req.pose.orientation.w = math.cos(half)
        future = self.client.call_async(req)
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if (time.monotonic() - start) > 10.0:
                raise RuntimeError(f'Timed out waiting for {self.service_name} response')
        result = future.result()
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(f'Set pose request failed at x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}')

    def capture_image_at_pose(self, *, x: float, y: float, yaw: float) -> np.ndarray:
        before = int(self.image_count)
        self._set_robot_pose(x=x, y=y, yaw=yaw)
        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        while rclpy.ok() and time.monotonic() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)
        deadline = time.monotonic() + max(self.image_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.image_count > before and self.latest_image is not None:
                return self.latest_image.copy()
        if self.latest_image is None:
            raise RuntimeError('No camera image available after teleport')
        return self.latest_image.copy()


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture a YOLO pose-keypoint dataset by teleporting the robot.')
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=str((REPO_ROOT / 'src' / 'experiments' / 'config' / 'world_profiles.yaml').resolve()))
    parser.add_argument('--out', default='', help='Output dataset folder; defaults under logs/')
    parser.add_argument('--sample-nx', type=int, default=12)
    parser.add_argument('--sample-ny', type=int, default=10)
    parser.add_argument('--yaw-samples', type=int, default=8,
                        help='More yaws than the bbox script: heading is the target.')
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-mode', choices=('cyclic', 'yaw_bucket', 'spatial_cell', 'spatial_yaw_bucket'), default='spatial_yaw_bucket')
    parser.add_argument('--spatial-block-size', type=int, default=2)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--settle-s', type=float, default=0.35)
    parser.add_argument('--image-timeout-s', type=float, default=2.0)
    parser.add_argument('--preview-count', type=int, default=80)
    parser.add_argument('--min-bbox-size-px', type=float, default=18.0)
    parser.add_argument('--box-length', type=float, default=0.16)
    parser.add_argument('--box-width', type=float, default=0.16)
    parser.add_argument('--box-height', type=float, default=0.20)
    parser.add_argument('--image-topic', default='/external_camera/image_raw')
    args = parser.parse_args()

    profile, intrinsics, world_path, camera_pose = load_profile(str(args.world_profiles), str(args.world))
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis.get('visibility_map_min_x', -3.0)) + float(args.wall_margin)
    xmax = float(vis.get('visibility_map_max_x', 3.0)) - float(args.wall_margin)
    ymin = float(vis.get('visibility_map_min_y', -3.0)) + float(args.wall_margin)
    ymax = float(vis.get('visibility_map_max_y', 3.0)) - float(args.wall_margin)
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError('Invalid sampling bounds after wall margin')

    cam_pos = camera_pose[:3]
    look_at = compute_look_at_from_pose(cam_pos, camera_pose[3], camera_pose[4], camera_pose[5])
    img_w = int(intrinsics['img_width'])
    img_h = int(intrinsics['img_height'])
    camera = ObliqueCameraModel(
        cam_pos=cam_pos, look_at=look_at, img_width=img_w, img_height=img_h,
        fov_h_rad=float(intrinsics['fov_h_rad']),
    )

    geom = DEFAULT_MARKER_GEOMETRY
    xs = np.linspace(xmin, xmax, max(int(args.sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(args.sample_ny), 1))
    yaw_samples = max(int(args.yaw_samples), 1)
    yaws = np.asarray(evenly_spaced_yaws(yaw_samples), dtype=float)
    pose_records = build_pose_records(xs, ys, yaws)
    planned = [(float(item['x']), float(item['y']), float(item['yaw'])) for item in pose_records]
    split_labels = assign_splits(
        pose_records, val_fraction=float(args.val_fraction),
        split_mode=str(args.split_mode), seed=int(args.split_seed),
        spatial_block_size=int(args.spatial_block_size),
    )

    if str(args.out).strip():
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = (REPO_ROOT / 'logs' / f'projected_keypoint_dataset_{_timestamp()}').resolve()
    if out_dir.exists():
        raise RuntimeError(f'Output folder already exists: {out_dir}')

    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=False)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=False)

    diagnostics: list[dict] = []
    accepted = 0
    skipped = 0
    preview_written = 0

    rclpy.init()
    node = TeleportKeypointCapture(
        world_name=str(profile['world_name']), image_topic=str(args.image_topic),
        robot_z=float(args.robot_z), settle_s=float(args.settle_s), image_timeout_s=float(args.image_timeout_s),
    )
    try:
        node.wait_for_ready(timeout_s=20.0)
        for sample_idx, (x, y, yaw) in enumerate(planned):
            image = node.capture_image_at_pose(x=x, y=y, yaw=yaw)

            front_world = _marker_world_position(
                x=x, y=y, yaw=yaw, robot_z=float(args.robot_z),
                offset_x=geom.front_x, offset_z=geom.marker_z_in_base_link,
            )
            rear_world = _marker_world_position(
                x=x, y=y, yaw=yaw, robot_z=float(args.robot_z),
                offset_x=geom.rear_x, offset_z=geom.marker_z_in_base_link,
            )
            front_uv = _project_world_point(camera, front_world)
            rear_uv = _project_world_point(camera, rear_world)
            bbox = _project_robot_aabb(
                camera, x=x, y=y, yaw=yaw,
                box_length=float(args.box_length), box_width=float(args.box_width),
                box_height=float(args.box_height), img_w=image.shape[1], img_h=image.shape[0],
            )

            reason = ''
            if front_uv is None or rear_uv is None:
                reason = 'keypoint_behind_camera'
            elif bbox is None:
                reason = 'bbox_not_projectable'
            else:
                bw = float(bbox[2] - bbox[0])
                bh = float(bbox[3] - bbox[1])
                if bw < float(args.min_bbox_size_px) or bh < float(args.min_bbox_size_px):
                    reason = 'bbox_too_small'

            if reason:
                skipped += 1
                diagnostics.append({
                    'sample_idx': int(sample_idx), 'x': float(x), 'y': float(y), 'yaw_rad': float(yaw),
                    'accepted': 0, 'rejection_reason': reason, 'split': '',
                    'image': '', 'label': '',
                    'front_u': math.nan, 'front_v': math.nan,
                    'rear_u': math.nan, 'rear_v': math.nan,
                    'front_visible': 0, 'rear_visible': 0,
                    'preview': '',
                })
                continue

            front_visible = _kpt_inside(front_uv, image.shape[1], image.shape[0])
            rear_visible = _kpt_inside(rear_uv, image.shape[1], image.shape[0])

            split = str(split_labels[sample_idx])
            stem = f'{accepted:06d}'
            image_path = out_dir / 'images' / split / f'{stem}.jpg'
            label_path = out_dir / 'labels' / split / f'{stem}.txt'
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f'Failed to write image {image_path}')
            label_path.write_text(
                _build_pose_label_line(
                    bbox=bbox, front_uv=front_uv, rear_uv=rear_uv,
                    front_visible=front_visible, rear_visible=rear_visible,
                    img_w=image.shape[1], img_h=image.shape[0],
                ),
                encoding='utf-8',
            )

            preview_rel = ''
            if preview_written < max(int(args.preview_count), 0):
                preview = image.copy()
                cv2.rectangle(
                    preview,
                    (int(round(bbox[0])), int(round(bbox[1]))),
                    (int(round(bbox[2])), int(round(bbox[3]))),
                    (255, 200, 0), 2,
                )
                cv2.circle(preview, (int(round(front_uv[0])), int(round(front_uv[1]))), 6, (0, 0, 255), 2)
                cv2.circle(preview, (int(round(rear_uv[0])), int(round(rear_uv[1]))), 6, (255, 0, 0), 2)
                cv2.line(preview,
                         (int(round(rear_uv[0])), int(round(rear_uv[1]))),
                         (int(round(front_uv[0])), int(round(front_uv[1]))),
                         (0, 255, 255), 2)
                cv2.putText(preview, f'x={x:.2f} y={y:.2f} yaw={yaw:.2f}',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                preview_path = out_dir / 'previews' / split / f'{stem}.jpg'
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), preview)
                preview_rel = str(preview_path.relative_to(out_dir))
                preview_written += 1

            diagnostics.append({
                'sample_idx': int(sample_idx), 'x': float(x), 'y': float(y), 'yaw_rad': float(yaw),
                'accepted': 1, 'rejection_reason': '', 'split': split,
                'image': str(image_path.relative_to(out_dir)), 'label': str(label_path.relative_to(out_dir)),
                'front_u': float(front_uv[0]), 'front_v': float(front_uv[1]),
                'rear_u': float(rear_uv[0]), 'rear_v': float(rear_uv[1]),
                'front_visible': int(front_visible), 'rear_visible': int(rear_visible),
                'preview': preview_rel,
            })
            accepted += 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    (out_dir / 'data.yaml').write_text(
        yaml.safe_dump(
            {
                'path': str(out_dir),
                'train': 'images/train',
                'val': 'images/val',
                'names': {0: 'robot'},
                'kpt_shape': [int(NUM_KEYPOINTS), 3],
                'flip_idx': [int(KEYPOINT_FRONT), int(KEYPOINT_REAR)],
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    (out_dir / 'capture_manifest.json').write_text(
        json.dumps(
            {
                'world': str(args.world), 'world_name': str(profile['world_name']),
                'world_path': str(world_path),
                'camera_pose': [float(v) for v in camera_pose],
                'look_at': [float(v) for v in look_at],
                'sample_nx': int(args.sample_nx), 'sample_ny': int(args.sample_ny),
                'yaw_samples': int(args.yaw_samples),
                'yaw_values_rad': [float(v) for v in yaws.tolist()],
                'planned_samples': int(len(planned)),
                'accepted_samples': int(accepted), 'skipped_samples': int(skipped),
                'split_mode': str(args.split_mode), 'split_seed': int(args.split_seed),
                'spatial_block_size': int(args.spatial_block_size),
                'wall_margin': float(args.wall_margin),
                'box_length': float(args.box_length), 'box_width': float(args.box_width),
                'box_height': float(args.box_height),
                'marker_geometry': {
                    'front_x': geom.front_x, 'rear_x': geom.rear_x,
                    'marker_y': geom.marker_y, 'marker_z_in_base_link': geom.marker_z_in_base_link,
                    'base_joint_z': geom.base_joint_z,
                },
                'keypoint_layout': {
                    '0': 'front_virtual_at_lidar_height',
                    '1': 'rear_lidar_centre',
                },
                'timestamp': _timestamp(),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    with (out_dir / 'capture_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        fieldnames = [
            'sample_idx', 'x', 'y', 'yaw_rad', 'accepted', 'rejection_reason',
            'split', 'image', 'label',
            'front_u', 'front_v', 'rear_u', 'rear_v',
            'front_visible', 'rear_visible', 'preview',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f'Wrote projected-keypoint dataset to {out_dir}')
    print(json.dumps({'accepted_samples': int(accepted), 'skipped_samples': int(skipped)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
