#!/usr/bin/env python3
"""Capture a YOLO-seg dataset directly from Gazebo semantic-segmentation output."""

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
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rclpy.node import Node
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/experiments', 'src/perception', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.world_profiles import load_profile
from dataset_split_utils import assign_splits, build_pose_records, evenly_spaced_yaws
from perception.core.ros_image import image_msg_to_bgr8


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _read_uint_label_map(msg: Image) -> np.ndarray:
    encoding = str(msg.encoding or '').strip().lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    if height <= 0 or width <= 0:
        raise ValueError(f'Invalid label image shape height={height}, width={width}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError(f'Label image has {raw.size} bytes, expected at least {expected}')
    rows = raw[:expected].reshape(height, step)

    if encoding in ('rgb8', 'bgr8', '8uc3'):
        min_step = width * 3
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        pixels = rows[:, :min_step].reshape(height, width, 3)
        return pixels[..., 2].astype(np.uint32)

    if encoding in ('mono8', '8uc1'):
        min_step = width
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        return rows[:, :min_step].reshape(height, width).astype(np.uint32)

    if encoding in ('mono16', '16uc1', '16sc1'):
        min_step = width * 2
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        view = rows[:, :min_step].reshape(height, width, 2)
        return view.view(np.uint16).reshape(height, width).astype(np.uint32)

    if encoding in ('32sc1', '32uc1'):
        min_step = width * 4
        if step < min_step:
            raise ValueError(f'Label image step {step} too small for encoding {encoding}')
        view = rows[:, :min_step].reshape(height, width, 4)
        dtype = np.int32 if encoding == '32sc1' else np.uint32
        return view.view(dtype).reshape(height, width).astype(np.uint32)

    raise ValueError(
        f'Unsupported segmentation labels encoding {msg.encoding!r}; '
        'expected rgb8/bgr8/mono8/mono16/32sc1/32uc1'
    )


def _polygon_from_mask(mask_u8: np.ndarray, epsilon_ratio: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None
    epsilon = max(1.0, float(epsilon_ratio) * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2).astype(float)
    if pts.shape[0] < 3:
        return None
    return pts


def _write_seg_label(path: Path, polygon: np.ndarray | None, img_w: int, img_h: int) -> None:
    if polygon is None or len(polygon) < 3:
        path.write_text('', encoding='utf-8')
        return
    values = ['0']
    for x, y in polygon:
        values.append(f'{np.clip(x / float(img_w), 0.0, 1.0):.8f}')
        values.append(f'{np.clip(y / float(img_h), 0.0, 1.0):.8f}')
    path.write_text(' '.join(values) + '\n', encoding='utf-8')


def _mask_bottom(polygon: np.ndarray | None, band_px: float) -> tuple[float, float]:
    if polygon is None or len(polygon) < 3:
        return math.nan, math.nan
    v_bottom = float(np.max(polygon[:, 1]))
    band = polygon[polygon[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(polygon[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


class TeleportSimSegCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        image_topic: str,
        labels_topic: str,
        robot_z: float,
        settle_s: float,
        image_timeout_s: float,
        sync_slop_s: float,
        odom_topic: str,
        pose_timeout_s: float,
        pose_tolerance_m: float,
        yaw_tolerance_rad: float,
        verify_with_odom: bool,
        min_new_frame_pairs: int,
    ):
        super().__init__('capture_simseg_dataset')
        self.world_name = str(world_name)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.image_timeout_s = float(image_timeout_s)
        self.sync_slop_s = float(max(sync_slop_s, 0.0))
        self.pose_timeout_s = float(max(pose_timeout_s, 0.0))
        self.pose_tolerance_m = float(max(pose_tolerance_m, 0.0))
        self.yaw_tolerance_rad = float(max(yaw_tolerance_rad, 0.0))
        self.verify_with_odom = bool(verify_with_odom)
        self.min_new_frame_pairs = max(int(min_new_frame_pairs), 1)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.rgb_count = 0
        self.label_count = 0
        self.latest_image: np.ndarray | None = None
        self.latest_labels: np.ndarray | None = None
        self.latest_image_stamp_ns: int | None = None
        self.latest_label_stamp_ns: int | None = None
        self.latest_odom_xyyaw: tuple[float, float, float] | None = None
        self.create_subscription(Image, str(image_topic), self._image_cb, 10)
        self.create_subscription(Image, str(labels_topic), self._labels_cb, 10)
        if self.verify_with_odom:
            self.create_subscription(Odometry, str(odom_topic), self._odom_cb, 10)

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = image_msg_to_bgr8(msg)
        self.rgb_count += 1
        self.latest_image_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _labels_cb(self, msg: Image) -> None:
        try:
            self.latest_labels = _read_uint_label_map(msg)
            self.label_count += 1
            self.latest_label_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        except Exception as exc:  # narrow callback guard to keep node alive while logging malformed frames
            self.get_logger().error(f'Failed to decode segmentation labels image: {exc}')

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.latest_odom_xyyaw = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(yaw),
        )

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(float(timeout_s), 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if (
                service_ready
                and
                self.latest_image is not None
                and self.latest_labels is not None
                and ((not self.verify_with_odom) or self.latest_odom_xyyaw is not None)
            ):
                return
        raise RuntimeError(
            f'Timed out waiting for {self.service_name}, RGB image, and segmentation labels. '
            'Launch the simulator first and confirm segmentation bridges are running.'
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
            raise RuntimeError(
                f'Set pose request failed at x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
            )

    def _pose_matches(self, *, x: float, y: float, yaw: float) -> bool:
        if self.latest_odom_xyyaw is None:
            return False
        x_now, y_now, yaw_now = self.latest_odom_xyyaw
        pos_err = math.hypot(float(x_now) - float(x), float(y_now) - float(y))
        yaw_err = math.atan2(math.sin(float(yaw_now) - float(yaw)), math.cos(float(yaw_now) - float(yaw)))
        return pos_err <= self.pose_tolerance_m and abs(yaw_err) <= self.yaw_tolerance_rad

    def capture_at_pose(self, *, x: float, y: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        before_rgb = int(self.rgb_count)
        before_label = int(self.label_count)
        self._set_robot_pose(x=x, y=y, yaw=yaw)

        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        while rclpy.ok() and time.monotonic() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)

        if self.verify_with_odom:
            pose_deadline = time.monotonic() + max(self.pose_timeout_s, 0.0)
            while rclpy.ok() and time.monotonic() < pose_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self._pose_matches(x=x, y=y, yaw=yaw):
                    break
            else:
                raise RuntimeError(
                    f'Robot pose did not converge to commanded sample x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
                )

        deadline = time.monotonic() + max(self.image_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (
                self.rgb_count >= (before_rgb + self.min_new_frame_pairs)
                and self.label_count >= (before_label + self.min_new_frame_pairs)
                and self.latest_image is not None
                and self.latest_labels is not None
                and self.latest_image_stamp_ns is not None
                and self.latest_label_stamp_ns is not None
            ):
                stamp_delta_s = abs(self.latest_image_stamp_ns - self.latest_label_stamp_ns) * 1e-9
                if stamp_delta_s <= self.sync_slop_s:
                    return self.latest_image.copy(), self.latest_labels.copy()
        if self.latest_image is None or self.latest_labels is None:
            raise RuntimeError('No RGB or segmentation labels available after teleport')
        if self.latest_image_stamp_ns is not None and self.latest_label_stamp_ns is not None:
            stamp_delta_s = abs(self.latest_image_stamp_ns - self.latest_label_stamp_ns) * 1e-9
            raise RuntimeError(
                'RGB and segmentation labels were captured but never synchronized closely enough '
                f'(latest delta={stamp_delta_s:.3f}s, required <= {self.sync_slop_s:.3f}s).'
            )
        return self.latest_image.copy(), self.latest_labels.copy()


def _in_any_region(x: float, y: float, regions: list[dict], shrink_m: float) -> bool:
    for r in regions:
        if (float(r['xmin']) + shrink_m <= x <= float(r['xmax']) - shrink_m and
                float(r['ymin']) + shrink_m <= y <= float(r['ymax']) - shrink_m):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture a YOLO-seg dataset directly from Gazebo semantic segmentation labels.')
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=str((REPO_ROOT / 'src' / 'experiments' / 'config' / 'world_profiles.yaml').resolve()))
    parser.add_argument('--out', default='', help='Output dataset folder; defaults under logs/')
    parser.add_argument('--sample-nx', type=int, default=12)
    parser.add_argument('--sample-ny', type=int, default=10)
    parser.add_argument('--yaw-samples', type=int, default=4)
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-mode', choices=('cyclic', 'yaw_bucket', 'spatial_cell', 'spatial_yaw_bucket'), default='spatial_yaw_bucket')
    parser.add_argument('--spatial-block-size', type=int, default=2)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--settle-s', type=float, default=0.35)
    parser.add_argument('--image-timeout-s', type=float, default=2.0)
    parser.add_argument('--sync-slop-ms', type=float, default=60.0)
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--pose-timeout-s', type=float, default=2.0)
    parser.add_argument('--pose-tolerance-m', type=float, default=0.05)
    parser.add_argument('--yaw-tolerance-rad', type=float, default=0.20)
    parser.add_argument('--verify-with-odom', action='store_true')
    parser.add_argument('--min-new-frame-pairs', type=int, default=3)
    parser.add_argument('--preview-count', type=int, default=80)
    parser.add_argument('--min-mask-area', type=float, default=20.0)
    parser.add_argument('--bottom-band-px', type=float, default=3.0)
    parser.add_argument('--epsilon-ratio', type=float, default=0.015)
    parser.add_argument('--image-topic', default='/external_camera/image_raw')
    parser.add_argument('--labels-topic', default='/external_camera/segmentation/labels_map')
    parser.add_argument('--robot-label', type=int, default=23)
    parser.add_argument('--skip-region-filter', action='store_true',
                        help='Disable known_2d_regions traversability filter even when the profile defines regions.')
    parser.add_argument('--region-shrink-m', type=float, default=0.05,
                        help='Shrink each traversable region boundary inward before testing grid points.')
    args = parser.parse_args()

    profile, _intrinsics, _world_path, _camera_pose = load_profile(str(args.world_profiles), str(args.world))
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis.get('visibility_map_min_x', -3.0)) + float(args.wall_margin)
    xmax = float(vis.get('visibility_map_max_x', 3.0)) - float(args.wall_margin)
    ymin = float(vis.get('visibility_map_min_y', -3.0)) + float(args.wall_margin)
    ymax = float(vis.get('visibility_map_max_y', 3.0)) - float(args.wall_margin)
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError('Invalid sampling bounds after wall margin')

    xs = np.linspace(xmin, xmax, max(int(args.sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(args.sample_ny), 1))
    yaw_samples = max(int(args.yaw_samples), 1)
    yaws = np.asarray(evenly_spaced_yaws(yaw_samples), dtype=float)
    pose_records = build_pose_records(xs, ys, yaws)
    known_regions = list(profile.get('known_2d_regions') or [])
    if known_regions and not args.skip_region_filter:
        traversable = [r for r in known_regions if str(r.get('type', '')) == 'traversable']
        if traversable:
            pose_records = [
                rec for rec in pose_records
                if _in_any_region(float(rec['x']), float(rec['y']), traversable, float(args.region_shrink_m))
            ]
    planned = [(float(item['x']), float(item['y']), float(item['yaw'])) for item in pose_records]
    split_labels = assign_splits(
        pose_records,
        val_fraction=float(args.val_fraction),
        split_mode=str(args.split_mode),
        seed=int(args.split_seed),
        spatial_block_size=int(args.spatial_block_size),
    )

    if str(args.out).strip():
        out_dir = Path(args.out).expanduser().resolve()
    else:
        out_dir = (REPO_ROOT / 'logs' / f'simseg_dataset_{_timestamp()}').resolve()
    if out_dir.exists():
        raise RuntimeError(f'Output folder already exists: {out_dir}')

    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=False)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=False)

    diagnostics = []
    accepted = 0
    rejected = 0
    preview_written = 0

    rclpy.init()
    node = TeleportSimSegCapture(
        world_name=str(profile['world_name']),
        image_topic=str(args.image_topic),
        labels_topic=str(args.labels_topic),
        robot_z=float(args.robot_z),
        settle_s=float(args.settle_s),
        image_timeout_s=float(args.image_timeout_s),
        sync_slop_s=float(args.sync_slop_ms) / 1000.0,
        odom_topic=str(args.odom_topic),
        pose_timeout_s=float(args.pose_timeout_s),
        pose_tolerance_m=float(args.pose_tolerance_m),
        yaw_tolerance_rad=float(args.yaw_tolerance_rad),
        verify_with_odom=bool(args.verify_with_odom),
        min_new_frame_pairs=int(args.min_new_frame_pairs),
    )
    try:
        node.wait_for_ready(timeout_s=15.0)
        for index, (x, y, yaw) in enumerate(planned):
            split = str(split_labels[index])
            rgb, labels = node.capture_at_pose(x=x, y=y, yaw=yaw)
            if labels.shape[:2] != rgb.shape[:2]:
                raise RuntimeError(
                    f'Segmentation label map shape {labels.shape[:2]} does not match RGB image shape {rgb.shape[:2]}'
                )

            raw_mask = (labels == int(args.robot_label)).astype(np.uint8) * 255
            polygon = None
            reason = ''
            mask_area = float(cv2.countNonZero(raw_mask))
            if mask_area < float(args.min_mask_area):
                reason = 'small_mask'
            else:
                polygon = _polygon_from_mask(raw_mask, float(args.epsilon_ratio))
                if polygon is None:
                    reason = 'empty_polygon'

            accepted += int(polygon is not None)
            rejected += int(polygon is None)

            stem = f'{index:06d}'
            image_path = out_dir / 'images' / split / f'{stem}.jpg'
            label_path = out_dir / 'labels' / split / f'{stem}.txt'
            cv2.imwrite(str(image_path), rgb)
            _write_seg_label(label_path, polygon, rgb.shape[1], rgb.shape[0])

            mask_bottom_u, mask_bottom_v = _mask_bottom(polygon, float(args.bottom_band_px))
            preview_rel = ''
            if preview_written < max(int(args.preview_count), 0):
                preview = rgb.copy()
                if polygon is not None:
                    pts = polygon.astype(np.int32).reshape(-1, 1, 2)
                    overlay = preview.copy()
                    cv2.fillPoly(overlay, [pts], (0, 180, 0))
                    preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0.0)
                    cv2.polylines(preview, [pts], True, (0, 200, 0), 2)
                if math.isfinite(mask_bottom_u) and math.isfinite(mask_bottom_v):
                    cv2.circle(preview, (int(round(mask_bottom_u)), int(round(mask_bottom_v))), 5, (0, 255, 0), -1)
                cv2.putText(
                    preview,
                    f'accepted={int(polygon is not None)} reason={reason or "ok"} label={int(args.robot_label)}',
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                preview_path = out_dir / 'previews' / split / f'{stem}.jpg'
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), preview)
                preview_rel = str(preview_path.relative_to(out_dir))
                preview_written += 1

            diagnostics.append({
                'split': split,
                'image': str(image_path.relative_to(out_dir)),
                'label': str(label_path.relative_to(out_dir)),
                'accepted': int(polygon is not None),
                'rejection_reason': reason,
                'robot_x': float(x),
                'robot_y': float(y),
                'robot_yaw': float(yaw),
                'mask_area_px': float(mask_area),
                'mask_bottom_u': float(mask_bottom_u),
                'mask_bottom_v': float(mask_bottom_v),
                'preview': preview_rel,
            })
    finally:
        node.destroy_node()
        rclpy.shutdown()

    data_yaml = {
        'path': str(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'robot'},
        'task': 'segment',
    }
    (out_dir / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding='utf-8')

    with (out_dir / 'label_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        fieldnames = list(diagnostics[0].keys()) if diagnostics else [
            'split', 'image', 'label', 'accepted', 'rejection_reason',
            'robot_x', 'robot_y', 'robot_yaw',
            'mask_area_px', 'mask_bottom_u', 'mask_bottom_v', 'preview',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    manifest = {
        'world': str(args.world),
        'source': 'gazebo_semantic_segmentation',
        'robot_label': int(args.robot_label),
        'accepted_labels': int(accepted),
        'rejected_labels': int(rejected),
        'sample_nx': int(args.sample_nx),
        'sample_ny': int(args.sample_ny),
        'yaw_samples': int(args.yaw_samples),
        'yaw_values_rad': [float(v) for v in yaws.tolist()],
        'yaw_sampling_strategy': 'fixed_uniform',
        'split_mode': str(args.split_mode),
        'split_seed': int(args.split_seed),
        'spatial_block_size': int(args.spatial_block_size),
        'timestamp': _timestamp(),
        'notes': [
            'Simulator-native segmentation is used offline only to create training masks.',
            'Runtime still uses YOLO on raw RGB images only.',
            'Robot poses are sampled over fixed evenly spaced headings so training covers orientation variation consistently.',
            'Train/val split is assigned deterministically at the group level to reduce adjacent-pose and adjacent-heading leakage.',
            f'RGB/label frame pairs are accepted only when their header timestamps differ by at most {float(args.sync_slop_ms):.1f} ms.',
            f'At least {int(args.min_new_frame_pairs)} fresh RGB and label frames are observed after each teleport before a sample is accepted.',
        ],
    }
    (out_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'Wrote simulator-segmentation dataset to {out_dir}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
