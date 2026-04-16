#!/usr/bin/env python3
"""Capture a canonical teleport-sampled visibility dataset for method comparison."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image

from common import (
    CAPTURE_COLUMNS,
    CURRENT_CAPTURE_DIR,
    LOGS_ROOT,
    bbox_bottom_center,
    ensure_repo_python_paths,
    project_robot_bbox,
    repo_relative,
    safe_reset_generated_dir,
    timestamp,
    write_csv,
    write_manifest,
)

ensure_repo_python_paths()

from experiments.core.world_profiles import compute_look_at_from_pose, load_profile
from perception.core.ros_image import image_msg_to_bgr8
from unav_common.camera_model import ObliqueCameraModel
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world, scene_to_json, segment_occluded


def _sample_positions(vis: dict[str, object], *, sample_nx: int, sample_ny: int, wall_margin_m: float):
    xmin = float(vis.get('visibility_map_min_x', -6.0)) + float(max(wall_margin_m, 0.0))
    xmax = float(vis.get('visibility_map_max_x', 6.0)) - float(max(wall_margin_m, 0.0))
    ymin = float(vis.get('visibility_map_min_y', -6.0)) + float(max(wall_margin_m, 0.0))
    ymax = float(vis.get('visibility_map_max_y', 6.0)) - float(max(wall_margin_m, 0.0))
    if not (xmin < xmax and ymin < ymax):
        raise RuntimeError('Invalid sample bounds after applying wall_margin_m')

    xs = np.linspace(xmin, xmax, max(int(sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(sample_ny), 1))
    positions = []
    for row_idx, y in enumerate(ys):
        x_iter = xs if (row_idx % 2 == 0) else xs[::-1]
        for x in x_iter:
            positions.append((float(x), float(y)))
    return positions


class TeleportImageCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        image_topic: str,
        odom_topic: str,
        robot_z: float,
        settle_s: float,
        image_timeout_s: float,
        pose_timeout_s: float,
        pose_tolerance_m: float,
        yaw_tolerance_rad: float,
        verify_with_odom: bool,
        min_new_frames: int,
    ):
        super().__init__('capture_visibility_samples')
        self.world_name = str(world_name)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.image_timeout_s = float(image_timeout_s)
        self.pose_timeout_s = float(pose_timeout_s)
        self.pose_tolerance_m = float(pose_tolerance_m)
        self.yaw_tolerance_rad = float(yaw_tolerance_rad)
        self.verify_with_odom = bool(verify_with_odom)
        self.min_new_frames = max(int(min_new_frames), 1)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.image_count = 0
        self.latest_image = None
        self.latest_odom_xyyaw = None
        self.create_subscription(Image, str(image_topic), self._image_cb, 10)
        if self.verify_with_odom:
            self.create_subscription(Odometry, str(odom_topic), self._odom_cb, 10)

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = image_msg_to_bgr8(msg)
        self.image_count += 1

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
            if self.latest_image is None:
                continue
            if self.verify_with_odom and self.latest_odom_xyyaw is None:
                continue
            if service_ready:
                return
        raise RuntimeError(
            f'Timed out waiting for {self.service_name}, first camera image'
            f'{" and /odom" if self.verify_with_odom else ""}. Launch the simulator first.'
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

    def _pose_matches(self, *, x: float, y: float, yaw: float) -> bool:
        if self.latest_odom_xyyaw is None:
            return False
        x_now, y_now, yaw_now = self.latest_odom_xyyaw
        pos_err = math.hypot(float(x_now) - float(x), float(y_now) - float(y))
        yaw_err = math.atan2(math.sin(float(yaw_now) - float(yaw)), math.cos(float(yaw_now) - float(yaw)))
        return pos_err <= self.pose_tolerance_m and abs(yaw_err) <= self.yaw_tolerance_rad

    def capture_image_at_pose(self, *, x: float, y: float, yaw: float) -> np.ndarray:
        before = int(self.image_count)
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
            if self.image_count >= (before + self.min_new_frames) and self.latest_image is not None:
                return self.latest_image.copy()
        if self.latest_image is None:
            raise RuntimeError('No camera image available after teleport')
        return self.latest_image.copy()


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture a canonical teleport-sampled visibility dataset.')
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=str((Path(__file__).resolve().parents[2] / 'src' / 'experiments' / 'config' / 'world_profiles.yaml').resolve()))
    parser.add_argument('--out', default=str(CURRENT_CAPTURE_DIR))
    parser.add_argument('--sample-nx', type=int, default=15)
    parser.add_argument('--sample-ny', type=int, default=15)
    parser.add_argument('--yaw-rad', type=float, default=0.0)
    parser.add_argument('--wall-margin-m', type=float, default=0.45)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--settle-s', type=float, default=0.35)
    parser.add_argument('--image-timeout-s', type=float, default=2.0)
    parser.add_argument('--image-topic', default='/external_camera/image_raw')
    parser.add_argument('--ready-timeout-s', type=float, default=15.0)
    parser.add_argument('--camera-frame', default='external_camera')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--pose-timeout-s', type=float, default=2.0)
    parser.add_argument('--pose-tolerance-m', type=float, default=0.05)
    parser.add_argument('--yaw-tolerance-rad', type=float, default=0.20)
    parser.add_argument('--verify-with-odom', action='store_true', help='Require /odom to converge to each commanded teleport pose before capture.')
    parser.add_argument('--min-new-frames', type=int, default=1, help='Minimum number of fresh camera frames to wait for after each teleport.')
    parser.add_argument('--target-height-m', type=float, default=0.0)
    parser.add_argument('--robot-box-length', type=float, default=0.22)
    parser.add_argument('--robot-box-width', type=float, default=0.22)
    parser.add_argument('--robot-box-height', type=float, default=0.20)
    parser.add_argument('--draw-oracle-box', action='store_true', help='Overlay the projected oracle reference box in preview images.')
    args = parser.parse_args()

    profile, intrinsics, world_path, camera_pose = load_profile(str(args.world_profiles), str(args.world))
    vis = dict(profile.get('visibility_defaults') or {})
    positions = _sample_positions(
        vis,
        sample_nx=int(args.sample_nx),
        sample_ny=int(args.sample_ny),
        wall_margin_m=float(args.wall_margin_m),
    )
    output_dir = safe_reset_generated_dir(Path(args.out), allowed_root=LOGS_ROOT)
    (output_dir / 'images').mkdir(parents=True, exist_ok=False)
    (output_dir / 'previews').mkdir(parents=True, exist_ok=False)

    cam_pos = np.asarray(camera_pose[:3], dtype=float)
    look_at = compute_look_at_from_pose(cam_pos, camera_pose[3], camera_pose[4], camera_pose[5])
    camera = ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=int(intrinsics['img_width']),
        img_height=int(intrinsics['img_height']),
        fov_h_rad=float(intrinsics['fov_h_rad']),
    )
    occlusion_scene = parse_occlusion_scene_from_world(str(world_path))

    rows: list[dict[str, str]] = []
    image_hashes: list[str] = []
    rclpy.init()
    node = TeleportImageCapture(
        world_name=str(profile['world_name']),
        image_topic=str(args.image_topic),
        odom_topic=str(args.odom_topic),
        robot_z=float(args.robot_z),
        settle_s=float(args.settle_s),
        image_timeout_s=float(args.image_timeout_s),
        pose_timeout_s=float(args.pose_timeout_s),
        pose_tolerance_m=float(args.pose_tolerance_m),
        yaw_tolerance_rad=float(args.yaw_tolerance_rad),
        verify_with_odom=bool(args.verify_with_odom),
        min_new_frames=int(args.min_new_frames),
    )
    try:
        node.wait_for_ready(timeout_s=float(args.ready_timeout_s))
        for sample_id, (x, y) in enumerate(positions):
            image = node.capture_image_at_pose(x=x, y=y, yaw=float(args.yaw_rad))
            image_path = output_dir / 'images' / f'{sample_id:06d}.jpg'
            preview_path = output_dir / 'previews' / f'{sample_id:06d}.jpg'
            cv2.imwrite(str(image_path), image)
            image_hashes.append(hashlib.md5(image_path.read_bytes()).hexdigest())

            bbox = project_robot_bbox(
                camera,
                x=float(x),
                y=float(y),
                yaw=float(args.yaw_rad),
                box_length=float(args.robot_box_length),
                box_width=float(args.robot_box_width),
                box_height=float(args.robot_box_height),
                img_w=image.shape[1],
                img_h=image.shape[0],
            )
            bottom_u, bottom_v = bbox_bottom_center(bbox)
            target_xyz = np.asarray([float(x), float(y), float(args.target_height_m)], dtype=float)
            occluded = segment_occluded(occlusion_scene.prisms, camera.cam_pos, target_xyz)
            in_frame = bbox is not None and math.isfinite(bottom_u) and math.isfinite(bottom_v)
            if not in_frame:
                oracle_visible = 0
                oracle_reason = 'outside_image'
            elif occluded:
                oracle_visible = 0
                oracle_reason = 'occluded'
            else:
                oracle_visible = 1
                oracle_reason = 'visible'

            preview = image.copy()
            if bool(args.draw_oracle_box) and bbox is not None:
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 200, 0), 2)
            if math.isfinite(bottom_u) and math.isfinite(bottom_v):
                color = (0, 255, 0) if oracle_visible else (0, 0, 255)
                cv2.circle(preview, (int(round(bottom_u)), int(round(bottom_v))), 5, color, -1)
            cv2.putText(
                preview,
                f'sample={sample_id} oracle={oracle_visible} reason={oracle_reason}',
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(preview_path), preview)

            rows.append({
                'sample_id': str(sample_id),
                'image_path': repo_relative(image_path, output_dir),
                'preview_path': repo_relative(preview_path, output_dir),
                'x': f'{float(x):.8f}',
                'y': f'{float(y):.8f}',
                'theta': f'{float(args.yaw_rad):.8f}',
                'timestamp': f'{time.time():.6f}',
                'world': str(args.world),
                'camera_frame': str(args.camera_frame),
                'oracle_visible': str(int(oracle_visible)),
                'oracle_bottom_u': '' if not math.isfinite(bottom_u) else f'{float(bottom_u):.8f}',
                'oracle_bottom_v': '' if not math.isfinite(bottom_v) else f'{float(bottom_v):.8f}',
                'oracle_occlusion_reason': oracle_reason,
                'segmentation_path': '',
                'labels_map_path': '',
                'robot_label': '',
            })
    finally:
        node.destroy_node()
        rclpy.shutdown()

    write_csv(output_dir / 'samples.csv', CAPTURE_COLUMNS, rows)
    unique_image_count = len(set(image_hashes))
    duplicate_image_count = max(len(image_hashes) - unique_image_count, 0)
    manifest = {
        'world': str(args.world),
        'world_name': str(profile['world_name']),
        'world_profiles_path': str(Path(args.world_profiles).resolve()),
        'world_path': str(Path(world_path).resolve()),
        'capture_mode': 'teleport',
        'camera_frame': str(args.camera_frame),
        'camera_pose': [float(v) for v in camera_pose],
        'camera_pos': [float(v) for v in cam_pos],
        'look_at': [float(v) for v in look_at],
        'img_width': int(intrinsics['img_width']),
        'img_height': int(intrinsics['img_height']),
        'fov_h_rad': float(intrinsics['fov_h_rad']),
        'visibility_bounds': {
            'xmin': float(vis.get('visibility_map_min_x', -6.0)),
            'xmax': float(vis.get('visibility_map_max_x', 6.0)),
            'ymin': float(vis.get('visibility_map_min_y', -6.0)),
            'ymax': float(vis.get('visibility_map_max_y', 6.0)),
            'nx': int(args.sample_nx),
            'ny': int(args.sample_ny),
            'wall_margin_m': float(args.wall_margin_m),
        },
        'oracle_source': 'geometry',
        'oracle_target_height_m': float(args.target_height_m),
        'oracle_preview_draws_reference_box': bool(args.draw_oracle_box),
        'robot_box': {
            'length': float(args.robot_box_length),
            'width': float(args.robot_box_width),
            'height': float(args.robot_box_height),
        },
        'geometry_json': scene_to_json(occlusion_scene),
        'sample_count': int(len(rows)),
        'unique_image_count': int(unique_image_count),
        'duplicate_image_count': int(duplicate_image_count),
        'output_dir': str(output_dir),
        'timestamp': timestamp(),
        'notes': [
            'This capture is the canonical shared raw input for visibility-method comparisons.',
            'samples.csv contains raw shared observations only; method-specific targets are created later.',
            'Each sample is captured after teleporting the robot with the ROS-bridged Gazebo set_pose service.',
            'A configurable number of fresh camera frames are skipped after each teleport to avoid stale images.',
        ],
    }
    write_manifest(output_dir / 'capture_manifest.json', manifest)
    print(f'Wrote capture to {output_dir}')
    print(f'Samples: {len(rows)}')
    if duplicate_image_count > 0:
        print(
            'Warning: duplicate frames detected during capture '
            f'({duplicate_image_count} duplicates across {len(rows)} samples).'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
