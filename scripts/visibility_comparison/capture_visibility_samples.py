#!/usr/bin/env python3
"""Capture a canonical teleport-sampled visibility dataset for method comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    ensure_repo_python_paths,
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


def _in_any_region(x: float, y: float, regions: list[dict], shrink_m: float) -> bool:
    for r in regions:
        if (float(r['xmin']) + shrink_m <= x <= float(r['xmax']) - shrink_m and
                float(r['ymin']) + shrink_m <= y <= float(r['ymax']) - shrink_m):
            return True
    return False


def _sample_positions(
    vis: dict[str, object],
    *,
    sample_nx: int,
    sample_ny: int,
    wall_margin_m: float,
    traversable_regions: list[dict] | None = None,
    region_shrink_m: float = 0.05,
):
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
            if traversable_regions is not None and not _in_any_region(x, y, traversable_regions, region_shrink_m):
                continue
            positions.append((float(x), float(y)))
    return positions


def _sample_yaws(*, yaw_rad: float, yaw_samples: int, yaw_list_rad: str) -> list[float]:
    raw_list = str(yaw_list_rad or '').strip()
    if raw_list:
        yaws = []
        for part in raw_list.replace(',', ' ').split():
            try:
                yaws.append(float(part))
            except ValueError as exc:
                raise RuntimeError(f'Invalid yaw value in --yaw-list-rad: {part!r}') from exc
        if not yaws:
            raise RuntimeError('--yaw-list-rad was provided but no finite yaw values were parsed')
        return yaws

    n = int(max(yaw_samples, 1))
    if n <= 1:
        return [float(yaw_rad)]
    return [float(yaw_rad) + (2.0 * math.pi * float(i) / float(n)) for i in range(n)]


class TeleportImageCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        image_topic: str,
        odom_topic: str,
        extra_cameras: tuple[tuple[str, str], ...] = (),
        robot_z: float,
        settle_s: float,
        image_timeout_s: float,
        pose_timeout_s: float,
        pose_tolerance_m: float,
        yaw_tolerance_rad: float,
        verify_with_odom: bool,
        min_new_frames: int,
        warn_on_odom_mismatch: bool = False,
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
        self.warn_on_odom_mismatch = bool(warn_on_odom_mismatch)
        self.min_new_frames = max(int(min_new_frames), 1)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.image_count = 0
        self.latest_image = None
        self.latest_odom_xyyaw = None
        self.create_subscription(Image, str(image_topic), self._image_cb, 10)
        # Extra cameras ride along on the SAME teleport. Capturing four cameras in one
        # pass instead of four sequential passes is a 4x saving on a grid this size, and
        # it guarantees the four views share one robot pose rather than four re-teleports.
        self.extra_latest: dict[str, object] = {}
        self.extra_count: dict[str, int] = {}
        for frame, topic in extra_cameras:
            self.extra_latest[frame] = None
            self.extra_count[frame] = 0
            self.create_subscription(
                Image, str(topic), self._make_extra_cb(str(frame)), 10
            )
        if self.verify_with_odom:
            self.create_subscription(Odometry, str(odom_topic), self._odom_cb, 10)

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = image_msg_to_bgr8(msg)
        self.image_count += 1

    def _make_extra_cb(self, frame: str):
        def _cb(msg: Image) -> None:
            self.extra_latest[frame] = image_msg_to_bgr8(msg)
            self.extra_count[frame] = self.extra_count.get(frame, 0) + 1
        return _cb

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
        # Long grid captures have occasionally seen one transient service timeout
        # after thousands of successful teleports.  Retrying the *same* idempotent
        # pose request is safer than discarding a 15k-frame capture; downstream
        # membership and stale-frame gates still fail the capture closed.
        failures = []
        for attempt in range(1, 4):
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
            timed_out = False
            while rclpy.ok() and not future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                if (time.monotonic() - start) > 10.0:
                    timed_out = True
                    break
            if timed_out:
                failures.append(f'attempt {attempt}: response timeout')
            else:
                result = future.result()
                if result is not None and bool(getattr(result, 'success', False)):
                    return
                failures.append(f'attempt {attempt}: service returned failure')
            if attempt < 3:
                self.get_logger().warning(
                    f'Set-pose transient at x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}; '
                    f'{failures[-1]}; retrying')
                # Spin briefly so a late response is drained before issuing the
                # identical request again.
                retry_until = time.monotonic() + 0.5
                while rclpy.ok() and time.monotonic() < retry_until:
                    rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            f'Set pose failed after 3 attempts at x={x:.3f}, y={y:.3f}, '
            f'yaw={yaw:.3f}: {"; ".join(failures)}')

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
                message = f'Robot pose did not converge to commanded sample x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
                if self.warn_on_odom_mismatch:
                    if self.latest_odom_xyyaw is None:
                        self.get_logger().warn(
                            f'{message}; no /odom pose received, continuing because --warn-on-odom-mismatch is set'
                        )
                    else:
                        ox, oy, oyaw = self.latest_odom_xyyaw
                        self.get_logger().warn(
                            f'{message}; latest /odom=({ox:.3f}, {oy:.3f}, {oyaw:.3f}), '
                            'continuing because --warn-on-odom-mismatch is set'
                        )
                else:
                    raise RuntimeError(message)

        before_extra = {f: int(c) for f, c in self.extra_count.items()}
        deadline = time.monotonic() + max(self.image_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            primary_ready = (self.image_count >= (before + self.min_new_frames)
                             and self.latest_image is not None)
            extras_ready = all(
                self.extra_count.get(f, 0) >= before_extra[f] + self.min_new_frames
                and self.extra_latest.get(f) is not None
                for f in before_extra
            )
            if primary_ready and extras_ready:
                return self._snapshot()
        if self.latest_image is None:
            raise RuntimeError('No camera image available after teleport')
        # Timed out waiting for one or more extras; the caller records which frames are
        # stale rather than silently pairing an old image with a new pose.
        return self._snapshot(before_extra=before_extra)

    def _snapshot(self, before_extra: dict | None = None):
        """Latest frame per camera. Frames that did not refresh are returned as None."""

        out = {None: self.latest_image.copy() if self.latest_image is not None else None}
        for frame, image in self.extra_latest.items():
            fresh = True
            if before_extra is not None:
                fresh = self.extra_count.get(frame, 0) > before_extra.get(frame, 0)
            out[frame] = image.copy() if (image is not None and fresh) else None
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture a canonical teleport-sampled visibility dataset.')
    parser.add_argument('--world', default='warehouse_aws.world.sdf')
    parser.add_argument('--world-profiles', default=str((Path(__file__).resolve().parents[2] / 'src' / 'experiments' / 'config' / 'world_profiles.yaml').resolve()))
    parser.add_argument('--out', default=str(CURRENT_CAPTURE_DIR))
    parser.add_argument('--sample-nx', type=int, default=15)
    parser.add_argument('--sample-ny', type=int, default=15)
    parser.add_argument('--yaw-rad', type=float, default=0.0, help='Base yaw in radians. Preserves the legacy single-heading capture when --yaw-samples=1.')
    parser.add_argument('--yaw-samples', type=int, default=4, help='Number of evenly spaced robot headings to capture at each (x,y).')
    parser.add_argument('--yaw-list-rad', default='', help='Optional explicit whitespace/comma-separated yaw list in radians; overrides --yaw-samples.')
    parser.add_argument('--wall-margin-m', type=float, default=0.45)
    parser.add_argument('--skip-region-filter', action='store_true',
                        help='Disable known_2d_regions traversability filter even when the profile defines regions.')
    parser.add_argument('--region-shrink-m', type=float, default=0.05,
                        help='Shrink each traversable region boundary inward before testing grid points.')
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--settle-s', type=float, default=0.35)
    parser.add_argument('--image-timeout-s', type=float, default=2.0)
    parser.add_argument('--image-topic', default='/external_camera/image_raw')
    parser.add_argument('--ready-timeout-s', type=float, default=15.0)
    parser.add_argument('--camera-frame', default='external_camera')
    parser.add_argument(
        '--extra-cameras', default='',
        help=('Comma-separated frame=topic pairs captured on the SAME teleport, e.g. '
              '"external_camera_b=/external_camera_b/image_raw,...". Empty (default) '
              'preserves the single-camera capture byte for byte.'),
    )
    parser.add_argument(
        '--no-previews', action='store_true',
        help=('Skip the annotated preview JPEGs. They double disk for no analytical value '
              'on large grids; the oracle pixel is already a CSV column.'),
    )
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--pose-timeout-s', type=float, default=2.0)
    parser.add_argument('--pose-tolerance-m', type=float, default=0.05)
    parser.add_argument('--yaw-tolerance-rad', type=float, default=0.20)
    parser.add_argument('--verify-with-odom', action='store_true', help='Require /odom to converge to each commanded teleport pose before capture.')
    parser.add_argument('--warn-on-odom-mismatch', action='store_true', help='With --verify-with-odom, warn and continue instead of aborting if /odom does not match the commanded world pose.')
    parser.add_argument('--min-new-frames', type=int, default=1, help='Minimum number of fresh camera frames to wait for after each teleport.')
    parser.add_argument('--target-height-m', type=float, default=0.0)
    parser.add_argument('--position-list-json', default='',
                        help='Optional explicit JSON list of [x,y] positions to capture '
                             '(overrides the grid sampler). For targeted column infill without a full recapture.')
    args = parser.parse_args()

    profile, intrinsics, world_path, camera_pose = load_profile(str(args.world_profiles), str(args.world))
    vis = dict(profile.get('visibility_defaults') or {})
    known_regions = list(profile.get('known_2d_regions') or [])
    traversable_regions: list[dict] | None = None
    if known_regions and not args.skip_region_filter:
        traversable_regions = [r for r in known_regions if str(r.get('type', '')) == 'traversable']
        if not traversable_regions:
            traversable_regions = None
    if str(args.position_list_json or '').strip():
        positions = [(float(p[0]), float(p[1])) for p in json.loads(args.position_list_json)]
        print(f'[capture] using explicit position list: {len(positions)} positions (grid sampler bypassed)')
    else:
        positions = _sample_positions(
            vis,
            sample_nx=int(args.sample_nx),
            sample_ny=int(args.sample_ny),
            wall_margin_m=float(args.wall_margin_m),
            traversable_regions=traversable_regions,
            region_shrink_m=float(args.region_shrink_m),
        )
    yaws = _sample_yaws(
        yaw_rad=float(args.yaw_rad),
        yaw_samples=int(args.yaw_samples),
        yaw_list_rad=str(args.yaw_list_rad),
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
    occlusion_scene = parse_occlusion_scene_from_world(str(world_path), geometry_tags=('collision',))

    rows: list[dict[str, str]] = []
    image_hashes: list[str] = []
    #: Teleports where a camera did not deliver a fresh frame in time. Recorded rather
    #: than dropped silently, so coverage gaps are auditable.
    stale_views: list[dict] = []
    rclpy.init()
    extra_cameras = tuple(
        tuple(part.split('=', 1))  # type: ignore[misc]
        for part in (p.strip() for p in str(args.extra_cameras).split(','))
        if part and '=' in part
    )
    extra_models = {}
    for frame, _topic in extra_cameras:
        # load_profile resolves the mount for a named camera model in the same world.
        _p, intr_e, _w, pose_e = load_profile(
            str(args.world_profiles), str(args.world), camera_model=str(frame)
        )
        pos_e = np.asarray(pose_e[:3], dtype=float)
        extra_models[frame] = ObliqueCameraModel(
            cam_pos=pos_e,
            look_at=compute_look_at_from_pose(pos_e, pose_e[3], pose_e[4], pose_e[5]),
            img_width=int(intr_e['img_width']),
            img_height=int(intr_e['img_height']),
            fov_h_rad=float(intr_e['fov_h_rad']),
        )
    if extra_cameras:
        print(f'[capture] multicam: {1 + len(extra_cameras)} cameras per teleport '
              f'({args.camera_frame} + {[f for f, _t in extra_cameras]})')

    node = TeleportImageCapture(
        world_name=str(profile['world_name']),
        image_topic=str(args.image_topic),
        odom_topic=str(args.odom_topic),
        extra_cameras=extra_cameras,
        robot_z=float(args.robot_z),
        settle_s=float(args.settle_s),
        image_timeout_s=float(args.image_timeout_s),
        pose_timeout_s=float(args.pose_timeout_s),
        pose_tolerance_m=float(args.pose_tolerance_m),
        yaw_tolerance_rad=float(args.yaw_tolerance_rad),
        verify_with_odom=bool(args.verify_with_odom),
        min_new_frames=int(args.min_new_frames),
        warn_on_odom_mismatch=bool(args.warn_on_odom_mismatch),
    )
    try:
        node.wait_for_ready(timeout_s=float(args.ready_timeout_s))
        sample_id = 0
        for position_idx, (x, y) in enumerate(positions):
            for heading_idx, yaw in enumerate(yaws):
                frames = node.capture_image_at_pose(x=x, y=y, yaw=float(yaw))
                # One teleport, one row per camera. `None` keys the primary camera so the
                # single-camera path below is unchanged when --extra-cameras is empty.
                views = [(None, str(args.camera_frame), camera)]
                views += [(f, f, extra_models[f]) for f, _t in extra_cameras]

                for key, frame_name, cam_model in views:
                    image = frames.get(key)
                    if image is None:
                        stale_views.append(
                            {'sample_id': sample_id, 'camera_frame': frame_name,
                             'x': float(x), 'y': float(y), 'theta': float(yaw)}
                        )
                        continue
                    suffix = '' if key is None else f'_{frame_name}'
                    stem = f'{sample_id:06d}_xy{position_idx:04d}_h{heading_idx:02d}{suffix}'
                    image_path = output_dir / 'images' / f'{stem}.jpg'
                    preview_path = output_dir / 'previews' / f'{stem}.jpg'
                    cv2.imwrite(str(image_path), image)
                    image_hashes.append(hashlib.md5(image_path.read_bytes()).hexdigest())

                    bottom_u, bottom_v, _ = cam_model.world_to_pixel(float(x), float(y), 0.0)
                    target_xyz = np.asarray(
                        [float(x), float(y), float(args.target_height_m)], dtype=float)
                    occluded = segment_occluded(
                        occlusion_scene.prisms, cam_model.cam_pos, target_xyz)
                    in_frame = (math.isfinite(bottom_u) and math.isfinite(bottom_v)
                                and 0.0 <= bottom_u < image.shape[1]
                                and 0.0 <= bottom_v < image.shape[0])
                    if not in_frame:
                        oracle_visible, oracle_reason = 0, 'outside_image'
                    elif occluded:
                        oracle_visible, oracle_reason = 0, 'occluded'
                    else:
                        oracle_visible, oracle_reason = 1, 'visible'

                    preview = image if args.no_previews else image.copy()
                    if (not args.no_previews) and math.isfinite(bottom_u) and math.isfinite(bottom_v):
                        color = (0, 255, 0) if oracle_visible else (0, 0, 255)
                        cv2.circle(preview, (int(round(bottom_u)), int(round(bottom_v))),
                                   5, color, -1)
                    if not args.no_previews:
                        cv2.putText(
                            preview,
                            f'sample={sample_id} {frame_name} xy={position_idx} '
                            f'h={heading_idx} oracle={oracle_visible} reason={oracle_reason}',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                            cv2.LINE_AA,
                        )
                    if not args.no_previews:
                        cv2.imwrite(str(preview_path), preview)

                    rows.append({
                        'sample_id': str(sample_id),
                        'image_path': repo_relative(image_path, output_dir),
                        'preview_path': repo_relative(preview_path, output_dir),
                        'x': f'{float(x):.8f}',
                        'y': f'{float(y):.8f}',
                        'theta': f'{float(yaw):.8f}',
                        'timestamp': f'{time.time():.6f}',
                        'world': str(args.world),
                        'camera_frame': frame_name,
                        'oracle_visible': str(int(oracle_visible)),
                        'oracle_bottom_u': '' if not math.isfinite(bottom_u) else f'{float(bottom_u):.8f}',
                        'oracle_bottom_v': '' if not math.isfinite(bottom_v) else f'{float(bottom_v):.8f}',
                        'oracle_occlusion_reason': oracle_reason,
                        'segmentation_path': '',
                        'labels_map_path': '',
                        'robot_label': '',
                    })
                sample_id += 1
                if sample_id % 100 == 0:
                    print(f'[capture] {sample_id}/{len(positions) * len(yaws)} poses, '
                          f'{len(rows)} rows, {len(stale_views)} stale views', flush=True)
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
        'capture_method': 'grid_teleport',
        'camera_frame': str(args.camera_frame),
        'extra_camera_frames': [f for f, _t in extra_cameras],
        'extra_camera_mounts': {
            f: [float(v) for v in load_profile(
                str(args.world_profiles), str(args.world), camera_model=str(f))[3]]
            for f, _t in extra_cameras
        },
        'stale_views': stale_views,
        'n_stale_views': len(stale_views),
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
        'heading_sampling': {
            'yaw_base_rad': float(args.yaw_rad),
            'yaw_samples': int(len(yaws)),
            'yaw_list_rad': [float(v) for v in yaws],
            'explicit_yaw_list_rad': str(args.yaw_list_rad).strip(),
            'samples_per_xy': int(len(yaws)),
            'position_count': int(len(positions)),
            'verify_with_odom': bool(args.verify_with_odom),
            'warn_on_odom_mismatch': bool(args.warn_on_odom_mismatch),
        },
        'oracle_source': 'geometry',
        'oracle_target_height_m': float(args.target_height_m),
        'oracle_reference_point': 'ground_plane_robot_pose',
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
            'When yaw_samples > 1, repeated rows at the same (x, y) are intentional and downstream GP targets average them into a heading-marginal 2D detectability target.',
        ],
    }
    write_manifest(output_dir / 'capture_manifest.json', manifest)
    print(f'Wrote capture to {output_dir}')
    print(f'Samples: {len(rows)}')
    print(f'Positions: {len(positions)}')
    print(f'Headings per position: {len(yaws)}')
    if duplicate_image_count > 0:
        print(
            'Warning: duplicate frames detected during capture '
            f'({duplicate_image_count} duplicates across {len(rows)} samples).'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
