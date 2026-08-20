#!/usr/bin/env python3
"""Capture a front/rear marker keypoint dataset by teleporting the robot.

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


def _marker_hits(
    image: np.ndarray,
    background: np.ndarray | None,
    uv: np.ndarray,
    kind: str,
    *,
    half: int = 4,
    change_threshold: float = 25.0,
) -> int:
    """Count pixels near ``uv`` that actually render the marker disk.

    A projected keypoint says where the marker *would* be; it says nothing about
    whether a rack is in the way. Two conditions must hold for a pixel to count:

    1. it carries the marker's colour (cyan front / magenta rear), and
    2. it differs from the same pixel in a robot-free background frame.

    Condition 1 alone is not enough: the blue rack rails anti-alias against the
    yellow racks into pixels that pass any colour test loose enough to survive a
    10 m view. Those pixels are static, so condition 2 removes them.
    """
    x0, y0 = int(round(float(uv[0]))) - half, int(round(float(uv[1]))) - half
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    x0, y0 = max(0, x0), max(0, y0)
    win = image[y0:y1, x0:x1].astype(float)
    if win.size == 0:
        return 0
    b, g, r = win[..., 0], win[..., 1], win[..., 2]
    if kind == 'front':      # cyan: blue and green together, no red
        hit = (b > 120.0) & (r < 0.35 * b) & (g > 0.85 * b)
    else:                    # rear, magenta: blue and red together, no green
        hit = (b > 120.0) & (g < 0.35 * b) & (r > 0.85 * b)
    if background is not None:
        bg = background[y0:y1, x0:x1].astype(float)
        if bg.shape == win.shape:
            hit &= np.max(np.abs(win - bg), axis=-1) > float(change_threshold)
    return int(np.count_nonzero(hit))


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
    """Teleport the robot and collect one frame per camera at each pose.

    Every camera is read at the same pose, so a four-camera world costs the same
    teleports as a one-camera world instead of four separate passes.
    """

    def __init__(
        self, *, world_name: str, image_topics: dict[str, str],
        robot_z: float, settle_s: float, image_timeout_s: float,
    ):
        super().__init__('capture_projected_keypoint_dataset')
        self.world_name = str(world_name)
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.image_timeout_s = float(image_timeout_s)
        self.service_name = f'/world/{self.world_name}/set_pose'
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.cameras = list(image_topics)
        self.image_count = {name: 0 for name in self.cameras}
        self.latest_image: dict[str, np.ndarray | None] = {name: None for name in self.cameras}
        for name, topic in image_topics.items():
            self.create_subscription(
                Image, str(topic), lambda msg, cam=name: self._image_cb(cam, msg), 10,
            )

    def _image_cb(self, camera: str, msg: Image) -> None:
        self.latest_image[camera] = image_msg_to_bgr8(msg)
        self.image_count[camera] += 1

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(float(timeout_s), 0.0)
        service_ready = False
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (not service_ready) and self.client.wait_for_service(timeout_sec=0.05):
                service_ready = True
            if service_ready and all(self.latest_image[c] is not None for c in self.cameras):
                return
        missing = [c for c in self.cameras if self.latest_image[c] is None]
        raise RuntimeError(
            f'Timed out waiting for {self.service_name} and a first frame from every camera. '
            f'Still silent: {missing or "none"}. Launch the simulator first.'
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

    def capture_images_at_pose(self, *, x: float, y: float, yaw: float) -> dict[str, np.ndarray]:
        """Teleport, then wait for a *fresh* frame from every camera."""
        before = dict(self.image_count)
        self._set_robot_pose(x=x, y=y, yaw=yaw)
        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        while rclpy.ok() and time.monotonic() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)
        deadline = time.monotonic() + max(self.image_timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(self.image_count[c] > before[c] for c in self.cameras):
                break
        # A camera that did not tick in time keeps its last frame; that frame is
        # from an earlier pose, so it must not be written as this pose's label.
        return {
            c: img.copy() for c in self.cameras
            if (img := self.latest_image[c]) is not None and self.image_count[c] > before[c]
        }


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
    parser.add_argument('--repeats', type=int, default=1,
                        help='Fresh captures per nominal pose. Repeats inherit the nominal pose split.')
    parser.add_argument('--position-jitter-m', type=float, default=0.0,
                        help='Balanced micro-jitter radius around each nominal pose.')
    parser.add_argument('--yaw-jitter-rad', type=float, default=0.0,
                        help='Balanced yaw micro-jitter around each nominal heading.')
    parser.add_argument('--session-id', default='session_0',
                        help='Independent simulator/capture session identifier written per row.')
    parser.add_argument('--plan-seed', type=int, default=0,
                        help='Rotates the balanced repeat pattern without changing grouped splits.')
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--settle-s', type=float, default=0.35)
    parser.add_argument('--image-timeout-s', type=float, default=2.0)
    parser.add_argument('--preview-count', type=int, default=80)
    parser.add_argument('--min-bbox-size-px', type=float, default=18.0)
    parser.add_argument('--box-length', type=float, default=0.16)
    parser.add_argument('--box-width', type=float, default=0.16)
    parser.add_argument('--box-height', type=float, default=0.20)
    parser.add_argument('--cameras', default='external_camera',
                        help='Comma-separated camera model names in the world SDF. Every camera '
                             'is read at the same teleported pose, so four cameras cost the same '
                             'teleports as one.')
    parser.add_argument('--image-topics', default='',
                        help='Comma-separated topics matching --cameras. Defaults to '
                             '/<camera>/image_raw for each.')
    parser.add_argument('--marker-window-half', type=int, default=4,
                        help='Half-width in px of the window searched for each rendered marker.')
    parser.add_argument('--min-marker-hits', type=int, default=2,
                        help='Marker pixels needed before a keypoint counts as rendered-visible. '
                             '0 disables the check and labels every projected keypoint as visible.')
    parser.add_argument('--background-x', type=float, default=0.0)
    parser.add_argument('--background-y', type=float, default=25.0,
                        help='Where to park the robot for the robot-free background frame.')
    args = parser.parse_args()

    camera_names = [c.strip() for c in str(args.cameras).split(',') if c.strip()]
    if not camera_names:
        raise RuntimeError('--cameras must name at least one camera')
    if str(args.image_topics).strip():
        topic_list = [t.strip() for t in str(args.image_topics).split(',') if t.strip()]
        if len(topic_list) != len(camera_names):
            raise RuntimeError('--image-topics must have one entry per --cameras entry')
    else:
        topic_list = [f'/{name}/image_raw' for name in camera_names]
    image_topics = dict(zip(camera_names, topic_list))

    profile, _intrinsics, world_path, _first_pose = load_profile(
        str(args.world_profiles), str(args.world), camera_names[0],
    )
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis.get('visibility_map_min_x', -3.0)) + float(args.wall_margin)
    xmax = float(vis.get('visibility_map_max_x', 3.0)) - float(args.wall_margin)
    ymin = float(vis.get('visibility_map_min_y', -3.0)) + float(args.wall_margin)
    ymax = float(vis.get('visibility_map_max_y', 3.0)) - float(args.wall_margin)
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError('Invalid sampling bounds after wall margin')

    cameras: dict[str, ObliqueCameraModel] = {}
    camera_poses: dict[str, list[float]] = {}
    for name in camera_names:
        _, cam_intrinsics, _, pose = load_profile(str(args.world_profiles), str(args.world), name)
        look_at = compute_look_at_from_pose(pose[:3], pose[3], pose[4], pose[5])
        cameras[name] = ObliqueCameraModel(
            cam_pos=pose[:3], look_at=look_at,
            img_width=int(cam_intrinsics['img_width']), img_height=int(cam_intrinsics['img_height']),
            fov_h_rad=float(cam_intrinsics['fov_h_rad']),
        )
        camera_poses[name] = [float(v) for v in pose]

    geom = DEFAULT_MARKER_GEOMETRY
    xs = np.linspace(xmin, xmax, max(int(args.sample_nx), 1))
    ys = np.linspace(ymin, ymax, max(int(args.sample_ny), 1))
    yaw_samples = max(int(args.yaw_samples), 1)
    yaws = np.asarray(evenly_spaced_yaws(yaw_samples), dtype=float)
    pose_records = build_pose_records(xs, ys, yaws)
    base_split_labels = assign_splits(
        pose_records, val_fraction=float(args.val_fraction),
        split_mode=str(args.split_mode), seed=int(args.split_seed),
        spatial_block_size=int(args.spatial_block_size),
    )
    repeats = max(int(args.repeats), 1)
    planned: list[tuple[float, float, float]] = []
    planned_meta: list[dict] = []
    split_labels: list[str] = []
    for pose_idx, item in enumerate(pose_records):
        # Equally spaced directions make the perturbations sum to zero within a
        # nominal condition. Session/plan seed rotates the pattern but does not
        # change which spatial group is held out.
        phase = 2.0 * math.pi * (
            (int(args.plan_seed) * 0.61803398875 + pose_idx * 0.38196601125) % 1.0
        )
        for repeat_idx in range(repeats):
            angle = phase + 2.0 * math.pi * repeat_idx / repeats
            dx = float(args.position_jitter_m) * math.cos(angle) if repeats > 1 else 0.0
            dy = float(args.position_jitter_m) * math.sin(angle) if repeats > 1 else 0.0
            dyaw = float(args.yaw_jitter_rad) * math.sin(angle) if repeats > 1 else 0.0
            nominal_yaw = float(item['yaw'])
            planned.append((
                float(item['x']) + dx,
                float(item['y']) + dy,
                (nominal_yaw + dyaw) % (2.0 * math.pi),
            ))
            planned_meta.append({
                'session_id': str(args.session_id),
                'anchor_id': f"xy_{int(item['x_idx']):03d}_{int(item['y_idx']):03d}",
                'x_idx': int(item['x_idx']), 'y_idx': int(item['y_idx']),
                'yaw_idx': int(item['yaw_idx']), 'repeat_idx': int(repeat_idx),
                'nominal_x': float(item['x']), 'nominal_y': float(item['y']),
                'nominal_yaw_rad': nominal_yaw,
            })
            split_labels.append(str(base_split_labels[pose_idx]))

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
        world_name=str(profile['world_name']), image_topics=image_topics,
        robot_z=float(args.robot_z), settle_s=float(args.settle_s), image_timeout_s=float(args.image_timeout_s),
    )
    try:
        node.wait_for_ready(timeout_s=20.0)
        backgrounds = node.capture_images_at_pose(
            x=float(args.background_x), y=float(args.background_y), yaw=0.0,
        )
        missing_bg = [c for c in camera_names if c not in backgrounds]
        if missing_bg:
            raise RuntimeError(f'No robot-free background frame from: {missing_bg}')

        for sample_idx, (x, y, yaw) in enumerate(planned):
            meta = planned_meta[sample_idx]
            images = node.capture_images_at_pose(x=x, y=y, yaw=yaw)
            # The split belongs to the POSE, so all cameras of one pose land on the
            # same side of it. Otherwise the same pose appears in train and val and
            # "held out" means nothing.
            split = str(split_labels[sample_idx])

            front_world = _marker_world_position(
                x=x, y=y, yaw=yaw, robot_z=float(args.robot_z),
                offset_x=geom.front_x, offset_z=geom.marker_z_in_base_link,
            )
            rear_world = _marker_world_position(
                x=x, y=y, yaw=yaw, robot_z=float(args.robot_z),
                offset_x=geom.rear_x, offset_z=geom.marker_z_in_base_link,
            )

            for camera_name in camera_names:
                camera = cameras[camera_name]
                image = images.get(camera_name)
                front_uv = _project_world_point(camera, front_world)
                rear_uv = _project_world_point(camera, rear_world)
                bbox = None if image is None else _project_robot_aabb(
                    camera, x=x, y=y, yaw=yaw,
                    box_length=float(args.box_length), box_width=float(args.box_width),
                    box_height=float(args.box_height), img_w=image.shape[1], img_h=image.shape[0],
                )

                front_hits = 0
                rear_hits = 0
                reason = ''
                if image is None:
                    reason = 'no_fresh_frame'
                elif front_uv is None or rear_uv is None:
                    reason = 'keypoint_behind_camera'
                elif bbox is None:
                    reason = 'bbox_not_projectable'
                else:
                    bw = float(bbox[2] - bbox[0])
                    bh = float(bbox[3] - bbox[1])
                    if bw < float(args.min_bbox_size_px) or bh < float(args.min_bbox_size_px):
                        reason = 'bbox_too_small'
                    else:
                        front_hits = _marker_hits(
                            image, backgrounds[camera_name], front_uv, 'front',
                            half=int(args.marker_window_half),
                        )
                        rear_hits = _marker_hits(
                            image, backgrounds[camera_name], rear_uv, 'rear',
                            half=int(args.marker_window_half),
                        )
                        if max(front_hits, rear_hits) < int(args.min_marker_hits):
                            # Both markers are behind something. The projection still
                            # lands on a pixel, so without this the label would claim
                            # a robot in front of a rack that hides it.
                            reason = 'markers_not_rendered'

                if reason:
                    skipped += 1
                    diagnostics.append({
                        'sample_idx': int(sample_idx), 'camera': camera_name,
                        **meta,
                        'x': float(x), 'y': float(y), 'yaw_rad': float(yaw),
                        'accepted': 0, 'rejection_reason': reason, 'split': '',
                        'image': '', 'label': '',
                        'front_u': math.nan, 'front_v': math.nan,
                        'rear_u': math.nan, 'rear_v': math.nan,
                        'front_visible': 0, 'rear_visible': 0,
                        'front_hits': int(front_hits), 'rear_hits': int(rear_hits),
                        'preview': '',
                    })
                    continue

                assert image is not None and bbox is not None
                assert front_uv is not None and rear_uv is not None
                front_visible = _kpt_inside(front_uv, image.shape[1], image.shape[0])
                rear_visible = _kpt_inside(rear_uv, image.shape[1], image.shape[0])
                if int(args.min_marker_hits) > 0:
                    front_visible = front_visible and front_hits >= int(args.min_marker_hits)
                    rear_visible = rear_visible and rear_hits >= int(args.min_marker_hits)

                stem = f'{sample_idx:06d}_{camera_name}'
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
                    cv2.putText(preview, f'{camera_name} x={x:.2f} y={y:.2f} yaw={yaw:.2f}',
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    preview_path = out_dir / 'previews' / split / f'{stem}.jpg'
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(preview_path), preview)
                    preview_rel = str(preview_path.relative_to(out_dir))
                    preview_written += 1

                diagnostics.append({
                    'sample_idx': int(sample_idx), 'camera': camera_name,
                    **meta,
                    'x': float(x), 'y': float(y), 'yaw_rad': float(yaw),
                    'accepted': 1, 'rejection_reason': '', 'split': split,
                    'image': str(image_path.relative_to(out_dir)), 'label': str(label_path.relative_to(out_dir)),
                    'front_u': float(front_uv[0]), 'front_v': float(front_uv[1]),
                    'rear_u': float(rear_uv[0]), 'rear_v': float(rear_uv[1]),
                    'front_visible': int(front_visible), 'rear_visible': int(rear_visible),
                    'front_hits': int(front_hits), 'rear_hits': int(rear_hits),
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
                # camera_pose/look_at describe the FIRST camera, kept so
                # single-camera captures read the same as they always did.
                'camera_pose': camera_poses[camera_names[0]],
                'look_at': [float(v) for v in compute_look_at_from_pose(
                    camera_poses[camera_names[0]][:3], *camera_poses[camera_names[0]][3:6],
                )],
                'cameras': {
                    name: {
                        'pose': camera_poses[name],
                        'image_topic': image_topics[name],
                        'img_width': cameras[name].img_width,
                        'img_height': cameras[name].img_height,
                        'fov_h_rad': cameras[name].fov_h_rad,
                    }
                    for name in camera_names
                },
                'sample_nx': int(args.sample_nx), 'sample_ny': int(args.sample_ny),
                'yaw_samples': int(args.yaw_samples),
                'yaw_values_rad': [float(v) for v in yaws.tolist()],
                'planned_samples': int(len(planned)),
                'nominal_pose_count': int(len(pose_records)),
                'repeats': int(repeats),
                'position_jitter_m': float(args.position_jitter_m),
                'yaw_jitter_rad': float(args.yaw_jitter_rad),
                'session_id': str(args.session_id),
                'plan_seed': int(args.plan_seed),
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
                    '0': 'front_marker_disk_cyan',
                    '1': 'rear_marker_disk_magenta',
                },
                'robot_z': float(args.robot_z),
                'keypoint_marker_world_z': float(geom.world_z(float(args.robot_z))),
                'marker_visibility_check': {
                    'min_marker_hits': int(args.min_marker_hits),
                    'window_half_px': int(args.marker_window_half),
                    'background_pose_xy': [float(args.background_x), float(args.background_y)],
                },
                'timestamp': _timestamp(),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    with (out_dir / 'capture_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        fieldnames = [
            'sample_idx', 'camera', 'session_id', 'anchor_id', 'x_idx', 'y_idx',
            'yaw_idx', 'repeat_idx', 'nominal_x', 'nominal_y', 'nominal_yaw_rad',
            'x', 'y', 'yaw_rad', 'accepted', 'rejection_reason',
            'split', 'image', 'label',
            'front_u', 'front_v', 'rear_u', 'rear_v',
            'front_visible', 'rear_visible', 'front_hits', 'rear_hits', 'preview',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f'Wrote projected-keypoint dataset to {out_dir}')
    print(json.dumps({'accepted_samples': int(accepted), 'skipped_samples': int(skipped)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
