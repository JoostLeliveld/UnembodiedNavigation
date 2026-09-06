#!/usr/bin/env python3
"""Capture one synchronized five-camera RGB batch at each warehouse pose.

This is an observation-characterization capture, not a detector-training dataset. It keeps
every fresh RGB frame, including an empty/occluded view, so later YOLO miss and
false-positive maps have an actual image behind every attempted opportunity. Semantic labels
are optional diagnostics and are not required by the main capture.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image

REPO = Path(__file__).resolve().parents[2]
for rel in ('scripts/perception', 'src/experiments', 'src/perception', 'src/unav_common'):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from capture_yolo_dataset import (  # noqa: E402
    _camera_from_profile,
    _capture_transport_environment,
    _filter_pose_records,
    _project_robot_bbox,
    _read_uint_label_map,
    _sha1_array,
    _sha256_file,
    _stamp_ns,
)
from dataset_split_utils import build_pose_records, evenly_spaced_yaws  # noqa: E402
from experiments.core.world_profiles import load_profile  # noqa: E402
from perception.core.ros_image import image_msg_to_bgr8  # noqa: E402
from unav_common.occlusion_geometry import parse_collision_scene_from_world  # noqa: E402


CAMERAS = (
    ('camera_A', 'external_camera'),
    ('camera_B', 'external_camera_b'),
    ('camera_C', 'external_camera_c'),
    ('camera_D', 'external_camera_d'),
    ('camera_E', 'external_camera_e'),
)
FIELDS = (
    'pose_id', 'position_id', 'x_idx', 'y_idx', 'heading_id', 'repetition_id',
    'source_batch_id',
    'dataset_split', 'random_draw_index',
    'camera_id', 'camera_model', 'image', 'capture_status', 'capture_error',
    'robot_x', 'robot_y', 'robot_yaw', 'camera_range_m',
    'image_stamp_s', 'label_stamp_s', 'stamp_delta_s', 'batch_image_span_s',
    'image_sha1', 'semantic_robot_pixels', 'line_of_sight',
    'mask_x0', 'mask_y0', 'mask_x1', 'mask_y1', 'mask_bottom_u', 'mask_bottom_v',
    'expected_x0', 'expected_y0', 'expected_x1', 'expected_y1', 'nominal_in_frame',
)


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    model: str
    image_topic: str
    labels_topic: str
    camera: object
    pose: tuple[float, ...]
    image_width: int
    image_height: int


@dataclass
class Pair:
    image: np.ndarray
    labels: np.ndarray | None
    image_stamp_ns: int
    label_stamp_ns: int | None
    stamp_delta_s: float


def _closest_timestamp_batch(
    candidates: dict[str, list[Pair]], *, max_span_s: float
) -> dict[str, Pair] | None:
    """Choose the closest all-camera timestamp combination.

    Camera callbacks do not necessarily arrive in timestamp order across topics. Selecting
    each topic's latest fresh frame independently can therefore merge adjacent 5 Hz rounds
    into one nominal batch. Anchor each candidate timestamp in turn, take the closest frame
    from every camera, and retain the newest minimum-span combination.
    """
    if not candidates or any(not items for items in candidates.values()):
        return None
    anchors = sorted({item.image_stamp_ns for items in candidates.values() for item in items})
    best: dict[str, Pair] | None = None
    best_key: tuple[int, int] | None = None
    for anchor in anchors:
        selected = {
            camera_id: min(
                items,
                key=lambda item: (abs(item.image_stamp_ns - anchor), -item.image_stamp_ns),
            )
            for camera_id, items in candidates.items()
        }
        stamps = [item.image_stamp_ns for item in selected.values()]
        span_ns = max(stamps) - min(stamps)
        # First minimize the span; then prefer the newest coherent round.
        key = (span_ns, -min(stamps))
        if best_key is None or key < best_key:
            best_key, best = key, selected
    assert best_key is not None
    if best_key[0] * 1e-9 > float(max_span_s):
        return None
    return best


class FiveCameraCapture(Node):
    def __init__(
        self,
        *,
        world_name: str,
        cameras: tuple[CameraSpec, ...],
        robot_z: float,
        settle_s: float,
        timeout_s: float,
        sync_slop_s: float,
        batch_sync_slop_s: float,
        min_new_rgb: int,
        min_new_labels: int,
        buffer_size: int,
        with_semantic: bool,
    ) -> None:
        super().__init__('capture_bbox_characterization')
        self.robot_z = float(robot_z)
        self.settle_s = float(settle_s)
        self.timeout_s = float(timeout_s)
        self.sync_slop_s = float(sync_slop_s)
        self.batch_sync_slop_s = float(batch_sync_slop_s)
        self.min_new_rgb = max(int(min_new_rgb), 1)
        self.min_new_labels = max(int(min_new_labels), 1)
        self.with_semantic = bool(with_semantic)
        self.rgb_count = {spec.camera_id: 0 for spec in cameras}
        self.label_count = {spec.camera_id: 0 for spec in cameras}
        self.rgb = {spec.camera_id: deque(maxlen=max(int(buffer_size), 8)) for spec in cameras}
        self.labels = {spec.camera_id: deque(maxlen=max(int(buffer_size), 8)) for spec in cameras}
        self.last_label_error = {spec.camera_id: '' for spec in cameras}
        self.client = self.create_client(SetEntityPose, f'/world/{world_name}/set_pose')
        self.zero_cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        for spec in cameras:
            self.create_subscription(
                Image,
                spec.image_topic,
                functools.partial(self._rgb_cb, spec.camera_id),
                10,
            )
            if self.with_semantic:
                self.create_subscription(
                    Image,
                    spec.labels_topic,
                    functools.partial(self._labels_cb, spec.camera_id),
                    10,
                )

    def _rgb_cb(self, camera_id: str, msg: Image) -> None:
        self.rgb_count[camera_id] += 1
        self.rgb[camera_id].append(
            (self.rgb_count[camera_id], _stamp_ns(msg), image_msg_to_bgr8(msg))
        )

    def _labels_cb(self, camera_id: str, msg: Image) -> None:
        try:
            labels = _read_uint_label_map(msg)
        except Exception as exc:  # pragma: no cover - ROS transport path
            self.last_label_error[camera_id] = str(exc)
            return
        self.label_count[camera_id] += 1
        self.labels[camera_id].append(
            (self.label_count[camera_id], _stamp_ns(msg), labels)
        )

    def _spin(self, timeout: float = 0.04) -> None:
        self.zero_cmd.publish(Twist())
        rclpy.spin_once(self, timeout_sec=float(timeout))

    def wait_ready(self, camera_ids: tuple[str, ...], timeout_s: float = 45.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin()
            service = self.client.wait_for_service(timeout_sec=0.02)
            ready = all(self.rgb[cid] for cid in camera_ids)
            if self.with_semantic:
                ready = ready and all(self.labels[cid] for cid in camera_ids)
            if service and ready:
                return
        missing = [
            cid for cid in camera_ids
            if not self.rgb[cid] or (self.with_semantic and not self.labels[cid])
        ]
        raise RuntimeError(f'Capture transport not ready; missing camera streams: {missing}')

    def _set_pose(self, x: float, y: float, yaw: float) -> None:
        request = SetEntityPose.Request()
        request.entity = Entity(name='turtlebot3', type=Entity.MODEL)
        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(self.robot_z)
        request.pose.orientation.z = math.sin(0.5 * float(yaw))
        request.pose.orientation.w = math.cos(0.5 * float(yaw))
        future = self.client.call_async(request)
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self._spin()
        result = future.result() if future.done() else None
        if result is None or not bool(getattr(result, 'success', False)):
            raise RuntimeError(f'set_pose failed at ({x:.3f}, {y:.3f}, {yaw:.3f})')

    def _pair_candidates(
        self, camera_id: str, before_rgb: int, before_labels: int
    ) -> list[Pair]:
        if self.rgb_count[camera_id] < before_rgb + self.min_new_rgb:
            return []
        if not self.with_semantic:
            candidates = [item for item in self.rgb[camera_id] if item[0] > before_rgb]
            return [
                Pair(image.copy(), None, rgb_stamp, None, math.nan)
                for _count, rgb_stamp, image in candidates
            ]
        if self.label_count[camera_id] < before_labels + self.min_new_labels:
            return []
        paired: list[Pair] = []
        for rgb_count, rgb_stamp, image in self.rgb[camera_id]:
            if rgb_count <= before_rgb:
                continue
            best = None
            best_key = None
            for label_count, label_stamp, labels in self.labels[camera_id]:
                if label_count <= before_labels:
                    continue
                delta = abs(rgb_stamp - label_stamp) * 1e-9
                if delta > self.sync_slop_s:
                    continue
                key = (min(rgb_count, label_count), -delta)
                if best_key is None or key > best_key:
                    best_key = key
                    best = Pair(image.copy(), labels.copy(), rgb_stamp, label_stamp, delta)
            if best is not None:
                paired.append(best)
        return paired

    def _batch(
        self,
        camera_ids: tuple[str, ...],
        before_rgb: dict[str, int],
        before_labels: dict[str, int],
    ) -> dict[str, Pair] | None:
        candidates = {
            camera_id: self._pair_candidates(
                camera_id, before_rgb[camera_id], before_labels[camera_id]
            )
            for camera_id in camera_ids
        }
        return _closest_timestamp_batch(candidates, max_span_s=self.batch_sync_slop_s)

    def capture(self, x: float, y: float, yaw: float, camera_ids: tuple[str, ...]) -> dict[str, Pair]:
        self._set_pose(x, y, yaw)
        settle_deadline = time.monotonic() + self.settle_s
        while rclpy.ok() and time.monotonic() < settle_deadline:
            self._spin()
        before_rgb = dict(self.rgb_count)
        before_labels = dict(self.label_count)
        deadline = time.monotonic() + self.timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self._spin()
            pairs = self._batch(camera_ids, before_rgb, before_labels)
            if pairs is not None:
                return pairs
        missing = [
            cid for cid in camera_ids
            if not self._pair_candidates(cid, before_rgb[cid], before_labels[cid])
        ]
        raise RuntimeError(
            'fresh synchronized frame timeout; '
            f'missing={missing}, required_batch_span_s<={self.batch_sync_slop_s}'
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _add_rgb_read_noise(
    image: np.ndarray,
    *,
    stddev_dn: float,
    seed: int,
    pose_id: int,
    repetition_id: int,
    camera_index: int,
) -> np.ndarray:
    """Apply a reproducible independent camera read-noise realization.

    ``stddev_dn`` is in 8-bit digital-number units. Seed components identify one
    camera-pose repetition, so resumed captures reproduce exactly the same image.
    """
    stddev = float(stddev_dn)
    if not math.isfinite(stddev) or stddev < 0.0:
        raise ValueError('rgb-noise-stddev-dn must be finite and non-negative')
    if stddev == 0.0:
        return image.copy()
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), int(pose_id), int(repetition_id), int(camera_index)
    ]))
    noisy = image.astype(np.float32) + rng.normal(0.0, stddev, size=image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _mask_geometry(labels: np.ndarray, robot_label: int) -> dict[str, float | int]:
    ys, xs = np.nonzero(labels == int(robot_label))
    if not xs.size:
        return {
            'semantic_robot_pixels': 0, 'line_of_sight': 0,
            'mask_x0': math.nan, 'mask_y0': math.nan,
            'mask_x1': math.nan, 'mask_y1': math.nan,
            'mask_bottom_u': math.nan, 'mask_bottom_v': math.nan,
        }
    bottom = int(ys.max())
    bottom_x = xs[ys >= bottom - 2]
    return {
        'semantic_robot_pixels': int(xs.size), 'line_of_sight': 1,
        'mask_x0': int(xs.min()), 'mask_y0': int(ys.min()),
        'mask_x1': int(xs.max()) + 1, 'mask_y1': int(ys.max()) + 1,
        'mask_bottom_u': float(np.mean(bottom_x)), 'mask_bottom_v': float(bottom + 1),
    }


def _expected_geometry(spec: CameraSpec, x: float, y: float, yaw: float) -> dict[str, float | int]:
    box = _project_robot_bbox(
        spec.camera,
        x=float(x), y=float(y), yaw=float(yaw), z=0.0,
        box_length=0.80, box_width=0.55, box_height=0.35,
    )
    if box is None:
        return {
            'expected_x0': math.nan, 'expected_y0': math.nan,
            'expected_x1': math.nan, 'expected_y1': math.nan, 'nominal_in_frame': 0,
        }
    x0, y0, x1, y1 = (float(value) for value in box)
    intersects = x1 > 0 and y1 > 0 and x0 < spec.image_width and y0 < spec.image_height
    return {
        'expected_x0': x0, 'expected_y0': y0,
        'expected_x1': x1, 'expected_y1': y1,
        'nominal_in_frame': int(intersects),
    }


def _camera_specs(world_profiles: Path, world: str) -> tuple[tuple[CameraSpec, ...], dict, str]:
    specs = []
    base_profile = None
    world_path = ''
    for camera_id, model in CAMERAS:
        profile, intrinsics, resolved_world, pose = load_profile(
            str(world_profiles), str(world), camera_model=model
        )
        base_profile = profile if base_profile is None else base_profile
        world_path = resolved_world
        specs.append(CameraSpec(
            camera_id=camera_id,
            model=model,
            image_topic=f'/{model}/image_raw',
            labels_topic=f'/{model}/segmentation/labels_map',
            camera=_camera_from_profile(pose, intrinsics),
            pose=tuple(float(value) for value in pose),
            image_width=int(intrinsics['img_width']),
            image_height=int(intrinsics['img_height']),
        ))
    assert base_profile is not None
    return tuple(specs), base_profile, world_path


def _pose_plan(profile: dict, world_path: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    vis = dict(profile.get('visibility_defaults') or {})
    xmin = float(vis['visibility_map_min_x']) + float(args.wall_margin)
    xmax = float(vis['visibility_map_max_x']) - float(args.wall_margin)
    ymin = float(vis['visibility_map_min_y']) + float(args.wall_margin)
    ymax = float(vis['visibility_map_max_y']) - float(args.wall_margin)
    known = list(profile.get('known_2d_regions') or [])
    traversable = [
        item for item in known
        if str(item.get('type', '')).strip().lower() == 'traversable'
    ]
    excluded = [
        item for item in known
        if str(item.get('type', '')).strip().lower() not in {'traversable', 'site_boundary'}
    ]
    prisms = parse_collision_scene_from_world(world_path).prisms
    def filter_records(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
        return _filter_pose_records(
            records,
            traversable_regions=traversable,
            excluded_regions=excluded,
            region_shrink_m=float(args.region_shrink_m),
            collision_prisms=tuple(prisms),
            collision_clearance_m=float(args.collision_clearance_m),
            camera_xy=(0.0, 0.0),
            min_camera_range_m=0.0,
            max_camera_range_m=float('inf'),
        )

    if args.pose_file is not None:
        pose_file = args.pose_file.expanduser().resolve()
        payload = json.loads(pose_file.read_text(encoding='utf-8'))
        source = payload.get('poses') if isinstance(payload, dict) else payload
        if not isinstance(source, list) or not source:
            raise ValueError('--pose-file must contain a non-empty JSON list or {"poses": [...]}')
        records = []
        positions: dict[tuple[float, float], int] = {}
        for index, item in enumerate(source):
            if not isinstance(item, dict):
                raise ValueError(f'pose-file entry {index} is not an object')
            x, y, yaw = (float(item[key]) for key in ('x', 'y', 'yaw'))
            position_key = (x, y)
            if position_key not in positions:
                positions[position_key] = len(positions)
            records.append({
                'x': x, 'y': y, 'yaw': yaw,
                'x_idx': index, 'y_idx': index, 'yaw_idx': index,
                'position_id': positions[position_key],
                'dataset_split': str(item.get('stratum', 'commissioning')),
                'random_draw_index': '',
            })
        kept, counts = filter_records(records)
        if len(kept) != len(records):
            rejected = sorted(set(range(len(records))) - {int(row['x_idx']) for row in kept})
            raise RuntimeError(f'pose-file contains collision-invalid entries: {rejected}')
        return kept, {
            'bounds': [xmin, xmax, ymin, ymax],
            'filter_counts': counts,
            'sampling': 'predeclared_pose_file',
            'pose_file': str(pose_file),
            'pose_file_sha256': _sha256(pose_file),
            'pose_labels': [str(item.get('stratum', f'pose_{index}'))
                            for index, item in enumerate(source)],
        }

    random_per_split = int(args.random_poses_per_split)
    if random_per_split > 0:
        if int(args.max_poses) > 0:
            raise ValueError('--max-poses cannot be combined with --random-poses-per-split')
        tile_m = float(args.spatial_holdout_tile_m)
        if tile_m <= 0.0:
            raise ValueError('--spatial-holdout-tile-m must be positive')
        rng = np.random.default_rng(int(args.random_seed))
        draw_count = max(20 * random_per_split, 10000)
        records = []
        for draw_index in range(draw_count):
            records.append({
                'x': float(rng.uniform(xmin, xmax)),
                'y': float(rng.uniform(ymin, ymax)),
                'yaw': float(rng.uniform(-math.pi, math.pi)),
                'x_idx': draw_index,
                'y_idx': draw_index,
                'yaw_idx': draw_index,
                'random_draw_index': draw_index,
            })
        kept, counts = filter_records(records)
        pools = {'train': [], 'validation': []}
        for record in kept:
            tile_x = math.floor((float(record['x']) - xmin) / tile_m)
            tile_y = math.floor((float(record['y']) - ymin) / tile_m)
            split = 'train' if (tile_x + tile_y) % 2 == 0 else 'validation'
            record['dataset_split'] = split
            pools[split].append(record)
        shortages = {
            split: len(rows) for split, rows in pools.items()
            if len(rows) < random_per_split
        }
        if shortages:
            raise RuntimeError(
                f'Not enough collision-free random poses for requested split sizes: {shortages}'
            )
        chosen = pools['train'][:random_per_split] + pools['validation'][:random_per_split]
        order = rng.permutation(len(chosen))
        kept = [chosen[int(index)] for index in order]
        for position_id, record in enumerate(kept):
            record['position_id'] = position_id
        return kept, {
            'bounds': [xmin, xmax, ymin, ymax],
            'filter_counts': counts,
            'sampling': 'seeded_continuous_uniform_xy_yaw_after_collision_filter',
            'random_seed': int(args.random_seed),
            'random_poses_per_split': random_per_split,
            'spatial_split': f'{tile_m:g} m checkerboard tiles',
            'spatial_holdout_tile_m': tile_m,
            'random_candidate_draws': draw_count,
            'eligible_by_split': {key: len(value) for key, value in pools.items()},
        }

    records = build_pose_records(
        np.linspace(xmin, xmax, int(args.sample_nx)),
        np.linspace(ymin, ymax, int(args.sample_ny)),
        np.asarray(evenly_spaced_yaws(int(args.yaw_samples)), dtype=float),
    )
    kept, counts = filter_records(records)
    if int(args.max_poses) > 0:
        kept = kept[:int(args.max_poses)]
    positions = {}
    for record in kept:
        key = (int(record['x_idx']), int(record['y_idx']))
        if key not in positions:
            positions[key] = len(positions)
        record['position_id'] = positions[key]
    return kept, {'bounds': [xmin, xmax, ymin, ymax], 'filter_counts': counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='warehouse_v2.world.sdf')
    parser.add_argument('--world-profiles', type=Path, default=REPO / 'src/experiments/config/world_profiles.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sample-nx', type=int, default=33)
    parser.add_argument('--sample-ny', type=int, default=28)
    parser.add_argument('--yaw-samples', type=int, default=8)
    parser.add_argument(
        '--pose-file', type=Path,
        help='Predeclared JSON poses [{x,y,yaw,stratum}, ...]; overrides grid/random sampling.',
    )
    parser.add_argument(
        '--random-poses-per-split', type=int, default=0,
        help='Use this many continuous random poses in each frozen train/validation split.',
    )
    parser.add_argument('--random-seed', type=int, default=20260901)
    parser.add_argument('--spatial-holdout-tile-m', type=float, default=2.0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-poses', type=int, default=0)
    parser.add_argument('--wall-margin', type=float, default=0.65)
    parser.add_argument('--region-shrink-m', type=float, default=0.05)
    parser.add_argument('--collision-clearance-m', type=float, default=0.25)
    parser.add_argument('--robot-z', type=float, default=0.0)
    parser.add_argument('--robot-label', type=int, default=23)
    parser.add_argument('--settle-s', type=float, default=0.80)
    parser.add_argument('--image-timeout-s', type=float, default=8.0)
    parser.add_argument('--sync-slop-ms', type=float, default=60.0)
    parser.add_argument(
        '--batch-sync-slop-ms', type=float, default=50.0,
        help='Maximum timestamp span across the five selected RGB frames.',
    )
    parser.add_argument('--min-new-rgb-frames', type=int, default=3)
    parser.add_argument('--min-new-label-frames', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=90)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument(
        '--rgb-noise-stddev-dn', type=float, default=0.0,
        help=(
            'Optional independent Gaussian RGB read noise, in 8-bit digital-number units, '
            'applied after ROS transport and before hashing/saving/detector inference.'
        ),
    )
    parser.add_argument(
        '--rgb-noise-seed', type=int, default=20260903,
        help='Base seed for --rgb-noise-stddev-dn; each camera-pose-repeat has its own stream.',
    )
    parser.add_argument(
        '--with-semantic', action='store_true',
        help='Also wait for semantic-label images. Optional diagnostic; not used by YOLO.',
    )
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume a status=running capture after its last successful pose batch.',
    )
    args = parser.parse_args()

    transport = _capture_transport_environment(allow_unisolated_transport=False)
    specs, profile, world_path = _camera_specs(args.world_profiles, args.world)
    poses, plan_meta = _pose_plan(profile, world_path, args)
    planned_rows = len(poses) * max(int(args.repeats), 1) * len(specs)
    plan = {
        'world': str(args.world),
        'pose_count': len(poses),
        'position_count': len({int(row['position_id']) for row in poses}),
        'heading_count': (
            'predeclared' if args.pose_file is not None
            else ('continuous' if int(args.random_poses_per_split) > 0 else int(args.yaw_samples))
        ),
        'camera_count': len(specs),
        'repeats': max(int(args.repeats), 1),
        'planned_rows': planned_rows,
        'batch_sync_slop_ms': float(args.batch_sync_slop_ms),
        **plan_meta,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    out = args.out.expanduser().resolve()
    manifest_path = out / 'capture_manifest.json'
    script = Path(__file__).resolve()
    helper = REPO / 'scripts/perception/capture_yolo_dataset.py'
    expected_manifest = {
        'status': 'running',
        'schema': 'bbox_characterization_capture.v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'plan': plan,
        'transport': transport,
        'world_path': str(Path(world_path).resolve()),
        'world_sha256': _sha256(Path(world_path)),
        'world_profiles_path': str(args.world_profiles.resolve()),
        'world_profiles_sha256': _sha256(args.world_profiles.resolve()),
        'capture_script_sha256': _sha256(script),
        'capture_helper_sha256': _sha256(helper),
        'cameras': [
            {
                'camera_id': spec.camera_id, 'camera_model': spec.model,
                'image_topic': spec.image_topic, 'labels_topic': spec.labels_topic,
                'pose_xyz_rpy': list(spec.pose),
                'image_width': spec.image_width, 'image_height': spec.image_height,
            }
            for spec in specs
        ],
        'evaluation_only_inputs': (
            ['commanded_robot_pose', 'semantic_label_map']
            if bool(args.with_semantic) else ['commanded_robot_pose']
        ),
        'sensor_input': 'raw_rgb',
        'sensor_perturbation': {
            'type': (
                'independent_gaussian_rgb_read_noise_after_ros_transport'
                if float(args.rgb_noise_stddev_dn) > 0.0 else 'none'
            ),
            'mean_dn': 0.0,
            'stddev_dn': float(args.rgb_noise_stddev_dn),
            'base_seed': int(args.rgb_noise_seed),
            'clipped_to_uint8': True,
            'detector_input': True,
            'scope': (
                'controlled sensor-read-noise repeat experiment; does not model scene, '
                'illumination, pose, calibration, or renderer variation'
            ),
        },
        'sensor_output_to_be_computed': 'one_frozen_yolo_bounding_box',
        'limited_diagnostic_capture': bool(int(args.max_poses) > 0),
    }
    index_path = out / 'capture_index.csv'
    start_pose_id = 0
    rows_written = 0
    failures = 0
    existing_rows: list[dict[str, str]] = []
    if bool(args.resume):
        if not out.is_dir() or not manifest_path.is_file() or not index_path.is_file():
            raise RuntimeError('--resume requires an existing capture manifest and index')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'running':
            raise RuntimeError(f"resume requires status=running, got {manifest.get('status')!r}")
        if manifest.get('plan') != plan:
            raise RuntimeError('resume plan differs from the frozen capture plan')
        with index_path.open(newline='', encoding='utf-8') as handle:
            existing_rows = list(csv.DictReader(handle))
        counts = Counter(int(row['pose_id']) for row in existing_rows)
        if any(value != len(specs) for value in counts.values()):
            raise RuntimeError('resume index contains an incomplete five-camera pose batch')
        successful = {
            int(row['pose_id']) for row in existing_rows if row['capture_status'] == 'ok'
        }
        start_pose_id = max(successful, default=-1) + 1
        # A renderer crash can leave complete, all-failed retry batches at the tail. They were
        # never images, so retry those poses after restart while retaining earlier real failures.
        existing_rows = [row for row in existing_rows if int(row['pose_id']) < start_pose_id]
        with index_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(existing_rows)
        rows_written = len(existing_rows)
        failures = len({
            (row['pose_id'], row['repetition_id'])
            for row in existing_rows if row['capture_status'] == 'failed'
        })
        manifest['resume_count'] = int(manifest.get('resume_count', 0)) + 1
        manifest.setdefault('resume_events', []).append({
            'utc': datetime.now(timezone.utc).isoformat(),
            'start_pose_id': start_pose_id,
            'rows_retained': rows_written,
            'retained_failed_batches': failures,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    else:
        if out.exists():
            raise RuntimeError(f'Output already exists: {out}')
        for spec in specs:
            (out / spec.camera_id / 'images').mkdir(parents=True, exist_ok=False)
        manifest = expected_manifest
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    camera_ids = tuple(spec.camera_id for spec in specs)
    spec_by_id = {spec.camera_id: spec for spec in specs}
    camera_index_by_id = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    rclpy.init()
    node = FiveCameraCapture(
        world_name=str(profile['world_name']), cameras=specs,
        robot_z=float(args.robot_z), settle_s=float(args.settle_s),
        timeout_s=float(args.image_timeout_s), sync_slop_s=float(args.sync_slop_ms) / 1000.0,
        batch_sync_slop_s=float(args.batch_sync_slop_ms) / 1000.0,
        min_new_rgb=int(args.min_new_rgb_frames), min_new_labels=int(args.min_new_label_frames),
        buffer_size=int(args.buffer_size), with_semantic=bool(args.with_semantic),
    )
    try:
        node.wait_ready(camera_ids)
        image_path_by_hash: dict[tuple[str, str], Path] = {
            (row['camera_id'], row['image_sha1']): out / row['image']
            for row in existing_rows if row['capture_status'] == 'ok' and row['image_sha1']
        }
        with index_path.open('a' if bool(args.resume) else 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not bool(args.resume):
                writer.writeheader()
            for pose_id, record in enumerate(poses):
                if pose_id < start_pose_id:
                    continue
                x, y, yaw = float(record['x']), float(record['y']), float(record['yaw'])
                for repetition in range(max(int(args.repeats), 1)):
                    pairs = None
                    error = ''
                    for attempt in range(max(int(args.max_attempts), 1)):
                        try:
                            pairs = node.capture(x, y, yaw, camera_ids)
                            break
                        except Exception as exc:  # pragma: no cover - ROS transport path
                            error = f'attempt_{attempt}:{exc}'
                    if pairs is None:
                        failures += 1
                        for camera_id in camera_ids:
                            spec = spec_by_id[camera_id]
                            row = {field: '' for field in FIELDS}
                            row.update({
                                'pose_id': pose_id, 'position_id': int(record['position_id']),
                                'x_idx': int(record['x_idx']), 'y_idx': int(record['y_idx']),
                                'heading_id': int(record['yaw_idx']), 'repetition_id': repetition,
                                'source_batch_id': f'pose_{pose_id:06d}_r{repetition:02d}',
                                'dataset_split': record.get('dataset_split', ''),
                                'random_draw_index': record.get('random_draw_index', ''),
                                'camera_id': camera_id, 'camera_model': spec.model,
                                'capture_status': 'failed', 'capture_error': error,
                                'robot_x': x, 'robot_y': y, 'robot_yaw': yaw,
                            })
                            writer.writerow(row)
                            rows_written += 1
                        handle.flush()
                        continue
                    stamps = [pair.image_stamp_ns for pair in pairs.values()]
                    batch_span_s = (max(stamps) - min(stamps)) * 1e-9
                    for camera_id, pair in pairs.items():
                        spec = spec_by_id[camera_id]
                        detector_image = _add_rgb_read_noise(
                            pair.image,
                            stddev_dn=float(args.rgb_noise_stddev_dn),
                            seed=int(args.rgb_noise_seed),
                            pose_id=pose_id,
                            repetition_id=repetition,
                            camera_index=camera_index_by_id[camera_id],
                        )
                        image_sha1 = _sha1_array(detector_image)
                        image_path = image_path_by_hash.get((camera_id, image_sha1))
                        if image_path is None:
                            image_path = (
                                out / camera_id / 'images'
                                / f'pose_{pose_id:06d}_r{repetition:02d}.png'
                            )
                            if not cv2.imwrite(str(image_path), detector_image):
                                raise RuntimeError(f'Failed to write {image_path}')
                            image_path_by_hash[(camera_id, image_sha1)] = image_path
                        expected = _expected_geometry(spec, x, y, yaw)
                        row = {field: '' for field in FIELDS}
                        row.update({
                            'pose_id': pose_id, 'position_id': int(record['position_id']),
                            'x_idx': int(record['x_idx']), 'y_idx': int(record['y_idx']),
                            'heading_id': int(record['yaw_idx']), 'repetition_id': repetition,
                            'source_batch_id': f'pose_{pose_id:06d}_r{repetition:02d}',
                            'dataset_split': record.get('dataset_split', ''),
                            'random_draw_index': record.get('random_draw_index', ''),
                            'camera_id': camera_id, 'camera_model': spec.model,
                            'image': str(image_path.relative_to(out)),
                            'capture_status': 'ok', 'capture_error': '',
                            'robot_x': x, 'robot_y': y, 'robot_yaw': yaw,
                            'camera_range_m': math.hypot(x - spec.pose[0], y - spec.pose[1]),
                            'image_stamp_s': pair.image_stamp_ns * 1e-9,
                            'label_stamp_s': (
                                pair.label_stamp_ns * 1e-9
                                if pair.label_stamp_ns is not None else ''
                            ),
                            'stamp_delta_s': pair.stamp_delta_s,
                            'batch_image_span_s': batch_span_s,
                            'image_sha1': image_sha1,
                            **expected,
                        })
                        if pair.labels is not None:
                            row.update(_mask_geometry(pair.labels, int(args.robot_label)))
                        writer.writerow(row)
                        rows_written += 1
                    handle.flush()
                if (pose_id + 1) % 25 == 0:
                    print(
                        f'captured {pose_id + 1}/{len(poses)} poses; '
                        f'rows={rows_written} failed_batches={failures}',
                        flush=True,
                    )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    manifest.update({
        'status': 'complete' if failures == 0 else 'complete_with_failed_batches',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'rows_written': rows_written,
        'failed_batches': failures,
        'capture_index_sha256': _sha256(index_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ('status', 'rows_written', 'failed_batches')}, indent=2))
    return 0 if failures == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
