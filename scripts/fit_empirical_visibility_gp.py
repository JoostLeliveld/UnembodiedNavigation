#!/usr/bin/env python3
"""Capture empirical detection data over sampled spawn poses and fit a GP visibility field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import Float64MultiArray
import yaml

from experiments.core.world_profiles import load_profile
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from planning.core.visibility_raycast_25d import SimpleRBFGP, _clip_prob, _logit, _sigmoid
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world, signed_distance_to_union_xy


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


@dataclass
class SampleResult:
    sample_idx: int
    x: float
    y: float
    yaw: float
    repeat_idx: int
    detected_rate: float
    detected_label: int
    n_msgs: int
    n_detected: int
    mean_u: float
    mean_v: float
    mean_red_area_px: float
    mean_border_margin_px: float


class EmpiricalVisibilityCapture(Node):
    def __init__(self, world_name: str, entity_name: str = 'turtlebot3'):
        super().__init__('empirical_visibility_capture')
        self.world_name = str(world_name)
        self.entity_name = str(entity_name)
        self._odom_msg = None
        self._diag_buffer: List[Dict[str, float]] = []
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._diag_cb, 10)
        self.set_pose_client = self.create_client(SetEntityPose, f'/world/{self.world_name}/set_pose')

    def _odom_cb(self, msg: Odometry):
        self._odom_msg = msg

    def _diag_cb(self, msg: Float64MultiArray):
        self._diag_buffer.append(diagnostics_from_message(msg))

    def spin_for(self, duration_s: float):
        end = time.monotonic() + max(0.0, float(duration_s))
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_ready(self, timeout_s: float = 10.0):
        end = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < end:
            if self._odom_msg is not None and self.set_pose_client.wait_for_service(timeout_sec=0.1):
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError('Timed out waiting for /odom and /world/.../set_pose service')

    def set_pose(self, x: float, y: float, z: float, yaw: float, timeout_s: float = 3.0):
        req = SetEntityPose.Request()
        req.entity.name = self.entity_name
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        qx, qy, qz, qw = _yaw_to_quaternion(yaw)
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw
        future = self.set_pose_client.call_async(req)
        end = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if future.done():
                result = future.result()
                if result is None or not bool(result.success):
                    raise RuntimeError(f'SetEntityPose failed for ({x:.2f}, {y:.2f}, yaw={yaw:.2f})')
                return
        raise RuntimeError('Timed out waiting for SetEntityPose response')

    def wait_for_odom_pose(self, x: float, y: float, yaw: float, timeout_s: float = 2.0, pos_tol: float = 0.08, yaw_tol: float = 0.20):
        end = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            msg = self._odom_msg
            if msg is None:
                continue
            pos = msg.pose.pose.position
            q = msg.pose.pose.orientation
            est_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if math.hypot(pos.x - x, pos.y - y) <= pos_tol and abs(_wrap_angle(est_yaw - yaw)) <= yaw_tol:
                return True
        return False

    def clear_diag_buffer(self):
        self._diag_buffer.clear()

    def collect_diagnostics(self, timeout_s: float, min_msgs: int = 1) -> List[Dict[str, float]]:
        start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - start) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
            if len(self._diag_buffer) >= min_msgs:
                break
        if len(self._diag_buffer) < min_msgs:
            # give a little more time so very low-rate camera streams still contribute if available
            end = time.monotonic() + 0.15
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
        return list(self._diag_buffer)


def _default_arg_path(relative: str) -> str:
    return str((Path(__file__).resolve().parents[1] / relative).resolve())


def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan
    mask = np.isfinite(arr)
    if not np.any(mask):
        return math.nan
    return float(np.mean(arr[mask]))


def _load_world_profile(world_profiles_path: Path, world: str):
    profile, _intrinsics, world_path, camera_pose = load_profile(str(world_profiles_path), world)
    vis = dict(profile.get('visibility_defaults') or {})
    return profile, vis, Path(world_path), camera_pose


def _resolve_sampling_bounds(vis: Dict[str, object], sampling_region: str, xmin_override, xmax_override, ymin_override, ymax_override, wall_margin_m: float):
    xmin = float(vis.get('visibility_map_min_x', -6.0)) + wall_margin_m
    xmax = float(vis.get('visibility_map_max_x', 6.0)) - wall_margin_m
    ymin = float(vis.get('visibility_map_min_y', -6.0)) + wall_margin_m
    ymax = float(vis.get('visibility_map_max_y', 6.0)) - wall_margin_m

    region = str(sampling_region or 'full').strip().lower()
    if region == 'open_shelves_core':
        xmin, xmax = -3.35, -0.95
        ymin, ymax = -3.35, 3.15
    elif region not in ('', 'full', 'default'):
        raise ValueError(f'Unknown sampling region {sampling_region!r}')

    if xmin_override is not None:
        xmin = float(xmin_override)
    if xmax_override is not None:
        xmax = float(xmax_override)
    if ymin_override is not None:
        ymin = float(ymin_override)
    if ymax_override is not None:
        ymax = float(ymax_override)

    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f'Invalid sampling bounds: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}')
    return xmin, xmax, ymin, ymax


def _sample_positions(profile: Dict[str, object], vis: Dict[str, object], world_path: Path, sample_nx: int, sample_ny: int, wall_margin_m: float, clearance_m: float, sample_limit: int, shuffle: bool, seed: int, sampling_region: str = 'full', xmin_override=None, xmax_override=None, ymin_override=None, ymax_override=None):
    xmin, xmax, ymin, ymax = _resolve_sampling_bounds(
        vis,
        sampling_region,
        xmin_override,
        xmax_override,
        ymin_override,
        ymax_override,
        wall_margin_m,
    )
    xs = np.linspace(xmin, xmax, int(max(sample_nx, 2)))
    ys = np.linspace(ymin, ymax, int(max(sample_ny, 2)))
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    scene = parse_occlusion_scene_from_world(str(world_path), model_name='warehouse_rack_occluders')
    if scene.prisms:
        signed = signed_distance_to_union_xy(scene.prisms, pts)
        pts = pts[signed >= float(clearance_m)]

    rng = np.random.default_rng(int(seed))
    if shuffle and pts.shape[0] > 1:
        pts = pts[rng.permutation(pts.shape[0])]
    if int(sample_limit) > 0:
        pts = pts[: int(sample_limit)]
    return pts, (xmin, xmax, ymin, ymax)


def _aggregate_results(rows: List[SampleResult]):
    grouped: Dict[tuple[float, float, float], List[SampleResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.x, row.y, row.yaw)].append(row)

    agg_rows = []
    for (x, y, yaw), group in grouped.items():
        det_rates = [g.detected_rate for g in group]
        n_msgs = sum(g.n_msgs for g in group)
        n_detected = sum(g.n_detected for g in group)
        agg_rows.append({
            'x': x,
            'y': y,
            'yaw': yaw,
            'n_repeats': len(group),
            'detected_rate_mean': float(np.mean(det_rates)),
            'detected_rate_std': float(np.std(det_rates)),
            'detected_label': int(float(np.mean(det_rates)) >= 0.5),
            'n_msgs_total': int(n_msgs),
            'n_detected_total': int(n_detected),
            'mean_u': _safe_nanmean([g.mean_u for g in group]),
            'mean_v': _safe_nanmean([g.mean_v for g in group]),
            'mean_red_area_px': _safe_nanmean([g.mean_red_area_px for g in group]),
            'mean_border_margin_px': _safe_nanmean([g.mean_border_margin_px for g in group]),
        })
    agg_rows.sort(key=lambda r: (r['y'], r['x']))
    return agg_rows


def _fit_gp_from_aggregates(agg_rows, vis: Dict[str, object], camera_pose, gp_length_scale: float, gp_noise_var: float, beta: float, min_prob: float):
    xs = np.linspace(float(vis.get('visibility_map_min_x', -6.0)), float(vis.get('visibility_map_max_x', 6.0)), int(vis.get('visibility_map_nx', 160)))
    ys = np.linspace(float(vis.get('visibility_map_min_y', -6.0)), float(vis.get('visibility_map_max_y', 6.0)), int(vis.get('visibility_map_ny', 160)))

    X_train = np.asarray([[r['x'], r['y']] for r in agg_rows], dtype=float)
    p_train = np.asarray([r['detected_rate_mean'] for r in agg_rows], dtype=float)
    p_train = np.clip(p_train, min_prob, 1.0 - min_prob)

    gp = SimpleRBFGP(
        length_scale=float(gp_length_scale),
        signal_var=1.0,
        noise_var=float(max(gp_noise_var, 1e-8)),
    ).fit(X_train, _logit(p_train))

    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    mu_f, sigma_f = gp.predict_mean_std(XY)
    p_mean = _sigmoid(mu_f)
    p_cons = _sigmoid(mu_f - float(beta) * sigma_f)

    return {
        'xs': xs,
        'ys': ys,
        'X_train': X_train,
        'p_train': p_train,
        'P_mean_map': _clip_prob(p_mean.reshape(Yg.shape), min_prob).astype(float),
        'P_conservative_map': _clip_prob(p_cons.reshape(Yg.shape), min_prob).astype(float),
        'P_map': _clip_prob(p_cons.reshape(Yg.shape), min_prob).astype(float),
        'camera_pos': np.asarray(camera_pose[:3], dtype=float),
    }


def _write_raw_csv(path: Path, rows: List[SampleResult]):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_idx', 'x', 'y', 'yaw', 'repeat_idx', 'detected_rate', 'detected_label',
            'n_msgs', 'n_detected', 'mean_u', 'mean_v', 'mean_red_area_px', 'mean_border_margin_px',
        ])
        for row in rows:
            writer.writerow([
                row.sample_idx, row.x, row.y, row.yaw, row.repeat_idx, row.detected_rate,
                row.detected_label, row.n_msgs, row.n_detected, row.mean_u, row.mean_v,
                row.mean_red_area_px, row.mean_border_margin_px,
            ])


def _write_agg_csv(path: Path, agg_rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows else [
            'x', 'y', 'yaw', 'n_repeats', 'detected_rate_mean', 'detected_rate_std',
            'detected_label', 'n_msgs_total', 'n_detected_total', 'mean_u', 'mean_v',
            'mean_red_area_px', 'mean_border_margin_px',
        ])
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)


def _plot_fit(path: Path, fit, agg_rows, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    extent = [fit['xs'][0], fit['xs'][-1], fit['ys'][0], fit['ys'][-1]]
    im0 = axes[0].imshow(fit['P_mean_map'], origin='lower', extent=extent, aspect='equal', vmin=0.0, vmax=1.0, cmap='viridis')
    axes[0].set_title('Empirical GP mean')
    im1 = axes[1].imshow(fit['P_conservative_map'], origin='lower', extent=extent, aspect='equal', vmin=0.0, vmax=1.0, cmap='viridis')
    axes[1].set_title('Empirical GP conservative')
    pts = np.asarray([[r['x'], r['y']] for r in agg_rows], dtype=float) if agg_rows else np.zeros((0, 2), dtype=float)
    vals = np.asarray([r['detected_rate_mean'] for r in agg_rows], dtype=float) if agg_rows else np.zeros((0,), dtype=float)
    if pts.size:
        for ax in axes:
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap='coolwarm', vmin=0.0, vmax=1.0, s=28, edgecolors='k', linewidths=0.3)
            ax.scatter([fit['camera_pos'][0]], [fit['camera_pos'][1]], c='white', s=50, marker='x')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='Capture empirical detection samples and fit a GP visibility field.')
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=_default_arg_path('src/experiments/config/world_profiles.yaml'))
    parser.add_argument('--output-root', default=_default_arg_path('logs/visibility_capture'))
    parser.add_argument('--publish-artifact', default='', help='Optional stable output path for the fitted GP artifact')
    parser.add_argument('--entity-name', default='turtlebot3')
    parser.add_argument('--sample-nx', type=int, default=15)
    parser.add_argument('--sample-ny', type=int, default=15)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--sampling-region', default='full', choices=['full', 'open_shelves_core'])
    parser.add_argument('--xmin', type=float, default=None)
    parser.add_argument('--xmax', type=float, default=None)
    parser.add_argument('--ymin', type=float, default=None)
    parser.add_argument('--ymax', type=float, default=None)
    parser.add_argument('--yaw-rad', type=float, default=0.0)
    parser.add_argument('--robot-z', type=float, default=0.05)
    parser.add_argument('--wall-margin-m', type=float, default=0.45)
    parser.add_argument('--clearance-m', type=float, default=0.28)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--settle-s', type=float, default=0.25)
    parser.add_argument('--diag-timeout-s', type=float, default=1.6)
    parser.add_argument('--min-diag-messages', type=int, default=1)
    parser.add_argument('--odom-timeout-s', type=float, default=2.0)
    parser.add_argument('--label-threshold', type=float, default=0.5)
    parser.add_argument('--gp-length-scale', type=float, default=-1.0)
    parser.add_argument('--gp-noise-var', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=-1.0)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    world_profiles_path = Path(args.world_profiles).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    profile, vis, world_path, camera_pose = _load_world_profile(world_profiles_path, args.world)
    points, sample_bounds = _sample_positions(
        profile,
        vis,
        world_path,
        args.sample_nx,
        args.sample_ny,
        args.wall_margin_m,
        args.clearance_m,
        args.sample_limit,
        args.shuffle,
        args.seed,
        sampling_region=args.sampling_region,
        xmin_override=args.xmin,
        xmax_override=args.xmax,
        ymin_override=args.ymin,
        ymax_override=args.ymax,
    )
    if points.shape[0] == 0:
        raise RuntimeError('No valid sample points remained after filtering')

    gp_length_scale = float(args.gp_length_scale if args.gp_length_scale > 0.0 else vis.get('visibility_gp_length_scale', 1.0))
    gp_noise_var = float(args.gp_noise_var if args.gp_noise_var > 0.0 else vis.get('visibility_gp_noise_var', 0.1))
    beta = float(args.beta if args.beta > 0.0 else vis.get('visibility_beta', 1.0))

    rclpy.init()
    node = EmpiricalVisibilityCapture(world_name=profile['world_name'], entity_name=args.entity_name)
    try:
        node.get_logger().info(f'Waiting for capture interfaces in world {profile["world_name"]!r}')
        node.wait_for_ready(timeout_s=12.0)
        node.spin_for(0.5)
        xmin, xmax, ymin, ymax = sample_bounds
        node.get_logger().info(
            'Sampling %d poses in region=%r with bounds x=[%.2f, %.2f], y=[%.2f, %.2f], repeats=%d' % (
                int(points.shape[0]),
                str(args.sampling_region),
                float(xmin),
                float(xmax),
                float(ymin),
                float(ymax),
                int(max(args.repeats, 1)),
            )
        )

        raw_rows: List[SampleResult] = []
        for sample_idx, (x, y) in enumerate(points):
            for repeat_idx in range(int(max(args.repeats, 1))):
                node.set_pose(float(x), float(y), float(args.robot_z), float(args.yaw_rad), timeout_s=3.0)
                node.wait_for_odom_pose(float(x), float(y), float(args.yaw_rad), timeout_s=args.odom_timeout_s)
                node.spin_for(args.settle_s)
                node.clear_diag_buffer()
                diags = node.collect_diagnostics(timeout_s=args.diag_timeout_s, min_msgs=max(args.min_diag_messages, 1))
                detected_flags = [1.0 if bool(d['detected']) else 0.0 for d in diags]
                detect_rate = float(np.mean(detected_flags)) if detected_flags else 0.0
                n_detected = int(sum(detected_flags))
                raw_rows.append(SampleResult(
                    sample_idx=sample_idx,
                    x=float(x),
                    y=float(y),
                    yaw=float(args.yaw_rad),
                    repeat_idx=repeat_idx,
                    detected_rate=detect_rate,
                    detected_label=int(detect_rate >= float(args.label_threshold)),
                    n_msgs=len(diags),
                    n_detected=n_detected,
                    mean_u=_safe_nanmean([d['u_mid'] for d in diags]),
                    mean_v=_safe_nanmean([d['v_mid'] for d in diags]),
                    mean_red_area_px=_safe_nanmean([d['red_area_px'] for d in diags]),
                    mean_border_margin_px=_safe_nanmean([d['border_margin_px'] for d in diags]),
                ))
                if ((sample_idx + 1) % 10 == 0 and repeat_idx == int(max(args.repeats, 1)) - 1) or sample_idx == 0:
                    node.get_logger().info(
                        f'Captured {sample_idx + 1}/{points.shape[0]} poses; latest rate={detect_rate:.2f} at ({x:.2f}, {y:.2f})'
                    )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    agg_rows = _aggregate_results(raw_rows)
    fit = _fit_gp_from_aggregates(
        agg_rows,
        vis,
        camera_pose,
        gp_length_scale=gp_length_scale,
        gp_noise_var=gp_noise_var,
        beta=beta,
        min_prob=float(max(args.min_prob, 1e-6)),
    )

    raw_csv = run_dir / 'raw_detection_samples.csv'
    agg_csv = run_dir / 'aggregated_detection_samples.csv'
    npz_path = run_dir / 'empirical_visibility_gp.npz'
    plot_path = run_dir / 'empirical_visibility_gp.png'
    meta_path = run_dir / 'capture_config.json'

    _write_raw_csv(raw_csv, raw_rows)
    _write_agg_csv(agg_csv, agg_rows)
    np.savez_compressed(
        npz_path,
        xs=fit['xs'],
        ys=fit['ys'],
        X_train=fit['X_train'],
        p_train=fit['p_train'],
        P_mean_map=fit['P_mean_map'],
        P_conservative_map=fit['P_conservative_map'],
        P_map=fit['P_map'],
        camera_pos=fit['camera_pos'],
    )
    _plot_fit(plot_path, fit, agg_rows, title=f'Empirical visibility GP: {args.world}')

    meta = {
        'world': args.world,
        'world_name': profile['world_name'],
        'world_path': str(world_path),
        'world_profiles_path': str(world_profiles_path),
        'output_dir': str(run_dir),
        'n_raw_samples': len(raw_rows),
        'n_unique_poses': len(agg_rows),
        'sample_nx': int(args.sample_nx),
        'sample_ny': int(args.sample_ny),
        'sample_limit': int(args.sample_limit),
        'sampling_region': str(args.sampling_region),
        'sample_bounds': {
            'xmin': float(sample_bounds[0]),
            'xmax': float(sample_bounds[1]),
            'ymin': float(sample_bounds[2]),
            'ymax': float(sample_bounds[3]),
        },
        'yaw_rad': float(args.yaw_rad),
        'robot_z': float(args.robot_z),
        'wall_margin_m': float(args.wall_margin_m),
        'clearance_m': float(args.clearance_m),
        'repeats': int(args.repeats),
        'settle_s': float(args.settle_s),
        'diag_timeout_s': float(args.diag_timeout_s),
        'min_diag_messages': int(args.min_diag_messages),
        'gp_length_scale': gp_length_scale,
        'gp_noise_var': gp_noise_var,
        'beta': beta,
        'mean_detected_rate': float(np.mean([r['detected_rate_mean'] for r in agg_rows])) if agg_rows else math.nan,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    publish_artifact = Path(str(args.publish_artifact or '').strip()).expanduser()
    if str(publish_artifact):
        publish_artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz_path, publish_artifact)
        print(f'Published empirical GP artifact to {publish_artifact.resolve()}')

    print(f'Wrote raw samples to {raw_csv}')
    print(f'Wrote aggregated samples to {agg_csv}')
    print(f'Wrote empirical GP artifact to {npz_path}')
    print(f'Wrote preview plot to {plot_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
