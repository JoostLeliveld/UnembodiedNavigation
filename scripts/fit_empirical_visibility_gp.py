#!/usr/bin/env python3
"""Collect driving-based visibility data and fit a scalar GP visibility field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float64MultiArray, String

REPO_ROOT = Path(__file__).resolve().parents[1]
for rel in ('src/experiments', 'src/perception', 'src/planning', 'src/unav_common'):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.core.visibility_capture import (
    VISIBILITY_CAPTURE_CONFIG_TOPIC,
    VISIBILITY_CAPTURE_DONE_TOPIC,
)
from experiments.core.world_profiles import load_profile
from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from planning.core.gp_visibility_helpers import (
    SimpleRBFGP,
    clip_prob as _clip_prob,
    logit as _logit,
    sigmoid as _sigmoid,
)


@dataclass
class CaptureSample:
    sample_idx: int
    stamp: float
    state_x: float
    state_y: float
    cov_x: float
    cov_y: float
    state_age_s: float
    state_is_fresh: int
    detected: int
    usable_label: int
    u_mid: float
    v_mid: float
    yaw_est: float
    red_area_px: float
    border_margin_px: float
    normalized_blob_area: float = math.nan


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


def _safe_nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan
    mask = np.isfinite(arr)
    if not np.any(mask):
        return math.nan
    return float(np.std(arr[mask]))


def _load_world_profile(world_profiles_path: Path, world: str):
    profile, _intrinsics, world_path, camera_pose = load_profile(str(world_profiles_path), world)
    vis = dict(profile.get('visibility_defaults') or {})
    return profile, vis, Path(world_path), camera_pose


class DrivingVisibilityCapture(Node):
    def __init__(self, *, usable_area_px_min: float, max_state_age_s: float):
        super().__init__('driving_visibility_capture')
        self.usable_area_px_min = float(usable_area_px_min)
        self.max_state_age_s = float(max_state_age_s)

        self.samples: List[CaptureSample] = []
        self._latest_state = None
        self._done = False
        self._sweep_config = None
        self._skipped_no_state = 0
        self._stale_state_rows = 0

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, 10)
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._diag_cb, 10)
        self.create_subscription(Bool, VISIBILITY_CAPTURE_DONE_TOPIC, self._done_cb, latched_qos)
        self.create_subscription(String, VISIBILITY_CAPTURE_CONFIG_TOPIC, self._config_cb, latched_qos)

    @property
    def skipped_no_state(self) -> int:
        return int(self._skipped_no_state)

    @property
    def stale_state_rows(self) -> int:
        return int(self._stale_state_rows)

    @property
    def sweep_config(self) -> Dict[str, object] | None:
        return dict(self._sweep_config) if self._sweep_config is not None else None

    def _state_cb(self, msg: PoseWithCovarianceStamped) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        cov = list(msg.pose.covariance)
        self._latest_state = {
            'stamp': stamp,
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'cov_x': float(cov[0]) if len(cov) > 0 else math.nan,
            'cov_y': float(cov[7]) if len(cov) > 7 else math.nan,
        }

    def _done_cb(self, msg: Bool) -> None:
        self._done = bool(msg.data)

    def _config_cb(self, msg: String) -> None:
        try:
            self._sweep_config = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Ignoring malformed sweep config message: {exc}')

    def _diag_cb(self, msg: Float64MultiArray) -> None:
        diag = diagnostics_from_message(msg)
        if self._latest_state is None:
            self._skipped_no_state += 1
            return

        stamp = float(diag.get('stamp', math.nan))
        if not math.isfinite(stamp):
            self._skipped_no_state += 1
            return

        state_age_s = abs(stamp - float(self._latest_state['stamp']))
        state_is_fresh = int(state_age_s <= self.max_state_age_s)
        if not state_is_fresh:
            self._stale_state_rows += 1

        detected = bool(diag.get('detected', False))
        red_area_px = float(diag.get('red_area_px', math.nan))
        usable_label = int(detected and math.isfinite(red_area_px) and red_area_px >= self.usable_area_px_min)

        self.samples.append(CaptureSample(
            sample_idx=len(self.samples),
            stamp=stamp,
            state_x=float(self._latest_state['x']),
            state_y=float(self._latest_state['y']),
            cov_x=float(self._latest_state['cov_x']),
            cov_y=float(self._latest_state['cov_y']),
            state_age_s=float(state_age_s),
            state_is_fresh=state_is_fresh,
            detected=int(detected),
            usable_label=usable_label,
            u_mid=float(diag.get('u_mid', math.nan)),
            v_mid=float(diag.get('v_mid', math.nan)),
            yaw_est=float(diag.get('yaw_est', math.nan)),
            red_area_px=red_area_px,
            border_margin_px=float(diag.get('border_margin_px', math.nan)),
        ))

        if len(self.samples) % 100 == 0:
            usable_rate = float(np.mean([s.usable_label for s in self.samples]))
            self.get_logger().info(
                f'Captured {len(self.samples)} diagnostic samples '
                f'(usable_rate={usable_rate:.3f}, stale_rows={self._stale_state_rows})'
            )

    def wait_for_ready(self, timeout_s: float) -> None:
        end = time.monotonic() + max(timeout_s, 0.0)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_state is not None and self._sweep_config is not None:
                return
        raise RuntimeError('Timed out waiting for /state/bev and /visibility_capture/config_json')

    def run_until_complete(self, max_capture_s: float, post_done_grace_s: float) -> None:
        start = time.monotonic()
        done_seen_at = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            if max_capture_s > 0.0 and (now - start) > max_capture_s:
                self.get_logger().warn('Capture reached max_capture_s before completion; stopping collection')
                return
            if self._done:
                if done_seen_at is None:
                    done_seen_at = now
                elif (now - done_seen_at) >= max(post_done_grace_s, 0.0):
                    return


def _compute_blob_area_ref(samples: List[CaptureSample], percentile: float, usable_area_px_min: float) -> float:
    areas = np.asarray(
        [s.red_area_px for s in samples if math.isfinite(s.red_area_px) and s.red_area_px > 0.0],
        dtype=float,
    )
    if areas.size == 0:
        return float(max(usable_area_px_min, 1.0))
    ref = float(np.percentile(areas, np.clip(percentile, 1.0, 100.0)))
    return float(max(ref, usable_area_px_min, 1.0))


def _inject_normalized_blob_area(samples: List[CaptureSample], ref_area_px: float) -> None:
    denom = float(max(ref_area_px, 1.0))
    for sample in samples:
        if math.isfinite(sample.red_area_px) and sample.red_area_px > 0.0:
            sample.normalized_blob_area = float(np.clip(sample.red_area_px / denom, 0.0, 1.0))
        else:
            sample.normalized_blob_area = 0.0


def _grid_vectors(vis: Dict[str, object]):
    xmin = float(vis.get('visibility_map_min_x', -6.0))
    xmax = float(vis.get('visibility_map_max_x', 6.0))
    ymin = float(vis.get('visibility_map_min_y', -6.0))
    ymax = float(vis.get('visibility_map_max_y', 6.0))
    nx = int(vis.get('visibility_map_nx', 160))
    ny = int(vis.get('visibility_map_ny', 160))
    xs = np.linspace(xmin, xmax, max(nx, 2))
    ys = np.linspace(ymin, ymax, max(ny, 2))
    return xs, ys


def _nearest_index(value: float, axis: np.ndarray) -> int:
    if axis.size <= 1:
        return 0
    step = float(axis[1] - axis[0])
    idx = int(round((value - float(axis[0])) / step))
    return int(np.clip(idx, 0, axis.size - 1))


def _aggregate_samples(samples: List[CaptureSample], vis: Dict[str, object]):
    xs, ys = _grid_vectors(vis)
    grouped: Dict[tuple[int, int], List[CaptureSample]] = defaultdict(list)
    for sample in samples:
        if not sample.state_is_fresh:
            continue
        if not (math.isfinite(sample.state_x) and math.isfinite(sample.state_y)):
            continue
        ix = _nearest_index(sample.state_x, xs)
        iy = _nearest_index(sample.state_y, ys)
        grouped[(ix, iy)].append(sample)

    agg_rows = []
    for (ix, iy), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        detected_flags = [float(s.detected) for s in group]
        usable_flags = [float(s.usable_label) for s in group]
        agg_rows.append({
            'ix': int(ix),
            'iy': int(iy),
            'x_center': float(xs[ix]),
            'y_center': float(ys[iy]),
            'n_samples': int(len(group)),
            'n_detected': int(sum(detected_flags)),
            'n_usable': int(sum(usable_flags)),
            'detected_rate_mean': float(np.mean(detected_flags)),
            'detected_rate_std': float(np.std(detected_flags)),
            'usable_rate_mean': float(np.mean(usable_flags)),
            'usable_rate_std': float(np.std(usable_flags)),
            'mean_red_area_px': _safe_nanmean([s.red_area_px for s in group]),
            'mean_border_margin_px': _safe_nanmean([s.border_margin_px for s in group]),
            'mean_blob_area_norm': _safe_nanmean([s.normalized_blob_area for s in group]),
            'mean_state_age_s': _safe_nanmean([s.state_age_s for s in group]),
            'mean_cov_x': _safe_nanmean([s.cov_x for s in group]),
            'mean_cov_y': _safe_nanmean([s.cov_y for s in group]),
        })
    return agg_rows


def _fit_gp_from_aggregates(agg_rows, vis: Dict[str, object], camera_pose, gp_length_scale: float, gp_noise_var: float, beta: float, min_prob: float):
    xs, ys = _grid_vectors(vis)
    if not agg_rows:
        raise RuntimeError('No fresh state-aligned capture samples remained for GP fitting')

    X_train = np.asarray([[r['x_center'], r['y_center']] for r in agg_rows], dtype=float)
    p_train = np.asarray([r['usable_rate_mean'] for r in agg_rows], dtype=float)
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


def _write_raw_csv(path: Path, rows: List[CaptureSample]) -> None:
    fieldnames = [
        'sample_idx',
        'stamp',
        'state_x',
        'state_y',
        'cov_x',
        'cov_y',
        'state_age_s',
        'state_is_fresh',
        'detected',
        'usable_label',
        'u_mid',
        'v_mid',
        'yaw_est',
        'red_area_px',
        'border_margin_px',
        'normalized_blob_area',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: getattr(row, key) for key in fieldnames
            })


def _write_agg_csv(path: Path, agg_rows) -> None:
    default_fields = [
        'ix', 'iy', 'x_center', 'y_center', 'n_samples', 'n_detected', 'n_usable',
        'detected_rate_mean', 'detected_rate_std', 'usable_rate_mean', 'usable_rate_std',
        'mean_red_area_px', 'mean_border_margin_px', 'mean_blob_area_norm',
        'mean_state_age_s', 'mean_cov_x', 'mean_cov_y',
    ]
    fieldnames = list(agg_rows[0].keys()) if agg_rows else default_fields
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)


def _plot_fit(path: Path, fit, agg_rows, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    extent = [fit['xs'][0], fit['xs'][-1], fit['ys'][0], fit['ys'][-1]]
    im0 = axes[0].imshow(
        fit['P_mean_map'],
        origin='lower',
        extent=extent,
        aspect='equal',
        vmin=0.0,
        vmax=1.0,
        cmap='viridis',
    )
    axes[0].set_title('Driving-capture GP mean')
    im1 = axes[1].imshow(
        fit['P_conservative_map'],
        origin='lower',
        extent=extent,
        aspect='equal',
        vmin=0.0,
        vmax=1.0,
        cmap='viridis',
    )
    axes[1].set_title('Driving-capture GP conservative')

    pts = np.asarray([[r['x_center'], r['y_center']] for r in agg_rows], dtype=float) if agg_rows else np.zeros((0, 2))
    vals = np.asarray([r['usable_rate_mean'] for r in agg_rows], dtype=float) if agg_rows else np.zeros((0,))
    if pts.size:
        for ax in axes:
            ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap='coolwarm', vmin=0.0, vmax=1.0, s=24, edgecolors='k', linewidths=0.25)
            ax.scatter([fit['camera_pos'][0]], [fit['camera_pos'][1]], c='white', s=50, marker='x')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Collect driving-based visibility samples from /state/bev and fit a GP visibility field.'
    )
    parser.add_argument('--world', default='warehouse_occ_light.world.sdf')
    parser.add_argument('--world-profiles', default=_default_arg_path('src/experiments/config/world_profiles.yaml'))
    parser.add_argument('--output-root', default=_default_arg_path('logs/visibility_capture'))
    parser.add_argument('--publish-artifact', default='', help='Optional stable output path for the fitted GP artifact')
    parser.add_argument('--usable-area-px-min', type=float, default=20.0)
    parser.add_argument('--max-state-age-s', type=float, default=0.50)
    parser.add_argument('--blob-area-ref-percentile', type=float, default=95.0)
    parser.add_argument('--ready-timeout-s', type=float, default=20.0)
    parser.add_argument('--max-capture-s', type=float, default=600.0)
    parser.add_argument('--post-done-grace-s', type=float, default=0.75)
    parser.add_argument('--gp-length-scale', type=float, default=-1.0)
    parser.add_argument('--gp-noise-var', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=-1.0)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    args = parser.parse_args()

    world_profiles_path = Path(args.world_profiles).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    profile, vis, world_path, camera_pose = _load_world_profile(world_profiles_path, args.world)
    gp_length_scale = float(args.gp_length_scale if args.gp_length_scale > 0.0 else vis.get('visibility_gp_length_scale', 1.0))
    gp_noise_var = float(args.gp_noise_var if args.gp_noise_var > 0.0 else vis.get('visibility_gp_noise_var', 0.1))
    beta = float(args.beta if args.beta > 0.0 else vis.get('visibility_beta', 1.0))
    min_prob = float(max(args.min_prob, 1e-6))

    rclpy.init()
    node = DrivingVisibilityCapture(
        usable_area_px_min=float(args.usable_area_px_min),
        max_state_age_s=float(args.max_state_age_s),
    )
    try:
        node.get_logger().info(
            f'Waiting for driving capture interfaces for world {profile["world_name"]!r}'
        )
        node.wait_for_ready(timeout_s=float(args.ready_timeout_s))
        node.get_logger().info('Starting driving-based visibility capture')
        node.run_until_complete(
            max_capture_s=float(args.max_capture_s),
            post_done_grace_s=float(args.post_done_grace_s),
        )
    finally:
        samples = list(node.samples)
        skipped_no_state = node.skipped_no_state
        stale_state_rows = node.stale_state_rows
        sweep_config = node.sweep_config or {}
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    blob_area_ref_px = _compute_blob_area_ref(samples, float(args.blob_area_ref_percentile), float(args.usable_area_px_min))
    _inject_normalized_blob_area(samples, blob_area_ref_px)
    agg_rows = _aggregate_samples(samples, vis)
    fit = _fit_gp_from_aggregates(
        agg_rows,
        vis,
        camera_pose,
        gp_length_scale=gp_length_scale,
        gp_noise_var=gp_noise_var,
        beta=beta,
        min_prob=min_prob,
    )

    raw_csv = run_dir / 'raw_detection_samples.csv'
    agg_csv = run_dir / 'aggregated_detection_samples.csv'
    npz_path = run_dir / 'empirical_visibility_gp.npz'
    plot_path = run_dir / 'empirical_visibility_gp.png'
    meta_path = run_dir / 'capture_config.json'

    _write_raw_csv(raw_csv, samples)
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
    _plot_fit(plot_path, fit, agg_rows, title=f'Driving-based empirical visibility GP: {args.world}')

    fresh_samples = [s for s in samples if s.state_is_fresh]
    meta = {
        'world': args.world,
        'world_name': profile['world_name'],
        'world_path': str(world_path),
        'world_profiles_path': str(world_profiles_path),
        'output_dir': str(run_dir),
        'state_input_topic': '/state/bev',
        'state_dimensions': ['x', 'y'],
        'label_mode': 'binary_usable_detection',
        'label_rule': 'detected && red_area_px >= usable_area_px_min',
        'usable_area_px_min': float(args.usable_area_px_min),
        'max_state_age_s': float(args.max_state_age_s),
        'blob_area_ref_percentile': float(args.blob_area_ref_percentile),
        'blob_area_ref_px': float(blob_area_ref_px),
        'n_raw_samples': int(len(samples)),
        'n_fresh_samples': int(len(fresh_samples)),
        'n_occupied_cells': int(len(agg_rows)),
        'skipped_no_state_samples': int(skipped_no_state),
        'stale_state_rows': int(stale_state_rows),
        'gp_length_scale': gp_length_scale,
        'gp_noise_var': gp_noise_var,
        'beta': beta,
        'mean_detected_rate_raw': float(np.mean([s.detected for s in samples])) if samples else math.nan,
        'mean_usable_rate_raw': float(np.mean([s.usable_label for s in samples])) if samples else math.nan,
        'mean_detected_rate_cells': float(np.mean([r['detected_rate_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'mean_usable_rate_cells': float(np.mean([r['usable_rate_mean'] for r in agg_rows])) if agg_rows else math.nan,
        'sweep_config': sweep_config,
        'published_artifact_path': str(Path(args.publish_artifact).expanduser().resolve()) if str(args.publish_artifact).strip() else '',
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
