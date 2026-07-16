#!/usr/bin/env python3
"""Build a planner-compatible initial GP from one fixed-camera depth frame."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from common import ARTIFACT_SCHEMA_VERSION, LOGS_ROOT, REPO_ROOT, sha256_file, write_csv, write_manifest


os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


GEOMETRY_DIR = REPO_ROOT / 'scripts' / 'geometry_visibility'
UNAV_COMMON_DIR = REPO_ROOT / 'src' / 'unav_common'
for path in (UNAV_COMMON_DIR, GEOMETRY_DIR):
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import geometry_visibility as gv  # noqa: E402


DEFAULT_REFERENCE_GP = REPO_ROOT / 'paper_artifacts' / 'gp' / 'warehouse_visibility_gp_v1' / 'yolo_score_raw_gp.npz'
DEFAULT_CAMPAIGN_CONFIG = REPO_ROOT / 'scripts' / 'visibility_comparison' / 'warehouse_visibility_campaign.yaml'
DEFAULT_DEPTH_NPY = Path(
    '/tmp/claude-1000/-home-joostleliveld-Thesis/2e584162-8150-4091-9318-d7f33ec3aeeb/scratchpad/depth.npy'
)
DEFAULT_OUT = LOGS_ROOT / 'depth_sensed_initial_gp_v1'


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -60.0, 60.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _load_campaign_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    driveable_json = json.loads(cfg['driveable_geometry_json'])
    return {
        'driveable_prisms': gv.prisms_from_json(driveable_json),
        'r_visible_uv': float(cfg.get('r_visible_uv', 2.5)),
        'r_miss_uv': float(cfg.get('r_miss_uv', 40.0)),
        'world': str(cfg.get('world', '')),
    }


def _reference_payload(path: Path) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in (
            'camera_pos',
            'camera_pose',
            'look_at',
            'img_width',
            'img_height',
            'fov_h_rad',
            'target_height',
            'geometry_json',
            'geometry_sha256',
            'world',
            'world_name',
            'world_path',
        ):
            if key in data.files:
                payload[key] = np.asarray(data[key])
    return payload


def _backproject_depth(cam, depth: np.ndarray, *, step: int, max_depth_m: float) -> np.ndarray:
    height, width = depth.shape
    vv, uu = np.mgrid[0:height:step, 0:width:step]
    d = depth[vv, uu].reshape(-1)
    uu = uu.reshape(-1).astype(float)
    vv = vv.reshape(-1).astype(float)
    finite = np.isfinite(d) & (d > 0.2) & (d < float(max_depth_m))
    uu = uu[finite]
    vv = vv[finite]
    d = d[finite].astype(float)
    if d.size == 0:
        raise RuntimeError('Depth frame contains no finite positive samples after filtering')

    rays = np.linalg.inv(cam.K) @ np.vstack([uu, vv, np.ones_like(uu)])
    camera_points = rays * d
    world_points = (cam.R.T @ camera_points) + np.asarray(cam.cam_pos, dtype=float)[:, None]
    return world_points.T


def _raycast_height_map(cam, xs: np.ndarray, ys: np.ndarray, height_map: np.ndarray, *, z_marker: float, n_samples: int) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    targets = np.stack([gx, gy, np.full_like(gx, float(z_marker))], axis=-1).reshape(-1, 3)
    cam_pos = np.asarray(cam.cam_pos, dtype=float)
    t = np.linspace(0.02, 0.98, int(n_samples)).reshape(1, -1, 1)
    samples = cam_pos.reshape(1, 1, 3) + t * (targets[:, None, :] - cam_pos.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, samples[..., 0]), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, samples[..., 1]), 0, len(ys) - 1)
    clearance = samples[..., 2] - height_map[iy, ix]
    return clearance.min(axis=1).reshape(gx.shape)


def _overlay_prisms(ax, prisms, *, color: str, linewidth: float = 0.8) -> None:
    for prism in prisms:
        ax.add_patch(
            Rectangle(
                (prism.xmin, prism.ymin),
                prism.xmax - prism.xmin,
                prism.ymax - prism.ymin,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def _plot_showcase(
    path: Path,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    height_map: np.ndarray,
    observed_mask: np.ndarray,
    unknown_mask: np.ndarray,
    p_mean: np.ndarray,
    p_plan: np.ndarray,
    driveable_mask: np.ndarray,
    meta: dict,
    driveable_prisms,
) -> None:
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    fig.suptitle('Depth-sensed initial GP from fixed-camera RGB-D', fontsize=14, fontweight='bold')

    fields = (
        (np.where(observed_mask, height_map, np.nan), 'Sensed height map', 'height [m]', 'cividis', 0.0, 2.8),
        (np.where(driveable_mask, observed_mask.astype(float), np.nan), 'Observed driveable cells', 'observed', 'Blues', 0.0, 1.0),
        (np.where(driveable_mask, p_mean, np.nan), 'P_mean from depth visibility', 'P_mean', 'viridis', 0.0, 1.0),
        (np.where(driveable_mask, p_plan, np.nan), 'Conservative planning map', 'P_plan', 'viridis', 0.0, 1.0),
    )
    for ax, (field, title, label, cmap, vmin, vmax) in zip(axes, fields):
        im = ax.imshow(field, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        _overlay_prisms(ax, driveable_prisms, color='#32a852', linewidth=0.7)
        cam = np.asarray(meta['camera_pos'], dtype=float)
        ax.plot([cam[0]], [cam[1]], '*', color='#e63b2e', markersize=13, markeredgecolor='black')
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(im, ax=ax, shrink=0.78, label=label)
    axes[1].contour(
        xs,
        ys,
        np.where(driveable_mask, unknown_mask.astype(float), 0.0),
        levels=[0.5],
        colors=['#f28e2b'],
        linewidths=0.9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build(args: argparse.Namespace) -> tuple[Path, dict[str, str]]:
    reference_gp = Path(args.reference_gp).expanduser().resolve()
    campaign_config = Path(args.campaign_config).expanduser().resolve()
    depth_path = Path(args.depth_npy).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    logs_root = LOGS_ROOT.resolve()
    if logs_root not in output_dir.parents and output_dir != logs_root:
        raise RuntimeError(f'Output directory must stay under {logs_root}: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = gv.load_gp_artifact_geometry(reference_gp)
    cfg = _load_campaign_config(campaign_config)
    xs = np.asarray(meta['xs'], dtype=float)
    ys = np.asarray(meta['ys'], dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or ys.size < 2:
        raise RuntimeError(f'Reference GP has an invalid grid: {reference_gp}')
    cam = gv.make_camera(meta)

    depth = np.asarray(np.load(depth_path), dtype=float)
    if depth.ndim != 2:
        raise RuntimeError(f'Depth frame must be HxW, got shape {depth.shape}: {depth_path}')
    points = _backproject_depth(cam, depth, step=int(args.depth_step), max_depth_m=float(args.max_depth_m))
    hm = gv.height_map_from_points(points, xs, ys)
    height_map = np.asarray(hm['h_max'], dtype=float)
    observed_mask = np.asarray(hm['observed'], dtype=bool)

    fov = gv.fov_projection_grid(cam, xs, ys, float(args.z_marker))
    jac = gv.projection_jacobian_scale(cam, xs, ys, float(args.z_marker))
    clearance = _raycast_height_map(
        cam,
        xs,
        ys,
        height_map,
        z_marker=float(args.z_marker),
        n_samples=int(args.ray_samples),
    )
    components = gv.compute_visibility(
        fov_mask=fov['fov_mask'],
        min_clearance=clearance,
        px_per_m_min=jac['px_per_m_min'],
        u=fov['u'],
        v=fov['v'],
        img_w=cam.img_width,
        img_h=cam.img_height,
        tau_clearance=float(args.tau_clearance),
        use_range=True,
        use_boundary=True,
    )
    p_mean = np.asarray(components['visibility_score'], dtype=float)

    driveable_mask = gv.in_any_prism(xs, ys, cfg['driveable_prisms'])
    unknown_mask = fov['fov_mask'] & driveable_mask & ~observed_mask
    if float(args.unknown_trust) >= 0.0:
        p_mean = np.where(unknown_mask, np.minimum(p_mean, float(args.unknown_trust)), p_mean)
    p_mean = np.clip(p_mean, float(args.min_prob), 1.0 - float(args.min_prob))

    boundary_uncertainty = np.exp(-np.square(clearance / max(float(args.tau_clearance), 1e-9)))
    unknown_extra = unknown_mask.astype(float)
    f_std = float(args.prior_logit_std) * (0.65 + 0.35 * np.clip(boundary_uncertainty, 0.0, 1.0))
    f_std = f_std + float(args.unknown_extra_logit_std) * unknown_extra
    f_std = np.clip(f_std, 1e-6, None)
    f_mean = _logit(p_mean)
    p_plan = np.clip(_sigmoid(f_mean - float(args.beta) * f_std), float(args.min_prob), 1.0 - float(args.min_prob))
    r_std, r_var = gv.trust_to_r_plan(p_plan, cfg['r_visible_uv'], cfg['r_miss_uv'])

    payload = {
        'xs': xs.astype(float),
        'ys': ys.astype(float),
        'X_train': np.empty((0, 2), dtype=float),
        'p_train': np.empty((0,), dtype=float),
        'F_mean_map': f_mean.astype(float),
        'F_std_map': f_std.astype(float),
        'P_mean_map': p_mean.astype(float),
        'P_conservative_plan_map': p_plan.astype(float),
        'rho_mean_map': p_mean.astype(float),
        'rho_std_map': f_std.astype(float),
        'rho_plan_map': p_plan.astype(float),
        'r_plan_std_map': r_std.astype(float),
        'r_plan_var_map': r_var.astype(float),
        'depth_height_map': height_map.astype(float),
        'depth_observed_mask': observed_mask.astype(np.uint8),
        'depth_unknown_mask': unknown_mask.astype(np.uint8),
        'depth_clearance_map': clearance.astype(float),
        'depth_fov_mask': np.asarray(fov['fov_mask'], dtype=np.uint8),
        'driveable_mask': driveable_mask.astype(np.uint8),
        'artifact_schema_version': np.asarray([int(ARTIFACT_SCHEMA_VERSION)], dtype=np.int32),
        'method_id': np.asarray(['depth_sensed_initial_gp'], dtype=np.str_),
        'target_id': np.asarray(['yolo_score_raw'], dtype=np.str_),
        'state_source': np.asarray(['DEPTH_SENSED_INITIAL'], dtype=np.str_),
        'reference_gp_path': np.asarray([str(reference_gp)], dtype=np.str_),
        'reference_gp_sha256': np.asarray([sha256_file(reference_gp)], dtype=np.str_),
        'depth_npy_path': np.asarray([str(depth_path)], dtype=np.str_),
        'depth_npy_sha256': np.asarray([sha256_file(depth_path)], dtype=np.str_),
        'campaign_config_path': np.asarray([str(campaign_config)], dtype=np.str_),
        'campaign_config_sha256': np.asarray([sha256_file(campaign_config)], dtype=np.str_),
        'z_marker': np.asarray([float(args.z_marker)], dtype=float),
        'tau_clearance': np.asarray([float(args.tau_clearance)], dtype=float),
        'depth_step': np.asarray([int(args.depth_step)], dtype=np.int32),
        'ray_samples': np.asarray([int(args.ray_samples)], dtype=np.int32),
        'unknown_trust': np.asarray([float(args.unknown_trust)], dtype=float),
        'beta': np.asarray([float(args.beta)], dtype=float),
        'min_prob': np.asarray([float(args.min_prob)], dtype=float),
        'r_visible_uv': np.asarray([cfg['r_visible_uv']], dtype=float),
        'r_miss_uv': np.asarray([cfg['r_miss_uv']], dtype=float),
    }
    payload.update(_reference_payload(reference_gp))

    artifact_path = output_dir / 'depth_sensed_initial_gp.npz'
    np.savez_compressed(artifact_path, **payload)

    figure_path = output_dir / 'depth_sensed_initial_gp_showcase.png'
    _plot_showcase(
        figure_path,
        xs=xs,
        ys=ys,
        height_map=height_map,
        observed_mask=observed_mask,
        unknown_mask=unknown_mask,
        p_mean=p_mean,
        p_plan=p_plan,
        driveable_mask=driveable_mask,
        meta=meta,
        driveable_prisms=cfg['driveable_prisms'],
    )

    driveable = driveable_mask.astype(bool)
    in_fov_driveable = driveable & np.asarray(fov['fov_mask'], dtype=bool)
    summary_row = {
        'artifact_path': str(artifact_path),
        'reference_gp_sha256': sha256_file(reference_gp),
        'depth_npy_sha256': sha256_file(depth_path),
        'depth_shape': f'{depth.shape[0]}x{depth.shape[1]}',
        'backprojected_points': str(int(points.shape[0])),
        'grid_nx': str(int(xs.size)),
        'grid_ny': str(int(ys.size)),
        'driveable_cells': str(int(np.sum(driveable))),
        'driveable_in_fov_cells': str(int(np.sum(in_fov_driveable))),
        'observed_driveable_in_fov_cells': str(int(np.sum(in_fov_driveable & observed_mask))),
        'unknown_driveable_in_fov_cells': str(int(np.sum(unknown_mask))),
        'p_mean_all': f'{float(np.mean(p_mean)):.10g}',
        'p_plan_all': f'{float(np.mean(p_plan)):.10g}',
        'p_mean_driveable_in_fov': f'{float(np.mean(p_mean[in_fov_driveable])):.10g}',
        'p_plan_driveable_in_fov': f'{float(np.mean(p_plan[in_fov_driveable])):.10g}',
        'height_max': f'{float(np.max(height_map)):.10g}',
        'clearance_min': f'{float(np.min(clearance)):.10g}',
        'clearance_max': f'{float(np.max(clearance)):.10g}',
    }
    summary_csv = output_dir / 'summary.csv'
    write_csv(
        summary_csv,
        (
            'artifact_path',
            'reference_gp_sha256',
            'depth_npy_sha256',
            'depth_shape',
            'backprojected_points',
            'grid_nx',
            'grid_ny',
            'driveable_cells',
            'driveable_in_fov_cells',
            'observed_driveable_in_fov_cells',
            'unknown_driveable_in_fov_cells',
            'p_mean_all',
            'p_plan_all',
            'p_mean_driveable_in_fov',
            'p_plan_driveable_in_fov',
            'height_max',
            'clearance_min',
            'clearance_max',
        ),
        [summary_row],
    )
    write_manifest(
        output_dir / 'manifest.json',
        {
            'artifact_path': str(artifact_path),
            'figure_png': str(figure_path),
            'summary_csv': str(summary_csv),
            'reference_gp': str(reference_gp),
            'reference_gp_sha256': sha256_file(reference_gp),
            'campaign_config': str(campaign_config),
            'campaign_config_sha256': sha256_file(campaign_config),
            'depth_npy': str(depth_path),
            'depth_npy_sha256': sha256_file(depth_path),
            'world': cfg['world'],
            'state_source': 'DEPTH_SENSED_INITIAL',
            'method': 'Back-project one depth frame to a sensed height map, raycast occlusion, then convert visibility to a planner-compatible GP prior.',
            'hyperparameters': {
                'z_marker': float(args.z_marker),
                'tau_clearance': float(args.tau_clearance),
                'depth_step': int(args.depth_step),
                'max_depth_m': float(args.max_depth_m),
                'ray_samples': int(args.ray_samples),
                'unknown_trust': float(args.unknown_trust),
                'prior_logit_std': float(args.prior_logit_std),
                'unknown_extra_logit_std': float(args.unknown_extra_logit_std),
                'beta': float(args.beta),
                'min_prob': float(args.min_prob),
            },
            'summary': summary_row,
            'notes': [
                'The depth frame supplies the occlusion height map. CAD occluder geometry is not used to compute the visibility map.',
                'Camera/grid metadata are copied from the locked GP artifact so the result can be consumed by the existing planner.',
                'P_mean_map is intended as the prior mean for trajectory updates; P_conservative_plan_map is the planner-facing conservative map.',
            ],
        },
    )

    return artifact_path, summary_row


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a depth-sensed initial GP artifact.')
    parser.add_argument('--reference-gp', default=str(DEFAULT_REFERENCE_GP))
    parser.add_argument('--campaign-config', default=str(DEFAULT_CAMPAIGN_CONFIG))
    parser.add_argument('--depth-npy', default=str(DEFAULT_DEPTH_NPY))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--z-marker', type=float, default=0.35)
    parser.add_argument('--tau-clearance', type=float, default=0.10)
    parser.add_argument('--depth-step', type=int, default=3)
    parser.add_argument('--max-depth-m', type=float, default=60.0)
    parser.add_argument('--ray-samples', type=int, default=48)
    parser.add_argument('--unknown-trust', type=float, default=0.05)
    parser.add_argument('--prior-logit-std', type=float, default=1.2)
    parser.add_argument('--unknown-extra-logit-std', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    args = parser.parse_args()

    artifact_path, summary = build(args)
    print(f'Wrote depth-sensed initial GP: {artifact_path}')
    print(f'Depth points: {summary["backprojected_points"]}')
    print(f'Driveable in-FOV cells: {summary["driveable_in_fov_cells"]}')
    print(f'P_mean driveable in-FOV: {summary["p_mean_driveable_in_fov"]}')
    print(f'P_plan driveable in-FOV: {summary["p_plan_driveable_in_fov"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
