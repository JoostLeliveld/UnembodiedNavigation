#!/usr/bin/env python3
"""Fit planner-compatible GP visibility artifacts from canonical scalar targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import (
    ACTIVE_METHOD_IDS,
    ARTIFACT_SCHEMA_VERSION,
    CURRENT_CAPTURE_DIR,
    CURRENT_GP_DIR,
    CURRENT_TARGETS_DIR,
    LOGS_ROOT,
    parse_float,
    read_csv_rows,
    repo_relative,
    safe_reset_generated_dir,
    visibility_geometry_sha256,
    write_csv,
    write_manifest,
)


def _clip_prob(p: np.ndarray | float, eps: float) -> np.ndarray | float:
    return np.clip(p, eps, 1.0 - eps)

def _sigmoid(x):
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-arr))

def _logit(p):
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


GP_METHOD_IDS = tuple(method for method in ACTIVE_METHOD_IDS if method != 'constant_R_efe')


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else {}


def _grid_vectors(rows: list[dict[str, str]], capture_manifest: dict, *, grid_nx: int, grid_ny: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = dict(capture_manifest.get('visibility_bounds') or {})
    sample_x = np.asarray([parse_float(row.get('x', ''), math.nan) for row in rows], dtype=float)
    sample_y = np.asarray([parse_float(row.get('y', ''), math.nan) for row in rows], dtype=float)
    finite_x = sample_x[np.isfinite(sample_x)]
    finite_y = sample_y[np.isfinite(sample_y)]
    if finite_x.size == 0 or finite_y.size == 0:
        raise RuntimeError('gp_targets.csv does not contain any finite x/y samples')

    xmin = float(bounds.get('xmin', np.min(finite_x)))
    xmax = float(bounds.get('xmax', np.max(finite_x)))
    ymin = float(bounds.get('ymin', np.min(finite_y)))
    ymax = float(bounds.get('ymax', np.max(finite_y)))
    if not (xmin < xmax and ymin < ymax):
        raise RuntimeError('Invalid grid bounds for GP fitting')

    if int(grid_nx) > 1 and int(grid_ny) > 1:
        xs = np.linspace(xmin, xmax, int(grid_nx))
        ys = np.linspace(ymin, ymax, int(grid_ny))
        return xs.astype(float), ys.astype(float)

    default_nx = int(bounds.get('nx', 0))
    default_ny = int(bounds.get('ny', 0))
    if default_nx > 1 and default_ny > 1:
        xs = np.linspace(xmin, xmax, default_nx)
        ys = np.linspace(ymin, ymax, default_ny)
        return xs.astype(float), ys.astype(float)

    uniq_x = np.unique(np.round(finite_x, 6))
    uniq_y = np.unique(np.round(finite_y, 6))
    if uniq_x.size >= 2 and uniq_y.size >= 2:
        return uniq_x.astype(float), uniq_y.astype(float)

    xs = np.linspace(xmin, xmax, 80)
    ys = np.linspace(ymin, ymax, 80)
    return xs.astype(float), ys.astype(float)


def _aggregate_targets(rows: list[dict[str, str]], target_key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[float, float], list[float]] = {}
    for row in rows:
        raw_value = str(row.get(target_key, '')).strip()
        if raw_value == '':
            continue
        x = parse_float(row.get('x', ''), math.nan)
        y = parse_float(row.get('y', ''), math.nan)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Missing x or y in row: {row}")
        val = parse_float(raw_value, math.nan)
        if not math.isfinite(val):
            continue
        key = (round(float(x), 6), round(float(y), 6))
        grouped.setdefault(key, []).append(float(val))

    if not grouped:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    keys = sorted(grouped.keys(), key=lambda item: (item[1], item[0]))
    X = np.asarray([[k[0], k[1]] for k in keys], dtype=float)
    p = np.asarray([float(np.mean(grouped[k])) for k in keys], dtype=float)
    return X, p


def _fit_gp_artifact(
    X_train: np.ndarray,
    p_train: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    min_prob: float,
    gp_length_scale: float,
    gp_noise_var: float,
    beta: float,
) -> dict[str, np.ndarray]:
    p_train = np.asarray(_clip_prob(p_train, min_prob), dtype=float)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    kernel = 1.0 * RBF(length_scale=float(gp_length_scale))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(max(gp_noise_var, 1e-8)),
        normalize_y=True,
        optimizer=None,
    ).fit(X_train, _logit(p_train))

    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    mu_f, sigma_f = gp.predict(XY, return_std=True)
    p_mean = _sigmoid(mu_f)
    p_cons = _sigmoid(mu_f - float(beta) * sigma_f)
    return {
        'xs': xs.astype(float),
        'ys': ys.astype(float),
        'X_train': X_train.astype(float),
        'p_train': p_train.astype(float),
        'F_mean_map': mu_f.reshape(Yg.shape).astype(float),
        'F_std_map': np.clip(sigma_f.reshape(Yg.shape), 0.0, None).astype(float),
        'P_mean_map': _clip_prob(p_mean.reshape(Yg.shape), min_prob).astype(float),
        'P_conservative_plan_map': _clip_prob(p_cons.reshape(Yg.shape), min_prob).astype(float),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Fit planner-compatible GP artifacts from gp_targets.csv.')
    parser.add_argument('--gp-targets', default=str(CURRENT_TARGETS_DIR / 'gp_targets_xy_aggregated.csv'))
    parser.add_argument('--capture-manifest', default=str(CURRENT_CAPTURE_DIR / 'capture_manifest.json'))
    parser.add_argument('--out', default=str(CURRENT_GP_DIR))
    parser.add_argument('--grid-nx', type=int, default=120)
    parser.add_argument('--grid-ny', type=int, default=120)
    parser.add_argument('--gp-length-scale', type=float, default=1.35)
    parser.add_argument('--gp-noise-var', type=float, default=0.12)
    parser.add_argument('--beta', type=float, default=0.75)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    parser.add_argument('--max-train-points', type=int, default=0, help='Optional deterministic xy-point subsample for low-data GP diagnostics.')
    parser.add_argument('--subsample-seed', type=int, default=0)
    args = parser.parse_args()

    gp_targets_path = Path(args.gp_targets).expanduser().resolve()
    if not gp_targets_path.is_file():
        legacy_path = (CURRENT_TARGETS_DIR / 'gp_targets.csv').resolve()
        if gp_targets_path == (CURRENT_TARGETS_DIR / 'gp_targets_xy_aggregated.csv').resolve() and legacy_path.is_file():
            gp_targets_path = legacy_path
        else:
            raise RuntimeError(f'GP targets CSV not found: {gp_targets_path}')

    capture_manifest_path = Path(args.capture_manifest).expanduser().resolve()
    capture_manifest = _load_json(capture_manifest_path)
    rows = read_csv_rows(gp_targets_path)
    if not rows:
        raise RuntimeError(f'GP targets CSV is empty: {gp_targets_path}')

    xs, ys = _grid_vectors(rows, capture_manifest, grid_nx=int(args.grid_nx), grid_ny=int(args.grid_ny))
    output_dir = safe_reset_generated_dir(Path(args.out), allowed_root=LOGS_ROOT)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict[str, str]] = []
    fitted_methods: list[str] = []

    for method_id in GP_METHOD_IDS:
        X_train, p_train = _aggregate_targets(rows, method_id)
        if int(args.max_train_points) > 0 and X_train.shape[0] > int(args.max_train_points):
            rng = np.random.default_rng(int(args.subsample_seed))
            keep = np.sort(rng.choice(X_train.shape[0], size=int(args.max_train_points), replace=False))
            X_train = X_train[keep]
            p_train = p_train[keep]
        available = bool(X_train.shape[0] >= 4)
        artifact_path = output_dir / f'{method_id}_gp.npz'
        if available:
            fit = _fit_gp_artifact(
                X_train,
                p_train,
                xs,
                ys,
                min_prob=float(args.min_prob),
                gp_length_scale=float(args.gp_length_scale),
                gp_noise_var=float(args.gp_noise_var),
                beta=float(args.beta),
            )
            metadata = {
                'artifact_schema_version': np.asarray([int(ARTIFACT_SCHEMA_VERSION)], dtype=np.int32),
                'camera_pos': np.asarray(capture_manifest.get('camera_pos', [-3.0, -3.0, 6.0]), dtype=float),
                'camera_pose': np.asarray(capture_manifest.get('camera_pose', [-3.0, -3.0, 6.0, 0.0, 0.0, 0.0]), dtype=float),
                'look_at': np.asarray(capture_manifest.get('look_at', [1.5, 1.5, 0.0]), dtype=float),
                'img_width': np.asarray([int(capture_manifest.get('img_width', 1280))], dtype=np.int32),
                'img_height': np.asarray([int(capture_manifest.get('img_height', 720))], dtype=np.int32),
                'fov_h_rad': np.asarray([float(capture_manifest.get('fov_h_rad', 1.5708))], dtype=float),
                'target_height': np.asarray([float(capture_manifest.get('oracle_target_height_m', 0.0))], dtype=float),
                'geometry_json': np.asarray([str(capture_manifest.get('geometry_json', ''))], dtype=np.str_),
                'geometry_sha256': np.asarray([visibility_geometry_sha256(str(capture_manifest.get('geometry_json', '')))], dtype=np.str_),
                'method_id': np.asarray([method_id], dtype=np.str_),
                'world': np.asarray([str(capture_manifest.get('world', ''))], dtype=np.str_),
                'world_name': np.asarray([str(capture_manifest.get('world_name', ''))], dtype=np.str_),
                'world_path': np.asarray([str(capture_manifest.get('world_path', ''))], dtype=np.str_),
                'gp_length_scale': np.asarray([float(args.gp_length_scale)], dtype=float),
                'gp_noise_var': np.asarray([float(args.gp_noise_var)], dtype=float),
                'beta': np.asarray([float(args.beta)], dtype=float),
                'min_prob': np.asarray([float(args.min_prob)], dtype=float),
                'max_train_points': np.asarray([int(args.max_train_points)], dtype=np.int32),
                'subsample_seed': np.asarray([int(args.subsample_seed)], dtype=np.int32),
            }
            np.savez_compressed(artifact_path, **fit, **metadata)
            fitted_methods.append(method_id)

        summary_rows.append({
            'method_id': method_id,
            'available': str(int(available)),
            'train_points': str(int(X_train.shape[0])),
            'target_mean': '' if p_train.size == 0 else f'{float(np.mean(p_train)):.8f}',
            'target_min': '' if p_train.size == 0 else f'{float(np.min(p_train)):.8f}',
            'target_max': '' if p_train.size == 0 else f'{float(np.max(p_train)):.8f}',
            'artifact_path': repo_relative(artifact_path, output_dir) if available else '',
        })

    write_csv(
        output_dir / 'gp_fit_summary.csv',
        ('method_id', 'available', 'train_points', 'target_mean', 'target_min', 'target_max', 'artifact_path'),
        summary_rows,
    )
    write_manifest(output_dir / 'gp_manifest.json', {
        'gp_targets_csv': str(gp_targets_path),
        'capture_manifest': str(capture_manifest_path) if capture_manifest_path.is_file() else '',
        'grid': {
            'nx': int(xs.size),
            'ny': int(ys.size),
            'xmin': float(xs[0]),
            'xmax': float(xs[-1]),
            'ymin': float(ys[0]),
            'ymax': float(ys[-1]),
        },
        'gp_length_scale': float(args.gp_length_scale),
        'gp_noise_var': float(args.gp_noise_var),
        'beta': float(args.beta),
        'min_prob': float(args.min_prob),
        'max_train_points': int(args.max_train_points),
        'subsample_seed': int(args.subsample_seed),
        'available_methods': fitted_methods,
        'missing_methods': [method for method in GP_METHOD_IDS if method not in fitted_methods],
        'notes': [
            'This shared-stage fitter scans the heading-aggregated GP target table and fits only target columns that are already populated.',
            'If gp_targets_xy_aggregated.csv is absent, the fitter falls back to gp_targets.csv and still averages repeated (x, y) rows internally.',
            'GP artifacts are planner-compatible via xs, ys, and P_conservative_plan_map. Extra metadata is stored for plotting and reporting.',
            'Strict schema cutover: legacy artifacts with P_map are no longer accepted by the planner or plotting scripts.',
        ],
    })
    print(f'Wrote GP artifacts to {output_dir}')
    print(f'Fitted methods: {", ".join(fitted_methods) if fitted_methods else "none"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
