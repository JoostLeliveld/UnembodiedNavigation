#!/usr/bin/env python3
"""Fit belief-aware detector-reliability GP baselines from event logs."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common import ARTIFACT_SCHEMA_VERSION, LOGS_ROOT, REPO_ROOT, read_csv_rows, repo_relative, sha256_file, write_csv, write_manifest

_SHARED_METRICS_DIR = Path(__file__).resolve().parents[1] / 'shared'
if str(_SHARED_METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_METRICS_DIR))
import metrics as M  # noqa: E402  # canonical paper metrics; never reimplement here


DEFAULT_EVENTS = LOGS_ROOT / 'belief_gp_events' / 'events.csv'
DEFAULT_OUT = LOGS_ROOT / 'belief_aware_gp_v1'
DEFAULT_GRID_GP = REPO_ROOT / 'paper_artifacts' / 'gp' / 'warehouse_visibility_gp_v1' / 'yolo_score_raw_gp.npz'

MODES = ('naive', 'uncertainty_weighted', 'belief_spread', 'expected_kernel')


@dataclass(frozen=True)
class EventData:
    X: np.ndarray
    y: np.ndarray
    cov: np.ndarray
    run_ids: np.ndarray
    rows_used: int
    target_id: str


@dataclass(frozen=True)
class AggregateData:
    X: np.ndarray
    y: np.ndarray
    cov: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class PriorMap:
    path: Path
    xs: np.ndarray
    ys: np.ndarray
    p_mean: np.ndarray
    p_plan: np.ndarray
    map_key: str
    sha256: str


def _clip_prob(p: np.ndarray | float, eps: float) -> np.ndarray | float:
    return np.clip(p, eps, 1.0 - eps)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-arr))


def _logit(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, '')
    if raw in (None, '', 'nan', 'NaN'):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _finite_cov(cov_xx: float, cov_xy: float, cov_yy: float, *, eig_floor: float = 1e-9) -> np.ndarray | None:
    if not (math.isfinite(cov_xx) and math.isfinite(cov_yy)):
        return None
    cov_xy = float(cov_xy) if math.isfinite(cov_xy) else 0.0
    S = np.asarray([[float(cov_xx), cov_xy], [cov_xy, float(cov_yy)]], dtype=float)
    S = 0.5 * (S + S.T)
    try:
        vals, vecs = np.linalg.eigh(S)
    except np.linalg.LinAlgError:
        return None
    vals = np.clip(vals, eig_floor, None)
    return (vecs * vals) @ vecs.T


def _load_events(path: Path, *, target: str, min_prob: float) -> EventData:
    rows = read_csv_rows(path)
    X_values: list[list[float]] = []
    y_values: list[float] = []
    cov_values: list[np.ndarray] = []
    run_ids: list[str] = []
    for row in rows:
        x = _f(row, 'm_x')
        y = _f(row, 'm_y')
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        S = _finite_cov(_f(row, 'S_xx'), _f(row, 'S_xy', 0.0), _f(row, 'S_yy'))
        if S is None:
            continue
        if target == 'hit':
            target_value = _f(row, 'det_hit')
        elif target == 'score':
            target_value = _f(row, 'yolo_score_raw')
            if not math.isfinite(target_value) and _f(row, 'det_hit') <= 0.0:
                target_value = 0.0
        else:
            raise ValueError(f'Unsupported target: {target}')
        if not math.isfinite(target_value):
            continue
        target_value = float(np.clip(target_value, 0.0, 1.0))
        X_values.append([float(x), float(y)])
        y_values.append(float(target_value))
        cov_values.append(S)
        run_ids.append(str(row.get('run_dir', '') or row.get('run_id', '')))
    if not X_values:
        raise RuntimeError(f'No finite belief GP events for target={target}: {path}')
    return EventData(
        X=np.asarray(X_values, dtype=float),
        y=np.asarray(y_values, dtype=float),
        cov=np.asarray(cov_values, dtype=float),
        run_ids=np.asarray(run_ids, dtype=np.str_),
        rows_used=int(len(X_values)),
        target_id='det_hit' if target == 'hit' else 'yolo_score_raw',
    )


def _aggregate_events(data: EventData, *, resolution_m: float, max_bin_weight: float) -> AggregateData:
    if resolution_m <= 0.0:
        count = np.ones((data.X.shape[0],), dtype=float)
        return AggregateData(data.X, data.y, data.cov, count)

    grouped: dict[tuple[int, int], list[int]] = {}
    scaled = np.round(data.X / float(resolution_m)).astype(int)
    for idx, key_values in enumerate(scaled):
        grouped.setdefault((int(key_values[0]), int(key_values[1])), []).append(idx)

    X_out: list[np.ndarray] = []
    y_out: list[float] = []
    cov_out: list[np.ndarray] = []
    count_out: list[float] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[1], item[0])):
        indices = np.asarray(grouped[key], dtype=int)
        points = data.X[indices]
        mean_xy = np.mean(points, axis=0)
        mean_y = float(np.mean(data.y[indices]))
        mean_cov = np.mean(data.cov[indices], axis=0)
        if points.shape[0] > 1:
            centered = points - mean_xy
            mean_cov = mean_cov + (centered.T @ centered) / float(points.shape[0])
        S = _finite_cov(float(mean_cov[0, 0]), float(mean_cov[0, 1]), float(mean_cov[1, 1]))
        if S is None:
            continue
        X_out.append(mean_xy)
        y_out.append(mean_y)
        cov_out.append(S)
        count_out.append(min(float(indices.size), float(max_bin_weight)))
    return AggregateData(
        X=np.asarray(X_out, dtype=float),
        y=np.asarray(y_out, dtype=float),
        cov=np.asarray(cov_out, dtype=float),
        count=np.asarray(count_out, dtype=float),
    )


def _sigma_points(mean: np.ndarray, cov: np.ndarray, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.clip(vals, 1e-9, None)
    axes = vecs @ np.diag(np.sqrt(vals) * float(scale))
    points = [np.asarray(mean, dtype=float)]
    weights = [0.2]
    for dim in range(2):
        offset = axes[:, dim]
        points.append(mean + offset)
        points.append(mean - offset)
        weights.extend([0.2, 0.2])
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def _fit_inputs(
    agg: AggregateData,
    *,
    mode: str,
    noise_var: float,
    gp_length_scale: float,
    pose_length_scale: float,
    min_certainty: float,
    spread_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    base_alpha = float(max(noise_var, 1e-8))
    effective_count = np.clip(agg.count, 1.0, None)
    if mode == 'naive':
        alpha = base_alpha / effective_count
        return agg.X, agg.y, alpha, {'mean_certainty': 1.0, 'fit_points_per_aggregate': 1.0}

    if mode == 'uncertainty_weighted':
        trace = np.trace(agg.cov, axis1=1, axis2=2)
        denom = max(float(pose_length_scale) ** 2, 1e-9)
        certainty = np.exp(-0.5 * np.clip(trace, 0.0, None) / denom)
        certainty = np.clip(certainty, float(min_certainty), 1.0)
        alpha = base_alpha / (effective_count * certainty)
        return agg.X, agg.y, alpha, {
            'mean_certainty': float(np.mean(certainty)),
            'fit_points_per_aggregate': 1.0,
        }

    if mode == 'belief_spread':
        X_fit: list[np.ndarray] = []
        y_fit: list[float] = []
        alpha_fit: list[float] = []
        for mean_xy, target, cov, count in zip(agg.X, agg.y, agg.cov, effective_count):
            points, weights = _sigma_points(mean_xy, cov, scale=float(spread_scale))
            for point, weight in zip(points, weights):
                X_fit.append(point)
                y_fit.append(float(target))
                alpha_fit.append(base_alpha / (float(count) * float(weight)))
        return (
            np.asarray(X_fit, dtype=float),
            np.asarray(y_fit, dtype=float),
            np.asarray(alpha_fit, dtype=float),
            {'mean_certainty': 1.0, 'fit_points_per_aggregate': 5.0},
        )

    if mode == 'expected_kernel':
        alpha = base_alpha / effective_count
        self_cov = _expected_rbf_self_covariance(agg.cov, length_scale=float(gp_length_scale))
        return agg.X, agg.y, alpha, {
            'mean_certainty': float(np.mean(self_cov)),
            'fit_points_per_aggregate': 1.0,
        }

    raise ValueError(f'Unsupported mode: {mode}')


def _expected_rbf_self_covariance(covs: np.ndarray, *, length_scale: float) -> np.ndarray:
    ell2 = float(length_scale) ** 2
    covs = np.asarray(covs, dtype=float)
    a = ell2 + 2.0 * covs[:, 0, 0]
    b = 2.0 * covs[:, 0, 1]
    c = ell2 + 2.0 * covs[:, 1, 1]
    det = np.maximum(a * c - b * b, 1e-18)
    return np.clip(ell2 / np.sqrt(det), 1e-12, 1.0)


def _expected_rbf_kernel(
    A_X: np.ndarray,
    A_cov: np.ndarray,
    B_X: np.ndarray,
    B_cov: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    """Return E[k(x_a, x_b)] for x_a~N(A_X,A_cov), x_b~N(B_X,B_cov)."""
    A_X = np.asarray(A_X, dtype=float)
    B_X = np.asarray(B_X, dtype=float)
    A_cov = np.asarray(A_cov, dtype=float)
    B_cov = np.asarray(B_cov, dtype=float)
    ell2 = float(length_scale) ** 2
    K = np.empty((A_X.shape[0], B_X.shape[0]), dtype=float)
    for j in range(B_X.shape[0]):
        S = A_cov + B_cov[j]
        a = ell2 + S[:, 0, 0]
        b = S[:, 0, 1]
        c = ell2 + S[:, 1, 1]
        det = np.maximum(a * c - b * b, 1e-18)
        dx = A_X[:, 0] - B_X[j, 0]
        dy = A_X[:, 1] - B_X[j, 1]
        mahal = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det
        factor = ell2 / np.sqrt(det)
        K[:, j] = factor * np.exp(-0.5 * mahal)
    return np.clip(K, 0.0, 1.0)


def _solve_spd_with_jitter(K: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
    eye = np.eye(K.shape[0], dtype=float)
    last_exc: Exception | None = None
    for jitter in (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        try:
            L = np.linalg.cholesky(K + jitter * eye)
            solved = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            return L, solved, float(jitter)
        except np.linalg.LinAlgError as exc:
            last_exc = exc
    raise RuntimeError('Expected-kernel GP covariance is not positive definite after jitter escalation') from last_exc


def _fit_predict_expected_kernel_gp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_cov: np.ndarray,
    alpha: np.ndarray,
    query_X: np.ndarray,
    *,
    query_cov: np.ndarray | None,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    return _fit_predict_expected_kernel_latent(
        train_X,
        _logit(train_y),
        train_cov,
        alpha,
        query_X,
        query_cov=query_cov,
        length_scale=float(length_scale),
    )


def _fit_predict_expected_kernel_latent(
    train_X: np.ndarray,
    train_latent: np.ndarray,
    train_cov: np.ndarray,
    alpha: np.ndarray,
    query_X: np.ndarray,
    *,
    query_cov: np.ndarray | None,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_X = np.asarray(train_X, dtype=float)
    train_latent = np.asarray(train_latent, dtype=float)
    train_cov = np.asarray(train_cov, dtype=float)
    query_X = np.asarray(query_X, dtype=float)
    if query_cov is None:
        query_cov = np.zeros((query_X.shape[0], 2, 2), dtype=float)
    else:
        query_cov = np.asarray(query_cov, dtype=float)

    y_mean = float(np.mean(train_latent))
    y_std = float(np.std(train_latent))
    if not math.isfinite(y_std) or y_std < 1e-12:
        y_std = 1.0
    y_norm = (train_latent - y_mean) / y_std

    K = _expected_rbf_kernel(train_X, train_cov, train_X, train_cov, length_scale=float(length_scale))
    K = 0.5 * (K + K.T)
    K[np.diag_indices_from(K)] = np.maximum(np.diag(K), _expected_rbf_self_covariance(train_cov, length_scale=float(length_scale)))
    K = K + np.diag(np.asarray(alpha, dtype=float))
    L, solved, jitter = _solve_spd_with_jitter(K, y_norm)

    K_star = _expected_rbf_kernel(query_X, query_cov, train_X, train_cov, length_scale=float(length_scale))
    mu_norm = K_star @ solved
    v = np.linalg.solve(L, K_star.T)
    prior_var = _expected_rbf_self_covariance(query_cov, length_scale=float(length_scale))
    var_norm = np.maximum(prior_var - np.sum(v * v, axis=0), 0.0)
    mu_f = y_mean + y_std * mu_norm
    sigma_f = y_std * np.sqrt(var_norm)
    return np.asarray(mu_f, dtype=float), np.asarray(sigma_f, dtype=float), float(jitter)


def _fit_predict_gp(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    alpha: np.ndarray,
    query_X: np.ndarray,
    *,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _fit_predict_latent_gp(
        X_fit,
        _logit(np.asarray(y_fit, dtype=float)),
        alpha,
        query_X,
        length_scale=float(length_scale),
    )


def _fit_predict_latent_gp(
    X_fit: np.ndarray,
    latent_fit: np.ndarray,
    alpha: np.ndarray,
    query_X: np.ndarray,
    *,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    kernel = 1.0 * RBF(length_scale=float(length_scale))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.asarray(alpha, dtype=float),
        normalize_y=True,
        optimizer=None,
    ).fit(np.asarray(X_fit, dtype=float), np.asarray(latent_fit, dtype=float))
    mu_f, sigma_f = gp.predict(np.asarray(query_X, dtype=float), return_std=True)
    return np.asarray(mu_f, dtype=float), np.clip(np.asarray(sigma_f, dtype=float), 0.0, None)


def _interp_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray, X: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    grid = np.asarray(grid, dtype=float)
    X = np.asarray(X, dtype=float)
    x = np.clip(X[:, 0], float(xs[0]), float(xs[-1]))
    y = np.clip(X[:, 1], float(ys[0]), float(ys[-1]))
    ix1 = np.searchsorted(xs, x, side='right')
    iy1 = np.searchsorted(ys, y, side='right')
    ix1 = np.clip(ix1, 1, xs.size - 1)
    iy1 = np.clip(iy1, 1, ys.size - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1
    x0 = xs[ix0]
    x1 = xs[ix1]
    y0 = ys[iy0]
    y1 = ys[iy1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 != y0))
    g00 = grid[iy0, ix0]
    g10 = grid[iy0, ix1]
    g01 = grid[iy1, ix0]
    g11 = grid[iy1, ix1]
    return (
        (1.0 - tx) * (1.0 - ty) * g00
        + tx * (1.0 - ty) * g10
        + (1.0 - tx) * ty * g01
        + tx * ty * g11
    )


def _load_prior_map(path: Path | str, *, map_key: str, min_prob: float) -> PriorMap | None:
    raw = str(path or '').strip()
    if not raw:
        return None
    prior_path = Path(raw).expanduser().resolve()
    if not prior_path.is_file():
        raise RuntimeError(f'Prior GP artifact not found: {prior_path}')
    with np.load(prior_path, allow_pickle=False) as data:
        xs = np.asarray(data['xs'], dtype=float)
        ys = np.asarray(data['ys'], dtype=float)
        p_mean = np.asarray(data['P_mean_map'], dtype=float)
        p_plan = np.asarray(data['P_conservative_plan_map'], dtype=float)
        if map_key == 'P_mean_map':
            chosen_key = 'P_mean_map'
        elif map_key == 'P_conservative_plan_map':
            chosen_key = 'P_conservative_plan_map'
        else:
            raise RuntimeError(f'Unsupported prior map key: {map_key}')
    expected_shape = (ys.size, xs.size)
    if p_mean.shape != expected_shape or p_plan.shape != expected_shape:
        raise RuntimeError(f'Prior GP map shape mismatch in {prior_path}')
    return PriorMap(
        path=prior_path,
        xs=xs,
        ys=ys,
        p_mean=np.clip(p_mean, float(min_prob), 1.0 - float(min_prob)),
        p_plan=np.clip(p_plan, float(min_prob), 1.0 - float(min_prob)),
        map_key=chosen_key,
        sha256=sha256_file(prior_path),
    )


def _prior_prob(prior: PriorMap | None, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if prior is None:
        return np.full((X.shape[0],), 0.5, dtype=float)
    grid = prior.p_mean if prior.map_key == 'P_mean_map' else prior.p_plan
    return np.clip(_interp_grid(prior.xs, prior.ys, grid, X), 1e-6, 1.0 - 1e-6)


def _prior_logit(prior: PriorMap | None, X: np.ndarray) -> np.ndarray:
    if prior is None:
        return np.zeros((np.asarray(X).shape[0],), dtype=float)
    return _logit(_prior_prob(prior, X))


def _prior_artifact_payload(prior: PriorMap | None) -> dict[str, np.ndarray]:
    if prior is None:
        return {
            'uses_prior_mean_function': np.asarray([0], dtype=np.int32),
            'prior_gp_path': np.asarray([''], dtype=np.str_),
            'prior_gp_sha256': np.asarray([''], dtype=np.str_),
            'prior_map_key': np.asarray([''], dtype=np.str_),
        }
    return {
        'uses_prior_mean_function': np.asarray([1], dtype=np.int32),
        'prior_gp_path': np.asarray([str(prior.path)], dtype=np.str_),
        'prior_gp_sha256': np.asarray([prior.sha256], dtype=np.str_),
        'prior_map_key': np.asarray([prior.map_key], dtype=np.str_),
        'prior_P_mean_map': prior.p_mean.astype(float),
        'prior_P_conservative_plan_map': prior.p_plan.astype(float),
    }


def _load_grid(path: Path, *, fallback_X: np.ndarray, grid_nx: int, grid_ny: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metadata: dict[str, np.ndarray] = {}
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            xs = np.asarray(data['xs'], dtype=float)
            ys = np.asarray(data['ys'], dtype=float)
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
                    metadata[key] = np.asarray(data[key])
            return xs, ys, metadata
    pad = 0.5
    xs = np.linspace(float(np.min(fallback_X[:, 0]) - pad), float(np.max(fallback_X[:, 0]) + pad), int(grid_nx))
    ys = np.linspace(float(np.min(fallback_X[:, 1]) - pad), float(np.max(fallback_X[:, 1]) + pad), int(grid_ny))
    return xs.astype(float), ys.astype(float), metadata


def _artifact_for_mode(
    agg: AggregateData,
    *,
    mode: str,
    xs: np.ndarray,
    ys: np.ndarray,
    metadata: dict[str, np.ndarray],
    prior: PriorMap | None,
    target_id: str,
    events_sha256: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    X_fit, y_fit, alpha, mode_summary = _fit_inputs(
        agg,
        mode=mode,
        noise_var=float(args.gp_noise_var),
        gp_length_scale=float(args.gp_length_scale),
        pose_length_scale=float(args.pose_length_scale),
        min_certainty=float(args.min_certainty),
        spread_scale=float(args.spread_scale),
    )
    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    prior_query = _prior_logit(prior, XY)
    prior_fit = _prior_logit(prior, X_fit)
    expected_kernel_jitter = 0.0
    if mode == 'expected_kernel':
        query_cov = np.zeros((XY.shape[0], 2, 2), dtype=float)
        train_latent = _logit(agg.y) - _prior_logit(prior, agg.X)
        resid_mu, sigma_f, expected_kernel_jitter = _fit_predict_expected_kernel_latent(
            agg.X,
            train_latent,
            agg.cov,
            alpha,
            XY,
            query_cov=query_cov,
            length_scale=float(args.gp_length_scale),
        )
        mu_f = prior_query + resid_mu
    else:
        latent_fit = _logit(y_fit) - prior_fit
        resid_mu, sigma_f = _fit_predict_latent_gp(X_fit, latent_fit, alpha, XY, length_scale=float(args.gp_length_scale))
        mu_f = prior_query + resid_mu
    p_mean = _clip_prob(_sigmoid(mu_f), float(args.min_prob))
    p_plan = _clip_prob(_sigmoid(mu_f - float(args.beta) * sigma_f), float(args.min_prob))
    artifact = {
        'xs': xs.astype(float),
        'ys': ys.astype(float),
        'X_train': agg.X.astype(float),
        'p_train': _clip_prob(agg.y, float(args.min_prob)).astype(float),
        'train_count': agg.count.astype(float),
        'train_cov': agg.cov.astype(float),
        'X_fit': X_fit.astype(float),
        'p_fit': _clip_prob(y_fit, float(args.min_prob)).astype(float),
        'alpha_fit': np.asarray(alpha, dtype=float),
        'expected_kernel_jitter': np.asarray([float(expected_kernel_jitter)], dtype=float),
        'F_mean_map': mu_f.reshape(Yg.shape).astype(float),
        'F_std_map': sigma_f.reshape(Yg.shape).astype(float),
        'P_mean_map': p_mean.reshape(Yg.shape).astype(float),
        'P_conservative_plan_map': p_plan.reshape(Yg.shape).astype(float),
        'rho_mean_map': p_mean.reshape(Yg.shape).astype(float),
        'rho_std_map': sigma_f.reshape(Yg.shape).astype(float),
        'rho_plan_map': p_plan.reshape(Yg.shape).astype(float),
        'artifact_schema_version': np.asarray([int(ARTIFACT_SCHEMA_VERSION)], dtype=np.int32),
        'method_id': np.asarray([f'{target_id}_{mode}'], dtype=np.str_),
        'target_id': np.asarray([target_id], dtype=np.str_),
        'belief_mode': np.asarray([mode], dtype=np.str_),
        'state_source': np.asarray(['BELIEF'], dtype=np.str_),
        'events_sha256': np.asarray([events_sha256], dtype=np.str_),
        'gp_length_scale': np.asarray([float(args.gp_length_scale)], dtype=float),
        'gp_noise_var': np.asarray([float(args.gp_noise_var)], dtype=float),
        'beta': np.asarray([float(args.beta)], dtype=float),
        'min_prob': np.asarray([float(args.min_prob)], dtype=float),
        'aggregate_resolution_m': np.asarray([float(args.aggregate_resolution_m)], dtype=float),
        'pose_length_scale': np.asarray([float(args.pose_length_scale)], dtype=float),
        'min_certainty': np.asarray([float(args.min_certainty)], dtype=float),
        'spread_scale': np.asarray([float(args.spread_scale)], dtype=float),
    }
    artifact.update(metadata)
    artifact.update(_prior_artifact_payload(prior))
    summary = {
        'aggregate_points': float(agg.X.shape[0]),
        'fit_points': float(X_fit.shape[0]),
        'target_mean': float(np.mean(agg.y)),
        'alpha_mean': float(np.mean(alpha)),
        'prior_map_mean': float('nan') if prior is None else float(np.mean(prior.p_mean if prior.map_key == 'P_mean_map' else prior.p_plan)),
        'p_mean_map_mean': float(np.mean(p_mean)),
        'p_conservative_plan_map_mean': float(np.mean(p_plan)),
        'expected_kernel_jitter': float(expected_kernel_jitter),
        **mode_summary,
    }
    return artifact, summary


def _subset_events(data: EventData, mask: np.ndarray) -> EventData:
    """Subset an event table while retaining route/run identifiers."""

    mask = np.asarray(mask, dtype=bool)
    return EventData(
        X=data.X[mask],
        y=data.y[mask],
        cov=data.cov[mask],
        run_ids=data.run_ids[mask],
        rows_used=int(np.sum(mask)),
        target_id=data.target_id,
    )


def _predict_mode_at_events(
    train_agg: AggregateData,
    test_data: EventData,
    *,
    mode: str,
    prior: PriorMap | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Predict a frozen GP mode at held-out uncertain belief inputs."""

    X_fit, y_fit, alpha, _summary = _fit_inputs(
        train_agg,
        mode=mode,
        noise_var=float(args.gp_noise_var),
        gp_length_scale=float(args.gp_length_scale),
        pose_length_scale=float(args.pose_length_scale),
        min_certainty=float(args.min_certainty),
        spread_scale=float(args.spread_scale),
    )
    if mode == 'expected_kernel':
        train_latent = _logit(train_agg.y) - _prior_logit(prior, train_agg.X)
        residual, _sigma_f, _jitter = _fit_predict_expected_kernel_latent(
            train_agg.X,
            train_latent,
            train_agg.cov,
            alpha,
            test_data.X,
            query_cov=test_data.cov,
            length_scale=float(args.gp_length_scale),
        )
        latent = _prior_logit(prior, test_data.X) + residual
    else:
        latent_fit = _logit(y_fit) - _prior_logit(prior, X_fit)
        residual, _sigma_f = _fit_predict_latent_gp(
            X_fit,
            latent_fit,
            alpha,
            test_data.X,
            length_scale=float(args.gp_length_scale),
        )
        latent = _prior_logit(prior, test_data.X) + residual
    return M.clip_prob(M.sigmoid(latent), eps=float(args.min_prob))


def _route_disjoint_validation(
    train_data: EventData,
    heldout_data: EventData,
    *,
    modes: tuple[str, ...],
    prior: PriorMap | None,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    """Evaluate GP variants on complete held-out runs, never mixed frames."""

    if not train_data.rows_used or not heldout_data.rows_used:
        return []
    train_agg = _aggregate_events(
        train_data,
        resolution_m=float(args.aggregate_resolution_m),
        max_bin_weight=float(args.max_bin_weight),
    )
    if train_agg.X.shape[0] < 4:
        raise RuntimeError(
            f'Route-disjoint evaluation needs at least 4 train aggregate points, got {train_agg.X.shape[0]}'
        )
    y = heldout_data.y
    predictions: dict[str, np.ndarray] = {
        'constant_train_mean': np.full_like(y, float(np.mean(train_data.y)), dtype=float),
    }
    if prior is not None:
        predictions['prior_only'] = M.clip_prob(_prior_prob(prior, heldout_data.X), eps=float(args.min_prob))
    for mode in modes:
        predictions[mode] = _predict_mode_at_events(
            train_agg,
            heldout_data,
            mode=mode,
            prior=prior,
            args=args,
        )

    rows: list[dict[str, str]] = []
    for mode, prediction in predictions.items():
        auroc = M.auroc(y, prediction)
        fhtr = M.fhtr(y, prediction, tau_high=0.70)
        rows.append(
            {
                'mode': mode,
                'train_runs': '|'.join(sorted({str(item) for item in train_data.run_ids})),
                'heldout_runs': '|'.join(sorted({str(item) for item in heldout_data.run_ids})),
                'train_events': str(train_data.rows_used),
                'heldout_events': str(heldout_data.rows_used),
                'brier': f'{M.brier(y, prediction):.10g}',
                'logloss': f'{M.logloss(y, prediction, eps=float(args.min_prob)):.10g}',
                'auroc': '' if math.isnan(auroc) else f'{auroc:.10g}',
                'ece': f'{M.ece(y, prediction):.10g}',
                'false_high_trust_rate': '' if math.isnan(fhtr) else f'{fhtr:.10g}',
                'prediction_mean': f'{float(np.mean(prediction)):.10g}',
                'target_mean': f'{float(np.mean(y)):.10g}',
            }
        )
    return rows


def _logloss(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float) -> float:
    p = np.clip(p_pred, eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _auc(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    if not np.all((y == 0.0) | (y == 1.0)):
        return math.nan
    pos = p_pred[y >= 0.5]
    neg = p_pred[y < 0.5]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    order_score = 0.0
    for value in pos:
        order_score += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return float(order_score / (pos.size * neg.size))


def _fold_ids(run_ids: np.ndarray, folds: int) -> dict[str, int]:
    unique_runs = sorted({str(run_id) for run_id in run_ids})
    fold_count = max(2, min(int(folds), len(unique_runs)))
    return {run_id: idx % fold_count for idx, run_id in enumerate(unique_runs)}


def _validate_modes(
    data: EventData,
    *,
    modes: tuple[str, ...],
    prior: PriorMap | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fold_map = _fold_ids(data.run_ids, int(args.folds))
    fold_count = max(fold_map.values()) + 1
    fold_rows: list[dict[str, str]] = []

    for fold in range(fold_count):
        train_mask = np.asarray([fold_map[str(run_id)] != fold for run_id in data.run_ids], dtype=bool)
        test_mask = ~train_mask
        if not np.any(train_mask) or not np.any(test_mask):
            continue
        train_data = EventData(
            X=data.X[train_mask],
            y=data.y[train_mask],
            cov=data.cov[train_mask],
            run_ids=data.run_ids[train_mask],
            rows_used=int(np.sum(train_mask)),
            target_id=data.target_id,
        )
        agg = _aggregate_events(
            train_data,
            resolution_m=float(args.aggregate_resolution_m),
            max_bin_weight=float(args.max_bin_weight),
        )
        y_test = data.y[test_mask]
        X_test = data.X[test_mask]
        constant = np.full_like(y_test, float(np.mean(train_data.y)), dtype=float)
        fold_rows.append(
            {
                'mode': 'constant_train_mean',
                'fold': str(fold),
                'train_events': str(int(np.sum(train_mask))),
                'train_aggregate_points': '1',
                'fit_points': '1',
                'heldout_events': str(int(np.sum(test_mask))),
                'brier': f'{float(np.mean((constant - y_test) ** 2)):.10g}',
                'logloss': f'{_logloss(y_test, constant, eps=float(args.min_prob)):.10g}',
                'auroc': f'{_auc(y_test, constant):.10g}',
                'prediction_mean': f'{float(np.mean(constant)):.10g}',
                'target_mean': f'{float(np.mean(y_test)):.10g}',
            }
        )
        if prior is not None:
            prior_pred = _clip_prob(_prior_prob(prior, X_test), float(args.min_prob))
            fold_rows.append(
                {
                    'mode': 'prior_only',
                    'fold': str(fold),
                    'train_events': '0',
                    'train_aggregate_points': '0',
                    'fit_points': '0',
                    'heldout_events': str(int(np.sum(test_mask))),
                    'brier': f'{float(np.mean((prior_pred - y_test) ** 2)):.10g}',
                    'logloss': f'{_logloss(y_test, prior_pred, eps=float(args.min_prob)):.10g}',
                    'auroc': f'{_auc(y_test, prior_pred):.10g}',
                    'prediction_mean': f'{float(np.mean(prior_pred)):.10g}',
                    'target_mean': f'{float(np.mean(y_test)):.10g}',
                }
            )
        for mode in modes:
            X_fit, y_fit, alpha, _ = _fit_inputs(
                agg,
                mode=mode,
                noise_var=float(args.gp_noise_var),
                gp_length_scale=float(args.gp_length_scale),
                pose_length_scale=float(args.pose_length_scale),
                min_certainty=float(args.min_certainty),
                spread_scale=float(args.spread_scale),
            )
            if mode == 'expected_kernel':
                train_latent = _logit(agg.y) - _prior_logit(prior, agg.X)
                resid_mu, _sigma_f, _jitter = _fit_predict_expected_kernel_latent(
                    agg.X,
                    train_latent,
                    agg.cov,
                    alpha,
                    X_test,
                    query_cov=data.cov[test_mask],
                    length_scale=float(args.gp_length_scale),
                )
                mu_f = _prior_logit(prior, X_test) + resid_mu
            else:
                latent_fit = _logit(y_fit) - _prior_logit(prior, X_fit)
                resid_mu, _sigma_f = _fit_predict_latent_gp(
                    X_fit,
                    latent_fit,
                    alpha,
                    X_test,
                    length_scale=float(args.gp_length_scale),
                )
                mu_f = _prior_logit(prior, X_test) + resid_mu
            p_pred = _clip_prob(_sigmoid(mu_f), float(args.min_prob))
            fold_rows.append(
                {
                    'mode': mode,
                    'fold': str(fold),
                    'train_events': str(int(np.sum(train_mask))),
                    'train_aggregate_points': str(int(agg.X.shape[0])),
                    'fit_points': str(int(X_fit.shape[0])),
                    'heldout_events': str(int(np.sum(test_mask))),
                    'brier': f'{float(np.mean((p_pred - y_test) ** 2)):.10g}',
                    'logloss': f'{_logloss(y_test, p_pred, eps=float(args.min_prob)):.10g}',
                    'auroc': f'{_auc(y_test, p_pred):.10g}',
                    'prediction_mean': f'{float(np.mean(p_pred)):.10g}',
                    'target_mean': f'{float(np.mean(y_test)):.10g}',
                }
            )

    summary_rows: list[dict[str, str]] = []
    summary_modes = ['constant_train_mean']
    if prior is not None:
        summary_modes.append('prior_only')
    summary_modes.extend(modes)
    for mode in summary_modes:
        mode_rows = [row for row in fold_rows if row['mode'] == mode]
        if not mode_rows:
            continue
        brier = np.asarray([float(row['brier']) for row in mode_rows], dtype=float)
        logloss = np.asarray([float(row['logloss']) for row in mode_rows], dtype=float)
        auroc = np.asarray([float(row['auroc']) for row in mode_rows if row['auroc'] not in ('', 'nan')], dtype=float)
        heldout = np.asarray([float(row['heldout_events']) for row in mode_rows], dtype=float)
        summary_rows.append(
            {
                'mode': mode,
                'folds': str(len(mode_rows)),
                'heldout_events': str(int(np.sum(heldout))),
                'brier_mean': f'{float(np.mean(brier)):.10g}',
                'brier_std': f'{float(np.std(brier)):.10g}',
                'logloss_mean': f'{float(np.mean(logloss)):.10g}',
                'logloss_std': f'{float(np.std(logloss)):.10g}',
                'auroc_mean': '' if auroc.size == 0 else f'{float(np.mean(auroc)):.10g}',
                'auroc_std': '' if auroc.size == 0 else f'{float(np.std(auroc)):.10g}',
            }
        )
    return fold_rows, summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description='Fit naive and belief-aware GP reliability artifacts from detector events.')
    parser.add_argument('--events', default=str(DEFAULT_EVENTS))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--grid-from', default=str(DEFAULT_GRID_GP))
    parser.add_argument('--prior-gp', default='', help='Optional planner-compatible GP artifact used as a prior mean function.')
    parser.add_argument('--prior-map-key', default='P_mean_map', choices=('P_mean_map', 'P_conservative_plan_map'))
    parser.add_argument('--target', choices=('hit', 'score'), default='hit')
    parser.add_argument('--modes', default=','.join(MODES))
    parser.add_argument('--aggregate-resolution-m', type=float, default=0.20)
    parser.add_argument('--max-bin-weight', type=float, default=20.0)
    parser.add_argument('--grid-nx', type=int, default=160)
    parser.add_argument('--grid-ny', type=int, default=140)
    parser.add_argument('--gp-length-scale', type=float, default=0.90)
    parser.add_argument('--gp-noise-var', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--min-prob', type=float, default=1e-4)
    parser.add_argument('--pose-length-scale', type=float, default=0.35)
    parser.add_argument('--min-certainty', type=float, default=0.05)
    parser.add_argument('--spread-scale', type=float, default=1.0)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument(
        '--holdout-run-id',
        action='append',
        default=[],
        help='Repeatable complete run ID reserved for route-disjoint evaluation; never used for fitting.',
    )
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()

    events_path = Path(args.events).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    logs_root = LOGS_ROOT.resolve()
    if logs_root not in output_dir.parents and output_dir != logs_root:
        raise RuntimeError(f'Output directory must stay under {logs_root}: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = tuple(mode.strip() for mode in str(args.modes).split(',') if mode.strip())
    invalid_modes = sorted(set(modes).difference(MODES))
    if invalid_modes:
        raise RuntimeError(f'Unsupported mode(s): {invalid_modes}. Available: {MODES}')

    source_data = _load_events(events_path, target=str(args.target), min_prob=float(args.min_prob))
    heldout_run_ids = tuple(sorted({str(item).strip() for item in args.holdout_run_id if str(item).strip()}))
    missing_holdouts = sorted(set(heldout_run_ids).difference({str(item) for item in source_data.run_ids}))
    if missing_holdouts:
        raise RuntimeError(f'--holdout-run-id values absent from events: {missing_holdouts}')
    if heldout_run_ids:
        heldout_mask = np.asarray([str(run_id) in heldout_run_ids for run_id in source_data.run_ids], dtype=bool)
        if not np.any(heldout_mask) or np.all(heldout_mask):
            raise RuntimeError('--holdout-run-id must reserve at least one run and leave at least one train run')
        data = _subset_events(source_data, ~heldout_mask)
        heldout_data = _subset_events(source_data, heldout_mask)
    else:
        data = source_data
        heldout_data = None
    prior = _load_prior_map(args.prior_gp, map_key=str(args.prior_map_key), min_prob=float(args.min_prob))
    agg = _aggregate_events(
        data,
        resolution_m=float(args.aggregate_resolution_m),
        max_bin_weight=float(args.max_bin_weight),
    )
    if agg.X.shape[0] < 4:
        raise RuntimeError(f'Need at least 4 aggregate points, got {agg.X.shape[0]}')

    xs, ys, metadata = _load_grid(
        Path(args.grid_from).expanduser().resolve(),
        fallback_X=data.X,
        grid_nx=int(args.grid_nx),
        grid_ny=int(args.grid_ny),
    )
    events_sha256 = sha256_file(events_path)

    fit_rows: list[dict[str, str]] = []
    for mode in modes:
        artifact, summary = _artifact_for_mode(
            agg,
            mode=mode,
            xs=xs,
            ys=ys,
            metadata=metadata,
            prior=prior,
            target_id=data.target_id,
            events_sha256=events_sha256,
            args=args,
        )
        artifact_path = output_dir / f'{data.target_id}_{mode}_gp.npz'
        np.savez_compressed(artifact_path, **artifact)
        fit_rows.append(
            {
                'mode': mode,
                'target_id': data.target_id,
                'event_count': str(int(data.rows_used)),
                'aggregate_points': str(int(summary['aggregate_points'])),
                'fit_points': str(int(summary['fit_points'])),
                'target_mean': f'{summary["target_mean"]:.10g}',
                'alpha_mean': f'{summary["alpha_mean"]:.10g}',
                'mean_certainty': f'{summary["mean_certainty"]:.10g}',
                'prior_map_mean': '' if math.isnan(float(summary['prior_map_mean'])) else f'{summary["prior_map_mean"]:.10g}',
                'expected_kernel_jitter': f'{summary["expected_kernel_jitter"]:.10g}',
                'p_mean_map_mean': f'{summary["p_mean_map_mean"]:.10g}',
                'p_conservative_plan_map_mean': f'{summary["p_conservative_plan_map_mean"]:.10g}',
                'artifact_path': repo_relative(artifact_path, output_dir),
            }
        )

    write_csv(
        output_dir / 'fit_summary.csv',
        (
            'mode',
            'target_id',
            'event_count',
            'aggregate_points',
            'fit_points',
            'target_mean',
            'alpha_mean',
            'mean_certainty',
            'prior_map_mean',
            'expected_kernel_jitter',
            'p_mean_map_mean',
            'p_conservative_plan_map_mean',
            'artifact_path',
        ),
        fit_rows,
    )

    validation_summary_rows: list[dict[str, str]] = []
    validation_fold_rows: list[dict[str, str]] = []
    if not bool(args.skip_validation):
        validation_fold_rows, validation_summary_rows = _validate_modes(data, modes=modes, prior=prior, args=args)
        write_csv(
            output_dir / 'validation_folds.csv',
            (
                'mode',
                'fold',
                'train_events',
                'train_aggregate_points',
                'fit_points',
                'heldout_events',
                'brier',
                'logloss',
                'auroc',
                'prediction_mean',
                'target_mean',
            ),
            validation_fold_rows,
        )
        write_csv(
            output_dir / 'validation_summary.csv',
            (
                'mode',
                'folds',
                'heldout_events',
                'brier_mean',
                'brier_std',
                'logloss_mean',
                'logloss_std',
                'auroc_mean',
                'auroc_std',
            ),
            validation_summary_rows,
        )

    route_disjoint_rows: list[dict[str, str]] = []
    if heldout_data is not None:
        route_disjoint_rows = _route_disjoint_validation(
            data,
            heldout_data,
            modes=modes,
            prior=prior,
            args=args,
        )
        write_csv(
            output_dir / 'route_disjoint_validation.csv',
            (
                'mode',
                'train_runs',
                'heldout_runs',
                'train_events',
                'heldout_events',
                'brier',
                'logloss',
                'auroc',
                'ece',
                'false_high_trust_rate',
                'prediction_mean',
                'target_mean',
            ),
            route_disjoint_rows,
        )

    write_manifest(
        output_dir / 'manifest.json',
        {
            'events_csv': str(events_path),
            'events_sha256': events_sha256,
            'source_event_count': int(source_data.rows_used),
            'train_event_count': int(data.rows_used),
            'heldout_event_count': 0 if heldout_data is None else int(heldout_data.rows_used),
            'heldout_run_ids': list(heldout_run_ids),
            'target': str(args.target),
            'target_id': data.target_id,
            'state_source': 'BELIEF',
            'modes': list(modes),
            'prior_gp': '' if prior is None else str(prior.path),
            'prior_gp_sha256': '' if prior is None else prior.sha256,
            'prior_map_key': '' if prior is None else prior.map_key,
            'event_count': int(data.rows_used),
            'aggregate_points': int(agg.X.shape[0]),
            'grid_from': str(Path(args.grid_from).expanduser().resolve()),
            'grid': {
                'nx': int(xs.size),
                'ny': int(ys.size),
                'xmin': float(xs[0]),
                'xmax': float(xs[-1]),
                'ymin': float(ys[0]),
                'ymax': float(ys[-1]),
            },
            'hyperparameters': {
                'aggregate_resolution_m': float(args.aggregate_resolution_m),
                'max_bin_weight': float(args.max_bin_weight),
                'gp_length_scale': float(args.gp_length_scale),
                'gp_noise_var': float(args.gp_noise_var),
                'beta': float(args.beta),
                'min_prob': float(args.min_prob),
                'pose_length_scale': float(args.pose_length_scale),
                'min_certainty': float(args.min_certainty),
                'spread_scale': float(args.spread_scale),
                'folds': int(args.folds),
            },
            'fit_summary_csv': str(output_dir / 'fit_summary.csv'),
            'validation_summary_csv': '' if bool(args.skip_validation) else str(output_dir / 'validation_summary.csv'),
            'route_disjoint_validation_csv': '' if heldout_data is None else str(output_dir / 'route_disjoint_validation.csv'),
            'notes': [
                'naive uses the belief mean as a deterministic GP input.',
                'uncertainty_weighted inflates per-bin observation noise when the belief covariance trace is large.',
                'belief_spread replaces each aggregate observation with five covariance sigma points while conserving total observation precision.',
                'expected_kernel is the paper-style uncertain-input GP: the RBF kernel is integrated under N(m,S) pose beliefs.',
                'When --prior-gp is supplied, every mode fits trajectory residuals in logit space on top of that prior mean map.',
                'When --holdout-run-id is supplied, complete held-out runs are excluded from fitting and are scored with canonical shared metrics.',
                'Ground truth from the event table is not used for fitting; it is retained only for evaluation/audit.',
            ],
        },
    )

    print(f'Wrote belief-aware GP artifacts to {output_dir}')
    print(f'Events used: {data.rows_used}')
    print(f'Aggregate points: {agg.X.shape[0]}')
    print(f'Modes: {", ".join(modes)}')
    if prior is not None:
        print(f'Prior GP: {prior.path}')
    if validation_summary_rows:
        best = min(validation_summary_rows, key=lambda row: float(row['brier_mean']))
        print(f'Best validation Brier: {best["mode"]} = {best["brier_mean"]}')
    if route_disjoint_rows:
        best = min(route_disjoint_rows, key=lambda row: float(row['brier']))
        print(f'Best route-disjoint Brier: {best["mode"]} = {best["brier"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
