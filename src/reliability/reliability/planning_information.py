"""Fit per-camera expected-information fields from complete camera opportunities.

An admitted opportunity contributes the inverse of its matched runtime
covariance. A miss or gate refusal contributes the zero matrix. The spatial
estimate is shrunk towards zero when few survey opportunities support it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PlanningInformationField:
    camera_ids: tuple[str, ...]
    xs: np.ndarray
    ys: np.ndarray
    expected_information: np.ndarray
    opportunity_support: np.ndarray
    raw_expected_information: np.ndarray


def realized_information(
    admitted: Sequence[bool],
    runtime_covariance_m2: np.ndarray,
) -> np.ndarray:
    """Return ``inv(R_run)`` for admissions and zero for non-admissions."""
    admitted_array = np.asarray(admitted, dtype=bool)
    covariance = np.asarray(runtime_covariance_m2, dtype=float)
    if covariance.shape != (len(admitted_array), 2, 2):
        raise ValueError('runtime covariance must have shape [opportunity,2,2]')
    information = np.zeros_like(covariance)
    selected = covariance[admitted_array]
    if len(selected):
        if not np.isfinite(selected).all():
            raise ValueError('admitted runtime covariance must be finite')
        if not np.allclose(selected, selected.swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
            raise ValueError('admitted runtime covariance must be symmetric')
        if np.linalg.eigvalsh(selected).min() <= 0.0:
            raise ValueError('admitted runtime covariance must be positive definite')
        information[admitted_array] = np.linalg.inv(selected)
    return information


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.dot(weights, weights))
    if denominator <= 0.0:
        return 0.0
    return float(weights.sum() ** 2 / denominator)


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    """Remove only round-off-sized negative eigenvalues from a symmetric mean."""
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _heading_pooled_values(
    positions: np.ndarray,
    headings_rad: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average repetitions within heading, then declared headings within position.

    Heading is deliberately not a field dimension. Equal declared headings at a
    surveyed position receive equal weight even when one heading has more repeated
    captures. The returned rows are therefore one independent value per physical
    position. Support must be computed from these rows as well: repeated frames and
    headings are evidence about the outcome at a position, not additional spatial
    coverage.
    """
    wrapped = np.mod(headings_rad, 2.0 * np.pi)
    position_keys, position_inverse = np.unique(positions, axis=0, return_inverse=True)
    pooled = np.zeros((len(position_keys), 2, 2), dtype=float)
    for position_index in range(len(position_keys)):
        at_position = np.flatnonzero(position_inverse == position_index)
        heading_keys, heading_inverse = np.unique(wrapped[at_position], return_inverse=True)
        heading_means = np.stack([
            values[at_position[heading_inverse == heading_index]].mean(axis=0)
            for heading_index in range(len(heading_keys))
        ])
        pooled[position_index] = _project_psd(heading_means.mean(axis=0))
    return position_keys, pooled


def fit_constant_planning_information(
    positions_xy_m: np.ndarray,
    opportunity_camera_ids: Sequence[str],
    admitted: Sequence[bool],
    runtime_covariance_m2: np.ndarray,
    *,
    camera_ids: Sequence[str],
    headings_rad: Sequence[float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit position-balanced global and per-camera expected-information constants."""
    positions = np.asarray(positions_xy_m, dtype=float)
    cameras = np.asarray(opportunity_camera_ids).astype(str)
    headings = np.asarray(headings_rad, dtype=float)
    outcomes = realized_information(admitted, runtime_covariance_m2)
    declared = tuple(str(value) for value in camera_ids)
    if (positions.shape != (len(cameras), 2) or headings.shape != (len(cameras),)
            or not np.isfinite(positions).all() or not np.isfinite(headings).all()):
        raise ValueError('constant-information inputs have incompatible shapes')
    per_camera = {}
    pooled_camera_values = []
    for camera_id in declared:
        use = cameras == camera_id
        if not np.any(use):
            raise ValueError(f'no opportunities for {camera_id}')
        _points, values = _heading_pooled_values(
            positions[use], headings[use], outcomes[use])
        per_camera[camera_id] = _project_psd(values.mean(axis=0))
        pooled_camera_values.extend(values)
    global_value = _project_psd(np.stack(pooled_camera_values).mean(axis=0))
    return global_value, per_camera


def fit_planning_information_field(
    positions_xy_m: np.ndarray,
    opportunity_camera_ids: Sequence[str],
    admitted: Sequence[bool],
    runtime_covariance_m2: np.ndarray,
    *,
    camera_ids: Sequence[str],
    xs: Sequence[float],
    ys: Sequence[float],
    length_scale_m: float,
    support_radius_m: float,
    support_tau: float,
    headings_rad: Sequence[float] | None = None,
) -> PlanningInformationField:
    """Kernel-fit ``E[a R_run^-1 | p]`` and shrink unsupported cells to zero."""
    positions = np.asarray(positions_xy_m, dtype=float)
    opportunity_cameras = np.asarray(opportunity_camera_ids).astype(str)
    admitted_array = np.asarray(admitted, dtype=bool)
    headings = None if headings_rad is None else np.asarray(headings_rad, dtype=float)
    cameras = tuple(str(value) for value in camera_ids)
    x_axis = np.asarray(xs, dtype=float)
    y_axis = np.asarray(ys, dtype=float)
    if positions.shape != (len(admitted_array), 2) or not np.isfinite(positions).all():
        raise ValueError('positions must be finite [opportunity,2]')
    if opportunity_cameras.shape != (len(admitted_array),):
        raise ValueError('one camera ID is required per opportunity')
    if headings is not None and (headings.shape != (len(admitted_array),)
                                 or not np.isfinite(headings).all()):
        raise ValueError('headings must be finite with one value per opportunity')
    if not cameras or len(cameras) != len(set(cameras)):
        raise ValueError('camera IDs must be nonempty and unique')
    if not set(opportunity_cameras).issubset(cameras):
        raise ValueError('opportunity contains a camera outside the declared roster')
    for axis, name in ((x_axis, 'xs'), (y_axis, 'ys')):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError(f'{name} must be a finite one-dimensional grid')
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError(f'{name} must be strictly increasing')
    length_scale = float(length_scale_m)
    radius = float(support_radius_m)
    tau = float(support_tau)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError('length scale must be positive')
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError('support radius must be positive')
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError('support tau must be nonnegative')

    outcomes = realized_information(admitted_array, runtime_covariance_m2)
    shape = (len(cameras), len(y_axis), len(x_axis))
    raw = np.zeros(shape + (2, 2), dtype=float)
    deployed = np.zeros_like(raw)
    support = np.zeros(shape, dtype=float)
    grid = np.stack(np.meshgrid(x_axis, y_axis, indexing='xy'), axis=-1)
    for camera_index, camera_id in enumerate(cameras):
        use = opportunity_cameras == camera_id
        points = positions[use]
        values = outcomes[use]
        if not len(points):
            continue
        if headings is None:
            value_points, pooled_values = points, values
        else:
            value_points, pooled_values = _heading_pooled_values(
                points, headings[use], values)
        for y_index in range(len(y_axis)):
            for x_index in range(len(x_axis)):
                # Spatial support is counted by independent physical position, not
                # by the number of frames, repetitions or headings captured there.
                squared_distance = np.sum(
                    (value_points - grid[y_index, x_index]) ** 2, axis=1)
                support_weights = np.exp(-0.5 * squared_distance / length_scale ** 2)
                support_weights[squared_distance > radius ** 2] = 0.0
                if float(support_weights.sum()) <= 0.0:
                    continue
                value_squared_distance = np.sum(
                    (value_points - grid[y_index, x_index]) ** 2, axis=1)
                value_weights = np.exp(
                    -0.5 * value_squared_distance / length_scale ** 2)
                value_weights[value_squared_distance > radius ** 2] = 0.0
                estimate = np.einsum(
                    'n,nij->ij', value_weights, pooled_values) / value_weights.sum()
                estimate = _project_psd(estimate)
                n_eff = effective_sample_size(support_weights)
                alpha = 1.0 if tau == 0.0 else n_eff / (n_eff + tau)
                raw[camera_index, y_index, x_index] = estimate
                deployed[camera_index, y_index, x_index] = alpha * estimate
                support[camera_index, y_index, x_index] = n_eff

    return PlanningInformationField(
        camera_ids=cameras,
        xs=x_axis,
        ys=y_axis,
        expected_information=deployed,
        opportunity_support=support,
        raw_expected_information=raw,
    )
