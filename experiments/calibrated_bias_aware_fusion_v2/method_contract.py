#!/usr/bin/env python3
"""ROS-independent dimensional and observability checks for the proposed method.

These utilities are not an estimator and produce no performance result.  They
make the most failure-prone mathematical contracts executable before ROS code is
allowed into the campaign: ordered 4-D keypoints, full covariance propagation,
valid Split-CI decomposition, and gated physical extrinsic directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


KEYPOINT_ORDER = ("front_u_px", "front_v_px", "rear_u_px", "rear_v_px")
DRIFT_ORDER = (
    "delta_tx_m",
    "delta_ty_m",
    "delta_tz_m",
    "delta_roll_rad",
    "delta_pitch_rad",
    "delta_yaw_rad",
)
BASELINE_IDS = (
    "B1_best_single",
    "B2_scalar_R",
    "B3_calibrated_R_independent",
    "B4_static_offset",
    "B5_NIS_gating",
    "B6a_covariance_intersection",
    "B6b_split_covariance_intersection",
    "B7_proposed_observability_gated_drift",
    "B8_oracle_exclusion",
)


class MethodContractError(ValueError):
    """A matrix, state order, or baseline violates the frozen method contract."""


def _symmetric_matrix(
    value: np.ndarray | Sequence[Sequence[float]],
    shape: tuple[int, int],
    label: str,
    symmetry_tolerance: float,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape:
        raise MethodContractError(f"{label} shape must be {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise MethodContractError(f"{label} contains a non-finite value")
    if not np.allclose(matrix, matrix.T, atol=symmetry_tolerance, rtol=0.0):
        raise MethodContractError(f"{label} is not symmetric")
    return 0.5 * (matrix + matrix.T)


def validate_keypoint_order(order: Sequence[str]) -> None:
    if tuple(order) != KEYPOINT_ORDER:
        raise MethodContractError(
            f"keypoint vector order must be {KEYPOINT_ORDER}, got {tuple(order)}"
        )


def validate_keypoint_covariance(
    covariance: np.ndarray | Sequence[Sequence[float]],
    *,
    eigenvalue_floor: float = 1.0e-9,
    symmetry_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return a symmetric 4x4 pixel covariance or fail closed.

    A final runtime R must be positive definite, rather than merely semidefinite,
    because Gaussian NLL and whitening require a stable inverse.  The caller may
    regularize a training covariance, but the regularizer is part of the frozen
    model and cannot be applied silently here.
    """
    matrix = _symmetric_matrix(
        covariance, (4, 4), "keypoint covariance", symmetry_tolerance
    )
    smallest = float(np.linalg.eigvalsh(matrix)[0])
    if smallest < eigenvalue_floor:
        raise MethodContractError(
            f"keypoint covariance min eigenvalue {smallest:.3e} is below "
            f"the frozen floor {eigenvalue_floor:.3e}"
        )
    return matrix


def propagate_keypoint_covariance(
    covariance_uv: np.ndarray | Sequence[Sequence[float]],
    jacobian_xy_uv: np.ndarray | Sequence[Sequence[float]],
    *,
    eigenvalue_floor: float = 1.0e-9,
) -> np.ndarray:
    """Propagate the ordered full 4-D R into world XY using ``J R J^T``."""
    r_uv = validate_keypoint_covariance(
        covariance_uv, eigenvalue_floor=eigenvalue_floor
    )
    jacobian = np.asarray(jacobian_xy_uv, dtype=float)
    if jacobian.shape != (2, 4):
        raise MethodContractError(
            f"projection Jacobian shape must be (2, 4), got {jacobian.shape}"
        )
    if not np.all(np.isfinite(jacobian)):
        raise MethodContractError("projection Jacobian contains a non-finite value")
    propagated = jacobian @ r_uv @ jacobian.T
    propagated = 0.5 * (propagated + propagated.T)
    minimum = float(np.linalg.eigvalsh(propagated)[0])
    if minimum < -1.0e-10:
        raise MethodContractError(
            f"propagated world covariance is not PSD (min eigenvalue {minimum:.3e})"
        )
    if np.linalg.matrix_rank(jacobian) < 2:
        raise MethodContractError("projection Jacobian is rank deficient in world XY")
    return propagated


def validate_split_ci_decomposition(
    total_covariance: np.ndarray | Sequence[Sequence[float]],
    shared_covariance: np.ndarray | Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``P_total = P_independent + P_shared`` for Split CI.

    The returned pair is ``(P_independent, P_shared)``.  Both components must be
    positive semidefinite; labelling an arbitrary part of P as shared is therefore
    caught before a Split-CI baseline is run.
    """
    total = np.asarray(total_covariance, dtype=float)
    shared = np.asarray(shared_covariance, dtype=float)
    if total.ndim != 2 or total.shape[0] != total.shape[1]:
        raise MethodContractError("total covariance must be square")
    shared = _symmetric_matrix(
        shared, total.shape, "Split-CI shared covariance", tolerance
    )
    total = _symmetric_matrix(total, total.shape, "Split-CI total covariance", tolerance)
    independent = 0.5 * ((total - shared) + (total - shared).T)
    for label, component in (("total", total), ("shared", shared), ("independent", independent)):
        minimum = float(np.linalg.eigvalsh(component)[0])
        if minimum < -tolerance:
            raise MethodContractError(
                f"Split-CI {label} component is not PSD (min eigenvalue {minimum:.3e})"
            )
    return independent, shared


def whiten_jacobian(
    jacobian: np.ndarray | Sequence[Sequence[float]],
    residual_covariance: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Left-whiten a residual Jacobian using a positive-definite covariance."""
    h = np.asarray(jacobian, dtype=float)
    covariance = np.asarray(residual_covariance, dtype=float)
    if h.ndim != 2 or h.shape[1] != len(DRIFT_ORDER):
        raise MethodContractError(
            f"drift Jacobian must have six ordered columns {DRIFT_ORDER}; got {h.shape}"
        )
    covariance = _symmetric_matrix(
        covariance, (h.shape[0], h.shape[0]), "residual covariance", 1.0e-10
    )
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise MethodContractError("residual covariance must be positive definite") from exc
    return np.linalg.solve(chol, h)


@dataclass(frozen=True)
class ObservabilityResult:
    singular_values: np.ndarray
    threshold: float
    rank: int
    observable_projector: np.ndarray
    nullspace_projector: np.ndarray


def observability_projectors(
    whitened_drift_jacobian: np.ndarray | Sequence[Sequence[float]],
    *,
    absolute_singular_value_min: float = 1.0e-3,
    relative_to_max_min: float = 1.0e-2,
) -> ObservabilityResult:
    """Compute frozen SVD gates and projectors for the six physical drift modes.

    The input is the *already whitened and robot-pose-marginalized* stacked
    Jacobian from an overlap window.  Passing an unmarginalized matrix would test
    a different problem and must be prevented by the ROS integration layer.
    """
    matrix = np.asarray(whitened_drift_jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(DRIFT_ORDER):
        raise MethodContractError(
            f"whitened drift Jacobian must have shape (n, 6), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise MethodContractError("whitened drift Jacobian contains non-finite values")
    if absolute_singular_value_min < 0 or not 0 <= relative_to_max_min <= 1:
        raise MethodContractError("invalid observability thresholds")
    if matrix.shape[0] == 0:
        singular_values = np.empty(0, dtype=float)
        vectors_t = np.eye(len(DRIFT_ORDER), dtype=float)
    else:
        _, singular_values, vectors_t = np.linalg.svd(matrix, full_matrices=True)
    maximum = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(absolute_singular_value_min, relative_to_max_min * maximum)
    passed = singular_values >= threshold
    rank = int(np.count_nonzero(passed))
    passed_vectors = vectors_t[: singular_values.size][passed]
    if passed_vectors.size:
        observable = passed_vectors.T @ passed_vectors
    else:
        observable = np.zeros((len(DRIFT_ORDER), len(DRIFT_ORDER)), dtype=float)
    observable = 0.5 * (observable + observable.T)
    nullspace = np.eye(len(DRIFT_ORDER), dtype=float) - observable
    return ObservabilityResult(
        singular_values=singular_values,
        threshold=threshold,
        rank=rank,
        observable_projector=observable,
        nullspace_projector=nullspace,
    )


def validate_observability_window(
    camera_ids: Iterable[str],
    associated_instants: Iterable[object],
    *,
    minimum_distinct_cameras: int = 2,
    minimum_associated_instants: int = 8,
) -> None:
    cameras = tuple(camera_ids)
    instants = tuple(associated_instants)
    unknown = sorted(set(cameras) - set("ABCDE"))
    if unknown:
        raise MethodContractError(f"unknown camera IDs in overlap window: {unknown}")
    if len(set(cameras)) < minimum_distinct_cameras:
        raise MethodContractError("overlap window lacks distinct cameras")
    if len(set(instants)) < minimum_associated_instants:
        raise MethodContractError("overlap window has too few associated instants")


def validate_baseline_ids(ids: Iterable[str], *, runtime: bool) -> tuple[str, ...]:
    ids_tuple = tuple(ids)
    if len(ids_tuple) != len(set(ids_tuple)):
        raise MethodContractError("duplicate baseline ID")
    unknown = sorted(set(ids_tuple) - set(BASELINE_IDS))
    if unknown:
        raise MethodContractError(f"unknown baseline IDs: {unknown}")
    if runtime and "B8_oracle_exclusion" in ids_tuple:
        raise MethodContractError("oracle exclusion is offline diagnostic only")
    return ids_tuple
