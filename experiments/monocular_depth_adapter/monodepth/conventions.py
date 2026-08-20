"""Conversions between depth conventions that need nothing but the intrinsics.

Only one conversion is free: optical-axis depth <-> range along the ray. It is
pure pinhole geometry, so it introduces no assumption about the scene.

Turning relative or inverse depth into metres is NOT here and must not be added
here. That needs an anchor — a known distance somewhere in the image — and
choosing an anchor is a scene decision this adapter is not allowed to make.
Whoever owns the anchor (floor plane, a measured object, a depth sensor) owns
that conversion, downstream.
"""

from __future__ import annotations

import numpy as np

from .types import CameraIntrinsics, DepthConvention


def ray_secant_field(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Per-pixel ``range / z`` factor: ``sqrt(1 + x_n^2 + y_n^2)``.

    1.0 at the principal point and growing towards the corners. For a 90-degree
    horizontal field of view this reaches about 1.5 in the image corners, so
    mixing the two conventions is a tens-of-percent error, not a rounding one.
    """
    u = np.arange(intrinsics.width, dtype=np.float64)
    v = np.arange(intrinsics.height, dtype=np.float64)
    xn = (u - intrinsics.cx) / intrinsics.fx
    yn = (v - intrinsics.cy) / intrinsics.fy
    return np.sqrt(1.0 + xn[None, :] ** 2 + yn[:, None] ** 2)


def z_to_euclidean(depth_z: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Optical-axis depth (metres) -> range along the ray (metres)."""
    return (np.asarray(depth_z, dtype=np.float64) * ray_secant_field(intrinsics)).astype(np.float32)


def euclidean_to_z(depth_range: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Range along the ray (metres) -> optical-axis depth (metres)."""
    return (np.asarray(depth_range, dtype=np.float64) / ray_secant_field(intrinsics)).astype(np.float32)


def convert(depth: np.ndarray, source: DepthConvention, target: DepthConvention,
            intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert between the two metric conventions; refuse everything else.

    Refusing loudly is the point: a silent relative->metric conversion is how a
    unitless number ends up being treated as metres three modules later.
    """
    if source is target:
        return np.asarray(depth, dtype=np.float32)
    if source is DepthConvention.METRIC_Z and target is DepthConvention.EUCLIDEAN_RANGE:
        return z_to_euclidean(depth, intrinsics)
    if source is DepthConvention.EUCLIDEAN_RANGE and target is DepthConvention.METRIC_Z:
        return euclidean_to_z(depth, intrinsics)
    raise ValueError(
        f"cannot convert {source.value} -> {target.value} from intrinsics alone. "
        "Non-metric conventions need a scene anchor, which the depth adapter is "
        "not permitted to choose."
    )


def align_affine(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                 *, fit_shift: bool = True) -> tuple[float, float]:
    """Least-squares ``a, b`` with ``a * prediction + b ~ reference`` over ``mask``.

    Needed to compare two runs of a *non-metric* model against each other: a
    relative-depth network is free to return a different scale for a flipped
    copy of the same image, and an unaligned difference would report that free
    parameter as uncertainty. Metric models are compared directly, unaligned.
    """
    p = np.asarray(prediction, dtype=np.float64)[mask]
    r = np.asarray(reference, dtype=np.float64)[mask]
    if p.size < 16:
        return 1.0, 0.0
    if fit_shift:
        A = np.stack([p, np.ones_like(p)], axis=1)
        sol, *_ = np.linalg.lstsq(A, r, rcond=None)
        return float(sol[0]), float(sol[1])
    denom = float(np.dot(p, p))
    if denom <= 0:
        return 1.0, 0.0
    return float(np.dot(p, r) / denom), 0.0


__all__ = ["ray_secant_field", "z_to_euclidean", "euclidean_to_z", "convert", "align_affine"]
