"""Choosing the pixels whose true depth is already known.

The whole method rests on one deployment-legal observation: for a fixed camera
with known calibration, a pixel whose ray lands on the warehouse floor has an
*analytically known* depth. No sensor, no ground truth, no learning -- just the
plane and the extrinsics. Those pixels are the ruler that turns a monocular
prediction into metres.

Selection is deliberately conservative and geometric:

1. the ray must descend to the floor plane in front of the camera, at a sane
   range;
2. the world point it lands on must lie inside the declared drivable floor
   (a deployment map the planner already has);
3. an optional image-space floor segmentation, if the caller has one, is ANDed
   in;
4. the model must have marked the pixel valid.

Anchors chosen this way are *mostly* floor. The minority that are not -- a box
standing in the aisle occludes floor that the map says is drivable -- are left
for the robust fit to reject, and they are exactly the obstacle we are trying
to detect, so they must never be trusted here.

Floor segmentation lives in this module on purpose: picking anchors *is* the
segmentation problem for this method, and it is not a separate contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, sobel

from .contracts import (
    AnchorConfig,
    CameraCalibration,
    ContractViolation,
    FloorPlane,
    covered_by_any,
)


@dataclass(frozen=True)
class FloorAnchors:
    """Pixels with an analytically known optical-axis depth."""

    u: np.ndarray
    v: np.ndarray
    #: analytic optical-axis depth to the floor plane, metres
    depth_m: np.ndarray
    #: where each ray meets the plane, world frame ``(N, 3)``
    world_xyz: np.ndarray
    #: how many sub-sampled pixels were considered before filtering
    n_candidates: int
    #: counts surviving each successive filter, for diagnosing an empty set
    stage_counts: dict
    #: Relative fit weights, normalised to mean one over retained anchors.
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.u.size)

    @property
    def depth_span_m(self) -> float:
        """Robust spread of anchor depths (p95 - p5).

        This is the identifiability resource for the scale: anchors bunched at
        one range determine the shift but barely constrain the slope, in the
        same way that two cameras on nearly the same bearing cannot separate
        their offsets. A frame that fails the span gate is refused, not fitted.
        """
        if self.depth_m.size < 2:
            return 0.0
        lo, hi = np.percentile(self.depth_m, [5.0, 95.0])
        return float(hi - lo)


def analytic_plane_depth(
    calib: CameraCalibration,
    plane: FloorPlane,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Optical-axis depth at which each pixel ray meets the plane.

    Returns ``(depth, world_xyz)``; depth is NaN for rays that run parallel to
    the plane or meet it behind the camera.
    """
    dirs = calib.rays_world(u, v)  # (3, N), per unit optical-axis depth
    denom = plane.normal @ dirs  # (N,)
    numer = plane.offset - plane.normal @ calib.cam_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(np.abs(denom) > 1e-12, numer / denom, np.nan)
    depth = np.where(depth > 0.0, depth, np.nan)
    world = calib.cam_pos[:, None] + dirs * depth[None, :]
    return depth, world.T


def select_floor_anchors(
    calib: CameraCalibration,
    plane: FloorPlane,
    drivable: Sequence[Any],
    *,
    config: AnchorConfig | None = None,
    valid_mask: np.ndarray | None = None,
    floor_segmentation: np.ndarray | None = None,
    prediction_values: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    uncertainty_kind: str | None = None,
    native_confidence: np.ndarray | None = None,
) -> FloorAnchors:
    """Geometric floor-anchor selection. See the module docstring for the rules.

    ``drivable`` is a sequence of axis-aligned world footprints (anything with
    ``xmin/xmax/ymin/ymax``, so ``geometry_visibility.Prism`` works). Pass an
    empty sequence together with ``config.require_drivable=False`` to anchor on
    the whole visible floor plane.
    """
    cfg = config or AnchorConfig()
    if cfg.require_drivable and len(drivable) == 0:
        raise ContractViolation(
            "require_drivable=True but no drivable footprints were supplied; the method "
            "will not anchor on an undeclared floor"
        )

    u, v = calib.pixel_grid(cfg.pixel_step)
    n_candidates = int(u.size)
    stage: dict[str, int] = {"candidates": n_candidates}

    depth, world = analytic_plane_depth(calib, plane, u, v)
    keep = np.isfinite(depth)
    stage["hits_plane_in_front"] = int(keep.sum())

    keep &= (depth >= cfg.min_depth_m) & (depth <= cfg.max_depth_m)
    stage["within_range"] = int(keep.sum())

    if cfg.require_drivable:
        keep &= covered_by_any(world[:, 0], world[:, 1], drivable)
        stage["inside_drivable"] = int(keep.sum())

    if floor_segmentation is not None:
        seg = np.asarray(floor_segmentation, dtype=bool)
        if seg.shape != (calib.height, calib.width):
            raise ContractViolation(
                f"floor_segmentation shape {seg.shape} != image {(calib.height, calib.width)}"
            )
        if cfg.quality_filter and cfg.segmentation_erosion_px > 0:
            seg = binary_erosion(
                seg,
                structure=np.ones((3, 3), dtype=bool),
                iterations=int(cfg.segmentation_erosion_px),
                border_value=0,
            )
            stage["inside_eroded_segmentation"] = int(
                np.count_nonzero(keep & seg[v.astype(int), u.astype(int)])
            )
        keep &= seg[v.astype(int), u.astype(int)]
        stage["inside_segmentation"] = int(keep.sum())

    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != (calib.height, calib.width):
            raise ContractViolation(
                f"valid_mask shape {vm.shape} != image {(calib.height, calib.width)}"
            )
        keep &= vm[v.astype(int), u.astype(int)]
        stage["model_valid"] = int(keep.sum())

    weights = np.ones(u.shape, dtype=float)
    if cfg.quality_filter:
        image_shape = (calib.height, calib.width)
        ui, vi = u.astype(int), v.astype(int)

        if prediction_values is not None:
            values = np.asarray(prediction_values, dtype=float)
            if values.shape != image_shape:
                raise ContractViolation(
                    f"prediction_values shape {values.shape} != image {image_shape}"
                )
            finite_values = np.isfinite(values)
            robust_scale = (
                float(np.nanpercentile(np.abs(values[finite_values]), 75.0))
                if finite_values.any()
                else 1.0
            )
            # Sobel spreads one NaN over its whole stencil.  Fill invalid
            # pixels only for the derivative calculation; they are still
            # rejected explicitly below and dilated as edges.
            fill_value = (
                float(np.nanmedian(values[finite_values])) if finite_values.any() else 0.0
            )
            derivative_values = np.where(finite_values, values, fill_value)
            transformed = np.arcsinh(derivative_values / max(robust_scale, 1e-9))
            grad = np.hypot(sobel(transformed, axis=0), sobel(transformed, axis=1))
            finite_grad = grad[np.isfinite(grad) & finite_values]
            if finite_grad.size:
                edge_threshold = float(np.quantile(
                    finite_grad,
                    np.clip(cfg.depth_edge_quantile, 0.0, 1.0),
                ))
                # Strict comparison matters for a flat prediction: its 90th
                # percentile is zero, which is evidence of no edge rather
                # than evidence that every pixel is an edge.
                edge = (~finite_values) | (grad > edge_threshold)
                if cfg.depth_edge_dilation_px > 0:
                    edge = binary_dilation(
                        edge,
                        structure=np.ones((3, 3), dtype=bool),
                        iterations=int(cfg.depth_edge_dilation_px),
                    )
                keep &= ~edge[vi, ui]
                stage["away_from_depth_edges"] = int(keep.sum())
                edge_quality = 1.0 / (
                    1.0 + grad[vi, ui] / max(np.nanmedian(finite_grad), 1e-9)
                )
                weights *= np.clip(edge_quality, cfg.min_quality_weight, 1.0)

        confidence_maps = []
        if native_confidence is not None:
            confidence = np.asarray(native_confidence, dtype=float)
            if confidence.shape != image_shape:
                raise ContractViolation(
                    f"native_confidence shape {confidence.shape} != image {image_shape}"
                )
            confidence_maps.append(confidence)
        if uncertainty is not None and str(uncertainty_kind or "") == "native_confidence":
            confidence = np.asarray(uncertainty, dtype=float)
            if confidence.shape != image_shape:
                raise ContractViolation(
                    f"confidence uncertainty shape {confidence.shape} != image {image_shape}"
                )
            confidence_maps.append(confidence)
        if confidence_maps:
            confidence = np.nanmean(np.stack(confidence_maps), axis=0)
            sampled = confidence[vi, ui]
            eligible = sampled[keep & np.isfinite(sampled)]
            if eligible.size:
                keep_fraction = float(np.clip(cfg.confidence_keep_fraction, 0.0, 1.0))
                threshold = float(np.quantile(eligible, 1.0 - keep_fraction))
                keep &= np.isfinite(sampled) & (sampled >= threshold)
                stage["high_confidence"] = int(keep.sum())
                lo, hi = np.quantile(eligible, [0.05, 0.95])
                normalised = (sampled - lo) / max(float(hi - lo), 1e-9)
                weights *= np.clip(normalised, cfg.min_quality_weight, 1.0)

        if uncertainty is not None and str(uncertainty_kind or "") != "native_confidence":
            uncertainty_map = np.asarray(uncertainty, dtype=float)
            if uncertainty_map.shape != image_shape:
                raise ContractViolation(
                    f"uncertainty shape {uncertainty_map.shape} != image {image_shape}"
                )
            sampled = uncertainty_map[vi, ui]
            eligible = sampled[keep & np.isfinite(sampled) & (sampled >= 0.0)]
            if eligible.size:
                keep_fraction = float(np.clip(cfg.uncertainty_keep_fraction, 0.0, 1.0))
                threshold = float(np.quantile(eligible, keep_fraction))
                keep &= np.isfinite(sampled) & (sampled >= 0.0) & (sampled <= threshold)
                stage["low_uncertainty"] = int(keep.sum())
                lo, hi = np.quantile(eligible, [0.05, 0.95])
                normalised = 1.0 - (sampled - lo) / max(float(hi - lo), 1e-9)
                weights *= np.clip(normalised, cfg.min_quality_weight, 1.0)

    retained_weights = np.asarray(weights[keep], dtype=float)
    if retained_weights.size:
        retained_weights = np.clip(retained_weights, cfg.min_quality_weight, np.inf)
        retained_weights /= float(np.mean(retained_weights))

    return FloorAnchors(
        u=u[keep],
        v=v[keep],
        depth_m=depth[keep],
        world_xyz=world[keep],
        n_candidates=n_candidates,
        stage_counts=stage,
        weights=retained_weights,
    )
