"""End-to-end single-frame method: RGB-derived depth -> line-of-sight belief.

    floor anchors (calibration + drivable map)
        -> robust scale/shift fit in the model's own depth convention
        -> metric depth + per-pixel sigma
        -> back-projection -> 2.5-D height map + observed mask
        -> probabilistic raycast to the robot volume
        -> p_visible / p_occluded / p_unknown

The default remains single-frame operation, so the frozen studies keep their
original behaviour.  Online callers may explicitly supply a per-camera
temporal filter for the two affine ground-anchor parameters; depth pixels and
obstacle evidence are still never carried between frames.

Nothing in this package reads oracle depth, simulator obstacle poses, or an
oracle visibility grid. Those exist for scoring the output, and the method is
only meaningful if it never saw them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    CameraCalibration,
    ContractViolation,
    DepthPrediction,
    FloorPlane,
    FrameStatus,
    MethodConfig,
    MetricDepth,
    VisibilityField,
    VisibilityResult,
)
from .conventions import fit_space_for, to_optical_axis
from .floor_anchors import FloorAnchors, select_floor_anchors
from .ground_fit import fit_ground_affine, predicted_depth_sigma
from .heightmap import (
    HeightMap,
    back_project,
    ground_visibility_mask,
    rasterize_heights,
)
from .raycast import line_of_sight_field
from .temporal import TemporalGroundAnchorFilter

METHOD_VERSION = "ground_anchoring/1.1"


def config_fingerprint(config: MethodConfig) -> str:
    """Short stable hash of the configuration, for reproducibility records."""
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _all_unknown_field(
    xs: np.ndarray, ys: np.ndarray, calib: CameraCalibration, config: MethodConfig
) -> VisibilityField:
    shape = (np.asarray(ys).size, np.asarray(xs).size)
    zeros = np.zeros(shape, dtype=float)
    from .raycast import _in_fov  # local: only needed on the refusal path

    z_probe = 0.5 * (config.target.z_min_m + config.target.z_max_m)
    return VisibilityField(
        xs=np.asarray(xs, dtype=float),
        ys=np.asarray(ys, dtype=float),
        p_visible=zeros,
        p_occluded=zeros.copy(),
        p_unknown=np.ones(shape, dtype=float),
        unknown_mask=np.ones(shape, dtype=bool),
        in_fov=_in_fov(calib, np.asarray(xs, float), np.asarray(ys, float), z_probe),
        height_map_m=zeros.copy(),
        height_sigma_m=np.full(shape, np.nan),
        observed=np.zeros(shape, dtype=bool),
    )


def estimate_visibility(
    prediction: DepthPrediction,
    calib: CameraCalibration,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    plane: FloorPlane | None = None,
    drivable: Sequence[Any] = (),
    config: MethodConfig | None = None,
    floor_segmentation: np.ndarray | None = None,
    temporal_filter: TemporalGroundAnchorFilter | None = None,
    scenario_id: str = "",
    frame_id: str = "",
    timestamp: float = float("nan"),
    extra_provenance: Mapping[str, Any] | None = None,
) -> VisibilityResult:
    """Run the method on one frame. Returns the full output contract.

    Raises :class:`DepthConventionError` when the declared depth convention
    contradicts the data (default ``config.fit.strict_convention``), and
    :class:`ContractViolation` when the prediction does not match the camera.
    """
    cfg = config or MethodConfig()
    plane = plane or FloorPlane()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if prediction.shape != (calib.height, calib.width):
        raise ContractViolation(
            f"prediction shape {prediction.shape} does not match camera "
            f"{(calib.height, calib.width)} for {calib.camera_id!r}; the adapter must not "
            "resize without also resizing the intrinsics"
        )

    provenance: dict[str, Any] = {
        "method_version": METHOD_VERSION,
        "model_name": prediction.model_name,
        "checkpoint": prediction.checkpoint,
        "depth_convention": prediction.convention.value,
        "fit_space": fit_space_for(prediction.convention),
        "inference_time_s": prediction.inference_time_s,
        "config_fingerprint": config_fingerprint(cfg),
        "camera_id": calib.camera_id,
        "grid_shape": [int(ys.size), int(xs.size)],
    }
    if extra_provenance:
        provenance.update(dict(extra_provenance))

    pred_full = to_optical_axis(prediction.values, prediction.convention, calib)
    anchors: FloorAnchors = select_floor_anchors(
        calib,
        plane,
        drivable,
        config=cfg.anchors,
        valid_mask=prediction.valid_mask,
        floor_segmentation=floor_segmentation,
        prediction_values=pred_full if cfg.anchors.quality_filter else None,
        uncertainty=prediction.uncertainty if cfg.anchors.quality_filter else None,
        uncertainty_kind=prediction.uncertainty_kind,
        native_confidence=prediction.native_confidence if cfg.anchors.quality_filter else None,
    )
    provenance["anchor_stage_counts"] = dict(anchors.stage_counts)
    provenance["enhanced_anchor_selection"] = bool(cfg.anchors.quality_filter)
    if len(anchors):
        provenance["anchor_weight_range"] = [
            float(np.min(anchors.weights)), float(np.max(anchors.weights))
        ]

    pred_anchor = (
        to_optical_axis(
            prediction.values[anchors.v.astype(int), anchors.u.astype(int)],
            prediction.convention,
            calib,
            u=anchors.u,
            v=anchors.v,
        )
        if len(anchors)
        else np.zeros(0)
    )

    fit = fit_ground_affine(
        pred_anchor,
        anchors.depth_m,
        prediction.convention,
        config=cfg.fit,
        anchor_depth_span_m=anchors.depth_span_m,
        weights=anchors.weights if cfg.anchors.quality_filter else None,
    )

    if temporal_filter is not None:
        fit, temporal_provenance = temporal_filter.update(
            fit,
            camera_id=calib.camera_id,
            model_name=prediction.model_name,
            timestamp_s=timestamp,
        )
        provenance["temporal_anchor"] = temporal_provenance

    if not fit.status.is_ok:
        shape = (calib.height, calib.width)
        return VisibilityResult(
            scenario_id=scenario_id,
            frame_id=frame_id or prediction.frame_id,
            camera_id=calib.camera_id or prediction.camera_id,
            timestamp=timestamp,
            status=fit.status,
            ground_fit=fit,
            metric_depth=MetricDepth(
                depth_m=np.full(shape, np.nan),
                sigma_m=np.full(shape, np.nan),
                valid=np.zeros(shape, dtype=bool),
            ),
            visibility=_all_unknown_field(xs, ys, calib, cfg),
            provenance=provenance,
        )

    depth = fit.apply(pred_full)
    # Adapter ``native_confidence`` is larger-is-better and has no metric
    # standard-deviation semantics.  It may rank anchors, but it must never be
    # added to the depth variance.  Flip-consistency and other uncertainty
    # products remain eligible for propagation.
    model_sigma = (
        None
        if str(prediction.uncertainty_kind or "") == "native_confidence"
        else prediction.uncertainty
    )
    sigma = predicted_depth_sigma(fit, pred_full, depth, model_sigma)
    valid = (
        np.asarray(prediction.valid_mask, dtype=bool)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth < cfg.max_backproject_depth_m)
        & np.isfinite(sigma)
    )
    metric = MetricDepth(depth_m=depth, sigma_m=sigma, valid=valid)

    if not bool(np.any(valid)):
        refused = dataclasses.replace(fit, status=FrameStatus.NO_VALID_DEPTH)
        return VisibilityResult(
            scenario_id=scenario_id,
            frame_id=frame_id or prediction.frame_id,
            camera_id=calib.camera_id or prediction.camera_id,
            timestamp=timestamp,
            status=FrameStatus.NO_VALID_DEPTH,
            ground_fit=refused,
            metric_depth=metric,
            visibility=_all_unknown_field(xs, ys, calib, cfg),
            provenance=provenance,
        )

    points, sigma_world = back_project(
        calib,
        depth,
        valid,
        step=cfg.backproject_step,
        max_depth_m=cfg.max_backproject_depth_m,
        sigma_m=sigma,
    )
    hmap: HeightMap = rasterize_heights(
        points, sigma_world, xs, ys, plane, min_sigma_m=cfg.raycast.min_height_sigma_m
    )
    # A cell the back-projection sub-sampling stepped over is still known if the
    # camera can see its floor; a cell whose floor is hidden is not.
    ground_seen = ground_visibility_mask(
        calib, depth, valid, sigma, xs, ys, plane,
        sigma_k=cfg.ground_visible_sigma_k, abs_tol_m=cfg.ground_visible_abs_tol_m,
    )
    hmap = dataclasses.replace(hmap, observed=hmap.observed | ground_seen)
    los = line_of_sight_field(
        calib, depth, sigma, valid, xs, ys,
        plane=plane, target=cfg.target, config=cfg.raycast,
    )
    provenance["backprojected_points"] = int(hmap.n_points)
    n_fov = int(np.count_nonzero(los.in_fov))
    provenance["unobserved_in_fov_fraction"] = (
        float(np.count_nonzero(los.in_fov & ~hmap.observed) / n_fov) if n_fov else float("nan")
    )

    field = VisibilityField(
        xs=xs,
        ys=ys,
        p_visible=los.p_visible,
        p_occluded=los.p_occluded,
        p_unknown=los.p_unknown,
        unknown_mask=los.p_unknown > 0.5,
        in_fov=los.in_fov,
        height_map_m=hmap.h_max,
        height_sigma_m=hmap.h_sigma,
        observed=hmap.observed,
    )
    return VisibilityResult(
        scenario_id=scenario_id,
        frame_id=frame_id or prediction.frame_id,
        camera_id=calib.camera_id or prediction.camera_id,
        timestamp=timestamp,
        status=FrameStatus.OK,
        ground_fit=fit,
        metric_depth=metric,
        visibility=field,
        provenance=provenance,
    )


def estimate_visibility_batch(
    predictions: Sequence[DepthPrediction],
    calibrations: Sequence[CameraCalibration],
    xs: np.ndarray,
    ys: np.ndarray,
    **kwargs: Any,
) -> list[VisibilityResult]:
    """Run the method independently per camera. No cross-camera fusion here --
    reconciling several cameras is the belief filter's job, not this method's."""
    if len(predictions) != len(calibrations):
        raise ContractViolation(
            f"{len(predictions)} predictions but {len(calibrations)} calibrations"
        )
    return [
        estimate_visibility(p, c, xs, ys, **kwargs)
        for p, c in zip(predictions, calibrations)
    ]
