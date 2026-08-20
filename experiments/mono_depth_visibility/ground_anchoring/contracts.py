"""Typed contracts for the ground-anchoring / visibility-inference method.

These dataclasses are the interface boundary between the three agents:

* **Agent 1** (dynamic world + oracle) emits per-frame records. Only the
  *method-visible* subset of that record may reach this package -- see
  :mod:`ground_anchoring.io_contract`. Oracle depth, simulator obstacle poses
  and ray-cast oracle visibility are EVALUATION-ONLY and are never read here.
* **Agent 2** (monocular depth adapter) emits a :class:`DepthPrediction`.
* **Agent 3** (this package) consumes both and emits a
  :class:`VisibilityResult`.

Everything here is plain numpy + dataclasses so that agents 1 and 2 can import
the contracts without pulling in the method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class DepthConventionError(ValueError):
    """A depth convention is unknown, or the data contradicts the declared one.

    Raised -- never swallowed -- because a silently mis-handled convention
    produces a plausible-looking metric depth map that is wrong everywhere.
    """


class ContractViolation(ValueError):
    """A frame record or prediction does not satisfy the agreed contract."""


# ---------------------------------------------------------------------------
# Depth conventions
# ---------------------------------------------------------------------------
class DepthConvention(Enum):
    """How to read the numbers a monocular model produced.

    ``METRIC_Z``          metres along the optical axis (the pinhole ``z_c``).
    ``EUCLIDEAN_RANGE``   metres along the ray from the camera centre.
    ``RELATIVE_DEPTH``    depth-like, up to an unknown positive scale and shift.
    ``INVERSE_DEPTH``     disparity-like: proportional to 1/depth, up to an
                          unknown positive scale and shift.

    The names and values match the depth adapter's ``monodepth.DepthConvention``
    so a prediction crosses the boundary without translation.

    The distinction that actually matters is the last one: an affine fit for an
    inverse-depth model must be performed in *inverse* space. Fitting
    ``z = a*p + b`` to a disparity map is not a worse fit, it is the wrong
    model, and it fails hardest exactly where the obstacles are.
    """

    METRIC_Z = "metric_z"
    EUCLIDEAN_RANGE = "euclidean_range"
    RELATIVE_DEPTH = "relative_depth"
    INVERSE_DEPTH = "inverse_depth"

    @property
    def is_metric(self) -> bool:
        """True when the model claims its numbers are already in metres."""
        return self in (DepthConvention.METRIC_Z, DepthConvention.EUCLIDEAN_RANGE)

    @property
    def is_inverse(self) -> bool:
        """True when the prediction rises as true depth falls."""
        return self is DepthConvention.INVERSE_DEPTH

    @classmethod
    def parse(cls, value: "str | DepthConvention") -> "DepthConvention":
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        text = _CONVENTION_ALIASES.get(text, text)
        try:
            return cls(text)
        except ValueError as exc:  # loud, with the permitted set spelled out
            known = ", ".join(sorted(c.value for c in cls))
            raise DepthConventionError(
                f"unknown depth convention {value!r}; expected one of: {known}"
            ) from exc


#: accepted alternative spellings; each maps to the same physical meaning
_CONVENTION_ALIASES = {"metric_range": "euclidean_range", "disparity": "inverse_depth"}


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics + world->camera extrinsics.

    ``R`` maps world into camera: ``x_cam = R @ (x_world - cam_pos)``. This is
    the convention of ``unav_common.camera_model.ObliqueCameraModel``; use
    :meth:`from_oblique` to adopt an existing repo camera unchanged.

    The method never modifies calibration -- it is an input, and every
    "correction" this package fits lives in the depth values, not in ``K`` or
    ``R``.
    """

    K: np.ndarray
    R: np.ndarray
    cam_pos: np.ndarray
    width: int
    height: int
    camera_id: str = "camera"

    def __post_init__(self) -> None:
        object.__setattr__(self, "K", np.asarray(self.K, dtype=float).reshape(3, 3))
        object.__setattr__(self, "R", np.asarray(self.R, dtype=float).reshape(3, 3))
        object.__setattr__(self, "cam_pos", np.asarray(self.cam_pos, dtype=float).reshape(3))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        if not np.isfinite(self.K).all() or not np.isfinite(self.R).all():
            raise ContractViolation("camera calibration contains non-finite entries")
        if abs(abs(np.linalg.det(self.R)) - 1.0) > 1e-6:
            raise ContractViolation(
                f"extrinsic rotation is not orthonormal (|det| = {abs(np.linalg.det(self.R)):.6f})"
            )

    @classmethod
    def from_oblique(cls, cam: Any, camera_id: str = "camera") -> "CameraCalibration":
        """Adopt an ``ObliqueCameraModel`` (or anything with K/R/cam_pos)."""
        return cls(
            K=cam.K,
            R=cam.R,
            cam_pos=cam.cam_pos,
            width=cam.img_width,
            height=cam.img_height,
            camera_id=camera_id,
        )

    def rays_cam(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``K^-1 [u, v, 1]`` per pixel, shape ``(3, N)``. The z-row is 1, so
        scaling a ray by ``z_c`` gives the camera-frame point directly."""
        u = np.asarray(u, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        homog = np.vstack([u, v, np.ones_like(u)])
        return np.linalg.inv(self.K) @ homog

    def rays_world(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """World-frame ray directions per unit optical-axis depth, ``(3, N)``."""
        return self.R.T @ self.rays_cam(u, v)

    def ray_norms(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``||K^-1 [u,v,1]||`` -- the Euclidean-range / optical-depth ratio."""
        return np.linalg.norm(self.rays_cam(u, v), axis=0)

    def pixel_grid(self, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sub-sampled integer pixel coordinates ``(u, v)``, both flat."""
        step = max(1, int(step))
        vv, uu = np.mgrid[0 : self.height : step, 0 : self.width : step]
        return uu.ravel().astype(float), vv.ravel().astype(float)


@dataclass(frozen=True)
class FloorPlane:
    """Warehouse floor as ``normal . X = offset`` in world coordinates."""

    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    offset: float = 0.0

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            raise ContractViolation("floor plane normal is degenerate")
        object.__setattr__(self, "normal", n / norm)
        object.__setattr__(self, "offset", float(self.offset) / norm)

    def height_above(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of world points from the plane, along the normal."""
        pts = np.asarray(points, dtype=float)
        return pts @ self.normal - self.offset


@dataclass(frozen=True)
class Footprint:
    """Axis-aligned world footprint (metres). Field names match
    ``geometry_visibility.Prism`` so repo prisms can be passed straight in."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    name: str = "region"

    def covers(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)


def covered_by_any(x: np.ndarray, y: np.ndarray, regions: Sequence[Any]) -> np.ndarray:
    """Boolean mask: point inside at least one footprint (duck-typed on
    ``xmin/xmax/ymin/ymax``, so ``geometry_visibility.Prism`` works)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for r in regions:
        mask |= (x >= r.xmin) & (x <= r.xmax) & (y >= r.ymin) & (y <= r.ymax)
    return mask


# ---------------------------------------------------------------------------
# Agent 2 input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepthPrediction:
    """One monocular prediction, exactly as the depth adapter emits it."""

    values: np.ndarray
    convention: DepthConvention
    valid_mask: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    uncertainty_kind: str | None = None
    native_confidence: np.ndarray | None = None
    model_name: str = "unknown"
    checkpoint: str = "unknown"
    inference_time_s: float = float("nan")
    frame_id: str = ""
    camera_id: str = ""

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ContractViolation(f"depth prediction must be 2-D, got shape {values.shape}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "convention", DepthConvention.parse(self.convention))
        if self.valid_mask is None:
            mask = np.isfinite(values)
        else:
            mask = np.asarray(self.valid_mask, dtype=bool)
            if mask.shape != values.shape:
                raise ContractViolation(
                    f"valid_mask shape {mask.shape} != prediction shape {values.shape}"
                )
            mask = mask & np.isfinite(values)
        object.__setattr__(self, "valid_mask", mask)
        if self.uncertainty is not None:
            unc = np.asarray(self.uncertainty, dtype=float)
            if unc.shape != values.shape:
                raise ContractViolation(
                    f"uncertainty shape {unc.shape} != prediction shape {values.shape}"
                )
            object.__setattr__(self, "uncertainty", unc)
        if self.native_confidence is not None:
            confidence = np.asarray(self.native_confidence, dtype=float)
            if confidence.shape != values.shape:
                raise ContractViolation(
                    f"native_confidence shape {confidence.shape} != prediction shape {values.shape}"
                )
            object.__setattr__(self, "native_confidence", confidence)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnchorConfig:
    """Which pixels are allowed to act as known-depth floor anchors."""

    pixel_step: int = 4
    min_depth_m: float = 0.5
    max_depth_m: float = 60.0
    require_drivable: bool = True
    #: Opt-in quality filtering. False preserves the original study byte for
    #: byte; online configurations can enable the filters below.
    quality_filter: bool = False
    #: Erode an externally supplied floor segmentation before it can provide
    #: anchors. This avoids mixed floor/object pixels along semantic edges.
    segmentation_erosion_px: int = 3
    #: Reject depth-discontinuity pixels and a small neighbourhood around them.
    depth_edge_quantile: float = 0.90
    depth_edge_dilation_px: int = 2
    #: When native confidence exists, retain this upper fraction. The raw
    #: confidence scale is model-specific, so selection is rank-based.
    confidence_keep_fraction: float = 0.95
    #: When a genuine uncertainty map exists, retain this least-uncertain
    #: fraction. Native-confidence maps are never interpreted as uncertainty.
    uncertainty_keep_fraction: float = 0.85
    #: Lower bound for quality weights passed to the robust affine fit.
    min_quality_weight: float = 0.05


@dataclass(frozen=True)
class FitConfig:
    """Robust scale/shift fit and the gates that reject an unusable frame."""

    ransac_iters: int = 400
    ransac_seed: int = 0
    inlier_rel_tol: float = 0.05
    inlier_abs_tol_m: float = 0.05
    min_anchor_pixels: int = 200
    min_inlier_fraction: float = 0.40
    min_depth_span_m: float = 1.0
    max_residual_rms_m: float = 0.50
    max_condition_number: float = 1.0e6
    #: only enforced when the model claims to be metric already
    metric_scale_band: tuple[float, float] = (0.5, 2.0)
    #: raise on a detected convention mismatch instead of returning "unknown"
    strict_convention: bool = True


@dataclass(frozen=True)
class TargetVolume:
    """The robot body the camera has to see, as a small upright cylinder.

    Defaults describe the TurtleBot3 Burger used in this thesis (~0.14 m
    radius, ~0.20 m tall). Note that parts of the repo use a 0.35 m marker
    height; that is taller than the robot and therefore *anti*-conservative for
    occlusion, which is why this method takes the volume explicitly.
    """

    radius_m: float = 0.14
    z_min_m: float = 0.05
    z_max_m: float = 0.20
    n_heights: int = 3
    n_ring: int = 0

    def sample_offsets(self) -> np.ndarray:
        """Offsets ``(M, 3)`` from the cell centre, spanning the body."""
        if self.n_heights < 1:
            raise ContractViolation("TargetVolume.n_heights must be >= 1")
        zs = (
            np.array([0.5 * (self.z_min_m + self.z_max_m)])
            if self.n_heights == 1
            else np.linspace(self.z_min_m, self.z_max_m, self.n_heights)
        )
        if self.n_ring <= 0:
            xy = np.zeros((1, 2))
        else:
            ang = np.linspace(0.0, 2.0 * np.pi, self.n_ring, endpoint=False)
            xy = np.column_stack(
                [np.concatenate([[0.0], self.radius_m * np.cos(ang)]),
                 np.concatenate([[0.0], self.radius_m * np.sin(ang)])]
            )
        offsets = [(x, y, z) for z in zs for (x, y) in xy]
        return np.asarray(offsets, dtype=float)


@dataclass(frozen=True)
class RaycastConfig:
    """Sightline test tolerances."""

    #: Floor on the depth sigma used in the sightline comparison. Without it a
    #: perfectly-fitting frame would claim certainty it has not earned, and the
    #: comparison would collapse to a hard threshold.
    min_depth_sigma_m: float = 0.01
    #: floor on per-cell height sigma in the published height map
    min_height_sigma_m: float = 0.02


@dataclass(frozen=True)
class MethodConfig:
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    target: TargetVolume = field(default_factory=TargetVolume)
    raycast: RaycastConfig = field(default_factory=RaycastConfig)
    #: Pixel sub-sampling for back-projection into the height map. The grid
    #: must be coarser than the resulting ground sample spacing at the far end
    #: of the image, or distant cells get no return and are reported unknown
    #: for a sampling reason rather than an occlusion reason. The pipeline
    #: records ``unobserved_in_fov_fraction`` so that shows up instead of
    #: hiding.
    backproject_step: int = 2
    max_backproject_depth_m: float = 60.0
    #: tolerance of the forward depth-buffer test that decides whether the
    #: camera really saw a cell's floor: this many sigmas plus an absolute slack
    ground_visible_sigma_k: float = 3.0
    ground_visible_abs_tol_m: float = 0.05


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
class FrameStatus(Enum):
    """Why a frame was accepted or refused. Anything but ``OK`` means the
    visibility field is all-unknown -- the method declines rather than guesses."""

    OK = "ok"
    INSUFFICIENT_FLOOR_PIXELS = "insufficient_floor_pixels"
    INSUFFICIENT_DEPTH_SPAN = "insufficient_depth_span"
    ILL_CONDITIONED = "ill_conditioned"
    LOW_INLIER_FRACTION = "low_inlier_fraction"
    HIGH_RESIDUAL = "high_residual"
    NON_PHYSICAL_SCALE = "non_physical_scale"
    CONVENTION_MISMATCH = "convention_mismatch"
    NO_VALID_DEPTH = "no_valid_depth"

    @property
    def is_ok(self) -> bool:
        return self is FrameStatus.OK


@dataclass(frozen=True)
class GroundFit:
    """The fitted mapping from model output to metres, and its evidence."""

    scale: float
    shift: float
    #: "depth" (z = a*p + b) or "inverse_depth" (1/z = a*p + b)
    fit_space: str
    convention: DepthConvention
    status: FrameStatus
    n_anchor: int
    n_inlier: int
    inlier_fraction: float
    residual_rms_m: float
    residual_p95_m: float
    anchor_depth_span_m: float
    condition_number: float
    sigma_fit: float
    ata_inv: np.ndarray
    #: Direct posterior covariance of [scale, shift]. Legacy single-frame fits
    #: leave this unset and reconstruct it as sigma_fit^2 * ata_inv.
    parameter_cov: np.ndarray | None = None
    #: anchors whose corrected depth is *shorter* than the analytic floor --
    #: physically these are things standing on the floor (a box, a forklift)
    n_shorter_than_floor: int = 0
    #: anchors *behind* the floor plane; physically impossible, so a nonzero
    #: count points at bad calibration or a wrong plane, not at occlusion
    n_beyond_floor: int = 0
    notes: str = ""

    def apply(self, prediction: np.ndarray) -> np.ndarray:
        """Model output -> metric optical-axis depth (may contain non-finite)."""
        y = self.scale * np.asarray(prediction, dtype=float) + self.shift
        if self.fit_space == "inverse_depth":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(y > 1e-9, 1.0 / y, np.nan)
        return y

    @property
    def parameter_covariance(self) -> np.ndarray:
        if self.parameter_cov is not None:
            return np.asarray(self.parameter_cov, dtype=float).reshape(2, 2)
        return np.asarray(self.ata_inv, dtype=float) * self.sigma_fit**2

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "shift": self.shift,
            "fit_space": self.fit_space,
            "convention": self.convention.value,
            "status": self.status.value,
            "n_anchor": self.n_anchor,
            "n_inlier": self.n_inlier,
            "inlier_fraction": self.inlier_fraction,
            "residual_rms_m": self.residual_rms_m,
            "residual_p95_m": self.residual_p95_m,
            "anchor_depth_span_m": self.anchor_depth_span_m,
            "condition_number": self.condition_number,
            "sigma_fit": self.sigma_fit,
            "parameter_cov": self.parameter_covariance.tolist(),
            "n_shorter_than_floor": self.n_shorter_than_floor,
            "n_beyond_floor": self.n_beyond_floor,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MetricDepth:
    """Corrected depth in metres along the optical axis, with a 1-sigma."""

    depth_m: np.ndarray
    sigma_m: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class VisibilityField:
    """Per-cell line-of-sight belief on the planning grid.

    ``p_visible + p_occluded + p_unknown == 1`` in every cell. ``p_unknown`` is
    not a fudge term: it is the mass contributed by sightlines that pass low
    over cells the camera returned no depth for, where an unseen obstacle could
    be hiding.
    """

    xs: np.ndarray
    ys: np.ndarray
    p_visible: np.ndarray
    p_occluded: np.ndarray
    p_unknown: np.ndarray
    unknown_mask: np.ndarray
    in_fov: np.ndarray
    height_map_m: np.ndarray
    height_sigma_m: np.ndarray
    observed: np.ndarray

    @property
    def p_los(self) -> np.ndarray:
        """Line-of-sight probability conditioned on the sightline being known.

        ``p_visible / (p_visible + p_occluded)``, i.e. what the method believes
        where it believes anything. Cells that are entirely unknown return NaN
        rather than a made-up 0.5.
        """
        known = self.p_visible + self.p_occluded
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(known > 1e-9, self.p_visible / known, np.nan)


@dataclass(frozen=True)
class VisibilityResult:
    """Everything the method promises to emit for one frame."""

    scenario_id: str
    frame_id: str
    camera_id: str
    timestamp: float
    status: FrameStatus
    ground_fit: GroundFit
    metric_depth: MetricDepth
    visibility: VisibilityField
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status.is_ok

    def summary(self) -> dict:
        f = self.visibility
        return {
            "scenario_id": self.scenario_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "ground_fit": self.ground_fit.to_dict(),
            "depth_valid_fraction": float(np.mean(self.metric_depth.valid)),
            "depth_sigma_median_m": float(
                np.median(self.metric_depth.sigma_m[self.metric_depth.valid])
            )
            if bool(np.any(self.metric_depth.valid))
            else float("nan"),
            "cells_total": int(f.p_visible.size),
            "mean_p_visible": float(np.mean(f.p_visible)),
            "mean_p_occluded": float(np.mean(f.p_occluded)),
            "mean_p_unknown": float(np.mean(f.p_unknown)),
            "unknown_cells": int(np.count_nonzero(f.unknown_mask)),
            "provenance": dict(self.provenance),
        }
