"""Value types for the monocular depth adapter.

Everything here is plain data. The adapter hands back exactly what the network
produced, in the network's own units, plus enough metadata that a later stage
can decide what to do with it. Nothing in this module knows about warehouses,
floors, cameras-on-walls, or ground truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class DepthConvention(str, enum.Enum):
    """What a depth number actually means.

    Four different things get called "depth" and they are not interchangeable:

    - ``metric_z``          metres measured along the optical axis (planar depth).
                            The distance to the plane through the point parallel
                            to the image plane.
    - ``euclidean_range``   metres measured along the ray from the camera centre
                            to the point. Always >= the metric_z of the same
                            point, by the secant of the angle off the axis.
    - ``relative_depth``    unitless, larger = further, but only up to an unknown
                            scale (and usually an unknown shift). Cannot be turned
                            into metres without an external anchor.
    - ``inverse_depth``     unitless disparity-like, larger = NEARER. Zero means
                            infinitely far, which is a legal value, not a hole.

    Converting between the two metric conventions is pure geometry and needs only
    the intrinsics (see :mod:`monodepth.conventions`). Converting a non-metric
    convention into metres needs a scene anchor, which this adapter is explicitly
    forbidden from having.
    """

    METRIC_Z = "metric_z"
    EUCLIDEAN_RANGE = "euclidean_range"
    RELATIVE_DEPTH = "relative_depth"
    INVERSE_DEPTH = "inverse_depth"

    @property
    def is_metric(self) -> bool:
        """True when the numbers are metres, so a threshold in metres is meaningful."""
        return self in (DepthConvention.METRIC_Z, DepthConvention.EUCLIDEAN_RANGE)

    @property
    def larger_is_nearer(self) -> bool:
        """True only for inverse depth, where the ordering is flipped."""
        return self is DepthConvention.INVERSE_DEPTH

    @property
    def unit(self) -> str:
        return "m" if self.is_metric else "unitless"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one image, in pixels.

    The adapter reads these and never writes them. It has no authority to
    re-estimate, refine, or otherwise alter a camera calibration; if a model
    reports its own intrinsics estimate (UniDepthV2 does), that estimate is
    recorded as an observation in ``extras`` and never substituted in here.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"focal lengths must be positive, got fx={self.fx} fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"image size must be positive, got {self.width}x{self.height}")

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            raise ValueError(f"expected a 3x3 intrinsic matrix, got {K.shape}")
        return cls(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                   cy=float(K[1, 2]), width=int(width), height=int(height))

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def mirrored_horizontally(self) -> "CameraIntrinsics":
        """Intrinsics of the same camera after the image is flipped left-right.

        Used by the flip-consistency uncertainty signal. Only the principal point
        moves; a flipped pixel column ``u`` lands at ``(width - 1) - u``.
        """
        return CameraIntrinsics(fx=self.fx, fy=self.fy,
                                cx=(self.width - 1) - self.cx, cy=self.cy,
                                width=self.width, height=self.height)

    def scaled(self, sx: float, sy: float) -> "CameraIntrinsics":
        return CameraIntrinsics(fx=self.fx * sx, fy=self.fy * sy,
                                cx=self.cx * sx, cy=self.cy * sy,
                                width=int(round(self.width * sx)),
                                height=int(round(self.height * sy)))

    def as_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height}


@dataclass(frozen=True)
class BackendInfo:
    """Identity of the thing that produced a prediction.

    Everything needed to say "this number came from that model with those weights
    on that machine", so a prediction can be re-derived or invalidated later.
    """

    backend: str                       # model family, e.g. "depth_anything_v2"
    model_name: str                    # registry key, e.g. "dav2_metric_indoor_large"
    checkpoint: str                    # HF repo id / hub entry + weights URL
    checkpoint_revision: str | None    # resolved commit hash when the hub reports one
    convention: DepthConvention        # what the raw output means
    provides_native_confidence: bool
    uses_intrinsics: bool              # does the model actually consume K?
    native_input_size: tuple[int, int] | None   # (h, w) the network runs at, if fixed
    device: str = "cpu"
    torch_dtype: str = "float32"
    parameter_count: int = 0
    library_versions: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["convention"] = self.convention.value
        d["library_versions"] = dict(self.library_versions)
        d["native_input_size"] = list(self.native_input_size) if self.native_input_size else None
        return d


@dataclass(frozen=True)
class InferenceTiming:
    """Wall-clock seconds per frame, split by what the time was spent on.

    ``forward_s`` is measured with the CUDA stream synchronised, so it is real
    elapsed GPU time rather than the time to enqueue the kernels. Each network
    wants its own input canvas, so its resize/normalise happens inside the
    backend and is counted in ``forward_s``; ``preprocess_s`` covers only the
    adapter's own marshalling and is correspondingly tiny.

    ``uncertainty_s`` is separate and matters: flip consistency doubles the
    forward passes, and a cost table that hid that would understate the price of
    the uncertainty signal by 2x.
    """

    preprocess_s: float
    forward_s: float
    postprocess_s: float
    uncertainty_s: float = 0.0
    batch_size: int = 1

    @property
    def total_s(self) -> float:
        return self.preprocess_s + self.forward_s + self.postprocess_s + self.uncertainty_s

    def as_dict(self) -> dict:
        return {"preprocess_s": self.preprocess_s, "forward_s": self.forward_s,
                "postprocess_s": self.postprocess_s, "uncertainty_s": self.uncertainty_s,
                "total_s": self.total_s, "batch_size": self.batch_size}


@dataclass(frozen=True)
class MemoryRecord:
    """Peak memory during one inference call.

    GPU figures are per-batch peaks from torch's allocator, reset immediately
    before the call. ``host_rss_peak_mib`` is the process resident set size after
    the call, which includes the loaded weights and is therefore cumulative
    rather than per-call.
    """

    gpu_peak_allocated_mib: float
    gpu_peak_reserved_mib: float
    host_rss_mib: float
    weights_mib: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DepthPrediction:
    """One model's answer for one image, in that model's own convention.

    ``depth`` is never rescaled, re-anchored, or converted on the way out. A
    relative-depth model's output stays relative. That is the whole point of
    carrying ``convention`` around.
    """

    image_id: str
    depth: np.ndarray                        # (H, W) float32, raw model units
    convention: DepthConvention
    valid: np.ndarray                        # (H, W) bool, True = usable pixel
    intrinsics: CameraIntrinsics
    model: BackendInfo
    timing: InferenceTiming
    memory: MemoryRecord
    uncertainty: np.ndarray | None = None    # (H, W) float32, same units as depth
    uncertainty_kind: str | None = None      # see monodepth.uncertainty
    uncertainty_detail: Mapping[str, Any] = field(default_factory=dict)
    native_confidence: np.ndarray | None = None   # (H, W) float32, raw model scale
    image_sha256: str | None = None
    source_path: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)        # JSON-safe scalars only
    extra_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {self.depth.shape}")
        if self.valid.shape != self.depth.shape:
            raise ValueError(f"valid mask {self.valid.shape} != depth {self.depth.shape}")
        if self.depth.shape != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError(
                f"depth {self.depth.shape} does not match the declared image size "
                f"{(self.intrinsics.height, self.intrinsics.width)}"
            )
        for name in ("uncertainty", "native_confidence"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != self.depth.shape:
                raise ValueError(f"{name} {arr.shape} != depth {self.depth.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape  # type: ignore[return-value]

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())

    def metadata(self) -> dict:
        """Everything except the pixel arrays, JSON-serialisable."""
        return {
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
            "source_path": self.source_path,
            "convention": self.convention.value,
            "unit": self.convention.unit,
            "larger_is_nearer": self.convention.larger_is_nearer,
            "height": int(self.depth.shape[0]),
            "width": int(self.depth.shape[1]),
            "valid_fraction": self.valid_fraction,
            "intrinsics": self.intrinsics.as_dict(),
            "model": self.model.as_dict(),
            "timing": self.timing.as_dict(),
            "memory": self.memory.as_dict(),
            "uncertainty_kind": self.uncertainty_kind,
            "uncertainty_detail": dict(self.uncertainty_detail),
            "has_uncertainty": self.uncertainty is not None,
            "has_native_confidence": self.native_confidence is not None,
            "extras": dict(self.extras),
            "extra_arrays": sorted(self.extra_arrays),
        }


@dataclass(frozen=True)
class DepthRequest:
    """One unit of work: an image and the calibration that goes with it."""

    image_id: str
    image: np.ndarray               # (H, W, 3) uint8 RGB
    intrinsics: CameraIntrinsics
    source_path: str | None = None
    image_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB, got {self.image.dtype}")
        h, w = self.image.shape[:2]
        if (h, w) != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError(
                f"image is {w}x{h} but the intrinsics declare "
                f"{self.intrinsics.width}x{self.intrinsics.height}"
            )


def build_valid_mask(
    depth: np.ndarray,
    convention: DepthConvention,
    *,
    extra_invalid: np.ndarray | None = None,
    min_metric_m: float = 0.0,
    max_metric_m: float = 300.0,
) -> np.ndarray:
    """Mark every pixel whose depth value should not be trusted.

    Always invalid: non-finite values. For the two metric conventions, values
    outside ``(min_metric_m, max_metric_m]`` are invalid too — that is where a
    network's saturation and its "I have no idea" outputs land. For inverse
    depth, zero is kept: it means infinitely far, not a hole.

    ``extra_invalid`` is the backend's own knowledge, e.g. pixels that only exist
    because the image was padded up to the network's input size.
    """
    valid = np.isfinite(depth)
    if convention.is_metric:
        valid &= depth > float(min_metric_m)
        valid &= depth <= float(max_metric_m)
    elif convention is DepthConvention.INVERSE_DEPTH:
        valid &= depth >= 0.0
    else:  # relative depth: only the ordering is meaningful, negatives are not
        valid &= depth > 0.0
    if extra_invalid is not None:
        valid &= ~np.asarray(extra_invalid, dtype=bool)
    return valid


__all__: Sequence[str] = [
    "DepthConvention",
    "CameraIntrinsics",
    "BackendInfo",
    "InferenceTiming",
    "MemoryRecord",
    "DepthPrediction",
    "DepthRequest",
    "build_valid_mask",
]
