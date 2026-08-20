"""Basic per-pixel uncertainty signals.

Three sources, in increasing order of cost:

``native``
    The model's own confidence head, if it has one (Metric3D v2 and UniDepthV2
    do, Depth Anything V2 does not). Each model's scale is its own; the raw
    values are passed through unnormalised and must not be compared across
    models without a calibration step that lives downstream.

``flip_consistency``
    Run the model again on the left-right mirrored image (with the principal
    point mirrored to match), unflip, and take the per-pixel spread of the two
    predictions. Costs one extra forward pass. It measures how much the answer
    depends on where things happen to sit in the frame — which for an overhead
    oblique view, well outside these networks' training distribution, is a real
    and observable failure mode.

``temporal_disagreement``
    Per-pixel spread across several frames from the same fixed camera. The
    static scene should give a static depth; whatever moves is either the robot
    or the model being unstable. Computed offline over a stack of predictions,
    so it costs nothing extra at inference.

A note that matters for non-metric models: a relative-depth network may return a
differently scaled answer for the mirrored image, and differencing the two
unaligned would report that free scale as uncertainty. So for non-metric
conventions the mirrored prediction is affine-aligned to the reference first.
Metric predictions are differenced as they are — for them, a scale difference
between two views of the same scene is a real error, not a gauge freedom.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .conventions import align_affine
from .types import DepthConvention, DepthPrediction

NATIVE = "native_confidence"
FLIP = "flip_consistency"
TEMPORAL = "temporal_disagreement"


def flip_consistency(
    depth: np.ndarray,
    depth_from_flipped: np.ndarray,
    valid: np.ndarray,
    convention: DepthConvention,
) -> tuple[np.ndarray, dict]:
    """Per-pixel spread between a prediction and its mirrored-input twin.

    ``depth_from_flipped`` must already be un-flipped back onto the original
    pixel grid. Returns the half-absolute-difference (the standard deviation of
    a two-point sample) and a small report of what was done.
    """
    a = np.asarray(depth, dtype=np.float64)
    b = np.asarray(depth_from_flipped, dtype=np.float64)
    detail: dict = {"method": FLIP, "affine_aligned": False, "scale": 1.0, "shift": 0.0}

    if not convention.is_metric:
        scale, shift = align_affine(b, a, valid)
        b = scale * b + shift
        detail.update(affine_aligned=True, scale=scale, shift=shift)

    spread = 0.5 * np.abs(a - b)
    spread[~valid] = np.nan
    finite = spread[np.isfinite(spread)]
    detail["median_spread"] = float(np.median(finite)) if finite.size else float("nan")
    detail["p95_spread"] = float(np.percentile(finite, 95)) if finite.size else float("nan")
    detail["unit"] = convention.unit
    return spread.astype(np.float32), detail


def temporal_disagreement(
    predictions: Sequence[DepthPrediction],
    *,
    min_valid_frames: int = 2,
) -> tuple[np.ndarray, dict]:
    """Per-pixel standard deviation across frames from one fixed camera.

    Only meaningful when the camera did not move between frames, which is the
    case for wall-mounted cameras. Pixels seen validly in fewer than
    ``min_valid_frames`` frames come back NaN.

    Raises if the stack mixes cameras, image sizes, models, or conventions —
    every one of those would turn a bookkeeping mistake into a plausible-looking
    uncertainty map.
    """
    if len(predictions) < min_valid_frames:
        raise ValueError(f"need at least {min_valid_frames} predictions, got {len(predictions)}")

    shapes = {p.depth.shape for p in predictions}
    models = {p.model.model_name for p in predictions}
    conventions = {p.convention for p in predictions}
    intrinsics = {tuple(sorted(p.intrinsics.as_dict().items())) for p in predictions}
    if len(shapes) != 1:
        raise ValueError(f"mixed image sizes in the stack: {sorted(shapes)}")
    if len(models) != 1:
        raise ValueError(f"mixed models in the stack: {sorted(models)}")
    if len(conventions) != 1:
        raise ValueError(f"mixed depth conventions in the stack: {sorted(c.value for c in conventions)}")
    if len(intrinsics) != 1:
        raise ValueError("mixed intrinsics in the stack; temporal spread assumes one fixed camera")

    convention = conventions.pop()
    stack = np.stack([p.depth.astype(np.float64) for p in predictions])
    valid = np.stack([p.valid for p in predictions])

    if not convention.is_metric:
        # Align every frame to the first before comparing, for the same gauge
        # reason as flip consistency.
        reference = stack[0]
        both = valid & valid[0]
        for i in range(1, stack.shape[0]):
            scale, shift = align_affine(stack[i], reference, both[i])
            stack[i] = scale * stack[i] + shift

    stack[~valid] = np.nan
    counts = valid.sum(axis=0)
    # A pixel invalid in every frame is an all-NaN slice; nanstd warns about it
    # and returns NaN, which is exactly the answer wanted. Silence the noise
    # rather than pre-filtering, so the pixel keeps its place in the map.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(stack, axis=0)
    spread[counts < min_valid_frames] = np.nan

    finite = spread[np.isfinite(spread)]
    detail = {
        "method": TEMPORAL,
        "n_frames": int(len(predictions)),
        "model": models.pop(),
        "unit": convention.unit,
        "median_spread": float(np.median(finite)) if finite.size else float("nan"),
        "p95_spread": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "affine_aligned": not convention.is_metric,
    }
    return spread.astype(np.float32), detail


def summarize(uncertainty: np.ndarray | None) -> dict:
    """Compact stats for a run report; NaN-safe and empty-safe."""
    if uncertainty is None:
        return {"available": False}
    finite = uncertainty[np.isfinite(uncertainty)]
    if finite.size == 0:
        return {"available": True, "n_finite": 0}
    return {
        "available": True,
        "n_finite": int(finite.size),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


__all__ = ["NATIVE", "FLIP", "TEMPORAL", "flip_consistency", "temporal_disagreement", "summarize"]
