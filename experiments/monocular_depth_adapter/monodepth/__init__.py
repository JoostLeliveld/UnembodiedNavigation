"""Monocular depth adapter — RGB + intrinsics in, depth + metadata out.

One wrapper over several monocular depth networks, so that swapping the model
does not change the calling code and never silently changes what the numbers
mean. Each model's own depth convention is preserved and carried alongside the
array, invalid pixels are marked explicitly, and every prediction records the
model, checkpoint, runtime and memory that produced it.

Scope, stated as much by what is missing as by what is here. This package has no
access to ground truth, no access to simulator depth, no floor-plane anchoring,
no opinion about whether anything is visible, and no ability to modify a camera
calibration. Those belong to later stages of the pipeline and are kept out by the
backend signature, which accepts an image and intrinsics and nothing else.

Typical use::

    from monodepth import MonocularDepthAdapter, DepthRequest, CameraIntrinsics

    K = CameraIntrinsics(fx=640, fy=640, cx=640, cy=360, width=1280, height=720)
    with MonocularDepthAdapter("unidepth_v2_vits14", batch_size=4) as adapter:
        preds = adapter.predict([DepthRequest("frame0", rgb, K)])
    print(preds[0].convention, preds[0].valid_fraction)

``determinism`` is imported first on purpose: it sets ``CUBLAS_WORKSPACE_CONFIG``
before anything can create a CUDA context.
"""

from __future__ import annotations

from . import determinism  # noqa: F401  (import order matters, see module docstring)
from . import conventions, storage, uncertainty
from .adapter import UNCERTAINTY_MODES, MonocularDepthAdapter
from .backends import MODEL_FAMILIES, available_models, create_backend, family_of
from .types import (
    BackendInfo,
    CameraIntrinsics,
    DepthConvention,
    DepthPrediction,
    DepthRequest,
    InferenceTiming,
    MemoryRecord,
    build_valid_mask,
)

__version__ = "0.1.0"

__all__ = [
    "MonocularDepthAdapter",
    "UNCERTAINTY_MODES",
    "DepthRequest",
    "DepthPrediction",
    "DepthConvention",
    "CameraIntrinsics",
    "BackendInfo",
    "InferenceTiming",
    "MemoryRecord",
    "build_valid_mask",
    "available_models",
    "create_backend",
    "family_of",
    "MODEL_FAMILIES",
    "conventions",
    "determinism",
    "storage",
    "uncertainty",
]
