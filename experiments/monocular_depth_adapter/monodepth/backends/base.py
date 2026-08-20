"""The contract every depth model has to satisfy to be usable through the adapter.

A backend is allowed to know exactly two things about the world: an RGB image and
the camera intrinsics that produced it. It returns raw depth in its own
convention plus whatever it natively knows about its own reliability.

Deliberately absent from the signature, and not obtainable from it: ground truth,
simulator depth, the floor plane, the robot, and any notion of whether a pixel is
"visible". Those live downstream and stay there.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..types import BackendInfo, CameraIntrinsics


@dataclass
class RawDepthOutput:
    """What one backend produced for one image, before the adapter dresses it up."""

    depth: np.ndarray                       # (H, W) float32, native units, native resolution
    invalid: np.ndarray | None = None       # (H, W) bool, True = backend knows this is junk
    native_confidence: np.ndarray | None = None   # (H, W) float32, backend's own scale
    extras: dict[str, Any] = field(default_factory=dict)


class DepthBackend(abc.ABC):
    """One monocular depth model, wrapped so the adapter can drive it uniformly."""

    #: registry key, e.g. "dav2_metric_indoor_large"
    model_name: str = ""
    #: model family, e.g. "depth_anything_v2"
    family: str = ""

    def __init__(self, device: str = "cuda", dtype: str = "float32") -> None:
        self.device = device
        self.dtype = dtype
        self._loaded = False

    # ------------------------------------------------------------------ lifecycle
    @abc.abstractmethod
    def load(self) -> None:
        """Fetch weights and move the network to the device. Idempotent."""

    def unload(self) -> None:
        """Drop the weights and free the device memory."""
        import gc

        import torch

        for attr in ("_model", "_processor"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------- identity
    @abc.abstractmethod
    def info(self) -> BackendInfo:
        """Identity and declared convention. Valid only after :meth:`load`."""

    # ------------------------------------------------------------------ inference
    @abc.abstractmethod
    def infer_batch(
        self,
        images: Sequence[np.ndarray],
        intrinsics: Sequence[CameraIntrinsics],
    ) -> list[RawDepthOutput]:
        """Run the network on a batch of same-shape uint8 RGB images.

        Implementations must return one output per input, at the input image's
        own resolution, and must not consult anything other than ``images`` and
        ``intrinsics``.
        """

    # --------------------------------------------------------------------- shared
    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"{self.model_name}: call load() before inference")

    @staticmethod
    def _check_uniform_shape(images: Sequence[np.ndarray]) -> tuple[int, int]:
        shapes = {img.shape[:2] for img in images}
        if len(shapes) != 1:
            raise ValueError(f"a batch must be one image size, got {sorted(shapes)}")
        h, w = shapes.pop()
        return int(h), int(w)

    @staticmethod
    def _library_versions() -> dict[str, str]:
        import torch

        versions = {"torch": str(torch.__version__)}
        try:
            import torchvision

            versions["torchvision"] = str(torchvision.__version__)
        except Exception:  # pragma: no cover - torchvision is always present here
            pass
        return versions


__all__ = ["DepthBackend", "RawDepthOutput"]
