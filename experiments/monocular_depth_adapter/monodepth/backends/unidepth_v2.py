"""UniDepthV2, from the authors' package (``pip install git+.../UniDepth``).

The only model here that returns both metric conventions at once:

- ``depth``   metres along the optical axis  -> ``metric_z``
- ``radius``  metres along the ray           -> ``euclidean_range``

Measured on a 1280x720 frame with fx = 640: ``radius/depth`` at the image corner
came out 1.5051 against the 1.5060 the intrinsics predict, so the two really are
the two conventions and not a rescaling of each other. ``metric_z`` is the
declared convention; ``radius`` is carried in ``extras`` for anyone who wants the
range instead of re-deriving it.

Two things worth knowing before trusting it:

1. When you pass K it *is* used — the depth field changes materially versus
   letting the model self-estimate. Pass it.
2. The ``intrinsics`` it returns is nevertheless its OWN estimate, not your
   input. It is recorded in ``extras`` as an observation about the model, never
   written back into the calibration.

Import shim: ``unidepth.utils.visualization`` imports ``wandb`` at module scope
for training-time logging. A stub module stands in, so the dependency (and its
protobuf/sentry tail) stays out of the environment.
"""

from __future__ import annotations

import sys
import types
from typing import Sequence

import numpy as np

from ..types import BackendInfo, CameraIntrinsics, DepthConvention
from .base import DepthBackend, RawDepthOutput

CHECKPOINTS: dict[str, dict] = {
    "unidepth_v2_vits14": {
        "repo": "lpiccinelli/unidepth-v2-vits14",
        "notes": "ViT-S/14 backbone",
    },
    "unidepth_v2_vitl14": {
        "repo": "lpiccinelli/unidepth-v2-vitl14",
        "notes": "ViT-L/14 backbone",
    },
}

#: 0-9; picks the pixel budget the network internally resizes to. 9 = finest.
DEFAULT_RESOLUTION_LEVEL = 9

#: The installed UniDepth (0.1) mis-slices a batched camera: ``BatchCamera.unproject``
#: hands the whole (B, ...) pixel grid to each of the B per-image cameras and then
#: reshapes it to one image's size, so any forward with more than one image and
#: supplied intrinsics dies with a shape error. Measured 2026-08-11 on this
#: checkout. The adapter's batch API still accepts many frames; this backend just
#: feeds the network one at a time and says so in `extras`.
SUPPORTS_BATCHED_FORWARD = False


def _install_wandb_stub() -> None:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


class UniDepthV2Backend(DepthBackend):
    family = "unidepth_v2"

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float32",
                 resolution_level: int = DEFAULT_RESOLUTION_LEVEL) -> None:
        if model_name not in CHECKPOINTS:
            raise KeyError(f"unknown UniDepthV2 checkpoint: {model_name}")
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self._spec = CHECKPOINTS[model_name]
        self.resolution_level = int(resolution_level)
        self._model = None
        self._params = 0

    def load(self) -> None:
        if self._loaded:
            return
        _install_wandb_stub()
        from unidepth.models import UniDepthV2

        model = UniDepthV2.from_pretrained(self._spec["repo"])
        model.resolution_level = self.resolution_level  # type: ignore[assignment]
        self._params = int(sum(p.numel() for p in model.parameters()))
        self._model = model.to(self.device).eval()
        self._loaded = True

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend=self.family,
            model_name=self.model_name,
            checkpoint=self._spec["repo"],
            checkpoint_revision=None,
            convention=DepthConvention.METRIC_Z,
            provides_native_confidence=True,
            uses_intrinsics=True,
            native_input_size=None,   # chosen at runtime from resolution_level
            device=self.device,
            torch_dtype=self.dtype,
            parameter_count=self._params,
            library_versions=self._unidepth_versions(),
            notes=(self._spec["notes"]
                   + f"; resolution_level={self.resolution_level}"
                   + "; also returns euclidean range in extras['radius']"),
        )

    def _unidepth_versions(self) -> dict[str, str]:
        versions = self._library_versions()
        for mod in ("timm", "einops"):
            try:
                versions[mod] = str(__import__(mod).__version__)
            except Exception:  # pragma: no cover
                pass
        try:
            from importlib.metadata import version

            versions["unidepth"] = version("unidepth")
        except Exception:  # pragma: no cover
            pass
        return versions

    def infer_batch(
        self,
        images: Sequence[np.ndarray],
        intrinsics: Sequence[CameraIntrinsics],
    ) -> list[RawDepthOutput]:
        self._require_loaded()
        import torch

        assert self._model is not None
        self._check_uniform_shape(images)

        rgb = torch.from_numpy(
            np.stack([np.ascontiguousarray(img) for img in images]).transpose(0, 3, 1, 2)
        ).to(self.device)
        K = torch.from_numpy(
            np.stack([intr.matrix() for intr in intrinsics]).astype(np.float32)
        ).to(self.device)

        forward_batch = len(images) if SUPPORTS_BATCHED_FORWARD else 1
        collected: list[dict] = []
        with torch.inference_mode():
            for start in range(0, len(images), forward_batch):
                sl = slice(start, start + forward_batch)
                collected.append(self._model.infer(rgb[sl], K[sl]))

        def gather(key: str, channel_slice) -> np.ndarray | None:
            if key not in collected[0]:
                return None
            return np.concatenate(
                [c[key][channel_slice].float().cpu().numpy() for c in collected]
            ).astype(np.float32)

        depth = gather("depth", (slice(None), 0))
        assert depth is not None
        radius = gather("radius", (slice(None), 0))
        conf = gather("confidence", (slice(None), 0))
        self_k = gather("intrinsics", slice(None))

        outputs: list[RawDepthOutput] = []
        for i in range(depth.shape[0]):
            extras: dict = {
                "confidence_semantics": "raw UniDepthV2 confidence head, unnormalised; larger = more confident",
                "euclidean_range_available": radius is not None,
                "requested_batch_size": len(images),
                "forward_batch_size": forward_batch,
                "supports_batched_forward": SUPPORTS_BATCHED_FORWARD,
            }
            if radius is not None:
                extras["radius_m"] = radius[i]
            if self_k is not None:
                # Recorded as a model observation only. It is NOT the calibration.
                extras["model_self_estimated_K"] = self_k[i].tolist()
                extras["model_self_estimated_fx_px"] = float(self_k[i][0, 0])
                extras["supplied_fx_px"] = float(intrinsics[i].fx)
            outputs.append(
                RawDepthOutput(
                    depth=depth[i],
                    native_confidence=None if conf is None else conf[i],
                    extras=extras,
                )
            )
        return outputs


__all__ = ["UniDepthV2Backend", "CHECKPOINTS"]
