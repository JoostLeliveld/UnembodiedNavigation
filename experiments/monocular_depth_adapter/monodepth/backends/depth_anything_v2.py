"""Depth Anything V2, through the transformers depth-estimation head.

Two families of checkpoint, and they do NOT return the same kind of number:

- ``Depth-Anything-V2-Metric-*``  metric depth in metres along the optical axis.
  These are the relative model fine-tuned on Hypersim (indoor) or Virtual KITTI
  (outdoor) with a metric head.
- ``Depth-Anything-V2-<size>-hf`` the plain relative model. Its output is
  disparity-like: larger means nearer, and there is no scale. Calling that
  "depth" and comparing it against metres is the classic way to get a wrong
  answer that looks plausible, so it is tagged ``inverse_depth`` here and the
  adapter never converts it.

The model does not consume intrinsics at all. That is itself a benchmark result:
whatever it thinks the field of view is, is baked into the weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import BackendInfo, CameraIntrinsics, DepthConvention
from .base import DepthBackend, RawDepthOutput

# Registered checkpoints. `convention` is a property of the checkpoint, not of
# the code path, which is exactly why it is declared per entry.
CHECKPOINTS: dict[str, dict] = {
    "dav2_metric_indoor_small": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "convention": DepthConvention.METRIC_Z,
        "notes": "relative DA-V2 Small fine-tuned on Hypersim for indoor metric depth",
    },
    "dav2_metric_indoor_large": {
        "repo": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "convention": DepthConvention.METRIC_Z,
        "notes": "relative DA-V2 Large fine-tuned on Hypersim for indoor metric depth",
    },
    "dav2_relative_small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "convention": DepthConvention.INVERSE_DEPTH,
        "notes": "plain DA-V2 Small; output is unitless disparity, larger = nearer",
    },
}


class DepthAnythingV2Backend(DepthBackend):
    family = "depth_anything_v2"

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float32") -> None:
        if model_name not in CHECKPOINTS:
            raise KeyError(f"unknown Depth Anything V2 checkpoint: {model_name}")
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self._spec = CHECKPOINTS[model_name]
        self._model = None
        self._processor = None
        self._revision: str | None = None
        self._params = 0

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = self._spec["repo"]
        self._processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForDepthEstimation.from_pretrained(
            repo, dtype=getattr(torch, self.dtype)
        )
        self._revision = getattr(getattr(model, "config", None), "_commit_hash", None)
        self._params = int(sum(p.numel() for p in model.parameters()))
        self._model = model.to(self.device).eval()
        self._loaded = True

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend=self.family,
            model_name=self.model_name,
            checkpoint=self._spec["repo"],
            checkpoint_revision=self._revision,
            convention=self._spec["convention"],
            provides_native_confidence=False,
            uses_intrinsics=False,
            native_input_size=None,   # the processor picks a size from the input aspect
            device=self.device,
            torch_dtype=self.dtype,
            parameter_count=self._params,
            library_versions=self._transformers_versions(),
            notes=self._spec["notes"],
        )

    def _transformers_versions(self) -> dict[str, str]:
        versions = self._library_versions()
        try:
            import transformers

            versions["transformers"] = str(transformers.__version__)
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
        import torch.nn.functional as F

        h, w = self._check_uniform_shape(images)
        assert self._processor is not None and self._model is not None

        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model(**inputs)

        pred = out.predicted_depth          # (B, h', w') at the processor's size
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        # The processor resizes to a multiple of 14; put it back on the pixel grid
        # the caller handed us, so the mask and the intrinsics still line up.
        pred = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
        pred_np = pred.squeeze(1).detach().cpu().numpy().astype(np.float32)

        model_h, model_w = int(inputs["pixel_values"].shape[-2]), int(inputs["pixel_values"].shape[-1])
        return [
            RawDepthOutput(
                depth=pred_np[i],
                extras={"model_input_size": [model_h, model_w],
                        "resampled_to_input_grid": True},
            )
            for i in range(pred_np.shape[0])
        ]


__all__ = ["DepthAnythingV2Backend", "CHECKPOINTS"]
