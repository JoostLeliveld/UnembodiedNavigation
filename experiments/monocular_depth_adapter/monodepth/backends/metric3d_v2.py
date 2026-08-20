"""Metric3D v2, loaded from the authors' torch.hub entry.

Metric3D is the one model here that genuinely consumes the intrinsics. It
predicts depth for a *canonical* camera of focal length 1000 px and then rescales
by ``fx_resized / 1000``, so the focal length you pass is a multiplicative factor
on every metre it reports. Pass the wrong fx and the whole map is wrong by that
ratio, silently.

The preprocessing below is the authors' own recipe from ``hubconf.py``:
aspect-preserving resize into a 616x1064 canvas, mean-value padding, ImageNet
normalisation, then unpad -> resize -> de-canonical rescale on the way out. It is
reproduced rather than imported because the hub entry only exposes it inside a
``__main__`` block.

Import shim: the hub checkout has one unguarded ``from mmcv.utils import
collect_env`` in ``mono/utils/comm.py``. mmcv proper needs a CUDA build for ops
this model never calls, so a three-symbol stub backed by mmengine is registered
instead. Everything else in the checkout already falls back to mmengine.

**CUDA only.** ``device="cpu"`` fails inside the RAFT decoder — its depth-bin
tensor is created with a hardcoded ``.cuda()``, so the expectation step hits
"found at least two devices, cuda:0 and cpu" no matter where the weights live
(measured 2026-08-11). The other two families run on CPU, slowly; this one has
no CPU fallback short of patching the vendored checkout.
"""

from __future__ import annotations

import sys
import types
from typing import Sequence

import numpy as np

from ..types import BackendInfo, CameraIntrinsics, DepthConvention
from .base import DepthBackend, RawDepthOutput

# ViT variants run at this canvas; the ConvNeXt variants use (544, 1216).
VIT_INPUT_SIZE = (616, 1064)
CANONICAL_FOCAL_PX = 1000.0
PAD_VALUE = (123.675, 116.28, 103.53)
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)

CHECKPOINTS: dict[str, dict] = {
    "metric3d_v2_vit_small": {
        "entrypoint": "metric3d_vit_small",
        "weights": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth",
        "input_size": VIT_INPUT_SIZE,
        "notes": "DINOv2-reg ViT-S backbone, RAFT-4iter head",
    },
    "metric3d_v2_vit_large": {
        "entrypoint": "metric3d_vit_large",
        "weights": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth",
        "input_size": VIT_INPUT_SIZE,
        "notes": "DINOv2-reg ViT-L backbone, RAFT-8iter head",
    },
}

HUB_REPO = "yvanyin/metric3d"


def _install_mmcv_shim() -> None:
    """Satisfy the checkout's one unguarded mmcv import using mmengine."""
    if "mmcv" in sys.modules:
        return
    import mmengine
    from mmengine.utils.dl_utils import collect_env

    mmcv = types.ModuleType("mmcv")
    utils = types.ModuleType("mmcv.utils")
    utils.collect_env = collect_env            # type: ignore[attr-defined]
    utils.get_git_hash = mmengine.utils.get_git_hash   # type: ignore[attr-defined]
    utils.Config = mmengine.Config             # type: ignore[attr-defined]
    utils.DictAction = mmengine.DictAction     # type: ignore[attr-defined]
    mmcv.utils = utils                         # type: ignore[attr-defined]
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.utils"] = utils


class Metric3DV2Backend(DepthBackend):
    family = "metric3d_v2"

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float32") -> None:
        if model_name not in CHECKPOINTS:
            raise KeyError(f"unknown Metric3D v2 checkpoint: {model_name}")
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self._spec = CHECKPOINTS[model_name]
        self._model = None
        self._params = 0

    def load(self) -> None:
        if self._loaded:
            return
        import torch

        _install_mmcv_shim()
        model = torch.hub.load(HUB_REPO, self._spec["entrypoint"], pretrain=True, trust_repo=True)
        self._params = int(sum(p.numel() for p in model.parameters()))  # type: ignore[attr-defined]
        self._model = model.to(self.device).eval()  # type: ignore[attr-defined]
        self._loaded = True

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend=self.family,
            model_name=self.model_name,
            checkpoint=f"torch.hub:{HUB_REPO}#{self._spec['entrypoint']} <- {self._spec['weights']}",
            checkpoint_revision=None,   # the hub entry is a branch zipball, not a pinned commit
            convention=DepthConvention.METRIC_Z,
            provides_native_confidence=True,
            uses_intrinsics=True,
            native_input_size=tuple(self._spec["input_size"]),
            device=self.device,
            torch_dtype=self.dtype,
            parameter_count=self._params,
            library_versions=self._metric3d_versions(),
            notes=self._spec["notes"] + "; metric scale is fx-dependent (canonical f=1000 px)",
        )

    def _metric3d_versions(self) -> dict[str, str]:
        versions = self._library_versions()
        for mod in ("timm", "mmengine"):
            try:
                versions[mod] = str(__import__(mod).__version__)
            except Exception:  # pragma: no cover
                pass
        return versions

    # ---------------------------------------------------------------- preprocess
    def _prepare(self, images: Sequence[np.ndarray]):
        import cv2
        import torch

        target_h, target_w = self._spec["input_size"]
        h, w = self._check_uniform_shape(images)
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        pad_h, pad_w = target_h - new_h, target_w - new_w
        top, left = pad_h // 2, pad_w // 2
        pad_info = (top, pad_h - top, left, pad_w - left)

        mean = torch.tensor(IMAGENET_MEAN).float()[:, None, None]
        std = torch.tensor(IMAGENET_STD).float()[:, None, None]

        tensors = []
        for img in images:
            resized = cv2.resize(np.ascontiguousarray(img), (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(resized, pad_info[0], pad_info[1], pad_info[2],
                                        pad_info[3], cv2.BORDER_CONSTANT, value=PAD_VALUE)
            t = torch.from_numpy(padded.transpose(2, 0, 1)).float()
            tensors.append((t - mean) / std)
        batch = torch.stack(tensors).to(self.device)
        return batch, pad_info, scale, (h, w)

    def infer_batch(
        self,
        images: Sequence[np.ndarray],
        intrinsics: Sequence[CameraIntrinsics],
    ) -> list[RawDepthOutput]:
        self._require_loaded()
        import torch
        import torch.nn.functional as F

        assert self._model is not None
        batch, pad_info, scale, (h, w) = self._prepare(images)

        with torch.inference_mode():
            pred_depth, confidence, _ = self._model.inference({"input": batch})

        def unpad_resize(t: "torch.Tensor") -> np.ndarray:
            t = t.float()
            if t.ndim == 3:
                t = t.unsqueeze(1)
            top, bottom, left, right = pad_info
            t = t[:, :, top: t.shape[2] - bottom, left: t.shape[3] - right]
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            return t.squeeze(1).detach().cpu().numpy().astype(np.float32)

        depth_canonical = unpad_resize(pred_depth)
        conf_np = unpad_resize(confidence) if confidence is not None else None

        outputs: list[RawDepthOutput] = []
        for i, intr in enumerate(intrinsics):
            # De-canonical rescale uses fx measured on the RESIZED image, per the
            # authors' recipe: the network saw the resized pixels, not ours.
            fx_resized = intr.fx * scale
            to_real = fx_resized / CANONICAL_FOCAL_PX
            outputs.append(
                RawDepthOutput(
                    depth=(depth_canonical[i] * to_real).astype(np.float32),
                    native_confidence=None if conf_np is None else conf_np[i],
                    extras={
                        "model_input_size": list(self._spec["input_size"]),
                        "resize_scale": float(scale),
                        "fx_resized_px": float(fx_resized),
                        "canonical_to_real_scale": float(to_real),
                        "confidence_semantics": "raw Metric3D confidence head, unnormalised; larger = more confident",
                    },
                )
            )
        return outputs


__all__ = ["Metric3DV2Backend", "CHECKPOINTS"]
