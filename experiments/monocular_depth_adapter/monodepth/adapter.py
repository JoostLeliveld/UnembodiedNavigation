"""The one entry point: RGB + intrinsics in, depth + metadata out.

    adapter = MonocularDepthAdapter("unidepth_v2_vits14", uncertainty="native+flip")
    adapter.load()
    predictions = adapter.predict([DepthRequest(...), ...])

What it does
    picks a backend, runs it in batches, records how long that took and how much
    memory it cost, marks the pixels it cannot vouch for, and attaches an
    uncertainty map if asked.

What it will not do, by construction rather than by convention
    look at ground truth or simulator depth, anchor anything to the floor,
    decide whether a pixel is "visible", change a calibration, or convert a
    unitless prediction into metres. The backend signature only accepts an image
    and intrinsics, so none of those inputs are reachable from in here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import determinism, uncertainty as unc
from .backends import DepthBackend, create_backend, family_of
from .types import (
    BackendInfo,
    DepthPrediction,
    DepthRequest,
    InferenceTiming,
    MemoryRecord,
    build_valid_mask,
)

#: Accepted values for the `uncertainty` argument.
UNCERTAINTY_MODES = ("none", "native", "flip", "native+flip")


@dataclass
class _BatchCost:
    forward_s: float
    gpu_peak_allocated_mib: float
    gpu_peak_reserved_mib: float
    host_rss_mib: float


def _host_rss_mib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:  # pragma: no cover - psutil is present in this environment
        return float("nan")


class MonocularDepthAdapter:
    """Uniform driver for one monocular depth model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "float32",
        batch_size: int = 1,
        uncertainty: str = "native",
        seed: int = determinism.DEFAULT_SEED,
        max_metric_m: float = 300.0,
        min_metric_m: float = 0.0,
        backend_kwargs: dict | None = None,
    ) -> None:
        if uncertainty not in UNCERTAINTY_MODES:
            raise ValueError(f"uncertainty must be one of {UNCERTAINTY_MODES}, got {uncertainty!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.family = family_of(model_name)
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.uncertainty_mode = uncertainty
        self.seed = seed
        self.max_metric_m = float(max_metric_m)
        self.min_metric_m = float(min_metric_m)
        self._backend: DepthBackend | None = None
        self._backend_kwargs = dict(backend_kwargs or {})
        self._weights_mib = 0.0
        self._determinism_config: dict = {}
        #: batch sizes that had to be reduced after an out-of-memory error
        self.oom_events: list[dict] = []

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> "MonocularDepthAdapter":
        if self._backend is not None and self._backend.is_loaded:
            return self
        self._determinism_config = determinism.set_deterministic(self.seed)
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA device is available")

        before = 0
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()

        self._backend = create_backend(
            self.model_name, device=self.device, dtype=self.dtype, **self._backend_kwargs
        )
        self._backend.load()

        if self.device.startswith("cuda"):
            self._weights_mib = (torch.cuda.memory_allocated() - before) / 2 ** 20
        return self

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._backend = None

    def __enter__(self) -> "MonocularDepthAdapter":
        return self.load()

    def __exit__(self, *exc) -> None:
        self.unload()

    @property
    def info(self) -> BackendInfo:
        if self._backend is None or not self._backend.is_loaded:
            raise RuntimeError("call load() first")
        return self._backend.info()

    @property
    def determinism_config(self) -> dict:
        return dict(self._determinism_config)

    # ------------------------------------------------------------------ inference
    def predict_one(self, request: DepthRequest) -> DepthPrediction:
        return self.predict([request])[0]

    def predict(self, requests: Iterable[DepthRequest]) -> list[DepthPrediction]:
        """Run the model over any number of frames, batching by image size.

        Frames of different sizes cannot share a batch, so they are grouped and
        each group is chunked to ``batch_size``. Order is preserved.
        """
        if self._backend is None or not self._backend.is_loaded:
            raise RuntimeError("call load() before predict()")
        reqs = list(requests)
        if not reqs:
            return []

        results: list[DepthPrediction | None] = [None] * len(reqs)
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, req in enumerate(reqs):
            groups.setdefault(req.image.shape[:2], []).append(idx)  # type: ignore[arg-type]

        for _shape, indices in groups.items():
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                for offset, pred in zip(chunk, self._run_chunk([reqs[i] for i in chunk])):
                    results[offset] = pred

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:  # pragma: no cover - defensive
            raise RuntimeError(f"no prediction produced for request indices {missing}")
        return [r for r in results if r is not None]

    # -------------------------------------------------------------------- internal
    def _run_chunk(self, chunk: Sequence[DepthRequest]) -> list[DepthPrediction]:
        """Infer one same-size chunk, halving the batch if the GPU runs out."""
        try:
            return self._infer_chunk(chunk)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is OOM
            if not self._is_oom(exc) or len(chunk) == 1:
                raise
            half = max(1, len(chunk) // 2)
            self.oom_events.append({
                "attempted_batch_size": len(chunk),
                "fell_back_to": half,
                "error": str(exc)[:200],
            })
            self._free_cuda()
            out: list[DepthPrediction] = []
            for start in range(0, len(chunk), half):
                out.extend(self._run_chunk(chunk[start:start + half]))
            return out

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _free_cuda(self) -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _infer_chunk(self, chunk: Sequence[DepthRequest]) -> list[DepthPrediction]:
        import torch

        assert self._backend is not None
        images = [r.image for r in chunk]
        intrinsics = [r.intrinsics for r in chunk]

        t0 = time.perf_counter()
        images = [np.ascontiguousarray(img) for img in images]
        t_pre = time.perf_counter() - t0

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        raw = self._backend.infer_batch(images, intrinsics)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_s = time.perf_counter() - t0

        cost = _BatchCost(
            forward_s=forward_s,
            gpu_peak_allocated_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                    if self.device.startswith("cuda") else 0.0),
            gpu_peak_reserved_mib=(torch.cuda.max_memory_reserved() / 2 ** 20
                                   if self.device.startswith("cuda") else 0.0),
            host_rss_mib=_host_rss_mib(),
        )

        flip_maps: list[np.ndarray] | None = None
        uncertainty_s = 0.0
        if "flip" in self.uncertainty_mode:
            t0 = time.perf_counter()
            flip_maps = self._flipped_pass(chunk)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            uncertainty_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        info = self._backend.info()
        predictions: list[DepthPrediction] = []
        for i, (req, out) in enumerate(zip(chunk, raw)):
            depth = np.asarray(out.depth, dtype=np.float32)
            valid = build_valid_mask(
                depth, info.convention,
                extra_invalid=out.invalid,
                min_metric_m=self.min_metric_m,
                max_metric_m=self.max_metric_m,
            )
            scalars, arrays = _split_extras(out.extras)
            uncertainty_map, kind, detail = self._assemble_uncertainty(
                depth, valid, info, out.native_confidence,
                None if flip_maps is None else flip_maps[i],
            )
            predictions.append(DepthPrediction(
                image_id=req.image_id,
                depth=depth,
                convention=info.convention,
                valid=valid,
                intrinsics=req.intrinsics,
                model=info,
                timing=InferenceTiming(
                    preprocess_s=t_pre / len(chunk),
                    forward_s=cost.forward_s / len(chunk),
                    postprocess_s=0.0,   # replaced below, once the loop is timed
                    uncertainty_s=uncertainty_s / len(chunk),
                    batch_size=len(chunk),
                ),
                memory=MemoryRecord(
                    gpu_peak_allocated_mib=cost.gpu_peak_allocated_mib,
                    gpu_peak_reserved_mib=cost.gpu_peak_reserved_mib,
                    host_rss_mib=cost.host_rss_mib,
                    weights_mib=self._weights_mib,
                ),
                uncertainty=uncertainty_map,
                uncertainty_kind=kind,
                uncertainty_detail=detail,
                native_confidence=out.native_confidence,
                image_sha256=req.image_sha256,
                source_path=req.source_path,
                extras=scalars,
                extra_arrays=arrays,
            ))
        post_s = (time.perf_counter() - t0) / len(chunk)
        for p in predictions:
            p.timing = InferenceTiming(
                preprocess_s=p.timing.preprocess_s,
                forward_s=p.timing.forward_s,
                postprocess_s=post_s,
                uncertainty_s=p.timing.uncertainty_s,
                batch_size=p.timing.batch_size,
            )
        return predictions

    def _flipped_pass(self, chunk: Sequence[DepthRequest]) -> list[np.ndarray]:
        """Second forward pass on the mirrored images, unflipped on the way back."""
        assert self._backend is not None
        images = [np.ascontiguousarray(r.image[:, ::-1, :]) for r in chunk]
        intrinsics = [r.intrinsics.mirrored_horizontally() for r in chunk]
        raw = self._backend.infer_batch(images, intrinsics)
        return [np.ascontiguousarray(np.asarray(o.depth, dtype=np.float32)[:, ::-1]) for o in raw]

    def _assemble_uncertainty(
        self,
        depth: np.ndarray,
        valid: np.ndarray,
        info: BackendInfo,
        native_confidence: np.ndarray | None,
        flipped: np.ndarray | None,
    ) -> tuple[np.ndarray | None, str | None, dict]:
        """Pick the uncertainty map according to the configured mode.

        ``native+flip`` prefers the flip spread as the returned map, because it
        is in the same units as the depth and therefore usable, and keeps the
        model's own confidence alongside in ``native_confidence``. Combining the
        two into one number would need a calibration this adapter has no basis
        for.
        """
        mode = self.uncertainty_mode
        if mode == "none":
            return None, None, {}

        if flipped is not None:
            spread, detail = unc.flip_consistency(depth, flipped, valid, info.convention)
            detail["native_confidence_also_available"] = native_confidence is not None
            return spread, unc.FLIP, detail

        if "native" in mode and native_confidence is not None:
            return (np.asarray(native_confidence, dtype=np.float32), unc.NATIVE,
                    {"method": unc.NATIVE,
                     "semantics": "raw model confidence, unnormalised; larger = more confident",
                     "note": "not an interval; needs calibration before it means anything in metres"})

        return None, None, {"requested": mode, "available": False,
                            "reason": f"{info.model_name} has no native confidence head"}


def _split_extras(extras) -> tuple[dict, dict]:
    """Separate JSON-safe scalars from arrays so metadata stays serialisable."""
    scalars: dict = {}
    arrays: dict = {}
    for key, value in dict(extras or {}).items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            scalars[key] = value
    return scalars, arrays


__all__ = ["MonocularDepthAdapter", "UNCERTAINTY_MODES"]
