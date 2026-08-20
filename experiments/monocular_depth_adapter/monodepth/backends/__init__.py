"""Registry of available depth backends.

Backend modules are imported lazily: loading the Metric3D module drags in
mmengine and a torch.hub checkout, and UniDepth pulls its own package tree.
Neither should be a cost paid by someone who only wants Depth Anything.
"""

from __future__ import annotations

from typing import Callable

from .base import DepthBackend, RawDepthOutput

# model_name -> (family, factory). Factories are thunks so nothing heavy imports
# until a model is actually requested.
_FAMILY_OF: dict[str, str] = {}
_FACTORIES: dict[str, Callable[..., DepthBackend]] = {}


def _register_depth_anything() -> None:
    from .depth_anything_v2 import CHECKPOINTS, DepthAnythingV2Backend

    for name in CHECKPOINTS:
        _FAMILY_OF[name] = "depth_anything_v2"
        _FACTORIES[name] = (
            lambda name=name, **kw: DepthAnythingV2Backend(name, **kw)
        )


def _register_metric3d() -> None:
    from .metric3d_v2 import CHECKPOINTS, Metric3DV2Backend

    for name in CHECKPOINTS:
        _FAMILY_OF[name] = "metric3d_v2"
        _FACTORIES[name] = (
            lambda name=name, **kw: Metric3DV2Backend(name, **kw)
        )


def _register_unidepth() -> None:
    from .unidepth_v2 import CHECKPOINTS, UniDepthV2Backend

    for name in CHECKPOINTS:
        _FAMILY_OF[name] = "unidepth_v2"
        _FACTORIES[name] = (
            lambda name=name, **kw: UniDepthV2Backend(name, **kw)
        )


#: Declared without importing anything, so `available_models()` is cheap and
#: works even when one model family's dependencies are missing.
MODEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "depth_anything_v2": ("dav2_metric_indoor_small", "dav2_metric_indoor_large",
                          "dav2_relative_small"),
    "metric3d_v2": ("metric3d_v2_vit_small", "metric3d_v2_vit_large"),
    "unidepth_v2": ("unidepth_v2_vits14", "unidepth_v2_vitl14"),
}

_REGISTRARS = {
    "depth_anything_v2": _register_depth_anything,
    "metric3d_v2": _register_metric3d,
    "unidepth_v2": _register_unidepth,
}


def available_models() -> list[str]:
    """Every registered model name, without importing any model code."""
    return [name for names in MODEL_FAMILIES.values() for name in names]


def family_of(model_name: str) -> str:
    for family, names in MODEL_FAMILIES.items():
        if model_name in names:
            return family
    raise KeyError(f"unknown depth model: {model_name!r}. Known: {available_models()}")


def create_backend(model_name: str, **kwargs) -> DepthBackend:
    """Instantiate a backend by registry name. Does not load weights."""
    family = family_of(model_name)
    if model_name not in _FACTORIES:
        _REGISTRARS[family]()
    return _FACTORIES[model_name](**kwargs)


__all__ = [
    "DepthBackend",
    "RawDepthOutput",
    "MODEL_FAMILIES",
    "available_models",
    "family_of",
    "create_backend",
]
