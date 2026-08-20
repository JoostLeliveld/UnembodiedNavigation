"""Save predictions to disk and read them back unchanged.

One prediction is two files:

    <image_id>__<model_name>.npz    depth, valid, and any optional maps
    <image_id>__<model_name>.json   everything else, human-readable

The split is deliberate. The JSON sidecar is what you grep, diff, and paste into
a report; the npz is what you load. Storing the convention in the sidecar means
a stray depth array can never be silently reinterpreted as metres later — the
loader refuses to reconstruct a prediction without it.

``depth`` is stored as float32, unmodified. Dropping to float16 to save disk
would quietly cap the precision of every downstream comparison, so it is not
offered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import (
    BackendInfo,
    CameraIntrinsics,
    DepthConvention,
    DepthPrediction,
    InferenceTiming,
    MemoryRecord,
)

SCHEMA_VERSION = 1


def prediction_stem(prediction: DepthPrediction) -> str:
    return f"{prediction.image_id}__{prediction.model.model_name}"


def save_prediction(prediction: DepthPrediction, out_dir: str | Path) -> tuple[Path, Path]:
    """Write one prediction; returns the (npz, json) paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prediction_stem(prediction)

    arrays: dict[str, np.ndarray] = {
        "depth": prediction.depth.astype(np.float32),
        "valid": prediction.valid.astype(bool),
    }
    if prediction.uncertainty is not None:
        arrays["uncertainty"] = prediction.uncertainty.astype(np.float32)
    if prediction.native_confidence is not None:
        arrays["native_confidence"] = prediction.native_confidence.astype(np.float32)
    for key, value in prediction.extra_arrays.items():
        arrays[f"extra__{key}"] = np.asarray(value)

    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **arrays)

    meta = prediction.metadata()
    meta["schema_version"] = SCHEMA_VERSION
    meta["arrays"] = sorted(arrays)
    meta["npz_file"] = npz_path.name
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, json_path


def save_all(predictions: Iterable[DepthPrediction], out_dir: str | Path) -> list[Path]:
    return [save_prediction(p, out_dir)[0] for p in predictions]


def load_prediction(json_path: str | Path) -> DepthPrediction:
    """Rebuild a prediction from its sidecar plus npz."""
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{json_path.name}: schema version {meta.get('schema_version')} "
            f"!= {SCHEMA_VERSION}; the writer and reader disagree"
        )
    if "convention" not in meta:
        raise ValueError(f"{json_path.name}: no depth convention recorded, refusing to guess")

    with np.load(json_path.with_name(meta["npz_file"])) as data:
        arrays = {k: data[k] for k in data.files}

    model_meta = dict(meta["model"])
    model_meta["convention"] = DepthConvention(model_meta["convention"])
    native = model_meta.pop("native_input_size", None)
    model = BackendInfo(native_input_size=tuple(native) if native else None, **model_meta)

    timing_meta = dict(meta["timing"])
    timing_meta.pop("total_s", None)   # derived property, not a field

    return DepthPrediction(
        image_id=meta["image_id"],
        depth=arrays["depth"],
        convention=DepthConvention(meta["convention"]),
        valid=arrays["valid"].astype(bool),
        intrinsics=CameraIntrinsics(**meta["intrinsics"]),
        model=model,
        timing=InferenceTiming(**timing_meta),
        memory=MemoryRecord(**meta["memory"]),
        uncertainty=arrays.get("uncertainty"),
        uncertainty_kind=meta.get("uncertainty_kind"),
        uncertainty_detail=meta.get("uncertainty_detail", {}),
        native_confidence=arrays.get("native_confidence"),
        image_sha256=meta.get("image_sha256"),
        source_path=meta.get("source_path"),
        extras=meta.get("extras", {}),
        extra_arrays={k[len("extra__"):]: v for k, v in arrays.items() if k.startswith("extra__")},
    )


def load_dir(directory: str | Path) -> list[DepthPrediction]:
    """Load every prediction in a directory, ordered by filename."""
    return [load_prediction(p) for p in sorted(Path(directory).glob("*.json"))
            if p.name not in ("run_manifest.json", "benchmark.json")]


__all__ = ["SCHEMA_VERSION", "prediction_stem", "save_prediction", "save_all",
           "load_prediction", "load_dir"]
