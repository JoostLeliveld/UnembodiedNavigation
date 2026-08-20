"""Reading agent-1 frame records and agent-2 predictions, and writing results.

The boundary between the method and the evaluation oracle is enforced here
rather than by convention. A frame record from the dynamic-world generator
carries both method inputs and oracle fields; this loader consumes only the
whitelist and *refuses* the rest, so "the method never saw the oracle" is a
property of the code, not a promise in a README.

Agent-1 frame record (the full emitted contract)::

    scenario_id, timestamp, camera_id, rgb_path, oracle_depth_path,
    camera_intrinsics, camera_extrinsics, obstacle_state, oracle_visibility_grid

Of those, exactly six are method-visible. ``oracle_depth_path``,
``obstacle_state`` and ``oracle_visibility_grid`` are evaluation-only and
raise if requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    CameraCalibration,
    ContractViolation,
    DepthConvention,
    DepthPrediction,
    VisibilityResult,
)

#: keys of an agent-1 frame record the method is allowed to read. Identifiers
#: and calibration only -- nothing that describes what is actually in the world.
METHOD_VISIBLE_KEYS = frozenset(
    {
        "scenario_id",
        "timestamp",
        "camera_id",
        "frame_id",
        "rgb_path",
        "camera_intrinsics",
        "camera_extrinsics",
    }
)

#: keys that exist for scoring only; reading one here is a method violation
ORACLE_ONLY_KEYS = frozenset(
    {"oracle_depth_path", "oracle_visibility_grid", "obstacle_state", "oracle_depth", "gt_pose"}
)


class OracleAccessError(PermissionError):
    """The method tried to read an evaluation-only field."""


def method_visible_record(record: Mapping[str, Any]) -> dict:
    """Strip a frame record down to the fields the method may use.

    Returns the whitelisted subset plus a ``_withheld`` list naming what was
    dropped, so a run log can show that the oracle fields were present and
    deliberately not consumed.
    """
    if not isinstance(record, Mapping):
        raise ContractViolation(f"frame record must be a mapping, got {type(record).__name__}")
    missing = {"camera_id", "camera_intrinsics", "camera_extrinsics"} - set(record)
    if missing:
        raise ContractViolation(f"frame record is missing required keys: {sorted(missing)}")
    visible = {k: record[k] for k in sorted(METHOD_VISIBLE_KEYS) if k in record}
    visible["_withheld"] = sorted(set(record) - METHOD_VISIBLE_KEYS)
    return visible


def assert_no_oracle_access(keys: Any) -> None:
    """Raise if any evaluation-only key is about to be read by the method."""
    requested = {keys} if isinstance(keys, str) else set(keys)
    leaked = sorted(requested & ORACLE_ONLY_KEYS)
    if leaked:
        raise OracleAccessError(
            f"{leaked} are evaluation-only fields. The ground-anchoring method must infer "
            "visibility from RGB-derived depth, calibration and the drivable map alone; "
            "score against the oracle outside this package."
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _first(block: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """First present key among ``names`` -- the peer agents spell some alike."""
    for name in names:
        if name in block:
            return block[name]
    return default


def camera_from_record(record: Mapping[str, Any]) -> CameraCalibration:
    """Build a :class:`CameraCalibration` from an agent-1 frame record."""
    visible = method_visible_record(record)
    return calibration_from_parts(
        visible["camera_intrinsics"],
        visible["camera_extrinsics"],
        camera_id=str(visible.get("camera_id", "camera")),
    )


def calibration_from_parts(
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    camera_id: str = "camera",
) -> CameraCalibration:
    """Parse the two calibration blocks. Unknown shapes fail loudly.

    Intrinsics: ``{fx, fy, cx, cy}`` or ``{K}``, plus an image size given as
    ``width``/``height``, ``img_width``/``img_height`` or
    ``image_width``/``image_height`` -- the dynamic-world generator emits the
    ``img_*`` spelling and the depth adapter emits the plain one.

    Extrinsics, any one of:
      * ``{R_world_to_cam (3x3), t_world_to_cam (3)}``  (canonical)
      * ``{R_world_to_cam (3x3), camera_position (3)}``
      * ``{T_cam_world (4x4)}``  -- the world->camera homogeneous transform
      * ``{cam_pos | camera_position (3), look_at (3), up (3, optional)}``
    """
    width = int(_first(intrinsics, "width", "img_width", "image_width", default=0))
    height = int(_first(intrinsics, "height", "img_height", "image_height", default=0))
    if "K" in intrinsics:
        K = np.asarray(intrinsics["K"], dtype=float).reshape(3, 3)
    elif {"fx", "fy", "cx", "cy"} <= set(intrinsics):
        K = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ]
        )
    else:
        raise ContractViolation(
            f"unrecognised intrinsics block with keys {sorted(intrinsics)}; expected 'K' or "
            "fx/fy/cx/cy"
        )
    if width <= 0 or height <= 0:
        raise ContractViolation("intrinsics must declare a positive image width and height")

    if "T_cam_world" in extrinsics:
        T = np.asarray(extrinsics["T_cam_world"], dtype=float).reshape(4, 4)
        R = T[:3, :3]
        cam_pos = -R.T @ T[:3, 3]
    elif "R_world_to_cam" in extrinsics:
        R = np.asarray(extrinsics["R_world_to_cam"], dtype=float).reshape(3, 3)
        if "camera_position" in extrinsics:
            cam_pos = np.asarray(extrinsics["camera_position"], dtype=float).reshape(3)
        elif "t_world_to_cam" in extrinsics:
            cam_pos = -R.T @ np.asarray(extrinsics["t_world_to_cam"], dtype=float).reshape(3)
        else:
            raise ContractViolation(
                "extrinsics gave a rotation but no translation ('t_world_to_cam' or "
                "'camera_position')"
            )
    elif "look_at" in extrinsics and _first(extrinsics, "cam_pos", "camera_position") is not None:
        cam_pos = np.asarray(
            _first(extrinsics, "cam_pos", "camera_position"), dtype=float
        ).reshape(3)
        look_at = np.asarray(extrinsics["look_at"], dtype=float).reshape(3)
        up = np.asarray(extrinsics.get("up", (0.0, 0.0, 1.0)), dtype=float).reshape(3)
        R = _look_at_rotation(cam_pos, look_at, up)
    else:
        raise ContractViolation(
            f"unrecognised extrinsics block with keys {sorted(extrinsics)}; see "
            "calibration_from_parts for the accepted forms"
        )
    return CameraCalibration(K=K, R=R, cam_pos=cam_pos, width=width, height=height,
                             camera_id=camera_id)


def _look_at_rotation(cam_pos: np.ndarray, look_at: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """World->camera rotation, matching ``ObliqueCameraModel._compute_lookat_rotation``."""
    z_cam = look_at - cam_pos
    n = np.linalg.norm(z_cam)
    if n < 1e-9:
        raise ContractViolation("look_at coincides with the camera position")
    z_cam = z_cam / n
    x_cam = np.cross(z_cam, up_hint)
    nx = np.linalg.norm(x_cam)
    if nx < 1e-9:
        raise ContractViolation("look_at is parallel to the up hint; rotation is undefined")
    x_cam = x_cam / nx
    y_cam = np.cross(z_cam, x_cam)
    return np.array([x_cam, y_cam / np.linalg.norm(y_cam), z_cam])


# ---------------------------------------------------------------------------
# Agent-2 predictions
# ---------------------------------------------------------------------------
def load_prediction(path: str | Path) -> DepthPrediction:
    """Load a depth-adapter prediction. Pass either its ``.json`` or ``.npz``.

    Reads the depth adapter's on-disk schema -- an npz of ``depth`` / ``valid``
    (plus optional ``uncertainty``) beside a JSON sidecar carrying the
    convention, model identity and timing -- and also accepts a bare npz whose
    arrays are named ``values`` / ``valid_mask``.

    ``convention`` is mandatory. A prediction that does not say how to read its
    numbers is not usable, and guessing is precisely the failure this method
    refuses to make.
    """
    path = Path(path)
    npz_path, meta = _prediction_sources(path)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    values = _first(arrays, "depth", "values")
    if values is None:
        raise ContractViolation(
            f"{npz_path.name} has no 'depth' or 'values' array (found {sorted(arrays)})"
        )
    if "convention" not in meta:
        raise ContractViolation(
            f"{path.name} does not declare a depth convention; the adapter must record it "
            "in the JSON sidecar"
        )

    raw_model, raw_timing = meta.get("model"), meta.get("timing")
    model: Mapping[str, Any] = raw_model if isinstance(raw_model, Mapping) else {}
    timing: Mapping[str, Any] = raw_timing if isinstance(raw_timing, Mapping) else {}
    return DepthPrediction(
        values=values,
        convention=DepthConvention.parse(str(meta["convention"])),
        valid_mask=_first(arrays, "valid", "valid_mask"),
        uncertainty=arrays.get("uncertainty"),
        uncertainty_kind=meta.get("uncertainty_kind"),
        native_confidence=arrays.get("native_confidence"),
        model_name=str(_first(meta, "model_name", default=model.get("model_name", "unknown"))),
        checkpoint=str(_first(meta, "checkpoint", default=model.get("checkpoint", "unknown"))),
        inference_time_s=float(
            _first(meta, "inference_time_s", default=timing.get("total_s", float("nan")))
        ),
        frame_id=str(_first(meta, "frame_id", "image_id", default="")),
        camera_id=str(meta.get("camera_id", "")),
    )


def _prediction_sources(path: Path) -> tuple[Path, dict]:
    """Resolve a prediction path to ``(npz_path, sidecar_metadata)``."""
    if path.suffix == ".json":
        meta = json.loads(path.read_text())
        npz_path = path.with_name(str(meta.get("npz_file", path.with_suffix(".npz").name)))
        if not npz_path.is_file():
            raise ContractViolation(f"{path.name} points at missing array file {npz_path.name}")
        return npz_path, meta
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    return path, meta


def prediction_index(directory: str | Path) -> dict[str, list[Path]]:
    """Map each prediction's image id to its sidecars, for joining to frames.

    The adapter names files ``<image_id>__<model_name>.json``; several models
    may have run on the same image, so the value is a list.
    """
    index: dict[str, list[Path]] = {}
    for sidecar in sorted(Path(directory).glob("*.json")):
        if sidecar.name in ("run_manifest.json", "benchmark.json"):
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        image_id = str(meta.get("image_id") or meta.get("frame_id") or sidecar.stem.split("__")[0])
        index.setdefault(image_id, []).append(sidecar)
    return index


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------
def save_result(result: VisibilityResult, out_dir: str | Path, stem: str | None = None) -> Path:
    """Write the output contract: one ``.npz`` of grids plus a ``.json`` summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{result.camera_id or 'camera'}_{result.frame_id or 'frame'}"
    npz_path = out_dir / f"{stem}.npz"
    f = result.visibility
    np.savez_compressed(
        npz_path,
        xs=f.xs,
        ys=f.ys,
        p_visible=f.p_visible,
        p_occluded=f.p_occluded,
        p_unknown=f.p_unknown,
        p_los=f.p_los,
        unknown_mask=f.unknown_mask,
        in_fov=f.in_fov,
        height_map_m=f.height_map_m,
        height_sigma_m=f.height_sigma_m,
        observed=f.observed,
        depth_m=result.metric_depth.depth_m.astype(np.float32),
        depth_sigma_m=result.metric_depth.sigma_m.astype(np.float32),
        depth_valid=result.metric_depth.valid,
    )
    (out_dir / f"{stem}.json").write_text(json.dumps(result.summary(), indent=2, default=str))
    return npz_path
