"""Shared loading for the operational-residual study.

Reads ONLY operational streams: ``raw/experiment.csv`` (odometry + its covariance)
and ``raw/camera_*_perception.csv`` (per-camera detections). Ground truth lives in
``evaluation_only/`` and is loaded by a separate, explicitly named function that no
inference path calls.

Measurements are re-projected from ``obs_u/obs_v`` through
``reliability.projection._project_pixel_to_world`` with the **deployed**
along-bearing calibration, exactly as ``experiments/external_camera_bias_model``
does, so the operational and oracle residuals are computed on identical
measurements and the comparison isolates the *reference*, not the projection.
"""

from __future__ import annotations

import pathlib

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "state"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools"))

from reliability.projection import (
    camera_model_from_world,
)

# The projection corrections below were deleted from `reliability.projection` on 2026-08-07
# (measured harmful: e7). This study is ABOUT them, so it reads the graveyard copy instead.
import importlib.util as _ilu
_lpc = _ilu.spec_from_file_location(
    "legacy_projection_corrections",
    str(pathlib.Path(__file__).resolve().parents[1] / "legacy_projection_corrections.py"),
)
legacy_projection = _ilu.module_from_spec(_lpc)
_lpc.loader.exec_module(legacy_projection)
load_projection_calibration = legacy_projection.load_projection_calibration
projection_kwargs_for_camera = legacy_projection.projection_kwargs_for_camera
_project_pixel_to_world = legacy_projection.project_pixel_to_world

import attach_evaluation_truth as AET  # noqa: E402  (canonical nearest-stamp join)
from state.core import trajectory_smoother as ts  # noqa: E402


# Shared with experiments/external_camera_bias_model (same world, same deployment).
WORLD_SDF = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DEPLOYED_CALIB = (
    REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2"
    / "projection_calibration.json"
)
#: Override the projection calibration for an A/B, e.g. the gated 2-DOF v3:
#:     RCOND_PROJECTION_CALIBRATION=logs/.../projection_calibration_v3/projection_calibration.json
#: The path is recorded in every output so a run can never be mistaken for the
#: other arm. Empty/unset keeps the deployed v2.
CALIBRATION_OVERRIDE_ENV = "RCOND_PROJECTION_CALIBRATION"
OUT_ROOT = REPO / "logs/studies/operational_residual_rcond"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
CONTACT_Z_M = 0.05
TRUTH_TOL_S = 0.05

#: Camera-to-odometry association tolerance. Measured median offset is one 50 Hz
#: odometry tick (0-10 ms) with a worst case of 390 ms; 0.15 s matches the exp5
#: anchor tolerance and covers the bulk without bridging a real gap.
ASSOC_TOL_S = 0.15

CAPTURES = {
    "smoke1_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716",
    "smoke2_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716",
    "fusion_handover_20260721": REPO
    / "logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data",
}


@dataclass(frozen=True)
class Detection:
    """One detected observation, operational fields only."""

    capture: str
    camera: str
    stamp: float
    u: float
    v: float
    world: tuple[float, float]
    range_m: float


@dataclass(frozen=True)
class OperationalCapture:
    """Odometry track plus per-camera detections, no truth anywhere."""

    name: str
    stamps: np.ndarray
    odom: np.ndarray
    odom_cov: np.ndarray
    detections: dict[str, list[Detection]]

    @property
    def n_steps(self) -> int:
        return int(self.stamps.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.stamps[-1] - self.stamps[0]) if self.n_steps > 1 else 0.0


def _float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def camera_models():
    return {
        cam: camera_model_from_world(WORLD_SDF, include_name=MODEL_INCLUDES[cam])
        for cam in CAMERAS
    }


def calibration_path() -> Path:
    """The projection calibration in force, honouring the A/B override."""

    import os

    override = os.environ.get(CALIBRATION_OVERRIDE_ENV, "").strip()
    if not override:
        return DEPLOYED_CALIB
    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = REPO / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"{CALIBRATION_OVERRIDE_ENV}={override} is not a file")
    return candidate


def deployed_calibration():
    return load_projection_calibration(calibration_path())


def load_operational_capture(name: str, models=None, calib=None) -> OperationalCapture:
    """Odometry + per-camera detections for one capture. Never touches truth."""
    root = CAPTURES[name]
    models = models if models is not None else camera_models()
    calib = calib if calib is not None else deployed_calibration()

    stamps, odom, cov = [], [], []
    with (root / "raw" / "experiment.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = _float(row, "stamp")
            x, y = _float(row, "odom_noisy_x"), _float(row, "odom_noisy_y")
            if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
                continue
            cxx = _float(row, "odom_noisy_cov_xx")
            cxy = _float(row, "odom_noisy_cov_xy")
            cyy = _float(row, "odom_noisy_cov_yy")
            stamps.append(t)
            odom.append((x, y))
            cov.append(((cxx, cxy), (cxy, cyy)))

    order = np.argsort(np.asarray(stamps, dtype=float))
    stamps_arr = np.asarray(stamps, dtype=float)[order]
    odom_arr = np.asarray(odom, dtype=float)[order]
    cov_arr = np.asarray(cov, dtype=float)[order]

    detections: dict[str, list[Detection]] = {}
    for cam in CAMERAS:
        src = root / "raw" / f"{cam}_perception.csv"
        found: list[Detection] = []
        if src.exists():
            cam_x = float(models[cam].cam_pos[0])
            cam_y = float(models[cam].cam_pos[1])
            # projection_kwargs_for_camera is THE mapping from a loaded calibration
            # onto the projection signature. Picking keys out of the dict by hand
            # here would have silently kept this path at one along-bearing degree of
            # freedom when the library gained the cross-bearing term.
            projection_kwargs = projection_kwargs_for_camera(
                calib, cam, contact_z_m=CONTACT_Z_M
            )
            with src.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("detected") != "1":
                        continue
                    u, v, t = _float(row, "obs_u"), _float(row, "obs_v"), _float(row, "diag_stamp")
                    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(t)):
                        continue
                    point = _project_pixel_to_world(u, v, models[cam], **projection_kwargs)
                    if point is None or not all(math.isfinite(c) for c in point[:2]):
                        continue
                    found.append(
                        Detection(
                            capture=name, camera=cam, stamp=t, u=u, v=v,
                            world=(float(point[0]), float(point[1])),
                            range_m=float(math.hypot(point[0] - cam_x, point[1] - cam_y)),
                        )
                    )
        detections[cam] = found

    return OperationalCapture(
        name=name, stamps=stamps_arr, odom=odom_arr, odom_cov=cov_arr, detections=detections
    )


def associate(capture: OperationalCapture, detection: Detection, tol_s: float = ASSOC_TOL_S):
    """Nearest odometry index for a detection stamp, or None outside ``tol_s``."""
    idx = int(np.argmin(np.abs(capture.stamps - detection.stamp)))
    if abs(float(capture.stamps[idx]) - detection.stamp) > tol_s:
        return None
    return idx


def measurements_for(
    capture: OperationalCapture,
    r_assumed: dict[str, float] | float,
    tol_s: float = ASSOC_TOL_S,
) -> tuple[list, dict[str, int]]:
    """Smoother measurements from every associable detection.

    ``r_assumed`` is the *anchor* covariance the smoother assumes (per-axis std, m).
    It is an input to the trajectory, deliberately separate from the ``R_cond`` this
    study estimates -- reusing the estimate here would close a feedback loop the
    one-pass method is not entitled to (that is the M9 alternating scheme, gated
    behind everything else).
    """
    out, dropped = [], {cam: 0 for cam in CAMERAS}
    for cam, detections in capture.detections.items():
        std = float(r_assumed[cam]) if isinstance(r_assumed, dict) else float(r_assumed)
        var = std**2
        for det in detections:
            idx = associate(capture, det, tol_s)
            if idx is None:
                dropped[cam] += 1
                continue
            out.append(
                ts.Measurement(
                    index=idx, z=det.world,
                    covariance=((var, 0.0), (0.0, var)), source=cam,
                )
            )
    return out, dropped


# --------------------------------------------------------------------------- #
# EVALUATION ONLY below this line
# --------------------------------------------------------------------------- #


def load_truth_table(name: str):
    """Nearest-stamp ground-truth table. **EVALUATION ONLY.**

    Named so that any call site is visibly a scoring path. Never called from
    smoothing, residual construction, or covariance estimation.
    """
    path = CAPTURES[name] / "evaluation_only" / "ground_truth.csv"
    stamps, poses = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = _float(row, "stamp")
            x, y = _float(row, "gt_x"), _float(row, "gt_y")
            if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
                continue
            stamps.append(t)
            poses.append((x, y, _float(row, "gt_yaw")))
    paired = sorted(zip(stamps, poses))
    return [p[0] for p in paired], [p[1] for p in paired]


def truth_at(table, stamp: float, tol_s: float = TRUTH_TOL_S):
    """**EVALUATION ONLY.** Canonical nearest-stamp join."""
    return AET._nearest(table[0], table[1], stamp, tol_s)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
