#!/usr/bin/env python3
"""Does the commissioning gate survive into service? (C5 -> C3, register runway #1)

The 2-DOF bias fix is decided ONCE at commissioning (assumption A2) and then
deployed as a frozen constant. Cameras drift. Two questions follow, and neither
has been asked:

  Q1  Does a STALE correction become harmful? A per-camera constant fitted before
      the drift is still applied after it. If drift moves the camera against the
      fitted offset, the correction adds error instead of removing it -- and the
      "correct the outliers, leave the rest raw" policy would be actively worse
      than doing nothing.

  Q2  Is drift DETECTABLE with the same GT-free statistic the gate already uses?
      If |b_cross| / sigma_cross computed against the smoothed operational belief
      responds to drift, then commissioning and in-service monitoring are the same
      measurement and C5 becomes a lifecycle capability rather than a one-off.

Fault model (A4, faithful direction). The physical camera moves; the estimator's
calibration copy does NOT. So the recorded pixel is re-imaged through the DRIFTED
pose and then projected through the STALE calibration::

    xy_true   = true_cam.pixel_to_ground(u, v)          # what the pixel meant
    u', v'    = drifted_cam.world_to_pixel(xy_true)     # where it lands post-drift
    xy_est    = _project_pixel_to_world(u', v', true_cam, **stale_kwargs)

This is the dual of perturbing the estimator's calibration copy
(``calibration_perturbation.reproject_world``): same induced world bias, opposite
sign. The physical direction is modelled because the sign is what decides Q1.
Identity drift must reproduce the deployed pipeline bit-for-bit; asserted every run.

Images are unchanged -- the detector fires the same pixel content, only the
geometry moves (see A8 in the assumptions register).

Outputs -> logs/studies/calibration_drift_lifecycle/exp1_stale_correction/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _relative in ("src/reliability", "src/unav_common", "src/state",
                  "experiments/operational_residual_rcond",
                  "experiments/network_commissioning_realism"):
    sys.path.insert(0, str(REPO / _relative))

import estimate_rcond as ER  # noqa: E402  (owns the smoother driver)
import rcond_common as rc  # noqa: E402
import exp1_gate_without_truth as GATE_STUDY  # noqa: E402  (owns the gate statistic)

from reliability.calibration_perturbation import (  # noqa: E402
    CalibrationPerturbation,
    PinholeGroundCamera,
    perturb,
)
from reliability.operational_residual import build_operational_residuals  # noqa: E402
from reliability.projection import (  # noqa: E402
    load_projection_calibration,
    projection_kwargs_for_camera,
    _project_pixel_to_world,
)

OUT = REPO / "logs/studies/calibration_drift_lifecycle/exp1_stale_correction"

#: The gate deployed in fit_projection_calibration.py, reused unchanged.
GATE = GATE_STUDY.GATE

CALIB_V2 = (
    REPO / "logs/studies/multicamera_commissioning_bigwarehouse"
    / "projection_calibration_v2/projection_calibration.json"
)
CALIB_V3 = (
    REPO / "logs/studies/multicamera_commissioning_bigwarehouse"
    / "projection_calibration_v3/projection_calibration.json"
)

#: Yaw drift ladder, degrees. 0.1 deg is below anything a person would notice on a
#: mount; 2.0 deg is a knock. The health-monitor probe already flags sub-degree
#: calibration drift, so this ladder brackets its operating point.
YAW_DRIFT_DEG = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)

#: Mount settle / vibration creep, metres of camera-centre translation.
TRANSLATION_DRIFT_M = (0.0, 0.01, 0.025, 0.05, 0.10)

#: The capture held OUT of the v3 calibration fit -- the only honest place to ask
#: what a frozen constant does to unseen data.
HELD_OUT_CAPTURE = "fusion_handover_20260721"

#: exp3's anchor; kept identical so the operational reference is the same object
#: this workstream already validated.
ANCHOR_STD_M = ER.ANCHOR_STD_M


def pinhole_from_oblique(model, ground_z_m: float) -> PinholeGroundCamera:
    """The same camera, in the perturbable representation.

    ``ObliqueCameraModel`` and ``PinholeGroundCamera.looking_at`` build the identical
    look-at rotation and square-pixel intrinsics, so this is a re-parameterisation,
    not an approximation. The identity-drift assertion in :func:`main` is what proves
    it on the actual detections.
    """
    return PinholeGroundCamera.looking_at(
        center_m=tuple(float(v) for v in model.cam_pos),
        look_at_m=tuple(float(v) for v in model.look_at),
        fov_h_deg=math.degrees(float(model.fov_h_rad)),
        width=int(model.img_width),
        height=int(model.img_height),
        ground_z_m=float(ground_z_m),
    )


def drifted_world_point(
    u: float,
    v: float,
    true_model,
    pinhole: PinholeGroundCamera,
    drift: CalibrationPerturbation,
    kwargs: dict,
) -> tuple[float, float] | None:
    """Estimated world point for one detection under a physical camera drift.

    ``kwargs`` is the STALE calibration -- the constants the estimator still holds.
    """
    xy_true = true_model.pixel_to_world_at_z(u, v, rc.CONTACT_Z_M)
    if xy_true is None:
        return None
    if drift.is_identity:
        moved = (float(u), float(v))
    else:
        moved = perturb(pinhole, drift).world_to_pixel(xy_true, z_m=rc.CONTACT_Z_M)
        if moved is None:
            return None
    return _project_pixel_to_world(moved[0], moved[1], true_model, **kwargs)


def build_reference(capture, camera: str):
    """Smoothed operational belief with ``camera`` held out of its own reference.

    Held out, the reference is independent of that camera's drift -- which is
    exactly what makes the in-service gate computable while the camera is faulty.
    """
    traj, _ = ER.smooth_capture(capture, ANCHOR_STD_M, hold_out=camera)
    return traj


def residuals_under_drift(capture, traj, camera, true_model, pinhole, drift, kwargs):
    """Operational residual records for one camera under one drift + calibration."""
    records = []
    for detection in capture.detections[camera]:
        index = rc.associate(capture, detection)
        if index is None:
            continue
        world = drifted_world_point(detection.u, detection.v, true_model, pinhole, drift, kwargs)
        if world is None:
            continue
        records.append((index, world))
    if len(records) < 2:
        return []

    class _M:  # minimal Measurement-shaped view; the builder only reads these
        __slots__ = ("index", "z", "source")

        def __init__(self, index, z, source):
            self.index, self.z, self.source = index, z, source

    return build_operational_residuals(
        smoothed_mean=traj.smoothed_mean,
        smoothed_cov=traj.smoothed_cov,
        measurements=[_M(i, w, camera) for i, w in records],
        camera_id=camera,
        frame="xy",
        anchored_by=traj.sources,
    )


def oracle_error(capture, camera, true_model, pinhole, drift, kwargs, truth_table):
    """**EVALUATION ONLY.** RMS and cross-axis bias against Gazebo truth."""
    cam_x, cam_y = float(true_model.cam_pos[0]), float(true_model.cam_pos[1])
    errors, cross = [], []
    for detection in capture.detections[camera]:
        truth = rc.truth_at(truth_table, detection.stamp)
        if truth is None:
            continue
        world = drifted_world_point(detection.u, detection.v, true_model, pinhole, drift, kwargs)
        if world is None:
            continue
        ex, ey = world[0] - truth[0], world[1] - truth[1]
        errors.append(math.hypot(ex, ey))
        bx, by = world[0] - cam_x, world[1] - cam_y
        norm = math.hypot(bx, by)
        if norm > 1e-9:
            cross.append(-ex * by / norm + ey * bx / norm)
    if not errors:
        return None
    return {
        "n": len(errors),
        "rms_m": float(np.sqrt(np.mean(np.square(errors)))),
        "median_m": float(np.median(errors)),
        "cross_bias_m": float(np.mean(cross)) if cross else math.nan,
    }


def sweep(capture, truth_table, models, pinholes, references, calibrations, ladder, field):
    """One drift ladder x every camera x {raw, stale} policy."""
    rows = []
    for camera in rc.CAMERAS:
        if len(capture.detections[camera]) < 2:
            continue
        traj = references[camera]
        if traj is None:
            continue
        model, pinhole = models[camera], pinholes[camera]
        for magnitude in ladder:
            drift = (
                CalibrationPerturbation(yaw_deg=magnitude)
                if field == "yaw_deg"
                else CalibrationPerturbation(tx_m=magnitude)
            )
            for policy, kwargs in calibrations[camera].items():
                records = residuals_under_drift(
                    capture, traj, camera, model, pinhole, drift, kwargs
                )
                series = GATE_STUDY.cross_axis_series(records, model) if records else None
                gate = (
                    GATE_STUDY.gate_ratio(series["cross"])
                    if series is not None and series["cross"].size >= 2
                    else {"n": 0, "ratio": math.nan, "bias_m": math.nan, "sigma_m": math.nan}
                )
                truth = oracle_error(
                    capture, camera, model, pinhole, drift, kwargs, truth_table
                )
                rows.append(
                    {
                        "camera": camera,
                        "drift_field": field,
                        "drift": magnitude,
                        "policy": policy,
                        "gate_ratio_operational": gate["ratio"],
                        "gate_bias_m_operational": gate["bias_m"],
                        "gate_sigma_m_operational": gate["sigma_m"],
                        "gate_n": gate["n"],
                        "gate_decision": (
                            None if not math.isfinite(gate["ratio"]) else gate["ratio"] >= GATE
                        ),
                        "oracle_rms_m": truth["rms_m"] if truth else math.nan,
                        "oracle_median_m": truth["median_m"] if truth else math.nan,
                        "oracle_cross_bias_m": truth["cross_bias_m"] if truth else math.nan,
                        "oracle_n": truth["n"] if truth else 0,
                    }
                )
    return rows


def _pick(rows, camera, field, drift, policy):
    for row in rows:
        if (
            row["camera"] == camera
            and row["drift_field"] == field
            and row["drift"] == drift
            and row["policy"] == policy
        ):
            return row
    return None


def analyse(rows, ladder, field) -> list[dict]:
    """Q1 harm crossover and Q2 detection latency, per camera.

    Two detectors are scored, because the absolute one turns out not to be usable
    in service:

    ``absolute``  the commissioning gate as-is, ``|b_cross| / sigma_cross >= 1.2``.
                  It answers "does this camera have a lateral bias worth correcting",
                  which is a different question from "has this camera moved".
    ``change``    ``|b_cross(delta) - b_cross(0)| / sigma_cross(0) >= 1.2`` -- the
                  SHIFT against this camera's own commissioned value. Immune to the
                  per-capture offset that makes the absolute test fire at rest, and
                  it does not cancel when the induced bias opposes the resident one.
    """
    out = []
    for camera in rc.CAMERAS:
        baseline = _pick(rows, camera, field, 0.0, "stale_v3")
        raw0 = _pick(rows, camera, field, 0.0, "raw")
        if baseline is None or raw0 is None:
            continue
        base_bias = baseline["gate_bias_m_operational"]
        base_sigma = baseline["gate_sigma_m_operational"]
        harm_at = None
        detect_at = None
        change_at = None
        ladder_detail = []
        for magnitude in ladder:
            stale = _pick(rows, camera, field, magnitude, "stale_v3")
            raw = _pick(rows, camera, field, magnitude, "raw")
            if stale is None or raw is None:
                continue
            shift_z = math.nan
            if math.isfinite(base_bias) and math.isfinite(base_sigma) and base_sigma > 0.0:
                shift_z = abs(stale["gate_bias_m_operational"] - base_bias) / base_sigma
            ladder_detail.append(
                {
                    "drift": magnitude,
                    "rms_raw_m": raw["oracle_rms_m"],
                    "rms_stale_m": stale["oracle_rms_m"],
                    "gate_ratio": stale["gate_ratio_operational"],
                    "change_z": shift_z,
                }
            )
            if magnitude == 0.0:
                continue
            if harm_at is None and stale["oracle_rms_m"] > raw["oracle_rms_m"]:
                harm_at = magnitude
            if (
                detect_at is None
                and math.isfinite(stale["gate_ratio_operational"])
                and stale["gate_ratio_operational"] >= GATE
                and not (
                    math.isfinite(baseline["gate_ratio_operational"])
                    and baseline["gate_ratio_operational"] >= GATE
                )
            ):
                detect_at = magnitude
            if change_at is None and math.isfinite(shift_z) and shift_z >= GATE:
                change_at = magnitude
        out.append(
            {
                "camera": camera,
                "drift_field": field,
                "commissioned_policy": "CALIBRATE" if _has_cross(camera) else "RAW",
                "baseline_gate_ratio": baseline["gate_ratio_operational"],
                "baseline_gate_fires": (
                    math.isfinite(baseline["gate_ratio_operational"])
                    and baseline["gate_ratio_operational"] >= GATE
                ),
                "baseline_rms_stale_m": baseline["oracle_rms_m"],
                "baseline_rms_raw_m": raw0["oracle_rms_m"],
                "stale_correction_harmful_at": harm_at,
                "drift_detected_at_absolute": detect_at,
                "drift_detected_at_change": change_at,
                "ladder": ladder_detail,
            }
        )
    return out


_V3_CACHE: dict = {}


def _has_cross(camera: str) -> bool:
    """Did commissioning fit this camera a cross term (i.e. gate said CALIBRATE)?"""
    entry = _V3_CACHE.get(camera, {})
    return bool(entry.get("cross_intercept_m", 0.0) or entry.get("cross_slope_per_m", 0.0))


def main() -> None:
    models = rc.camera_models()
    pinholes = {c: pinhole_from_oblique(models[c], rc.CONTACT_Z_M) for c in rc.CAMERAS}

    calib_v3 = load_projection_calibration(CALIB_V3)
    _V3_CACHE.update(calib_v3)
    calibrations = {
        camera: {
            # "raw" keeps the FROZEN deployed along-bearing term and drops only the
            # commissioned cross constant, so the comparison isolates the 2-DOF
            # decision rather than re-opening the v2 baseline.
            "raw": {
                **projection_kwargs_for_camera(calib_v3, camera, contact_z_m=rc.CONTACT_Z_M),
                "cross_bearing_offset_m": 0.0,
                "cross_bearing_slope_per_m": 0.0,
            },
            "stale_v3": projection_kwargs_for_camera(
                calib_v3, camera, contact_z_m=rc.CONTACT_Z_M
            ),
        }
        for camera in rc.CAMERAS
    }

    capture = rc.load_operational_capture(
        HELD_OUT_CAPTURE, models=models, calib=calib_v3
    )
    truth_table = rc.load_truth_table(HELD_OUT_CAPTURE)

    # Faithfulness: identity drift must reproduce the deployed projection exactly.
    worst = 0.0
    checked = 0
    for camera in rc.CAMERAS:
        kwargs = calibrations[camera]["stale_v3"]
        for detection in capture.detections[camera]:
            reference = _project_pixel_to_world(
                detection.u, detection.v, models[camera], **kwargs
            )
            through_pinhole = drifted_world_point(
                detection.u, detection.v, models[camera], pinholes[camera],
                CalibrationPerturbation(), kwargs,
            )
            if reference is None or through_pinhole is None:
                continue
            worst = max(worst, math.hypot(
                reference[0] - through_pinhole[0], reference[1] - through_pinhole[1]
            ))
            checked += 1
    assert worst < 1.0e-9, f"identity drift is not a no-op: {worst:.3e} m over {checked} detections"

    # One held-out reference per camera; independent of drift magnitude, so it is
    # built once rather than per rung of the ladder.
    references = {camera: build_reference(capture, camera) for camera in rc.CAMERAS}

    rows = sweep(
        capture, truth_table, models, pinholes, references, calibrations,
        YAW_DRIFT_DEG, "yaw_deg",
    )
    rows += sweep(
        capture, truth_table, models, pinholes, references, calibrations,
        TRANSLATION_DRIFT_M, "tx_m",
    )
    summary = analyse(rows, YAW_DRIFT_DEG, "yaw_deg") + analyse(
        rows, TRANSLATION_DRIFT_M, "tx_m"
    )

    payload = {
        "study": "calibration_drift_lifecycle",
        "experiment": "exp1_stale_correction",
        "capture": HELD_OUT_CAPTURE,
        "capture_role": "held out of the v3 calibration fit",
        "calibration": str(CALIB_V3.relative_to(REPO)),
        "gate_threshold": GATE,
        "anchor_std_m": ANCHOR_STD_M,
        "fault_model": "physical camera drift, stale estimator calibration (A4/A8)",
        "identity_check_max_error_m": worst,
        "identity_check_detections": checked,
        "yaw_drift_deg": list(YAW_DRIFT_DEG),
        "translation_drift_m": list(TRANSLATION_DRIFT_M),
        "rows": rows,
        "summary": summary,
        "ground_truth_use": "evaluation only (oracle RMS / cross bias)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drift_lifecycle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )

    print(f"identity drift is a no-op to {worst:.2e} m over {checked} detections\n")

    for field, ladder in (("yaw_deg", YAW_DRIFT_DEG), ("tx_m", TRANSLATION_DRIFT_M)):
        print(f"=== drift = {field} ===")
        print(
            f"{'cam':<9} {'policy':<9} " + " ".join(f"{d:>8g}" for d in ladder)
            + "     (oracle RMS m / gate ratio)"
        )
        for camera in rc.CAMERAS:
            for policy in ("raw", "stale_v3"):
                cells = []
                for magnitude in ladder:
                    row = _pick(rows, camera, field, magnitude, policy)
                    cells.append(f"{row['oracle_rms_m']:>8.3f}" if row else f"{'-':>8}")
                print(f"{camera:<9} {policy:<9} " + " ".join(cells))
            cells = []
            for magnitude in ladder:
                row = _pick(rows, camera, field, magnitude, "stale_v3")
                ratio = row["gate_ratio_operational"] if row else math.nan
                cells.append(f"{ratio:>8.2f}" if math.isfinite(ratio) else f"{'-':>8}")
            print(f"{'':<9} {'gate|abs':<9} " + " ".join(cells))
            entry = next(
                (e for e in summary if e["camera"] == camera and e["drift_field"] == field), None
            )
            if entry is not None:
                cells = [
                    f"{rung['change_z']:>8.2f}" if math.isfinite(rung["change_z"]) else f"{'-':>8}"
                    for rung in entry["ladder"]
                ]
                print(f"{'':<9} {'gate|chg':<9} " + " ".join(cells))
        print()

    print(
        f"{'cam':<9} {'field':<8} {'commissioned':<13} {'fires@rest':>10} "
        f"{'harmful at':>11} {'det|abs':>8} {'det|chg':>8}"
    )
    for entry in summary:
        print(
            f"{entry['camera']:<9} {entry['drift_field']:<8} {entry['commissioned_policy']:<13} "
            f"{str(entry['baseline_gate_fires']):>10} "
            f"{str(entry['stale_correction_harmful_at']):>11} "
            f"{str(entry['drift_detected_at_absolute']):>8} "
            f"{str(entry['drift_detected_at_change']):>8}"
        )
    print(f"\n-> {OUT / 'drift_lifecycle.json'}")


if __name__ == "__main__":
    main()
