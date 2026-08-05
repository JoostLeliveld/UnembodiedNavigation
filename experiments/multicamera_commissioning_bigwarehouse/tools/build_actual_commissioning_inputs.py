#!/usr/bin/env python3
"""Prepare real four-camera commissioning logs for the canonical GP fitter.

The collector records detector observations and the independent noisy odometry
stream separately.  This small, deterministic adapter aligns every operational
camera observation to its nearest noisy-odometry belief and writes the event
schema consumed by ``scripts/visibility_comparison/fit_belief_aware_gp.py``.

No model input ever comes from ground truth.  New commissioning recordings
carry a propagated noisy-odometry covariance, rotated into the fixed warehouse
frame by the recorder.  A small explicit floor and a timestamp-alignment term
are added at the event level.  Legacy recordings without covariance fields
remain loadable through a clearly labelled isotropic fallback; they cannot be
mistaken for calibrated covariance evidence.

When a run has already had evaluation truth attached (the sibling
``evaluation_inputs/`` directory written by ``attach_evaluation_truth.py``),
the adapter additionally emits an EVALUATION-ONLY ``eval_`` block carrying the
SIGNED localization residual ``pred_world - true`` per detection.  Those
columns exist so a per-camera bias and conditional covariance can be fitted
offline; the canonical fitter reads none of them, and they must never become a
model or deployment input.  Runs without an attached-truth directory keep
writing blanks, exactly as before.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DEFAULT_DAYZERO = (
    REPO
    / "paper_artifacts/gp/warehouse_full_4cam_dayzero_v1/"
    "camera_a_planner_with_four_camera_maps.npz"
)
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")

# Signed localization residual, EVALUATION-ONLY.
#
# ``pred_world - true`` in metres, as a 2-vector.  A magnitude alone (the shape
# the rest of the stack logs) destroys the sign and therefore cannot identify a
# per-camera bias direction b_c(x); these columns are what the offline bias /
# conditional-covariance fit consumes.  They are derived from ground truth, so
# they are audit/evaluation evidence only and are never a GP or deployment
# input -- the canonical fitter selects its features by name and reads none of
# them.
RESIDUAL_FIELDS = (
    "eval_pred_world_x",
    "eval_pred_world_y",
    "eval_res_x",
    "eval_res_y",
    "eval_res_gt_source",
)
EVENT_FIELDS = (
    "m_x",
    "m_y",
    "S_xx",
    "S_xy",
    "S_yy",
    "det_hit",
    "yolo_score_raw",
    "run_id",
    "camera_id",
    "observation_stamp_s",
    "odom_stamp_s",
    "odom_alignment_age_s",
    "state_covariance_source",
) + RESIDUAL_FIELDS

# Sibling directory of ``raw/`` holding truth-attached copies of the per-camera
# perception CSVs (written by ``attach_evaluation_truth.py``).
DEFAULT_EVALUATION_TRUTH_SUBDIR = "evaluation_inputs"


def _float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _closest_odom(
    odometry: list[tuple[float, float, float, float, float, float]], stamp_s: float
) -> tuple[float, float, float, float, float, float] | None:
    if not odometry or not math.isfinite(stamp_s):
        return None
    stamps = [row[0] for row in odometry]
    index = bisect.bisect_left(stamps, stamp_s)
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(odometry)]
    if not candidates:
        return None
    return min((odometry[candidate] for candidate in candidates), key=lambda item: abs(item[0] - stamp_s))


def _recorded_covariance_or_none(
    covariance: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Return a finite SPD planar covariance or ``None`` for a legacy row."""

    xx, xy, yy = covariance
    if not all(math.isfinite(value) for value in covariance):
        return None
    if xx <= 0.0 or yy <= 0.0 or xx * yy - xy * xy <= 1.0e-12:
        return None
    return float(xx), float(xy), float(yy)


def _blank_residual() -> dict[str, Any]:
    return {field: "" for field in RESIDUAL_FIELDS}


def _load_truth_by_stamp(truth_csv: Path) -> dict[str, tuple[float, float]]:
    """Map ``diag_stamp`` -> ``(true_x, true_y)`` from a truth-attached copy.

    ``attach_evaluation_truth.py`` writes row-for-row copies of the operational
    per-camera CSV with ``true_x``/``true_y``/``true_yaw`` appended, so the
    ``diag_stamp`` text is byte-identical on both sides and keying on the raw
    string is an exact join rather than a float re-parse.
    """

    truth: dict[str, tuple[float, float]] = {}
    if not truth_csv.is_file():
        return truth
    for row in _read_csv(truth_csv):
        stamp_key = str(row.get("diag_stamp", "")).strip()
        if not stamp_key:
            continue
        true_x = _float(row, "true_x")
        true_y = _float(row, "true_y")
        if not (math.isfinite(true_x) and math.isfinite(true_y)):
            continue
        truth.setdefault(stamp_key, (float(true_x), float(true_y)))
    return truth


def _residual_fields(
    row: dict[str, str],
    truth_by_stamp: dict[str, tuple[float, float]],
    *,
    detected: bool,
) -> dict[str, Any]:
    """Signed residual columns for one perception row (EVALUATION-ONLY).

    Every field stays EMPTY -- never ``0.0`` -- when the quantity was not
    measured: no detection, no finite ``pred_world_x/y``, or no attached ground
    truth for this frame.  A zero would read to a bias fit as "unbiased here"
    and silently pull b_c(x) toward the origin.
    """

    fields = _blank_residual()
    if not detected:
        return fields
    pred_x = _float(row, "pred_world_x")
    pred_y = _float(row, "pred_world_y")
    if not (math.isfinite(pred_x) and math.isfinite(pred_y)):
        return fields
    fields["eval_pred_world_x"] = float(pred_x)
    fields["eval_pred_world_y"] = float(pred_y)

    stamp_key = str(row.get("diag_stamp", "")).strip()
    truth = truth_by_stamp.get(stamp_key)
    if truth is None:
        return fields
    fields["eval_res_x"] = float(pred_x) - truth[0]
    fields["eval_res_y"] = float(pred_y) - truth[1]
    fields["eval_res_gt_source"] = "attached_evaluation_true_xy"
    return fields


def _raw_run_dirs(run_root: Path) -> Iterable[Path]:
    # A completed route is materialised as <ordinal>_<name>/raw.  Failed
    # diagnostic recordings have no valid observations and are intentionally
    # ignored by the required odometry/per-camera file check below.
    for raw_dir in sorted(run_root.glob("*/raw")):
        if (raw_dir / "experiment.csv").is_file() and any(
            (raw_dir / f"{camera}_perception.csv").is_file() for camera in CAMERAS
        ):
            yield raw_dir


def _events_from_run(
    raw_dir: Path,
    *,
    sigma_floor_m: float,
    max_alignment_age_s: float,
    evaluation_truth_subdir: str = DEFAULT_EVALUATION_TRUTH_SUBDIR,
) -> dict[str, list[dict[str, Any]]]:
    odometry_rows = _read_csv(raw_dir / "experiment.csv")
    odometry = sorted(
        (
            (
                _float(row, "stamp"),
                _float(row, "odom_noisy_x"),
                _float(row, "odom_noisy_y"),
                _float(row, "odom_noisy_cov_xx"),
                _float(row, "odom_noisy_cov_xy"),
                _float(row, "odom_noisy_cov_yy"),
            )
            for row in odometry_rows
        ),
        key=lambda row: row[0],
    )
    odometry = [row for row in odometry if all(math.isfinite(value) for value in row[:3])]
    events = {camera: [] for camera in CAMERAS}
    if not odometry:
        return events

    truth_dir = (
        raw_dir.parent / evaluation_truth_subdir if evaluation_truth_subdir else None
    )
    for camera in CAMERAS:
        path = raw_dir / f"{camera}_perception.csv"
        if not path.is_file():
            continue
        # EVALUATION-ONLY read: truth is used to *measure* the signed residual
        # and never touches m_x/m_y/S_* or any other model input below.
        truth_by_stamp = (
            _load_truth_by_stamp(truth_dir / f"{camera}_perception.csv")
            if truth_dir is not None
            else {}
        )
        for row in _read_csv(path):
            stamp_s = _float(row, "diag_stamp")
            closest = _closest_odom(odometry, stamp_s)
            if closest is None:
                continue
            odom_stamp_s, x_m, y_m, cov_xx, cov_xy, cov_yy = closest
            age_s = abs(stamp_s - odom_stamp_s)
            if age_s > max_alignment_age_s:
                continue
            # The age term is independent of the propagated covariance and
            # makes a detector frame conservative when it has to align to a
            # more distant operational belief sample.
            age_var = (0.10 * age_s) ** 2
            recorded_covariance = _recorded_covariance_or_none((cov_xx, cov_xy, cov_yy))
            if recorded_covariance is None:
                event_xx = sigma_floor_m * sigma_floor_m + age_var
                event_xy = 0.0
                event_yy = sigma_floor_m * sigma_floor_m + age_var
                covariance_source = "legacy_declared_isotropic_fallback"
            else:
                event_xx = recorded_covariance[0] + sigma_floor_m * sigma_floor_m + age_var
                event_xy = recorded_covariance[1]
                event_yy = recorded_covariance[2] + sigma_floor_m * sigma_floor_m + age_var
                covariance_source = "propagated_odom_covariance_plus_floor_and_alignment"
            det_hit = int(_float(row, "detected") >= 0.5)
            event = {
                "m_x": x_m,
                "m_y": y_m,
                "S_xx": event_xx,
                "S_xy": event_xy,
                "S_yy": event_yy,
                "det_hit": det_hit,
                "yolo_score_raw": _float(row, "yolo_score_raw"),
                "run_id": raw_dir.parent.name,
                "camera_id": camera,
                "observation_stamp_s": stamp_s,
                "odom_stamp_s": odom_stamp_s,
                "odom_alignment_age_s": age_s,
                "state_covariance_source": covariance_source,
            }
            event.update(_residual_fields(row, truth_by_stamp, detected=det_hit == 1))
            events[camera].append(event)
    return events


def _residual_counts(combined: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Audit how many detection events ended up with a usable signed residual."""

    by_camera: dict[str, int] = {}
    total = 0
    detections = 0
    for camera, rows in combined.items():
        camera_total = 0
        for row in rows:
            if int(row.get("det_hit", 0) or 0) == 1:
                detections += 1
            res_x = row.get("eval_res_x", "")
            res_y = row.get("eval_res_y", "")
            if isinstance(res_x, float) and isinstance(res_y, float):
                if math.isfinite(res_x) and math.isfinite(res_y):
                    camera_total += 1
        by_camera[camera] = camera_total
        total += camera_total
    return {
        "events_with_signed_residual": int(total),
        "detection_events": int(detections),
        "by_camera": by_camera,
    }


def _write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_camera_priors(dayzero_path: Path, out_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    with np.load(dayzero_path, allow_pickle=False) as source:
        xs = np.asarray(source["xs"], dtype=float)
        ys = np.asarray(source["ys"], dtype=float)
        for camera in CAMERAS:
            p_map = np.asarray(source[f"P_{camera}_map"], dtype=float)
            path = out_dir / f"{camera}_dayzero_prior.npz"
            np.savez_compressed(
                path,
                xs=xs,
                ys=ys,
                P_mean_map=p_map,
                P_conservative_plan_map=p_map,
                prior_source=np.asarray([str(dayzero_path)], dtype=np.str_),
                prior_camera_id=np.asarray([camera], dtype=np.str_),
            )
            outputs[camera] = str(path)
        union = np.asarray(source["P_union_4cam_map"], dtype=float)
        pooled = out_dir / "pooled_dayzero_prior.npz"
        np.savez_compressed(
            pooled,
            xs=xs,
            ys=ys,
            P_mean_map=union,
            P_conservative_plan_map=union,
            prior_source=np.asarray([str(dayzero_path)], dtype=np.str_),
            prior_camera_id=np.asarray(["pooled_operational_events"], dtype=np.str_),
        )
        outputs["pooled"] = str(pooled)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dayzero-prior", type=Path, default=DEFAULT_DAYZERO)
    parser.add_argument(
        "--state-sigma-floor-m",
        type=float,
        default=0.02,
        help="Additive 1-sigma covariance floor in metres for recorded noisy odometry.",
    )
    parser.add_argument(
        "--state-sigma-m",
        dest="state_sigma_floor_m",
        type=float,
        default=argparse.SUPPRESS,
        help="Deprecated compatibility alias for --state-sigma-floor-m.",
    )
    parser.add_argument("--max-alignment-age-s", type=float, default=0.15)
    parser.add_argument(
        "--evaluation-truth-subdir",
        default=DEFAULT_EVALUATION_TRUTH_SUBDIR,
        help=(
            "Sibling directory of raw/ holding truth-attached per-camera CSVs, used ONLY to "
            "populate the evaluation-only eval_* signed-residual columns. Pass an empty string "
            "to leave those columns blank."
        ),
    )
    args = parser.parse_args()
    if args.state_sigma_floor_m <= 0.0 or args.max_alignment_age_s <= 0.0:
        parser.error("--state-sigma-floor-m and --max-alignment-age-s must be positive")

    run_root = args.run_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    dayzero_path = args.dayzero_prior.expanduser().resolve()
    if not dayzero_path.is_file():
        raise RuntimeError(f"Day-zero prior not found: {dayzero_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = {camera: [] for camera in CAMERAS}
    included_runs: list[str] = []
    for raw_dir in _raw_run_dirs(run_root):
        per_camera = _events_from_run(
            raw_dir,
            sigma_floor_m=float(args.state_sigma_floor_m),
            max_alignment_age_s=float(args.max_alignment_age_s),
            evaluation_truth_subdir=str(args.evaluation_truth_subdir),
        )
        if any(per_camera.values()):
            included_runs.append(raw_dir.parent.name)
        for camera, rows in per_camera.items():
            combined[camera].extend(rows)

    for camera, rows in combined.items():
        _write_events(out_dir / f"{camera}_events.csv", rows)
    pooled = sorted(
        (row for rows in combined.values() for row in rows),
        key=lambda row: (float(row["observation_stamp_s"]), str(row["camera_id"])),
    )
    _write_events(out_dir / "pooled_all_cameras_events.csv", pooled)
    priors = _write_camera_priors(dayzero_path, out_dir)
    covariance_sources = {
        source: sum(
            1
            for rows in combined.values()
            for row in rows
            if row["state_covariance_source"] == source
        )
        for source in sorted(
            {
                str(row["state_covariance_source"])
                for rows in combined.values()
                for row in rows
            }
        )
    }

    residual_counts = _residual_counts(combined)

    summary = {
        "source_run_root": str(run_root),
        "included_runs": included_runs,
        "event_counts": {camera: len(rows) for camera, rows in combined.items()},
        "pooled_event_count": len(pooled),
        "state_source": "noisy_operational_odometry",
        "state_covariance": {
            "kind": "propagated_odom_covariance_plus_floor_or_legacy_fallback",
            "sigma_floor_m": float(args.state_sigma_floor_m),
            "age_growth_m_per_s": 0.10,
            "event_count_by_source": covariance_sources,
            "reason": "new recorder logs propagated noisy-odometry covariance; legacy logs are retained with an explicit fallback",
        },
        "alignment": {"nearest_odom": True, "max_age_s": float(args.max_alignment_age_s)},
        "dayzero_prior": str(dayzero_path),
        "prior_artifacts": priors,
        # True only when an attached-truth directory was found and signed
        # residuals were actually written. Runs built from operational logs
        # alone keep reporting False, exactly as before.
        "contains_ground_truth": bool(residual_counts["events_with_signed_residual"]),
        "contains_ground_truth_in_model_inputs": False,
        "evaluation_residuals": {
            "columns": list(RESIDUAL_FIELDS),
            "definition": "eval_res = pred_world - true (signed 2-vector, metres)",
            "truth_subdir": str(args.evaluation_truth_subdir),
            "events_with_signed_residual": residual_counts["events_with_signed_residual"],
            "detection_events": residual_counts["detection_events"],
            "events_with_signed_residual_by_camera": residual_counts["by_camera"],
            "missing_policy": "empty, never 0.0",
            "note": (
                "EVALUATION-ONLY. Derived from ground truth so it may never be a model or "
                "deployment input; the canonical fitter selects m_x/m_y/S_xx/S_xy/S_yy/det_hit/"
                "yolo_score_raw by name and reads none of these columns."
            ),
        },
        "pooled_gp_note": "Pooled GP is a commissioning diagnostic only; operational handover selects among per-camera posteriors.",
    }
    (out_dir / "inputs_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
