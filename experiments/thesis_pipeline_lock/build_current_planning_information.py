#!/usr/bin/env python3
"""Build a direct expected-information planner artifact from the complete old drives.

This is Track-A draft evidence.  It uses whole-drive out-of-fold R2 predictions for every
admitted development-drive opportunity, retains misses/refusals as zero-information rows,
and writes the exact v3 artifact consumed by ``CameraNetworkModel``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "src/reliability"), str(REPO / "src/unav_common"),
    str(REPO / "experiments/archive/pre_canonical_20260920/reference_controlled_commissioning_v1"),
    str(REPO / "experiments/reference_controlled_commissioning_v1"),
]
from reliability.planning_information import fit_planning_information_field  # noqa: E402
from run_current_residual_covariance import fit_constants, predict_spatial  # noqa: E402


CAMPAIGN = REPO / "logs/commissioning/reference_controlled_v5_complete_20260913"
PREDICTIONS = REPO / "logs/thesis_final_pipeline_v1/stage07_correction_v2/candidate_predictions.npz"
COMPARISON = REPO / "logs/thesis_final_pipeline_v1/stage07_correction_v2/visibility_patch_comparison.json"
RUNTIME_MODEL = REPO / (
    "logs/thesis_final_pipeline_v1/stage07_matched_covariance_v2/"
    "commissioned_visibility_runtime_model.json"
)
OUTPUT = REPO / "logs/thesis_current/planning_information"
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
GRID_STEP_M = 0.20
LENGTH_SCALE_M = 1.0
SUPPORT_RADIUS_M = 2.0
SUPPORT_TAU = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_axis(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(np.ceil((upper - lower) / step)) + 1
    return np.linspace(lower, upper, count)


def correction_residuals() -> dict[str, np.ndarray]:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    if comparison.get("selected_correction") != "box_mlp_visibility_residual":
        raise RuntimeError("the selected old-data correction differs")
    runtime = json.loads(RUNTIME_MODEL.read_text(encoding="utf-8"))
    camera_xy = {key: np.asarray(value, dtype=float)
                 for key, value in runtime["camera_xy_m"].items()}
    record_by_identity = {}
    for drive_id in comparison["development_drives"]:
        path = CAMPAIGN / drive_id / "tables/camera_measurements.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                identity = (drive_id, row["camera_id"], int(row["capture_stamp_ns"]))
                record_by_identity[identity] = row
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        predicted = {key: np.asarray(archive[key]) for key in archive.files}
    selected = np.asarray(predicted["partition"]) == "development"
    drive = np.asarray(predicted["drive_id"])[selected].astype(str)
    camera = np.asarray(predicted["camera_id"])[selected].astype(str)
    stamp = np.asarray(predicted["capture_stamp_ns"], dtype=np.int64)[selected]
    target = np.asarray(predicted["target_ray_m"], dtype=float)[selected]
    correction = np.asarray(predicted["candidate_prediction_ray_m"], dtype=float)[selected]
    matched = [record_by_identity[(d, c, int(t))] for d, c, t in zip(drive, camera, stamp)]
    position = np.asarray([[float(row["reference_x"]), float(row["reference_y"])]
                           for row in matched])
    raw = np.asarray([[float(row["raw_projected_x"]), float(row["raw_projected_y"])]
                      for row in matched])
    bases = []
    for camera_id, observation in zip(camera, raw, strict=True):
        along = observation - camera_xy[camera_id]
        along /= np.linalg.norm(along)
        bases.append(np.stack([along, np.asarray([-along[1], along[0]])], axis=1))
    basis = np.stack(bases)
    residual = np.einsum("nij,nj->ni", basis, correction - target)
    expected_drives = sorted(comparison["development_drives"])
    if sorted(set(drive)) != expected_drives:
        raise RuntimeError("development-drive roster differs")

    covariance = np.empty((len(drive), 2, 2), dtype=float)
    for held_drive in expected_drives:
        train = drive != held_drive
        test = drive == held_drive
        _, r1 = fit_constants(residual[train], drive[train], camera[train])
        covariance[test] = predict_spatial(
            position[train], residual[train], drive[train], camera[train],
            position[test], camera[test], r1,
        )
    return {
        "drive": drive,
        "camera": camera,
        "stamp": stamp,
        "position": position,
        "residual": residual,
        "covariance": covariance,
        "campaign_sha256": sha256(CAMPAIGN / "campaign_execution.json"),
    }


def all_opportunities(prediction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    covariance_by_identity = {
        (d, c, int(t)): R for d, c, t, R in zip(
            prediction["drive"], prediction["camera"], prediction["stamp"],
            prediction["covariance"], strict=True)
    }
    rows = []
    for drive in sorted(set(prediction["drive"])):
        path = CAMPAIGN / drive / "tables/camera_opportunities.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["reference_supported"] != "1" or row["controller_state"].startswith("terminal"):
                    continue
                rows.append(row)
    covariance = np.full((len(rows), 2, 2), np.nan, dtype=float)
    admitted = np.zeros(len(rows), dtype=bool)
    gate_admitted = np.zeros(len(rows), dtype=bool)
    unmatched_admissions = []
    for index, row in enumerate(rows):
        is_admitted = row["sensor_gate_admitted"] == "1"
        gate_admitted[index] = is_admitted
        identity = (row["drive_id"], row["camera_id"], int(row["capture_stamp_ns"]))
        if is_admitted:
            if identity in covariance_by_identity:
                admitted[index] = True
                covariance[index] = covariance_by_identity.pop(identity)
            else:
                # The old visibility-correction comparison did not emit a prediction
                # for every gate admission.  Such rows cannot be assigned a matched
                # OOF covariance and conservatively contribute zero in this Track-A
                # artifact.  The final position campaign must not have this gap.
                unmatched_admissions.append(identity)
        elif identity in covariance_by_identity:
            raise RuntimeError(f"non-admission has an R2 prediction: {identity}")
    if covariance_by_identity:
        raise RuntimeError(f"{len(covariance_by_identity)} admitted predictions were not joined")
    return {
        "positions": np.asarray([[float(r["reference_x"]), float(r["reference_y"])] for r in rows]),
        "headings": np.asarray([float(r["reference_yaw"]) for r in rows]),
        "camera": np.asarray([r["camera_id"] for r in rows]),
        "drive": np.asarray([r["drive_id"] for r in rows]),
        "stamp": np.asarray([int(r["capture_stamp_ns"]) for r in rows], dtype=np.int64),
        "admitted": admitted,
        "gate_admitted": gate_admitted,
        "unmatched_admissions": unmatched_admissions,
        "covariance": covariance,
    }


def percentile(values: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.percentile(values, q)) for q in (0, 1, 5, 50, 95, 99, 100)}


def fused_calibration(prediction: dict[str, np.ndarray]) -> dict:
    """Evaluate the same independent information-form fusion used at runtime."""
    groups: dict[tuple[str, int], list[int]] = {}
    for index, (drive, stamp) in enumerate(zip(
            prediction["drive"], prediction["stamp"], strict=True)):
        groups.setdefault((str(drive), int(stamp)), []).append(index)
    residuals = []
    covariances = []
    camera_counts = []
    for indices in groups.values():
        local_covariance = prediction["covariance"][indices]
        local_residual = prediction["residual"][indices]
        precision = np.linalg.inv(local_covariance)
        fused_covariance = np.linalg.inv(precision.sum(axis=0))
        fused_residual = fused_covariance @ np.einsum(
            "nij,nj->i", precision, local_residual)
        residuals.append(fused_residual)
        covariances.append(fused_covariance)
        camera_counts.append(len(indices))
    residuals = np.stack(residuals)
    covariances = np.stack(covariances)
    camera_counts = np.asarray(camera_counts)
    distance = np.einsum(
        "ni,nij,nj->n", residuals, np.linalg.inv(covariances), residuals)
    determinant = np.linalg.det(covariances)
    nll = 0.5 * (2.0 * np.log(2.0 * np.pi) + np.log(determinant) + distance)

    def metrics(mask: np.ndarray) -> dict:
        return {
            "rounds": int(mask.sum()),
            "position_rmse_cm": float(100.0 * np.sqrt(np.mean(np.sum(residuals[mask] ** 2, axis=1)))),
            "mean_nll": float(np.mean(nll[mask])),
            "coverage_50": float(np.mean(distance[mask] <= 1.38629436112)),
            "coverage_90": float(np.mean(distance[mask] <= 4.60517018599)),
            "coverage_95": float(np.mean(distance[mask] <= 5.99146454711)),
            "coverage_99": float(np.mean(distance[mask] <= 9.21034037198)),
            "mahalanobis_squared_mean": float(np.mean(distance[mask])),
            "covariance_scale_for_mean_chi2_2": float(np.mean(distance[mask]) / 2.0),
            "predicted_equivalent_sd_cm": percentile(
                100.0 * np.sqrt(0.5 * np.trace(covariances[mask], axis1=-2, axis2=-1))),
        }

    output = {"all_admitted_rounds": metrics(np.ones(len(distance), dtype=bool)),
              "by_camera_count": {}}
    for count in sorted(set(camera_counts.tolist())):
        output["by_camera_count"][str(count)] = metrics(camera_counts == count)
    return output


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    staging = OUTPUT.with_name(OUTPUT.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    prediction = correction_residuals()
    fusion_diagnostic = fused_calibration(prediction)
    opportunities = all_opportunities(prediction)
    xs = grid_axis(-10.8, 10.8, GRID_STEP_M)
    ys = grid_axis(-8.8, 8.8, GRID_STEP_M)
    field = fit_planning_information_field(
        opportunities["positions"], opportunities["camera"], opportunities["admitted"],
        opportunities["covariance"], camera_ids=CAMERAS, xs=xs, ys=ys,
        length_scale_m=LENGTH_SCALE_M, support_radius_m=SUPPORT_RADIUS_M,
        support_tau=SUPPORT_TAU, headings_rad=opportunities["headings"],
    )

    source_hashes = {
        str(PREDICTIONS.relative_to(REPO)): sha256(PREDICTIONS),
        str(COMPARISON.relative_to(REPO)): sha256(COMPARISON),
        "experiments/thesis_pipeline_lock/build_current_planning_information.py": sha256(Path(__file__)),
        "src/reliability/reliability/planning_information.py": sha256(
            REPO / "src/reliability/reliability/planning_information.py"),
    }
    metadata = {
        "schema": "camera_network.thesis_stage09.v3",
        "pipeline_id": "THESIS-TRACK-A-OLD-DATA",
        "reference": "robot_ground_reference_xy",
        "frame": "map_bev",
        "covariance_units": "m2",
        "information_units": "m-2",
        "planning_target": "admitted_runtime_precision_else_zero",
        "belief_update": "deterministic_information_approximation",
        "support_population": "all_camera_opportunities",
        "heading_treatment": "equal_observed_heading_pool_within_identical_reference_position",
        "unsupported_limit": "zero_information",
        "runtime_covariance_source": "R2_spatial_full_whole_drive_out_of_fold",
        "runtime_covariance_is_out_of_fold": True,
        "correction_prediction_is_out_of_fold": True,
        "resampling_unit": "complete_drive",
        "prediction_drive_excluded_from_fit": True,
        "final_audit_accessed": False,
        "effective_planning_rate_hz": 1.0,
        "fit_parameters": {
            "grid_step_m": GRID_STEP_M,
            "length_scale_m": LENGTH_SCALE_M,
            "support_radius_m": SUPPORT_RADIUS_M,
            "support_tau": SUPPORT_TAU,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
    }
    artifact = staging / "camera_network_direct_information.npz"
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer, xs=field.xs, ys=field.ys, camera_ids=np.asarray(field.camera_ids),
        expected_information_m2_inv=field.expected_information,
        opportunity_support=field.opportunity_support,
        raw_expected_information_m2_inv=field.raw_expected_information,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    artifact.write_bytes(buffer.getvalue())

    admitted_covariance = opportunities["covariance"][opportunities["admitted"]]
    admitted_camera = opportunities["camera"][opportunities["admitted"]]
    field_trace = np.trace(field.expected_information, axis1=-2, axis2=-1)
    network_trace = field_trace.sum(axis=0)
    max_index = np.unravel_index(np.argmax(network_trace), network_trace.shape)
    report = {
        "schema": "track_a_direct_information_diagnostics.v1",
        "artifact": {"path": artifact.name, "sha256": sha256(artifact)},
        "method": {
            "target": "inverse whole-drive OOF R2 when admitted; zero otherwise",
            "planner_consumption": "CameraNetworkModel v3 direct-information interface",
            "effective_planning_rate_hz": 1.0,
            "expected_updates_for_global_dt_1s": 1,
        },
        "opportunities": {
            camera: {
                "count": int(np.sum(opportunities["camera"] == camera)),
                "admitted": int(np.sum((opportunities["camera"] == camera) & opportunities["admitted"])),
                "admitted_fraction": float(np.mean(opportunities["admitted"][opportunities["camera"] == camera])),
                "gate_admitted": int(np.sum((opportunities["camera"] == camera) & opportunities["gate_admitted"])),
            } for camera in CAMERAS
        },
        "admitted_R2": {
            camera: {
                "standard_deviation_eigenvalues_cm": percentile(
                    100.0 * np.sqrt(np.linalg.eigvalsh(admitted_covariance[admitted_camera == camera])).ravel()),
                "precision_trace_m2_inv": percentile(
                    np.trace(np.linalg.inv(admitted_covariance[admitted_camera == camera]), axis1=-2, axis2=-1)),
            } for camera in CAMERAS
        },
        "independent_runtime_fusion_diagnostic": fusion_diagnostic,
        "deployed_field": {
            "per_camera_trace_m2_inv": {
                camera: percentile(field_trace[index]) for index, camera in enumerate(CAMERAS)
            },
            "network_trace_m2_inv": percentile(network_trace),
            "maximum_network_trace_location_m": [
                float(field.xs[max_index[1]]), float(field.ys[max_index[0]])],
            "contributions_at_maximum_m2_inv": {
                camera: float(field_trace[index][max_index])
                for index, camera in enumerate(CAMERAS)
            },
            "opportunity_support": {
                camera: percentile(field.opportunity_support[index])
                for index, camera in enumerate(CAMERAS)
            },
        },
        "limitations": [
            "Track-A artifact from old complete development drives; it will be replaced by the final reference-position survey.",
            "Field parameters are fixed draft values and are not final-audit selected.",
            "Independent camera information is summed; held-out fused calibration remains a required diagnostic.",
            f"{len(opportunities['unmatched_admissions'])} old-data gate admissions lacked a frozen correction/R2 prediction and conservatively contribute zero.",
        ],
        "source_hashes": source_hashes,
    }
    report_path = staging / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, OUTPUT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
