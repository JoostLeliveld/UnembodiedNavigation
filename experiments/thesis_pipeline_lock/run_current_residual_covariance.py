#!/usr/bin/env python3
"""Export correction-held-out residuals and fit the current R0--R2 ladder.

The frozen visibility-informed MLP was trained on the campaign's fit drives.
Only development-drive observations are used here.  Covariance diagnostics are
whole-drive out of fold; the deployment artifact is refitted on all development
drives after scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiments" / "reference_controlled_commissioning_v1"
sys.path.insert(0, str(REFERENCE))

from analyze_commissioned_model import load_capture  # noqa: E402


CHI2 = {"50": 1.38629436112, "90": 4.60517018599,
        "95": 5.99146454711, "99": 9.21034037198}
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
EIGENVALUE_FLOOR_M2 = 1.0e-6
K_NEIGHBORS = 16
LENGTH_SCALE_M = 0.4
PRIOR_STD_M = 10.0
PRIOR_STRENGTH = 2.5e-6
PRIOR_COVARIANCE = PRIOR_STD_M ** 2 * np.eye(2)
PRIOR_SCATTER = PRIOR_STRENGTH * PRIOR_COVARIANCE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def spd(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, EIGENVALUE_FLOOR_M2)) @ vectors.T


def equal_drive_second_moment(residual: np.ndarray, drive: np.ndarray) -> np.ndarray:
    unique = sorted(set(drive.tolist()))
    moments = []
    for drive_id in unique:
        value = residual[drive == drive_id]
        moments.append(value.T @ value / len(value))
    return spd(np.mean(moments, axis=0))


def fit_constants(residual: np.ndarray, drive: np.ndarray,
                  camera: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_count = len(set(drive.tolist()))
    r0 = spd((global_count * equal_drive_second_moment(residual, drive) + PRIOR_SCATTER)
             / (global_count + PRIOR_STRENGTH))
    r1 = {
        camera_id: spd((
            len(set(drive[camera == camera_id].tolist())) * equal_drive_second_moment(
                residual[camera == camera_id], drive[camera == camera_id])
            + PRIOR_SCATTER
        ) / (len(set(drive[camera == camera_id].tolist())) + PRIOR_STRENGTH))
        for camera_id in CAMERAS
    }
    return r0, r1


def predict_spatial(train_position: np.ndarray, train_residual: np.ndarray,
                    train_drive: np.ndarray, train_camera: np.ndarray,
                    query_position: np.ndarray, query_camera: np.ndarray,
                    r1: dict[str, np.ndarray]) -> np.ndarray:
    output = np.empty((len(query_position), 2, 2), dtype=float)
    for camera_id in CAMERAS:
        train_index = np.flatnonzero(train_camera == camera_id)
        query_index = np.flatnonzero(query_camera == camera_id)
        if not len(query_index):
            continue
        points = train_position[train_index]
        residual = train_residual[train_index]
        drives = train_drive[train_index]
        tree = cKDTree(points)
        count = min(K_NEIGHBORS, len(points))
        distance, neighbor = tree.query(query_position[query_index], k=count)
        if count == 1:
            distance = distance[:, None]
            neighbor = neighbor[:, None]
        for destination, local_distance, local_index in zip(
                query_index, distance, neighbor, strict=True):
            weight = np.exp(-0.5 * (local_distance / LENGTH_SCALE_M) ** 2)
            local_drive = drives[local_index]
            # A densely sampled drive receives at most unit total kernel mass.
            for drive_id in np.unique(local_drive):
                selected = local_drive == drive_id
                total = weight[selected].sum()
                if total > 0:
                    weight[selected] /= total
            scatter = np.einsum(
                "n,ni,nj->ij", weight, residual[local_index], residual[local_index]
            )
            output[destination] = spd(
                (scatter + PRIOR_SCATTER) / (weight.sum() + PRIOR_STRENGTH)
            )
    return output


def predict_constants(camera: np.ndarray, r0: np.ndarray,
                      r1: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    global_covariance = np.repeat(r0[None, :, :], len(camera), axis=0)
    camera_covariance = np.stack([r1[value] for value in camera])
    return global_covariance, camera_covariance


def metric_rows(residual: np.ndarray, covariance: np.ndarray,
                drive: np.ndarray) -> dict:
    inverse = np.linalg.inv(covariance)
    distance = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    determinant = np.maximum(np.linalg.det(covariance), 1.0e-18)
    nll = 0.5 * (2.0 * math.log(2.0 * math.pi) + np.log(determinant) + distance)
    area = math.pi * CHI2["95"] * np.sqrt(determinant)
    by_drive = {}
    for drive_id in sorted(set(drive.tolist())):
        selected = drive == drive_id
        by_drive[drive_id] = {
            "observations": int(selected.sum()),
            "mean_nll": float(nll[selected].mean()),
            "coverage": {
                level: float(np.mean(distance[selected] <= threshold))
                for level, threshold in CHI2.items()
            },
            "median_95_ellipse_area_cm2": float(1.0e4 * np.median(area[selected])),
            "mahalanobis_above_99_fraction": float(
                np.mean(distance[selected] > CHI2["99"])
            ),
        }
    values = list(by_drive.values())
    return {
        "observations": int(len(residual)),
        "complete_drives": int(len(values)),
        "equal_drive_mean_nll": float(np.mean([value["mean_nll"] for value in values])),
        "equal_drive_coverage": {
            level: float(np.mean([value["coverage"][level] for value in values]))
            for level in CHI2
        },
        "equal_drive_median_95_ellipse_area_cm2": float(np.mean([
            value["median_95_ellipse_area_cm2"] for value in values
        ])),
        "equal_drive_mahalanobis_above_99_fraction": float(np.mean([
            value["mahalanobis_above_99_fraction"] for value in values
        ])),
        "by_drive": by_drive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    campaign_root = args.campaign_root.resolve()
    prediction_root = args.prediction_root.resolve()
    execution, records, _ = load_capture(campaign_root)
    prediction_path = prediction_root / "candidate_predictions.npz"
    comparison_path = prediction_root / "visibility_patch_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    expected_hash = comparison["artifacts"]["predictions"]["sha256"]
    if sha256(prediction_path) != expected_hash:
        raise RuntimeError("frozen prediction bundle hash mismatch")
    if comparison.get("selected_correction") != "box_mlp_visibility_residual":
        raise RuntimeError("visibility-informed correction is not selected")
    record_by_identity = {
        (record["drive"], record["camera"], record["stamp_ns"], record["source_frame_id"]): record
        for record in records
    }
    with np.load(prediction_path, allow_pickle=False) as archive:
        predicted = {key: np.asarray(archive[key]) for key in archive.files}
    identities = [
        (str(drive_id), str(camera_id), int(stamp_ns), str(frame_id))
        for drive_id, camera_id, stamp_ns, frame_id in zip(
            predicted["drive_id"], predicted["camera_id"],
            predicted["capture_stamp_ns"], predicted["source_frame_id"], strict=True
        )
    ]
    try:
        matched_records = [record_by_identity[identity] for identity in identities]
    except KeyError as error:
        raise RuntimeError(f"prediction identity absent from capture: {error.args[0]}") from error
    partition = np.asarray(predicted["partition"])
    selected = partition == "development"
    drive = np.asarray(predicted["drive_id"])[selected]
    camera = np.asarray(predicted["camera_id"])[selected]
    position = np.stack([record["reference"] for record in matched_records])[selected]
    basis = np.stack([record["basis"] for record in matched_records])[selected]
    raw = np.stack([record["raw"] for record in matched_records])[selected]
    target = np.asarray(predicted["target_ray_m"], dtype=float)[selected]
    correction = np.asarray(predicted["candidate_prediction_ray_m"], dtype=float)[selected]
    residual_ray_raw_basis = correction - target
    residual_world = np.einsum("nij,nj->ni", basis, residual_ray_raw_basis)
    corrected = raw + np.einsum("nij,nj->ni", basis, correction)
    camera_xy = {
        camera_id: np.asarray(execution["camera_registry"][camera_id]["xy_m"], dtype=float)
        for camera_id in CAMERAS
    } if "camera_registry" in execution else None
    if camera_xy is None:
        # Archived captures store the correction basis but not always the registry.
        # In that case retain the equivalent raw-ray basis for this diagnostic only.
        covariance_basis = basis
    else:
        covariance_basis = []
        for camera_id, point in zip(camera, corrected, strict=True):
            along = point - camera_xy[camera_id]
            along /= np.linalg.norm(along)
            covariance_basis.append(np.column_stack((along, [-along[1], along[0]])))
        covariance_basis = np.stack(covariance_basis)
    residual_ray = np.einsum("nji,nj->ni", covariance_basis, residual_world)

    expected_drives = sorted(comparison["development_drives"])
    if sorted(set(drive.tolist())) != expected_drives or len(expected_drives) != 6:
        raise RuntimeError("expected exactly six correction-held-out development drives")
    if not np.isfinite(residual_world).all():
        raise RuntimeError("non-finite corrected residual")

    residual_path = staging / "corrected_residuals.npz"
    np.savez_compressed(
        residual_path,
        drive=drive,
        camera=camera,
        reference_xy_m=position,
        corrected_xy_m=corrected,
        residual_world_m=residual_world,
        residual_ray_m=residual_ray,
        ray_basis=basis,
    )
    csv_path = staging / "corrected_residuals.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "drive_id", "camera_id", "reference_x_m", "reference_y_m",
            "corrected_x_m", "corrected_y_m", "residual_x_m", "residual_y_m",
            "residual_along_m", "residual_cross_m",
        ])
        for row in zip(drive, camera, position, corrected, residual_world,
                       residual_ray, strict=True):
            writer.writerow([row[0], row[1], *row[2], *row[3], *row[4], *row[5]])

    predictions = {
        "R0_global_full": np.empty((len(drive), 2, 2)),
        "R1_per_camera_full": np.empty((len(drive), 2, 2)),
        "R2_spatial_full": np.empty((len(drive), 2, 2)),
    }
    for held_drive in expected_drives:
        train = drive != held_drive
        test = drive == held_drive
        r0, r1 = fit_constants(residual_ray[train], drive[train], camera[train])
        global_covariance, camera_covariance = predict_constants(camera[test], r0, r1)
        predictions["R0_global_full"][test] = global_covariance
        predictions["R1_per_camera_full"][test] = camera_covariance
        predictions["R2_spatial_full"][test] = predict_spatial(
            position[train], residual_ray[train], drive[train], camera[train],
            position[test], camera[test], r1,
        )

    metrics = {
        name: metric_rows(residual_ray, covariance, drive)
        for name, covariance in predictions.items()
    }

    r0, r1 = fit_constants(residual_ray, drive, camera)
    model_path = staging / "covariance_models.npz"
    np.savez_compressed(
        model_path,
        global_covariance_ray_m2=r0,
        camera_order=np.asarray(CAMERAS),
        per_camera_covariance_ray_m2=np.stack([r1[value] for value in CAMERAS]),
        spatial_reference_xy_m=position,
        spatial_residual_ray_m=residual_ray,
        spatial_drive=drive,
        spatial_camera=camera,
        k_neighbors=np.asarray([K_NEIGHBORS]),
        length_scale_m=np.asarray([LENGTH_SCALE_M]),
        prior_covariance_ray_m2=PRIOR_COVARIANCE,
        prior_strength=np.asarray([PRIOR_STRENGTH]),
        eigenvalue_floor_m2=np.asarray([EIGENVALUE_FLOOR_M2]),
    )

    error = np.linalg.norm(residual_world, axis=1)
    report = {
        "schema": "current_mlp_residual_covariance.v1",
        "status": "complete_current_data_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "correction": "box_mlp_visibility_residual",
        "residual_population": {
            "role": "development drives not used to train the correction",
            "observations": int(len(drive)),
            "complete_drives": expected_drives,
            "mean_error_cm": float(100.0 * error.mean()),
            "median_error_cm": float(100.0 * np.median(error)),
            "rmse_cm": float(100.0 * np.sqrt(np.mean(error ** 2))),
            "p95_error_cm": float(100.0 * np.quantile(error, 0.95)),
            "mean_world_residual_cm": (100.0 * residual_world.mean(axis=0)).tolist(),
        },
        "evaluation": {
            "unit": "complete drive",
            "procedure": "leave one development drive out for every covariance prediction",
            "aggregation": "compute within drive, then arithmetic mean over six drives",
            "residual_recentering": False,
            "covariance_parameterization": "camera_to_corrected_observation_ray_frame",
        },
        "models": {
            "R0_global_full": "pooled full second moment about zero",
            "R1_per_camera_full": "one full second moment about zero per camera",
            "R2_spatial_full": {
                "description": "per-camera kernel-weighted local second moment with broad covariance prior",
                "k_neighbors": K_NEIGHBORS,
                "length_scale_m": LENGTH_SCALE_M,
                "prior_standard_deviation_m": PRIOR_STD_M,
                "prior_strength": PRIOR_STRENGTH,
                "drive_weighting": "unit total kernel mass per represented drive",
            },
        },
        "metrics": metrics,
        "limitations": [
            "The six development drives selected the old correction architecture.",
            "These are current-data diagnostics, not results from the new position-level audit.",
            "The covariance hyperparameters are the current fixed values and were not selected in this run.",
        ],
        "sources": {
            "campaign_root": str(campaign_root),
            "campaign_execution_sha256": sha256(campaign_root / "campaign_execution.json"),
            "prediction_root": str(prediction_root),
            "prediction_sha256": sha256(prediction_path),
            "comparison_sha256": sha256(comparison_path),
            "correction_fit_drives": comparison["fit_drives"],
            "correction_held_out_development_drives": expected_drives,
            "audit_opened": False,
            "campaign_status": execution.get("status"),
        },
        "artifacts": {
            "residual_npz": {"path": residual_path.name, "sha256": sha256(residual_path)},
            "residual_csv": {"path": csv_path.name, "sha256": sha256(csv_path)},
            "covariance_models": {"path": model_path.name, "sha256": sha256(model_path)},
        },
        "implementation_sha256": sha256(Path(__file__)),
    }
    report_path = staging / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    os.replace(staging, output)

    print(json.dumps({
        "residual_population": report["residual_population"],
        "metrics": {
            name: {
                "equal_drive_mean_nll": value["equal_drive_mean_nll"],
                "coverage_95": value["equal_drive_coverage"]["95"],
                "median_95_ellipse_area_cm2": value[
                    "equal_drive_median_95_ellipse_area_cm2"
                ],
                "above_99_fraction": value[
                    "equal_drive_mahalanobis_above_99_fraction"
                ],
            }
            for name, value in metrics.items()
        },
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
