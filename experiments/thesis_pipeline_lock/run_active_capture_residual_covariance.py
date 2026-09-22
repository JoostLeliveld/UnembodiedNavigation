#!/usr/bin/env python3
"""Evaluate the old visibility MLP and R0--R2 on a live-capture snapshot.

Only complete D_R and D_dev positions present in one captured index byte
snapshot are used.  Final-audit rows are never selected or decoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src/perception"), str(REPO / "src/reliability"),
                str(REPO / "src/unav_common")]

from perception.core.yolo_selection import select_best_detection, target_class_ids  # noqa: E402
from reliability.contracts import CameraObservation  # noqa: E402
from reliability.commissioned_visibility import CommissionedVisibilitySensorModel  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world, project_observation_to_world_with_covariance,
)
from unav_common.visibility_patch import visibility_grid_from_bgr_frame  # noqa: E402

from run_current_residual_covariance import (  # noqa: E402
    CAMERAS, CHI2, EIGENVALUE_FLOOR_M2, K_NEIGHBORS, LENGTH_SCALE_M,
    PRIOR_COVARIANCE, PRIOR_SCATTER, PRIOR_STD_M, PRIOR_STRENGTH, metric_rows, spd,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def captured_image_hash(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return ""
    digest = hashlib.sha1()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def complete_positions(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["position_id"])].append(row)
    complete = {}
    for position_id, values in grouped.items():
        identities = {(row["pose_id"], row["camera_id"]) for row in values}
        roles = {row["dataset_split"] for row in values}
        if len(values) == 20 and len(identities) == 20 and len(roles) == 1:
            complete[position_id] = values
    return complete


def maximin_position_ids(values: dict[int, list[dict[str, str]]], count: int) -> list[int]:
    ids = sorted(values)
    if len(ids) <= count:
        return ids
    points = np.asarray([
        [float(values[position_id][0]["robot_x"]), float(values[position_id][0]["robot_y"])]
        for position_id in ids
    ])
    selected = [int(np.lexsort((points[:, 1], points[:, 0]))[0])]
    minimum_distance2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        minimum_distance2[selected] = -1.0
        candidate = int(np.argmax(minimum_distance2))
        selected.append(candidate)
        minimum_distance2 = np.minimum(
            minimum_distance2, np.sum((points - points[candidate]) ** 2, axis=1)
        )
    return [ids[index] for index in selected]


def second_moment(residual: np.ndarray) -> np.ndarray:
    return spd(residual.T @ residual / len(residual))


def grouped_moments(position: np.ndarray, position_id: np.ndarray, residual: np.ndarray,
                    camera: np.ndarray, *, separate_camera: bool):
    """Return one second moment for every independent physical-position group."""
    keys = sorted({
        (int(pid), str(cam) if separate_camera else "all")
        for pid, cam in zip(position_id, camera, strict=True)
    })
    points, cameras, moments = [], [], []
    for pid, camera_key in keys:
        use = position_id == pid
        if separate_camera:
            use &= camera == camera_key
        points.append(np.mean(position[use], axis=0))
        cameras.append(camera_key)
        moments.append(second_moment(residual[use]))
    return np.asarray(points), np.asarray(cameras), np.stack(moments)


def fit_constants(position: np.ndarray, position_id: np.ndarray,
                  residual: np.ndarray, camera: np.ndarray):
    _points, _all, global_moments = grouped_moments(
        position, position_id, residual, camera, separate_camera=False)
    _points, grouped_camera, camera_moments = grouped_moments(
        position, position_id, residual, camera, separate_camera=True)
    r0 = spd((global_moments.sum(axis=0) + PRIOR_SCATTER)
             / (len(global_moments) + PRIOR_STRENGTH))
    return r0, {
        camera_id: spd((camera_moments[grouped_camera == camera_id].sum(axis=0)
                        + PRIOR_SCATTER)
                       / (np.sum(grouped_camera == camera_id) + PRIOR_STRENGTH))
        for camera_id in CAMERAS
    }


def fit_isotropic_pixel_sigma(position_id: np.ndarray, residual_world: np.ndarray,
                              unit_pixel_covariance_world: np.ndarray) -> float:
    """MLE scale for ``R_n = sigma_px^2 J_n J_n^T``, balanced by position."""
    mahalanobis = np.einsum(
        "ni,nij,nj->n", residual_world,
        np.linalg.inv(unit_pixel_covariance_world), residual_world)
    position_scores = [
        float(mahalanobis[position_id == value].mean())
        for value in sorted(set(position_id.tolist()))
    ]
    sigma2 = float(np.mean(position_scores) / 2.0)
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        raise RuntimeError("homography baseline produced an invalid pixel-noise scale")
    return math.sqrt(sigma2)


def spatial_prediction(train_position, train_residual, train_position_id, train_camera,
                       query_position, query_camera, r1):
    grouped_position, grouped_camera, grouped_second_moment = grouped_moments(
        train_position, train_position_id, train_residual, train_camera,
        separate_camera=True)
    output = np.empty((len(query_position), 2, 2), dtype=float)
    for camera_id in CAMERAS:
        train_index = np.flatnonzero(grouped_camera == camera_id)
        query_index = np.flatnonzero(query_camera == camera_id)
        tree = cKDTree(grouped_position[train_index])
        count = min(K_NEIGHBORS, len(train_index))
        distance, neighbor = tree.query(query_position[query_index], k=count)
        if count == 1:
            distance, neighbor = distance[:, None], neighbor[:, None]
        local_moment = grouped_second_moment[train_index]
        for destination, distances, indexes in zip(query_index, distance, neighbor, strict=True):
            weight = np.exp(-0.5 * (distances / LENGTH_SCALE_M) ** 2)
            scatter = np.einsum("n,nij->ij", weight, local_moment[indexes])
            output[destination] = spd(
                (scatter + PRIOR_SCATTER) / (weight.sum() + PRIOR_STRENGTH)
            )
    return output


def position_metrics(residual: np.ndarray, covariance: np.ndarray,
                     position_id: np.ndarray) -> dict:
    # Reuse the complete-group scorer with position IDs in place of drive IDs.
    result = metric_rows(residual, covariance, position_id.astype(str))
    result["complete_positions"] = result.pop("complete_drives")
    result["aggregation"] = "within complete position, then arithmetic mean over positions"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--old-mlp-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positions-per-role", type=int, default=25)
    args = parser.parse_args()

    capture = args.capture.resolve()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    index_path = capture / "capture_index.csv"
    index_bytes = index_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(index_bytes.decode("utf-8"))))
    complete = complete_positions(rows)
    selected_positions = {
        position_id: values for position_id, values in complete.items()
        if values[0]["dataset_split"] in {"D_R", "D_dev"}
    }
    sampled_ids = set()
    for role_name in ("D_R", "D_dev"):
        candidates = {
            position_id: values for position_id, values in selected_positions.items()
            if values[0]["dataset_split"] == role_name
        }
        sampled_ids.update(maximin_position_ids(candidates, int(args.positions_per_role)))
    selected_positions = {
        position_id: values for position_id, values in selected_positions.items()
        if position_id in sampled_ids
    }
    digest_cache: dict[str, str | None] = {}
    def image_matches(row: dict[str, str]) -> bool:
        if row["capture_status"] != "ok":
            return True
        path = capture / row["image"]
        key = str(path)
        if key not in digest_cache:
            digest_cache[key] = captured_image_hash(path) if path.is_file() else None
        return digest_cache[key] == row["image_sha1"]
    corrupt_position_ids = {
        position_id for position_id, values in selected_positions.items()
        if not all(image_matches(row) for row in values)
    }
    selected_positions = {
        position_id: values for position_id, values in selected_positions.items()
        if position_id not in corrupt_position_ids
    }
    selected_rows = [
        row for position_id in sorted(selected_positions)
        for row in selected_positions[position_id]
    ]
    if not selected_rows or any(row["dataset_split"] == "final_audit" for row in selected_rows):
        raise RuntimeError("snapshot role selection failed")

    manifest = json.loads((capture / "capture_manifest.json").read_text(encoding="utf-8"))
    world = Path(manifest["world_path"])
    camera_models = {
        item["camera_id"]: camera_model_from_world(world, include_name=item["camera_model"])
        for item in manifest["cameras"]
    }
    if set(camera_models) != set(CAMERAS):
        raise RuntimeError("capture camera registry differs from the MLP registry")

    image_by_hash: dict[str, Path] = {}
    rows_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    failed_rows = []
    for row in selected_rows:
        if row["capture_status"] != "ok":
            failed_rows.append(row)
            continue
        path = capture / row["image"]
        image_by_hash.setdefault(row["image_sha1"], path)
        rows_by_hash[row["image_sha1"]].append(row)
    weights = args.weights.resolve()
    old_manifest = args.old_mlp_manifest.resolve()
    detector = YOLO(str(weights))
    target_ids = target_class_ids(getattr(detector, "names", {}), "robot", -1)
    if target_ids == set() and set(getattr(detector, "names", {})) == {0}:
        target_ids = {0}
    if target_ids == set():
        raise RuntimeError("frozen detector has no unambiguous target class")
    sensor = CommissionedVisibilitySensorModel(old_manifest)
    hashes = sorted(image_by_hash)
    detections = {}
    for start in range(0, len(hashes), args.batch_size):
        chunk = hashes[start:start + args.batch_size]
        images = [cv2.imread(str(image_by_hash[value]), cv2.IMREAD_COLOR) for value in chunk]
        if any(image is None or image.shape != (720, 1280, 3) for image in images):
            raise RuntimeError("snapshot contains an undecodable or wrong-sized image")
        results = detector.predict(
            source=images, imgsz=960, conf=0.001, iou=0.45,
            batch=len(images), device=args.device, verbose=False, stream=False,
        )
        for image_hash, result in zip(chunk, results, strict=True):
            detections[image_hash] = select_best_detection(
                result, target_ids=target_ids, confidence_threshold=0.25,
                use_masks=False, mask_min_area=0.0, mask_bottom_band_px=3.0,
            )
        if start % (args.batch_size * 25) == 0:
            print(f"detector {min(start + args.batch_size, len(hashes))}/{len(hashes)}", flush=True)

    opportunities = {"D_R": 0, "D_dev": 0}
    misses = {"D_R": 0, "D_dev": 0}
    refusals = {"D_R": 0, "D_dev": 0}
    admitted = []
    for row in selected_rows:
        role = row["dataset_split"]
        opportunities[role] += 1
        if row["capture_status"] != "ok":
            misses[role] += 1
            continue
        selection = detections[row["image_sha1"]]
        if not selection["detected"] or selection["bbox_xyxy"] is None:
            misses[role] += 1
            continue
        box = np.asarray(selection["bbox_xyxy"], dtype=float)
        confidence = float(selection["confidence"])
        width, height = box[2] - box[0], box[3] - box[1]
        if (width < 16.0 or height < 16.0 or box[0] < 5.0 or box[1] < 5.0
                or box[2] > 1275.0 or box[3] > 715.0):
            refusals[role] += 1
            continue
        bottom = (0.5 * (box[0] + box[2]), box[3])
        raw = camera_models[row["camera_id"]].pixel_to_world_at_z(bottom[0], bottom[1], 0.0)
        if raw is None or not np.isfinite(raw[:2]).all():
            refusals[role] += 1
            continue
        image = cv2.imread(str(image_by_hash[row["image_sha1"]]), cv2.IMREAD_COLOR)
        grid = visibility_grid_from_bgr_frame(image, box)
        correction = sensor.correction_ray(row["camera_id"], raw[:2], box, confidence, grid)
        camera_xy = sensor.camera_xy[row["camera_id"]]
        along = np.asarray(raw[:2]) - camera_xy
        along /= np.linalg.norm(along)
        basis = np.column_stack((along, np.asarray([-along[1], along[0]])))
        corrected = np.asarray(raw[:2]) + basis @ correction
        truth = np.asarray([float(row["robot_x"]), float(row["robot_y"])])
        covariance_along = corrected - camera_xy
        covariance_along /= np.linalg.norm(covariance_along)
        covariance_basis = np.column_stack((
            covariance_along,
            np.asarray([-covariance_along[1], covariance_along[0]]),
        ))
        residual_world = corrected - truth
        pixel_observation = CameraObservation(
            camera_id=row["camera_id"], timestamp_s=0.0,
            pixel_uv=(float(bottom[0]), float(bottom[1])), detection_valid=True,
            detector_score=confidence, conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
        )
        projected_unit = project_observation_to_world_with_covariance(
            pixel_observation, camera_models[row["camera_id"]])
        if projected_unit is None:
            refusals[role] += 1
            continue
        admitted.append({
            "role": role, "position_id": int(row["position_id"]),
            "camera": row["camera_id"], "truth": truth, "raw": np.asarray(raw[:2]),
            "corrected": corrected, "residual_world": residual_world,
            "residual_ray": covariance_basis.T @ residual_world,
            "unit_pixel_covariance_world": np.asarray(projected_unit[1], dtype=float),
            "correction_ray": correction, "bbox": box, "confidence": confidence,
            "image_sha1": row["image_sha1"],
        })

    role = np.asarray([row["role"] for row in admitted])
    position_id = np.asarray([row["position_id"] for row in admitted])
    camera = np.asarray([row["camera"] for row in admitted])
    truth = np.stack([row["truth"] for row in admitted])
    residual_world = np.stack([row["residual_world"] for row in admitted])
    residual_ray = np.stack([row["residual_ray"] for row in admitted])
    corrected = np.stack([row["corrected"] for row in admitted])
    unit_pixel_covariance = np.stack([
        row["unit_pixel_covariance_world"] for row in admitted])
    train, test = role == "D_R", role == "D_dev"
    if not train.any() or not test.any():
        raise RuntimeError("snapshot has no admitted D_R or D_dev observations")
    r0, r1 = fit_constants(
        truth[train], position_id[train], residual_ray[train], camera[train])
    sigma_px = fit_isotropic_pixel_sigma(
        position_id[train], residual_world[train], unit_pixel_covariance[train])
    covariance = {
        "R0_global_full": np.repeat(r0[None], int(test.sum()), axis=0),
        "R1_per_camera_full": np.stack([r1[value] for value in camera[test]]),
        "R2_spatial_full": spatial_prediction(
            truth[train], residual_ray[train], position_id[train], camera[train],
            truth[test], camera[test], r1,
        ),
    }
    metrics = {
        name: position_metrics(residual_ray[test], value, position_id[test])
        for name, value in covariance.items()
    }
    projection_covariance = sigma_px ** 2 * unit_pixel_covariance[test]
    metrics["Rproj_homography_pixel"] = position_metrics(
        residual_world[test], projection_covariance, position_id[test])

    residual_path = staging / "corrected_residuals.npz"
    np.savez_compressed(
        residual_path, role=role, position_id=position_id, camera=camera,
        truth_xy_m=truth, corrected_xy_m=corrected,
        residual_world_m=residual_world, residual_ray_m=residual_ray,
    )
    model_path = staging / "covariance_models.npz"
    spatial_position, spatial_camera, spatial_moment = grouped_moments(
        truth[train], position_id[train], residual_ray[train], camera[train],
        separate_camera=True)
    np.savez_compressed(
        model_path, global_covariance_ray_m2=r0, camera_order=np.asarray(CAMERAS),
        per_camera_covariance_ray_m2=np.stack([r1[value] for value in CAMERAS]),
        spatial_reference_xy_m=spatial_position,
        spatial_second_moment_ray_m2=spatial_moment,
        spatial_camera=spatial_camera,
        k_neighbors=np.asarray([K_NEIGHBORS]), length_scale_m=np.asarray([LENGTH_SCALE_M]),
        prior_covariance_ray_m2=PRIOR_COVARIANCE,
        prior_strength=np.asarray([PRIOR_STRENGTH]),
        eigenvalue_floor_m2=np.asarray([EIGENVALUE_FLOOR_M2]),
        homography_baseline_sigma_px=np.asarray([sigma_px]),
    )
    error = np.linalg.norm(residual_world, axis=1)
    report = {
        "schema": "active_capture_old_mlp_residual_covariance.v1",
        "status": "complete_snapshot_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": {
            "capture": str(capture), "capture_status_at_snapshot": manifest.get("status"),
            "capture_index_sha256": sha256_bytes(index_bytes),
            "complete_positions": {
                role_name: sum(values[0]["dataset_split"] == role_name
                               for values in complete.values())
                for role_name in ("D_mu", "D_R", "D_dev", "final_audit")
            },
            "used_roles": ["D_R", "D_dev"], "final_audit_images_opened": False,
            "position_sampling": {
                "method": "deterministic planar maximin within role",
                "requested_per_role": int(args.positions_per_role),
            },
            "excluded_complete_positions_with_overwritten_images": len(corrupt_position_ids),
            "verified_positions_used": {
                role_name: sum(values[0]["dataset_split"] == role_name
                               for values in selected_positions.values())
                for role_name in opportunities
            },
            "selected_opportunities": opportunities,
            "acquisition_failures": {
                role_name: sum(row["dataset_split"] == role_name for row in failed_rows)
                for role_name in opportunities
            },
            "unique_images_inferred": len(hashes),
        },
        "models": {
            "detector_checkpoint": str(weights), "detector_sha256": sha256(weights),
            "correction": "old box_mlp_visibility_residual",
            "correction_manifest": str(old_manifest),
            "correction_manifest_sha256": sha256(old_manifest),
        },
        "opportunity_outcomes": {
            role_name: {"total": opportunities[role_name], "miss": misses[role_name],
                        "refused": refusals[role_name],
                        "admitted": int(np.sum(role == role_name))}
            for role_name in opportunities
        },
        "correction_error": {
            role_name: {
                "observations": int(np.sum(role == role_name)),
                "positions": int(len(set(position_id[role == role_name].tolist()))),
                "mean_cm": float(100 * error[role == role_name].mean()),
                "median_cm": float(100 * np.median(error[role == role_name])),
                "rmse_cm": float(100 * np.sqrt(np.mean(error[role == role_name] ** 2))),
                "p95_cm": float(100 * np.quantile(error[role == role_name], 0.95)),
            } for role_name in opportunities
        },
        "covariance_fit": {
            "fit_role": "D_R", "evaluation_role": "D_dev",
            "parameterization": "camera_to_corrected_observation_ray_frame",
            "runtime_output_frame": "map_bev_after_query_dependent_rotation",
            "residual_recentering": False, "spatial_grouping_unit": "position_id",
            "k_neighbors": K_NEIGHBORS, "length_scale_m": LENGTH_SCALE_M,
            "prior_standard_deviation_m": PRIOR_STD_M,
            "prior_strength": PRIOR_STRENGTH, "metrics": metrics,
            "homography_baseline": {
                "pixel_covariance": "sigma_px^2 I2",
                "propagation": "R_world = J_pixel_to_ground R_pixel J_pixel_to_ground^T",
                "position_balanced_sigma_px": sigma_px,
            },
        },
        "artifacts": {
            "residuals": {"path": residual_path.name, "sha256": sha256(residual_path)},
            "models": {"path": model_path.name, "sha256": sha256(model_path)},
        },
        "implementation_sha256": sha256(Path(__file__)),
    }
    (staging / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    print(json.dumps({
        "snapshot": report["snapshot"],
        "opportunity_outcomes": report["opportunity_outcomes"],
        "correction_error": report["correction_error"],
        "covariance_metrics": {
            name: {"nll": value["equal_drive_mean_nll"],
                   "coverage_95": value["equal_drive_coverage"]["95"],
                   "area_95_cm2": value["equal_drive_median_95_ellipse_area_cm2"]}
            for name, value in metrics.items()
        },
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
