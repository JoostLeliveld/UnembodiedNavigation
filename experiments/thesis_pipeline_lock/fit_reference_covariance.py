#!/usr/bin/env python3
"""Fit canonical ray-frame R0/R1/R2 on D_R; select R2 on D_dev."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common"), str(REPO)]
from experiments.warehouse_v2_sketches.reference_dataset import load_rows  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
EIGENVALUE_FLOOR_M2 = 1.0e-6
PRIOR_STD_M = 10.0
PRIOR_STRENGTH = 2.5e-6
PRIOR_COVARIANCE = PRIOR_STD_M ** 2 * np.eye(2)
PRIOR_SCATTER = PRIOR_STRENGTH * PRIOR_COVARIANCE
K_GRID = (8, 16, 32)
LENGTH_GRID_M = (0.2, 0.3, 0.4, 0.6, 1.0, 1.4)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def psd(matrix: np.ndarray) -> np.ndarray:
    eig, vec = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vec * np.maximum(eig, EIGENVALUE_FLOOR_M2)) @ vec.T


def grouped_moments(xy, position, residual, camera=None):
    groups = defaultdict(list)
    for index, key in enumerate(position):
        group = (str(key), str(camera[index])) if camera is not None else (str(key),)
        groups[group].append(index)
    points, cameras, moments = [], [], []
    for key in sorted(groups):
        use = np.asarray(groups[key], dtype=int)
        points.append(np.mean(xy[use], axis=0))
        cameras.append(key[-1] if camera is not None else "all")
        moments.append(np.mean(np.einsum("ni,nj->nij", residual[use], residual[use]), axis=0))
    return np.asarray(points), np.asarray(cameras), np.asarray(moments)


def posterior_mean(scatter: np.ndarray, support: float | np.ndarray) -> np.ndarray:
    denominator = np.asarray(support, dtype=float) + PRIOR_STRENGTH
    if denominator.ndim == 0:
        return psd((scatter + PRIOR_SCATTER) / float(denominator))
    result = (scatter + PRIOR_SCATTER) / denominator[..., None, None]
    return np.stack([psd(matrix) for matrix in result])


def spatial_predict(train_xy, train_camera, train_moment, query_xy, query_camera,
                    k, length):
    result = np.empty((len(query_xy), 2, 2), dtype=float)
    for camera_id in CAMERAS:
        support = train_camera == camera_id; queries = np.flatnonzero(query_camera == camera_id)
        points, moments = train_xy[support], train_moment[support]
        for index in queries:
            distance2 = np.sum((points - query_xy[index]) ** 2, axis=1)
            count = min(k, len(points)); selected = np.argpartition(distance2, count - 1)[:count]
            weight = np.exp(-0.5 * distance2[selected] / length ** 2)
            scatter = np.einsum("n,nij->ij", weight, moments[selected])
            result[index] = posterior_mean(scatter, float(weight.sum()))
    return result


def metrics(residual, covariance, position):
    nll, nis = [], []
    for value, matrix in zip(residual, covariance):
        matrix = psd(matrix); inverse = np.linalg.inv(matrix)
        score = float(value @ inverse @ value)
        nis.append(score)
        nll.append(0.5 * (math.log(np.linalg.det(matrix)) + score + 2 * math.log(2 * math.pi)))
    nll, nis = np.asarray(nll), np.asarray(nis)
    by_position = []
    for key in sorted(set(position.tolist())):
        use = position == key
        by_position.append((float(nll[use].mean()), float(nis[use].mean()), float((nis[use] <= 5.991).mean())))
    return {
        "observations": len(residual), "positions": len(by_position),
        "equal_position_mean_nll": float(np.mean([x[0] for x in by_position])),
        "equal_position_mean_nis": float(np.mean([x[1] for x in by_position])),
        "equal_position_95pct_coverage": float(np.mean([x[2] for x in by_position])),
        "pooled_mean_nll": float(nll.mean()), "pooled_mean_nis": float(nis.mean()),
        "pooled_95pct_coverage": float((nis <= 5.991).mean()),
        "mean_logdet": float(np.mean([math.log(np.linalg.det(psd(x))) for x in covariance])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correction, output = args.correction.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists(): raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    correction_manifest_path = correction / "manifest.json"
    correction_manifest = json.loads(correction_manifest_path.read_text(encoding="utf-8"))
    if correction_manifest.get("status") != "frozen_before_covariance_fit":
        raise RuntimeError("correction is not frozen")
    prediction_path = correction / correction_manifest["artifacts"]["predictions"]["path"]
    if sha256(prediction_path) != correction_manifest["artifacts"]["predictions"]["sha256"]:
        raise RuntimeError("correction predictions hash drift")
    with np.load(prediction_path, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    rows = load_rows()
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    models = {c: camera_model_from_world(world, include_name=next(
        r["camera_model"] for r in rows if r["camera_id"] == c)) for c in CAMERAS}
    camera_xy = {c: np.asarray(models[c].cam_pos[:2], dtype=float) for c in CAMERAS}
    residual_world = data["residual_world_m"].astype(float)
    corrected = data["corrected_xy_m"].astype(float)
    residual_ray = np.empty_like(residual_world)
    for i, (camera_id, point, residual) in enumerate(zip(data["camera"], corrected, residual_world)):
        along = point - camera_xy[str(camera_id)]; along /= np.linalg.norm(along)
        basis = np.column_stack((along, np.asarray([-along[1], along[0]])))
        residual_ray[i] = basis.T @ residual
    fit, dev = data["role"] == "D_R", data["role"] == "D_dev"
    fit_xy, _, global_moments = grouped_moments(data["truth_xy_m"][fit], data["position_key"][fit], residual_ray[fit])
    r0 = posterior_mean(global_moments.sum(axis=0), float(len(global_moments)))
    r1 = {}
    for camera_id in CAMERAS:
        use = fit & (data["camera"] == camera_id)
        _, _, moments = grouped_moments(data["truth_xy_m"][use], data["position_key"][use], residual_ray[use])
        r1[camera_id] = posterior_mean(moments.sum(axis=0), float(len(moments)))
    spatial_xy, spatial_camera, spatial_moment = grouped_moments(
        data["truth_xy_m"][fit], data["position_key"][fit], residual_ray[fit], data["camera"][fit]
    )
    dev_r0 = np.repeat(r0[None], int(dev.sum()), axis=0)
    dev_r1 = np.stack([r1[str(value)] for value in data["camera"][dev]])
    candidates = []
    for k in K_GRID:
        for length in LENGTH_GRID_M:
            covariance = spatial_predict(
                spatial_xy, spatial_camera, spatial_moment, corrected[dev], data["camera"][dev],
                k, length,
            )
            score = metrics(residual_ray[dev], covariance, data["position_key"][dev])
            candidates.append({"k_neighbors": k, "length_scale_m": length, "metrics": score})
    calibrated = [candidate for candidate in candidates if abs(
        candidate["metrics"]["equal_position_95pct_coverage"] - 0.95) <= 0.01]
    selected = min(calibrated or candidates, key=lambda x: (
        x["metrics"]["equal_position_mean_nll"],
        abs(x["metrics"]["equal_position_95pct_coverage"] - 0.95),
        x["k_neighbors"], x["length_scale_m"],
    ))
    dev_r2 = spatial_predict(
        spatial_xy, spatial_camera, spatial_moment, corrected[dev], data["camera"][dev],
        selected["k_neighbors"], selected["length_scale_m"],
    )
    model_path = staging / "covariance_models.npz"
    np.savez_compressed(
        model_path, global_covariance_ray_m2=r0, camera_order=np.asarray(CAMERAS),
        prior_covariance_ray_m2=PRIOR_COVARIANCE,
        prior_strength=np.asarray([PRIOR_STRENGTH]),
        per_camera_covariance_ray_m2=np.stack([r1[c] for c in CAMERAS]),
        spatial_reference_xy_m=spatial_xy, spatial_second_moment_ray_m2=spatial_moment,
        spatial_camera=spatial_camera, k_neighbors=np.asarray([selected["k_neighbors"]]),
        length_scale_m=np.asarray([selected["length_scale_m"]]),
        eigenvalue_floor_m2=np.asarray([EIGENVALUE_FLOOR_M2]),
    )
    residual_path = staging / "corrected_residuals.npz"
    np.savez_compressed(
        residual_path, role=data["role"], position_key=data["position_key"], camera=data["camera"],
        truth_xy_m=data["truth_xy_m"], corrected_xy_m=corrected,
        residual_world_m=residual_world, residual_ray_m=residual_ray,
    )
    report = {
        "schema": "thesis_reference_covariance.v1", "status": "frozen_before_planning_information",
        "created_utc": datetime.now(timezone.utc).isoformat(), "final_audit_accessed": False,
        "fit_role": "D_R", "selection_role": "D_dev",
        "parameterization": "posterior mean of proper inverse-Wishart covariance model",
        "position_weighting": "average second moment within position, then equally across positions",
        "residual_recentering": False, "eigenvalue_floor_m2": EIGENVALUE_FLOOR_M2,
        "prior": {"standard_deviation_m": PRIOR_STD_M,
                  "covariance_ray_m2": PRIOR_COVARIANCE.tolist(),
                  "strength": PRIOR_STRENGTH,
                  "zero_support_covariance_ray_m2": PRIOR_COVARIANCE.tolist()},
        "candidate_grid": {"k_neighbors": K_GRID, "length_scale_m": LENGTH_GRID_M},
        "selection_rule": "minimum D_dev equal-position Gaussian NLL among candidates within one percentage point of 95-percent coverage",
        "selected_R2": selected,
        "D_dev_metrics": {
            "R0_global_full": metrics(residual_ray[dev], dev_r0, data["position_key"][dev]),
            "R1_per_camera_full": metrics(residual_ray[dev], dev_r1, data["position_key"][dev]),
            "R2_spatial_full": metrics(residual_ray[dev], dev_r2, data["position_key"][dev]),
        },
        "correction_manifest": str(correction_manifest_path.relative_to(REPO)),
        "correction_manifest_sha256": sha256(correction_manifest_path),
        "artifacts": {
            "models": {"path": model_path.name, "sha256": sha256(model_path)},
            "residuals": {"path": residual_path.name, "sha256": sha256(residual_path)},
        },
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = staging / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(report_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps({"status": report["status"], "selected_R2": selected,
                      "D_dev_metrics": report["D_dev_metrics"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
