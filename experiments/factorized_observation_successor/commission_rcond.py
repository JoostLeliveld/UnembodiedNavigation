#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math

import numpy as np
from scipy.optimize import nnls

import common as C

C.add_import_paths()
from reliability.projection import camera_model_from_world  # noqa: E402


def rows() -> list[dict]:
    detections = {}
    with C.R_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["detected"] == "1" and row["pu0"]:
                detections[row["sample_id"]] = row
    result = []
    with C.R_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            det = detections.get(row["sample_id"])
            if det is None:
                continue
            result.append({
                "sample_id": row["sample_id"], "camera": row["camera_id"],
                "x": float(row["robot_x"]), "y": float(row["robot_y"]),
                "yaw": float(row["robot_yaw"]), "range": float(row["camera_range_m"]),
                "u": 0.5 * (float(det["pu0"]) + float(det["pu1"])),
                "v": float(det["pv1"]),
            })
    return result


def block_id(x: float, y: float) -> int:
    bx = int(np.searchsorted(np.asarray([-3.9, 3.9]), x))
    by = int(np.searchsorted(np.asarray([-0.25]), y))
    return 2 * bx + by


def design(data: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    models = {
        camera: camera_model_from_world(C.WORLD, include_name={
            "camera_A": "external_camera", "camera_B": "external_camera_b",
            "camera_C": "external_camera_c", "camera_D": "external_camera_d",
        }[camera]) for camera in C.CAMERAS
    }
    error, ranges, cameras, blocks, xy = [], [], [], [], []
    for row in data:
        u0, v0, _ = models[row["camera"]].world_to_pixel(row["x"], row["y"], 0.0)
        error.append([row["u"] - u0, row["v"] - v0])
        ranges.append(row["range"])
        cameras.append(C.CAMERAS.index(row["camera"]))
        blocks.append(block_id(row["x"], row["y"]))
        xy.append([row["x"], row["y"]])
    return tuple(np.asarray(v) for v in (error, ranges, cameras, blocks, xy))


def fit(error: np.ndarray, ranges: np.ndarray, cameras: np.ndarray, geometry: bool) -> dict:
    bias = np.zeros((len(C.CAMERAS), 2))
    for camera in range(len(C.CAMERAS)):
        bias[camera] = np.mean(error[cameras == camera], axis=0)
    residual = error - bias[cameras]
    if geometry:
        X = np.column_stack([np.ones(len(ranges)), ranges**2])
        coeff = np.vstack([nnls(X, residual[:, axis] ** 2)[0] for axis in range(2)])
    else:
        coeff = np.column_stack([np.mean(residual**2, axis=0), np.zeros(2)])
    vu = np.maximum(coeff[0, 0] + coeff[0, 1] * ranges**2, 1e-3)
    vv = np.maximum(coeff[1, 0] + coeff[1, 1] * ranges**2, 1e-3)
    rho = float(np.clip(np.mean(residual[:, 0] * residual[:, 1] / np.sqrt(vu * vv)), -0.95, 0.95))
    return {"bias": bias, "coeff": coeff, "rho": rho, "geometry": geometry}


def predict(model: dict, ranges: np.ndarray, cameras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = model["coeff"]
    vu = np.maximum(coeff[0, 0] + coeff[0, 1] * ranges**2, 1e-3)
    vv = np.maximum(coeff[1, 0] + coeff[1, 1] * ranges**2, 1e-3)
    cov = np.zeros((len(ranges), 2, 2))
    cov[:, 0, 0], cov[:, 1, 1] = vu, vv
    cov[:, 0, 1] = cov[:, 1, 0] = model["rho"] * np.sqrt(vu * vv)
    return model["bias"][cameras], cov


def score(error: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> dict:
    centered = error - mean
    inv = np.linalg.inv(cov)
    d2 = np.einsum("ni,nij,nj->n", centered, inv, centered)
    logdet = np.linalg.slogdet(cov)[1]
    nll = 0.5 * (2 * math.log(2 * math.pi) + logdet + d2)
    return {
        "n": int(len(error)), "mean_nll": float(np.mean(nll)),
        "coverage_95": float(np.mean(d2 <= 5.9914645471)),
        "median_mahalanobis2": float(np.median(d2)),
        "rmse_u_px": float(np.sqrt(np.mean(centered[:, 0] ** 2))),
        "rmse_v_px": float(np.sqrt(np.mean(centered[:, 1] ** 2))),
    }


def main() -> None:
    manifest = C.OUT / "frozen/manifest.json"
    if not manifest.is_file():
        raise RuntimeError("run freeze_inputs.py first")
    C.assert_frozen((C.R_ROWS, C.R_INDEX, C.R_REGISTRY, C.WORLD))
    error, ranges, cameras, blocks, xy = design(rows())
    if set(np.unique(blocks)) != set(range(6)):
        raise RuntimeError(f"spatial folds incomplete: {np.unique(blocks).tolist()}")
    oof = {name: {"error": [], "mean": [], "cov": [], "folds": []} for name in ("constant", "geometry")}
    per_fold = []
    for fold in range(6):
        train, test = blocks != fold, blocks == fold
        entry = {"fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum())}
        for name, geometry in (("constant", False), ("geometry", True)):
            model = fit(error[train], ranges[train], cameras[train], geometry)
            mean, cov = predict(model, ranges[test], cameras[test])
            entry[name] = score(error[test], mean, cov)
            for key, value in (("error", error[test]), ("mean", mean), ("cov", cov)):
                oof[name][key].append(value)
        per_fold.append(entry)
    aggregate = {}
    for name in oof:
        aggregate[name] = score(*[np.concatenate(oof[name][key], axis=0) for key in ("error", "mean", "cov")])
    full_geometry = fit(error, ranges, cameras, True)
    range_endpoints = np.asarray([ranges.min(), ranges.max()])
    _, endpoint_cov = predict(full_geometry, range_endpoints, np.zeros(2, dtype=int))
    variance_change = float(np.max(np.abs(np.diagonal(endpoint_cov[1]) / np.diagonal(endpoint_cov[0]) - 1.0)))
    geometry_pass = (
        aggregate["geometry"]["mean_nll"] <= aggregate["constant"]["mean_nll"] - 0.001
        and abs(aggregate["geometry"]["coverage_95"] - 0.95)
        <= abs(aggregate["constant"]["coverage_95"] - 0.95) + 0.01
        and variance_change >= 0.01
    )
    selected = "geometry" if geometry_pass else "constant"
    final = fit(error, ranges, cameras, selected == "geometry")
    p = C.load_p()
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    gx, gy = np.meshgrid(xs, ys)
    all_ranges = []
    C.add_import_paths()
    models = {camera: camera_model_from_world(C.WORLD, include_name={
        "camera_A": "external_camera", "camera_B": "external_camera_b",
        "camera_C": "external_camera_c", "camera_D": "external_camera_d",
    }[camera]) for camera in C.CAMERAS}
    for camera in C.CAMERAS:
        pos = models[camera].cam_pos
        all_ranges.append(np.hypot(gx - pos[0], gy - pos[1]))
    grid_ranges = np.stack(all_ranges)
    grid_cameras = np.repeat(np.arange(4)[:, None, None], len(ys), axis=1)
    grid_cameras = np.repeat(grid_cameras, len(xs), axis=2)
    mean, cov = predict(final, grid_ranges.ravel(), grid_cameras.ravel())
    cov = cov.reshape(4, len(ys), len(xs), 2, 2)
    mean = mean.reshape(4, len(ys), len(xs), 2)
    artifact = C.OUT / "rcond/r_cond_uv.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact, xs=xs, ys=ys, camera_ids=np.asarray(C.CAMERAS),
        bias_u_px=mean[..., 0], bias_v_px=mean[..., 1],
        R_uu_px2=cov[..., 0, 0], R_uv_px2=cov[..., 0, 1], R_vv_px2=cov[..., 1, 1],
        selected_model=np.asarray([selected]), range_min_m=np.asarray([ranges.min()]),
        range_max_m=np.asarray([ranges.max()]),
    )
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registry_id": "PG-IPM-CURRENT", "n_detections": int(len(error)),
        "response": "detected bounding-box bottom-centre minus floor-contact projection, pixels",
        "split": "six leave-one-spatial-block-out folds; yaw replicates grouped by (x,y)",
        "models": aggregate, "per_fold": per_fold,
        "geometry_gate": "PASS" if geometry_pass else "FAIL",
        "geometry_variance_relative_change_over_range": variance_change,
        "geometry_gate_thresholds": {"minimum_nll_improvement": 0.001, "maximum_coverage_error_increase": 0.01, "minimum_variance_relative_change": 0.01},
        "selected_model": selected,
        "claim_allowed": "spatial R_cond" if geometry_pass else "constant R_cond only",
        "final_parameters": {"bias_uv_px": final["bias"].tolist(), "variance_coeff_a_b": final["coeff"].tolist(), "rho": final["rho"]},
        "artifact": str(artifact.relative_to(C.REPO)), "artifact_sha256": C.sha256(artifact),
        "ground_truth_role": "commissioning/evaluation only; not a runtime input",
    }
    C.write_json(C.OUT / "rcond/summary.json", summary)
    print(f"R_cond geometry gate: {summary['geometry_gate']} (selected {selected})")
    print(C.OUT / "rcond/summary.json")


if __name__ == "__main__":
    main()
