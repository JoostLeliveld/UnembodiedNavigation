#!/usr/bin/env python3
"""Commission Bayesian bias and conditional covariance without runtime truth.

The current registered response is the YOLO bounding-box bottom centre in
pixels.  A finite-rank RBF Gaussian-process candidate is compared with a
per-camera/per-operational-visibility Normal-inverse-Wishart model using nested
spatial folds.  A failed spatial gate selects the constant model; it does not
turn a null spatial result into a planner failure.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge

import field_common as C

C.add_import_paths()
from reliability.projection import camera_model_from_world  # noqa: E402


CHI2_95 = 5.9914645471
PRIOR_SIGMA_PX = 5.0
PRIOR_KAPPA = 0.01
PRIOR_NU = 4.0
RBF_LENGTH_M = 3.0
MIN_SPATIAL_GROUP = 18
LOG_CHI_SQUARE_MEAN = -1.2703628454614782


def nearest_psd(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def spatial_block(x: float, y: float) -> int:
    bx = int(np.searchsorted(np.asarray([-3.9, 3.9]), x))
    by = int(np.searchsorted(np.asarray([-0.25]), y))
    return 2 * bx + by


def load_data() -> dict[str, np.ndarray]:
    C.assert_frozen()
    detections: dict[str, dict] = {}
    with C.R_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["detected"] == "1" and row["pu0"]:
                detections[row["sample_id"]] = row
    p = C.load_p()
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    models = {
        camera: camera_model_from_world(C.WORLD, include_name=C.INCLUDE_NAMES[camera])
        for camera in C.CAMERAS
    }
    records = []
    with C.R_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            detection = detections.get(row["sample_id"])
            if detection is None:
                continue
            camera = row["camera_id"]
            x, y = float(row["robot_x"]), float(row["robot_y"])
            u = 0.5 * (float(detection["pu0"]) + float(detection["pu1"]))
            v = float(detection["pv1"])
            expected_u, expected_v, _ = models[camera].world_to_pixel(x, y, 0.0)
            probability = float(C.sample(p[f"P_{camera}_map"], xs, ys, np.asarray([[x, y]]))[0])
            records.append((
                row["sample_id"], camera, x, y, float(row["robot_yaw"]),
                probability, C.visibility_mode(probability), u - expected_u, v - expected_v,
                spatial_block(x, y),
            ))
    if not records:
        raise RuntimeError("no registered detected observations")
    columns = list(zip(*records))
    return {
        "sample_id": np.asarray(columns[0]),
        "camera": np.asarray(columns[1]),
        "xy": np.column_stack([columns[2], columns[3]]).astype(float),
        "yaw": np.asarray(columns[4], dtype=float),
        "p_use": np.asarray(columns[5], dtype=float),
        "mode": np.asarray(columns[6]),
        "error": np.column_stack([columns[7], columns[8]]).astype(float),
        "block": np.asarray(columns[9], dtype=int),
    }


def niw_group(error: np.ndarray) -> dict:
    error = np.asarray(error, dtype=float)
    d = error.shape[1]
    n = len(error)
    mean = error.mean(axis=0) if n else np.zeros(d)
    centred = error - mean
    scatter = centred.T @ centred
    kappa = PRIOR_KAPPA + n
    nu = PRIOR_NU + n
    mu = (PRIOR_KAPPA * np.zeros(d) + n * mean) / kappa
    delta = mean
    psi0 = np.eye(d) * PRIOR_SIGMA_PX**2 * (PRIOR_NU - d - 1.0)
    psi = psi0 + scatter + PRIOR_KAPPA * n / kappa * np.outer(delta, delta)
    covariance = nearest_psd(psi / (nu - d - 1.0))
    return {
        "n": n,
        "bias": mu,
        "R": covariance,
        "bias_cov": covariance / kappa,
        "kappa": kappa,
        "nu": nu,
    }


def group_keys(camera: np.ndarray, mode: np.ndarray) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in zip(camera, mode)]


def fit_constant(data: dict[str, np.ndarray], take: np.ndarray) -> dict:
    fallback = niw_group(data["error"][take])
    groups = {}
    keys = group_keys(data["camera"][take], data["mode"][take])
    for key in sorted(set(keys)):
        local = take.copy()
        local &= data["camera"] == key[0]
        local &= data["mode"] == key[1]
        groups[key] = niw_group(data["error"][local])
    return {"kind": "constant", "groups": groups, "fallback": fallback}


def constant_predict(model: dict, camera: np.ndarray, mode: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(camera)
    mean = np.zeros((n, 2)); covariance = np.zeros((n, 2, 2)); bias_cov = np.zeros_like(covariance)
    for index, key in enumerate(group_keys(camera, mode)):
        group = model["groups"].get(key, model["fallback"])
        mean[index], covariance[index], bias_cov[index] = group["bias"], group["R"], group["bias_cov"]
    return mean, covariance, bias_cov


def rbf_centres() -> np.ndarray:
    return np.asarray([[x, y] for y in np.linspace(-7.5, 7.5, 5) for x in np.linspace(-10.0, 10.0, 6)])


def rbf_features(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    centres = rbf_centres()
    squared = np.sum((xy[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return np.column_stack([np.ones(len(xy)), np.exp(-0.5 * squared / RBF_LENGTH_M**2)])


def fit_spatial_group(xy: np.ndarray, error: np.ndarray, constant: dict) -> dict | None:
    if len(error) < MIN_SPATIAL_GROUP:
        return None
    features = rbf_features(xy)
    bias_models = []
    predicted = np.zeros_like(error)
    for axis in range(2):
        model = BayesianRidge(fit_intercept=False, compute_score=True, max_iter=300, tol=1e-6)
        model.fit(features, error[:, axis])
        bias_models.append(model)
        predicted[:, axis] = model.predict(features)
    residual = error - predicted
    variance_models = []
    for axis in range(2):
        target = np.log(residual[:, axis] ** 2 + 0.25)
        model = BayesianRidge(fit_intercept=False, compute_score=True, max_iter=300, tol=1e-6)
        model.fit(features, target)
        variance_models.append(model)
    raw_rho = float(np.corrcoef(residual.T)[0, 1]) if len(residual) > 2 else 0.0
    rho = float(np.clip(np.nan_to_num(raw_rho), -0.90, 0.90))
    return {
        "bias_models": bias_models,
        "variance_models": variance_models,
        "rho": rho,
        "constant": constant,
        "n": len(error),
    }


def fit_spatial(data: dict[str, np.ndarray], take: np.ndarray) -> dict:
    constant = fit_constant(data, take)
    groups = {}
    keys = group_keys(data["camera"][take], data["mode"][take])
    for key in sorted(set(keys)):
        local = take.copy()
        local &= data["camera"] == key[0]
        local &= data["mode"] == key[1]
        groups[key] = fit_spatial_group(data["xy"][local], data["error"][local], constant["groups"][key])
    return {"kind": "finite_rank_rbf_gp", "groups": groups, "constant": constant}


def spatial_group_predict(group: dict, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = rbf_features(xy)
    n = len(xy)
    mean = np.zeros((n, 2)); covariance = np.zeros((n, 2, 2)); bias_cov = np.zeros_like(covariance)
    constant_diag = np.diag(group["constant"]["R"])
    for axis, model in enumerate(group["bias_models"]):
        mean[:, axis] = model.predict(features)
        bias_cov[:, axis, axis] = np.einsum("ni,ij,nj->n", features, model.sigma_, features)
    diagonal = np.zeros((n, 2))
    for axis, model in enumerate(group["variance_models"]):
        log_variance = model.predict(features) - LOG_CHI_SQUARE_MEAN
        value = np.exp(np.clip(log_variance, -8.0, 12.0))
        diagonal[:, axis] = np.clip(value, 0.25 * constant_diag[axis], 4.0 * constant_diag[axis])
        covariance[:, axis, axis] = diagonal[:, axis]
    covariance[:, 0, 1] = covariance[:, 1, 0] = group["rho"] * np.sqrt(diagonal[:, 0] * diagonal[:, 1])
    return mean, covariance, bias_cov


def spatial_predict(model: dict, xy: np.ndarray, camera: np.ndarray, mode: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, covariance, bias_cov = constant_predict(model["constant"], camera, mode)
    keys = group_keys(camera, mode)
    for key in sorted(set(keys)):
        group = model["groups"].get(key)
        if group is None:
            continue
        local = np.asarray([item == key for item in keys], dtype=bool)
        mean[local], covariance[local], bias_cov[local] = spatial_group_predict(group, xy[local])
    return mean, covariance, bias_cov


def predict(model: dict, xy: np.ndarray, camera: np.ndarray, mode: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model["kind"] == "constant":
        return constant_predict(model, camera, mode)
    return spatial_predict(model, xy, camera, mode)


def score(error: np.ndarray, mean: np.ndarray, covariance: np.ndarray, camera: np.ndarray | None = None) -> dict:
    centred = np.asarray(error) - np.asarray(mean)
    covariance = np.asarray([nearest_psd(value) for value in covariance])
    inverse = np.linalg.inv(covariance)
    d2 = np.einsum("ni,nij,nj->n", centred, inverse, centred)
    nll = 0.5 * (2 * math.log(2 * math.pi) + np.linalg.slogdet(covariance)[1] + d2)
    result = {
        "n": int(len(error)),
        "mean_nll": float(np.mean(nll)),
        "coverage_95": float(np.mean(d2 <= CHI2_95)),
        "median_mahalanobis2": float(np.median(d2)),
        "rmse_u_px": float(np.sqrt(np.mean(centred[:, 0] ** 2))),
        "rmse_v_px": float(np.sqrt(np.mean(centred[:, 1] ** 2))),
    }
    if camera is not None:
        result["per_camera"] = {
            name: score(error[camera == name], mean[camera == name], covariance[camera == name])
            for name in C.CAMERAS if np.any(camera == name)
        }
    return result


def calibration_scale(error: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    centred = error - mean
    d2 = np.einsum("ni,nij,nj->n", centred, np.linalg.inv(covariance), centred)
    try:
        quantile = float(np.quantile(d2, 0.95, method="higher"))
    except TypeError:
        quantile = float(np.quantile(d2, 0.95, interpolation="higher"))
    return max(1.0, quantile / CHI2_95)


def nested_spatial_audit(data: dict[str, np.ndarray]) -> tuple[dict, dict]:
    blocks = data["block"]
    if set(np.unique(blocks)) != set(range(6)):
        raise RuntimeError(f"six spatial blocks required, got {np.unique(blocks).tolist()}")
    outputs = {
        name: {key: [] for key in ("error", "mean", "covariance", "camera")} | {"scales": []}
        for name in ("constant", "spatial")
    }
    folds = []
    for test_fold in range(6):
        calibration_fold = (test_fold + 1) % 6
        fit_mask = (blocks != test_fold) & (blocks != calibration_fold)
        calibration_mask = blocks == calibration_fold
        test_mask = blocks == test_fold
        entry = {
            "test_fold": test_fold,
            "calibration_fold": calibration_fold,
            "n_fit": int(fit_mask.sum()),
            "n_calibration": int(calibration_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        for name, fitter in (("constant", fit_constant), ("spatial", fit_spatial)):
            model = fitter(data, fit_mask)
            cal_mean, cal_r, cal_bias = predict(
                model, data["xy"][calibration_mask], data["camera"][calibration_mask], data["mode"][calibration_mask]
            )
            scale = calibration_scale(data["error"][calibration_mask], cal_mean, cal_r + cal_bias)
            mean, r_value, bias_value = predict(
                model, data["xy"][test_mask], data["camera"][test_mask], data["mode"][test_mask]
            )
            covariance = (r_value + bias_value) * scale
            outputs[name]["error"].append(data["error"][test_mask])
            outputs[name]["mean"].append(mean)
            outputs[name]["covariance"].append(covariance)
            outputs[name]["camera"].append(data["camera"][test_mask])
            outputs[name]["scales"].append(scale)
            entry[name] = score(data["error"][test_mask], mean, covariance)
        folds.append(entry)
    aggregate = {}
    for name, values in outputs.items():
        joined = {key: np.concatenate(values[key], axis=0) for key in ("error", "mean", "covariance", "camera")}
        aggregate[name] = score(joined["error"], joined["mean"], joined["covariance"], joined["camera"])
        aggregate[name]["nested_calibration_scales"] = [float(value) for value in values["scales"]]
        aggregate[name]["deployment_scale"] = float(np.median(values["scales"]))
    return aggregate, {"folds": folds}


def serial_group(group: dict) -> dict:
    return {
        "n": int(group["n"]),
        "bias_uv_px": np.asarray(group["bias"]).tolist(),
        "R_uv_px2": np.asarray(group["R"]).tolist(),
        "bias_posterior_cov_uv_px2": np.asarray(group["bias_cov"]).tolist(),
        "posterior_kappa": float(group["kappa"]),
        "posterior_nu": float(group["nu"]),
    }


def write_grid_artifact(data: dict[str, np.ndarray], selected_model: dict, selected: str, scale: float) -> Path:
    p = C.load_p()
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    shape = (len(C.CAMERAS), len(ys), len(xs))
    bias = np.zeros(shape + (2,)); r_value = np.zeros(shape + (2, 2)); bias_cov = np.zeros_like(r_value)
    modes = np.empty(shape, dtype="U8")
    for camera_index, camera in enumerate(C.CAMERAS):
        probability = np.asarray(p[f"P_{camera}_map"], float).ravel()
        local_modes = C.visibility_mode(probability)
        cameras = np.repeat(camera, len(xy))
        mean, covariance, mean_covariance = predict(selected_model, xy, cameras, local_modes)
        bias[camera_index] = mean.reshape(len(ys), len(xs), 2)
        r_value[camera_index] = (covariance * scale).reshape(len(ys), len(xs), 2, 2)
        bias_cov[camera_index] = (mean_covariance * scale).reshape(len(ys), len(xs), 2, 2)
        modes[camera_index] = local_modes.reshape(len(ys), len(xs))
    pooled = niw_group(data["error"])
    artifact = C.OUT / "commissioned/observation_field.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        xs=xs, ys=ys, camera_ids=np.asarray(C.CAMERAS), visibility_mode=modes,
        bias_u_px=bias[..., 0], bias_v_px=bias[..., 1],
        R_uu_px2=r_value[..., 0, 0], R_uv_px2=r_value[..., 0, 1], R_vv_px2=r_value[..., 1, 1],
        B_uu_px2=bias_cov[..., 0, 0], B_uv_px2=bias_cov[..., 0, 1], B_vv_px2=bias_cov[..., 1, 1],
        pooled_R_uv_px2=pooled["R"] * scale,
        pooled_bias_uv_px=pooled["bias"],
        pooled_bias_posterior_cov_uv_px2=pooled["bias_cov"] * scale,
        selected_model=np.asarray([selected]),
        visibility_threshold=np.asarray([C.VISIBILITY_THRESHOLD]),
        calibration_scale=np.asarray([scale]),
    )
    return artifact


def main() -> None:
    data = load_data()
    audit, detail = nested_spatial_audit(data)
    constant, spatial = audit["constant"], audit["spatial"]
    per_camera_ok = all(
        spatial["per_camera"][camera]["mean_nll"] <= constant["per_camera"][camera]["mean_nll"] + 0.05
        for camera in C.CAMERAS
    )
    full_spatial = fit_spatial(data, np.ones(len(data["error"]), dtype=bool))
    spatial_mean, spatial_r, _ = predict(full_spatial, data["xy"], data["camera"], data["mode"])
    relative_span = float(max(
        np.quantile(spatial_r[:, axis, axis], 0.90) / max(np.quantile(spatial_r[:, axis, axis], 0.10), 1e-9) - 1.0
        for axis in range(2)
    ))
    bias_span = float(np.max(np.linalg.norm(spatial_mean - np.median(spatial_mean, axis=0), axis=1)))
    checks = {
        "mean_nll_improves_by_0p01": spatial["mean_nll"] <= constant["mean_nll"] - 0.01,
        "coverage_error_not_worse_by_more_than_0p015": abs(spatial["coverage_95"] - 0.95) <= abs(constant["coverage_95"] - 0.95) + 0.015,
        "no_camera_nll_worse_by_more_than_0p05": per_camera_ok,
        "nondegenerate_spatial_effect": relative_span >= 0.10 or bias_span >= 0.50,
    }
    spatial_pass = all(checks.values())
    selected = "finite_rank_rbf_gp" if spatial_pass else "per_camera_visibility_niw"
    selected_model = full_spatial if spatial_pass else fit_constant(data, np.ones(len(data["error"]), dtype=bool))
    selected_audit = spatial if spatial_pass else constant
    scale = float(selected_audit["deployment_scale"])
    artifact = write_grid_artifact(data, selected_model, selected, scale)
    counts = {
        f"{camera}/{mode}": int(np.sum((data["camera"] == camera) & (data["mode"] == mode)))
        for camera in C.CAMERAS for mode in ("clear", "marginal")
    }
    minimum_group_ok = all(value >= 20 for value in counts.values())
    commissioning_checks = {
        "all_camera_visibility_groups_have_at_least_20_samples": minimum_group_ok,
        "nested_spatial_95pct_coverage_between_0p90_and_0p99": 0.90 <= selected_audit["coverage_95"] <= 0.99,
        "registered_current_response_only": True,
    }
    constant_model = fit_constant(data, np.ones(len(data["error"]), dtype=bool))
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "OFFLINE_COMMISSIONED" if all(commissioning_checks.values()) else "FAIL_CLOSED",
        "registry_id": "PG-IPM-CURRENT",
        "n_observations": int(len(data["error"])),
        "measurement_response": "detected YOLO bounding-box bottom centre minus floor-contact projection, pixels",
        "availability_input": "frozen A3 monocular-depth prior plus learned GP residual, per camera",
        "visibility_modes": {
            "definition": f"operational A3 p_use stratum: clear >= {C.VISIBILITY_THRESHOLD:.2f}, marginal < {C.VISIBILITY_THRESHOLD:.2f}",
            "counts": counts,
            "caveat": "These are operational probability strata, not rendered front/rear keypoint or physical occlusion labels.",
        },
        "bayesian_constant_model": {
            "prior": {"family": "Normal-inverse-Wishart", "sigma_px": PRIOR_SIGMA_PX, "kappa": PRIOR_KAPPA, "nu": PRIOR_NU},
            "groups": {f"{camera}/{mode}": serial_group(group) for (camera, mode), group in constant_model["groups"].items()},
        },
        "heldout_protocol": "six outer spatial blocks; the next block calibrates covariance; remaining four fit the model",
        "models": audit,
        "fold_detail": detail,
        "spatial_candidate": {
            "name": "finite-rank RBF Gaussian-process approximation for bias and log conditional variance",
            "length_scale_m": RBF_LENGTH_M,
            "relative_R_span_90_to_10": relative_span,
            "maximum_bias_departure_px": bias_span,
            "checks": checks,
            "gate": "PASS" if spatial_pass else "FAIL",
        },
        "selected_model": selected,
        "deployment_covariance_scale": scale,
        "commissioning_checks": commissioning_checks,
        "artifact": str(artifact.relative_to(C.REPO)),
        "artifact_sha256": C.sha256(artifact),
        "claim_allowed": (
            "spatial conditional bias and R field" if spatial_pass
            else "per-camera/per-operational-visibility constant conditional bias and R; ground ellipses may still vary through projection geometry"
        ),
        "keypoint_status": "NOT_COMMISSIONED_MULTI_CAMERA",
        "keypoint_next_evidence": "repeat the same grouped protocol on at least two sessions of four-camera keypoint data with usable front/rear-mode counts",
        "ground_truth_role": "offline commissioning and evaluation only; absent from runtime artifact inputs",
    }
    C.write_json(C.OUT / "commissioned/summary.json", payload)
    print(f"spatial field gate: {payload['spatial_candidate']['gate']}")
    print(f"selected: {selected}; commissioning: {payload['status']}")
    print(C.OUT / "commissioned/summary.json")


if __name__ == "__main__":
    main()
