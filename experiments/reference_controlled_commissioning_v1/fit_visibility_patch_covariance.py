#!/usr/bin/env python3
"""Fit covariance candidates matched to the visibility-residual correction.

Only that correction's whole-drive out-of-fold fitting residuals train covariance models. The
development drives select among constant, camera-specific, spatial, and
image-conditioned candidates.  Audit drives remain sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from analyze_commissioned_model import load_capture  # noqa: E402
from analyze_matched_covariance import (  # noqa: E402
    CHI2,
    CAMERAS,
    ConstantCovariance,
    SmoothCovariance,
    balanced_weights,
    dependence_diagnostics,
    gaussian_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def data_subset(data: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    return {
        key: value[selected] if isinstance(value, np.ndarray) and len(value) == len(data["drive"])
        else value
        for key, value in data.items()
    }


def isotropic(covariance: np.ndarray) -> np.ndarray:
    variance = 0.5 * np.trace(covariance, axis1=-2, axis2=-1)
    output = np.zeros_like(covariance)
    output[:, 0, 0] = variance
    output[:, 1, 1] = variance
    return output


def fit_only_calibration_scale(residual: np.ndarray, covariance: np.ndarray) -> float:
    inverse = np.linalg.inv(covariance)
    distance = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    return max(
        1.0,
        float(np.quantile(distance, 0.90) / CHI2[0.90]),
        float(np.quantile(distance, 0.95) / CHI2[0.95]),
    )


def development_calibration_scale(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Conservative scalar calibration on the designated development partition."""

    inverse = np.linalg.inv(covariance)
    distance = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    return max(
        1.0,
        float(np.mean(distance) / 2.0),
        float(np.quantile(distance, 0.90) / CHI2[0.90]),
        float(np.quantile(distance, 0.95) / CHI2[0.95]),
    )


def drive_oof_covariance(
    data: dict[str, Any],
    residual: np.ndarray,
    builder,
    query_data: dict[str, Any] | None = None,
) -> np.ndarray:
    """Predict every fitting drive from covariance models that exclude that drive."""

    result = np.empty((len(residual), 2, 2), dtype=float)
    drives = np.asarray(data["drive"])
    query_data = data if query_data is None else query_data
    for held in sorted(set(drives.tolist())):
        train_index = np.flatnonzero(drives != held)
        test_index = np.flatnonzero(drives == held)
        train = data_subset(data, train_index)
        test = data_subset(query_data, test_index)
        result[test_index] = builder(train, residual[train_index]).predict(test)
    return result


def temporal_dependence_at_stride(
    data: dict[str, Any], residual: np.ndarray, covariance: np.ndarray, stride: int
) -> dict[str, Any]:
    standardized = np.empty_like(residual)
    for index, (value, matrix) in enumerate(zip(residual, covariance, strict=True)):
        standardized[index] = np.linalg.solve(np.linalg.cholesky(matrix), value)
    values: list[float] = []
    series: dict[str, list[float]] = {}
    for drive_id in sorted(set(data["drive"].tolist())):
        for camera_id in CAMERAS:
            selected = np.flatnonzero(
                (data["drive"] == drive_id) & (data["camera"] == camera_id)
            )
            if len(selected) <= stride + 2:
                continue
            selected = selected[np.argsort(data["stamp"][selected])]
            local: list[float] = []
            for axis in range(2):
                correlation = float(np.corrcoef(
                    standardized[selected[:-stride], axis],
                    standardized[selected[stride:], axis],
                )[0, 1])
                if math.isfinite(correlation):
                    values.append(correlation)
                    local.append(correlation)
            series[f"{drive_id}:{camera_id}"] = local
    return {
        "stride_frames": int(stride),
        "median": float(np.median(values)),
        "median_absolute": float(np.median(np.abs(values))),
        "mean": float(np.mean(values)),
        "series": series,
    }


class ImageConditionedScale:
    """Positive scalar refinement of a spatial covariance.

    The spatial covariance remains the planner marginal.  Runtime-only box and
    visibility features scale its two-dimensional shape without rotating it.
    """

    def __init__(self, spatial: SmoothCovariance, alpha: float) -> None:
        self.spatial = spatial
        self.alpha = float(alpha)
        self.regressor = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.log_calibration = 0.0

    def fit(self, data: dict[str, Any], residual: np.ndarray) -> "ImageConditionedScale":
        base = self.spatial.predict(data)
        inverse = np.linalg.inv(base)
        energy = 0.5 * np.einsum("ni,nij,nj->n", residual, inverse, residual)
        target = np.log(np.clip(energy, 0.05, 20.0))
        weight = balanced_weights(data["drive"]) * len(data["drive"])
        self.regressor.fit(data["runtime_features"], target, ridge__sample_weight=weight)
        raw = self.regressor.predict(data["runtime_features"])
        scale = np.exp(np.clip(raw, math.log(0.25), math.log(4.0)))
        # One fit-only scalar restores E[d^2]=2 after the regression shrinkage.
        d2 = np.einsum("ni,nij,nj->n", residual, inverse, residual) / scale
        self.log_calibration = float(math.log(max(float(np.average(d2, weights=weight)) / 2.0, 1e-6)))
        return self

    def predict_scale(self, data: dict[str, Any]) -> np.ndarray:
        raw = self.regressor.predict(data["runtime_features"]) + self.log_calibration
        return np.exp(np.clip(raw, math.log(0.25), math.log(4.0)))

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        return self.spatial.predict(data) * self.predict_scale(data)[:, None, None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--visibility-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture_root = args.capture_root.resolve()
    visibility_root = args.visibility_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    comparison_path = visibility_root / "visibility_patch_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("selected") is not True or comparison.get("audit_analysis_permitted") is not False:
        raise RuntimeError("the visibility-residual correction must pass its frozen development gates with audit sealed")
    predictions_path = visibility_root / "candidate_predictions.npz"
    if sha256(predictions_path) != comparison["artifacts"]["predictions"]["sha256"]:
        raise RuntimeError("visibility-residual prediction artifact changed after selection")

    execution, records, rounds = load_capture(capture_root)
    with np.load(predictions_path, allow_pickle=False) as archive:
        drive = np.asarray(archive["drive_id"])
        partition = np.asarray(archive["partition"])
        camera = np.asarray(archive["camera_id"])
        stamp = np.asarray(archive["capture_stamp_ns"], dtype=np.int64)
        source_frame = np.asarray(archive["source_frame_id"])
        target = np.asarray(archive["target_ray_m"], dtype=float)
        correction = np.asarray(archive["candidate_prediction_ray_m"], dtype=float)
        visibility = np.asarray(archive["visibility_score"], dtype=float)
    if len(records) != len(drive):
        raise RuntimeError("capture and visibility-residual prediction row counts differ")
    for index, record in enumerate(records):
        identity = (record["drive"], record["camera"], record["stamp_ns"], record["source_frame_id"])
        predicted_identity = (str(drive[index]), str(camera[index]), int(stamp[index]), str(source_frame[index]))
        if identity != predicted_identity:
            raise RuntimeError(f"visibility-residual prediction identity mismatch at row {index}")
    if not np.isfinite(correction).all():
        raise RuntimeError("visibility-residual predictions contain non-finite rows")

    yaw = {(row["drive"], row["stamp_ns"]): row["reference_yaw"] for row in rounds}
    reference_xy = np.stack([record["reference"] for record in records])
    heading = np.asarray([yaw[(record["drive"], record["stamp_ns"])] for record in records])
    range_m = np.asarray([record["range_m"] for record in records])
    bearing = np.asarray([record["bearing_rad"] for record in records])
    runtime_features = np.column_stack((
        visibility,
        np.asarray([record["bbox_w"] / 1280.0 for record in records]),
        np.asarray([record["bbox_h"] / 720.0 for record in records]),
        np.asarray([record["confidence"] for record in records]),
        range_m,
        np.sin(bearing),
        np.cos(bearing),
        *[np.asarray(camera == value, dtype=float) for value in CAMERAS],
    ))
    basis = np.stack([record["basis"] for record in records])
    residual = correction - target
    raw_xy = np.stack([record["raw"] for record in records])
    corrected_xy = raw_xy + np.einsum("nij,nj->ni", basis, correction)
    data = {
        "drive": drive,
        "camera": camera,
        "stamp": stamp,
        "features": np.column_stack((reference_xy, heading, range_m)),
        "runtime_features": runtime_features,
        "basis": basis,
    }
    runtime_data = dict(data)
    runtime_data["features"] = np.column_stack((corrected_xy, heading, range_m))
    fit_index = np.flatnonzero(partition == "fit")
    development_index = np.flatnonzero(partition == "development")
    fit = data_subset(data, fit_index)
    fit_runtime = data_subset(runtime_data, fit_index)
    development = data_subset(runtime_data, development_index)
    development_planner = data_subset(data, development_index)
    fit_residual = residual[fit_index]
    development_residual = residual[development_index]

    candidates: dict[str, tuple[Any, np.ndarray, dict[str, Any]]] = {}
    global_builder = lambda d, r: ConstantCovariance(False).fit(d, r)
    global_full = global_builder(fit, fit_residual)
    global_oof = isotropic(drive_oof_covariance(
        fit, fit_residual, global_builder, fit_runtime
    ))
    global_scale = fit_only_calibration_scale(fit_residual, global_oof)
    global_cov = isotropic(global_full.predict(development)) * global_scale
    global_metrics = gaussian_metrics(development, development_residual, global_cov)
    global_metrics["fit_oof_calibration_scale"] = global_scale
    candidates["R0_global_isotropic"] = (
        global_full, global_cov, global_metrics
    )
    camera_builder = lambda d, r: ConstantCovariance(True).fit(d, r)
    camera_full = camera_builder(fit, fit_residual)
    camera_oof = drive_oof_covariance(fit, fit_residual, camera_builder, fit_runtime)
    camera_iso_oof = isotropic(camera_oof)
    camera_iso_scale = fit_only_calibration_scale(fit_residual, camera_iso_oof)
    camera_full_scale = fit_only_calibration_scale(fit_residual, camera_oof)
    camera_cov = camera_full.predict(development) * camera_full_scale
    camera_iso = isotropic(camera_full.predict(development)) * camera_iso_scale
    camera_iso_metrics = gaussian_metrics(development, development_residual, camera_iso)
    camera_iso_metrics["fit_oof_calibration_scale"] = camera_iso_scale
    camera_full_metrics = gaussian_metrics(development, development_residual, camera_cov)
    camera_full_metrics["fit_oof_calibration_scale"] = camera_full_scale
    candidates["R1_per_camera_isotropic"] = (
        camera_full, camera_iso, camera_iso_metrics
    )
    candidates["R2_per_camera_full"] = (
        camera_full, camera_cov, camera_full_metrics
    )

    spatial_options = []
    for position_scale in (0.75, 1.5, 3.0):
        for heading_scale in (0.75, 1.5, 100.0):
            for prior in (10.0, 30.0, 100.0):
                model = SmoothCovariance(position_scale, heading_scale, prior).fit(fit, fit_residual)
                builder = lambda d, r, ps=position_scale, hs=heading_scale, pr=prior: (
                    SmoothCovariance(ps, hs, pr).fit(d, r)
                )
                oof_covariance = drive_oof_covariance(
                    fit, fit_residual, builder, fit_runtime
                )
                scale = fit_only_calibration_scale(fit_residual, oof_covariance)
                covariance = model.predict(development) * scale
                metrics = gaussian_metrics(development, development_residual, covariance)
                metrics["fit_oof_calibration_scale"] = scale
                spatial_options.append((metrics["equal_drive_nll"], position_scale,
                                        heading_scale, prior, model, covariance, metrics))
    _, pscale, hscale, prior, spatial, spatial_cov, spatial_metrics = min(spatial_options)
    spatial_metrics.update({
        "position_scale_m": pscale,
        "heading_scale": hscale,
        "prior_strength": prior,
    })
    candidates["R3_spatial_full"] = (spatial, spatial_cov, spatial_metrics)

    image_options = []
    for alpha in (1.0, 10.0, 100.0):
        model = ImageConditionedScale(spatial, alpha).fit(fit, fit_residual)
        def image_builder(d, r, a=alpha, ps=pscale, hs=hscale, pr=prior):
            base = SmoothCovariance(ps, hs, pr).fit(d, r)
            return ImageConditionedScale(base, a).fit(d, r)
        oof_covariance = drive_oof_covariance(
            fit, fit_residual, image_builder, fit_runtime
        )
        scale = fit_only_calibration_scale(fit_residual, oof_covariance)
        covariance = model.predict(development) * scale
        metrics = gaussian_metrics(development, development_residual, covariance)
        metrics["fit_oof_calibration_scale"] = scale
        image_options.append((metrics["equal_drive_nll"], alpha, model, covariance, metrics))
    _, alpha, image_model, image_cov, image_metrics = min(image_options)
    image_metrics.update({
        "ridge_alpha": alpha,
        "planner_marginal": "R3_spatial_full",
        "runtime_scale_range": [0.25, 4.0],
    })
    candidates["R4_image_conditioned_scale"] = (
        image_model, image_cov, image_metrics
    )

    # Development is explicitly the pre-audit selection and calibration partition.
    # Retain the uncalibrated metrics so the transfer gap is visible, then freeze one
    # scalar per candidate.  The audit is still sealed at this point.
    calibrated: dict[str, tuple[Any, np.ndarray, dict[str, Any]]] = {}
    for name, (model, covariance, metrics) in candidates.items():
        scale = development_calibration_scale(development_residual, covariance)
        calibrated_covariance = covariance * scale
        calibrated_metrics = gaussian_metrics(
            development, development_residual, calibrated_covariance
        )
        calibrated_metrics["development_calibration_scale"] = scale
        calibrated_metrics["pre_development_calibration"] = metrics
        calibrated[name] = (model, calibrated_covariance, calibrated_metrics)
    candidates = calibrated

    selected_name = min(candidates, key=lambda name: candidates[name][2]["equal_drive_nll"])
    selected_covariance = candidates[selected_name][1]
    dependence = dependence_diagnostics(development, development_residual, selected_covariance)
    deployed_temporal = temporal_dependence_at_stride(
        development, development_residual, selected_covariance, stride=5
    )
    persistent_required = bool(
        deployed_temporal["median_absolute"] > 0.30
        or abs(dependence["simultaneous_pair_cosine_mean"]) > 0.20
    )
    report = {
        "schema": "visibility_patch_matched_covariance.v1",
        "status": "development_selection_complete",
        "audit_analysis_permitted": False,
        "capture_root": str(capture_root),
        "visibility_root": str(visibility_root),
        "mean_model": "box_mlp_visibility_residual",
        "fit_drives": sorted(set(drive[fit_index].tolist())),
        "development_drives": sorted(set(drive[development_index].tolist())),
        "fit_rows": int(len(fit_index)),
        "development_rows": int(len(development_index)),
        "candidates": {name: value[2] for name, value in candidates.items()},
        "selected_covariance": selected_name,
        "selection_rule": "minimum equal-development-drive Gaussian negative log likelihood",
        "calibration_rule": (
            "fit covariance structure on out-of-fold fit residuals; calibrate one "
            "conservative scalar on development before opening audit"
        ),
        "dependence_diagnostic": {
            "persistent_bias_model_required": persistent_required,
            "capture_rate_hz": 5.0,
            "deployed_fusion_rate_hz": 1.0,
            "deployed_temporal_dependence": deployed_temporal,
            "thresholds": {
                "deployed_lag_median_absolute": 0.30,
                "simultaneous_pair_cosine_absolute_mean": 0.20,
            },
            "five_hz_diagnostic": dependence,
        },
        "planner_covariance": (
            "R3_spatial_full" if selected_name == "R4_image_conditioned_scale" else selected_name
        ),
        "source_hashes": {
            "campaign_execution.json": sha256(capture_root / "campaign_execution.json"),
            "visibility_patch_comparison.json": sha256(comparison_path),
            "candidate_predictions.npz": sha256(predictions_path),
            str(Path(__file__).name): sha256(Path(__file__)),
        },
    }
    atomic_json(output / "matched_covariance_report.json", report)
    joblib.dump({
        "schema": "visibility_patch_covariance_development.v1",
        "selected_covariance": selected_name,
        "planner_covariance": report["planner_covariance"],
        "selected_development_calibration_scale": candidates[selected_name][2][
            "development_calibration_scale"
        ],
        "spatial_model": spatial,
        "image_model": image_model,
        "runtime_feature_names": [
            "visibility_score", "bbox_width_fraction", "bbox_height_fraction",
            "detector_confidence", "range_m", "bearing_sin", "bearing_cos",
            *[f"is_{camera_id}" for camera_id in CAMERAS],
        ],
    }, output / "covariance_development_models.joblib")
    report["artifacts"] = {
        "models": {
            "path": "covariance_development_models.joblib",
            "sha256": sha256(output / "covariance_development_models.joblib"),
        }
    }
    selected_metrics = candidates[selected_name][2]
    selected_pre = selected_metrics["pre_development_calibration"]
    selected_external_scale = float(
        selected_pre.get("fit_oof_calibration_scale", 1.0)
        * selected_metrics["development_calibration_scale"]
    )
    planner_metrics = candidates["R3_spatial_full"][2]
    planner_pre = planner_metrics["pre_development_calibration"]
    planner_external_scale = float(
        planner_pre.get("fit_oof_calibration_scale", 1.0)
        * planner_metrics["development_calibration_scale"]
    )
    planner_development_covariance = (
        spatial.predict(development_planner) * planner_external_scale
    )
    report["planner_marginal_development"] = gaussian_metrics(
        development_planner, development_residual, planner_development_covariance
    )
    spatial_camera = np.asarray(spatial.data["camera"])
    spatial_base = np.stack([spatial.base.values[value] for value in CAMERAS])
    ridge = image_model.regressor.named_steps["ridge"]
    scaler = image_model.regressor.named_steps["standardscaler"]
    parameter_path = output / "commissioned_visibility_parameters.npz"
    np.savez_compressed(
        parameter_path,
        spatial_camera=spatial_camera,
        spatial_features=np.asarray(spatial.data["features"], dtype=float),
        spatial_residual=np.asarray(spatial.residual, dtype=float),
        spatial_sample_weight=np.asarray(spatial.sample_weight, dtype=float),
        spatial_base=spatial_base,
        image_feature_mean=np.asarray(scaler.mean_, dtype=float),
        image_feature_scale=np.asarray(scaler.scale_, dtype=float),
        image_ridge_coef=np.asarray(ridge.coef_, dtype=float),
        image_ridge_intercept=np.asarray(ridge.intercept_, dtype=float),
    )
    protocol_path = Path(execution["protocol"]).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    from build_tables import _camera_positions
    camera_xy = _camera_positions(protocol)
    correction_base_path = visibility_root / "box_mlp_fit_only.joblib"
    correction_patch_path = visibility_root / "box_mlp_visibility_residual_fit_only.pt"
    runtime_model = {
        "schema": "commissioned_visibility_sensor_model.v1",
        "status": "frozen_before_audit",
        "audit_accessed": False,
        "mean_model": "box_mlp_visibility_residual",
        "runtime_covariance_model": selected_name,
        "planner_covariance_model": "R3_spatial_full",
        "camera_order": list(CAMERAS),
        "camera_xy_m": {key: list(map(float, value)) for key, value in camera_xy.items()},
        "image_shape_hw": [720, 1280],
        "correction_feature_names": [
            "range_m", "inv_range", "bearing_sin", "bearing_cos",
            "bbox_width_fraction", "bbox_height_fraction", "bbox_aspect",
            "bbox_bottom_u_fraction", "bbox_bottom_v_fraction",
            "detector_confidence", *[f"is_{value}" for value in CAMERAS],
        ],
        "visibility_grid": {"shape": [1, 16, 16], "definition": "soft_target_blue_in_bbox"},
        "correction_base": {
            "path": str(correction_base_path.resolve()),
            "sha256": sha256(correction_base_path),
        },
        "correction_patch": {
            "path": str(correction_patch_path.resolve()),
            "sha256": sha256(correction_patch_path),
        },
        "parameters": {
            "path": str(parameter_path.resolve()),
            "sha256": sha256(parameter_path),
        },
        "spatial": {
            "position_scale_m": float(spatial.position_scale),
            "heading_scale": float(spatial.heading_scale),
            "prior_strength": float(spatial.prior_strength),
            "external_calibration_scale": planner_external_scale,
        },
        "image_scale": {
            "ridge_alpha": float(alpha),
            "internal_log_calibration": float(image_model.log_calibration),
            "minimum": 0.25,
            "maximum": 4.0,
            "external_calibration_scale": selected_external_scale,
            "feature_names": [
                "visibility_score", "bbox_width_fraction", "bbox_height_fraction",
                "detector_confidence", "range_m", "bearing_sin", "bearing_cos",
                *[f"is_{camera_id}" for camera_id in CAMERAS],
            ],
        },
        "runtime_query": {
            "position": "visibility-residual-corrected observation position",
            "heading": "capture-time prior belief heading",
            "fusion_rate_hz": 1.0,
            "detector_rate_hz": 5.0,
        },
        "planner_query": {
            "position": "predicted belief mean position",
            "heading": "predicted belief mean heading",
        },
        "source_hashes": report["source_hashes"],
    }
    runtime_path = output / "commissioned_visibility_runtime_model.json"
    atomic_json(runtime_path, runtime_model)
    report["artifacts"].update({
        "runtime_model": {"path": runtime_path.name, "sha256": sha256(runtime_path)},
        "parameters": {"path": parameter_path.name, "sha256": sha256(parameter_path)},
    })
    atomic_json(output / "matched_covariance_report.json", report)
    print(json.dumps({
        "selected_covariance": selected_name,
        "planner_covariance": report["planner_covariance"],
        "equal_drive_nll": {
            name: round(value[2]["equal_drive_nll"], 4) for name, value in candidates.items()
        },
        "coverage_95": {
            name: round(value[2]["coverage"]["0.95"], 4) for name, value in candidates.items()
        },
        "persistent_bias_model_required": persistent_required,
        "deployed_temporal_dependence": deployed_temporal,
        "simultaneous_pair_cosine_mean": dependence["simultaneous_pair_cosine_mean"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
