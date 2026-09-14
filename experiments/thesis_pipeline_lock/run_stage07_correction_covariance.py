#!/usr/bin/env python3
"""Run spatially cross-validated correction and conditional-covariance commissioning."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from reliability.projection import camera_model_from_world


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FEATURE_NAMES = (
    "raw_range_m", "inverse_raw_range", "bbox_width_px", "bbox_height_px",
    "bbox_aspect", "bottom_u_norm", "bottom_v_norm", "confidence",
    "width_ratio", "height_ratio", "sin_heading_to_camera", "cos_heading_to_camera",
    "camera_A", "camera_B", "camera_C", "camera_D", "camera_E",
)
CHI2 = {"50": 1.38629436112, "90": 4.60517018599, "95": 5.99146454711, "99": 9.21034037198}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def percentile(values, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if not len(ordered):
        return math.nan
    index = (len(ordered) - 1) * probability
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "median_m": float(np.median(values)),
        "rms_m": float(np.sqrt(np.mean(values ** 2))),
        "p90_m": percentile(values, 0.90),
        "p95_m": percentile(values, 0.95),
        "maximum_m": float(np.max(values)),
        "above_0_25_m": int(np.sum(values > 0.25)),
        "above_0_25_fraction": float(np.mean(values > 0.25)),
    }


def load_rows(path: Path) -> list[dict]:
    numeric = {
        "pose_id", "position_id", "fold", "heading_id", "semantic_robot_pixels",
        "raw_best_confidence", "best_box_x0", "best_box_y0", "best_box_x1", "best_box_y1",
        "expected_box_x0", "expected_box_y0", "expected_box_x1", "expected_box_y1",
        "raw_ground_x", "raw_ground_y", "equivalent_x", "equivalent_y", "robot_x",
        "robot_y", "robot_yaw", "camera_range_m", "raw_ground_error_m", "equivalent_error_m",
    }
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if source["admitted"] != "1":
                continue
            row = dict(source)
            for key in numeric:
                row[key] = float(source[key])
            row["pose_id"] = int(row["pose_id"])
            row["position_id"] = int(row["position_id"])
            row["fold"] = int(row["fold"])
            row["heading_id"] = int(row["heading_id"])
            rows.append(row)
    if not rows or any(row["camera_id"] not in CAMERAS for row in rows):
        raise RuntimeError("Invalid admitted Stage-06 population")
    return rows


def feature_matrix(rows: list[dict], camera_xy: dict[str, tuple[float, float]]) -> np.ndarray:
    features = []
    for row in rows:
        width = row["best_box_x1"] - row["best_box_x0"]
        height = row["best_box_y1"] - row["best_box_y0"]
        predicted_width = row["expected_box_x1"] - row["expected_box_x0"]
        predicted_height = row["expected_box_y1"] - row["expected_box_y0"]
        cx, cy = camera_xy[row["camera_id"]]
        raw_range = math.hypot(row["raw_ground_x"] - cx, row["raw_ground_y"] - cy)
        bearing = math.atan2(cy - row["robot_y"], cx - row["robot_x"])
        relative = row["robot_yaw"] - bearing
        one_hot = [float(row["camera_id"] == camera) for camera in CAMERAS]
        features.append([
            raw_range,
            1.0 / max(raw_range, 1e-6),
            width,
            height,
            width / max(height, 1e-6),
            0.5 * (row["best_box_x0"] + row["best_box_x1"]) / 1280.0,
            row["best_box_y1"] / 720.0,
            row["raw_best_confidence"],
            width / predicted_width,
            height / predicted_height,
            math.sin(relative),
            math.cos(relative),
            *one_hot,
        ])
    matrix = np.asarray(features, dtype=float)
    if matrix.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise RuntimeError("Stage-07 runtime feature matrix is invalid")
    return matrix


def ray_bases(rows: list[dict], camera_xy: dict[str, tuple[float, float]]) -> np.ndarray:
    bases = []
    for row in rows:
        cx, cy = camera_xy[row["camera_id"]]
        vector = np.array([row["raw_ground_x"] - cx, row["raw_ground_y"] - cy], dtype=float)
        vector /= np.linalg.norm(vector)
        bases.append(np.column_stack([vector, np.array([-vector[1], vector[0]])]))
    return np.asarray(bases)


def apply_bias(
    predictions: np.ndarray,
    truth: np.ndarray,
    camera_ids: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    corrected = predictions.copy()
    biases = {}
    global_bias = np.mean(truth[train] - predictions[train], axis=0)
    for camera in CAMERAS:
        indexes = train[camera_ids[train] == camera]
        bias = global_bias if not len(indexes) else np.mean(truth[indexes] - predictions[indexes], axis=0)
        biases[camera] = [float(value) for value in bias]
        target = test[camera_ids[test] == camera]
        corrected[target] += bias
    return corrected, biases


class ResidualMLP(torch.nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputs, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, values):
        return self.layers(values)


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int,
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
    patience: int = 20,
) -> tuple[ResidualMLP, int, float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ResidualMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_loss = math.inf
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(len(x_tensor), generator=generator)
        model.train()
        for start in range(0, len(permutation), 128):
            indexes = permutation[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x_tensor[indexes]) - y_tensor[indexes]) ** 2)
            loss.backward()
            optimizer.step()
        if x_validation is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_loss = float(loss.detach())
            continue
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor(x_validation, dtype=torch.float32))
            validation_loss = float(torch.mean(
                (prediction - torch.tensor(y_validation, dtype=torch.float32)) ** 2
            ))
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def mlp_predict(model: ResidualMLP, matrix: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(matrix, dtype=torch.float32)).numpy().astype(float)


def correction_summaries(
    predictions: dict[str, np.ndarray], truth: np.ndarray, rows: list[dict]
) -> dict[str, dict]:
    blocks = sorted({row["block_id"] for row in rows})
    cameras = np.asarray([row["camera_id"] for row in rows])
    folds = np.asarray([row["fold"] for row in rows])
    block_ids = np.asarray([row["block_id"] for row in rows])
    reports = {}
    for name, values in predictions.items():
        errors = np.linalg.norm(values - truth, axis=1)
        block_medians = [float(np.median(errors[block_ids == block])) for block in blocks]
        block_p95s = [percentile(errors[block_ids == block], 0.95) for block in blocks]
        reports[name] = {
            "pooled": summary(errors),
            "block_macro_median_m": float(np.mean(block_medians)),
            "block_macro_p95_m": float(np.mean(block_p95s)),
            "by_camera": {camera: summary(errors[cameras == camera]) for camera in CAMERAS},
            "by_fold": {str(fold): summary(errors[folds == fold]) for fold in range(5)},
        }
    return reports


def bootstrap_differences(
    errors_a: np.ndarray, errors_b: np.ndarray, blocks: np.ndarray, seed: int
) -> dict:
    unique = sorted(set(blocks.tolist()))
    per_block = {}
    for block in unique:
        selector = blocks == block
        per_block[block] = (
            float(np.median(errors_a[selector]) - np.median(errors_b[selector])),
            percentile(errors_a[selector], 0.95) - percentile(errors_b[selector], 0.95),
        )
    rng = np.random.default_rng(seed)
    samples = np.empty((2000, 2), dtype=float)
    values = np.asarray([per_block[block] for block in unique])
    for index in range(len(samples)):
        draw = rng.integers(0, len(unique), size=len(unique))
        samples[index] = np.mean(values[draw], axis=0)
    return {
        "definition": "first model minus second model; whole-block bootstrap",
        "block_count": len(unique),
        "median_difference_m": {
            "estimate": float(np.mean(values[:, 0])),
            "ci95": [percentile(samples[:, 0], 0.025), percentile(samples[:, 0], 0.975)],
        },
        "p95_difference_m": {
            "estimate": float(np.mean(values[:, 1])),
            "ci95": [percentile(samples[:, 1], 0.025), percentile(samples[:, 1], 0.975)],
        },
    }


def spd(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    clipped = int(np.any(values < 1e-6))
    values = np.maximum(values, 1e-6)
    return vectors @ np.diag(values) @ vectors.T, clipped


def covariance(values: np.ndarray) -> tuple[np.ndarray, int]:
    if len(values) < 2:
        return np.eye(2) * 1e-4, 1
    centered = values - np.mean(values, axis=0)
    return spd(centered.T @ centered / len(values))


def covariance_scores(covariances: np.ndarray, residuals: np.ndarray, blocks: np.ndarray) -> dict:
    nll, d2, areas = [], [], []
    for matrix, residual in zip(covariances, residuals, strict=True):
        inverse = np.linalg.inv(matrix)
        determinant = float(np.linalg.det(matrix))
        distance = float(residual @ inverse @ residual)
        d2.append(distance)
        nll.append(0.5 * (2 * math.log(2 * math.pi) + math.log(determinant) + distance))
        areas.append(math.pi * CHI2["95"] * math.sqrt(determinant))
    nll = np.asarray(nll); d2 = np.asarray(d2); areas = np.asarray(areas)
    block_macro = np.mean([np.mean(nll[blocks == block]) for block in sorted(set(blocks.tolist()))])
    return {
        "mean_nll": float(np.mean(nll)),
        "block_macro_mean_nll": float(block_macro),
        "containment": {key: float(np.mean(d2 <= value)) for key, value in CHI2.items()},
        "mean_95_ellipse_area_m2": float(np.mean(areas)),
        "median_95_ellipse_area_m2": float(np.median(areas)),
        "mahalanobis_above_99_fraction": float(np.mean(d2 > CHI2["99"])),
    }


def fit_covariance_candidates(
    residuals: np.ndarray,
    features: np.ndarray,
    rows: list[dict],
) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    folds = np.asarray([row["fold"] for row in rows])
    cameras = np.asarray([row["camera_id"] for row in rows])
    blocks = np.asarray([row["block_id"] for row in rows])
    widths = np.asarray([row["best_box_x1"] - row["best_box_x0"] for row in rows])
    predictions = {
        name: np.empty((len(rows), 2, 2))
        for name in ("R0", "R1", "R2", "R2C", "R3")
    }
    scored_residuals = np.empty_like(residuals)
    floor_counts = Counter()
    fold_parameters = {}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        means = {}
        centered_train = residuals[train].copy()
        centered_test = residuals[test].copy()
        for camera in CAMERAS:
            train_local = np.flatnonzero(cameras[train] == camera)
            mean = np.mean(centered_train[train_local], axis=0)
            means[camera] = mean
            centered_train[cameras[train] == camera] -= mean
            centered_test[cameras[test] == camera] -= mean
        scored_residuals[test] = centered_test

        pooled, clipped = covariance(centered_train)
        floor_counts["R0"] += clipped
        predictions["R0"][test] = pooled
        per_camera = {}
        for camera in CAMERAS:
            matrix, clipped = covariance(centered_train[cameras[train] == camera])
            per_camera[camera] = matrix
            floor_counts["R1"] += clipped
            predictions["R1"][test[cameras[test] == camera]] = matrix

        width_entries = {}
        for camera in CAMERAS:
            train_mask = cameras[train] == camera
            camera_widths = widths[train][train_mask]
            edges = np.quantile(camera_widths, [1 / 3, 2 / 3])
            entries = []
            for bin_index in range(3):
                low = -math.inf if bin_index == 0 else edges[bin_index - 1]
                high = math.inf if bin_index == 2 else edges[bin_index]
                mask = train_mask & (widths[train] > low) & (widths[train] <= high)
                sample, clipped = covariance(centered_train[mask])
                floor_counts["R2_bin"] += clipped
                n = int(np.sum(mask)); weight = n / (n + 20.0)
                shrunk, clipped = spd(weight * sample + (1.0 - weight) * per_camera[camera])
                floor_counts["R2_shrunk"] += clipped
                entries.append(shrunk)
            width_entries[camera] = {"edges": edges, "covariances": entries}
            for index in test[cameras[test] == camera]:
                bin_index = int(np.searchsorted(edges, widths[index], side="left"))
                predictions["R2"][index] = entries[bin_index]

        nested_d2 = []
        for inner_fold in sorted(set(folds[train].tolist())):
            inner_fit = np.flatnonzero((folds != fold) & (folds != inner_fold))
            inner_cal = np.flatnonzero(folds == inner_fold)
            inner_fit_centered = residuals[inner_fit].copy()
            inner_cal_centered = residuals[inner_cal].copy()
            inner_per_camera = {}
            for camera in CAMERAS:
                inner_mean = np.mean(inner_fit_centered[cameras[inner_fit] == camera], axis=0)
                inner_fit_centered[cameras[inner_fit] == camera] -= inner_mean
                inner_cal_centered[cameras[inner_cal] == camera] -= inner_mean
                inner_per_camera[camera] = covariance(
                    inner_fit_centered[cameras[inner_fit] == camera]
                )[0]
            inner_entries = {}
            for camera in CAMERAS:
                camera_mask = cameras[inner_fit] == camera
                camera_widths = widths[inner_fit][camera_mask]
                edges = np.quantile(camera_widths, [1 / 3, 2 / 3])
                matrices = []
                for bin_index in range(3):
                    low = -math.inf if bin_index == 0 else edges[bin_index - 1]
                    high = math.inf if bin_index == 2 else edges[bin_index]
                    bin_mask = camera_mask & (widths[inner_fit] > low) & (widths[inner_fit] <= high)
                    sample = covariance(inner_fit_centered[bin_mask])[0]
                    n = int(np.sum(bin_mask)); weight = n / (n + 20.0)
                    matrices.append(spd(
                        weight * sample + (1.0 - weight) * inner_per_camera[camera]
                    )[0])
                inner_entries[camera] = {"edges": edges, "covariances": matrices}
            for local, index in enumerate(inner_cal):
                entry = inner_entries[cameras[index]]
                bin_index = int(np.searchsorted(entry["edges"], widths[index], side="left"))
                matrix = entry["covariances"][bin_index]
                residual = inner_cal_centered[local]
                nested_d2.append(float(residual @ np.linalg.inv(matrix) @ residual))
        nested_d2 = np.asarray(nested_d2)
        r2c_scale = max(
            1.0,
            float(np.quantile(nested_d2, 0.90) / CHI2["90"]),
            float(np.quantile(nested_d2, 0.95) / CHI2["95"]),
        )
        predictions["R2C"][test] = r2c_scale * predictions["R2"][test]

        scaler = StandardScaler().fit(features[train])
        base_d2 = np.empty(len(train))
        for local, index in enumerate(train):
            matrix = per_camera[cameras[index]]
            base_d2[local] = centered_train[local] @ np.linalg.inv(matrix) @ centered_train[local]
        target = np.log(np.maximum(base_d2 / 2.0, 1e-4))
        scale_model = Ridge(alpha=1.0).fit(scaler.transform(features[train]), target)
        raw_scale_train = np.clip(np.exp(scale_model.predict(scaler.transform(features[train]))), 0.05, 20.0)
        multiplier = float(np.mean(base_d2 / raw_scale_train) / 2.0)
        raw_scale_test = np.clip(np.exp(scale_model.predict(scaler.transform(features[test]))), 0.05, 20.0)
        scales = np.clip(raw_scale_test * multiplier, 0.05, 20.0)
        for local, index in enumerate(test):
            matrix, clipped = spd(scales[local] * per_camera[cameras[index]])
            floor_counts["R3"] += clipped
            predictions["R3"][index] = matrix
        fold_parameters[str(fold)] = {
            "mean_bias_by_camera": {camera: means[camera].tolist() for camera in CAMERAS},
            "R0": pooled.tolist(),
            "R1": {camera: per_camera[camera].tolist() for camera in CAMERAS},
            "R2_edges": {camera: width_entries[camera]["edges"].tolist() for camera in CAMERAS},
            "R2C_scale": r2c_scale,
            "R3_multiplier": multiplier,
        }

    reports = {}
    for name, values in predictions.items():
        clipping = floor_counts[name]
        if name in ("R2", "R2C"):
            clipping = floor_counts["R2_bin"] + floor_counts["R2_shrunk"]
        reports[name] = {
            **covariance_scores(values, scored_residuals, blocks),
            "eigenvalue_clipping_events": int(clipping),
        }
    selected = "R0"
    selection_steps = []
    for challenger in ("R1", "R2", "R3"):
        current = reports[selected]; candidate = reports[challenger]
        current_cal = abs(current["containment"]["90"] - 0.90) + abs(current["containment"]["95"] - 0.95)
        candidate_cal = abs(candidate["containment"]["90"] - 0.90) + abs(candidate["containment"]["95"] - 0.95)
        passed = (
            candidate["block_macro_mean_nll"] < current["block_macro_mean_nll"]
            and candidate_cal < current_cal
            and candidate["mean_95_ellipse_area_m2"] <= 1.2 * current["mean_95_ellipse_area_m2"]
        )
        selection_steps.append({
            "incumbent": selected, "challenger": challenger, "advance": passed,
            "incumbent_calibration_distance": current_cal,
            "challenger_calibration_distance": candidate_cal,
        })
        if passed:
            selected = challenger
    if selected == "R2":
        current = reports["R2"]
        candidate = reports["R2C"]
        current_cal = abs(current["containment"]["90"] - 0.90) + abs(current["containment"]["95"] - 0.95)
        candidate_cal = abs(candidate["containment"]["90"] - 0.90) + abs(candidate["containment"]["95"] - 0.95)
        passed = candidate_cal < current_cal
        selection_steps.append({
            "incumbent": "R2", "challenger": "R2C", "advance": passed,
            "reason": "post-selection train-fold-only containment calibration",
            "incumbent_calibration_distance": current_cal,
            "challenger_calibration_distance": candidate_cal,
            "ellipse_area_ratio": (
                candidate["mean_95_ellipse_area_m2"] / current["mean_95_ellipse_area_m2"]
            ),
        })
        if passed:
            selected = "R2C"
    deployment_d2 = np.asarray([
        scored_residuals[index]
        @ np.linalg.inv(predictions["R2"][index])
        @ scored_residuals[index]
        for index in range(len(rows))
    ])
    report = {
        "candidates": reports,
        "selection_steps": selection_steps,
        "selected": selected,
        "fold_parameters": fold_parameters,
        "deployment_R2C_scale": max(
            1.0,
            float(np.quantile(deployment_d2, 0.90) / CHI2["90"]),
            float(np.quantile(deployment_d2, 0.95) / CHI2["95"]),
        ),
    }
    return report, predictions, scored_residuals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage06", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_stage07_model_fitting":
        raise RuntimeError("Stage-07 protocol is not frozen")
    stage06 = args.stage06.resolve()
    stage06_manifest_path = REPO / "experiments/thesis_pipeline_lock/stage06_gate_manifest.json"
    pipeline = json.loads((REPO / "experiments/thesis_pipeline_lock/pipeline_lock.json").read_text())
    stage = next(item for item in pipeline["stages"] if item["id"] == "06_detector_gate")
    if stage.get("status") != "locked" or sha256(stage06_manifest_path) != stage["manifest_sha256"]:
        raise RuntimeError("Stage 06 is not locked")
    stage06_selection_manifest = json.loads((stage06 / "manifest.json").read_text())
    source = stage06 / stage06_selection_manifest.get("admission_records", "admission_records.csv")
    if not source.exists():
        source = stage06 / "admission_records.csv"
    if sha256(source) != "4599e7c4a529a4560cc54e362cbe27a34eb8872a386c5513c1c05c9c4f04a2a8":
        raise RuntimeError("Stage-06 admission records differ from the locked selection")
    rows = load_rows(source)
    if len(rows) != 2854:
        raise RuntimeError(f"Expected 2,854 admitted observations, found {len(rows)}")

    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    include_names = dict(zip(CAMERAS, (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )))
    models = {camera: camera_model_from_world(world, include_name=include) for camera, include in include_names.items()}
    camera_xy = {camera: (float(model.cam_pos[0]), float(model.cam_pos[1])) for camera, model in models.items()}
    features = feature_matrix(rows, camera_xy)
    bases = ray_bases(rows, camera_xy)
    truth = np.asarray([[row["robot_x"], row["robot_y"]] for row in rows])
    raw = np.asarray([[row["raw_ground_x"], row["raw_ground_y"]] for row in rows])
    analytic = np.asarray([[row["equivalent_x"], row["equivalent_y"]] for row in rows])
    camera_ids = np.asarray([row["camera_id"] for row in rows])
    folds = np.asarray([row["fold"] for row in rows])
    blocks = np.asarray([row["block_id"] for row in rows])
    target_ray = np.einsum("nij,nj->ni", np.transpose(bases, (0, 2, 1)), truth - analytic)

    predictions = {name: np.empty_like(truth) for name in ("C0_raw", "C1_visual_hull", "C2_ridge_residual", "C3_mlp_residual")}
    ridge_alphas = {}; mlp_epochs = {}; bias_records = defaultdict(dict)
    for fold in range(5):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        for name, base in (("C0_raw", raw), ("C1_visual_hull", analytic)):
            corrected, biases = apply_bias(base, truth, camera_ids, train, test)
            predictions[name][test] = corrected[test]
            bias_records[name][str(fold)] = biases

        validation_fold = (fold + 1) % 5
        inner_validation = np.flatnonzero(folds == validation_fold)
        inner_train = np.flatnonzero((folds != fold) & (folds != validation_fold))
        best_alpha = None; best_p95 = math.inf
        for alpha in protocol["fit_constants"]["ridge_alpha_grid"]:
            scaler = StandardScaler().fit(features[inner_train])
            model = Ridge(alpha=float(alpha)).fit(scaler.transform(features[inner_train]), target_ray[inner_train])
            residual_prediction = model.predict(scaler.transform(features))
            candidate = analytic + np.einsum("nij,nj->ni", bases, residual_prediction)
            candidate, _bias = apply_bias(candidate, truth, camera_ids, inner_train, inner_validation)
            p95 = percentile(np.linalg.norm(candidate[inner_validation] - truth[inner_validation], axis=1), 0.95)
            if p95 < best_p95:
                best_p95, best_alpha = p95, float(alpha)
        ridge_alphas[str(fold)] = best_alpha
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=best_alpha).fit(scaler.transform(features[train]), target_ray[train])
        residual_prediction = model.predict(scaler.transform(features))
        candidate = analytic + np.einsum("nij,nj->ni", bases, residual_prediction)
        candidate, biases = apply_bias(candidate, truth, camera_ids, train, test)
        predictions["C2_ridge_residual"][test] = candidate[test]
        bias_records["C2_ridge_residual"][str(fold)] = biases

        inner_scaler = StandardScaler().fit(features[inner_train])
        inner_model, best_epoch, validation_loss = fit_mlp(
            inner_scaler.transform(features[inner_train]), target_ray[inner_train],
            seed=20260910 + fold, epochs=200,
            x_validation=inner_scaler.transform(features[inner_validation]),
            y_validation=target_ray[inner_validation], patience=20,
        )
        mlp_epochs[str(fold)] = {"best_epoch": best_epoch, "inner_validation_mse": validation_loss}
        outer_scaler = StandardScaler().fit(features[train])
        outer_model, _epoch, _loss = fit_mlp(
            outer_scaler.transform(features[train]), target_ray[train],
            seed=20261910 + fold, epochs=best_epoch,
        )
        residual_prediction = mlp_predict(outer_model, outer_scaler.transform(features))
        candidate = analytic + np.einsum("nij,nj->ni", bases, residual_prediction)
        candidate, biases = apply_bias(candidate, truth, camera_ids, train, test)
        predictions["C3_mlp_residual"][test] = candidate[test]
        bias_records["C3_mlp_residual"][str(fold)] = biases

    correction_reports = correction_summaries(predictions, truth, rows)
    specified = protocol.get("deployed_correction_model")
    if specified is not None:
        if specified not in correction_reports:
            raise ValueError(f"unsupported deployed_correction_model: {specified!r}")
        selected = str(specified)
        selection_steps = [{
            "specified_before_refit": selected,
            "reason": protocol.get("deployed_correction_reason", "protocol-specified model"),
        }]
    else:
        selected = min(("C0_raw", "C1_visual_hull"), key=lambda name: (
            correction_reports[name]["pooled"]["above_0_25_m"],
            correction_reports[name]["block_macro_p95_m"],
        ))
        selection_steps = [{"initial_simpler_selection": selected}]
        for candidate in ("C2_ridge_residual", "C3_mlp_residual"):
            incumbent = correction_reports[selected]; challenger = correction_reports[candidate]
            p95_improvement = incumbent["pooled"]["p95_m"] - challenger["pooled"]["p95_m"]
            macro_median_improvement = incumbent["block_macro_median_m"] - challenger["block_macro_median_m"]
            passed = (
                p95_improvement >= 0.02
                and macro_median_improvement >= 0.01
                and challenger["pooled"]["above_0_25_m"] <= incumbent["pooled"]["above_0_25_m"]
            )
            selection_steps.append({
                "incumbent": selected, "challenger": candidate,
                "pooled_p95_improvement_m": p95_improvement,
                "block_macro_median_improvement_m": macro_median_improvement,
                "catastrophic_incumbent": incumbent["pooled"]["above_0_25_m"],
                "catastrophic_challenger": challenger["pooled"]["above_0_25_m"],
                "advance": passed,
            })
            if passed:
                selected = candidate

    errors = {name: np.linalg.norm(values - truth, axis=1) for name, values in predictions.items()}
    paired = {
        f"{name}_minus_{selected}": bootstrap_differences(errors[name], errors[selected], blocks, 20260910 + index)
        for index, name in enumerate(predictions) if name != selected
    }
    correction_report = {
        "schema": "thesis_stage07_correction_selection.v1",
        "status": "pass",
        "population": "Stage-06 admitted commissioning_fit observations",
        "observations": len(rows),
        "positions": len({row["position_id"] for row in rows}),
        "spatial_blocks": len(set(blocks.tolist())),
        "final_audit_accessed": False,
        "candidate_reports": correction_reports,
        "selection_steps": selection_steps,
        "selected": selected,
        "ridge_alpha_by_outer_fold": ridge_alphas,
        "mlp_epoch_by_outer_fold": mlp_epochs,
        "paired_block_bootstrap": paired,
        "fold_biases": bias_records,
    }

    residuals = predictions[selected] - truth
    covariance_report, covariance_predictions, covariance_scored_residuals = (
        fit_covariance_candidates(residuals, features, rows)
    )
    specified_covariance = protocol.get("deployed_covariance_model")
    if specified_covariance is not None:
        if specified_covariance not in covariance_report["candidates"]:
            raise ValueError(
                f"unsupported deployed_covariance_model: {specified_covariance!r}"
            )
        covariance_report["data_driven_selection"] = covariance_report["selected"]
        covariance_report["selected"] = str(specified_covariance)
        covariance_report["deployment_choice"] = {
            "specified_before_refit": str(specified_covariance),
            "reason": protocol.get(
                "deployed_covariance_reason", "protocol-specified covariance family"
            ),
        }
    covariance_report.update({
        "schema": "thesis_stage07_covariance_selection.v1",
        "status": "pass",
        "population": "out-of-fold residuals from the selected correction",
        "selected_correction": selected,
        "final_audit_accessed": False,
    })

    # Deployment correction fit.  Hyperparameters are summaries of outer-fold choices,
    # never selected from an in-sample deployment loss.
    deployment = {
        "schema": "thesis_commissioned_measurement_model.v1",
        "correction_model": selected,
        "feature_names": list(FEATURE_NAMES),
        "camera_order": list(CAMERAS),
        "covariance_model": covariance_report["selected"],
        "training_population": "all Stage-06 admitted commissioning_fit observations",
        "selection_population": "spatial out-of-fold commissioning predictions",
        "final_audit_accessed": False,
    }
    base_all = raw.copy() if selected == "C0_raw" else analytic.copy()
    learned_parameters = None
    model_state = None
    if selected == "C2_ridge_residual":
        alpha = float(statistics.median(ridge_alphas.values()))
        scaler = StandardScaler().fit(features)
        model = Ridge(alpha=alpha).fit(scaler.transform(features), target_ray)
        ray_prediction = model.predict(scaler.transform(features))
        base_all = analytic + np.einsum("nij,nj->ni", bases, ray_prediction)
        learned_parameters = {
            "ridge_alpha": alpha,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coefficient": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
        }
    elif selected == "C3_mlp_residual":
        epochs = int(statistics.median(value["best_epoch"] for value in mlp_epochs.values()))
        scaler = StandardScaler().fit(features)
        model, _epoch, _loss = fit_mlp(
            scaler.transform(features), target_ray, seed=20262910, epochs=epochs
        )
        ray_prediction = mlp_predict(model, scaler.transform(features))
        base_all = analytic + np.einsum("nij,nj->ni", bases, ray_prediction)
        learned_parameters = {
            "epochs": epochs, "hidden_layers": [64, 64],
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        }
        model_state = model.state_dict()
    bias_by_camera = {}
    for camera in CAMERAS:
        selector = camera_ids == camera
        bias_by_camera[camera] = np.mean(truth[selector] - base_all[selector], axis=0).tolist()
    deployment["correction_parameters"] = learned_parameters
    deployment["per_camera_residual_bias_xy_m"] = bias_by_camera

    # Fit the selected covariance family on all out-of-fold residuals after removing the
    # deployment bias.  Store all parameters needed by runtime; final-audit data are absent.
    centered = residuals.copy()
    residual_mean = {}
    for camera in CAMERAS:
        selector = camera_ids == camera
        mean = np.mean(centered[selector], axis=0)
        residual_mean[camera] = mean.tolist()
        centered[selector] -= mean
    covariance_kind = covariance_report["selected"]
    covariance_parameters = {"residual_mean_xy_m": residual_mean}
    pooled, _ = covariance(centered)
    per_camera = {camera: covariance(centered[camera_ids == camera])[0] for camera in CAMERAS}
    if covariance_kind == "R0":
        covariance_parameters["pooled_covariance_m2"] = pooled.tolist()
    elif covariance_kind == "R1":
        covariance_parameters["per_camera_covariance_m2"] = {
            camera: per_camera[camera].tolist() for camera in CAMERAS
        }
    elif covariance_kind in ("R2", "R2C"):
        entries = {}
        deployment_r2_covariances = np.empty((len(rows), 2, 2))
        widths = np.asarray([row["best_box_x1"] - row["best_box_x0"] for row in rows])
        for camera in CAMERAS:
            selector = camera_ids == camera
            edges = np.quantile(widths[selector], [1 / 3, 2 / 3])
            matrices = []
            for bin_index in range(3):
                low = -math.inf if bin_index == 0 else edges[bin_index - 1]
                high = math.inf if bin_index == 2 else edges[bin_index]
                mask = selector & (widths > low) & (widths <= high)
                sample, _ = covariance(centered[mask]); n = int(np.sum(mask)); weight = n / (n + 20.0)
                matrices.append(spd(weight * sample + (1 - weight) * per_camera[camera])[0].tolist())
            entries[camera] = {"width_edges_px": edges.tolist(), "covariance_m2": matrices}
            for index in np.flatnonzero(selector):
                bin_index = int(np.searchsorted(edges, widths[index], side="left"))
                deployment_r2_covariances[index] = np.asarray(matrices[bin_index])
        covariance_parameters["per_camera_width_bins"] = entries
        if covariance_kind == "R2C":
            covariance_parameters["calibration_scale"] = covariance_report[
                "deployment_R2C_scale"
            ]
    else:
        scaler = StandardScaler().fit(features)
        base_d2 = np.asarray([
            centered[index] @ np.linalg.inv(per_camera[camera_ids[index]]) @ centered[index]
            for index in range(len(rows))
        ])
        scale_model = Ridge(alpha=1.0).fit(
            scaler.transform(features), np.log(np.maximum(base_d2 / 2.0, 1e-4))
        )
        raw_scale = np.clip(np.exp(scale_model.predict(scaler.transform(features))), 0.05, 20.0)
        multiplier = float(np.mean(base_d2 / raw_scale) / 2.0)
        covariance_parameters.update({
            "base_per_camera_covariance_m2": {camera: per_camera[camera].tolist() for camera in CAMERAS},
            "scale_scaler_mean": scaler.mean_.tolist(), "scale_scaler_scale": scaler.scale_.tolist(),
            "scale_coefficient": scale_model.coef_.tolist(),
            "scale_intercept": float(scale_model.intercept_), "scale_multiplier": multiplier,
            "scale_clip": [0.05, 20.0],
        })
    deployment["covariance_parameters"] = covariance_parameters

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"Staging directory exists: {staging}")
    staging.mkdir(parents=True)
    correction_path = staging / "correction_report.json"
    covariance_path = staging / "covariance_report.json"
    deployment_path = staging / "measurement_model.json"
    correction_path.write_text(json.dumps(correction_report, indent=2, sort_keys=True) + "\n")
    covariance_path.write_text(json.dumps(covariance_report, indent=2, sort_keys=True) + "\n")
    if model_state is not None:
        state_path = staging / "correction_mlp_state.pt"
        torch.save(model_state, state_path)
        deployment["correction_state"] = state_path.name
        deployment["correction_state_sha256"] = sha256(state_path)
    deployment_path.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n")

    predictions_path = staging / "oof_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["pose_id", "position_id", "block_id", "fold", "camera_id", "truth_x", "truth_y"]
        for name in predictions:
            fields += [f"{name}_x", f"{name}_y", f"{name}_error_m"]
        fields += [
            "selected_R_xx_m2", "selected_R_xy_m2", "selected_R_yy_m2",
            "selected_R_mahalanobis2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(rows):
            record = {
                "pose_id": row["pose_id"], "position_id": row["position_id"],
                "block_id": row["block_id"], "fold": row["fold"], "camera_id": row["camera_id"],
                "truth_x": truth[index, 0], "truth_y": truth[index, 1],
            }
            for name, values in predictions.items():
                record[f"{name}_x"] = values[index, 0]; record[f"{name}_y"] = values[index, 1]
                record[f"{name}_error_m"] = errors[name][index]
            selected_matrix = covariance_predictions[covariance_kind][index]
            scored_residual = covariance_scored_residuals[index]
            record["selected_R_xx_m2"] = selected_matrix[0, 0]
            record["selected_R_xy_m2"] = selected_matrix[0, 1]
            record["selected_R_yy_m2"] = selected_matrix[1, 1]
            record["selected_R_mahalanobis2"] = float(
                scored_residual @ np.linalg.inv(selected_matrix) @ scored_residual
            )
            writer.writerow(record)

    manifest = {
        "schema": "thesis_stage07_run.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path.relative_to(REPO)), "protocol_sha256": sha256(protocol_path),
        "stage06_manifest": str(stage06_manifest_path.relative_to(REPO)),
        "stage06_manifest_sha256": sha256(stage06_manifest_path),
        "source_admission_records_sha256": sha256(source),
        "final_audit_accessed": False,
        "selected_correction": selected, "selected_covariance": covariance_kind,
        "artifacts": {
            "correction_report.json": sha256(correction_path),
            "covariance_report.json": sha256(covariance_path),
            "measurement_model.json": sha256(deployment_path),
            "oof_predictions.csv": sha256(predictions_path),
        },
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps({
        "selected_correction": selected,
        "correction": correction_reports,
        "selected_covariance": covariance_kind,
        "covariance": covariance_report["candidates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
