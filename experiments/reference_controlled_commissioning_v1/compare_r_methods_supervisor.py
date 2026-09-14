#!/usr/bin/env python3
"""Exploratory supervisor comparison of commissioned measurement covariance models.

The script uses only the unsealed fit and development drives of the current
reference-controlled campaign.  All model parameters and calibration factors are
estimated from the fit drives.  The development drives are used once for the
reported comparison.  Audit paths are rejected without being opened.

The analysis deliberately separates three questions:

1. the covariance of one corrected camera observation
2. dependence between cameras and between frames
3. direct filtering versus a double Gaussian cascade

The output is supervisor material, not paper-facing experimental evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CAMERA_XYZ = {
    "camera_A": np.asarray([-11.45, -9.45, 5.00]),
    "camera_B": np.asarray([-1.50, -9.72, 5.00]),
    "camera_C": np.asarray([-6.95, 9.45, 5.00]),
    "camera_D": np.asarray([11.45, 7.20, 5.00]),
    "camera_E": np.asarray([11.45, -9.45, 5.00]),
}
CHI2 = {0.50: 1.38629436112, 0.90: 4.60517018599,
        0.95: 5.99146454711, 0.99: 9.21034037198}
FLOOR_M2 = 1.0e-6
COLORS = {
    "global_constant": "#787878",
    "geometric_range_angle": "#222222",
    "per_camera_constant": "#4c78a8",
    "range_cholesky": "#f58518",
    "range_bearing_cholesky": "#e45756",
    "spatial_bins": "#72b7b2",
    "spatial_kernel": "#54a24b",
    "spatial_gp_scale": "#b279a2",
    "spatial_mlp_cholesky": "#ff9da6",
    "image_conditioned_scale": "#9d755d",
}
DISPLAY = {
    "global_constant": "Global constant",
    "geometric_range_angle": "Geometric range + angle",
    "per_camera_constant": "Per-camera constant",
    "range_cholesky": "Range Cholesky",
    "range_bearing_cholesky": "Range + bearing Cholesky",
    "spatial_bins": "Spatial bins",
    "spatial_kernel": "Spatial kernel",
    "spatial_gp_scale": "Spatial GP scale",
    "spatial_mlp_cholesky": "Spatial MLP Cholesky",
    "image_conditioned_scale": "Image-conditioned scale",
}


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


def spd(matrix: np.ndarray, floor: float = FLOOR_M2) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return vectors @ np.diag(values) @ vectors.T


def balanced_weights(drives: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(drives), dtype=float)
    unique = sorted(set(drives.tolist()))
    for drive in unique:
        selected = drives == drive
        weights[selected] = 1.0 / (len(unique) * selected.sum())
    return weights / weights.sum()


def empirical_covariance(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()
    matrix = np.einsum("n,ni,nj->ij", weights, residual, residual)
    return spd(matrix)


def subset(data: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[selected] for key, value in data.items()}


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    ray = raw_xy - camera_xy
    along = ray / np.linalg.norm(ray)
    return np.column_stack((along, np.asarray([-along[1], along[0]])))


def load_data(campaign_root: Path, prediction_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    execution_path = campaign_root / "campaign_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("campaign is not complete with audit sealed")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("audit must remain sealed")

    sources = [entry for entry in execution["drive_sources"]
               if entry["partition"] in {"fit", "development"}]
    if any("audit" in entry["drive_id"] or "audit" in entry["source"] for entry in sources):
        raise RuntimeError("refusing an audit source")
    fit_drives = sorted(entry["drive_id"] for entry in sources if entry["partition"] == "fit")
    development_drives = sorted(
        entry["drive_id"] for entry in sources if entry["partition"] == "development"
    )
    if len(fit_drives) != 6 or len(development_drives) != 6:
        raise RuntimeError("expected six fit and six development drives")

    rows: list[dict[str, str]] = []
    for entry in sorted(sources, key=lambda value: value["drive_id"]):
        table = Path(entry["source"]) / "tables/camera_measurements.csv"
        with table.open(newline="", encoding="utf-8") as handle:
            local = list(csv.DictReader(handle))
        for row in local:
            if row["drive_id"] != entry["drive_id"] or row["partition"] != entry["partition"]:
                raise RuntimeError("measurement row crossed its declared partition")
            if row["reference_supported"] != "1" or row["sensor_gate_admitted"] != "1":
                raise RuntimeError("camera_measurements contains an unsupported row")
        rows.extend(local)
    rows.sort(key=lambda row: (row["drive_id"], int(row["capture_stamp_ns"]), row["camera_id"]))

    prediction_path = prediction_root / "candidate_predictions.npz"
    comparison_path = prediction_root / "visibility_patch_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("audit_analysis_permitted") is not False:
        raise RuntimeError("prediction artifact does not preserve the audit seal")
    if "artifacts" in comparison and "predictions" in comparison["artifacts"]:
        expected = comparison["artifacts"]["predictions"].get("sha256")
        if expected and sha256(prediction_path) != expected:
            raise RuntimeError("prediction artifact hash changed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        predicted = {key: np.asarray(archive[key]) for key in archive.files}
    if len(rows) != len(predicted["drive_id"]):
        raise RuntimeError("measurement and prediction row counts differ")

    drive = np.asarray([row["drive_id"] for row in rows])
    partition = np.asarray([row["partition"] for row in rows])
    camera = np.asarray([row["camera_id"] for row in rows])
    stamp = np.asarray([int(row["capture_stamp_ns"]) for row in rows], dtype=np.int64)
    source_frame = np.asarray([row["source_frame_id"] for row in rows])
    identities = zip(drive, camera, stamp, source_frame, strict=True)
    predicted_identities = zip(
        predicted["drive_id"], predicted["camera_id"], predicted["capture_stamp_ns"],
        predicted["source_frame_id"], strict=True,
    )
    for index, (observed, expected) in enumerate(zip(identities, predicted_identities, strict=True)):
        normalized = (str(expected[0]), str(expected[1]), int(expected[2]), str(expected[3]))
        if observed != normalized:
            raise RuntimeError(f"prediction identity mismatch at row {index}")

    raw = np.asarray([[float(row["raw_projected_x"]), float(row["raw_projected_y"])]
                      for row in rows])
    reference = np.asarray([[float(row["reference_x"]), float(row["reference_y"])]
                            for row in rows])
    basis = np.stack([ray_basis(CAMERA_XYZ[value][:2], point)
                      for value, point in zip(camera, raw, strict=True)])
    correction = np.asarray(predicted["candidate_prediction_ray_m"], dtype=float)
    target = np.asarray(predicted["target_ray_m"], dtype=float)
    residual = correction - target
    corrected = raw + np.einsum("nij,nj->ni", basis, correction)
    horizontal_range = np.asarray([
        np.linalg.norm(point - CAMERA_XYZ[value][:2])
        for value, point in zip(camera, reference, strict=True)
    ])
    bearing = np.asarray([
        math.atan2(point[1] - CAMERA_XYZ[value][1], point[0] - CAMERA_XYZ[value][0])
        for value, point in zip(camera, reference, strict=True)
    ])
    height = np.asarray([CAMERA_XYZ[value][2] for value in camera])
    slant_range = np.sqrt(horizontal_range ** 2 + height ** 2)
    floor_normal_cosine = height / slant_range
    bbox_width = np.asarray([float(row["bbox_xmax"]) - float(row["bbox_xmin"]) for row in rows])
    bbox_height = np.asarray([float(row["bbox_ymax"]) - float(row["bbox_ymin"]) for row in rows])
    confidence = np.asarray([float(row["detector_score"]) for row in rows])
    visibility = np.asarray(predicted["visibility_score"], dtype=float)
    camera_one_hot = np.column_stack([(camera == value).astype(float) for value in CAMERAS])
    image_features = np.column_stack((
        reference,
        horizontal_range,
        np.sin(bearing),
        np.cos(bearing),
        visibility,
        bbox_width / 1280.0,
        bbox_height / 720.0,
        confidence,
        camera_one_hot,
    ))
    data = {
        "drive": drive,
        "partition": partition,
        "camera": camera,
        "stamp": stamp,
        "reference": reference,
        "raw": raw,
        "corrected": corrected,
        "basis": basis,
        "residual": residual,
        "range": horizontal_range,
        "bearing": bearing,
        "floor_normal_cosine": floor_normal_cosine,
        "visibility": visibility,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "confidence": confidence,
        "image_features": image_features,
    }
    metadata = {
        "campaign_root": str(campaign_root),
        "campaign_execution_sha256": sha256(execution_path),
        "prediction_root": str(prediction_root),
        "prediction_sha256": sha256(prediction_path),
        "fit_drives": fit_drives,
        "development_drives": development_drives,
        "fit_rows": int(np.sum(partition == "fit")),
        "development_rows": int(np.sum(partition == "development")),
        "audit_opened": False,
    }
    return data, metadata


class CovarianceModel:
    def fit(self, data: dict[str, np.ndarray]) -> "CovarianceModel":
        raise NotImplementedError

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class ConstantModel(CovarianceModel):
    def __init__(self, per_camera: bool) -> None:
        self.per_camera = per_camera
        self.values: dict[str, np.ndarray] = {}

    def fit(self, data: dict[str, np.ndarray]) -> "ConstantModel":
        groups = CAMERAS if self.per_camera else ("all",)
        for group in groups:
            selected = np.ones(len(data["drive"]), dtype=bool)
            if group != "all":
                selected = data["camera"] == group
            self.values[group] = empirical_covariance(
                data["residual"][selected], balanced_weights(data["drive"][selected])
            )
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if not self.per_camera:
            return np.repeat(self.values["all"][None, :, :], len(data["drive"]), axis=0)
        return np.stack([self.values[value] for value in data["camera"]])


class GeometricRangeAngleModel(CovarianceModel):
    """Isotropic engineering baseline based on range and floor-incidence angle.

    q = cos(alpha) / d_horizontal^2 and R = sigma0^2 / q I.
    Only sigma0 is estimated on the fit partition.
    """

    def fit(self, data: dict[str, np.ndarray]) -> "GeometricRangeAngleModel":
        q = data["floor_normal_cosine"] / np.maximum(data["range"] ** 2, 0.25)
        energy = np.sum(data["residual"] ** 2, axis=1)
        weights = balanced_weights(data["drive"])
        self.sigma0_squared = float(np.sum(weights * q * energy) / 2.0)
        self.sigma0_squared = max(self.sigma0_squared, FLOOR_M2)
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        q = data["floor_normal_cosine"] / np.maximum(data["range"] ** 2, 0.25)
        variance = self.sigma0_squared / np.maximum(q, 1.0e-8)
        result = np.zeros((len(variance), 2, 2))
        result[:, 0, 0] = variance
        result[:, 1, 1] = variance
        return result


def range_features(data: dict[str, np.ndarray]) -> np.ndarray:
    value = data["range"] / 10.0
    return np.column_stack((np.ones(len(value)), value, value ** 2))


def range_bearing_features(data: dict[str, np.ndarray]) -> np.ndarray:
    value = data["range"] / 10.0
    angle = data["bearing"]
    return np.column_stack((
        np.ones(len(value)), value, value ** 2, np.sin(angle), np.cos(angle),
        value * np.sin(angle), value * np.cos(angle),
    ))


class CholeskyRegression(CovarianceModel):
    def __init__(self, feature_function: Callable[[dict[str, np.ndarray]], np.ndarray]) -> None:
        self.feature_function = feature_function
        self.parameters: dict[str, np.ndarray] = {}
        self.fallback = ConstantModel(per_camera=True)

    def fit(self, data: dict[str, np.ndarray]) -> "CholeskyRegression":
        self.fallback.fit(data)
        features = self.feature_function(data)
        for camera in CAMERAS:
            selected = data["camera"] == camera
            x = features[selected]
            residual = data["residual"][selected]
            weights = balanced_weights(data["drive"][selected])
            k = x.shape[1]
            factor = np.linalg.cholesky(self.fallback.values[camera])
            start = np.zeros(3 * k)
            start[0] = math.log(factor[0, 0])
            start[k] = math.log(factor[1, 1])
            start[2 * k] = factor[1, 0]

            def objective(theta: np.ndarray) -> float:
                log_a = np.clip(x @ theta[:k], -8.0, 2.0)
                log_c = np.clip(x @ theta[k:2 * k], -8.0, 2.0)
                b = np.clip(x @ theta[2 * k:], -2.0, 2.0)
                a = np.exp(log_a)
                c = np.exp(log_c)
                r11 = a ** 2
                r12 = a * b
                r22 = b ** 2 + c ** 2
                determinant = np.maximum(r11 * r22 - r12 ** 2, 1.0e-12)
                e1 = residual[:, 0]
                e2 = residual[:, 1]
                distance = (r22 * e1 ** 2 - 2.0 * r12 * e1 * e2 + r11 * e2 ** 2) / determinant
                nll = 0.5 * (np.log(determinant) + distance)
                regularization = 1.0e-4 * float(np.sum(theta[1:] ** 2))
                return float(np.sum(weights * nll) + regularization)

            result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 350})
            self.parameters[camera] = result.x
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        features = self.feature_function(data)
        result = np.empty((len(data["drive"]), 2, 2))
        for camera in CAMERAS:
            selected = np.flatnonzero(data["camera"] == camera)
            if not len(selected):
                continue
            x = features[selected]
            theta = self.parameters[camera]
            k = x.shape[1]
            a = np.exp(np.clip(x @ theta[:k], -8.0, 2.0))
            c = np.exp(np.clip(x @ theta[k:2 * k], -8.0, 2.0))
            b = np.clip(x @ theta[2 * k:], -2.0, 2.0)
            result[selected, 0, 0] = a ** 2
            result[selected, 0, 1] = result[selected, 1, 0] = a * b
            result[selected, 1, 1] = b ** 2 + c ** 2
        return np.stack([spd(value) for value in result])


class SpatialBins(CovarianceModel):
    def __init__(self, cell_size: float = 2.0, prior: float = 40.0) -> None:
        self.cell_size = cell_size
        self.prior = prior
        self.base = ConstantModel(per_camera=True)
        self.values: dict[tuple[str, int, int], np.ndarray] = {}

    def fit(self, data: dict[str, np.ndarray]) -> "SpatialBins":
        self.base.fit(data)
        for camera in CAMERAS:
            selected = np.flatnonzero(data["camera"] == camera)
            cells = np.floor(data["reference"][selected] / self.cell_size).astype(int)
            for cell in sorted(set(map(tuple, cells.tolist()))):
                local = selected[np.all(cells == np.asarray(cell), axis=1)]
                count = len(local)
                covariance = empirical_covariance(
                    data["residual"][local], balanced_weights(data["drive"][local])
                )
                self.values[(camera, cell[0], cell[1])] = spd(
                    (count * covariance + self.prior * self.base.values[camera])
                    / (count + self.prior)
                )
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        cells = np.floor(data["reference"] / self.cell_size).astype(int)
        return np.stack([
            self.values.get((camera, int(cell[0]), int(cell[1])), self.base.values[camera])
            for camera, cell in zip(data["camera"], cells, strict=True)
        ])


class SpatialKernel(CovarianceModel):
    def __init__(self, bandwidth: float = 2.0, cell_size: float = 1.0, prior: float = 30.0) -> None:
        self.bandwidth = bandwidth
        self.cell_size = cell_size
        self.prior = prior
        self.base = ConstantModel(per_camera=True)
        self.centres: dict[str, np.ndarray] = {}
        self.scatter: dict[str, np.ndarray] = {}
        self.counts: dict[str, np.ndarray] = {}

    def fit(self, data: dict[str, np.ndarray]) -> "SpatialKernel":
        self.base.fit(data)
        for camera in CAMERAS:
            selected = np.flatnonzero(data["camera"] == camera)
            cells = np.floor(data["reference"][selected] / self.cell_size).astype(int)
            centres, scatter, counts = [], [], []
            for cell in sorted(set(map(tuple, cells.tolist()))):
                local = selected[np.all(cells == np.asarray(cell), axis=1)]
                weights = balanced_weights(data["drive"][local])
                centres.append(np.average(data["reference"][local], axis=0, weights=weights))
                scatter.append(empirical_covariance(data["residual"][local], weights))
                counts.append(len(local))
            self.centres[camera] = np.asarray(centres)
            self.scatter[camera] = np.asarray(scatter)
            self.counts[camera] = np.asarray(counts, dtype=float)
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        result = np.empty((len(data["drive"]), 2, 2))
        for camera in CAMERAS:
            selected = np.flatnonzero(data["camera"] == camera)
            if not len(selected):
                continue
            query = data["reference"][selected]
            distance2 = np.sum((query[:, None, :] - self.centres[camera][None, :, :]) ** 2, axis=2)
            weights = np.exp(-0.5 * distance2 / self.bandwidth ** 2)
            weights *= self.counts[camera][None, :]
            numerator = np.einsum("qn,nij->qij", weights, self.scatter[camera])
            denominator = weights.sum(axis=1)
            numerator += self.prior * self.base.values[camera][None, :, :]
            denominator += self.prior
            result[selected] = numerator / denominator[:, None, None]
        return np.stack([spd(value) for value in result])


class SpatialGPScale(CovarianceModel):
    def __init__(self, cell_size: float = 1.0, length_scale: float = 2.0) -> None:
        self.cell_size = cell_size
        self.length_scale = length_scale
        self.base = ConstantModel(per_camera=True)
        self.models: dict[str, GaussianProcessRegressor] = {}

    def fit(self, data: dict[str, np.ndarray]) -> "SpatialGPScale":
        self.base.fit(data)
        for camera in CAMERAS:
            selected = np.flatnonzero(data["camera"] == camera)
            inverse = np.linalg.inv(self.base.values[camera])
            energy = 0.5 * np.einsum(
                "ni,ij,nj->n", data["residual"][selected], inverse, data["residual"][selected]
            )
            cells = np.floor(data["reference"][selected] / self.cell_size).astype(int)
            x, y, alpha = [], [], []
            for cell in sorted(set(map(tuple, cells.tolist()))):
                use = np.all(cells == np.asarray(cell), axis=1)
                x.append(np.mean(data["reference"][selected][use], axis=0))
                y.append(float(np.mean(np.log(np.clip(energy[use], 0.08, 12.0)))))
                alpha.append(0.15 + 1.0 / max(int(np.sum(use)), 1))
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                self.length_scale, length_scale_bounds="fixed"
            ) + WhiteKernel(0.15, noise_level_bounds="fixed")
            model = GaussianProcessRegressor(
                kernel=kernel, alpha=np.asarray(alpha), optimizer=None, normalize_y=True
            )
            model.fit(np.asarray(x), np.asarray(y))
            self.models[camera] = model
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        result = np.empty((len(data["drive"]), 2, 2))
        for camera in CAMERAS:
            selected = np.flatnonzero(data["camera"] == camera)
            if not len(selected):
                continue
            log_scale = self.models[camera].predict(data["reference"][selected])
            scale = np.exp(np.clip(log_scale, math.log(0.2), math.log(5.0)))
            result[selected] = self.base.values[camera][None, :, :] * scale[:, None, None]
        return result


class ImageConditionedScale(CovarianceModel):
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.base = ConstantModel(per_camera=True)
        self.regressor = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))

    def fit(self, data: dict[str, np.ndarray]) -> "ImageConditionedScale":
        self.base.fit(data)
        base = self.base.predict(data)
        inverse = np.linalg.inv(base)
        energy = 0.5 * np.einsum("ni,nij,nj->n", data["residual"], inverse, data["residual"])
        target = np.log(np.clip(energy, 0.08, 12.0))
        weights = balanced_weights(data["drive"]) * len(data["drive"])
        self.regressor.fit(data["image_features"], target, ridge__sample_weight=weights)
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        base = self.base.predict(data)
        log_scale = self.regressor.predict(data["image_features"])
        scale = np.exp(np.clip(log_scale, math.log(0.2), math.log(5.0)))
        return base * scale[:, None, None]


class SpatialMLPCholesky(CovarianceModel):
    def __init__(self, epochs: int = 90, seed: int = 7) -> None:
        self.epochs = epochs
        self.seed = seed

    def fit(self, data: dict[str, np.ndarray]) -> "SpatialMLPCholesky":
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        camera_one_hot = np.column_stack([(data["camera"] == value).astype(float) for value in CAMERAS])
        x = np.column_stack((data["reference"], camera_one_hot)).astype(np.float32)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1.0e-4)
        x = (x - self.mean) / self.std
        y = data["residual"].astype(np.float32)
        weights = (balanced_weights(data["drive"]) * len(data["drive"])).astype(np.float32)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 3),
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.008, weight_decay=1.0e-5)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        wt = torch.from_numpy(weights)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            raw = self.network(xt)
            a = torch.exp(torch.clamp(raw[:, 0], -8.0, 2.0))
            c = torch.exp(torch.clamp(raw[:, 1], -8.0, 2.0))
            b = torch.clamp(raw[:, 2], -2.0, 2.0)
            determinant = torch.clamp(a * a * c * c, min=1.0e-12)
            r11 = a * a
            r12 = a * b
            r22 = b * b + c * c
            distance = (r22 * yt[:, 0] ** 2 - 2.0 * r12 * yt[:, 0] * yt[:, 1]
                        + r11 * yt[:, 1] ** 2) / determinant
            loss = torch.sum(wt * 0.5 * (torch.log(determinant) + distance)) / torch.sum(wt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
            optimizer.step()
        return self

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        import torch

        camera_one_hot = np.column_stack([(data["camera"] == value).astype(float) for value in CAMERAS])
        x = np.column_stack((data["reference"], camera_one_hot)).astype(np.float32)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            raw = self.network(torch.from_numpy(x)).numpy()
        a = np.exp(np.clip(raw[:, 0], -8.0, 2.0))
        c = np.exp(np.clip(raw[:, 1], -8.0, 2.0))
        b = np.clip(raw[:, 2], -2.0, 2.0)
        result = np.empty((len(x), 2, 2))
        result[:, 0, 0] = a ** 2
        result[:, 0, 1] = result[:, 1, 0] = a * b
        result[:, 1, 1] = b ** 2 + c ** 2
        return np.stack([spd(value) for value in result])


BUILDERS: dict[str, Callable[[], CovarianceModel]] = {
    "global_constant": lambda: ConstantModel(per_camera=False),
    "geometric_range_angle": GeometricRangeAngleModel,
    "per_camera_constant": lambda: ConstantModel(per_camera=True),
    "range_cholesky": lambda: CholeskyRegression(range_features),
    "range_bearing_cholesky": lambda: CholeskyRegression(range_bearing_features),
    "spatial_bins": SpatialBins,
    "spatial_kernel": SpatialKernel,
    "spatial_gp_scale": SpatialGPScale,
    "spatial_mlp_cholesky": SpatialMLPCholesky,
    "image_conditioned_scale": ImageConditionedScale,
}


def score(data: dict[str, np.ndarray], covariance: np.ndarray) -> dict[str, Any]:
    residual = data["residual"]
    inverse = np.linalg.inv(covariance)
    d2 = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    logdet = np.linalg.slogdet(covariance)[1]
    nll = 0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + d2)
    area = math.pi * CHI2[0.95] * np.sqrt(np.linalg.det(covariance)) * 1.0e4
    by_drive = {
        drive: float(np.mean(nll[data["drive"] == drive]))
        for drive in sorted(set(data["drive"].tolist()))
    }
    return {
        "n": int(len(residual)),
        "equal_drive_nll": float(np.mean(list(by_drive.values()))),
        "pooled_nll": float(np.mean(nll)),
        "mean_mahalanobis_d2": float(np.mean(d2)),
        "coverage": {str(level): float(np.mean(d2 <= threshold))
                     for level, threshold in CHI2.items()},
        "tail_over_99": float(np.mean(d2 > CHI2[0.99])),
        "mean_95_ellipse_area_cm2": float(np.mean(area)),
        "median_95_ellipse_area_cm2": float(np.median(area)),
        "by_drive_nll": by_drive,
    }


def fit_oof(builder: Callable[[], CovarianceModel], fit: dict[str, np.ndarray]) -> np.ndarray:
    result = np.empty((len(fit["drive"]), 2, 2))
    for held in sorted(set(fit["drive"].tolist())):
        train_index = np.flatnonzero(fit["drive"] != held)
        test_index = np.flatnonzero(fit["drive"] == held)
        model = builder().fit(subset(fit, train_index))
        result[test_index] = model.predict(subset(fit, test_index))
    return result


def fit_static_models(fit: dict[str, np.ndarray], development: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
    report: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, CovarianceModel] = {}
    for index, (name, builder) in enumerate(BUILDERS.items(), start=1):
        print(f"[{index}/{len(BUILDERS)}] fitting {name}", flush=True)
        oof = fit_oof(builder, fit)
        inverse = np.linalg.inv(oof)
        d2 = np.einsum("ni,nij,nj->n", fit["residual"], inverse, fit["residual"])
        calibration = float(np.average(d2, weights=balanced_weights(fit["drive"])) / 2.0)
        calibration = float(np.clip(calibration, 0.2, 5.0))
        oof *= calibration
        model = builder().fit(fit)
        covariance = model.predict(development) * calibration
        report[name] = {
            "fit_oof_calibration_scale": calibration,
            "fit_oof": score(fit, oof),
            "development": score(development, covariance),
        }
        predictions[name] = covariance
        models[name] = model
    return report, predictions, models


def normalized_residuals(data: dict[str, np.ndarray], covariance: np.ndarray) -> np.ndarray:
    result = np.empty_like(data["residual"])
    for index, (residual, matrix) in enumerate(zip(data["residual"], covariance, strict=True)):
        result[index] = np.linalg.solve(np.linalg.cholesky(matrix), residual)
    return result


def autocorrelations(data: dict[str, np.ndarray], covariance: np.ndarray) -> dict[str, Any]:
    normalized = normalized_residuals(data, covariance)
    result: dict[str, Any] = {}
    for stride in (1, 5):
        values = []
        for drive in sorted(set(data["drive"].tolist())):
            for camera in CAMERAS:
                selected = np.flatnonzero((data["drive"] == drive) & (data["camera"] == camera))
                selected = selected[np.argsort(data["stamp"][selected])]
                if len(selected) <= stride + 2:
                    continue
                for axis in range(2):
                    value = np.corrcoef(
                        normalized[selected[:-stride], axis], normalized[selected[stride:], axis]
                    )[0, 1]
                    if math.isfinite(value):
                        values.append(float(value))
        result[f"stride_{stride}"] = {
            "median": float(np.median(values)),
            "median_absolute": float(np.median(np.abs(values))),
            "mean": float(np.mean(values)),
            "values": values,
        }
    return result


def fit_temporal_model(fit: dict[str, np.ndarray], covariance: np.ndarray) -> dict[str, Any]:
    normalized = normalized_residuals(fit, covariance)
    per_camera: dict[str, float] = {}
    for camera in CAMERAS:
        numerator = 0.0
        denominator = 0.0
        for drive in sorted(set(fit["drive"].tolist())):
            selected = np.flatnonzero((fit["drive"] == drive) & (fit["camera"] == camera))
            selected = selected[np.argsort(fit["stamp"][selected])]
            if len(selected) < 3:
                continue
            previous = normalized[selected[:-1]]
            current = normalized[selected[1:]]
            numerator += float(np.sum(previous * current))
            denominator += float(np.sum(previous ** 2))
        per_camera[camera] = float(np.clip(numerator / max(denominator, 1.0e-9), 0.0, 0.95))
    return {"phi_per_frame": per_camera}


def temporal_conditional_score(data: dict[str, np.ndarray], covariance: np.ndarray,
                               model: dict[str, Any]) -> dict[str, Any]:
    normalized = normalized_residuals(data, covariance)
    independent_nll, conditional_nll = [], []
    by_drive_independent: dict[str, list[float]] = {}
    by_drive_conditional: dict[str, list[float]] = {}
    for drive in sorted(set(data["drive"].tolist())):
        for camera in CAMERAS:
            selected = np.flatnonzero((data["drive"] == drive) & (data["camera"] == camera))
            selected = selected[np.argsort(data["stamp"][selected])]
            if not len(selected):
                continue
            phi = model["phi_per_frame"][camera]
            for local_index, row_index in enumerate(selected):
                z = normalized[row_index]
                iid_value = math.log(2.0 * math.pi) + 0.5 * float(z @ z)
                if local_index == 0:
                    conditional_value = iid_value
                else:
                    innovation = z - phi * normalized[selected[local_index - 1]]
                    variance = max(1.0 - phi ** 2, 0.05)
                    conditional_value = math.log(2.0 * math.pi * variance) + 0.5 * float(
                        innovation @ innovation
                    ) / variance
                independent_nll.append(iid_value)
                conditional_nll.append(conditional_value)
                by_drive_independent.setdefault(drive, []).append(iid_value)
                by_drive_conditional.setdefault(drive, []).append(conditional_value)
    iid_drive = {key: float(np.mean(value)) for key, value in by_drive_independent.items()}
    cond_drive = {key: float(np.mean(value)) for key, value in by_drive_conditional.items()}
    return {
        "independent_standardized_equal_drive_nll": float(np.mean(list(iid_drive.values()))),
        "gauss_markov_standardized_equal_drive_nll": float(np.mean(list(cond_drive.values()))),
        "delta_nll_per_measurement": float(np.mean(list(cond_drive.values())) - np.mean(list(iid_drive.values()))),
        "by_drive_independent": iid_drive,
        "by_drive_gauss_markov": cond_drive,
    }


def simultaneous_groups(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    groups = []
    keys = np.asarray([f"{drive}:{stamp}" for drive, stamp in zip(data["drive"], data["stamp"], strict=True)])
    for key in sorted(set(keys.tolist())):
        selected = np.flatnonzero(keys == key)
        if len(selected) >= 2:
            groups.append(selected)
    return groups


def fit_shared_correlation(data: dict[str, np.ndarray], covariance: np.ndarray) -> dict[str, Any]:
    normalized = normalized_residuals(data, covariance)
    values = []
    blocks = []
    pair_values: dict[str, list[float]] = {}
    for selected in simultaneous_groups(data):
        for a in range(len(selected)):
            for b in range(a + 1, len(selected)):
                i, j = selected[a], selected[b]
                value = float(normalized[i] @ normalized[j] / 2.0)
                values.append(value)
                block = 0.5 * (
                    np.outer(normalized[i], normalized[j])
                    + np.outer(normalized[j], normalized[i])
                )
                blocks.append(block)
                key = "|".join(sorted((str(data["camera"][i]), str(data["camera"][j]))))
                pair_values.setdefault(key, []).append(value)
    shared_block = np.mean(blocks, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (shared_block + shared_block.T))
    eigenvalues = np.clip(eigenvalues, -0.20, 0.75)
    shared_block = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return {
        "shared_block": shared_block.tolist(),
        "mean_same_axis_strength": float(np.trace(shared_block) / 2.0),
        "pairwise_fit": {key: {"n": len(value), "mean": float(np.mean(value))}
                         for key, value in sorted(pair_values.items())},
    }


def cross_camera_score(data: dict[str, np.ndarray], covariance: np.ndarray,
                       correlation: dict[str, Any]) -> dict[str, Any]:
    normalized = normalized_residuals(data, covariance)
    shared_block = np.asarray(correlation["shared_block"])
    independent, shared = [], []
    sizes = []
    pair_values: dict[str, list[float]] = {}
    for selected in simultaneous_groups(data):
        z = normalized[selected].reshape(-1)
        count = len(selected)
        independent.append(count * math.log(2.0 * math.pi) + 0.5 * float(z @ z))
        joint = np.eye(2 * count)
        for a in range(count):
            for b in range(a):
                joint[2 * a:2 * a + 2, 2 * b:2 * b + 2] = shared_block
                joint[2 * b:2 * b + 2, 2 * a:2 * a + 2] = shared_block.T
        joint = spd(joint, floor=1.0e-4)
        shared.append(
            count * math.log(2.0 * math.pi)
            + 0.5 * float(np.linalg.slogdet(joint)[1] + z @ np.linalg.solve(joint, z))
        )
        sizes.append(count)
        for a in range(count):
            for b in range(a + 1, count):
                i, j = selected[a], selected[b]
                key = "|".join(sorted((str(data["camera"][i]), str(data["camera"][j]))))
                pair_values.setdefault(key, []).append(float(normalized[i] @ normalized[j] / 2.0))
    total_measurements = max(int(np.sum(sizes)), 1)
    return {
        "rounds": len(sizes),
        "measurements": total_measurements,
        "independent_joint_nll_per_measurement": float(np.sum(independent) / total_measurements),
        "shared_latent_joint_nll_per_measurement": float(np.sum(shared) / total_measurements),
        "delta_nll_per_measurement": float((np.sum(shared) - np.sum(independent)) / total_measurements),
        "pairwise_development": {key: {"n": len(value), "mean": float(np.mean(value))}
                                 for key, value in sorted(pair_values.items())},
    }


def world_covariances(data: dict[str, np.ndarray], covariance_ray: np.ndarray) -> np.ndarray:
    return np.einsum("nik,nkl,njl->nij", data["basis"], covariance_ray, data["basis"])


def batch_update(mean: np.ndarray, covariance: np.ndarray, observations: np.ndarray,
                 measurement_covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(observations)
    h = np.tile(np.eye(2), (count, 1))
    innovation = observations.reshape(-1) - h @ mean
    innovation_covariance = h @ covariance @ h.T + measurement_covariance
    gain = covariance @ h.T @ np.linalg.inv(innovation_covariance)
    updated_mean = mean + gain @ innovation
    updated_covariance = spd((np.eye(2) - gain @ h) @ covariance)
    return updated_mean, updated_covariance


def replay_architecture(data: dict[str, np.ndarray], covariance_ray: np.ndarray,
                        correlation: dict[str, Any], arm: str) -> dict[str, Any]:
    covariance_world = world_covariances(data, covariance_ray)
    shared_block = np.asarray(correlation["shared_block"])
    drive_results: dict[str, Any] = {}
    all_squared_error, all_nees = [], []
    for drive in sorted(set(data["drive"].tolist())):
        drive_index = np.flatnonzero(data["drive"] == drive)
        stamps = sorted(set(data["stamp"][drive_index].tolist()))
        first = np.flatnonzero((data["drive"] == drive) & (data["stamp"] == stamps[0]))[0]
        mean = data["reference"][first].copy()
        covariance = np.eye(2) * 0.01
        camera_tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        previous_reference = data["reference"][first].copy()
        errors, nees = [], []
        for stamp in stamps:
            selected = np.flatnonzero((data["drive"] == drive) & (data["stamp"] == stamp))
            reference = np.mean(data["reference"][selected], axis=0)
            delta = reference - previous_reference
            previous_reference = reference
            process_variance = (0.006 + 0.012 * np.linalg.norm(delta)) ** 2
            mean = mean + delta
            covariance = covariance + np.eye(2) * process_variance
            for camera, (track_mean, track_covariance) in list(camera_tracks.items()):
                camera_tracks[camera] = (
                    track_mean + delta,
                    track_covariance + np.eye(2) * process_variance,
                )
            if arm == "double_cascade":
                track_observations, track_covariances = [], []
                for index in selected:
                    camera = str(data["camera"][index])
                    if camera not in camera_tracks:
                        camera_tracks[camera] = (
                            data["corrected"][index].copy(), covariance_world[index].copy()
                        )
                    else:
                        track_mean, track_covariance = camera_tracks[camera]
                        track_mean, track_covariance = batch_update(
                            track_mean, track_covariance,
                            data["corrected"][index][None, :], covariance_world[index]
                        )
                        camera_tracks[camera] = (track_mean, track_covariance)
                    track_observations.append(camera_tracks[camera][0])
                    track_covariances.append(camera_tracks[camera][1])
                measurement_covariance = np.zeros((2 * len(selected), 2 * len(selected)))
                for local, matrix in enumerate(track_covariances):
                    measurement_covariance[2 * local:2 * local + 2, 2 * local:2 * local + 2] = matrix
                mean, covariance = batch_update(
                    mean, covariance, np.asarray(track_observations), measurement_covariance
                )
            else:
                count = len(selected)
                measurement_covariance = np.zeros((2 * count, 2 * count))
                for a, index_a in enumerate(selected):
                    measurement_covariance[2 * a:2 * a + 2, 2 * a:2 * a + 2] = covariance_world[index_a]
                    if arm == "direct_shared":
                        for b, index_b in enumerate(selected[:a]):
                            cross = (
                                np.linalg.cholesky(covariance_world[index_a])
                                @ shared_block
                                @ np.linalg.cholesky(covariance_world[index_b]).T
                            )
                            measurement_covariance[2 * a:2 * a + 2, 2 * b:2 * b + 2] = cross
                            measurement_covariance[2 * b:2 * b + 2, 2 * a:2 * a + 2] = cross.T
                measurement_covariance = spd(measurement_covariance)
                mean, covariance = batch_update(
                    mean, covariance, data["corrected"][selected], measurement_covariance
                )
            error = mean - reference
            value = float(error @ np.linalg.solve(covariance, error))
            errors.append(float(error @ error))
            nees.append(value)
        drive_results[drive] = {
            "rmse_m": float(math.sqrt(np.mean(errors))),
            "mean_nees": float(np.mean(nees)),
            "nees_95_coverage": float(np.mean(np.asarray(nees) <= CHI2[0.95])),
            "rounds": len(stamps),
        }
        all_squared_error.extend(errors)
        all_nees.extend(nees)
    return {
        "equal_drive_rmse_m": float(np.mean([value["rmse_m"] for value in drive_results.values()])),
        "pooled_rmse_m": float(math.sqrt(np.mean(all_squared_error))),
        "mean_nees": float(np.mean(all_nees)),
        "nees_95_coverage": float(np.mean(np.asarray(all_nees) <= CHI2[0.95])),
        "by_drive": drive_results,
    }


def write_static_csv(path: Path, static: dict[str, Any]) -> None:
    fields = [
        "model", "equal_drive_nll", "mean_mahalanobis_d2", "coverage_50",
        "coverage_90", "coverage_95", "coverage_99", "tail_over_99",
        "mean_95_ellipse_area_cm2", "fit_oof_calibration_scale",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, entry in static.items():
            metrics = entry["development"]
            writer.writerow({
                "model": name,
                "equal_drive_nll": metrics["equal_drive_nll"],
                "mean_mahalanobis_d2": metrics["mean_mahalanobis_d2"],
                "coverage_50": metrics["coverage"]["0.5"],
                "coverage_90": metrics["coverage"]["0.9"],
                "coverage_95": metrics["coverage"]["0.95"],
                "coverage_99": metrics["coverage"]["0.99"],
                "tail_over_99": metrics["tail_over_99"],
                "mean_95_ellipse_area_cm2": metrics["mean_95_ellipse_area_cm2"],
                "fit_oof_calibration_scale": entry["fit_oof_calibration_scale"],
            })


def page_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.055, 0.913, subtitle, fontsize=9.5, color="#555555", va="top")


def add_footer(fig: plt.Figure, page: int) -> None:
    fig.text(0.055, 0.025, "Supervisor comparison | exploratory | audit sealed", fontsize=8, color="#777777")
    fig.text(0.945, 0.025, str(page), fontsize=8, color="#777777", ha="right")


def table_on_axis(axis: plt.Axes, columns: list[str], rows: list[list[str]], widths: list[float] | None = None,
                  font_size: float = 8.0, scale_y: float = 1.45) -> None:
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colLoc="left",
                       colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#eaf1f8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")


def generate_report(path: Path, result: dict[str, Any], data: dict[str, np.ndarray],
                    development: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> None:
    static = result["static_models"]
    ranked = sorted(static, key=lambda name: static[name]["development"]["equal_drive_nll"])
    best = result["selection"]["best_learned_model"]
    page = 0
    with PdfPages(path) as pdf:
        page += 1
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        page_title(fig, "Commissioned covariance models", "A development-split comparison for supervisor discussion")
        fig.text(0.055, 0.82, "Question", fontsize=11, fontweight="bold", color="#315b7d")
        fig.text(0.055, 0.775,
                 "Which uncertainty model best describes the residual error after the commissioned mean correction?",
                 fontsize=14, wrap=True)
        fig.text(0.055, 0.70, "Data split", fontsize=11, fontweight="bold", color="#315b7d")
        fig.text(0.055, 0.645,
                 f"Six fit drives, {result['metadata']['fit_rows']:,} admitted measurements\n"
                 f"Six development drives, {result['metadata']['development_rows']:,} admitted measurements\n"
                 "Audit drives were not opened",
                 fontsize=12, linespacing=1.6, va="top")
        fig.text(0.055, 0.47, "Result", fontsize=11, fontweight="bold", color="#315b7d")
        winner = static[best]["development"]
        fig.text(0.055, 0.42,
                 f"Best learned model on equal-drive development NLL: {DISPLAY[best]}\n"
                 f"NLL {winner['equal_drive_nll']:.3f}, 95% containment {100 * winner['coverage']['0.95']:.1f}%, "
                 f"mean 95% area {winner['mean_95_ellipse_area_cm2']:.1f} cm²",
                 fontsize=13, linespacing=1.55, va="top")
        fig.text(0.055, 0.26, "Interpretation boundary", fontsize=11, fontweight="bold", color="#a04431")
        fig.text(0.055, 0.215,
                 "This report supports method selection and supervisor discussion. It is not an audit result and is not yet paper-facing evidence.\n"
                 "Covariance changes uncertainty calibration, not the corrected position itself.",
                 fontsize=10.5, linespacing=1.55, va="top")
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), gridspec_kw={"width_ratios": [1.12, 1]})
        fig.subplots_adjust(top=0.82, bottom=0.12, left=0.19, right=0.96, wspace=0.34)
        page_title(fig, "Single-observation covariance", "Lower NLL is better. Area shows how sharp the reported uncertainty is.")
        names = ranked
        nll = [static[name]["development"]["equal_drive_nll"] for name in names]
        axes[0].barh(range(len(names)), nll, color=[COLORS[name] for name in names])
        axes[0].set_yticks(range(len(names)), [DISPLAY[name] for name in names], fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Equal-drive Gaussian NLL")
        axes[0].grid(axis="x", alpha=0.2)
        for index, value in enumerate(nll):
            axes[0].text(value, index, f"  {value:.3f}", va="center", fontsize=8)
        for name in names:
            metrics = static[name]["development"]
            marker = "*" if name == best else "o"
            size = 115 if name == best else 45
            axes[1].scatter(metrics["mean_95_ellipse_area_cm2"], metrics["equal_drive_nll"],
                            color=COLORS[name], marker=marker, s=size, label=DISPLAY[name])
        axes[1].set_xlabel("Mean 95% ellipse area [cm²]")
        axes[1].set_ylabel("Equal-drive Gaussian NLL")
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=7, loc="best")
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        page_title(fig, "Development results", "All parameters and scalar calibration factors come from fit drives only")
        axis = fig.add_axes([0.045, 0.11, 0.91, 0.72])
        rows = []
        for rank, name in enumerate(ranked, start=1):
            metrics = static[name]["development"]
            rows.append([
                str(rank), DISPLAY[name], f"{metrics['equal_drive_nll']:.3f}",
                f"{metrics['mean_mahalanobis_d2']:.2f}",
                f"{100 * metrics['coverage']['0.9']:.1f}%",
                f"{100 * metrics['coverage']['0.95']:.1f}%",
                f"{metrics['mean_95_ellipse_area_cm2']:.1f}",
            ])
        table_on_axis(axis, ["Rank", "Model", "NLL", "Mean d²", "90%", "95%", "Area cm²"], rows,
                      widths=[0.06, 0.29, 0.11, 0.11, 0.10, 0.10, 0.13], font_size=8.4, scale_y=1.5)
        fig.text(0.055, 0.065,
                 "Ideal mean d² is 2.0. Nominal containment should match the stated percentage. A small ellipse is useful only when it remains calibrated.",
                 fontsize=9, color="#555555")
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.82, bottom=0.14, left=0.08, right=0.96, wspace=0.28)
        page_title(fig, "Calibration and route stability", "Observed containment and complete-drive NLL reveal different failure modes")
        levels = list(CHI2)
        axes[0].plot(levels, levels, "--", color="#222222", linewidth=1.2, label="Ideal")
        selected_names = ["global_constant", "geometric_range_angle", best]
        selected_names = list(dict.fromkeys(selected_names))
        for name in selected_names:
            coverage = static[name]["development"]["coverage"]
            axes[0].plot(levels, [coverage[str(level)] for level in levels], marker="o",
                         color=COLORS[name], label=DISPLAY[name])
        axes[0].set(xlabel="Nominal probability", ylabel="Observed containment", xlim=(0.47, 1.0), ylim=(0.47, 1.0))
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        drives = result["metadata"]["development_drives"]
        matrix = np.asarray([[static[name]["development"]["by_drive_nll"][drive] for drive in drives]
                             for name in ranked])
        image = axes[1].imshow(matrix, aspect="auto", cmap="viridis_r")
        axes[1].set_yticks(range(len(ranked)), [DISPLAY[name] for name in ranked], fontsize=7.5)
        axes[1].set_xticks(range(len(drives)), [drive.replace("development_", "").replace("_settle", "")
                                                for drive in drives], rotation=40, ha="right", fontsize=7)
        axes[1].set_title("NLL by complete development drive", fontsize=10)
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True, sharey=True)
        fig.subplots_adjust(top=0.83, bottom=0.11, left=0.09, right=0.94, hspace=0.25, wspace=0.13)
        page_title(fig, f"Where {DISPLAY[best]} is uncertain", "Each point is a development measurement. Colour is predicted 95% ellipse area.")
        area = math.pi * CHI2[0.95] * np.sqrt(np.linalg.det(predictions[best])) * 1.0e4
        low, high = np.quantile(area, [0.02, 0.98])
        scatter = None
        for axis, camera in zip(axes.flat, CAMERAS, strict=False):
            selected = development["camera"] == camera
            scatter = axis.scatter(development["reference"][selected, 0], development["reference"][selected, 1],
                                   c=area[selected], s=7, cmap="magma", vmin=low, vmax=high, alpha=0.8)
            axis.scatter(CAMERA_XYZ[camera][0], CAMERA_XYZ[camera][1], marker="^", s=60, color="#238b45")
            axis.set_title(camera.replace("camera_", "Camera "), fontsize=10)
            axis.grid(alpha=0.15)
            axis.set_aspect("equal", adjustable="box")
        axes[1, 2].axis("off")
        if scatter is not None:
            colorbar_axis = fig.add_axes([0.77, 0.18, 0.018, 0.20])
            fig.colorbar(scatter, cax=colorbar_axis, label="95% area [cm²]")
        fig.text(0.49, 0.055, "Warehouse x [m]", ha="center", fontsize=9)
        fig.text(0.025, 0.46, "Warehouse y [m]", va="center", rotation=90, fontsize=9)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.96, wspace=0.30)
        page_title(fig, "Dependence is a separate modelling decision", "Static R describes one observation. These tests ask whether observations are independent.")
        temporal = result["temporal_dependence"]
        labels = ["5 Hz\nlag 1", "Approx. 1 Hz\nlag 5"]
        before = [temporal["development_autocorrelation"]["stride_1"]["median_absolute"],
                  temporal["development_autocorrelation"]["stride_5"]["median_absolute"]]
        axes[0].bar(labels, before, color=["#e45756", "#4c78a8"])
        axes[0].axhline(0.30, color="#222222", linestyle="--", linewidth=1, label="Diagnostic threshold")
        axes[0].set_ylabel("Median absolute residual correlation")
        axes[0].set_ylim(0, max(0.65, max(before) * 1.2))
        axes[0].set_title("Between-frame dependence")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.2)
        cross = result["cross_camera_dependence"]
        values = [cross["development"]["independent_joint_nll_per_measurement"],
                  cross["development"]["shared_latent_joint_nll_per_measurement"]]
        axes[1].bar(["Independent", f"Shared block\nstrength={cross['fit']['mean_same_axis_strength']:.2f}"], values,
                    color=["#787878", "#b279a2"])
        axes[1].set_ylabel("Joint standardized NLL per measurement")
        axes[1].set_title("Simultaneous-camera dependence")
        axes[1].grid(axis="y", alpha=0.2)
        fig.text(0.08, 0.075,
                 f"Fit-only Gauss–Markov model changes development standardized NLL by "
                 f"{temporal['development_conditional_score']['delta_nll_per_measurement']:+.3f} per measurement. "
                 "Negative means the dependence model predicts better.", fontsize=9.5)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.96, wspace=0.30)
        page_title(fig, "Estimator architecture replay", "Same corrected observations and reference motion increments on development drives")
        replay = result["architecture_replay"]
        arms = ["direct_independent", "direct_shared", "double_cascade"]
        labels = ["Direct\nindependent", "Direct\ncorrelated", "Double Gaussian\ncascade"]
        axes[0].bar(labels, [100 * replay[arm]["equal_drive_rmse_m"] for arm in arms],
                    color=["#4c78a8", "#b279a2", "#e45756"])
        axes[0].set_ylabel("Equal-drive belief RMSE [cm]")
        axes[0].grid(axis="y", alpha=0.2)
        axes[1].bar(labels, [replay[arm]["mean_nees"] for arm in arms],
                    color=["#4c78a8", "#b279a2", "#e45756"])
        axes[1].axhline(2.0, color="#222222", linestyle="--", label="Ideal mean NEES")
        axes[1].set_ylabel("Mean position NEES")
        axes[1].set_yscale("log")
        axes[1].grid(axis="y", alpha=0.2)
        axes[1].legend(fontsize=8)
        fig.text(0.08, 0.082,
                 "The cascade is an architecture baseline. Its camera-track posteriors reuse earlier frames.\n"
                 "Treating those posteriors as fresh independent measurements can make belief covariance overconfident.",
                 fontsize=9.2, linespacing=1.35)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        page_title(fig, "Recommended paper shortlist", "Keep the broad ladder for supervision. Report only the decisive comparison in the paper.")
        shortlist = result["selection"]["paper_shortlist"]
        axis = fig.add_axes([0.07, 0.46, 0.86, 0.33])
        rows = []
        for role, name in shortlist.items():
            metrics = static[name]["development"]
            rows.append([
                role.replace("_", " ").title(), DISPLAY[name], f"{metrics['equal_drive_nll']:.3f}",
                f"{100 * metrics['coverage']['0.95']:.1f}%", f"{metrics['mean_95_ellipse_area_cm2']:.1f}",
            ])
        table_on_axis(axis, ["Role", "Method", "NLL", "95% containment", "Area cm²"], rows,
                      widths=[0.22, 0.32, 0.12, 0.18, 0.13], font_size=9.5, scale_y=1.7)
        fig.text(0.07, 0.37, "Decision rule", fontsize=11, fontweight="bold", color="#315b7d")
        fig.text(0.07, 0.315,
                 "Select the learned commissioned model by equal-drive development NLL. Check containment and ellipse area before accepting it. "
                 "Keep temporal and cross-camera dependence as explicit estimator assumptions.",
                 fontsize=11, wrap=True)
        best_coverage = static[best]["development"]["coverage"]["0.95"]
        fig.text(0.07, 0.255,
                 f"Current warning: the selected model contains only {100 * best_coverage:.1f}% of development residuals in its nominal 95% ellipses. "
                 "The Gaussian tail model is therefore not ready to freeze.",
                 fontsize=10.5, color="#a04431", wrap=True)
        fig.text(0.07, 0.20, "Before paper use", fontsize=11, fontweight="bold", color="#a04431")
        fig.text(0.07, 0.145,
                 "Freeze the chosen method and evaluation protocol, then run the sealed audit once. Do not copy these development numbers into the thesis as final evidence.",
                 fontsize=11, wrap=True)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    data, metadata = load_data(args.campaign_root.resolve(), args.prediction_root.resolve())
    fit_index = np.flatnonzero(data["partition"] == "fit")
    development_index = np.flatnonzero(data["partition"] == "development")
    fit = subset(data, fit_index)
    development = subset(data, development_index)

    static, predictions, models = fit_static_models(fit, development)
    learned_candidates = [name for name in BUILDERS
                          if name not in {"global_constant", "geometric_range_angle"}]
    best = min(learned_candidates, key=lambda name: static[name]["development"]["equal_drive_nll"])

    best_model = models[best]
    calibration = static[best]["fit_oof_calibration_scale"]
    fit_covariance = best_model.predict(fit) * calibration
    development_covariance = predictions[best]
    temporal_model = fit_temporal_model(fit, fit_covariance)
    temporal = {
        "fit_model": temporal_model,
        "fit_autocorrelation": autocorrelations(fit, fit_covariance),
        "development_autocorrelation": autocorrelations(development, development_covariance),
        "development_conditional_score": temporal_conditional_score(
            development, development_covariance, temporal_model
        ),
    }
    correlation = fit_shared_correlation(fit, fit_covariance)
    cross = {
        "fit": correlation,
        "development": cross_camera_score(development, development_covariance, correlation),
    }
    replay = {
        arm: replay_architecture(development, development_covariance, correlation, arm)
        for arm in ("direct_independent", "direct_shared", "double_cascade")
    }
    result = {
        "schema": "commissioned_r_supervisor_comparison.v1",
        "status": "complete_exploratory_development_split",
        "paper_facing_evidence": False,
        "audit_opened": False,
        "metadata": metadata,
        "methodology": {
            "fit": "six fit drives with whole-drive out-of-fold calibration",
            "evaluation": "six untouched development drives, equal weight per complete drive",
            "primary_metric": "equal-drive Gaussian negative log likelihood",
            "robot_heading_used_by_covariance_models": False,
            "geometric_baseline": "q = cos(alpha) / horizontal_range^2, R = sigma0^2 / q I",
            "mean_correction": "fixed commissioned visibility-residual correction",
        },
        "static_models": static,
        "selection": {
            "best_learned_model": best,
            "paper_shortlist": {
                "global_constant": "global_constant",
                "literature_geometric_baseline": "geometric_range_angle",
                "best_learned_commissioned": best,
            },
        },
        "temporal_dependence": temporal,
        "cross_camera_dependence": cross,
        "architecture_replay": replay,
    }

    write_static_csv(output / "static_model_comparison.csv", static)
    atomic_json(output / "supervisor_comparison.json", result)
    np.savez_compressed(
        output / "development_covariance_predictions.npz",
        drive_id=development["drive"],
        camera_id=development["camera"],
        capture_stamp_ns=development["stamp"],
        best_model=np.asarray(best),
        covariance_ray_m2=development_covariance,
    )
    report_path = output / "commissioning_R_supervisor_comparison.pdf"
    generate_report(report_path, result, data, development, predictions)
    completion = {
        "schema": "commissioned_r_supervisor_completion.v1",
        "status": "complete_exploratory_development_split",
        "audit_opened": False,
        "best_learned_model": best,
        "artifacts": {name: sha256(output / name) for name in (
            "static_model_comparison.csv",
            "supervisor_comparison.json",
            "development_covariance_predictions.npz",
            "commissioning_R_supervisor_comparison.pdf",
        )},
        "source_sha256": sha256(Path(__file__)),
    }
    atomic_json(output / "completion.json", completion)
    print(json.dumps({
        "best_learned_model": best,
        "development": static[best]["development"],
        "temporal_delta_nll": temporal["development_conditional_score"]["delta_nll_per_measurement"],
        "cross_camera_strength": correlation["mean_same_axis_strength"],
        "cross_camera_delta_nll": cross["development"]["delta_nll_per_measurement"],
        "report": str(report_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
