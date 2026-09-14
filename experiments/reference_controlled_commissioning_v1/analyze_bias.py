#!/usr/bin/env python3
"""Fit and compare bias corrections on unsealed reference-controlled drives.

The script never opens an audit-drive table or crop.  B0--B4 predict only the conditional
mean correction from runtime-available fields.  Covariance, NIS, fusion, belief and planner
outputs are intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from build_tables import _camera_positions  # noqa: E402


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
MODELS = ("B0_raw", "B1_camera_constant", "B2_smooth_geometry", "B3_box_mlp", "B4_rgb_context")
COMPLEXITY = {name: index for index, name in enumerate(MODELS)}
COLORS = {
    "B0_raw": "#8b8b85",
    "B1_camera_constant": "#2474b5",
    "B2_smooth_geometry": "#15956f",
    "B3_box_mlp": "#e0642c",
    "B4_rgb_context": "#ba2d72",
}
CAMERA_COLORS = {
    "camera_A": "#2b6cb0", "camera_B": "#dd6b20", "camera_C": "#2f855a",
    "camera_D": "#805ad5", "camera_E": "#c53030",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {key} in {row.get('drive_id')} {row.get('source_frame_id')}")
    return value


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    ray = raw_xy - camera_xy
    distance = float(np.linalg.norm(ray))
    if not math.isfinite(distance) or distance <= 1e-9:
        raise RuntimeError("invalid camera-to-reading ray")
    along = ray / distance
    return np.column_stack((along, np.asarray([-along[1], along[0]])))


def load_data(campaign_root: Path, output: Path) -> dict[str, np.ndarray]:
    execution_path = campaign_root / "campaign_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("campaign is not complete with audit sealed")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("expected audit_analysis_permitted=false")

    protocol_path = Path(execution["protocol"]).resolve()
    if sha256(protocol_path) != execution["protocol_sha256"]:
        raise RuntimeError("campaign protocol hash changed")
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    gate_path = REPO / "config/commissioning_sensor_gate_v2.yaml"
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if gate.get("gate_id") != "commissioning_sensor_gate_v2" or gate.get("edge_check_mode") != "full_bbox":
        raise RuntimeError("expected the frozen commissioning sensor gate v2")
    camera_xy = {key: np.asarray(value, dtype=float) for key, value in _camera_positions(protocol).items()}
    selected = execution["selected_run_dirs"]
    audit_ids = sorted(key for key in selected if key.startswith("audit_"))
    usable_ids = sorted(key for key in selected if key.startswith(("fit_", "development_")))
    if len(audit_ids) != 8 or len(usable_ids) != 16:
        raise RuntimeError("expected eight sealed audit and sixteen unsealed drives")

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    gate_refusals = {"fit": 0, "development": 0}
    for drive_id in usable_ids:
        run = Path(selected[drive_id]).resolve()
        if "audit" in run.name or any(part.startswith("audit_") for part in run.parts):
            raise RuntimeError(f"refusing audit path {run}")
        integrity_path = run / "tables/integrity.json"
        table_path = run / "tables/camera_measurements.csv"
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity.get("passed") is not True or integrity.get("drive_id") != drive_id:
            raise RuntimeError(f"invalid table integrity for {drive_id}")
        if integrity.get("frame_identity_match_fraction") != 1.0:
            raise RuntimeError(f"incomplete frame identity for {drive_id}")
        sources.append({
            "drive_id": drive_id,
            "partition": integrity["partition"],
            "run_dir": str(run),
            "camera_measurements": str(table_path),
            "camera_measurements_sha256": sha256(table_path),
            "integrity_sha256": sha256(integrity_path),
            "rows": int(integrity["measurement_rows"]),
        })
        with table_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(integrity["measurement_rows"]):
            raise RuntimeError(f"measurement-row count changed for {drive_id}")
        for row in rows:
            if row["drive_id"] != drive_id or row["partition"] not in {"fit", "development"}:
                raise RuntimeError("measurement row crossed the declared partition")
            if row["sensor_gate_admitted"] != "1" or row["reference_supported"] != "1":
                raise RuntimeError("camera_measurements contains a non-admitted or unsupported row")
            crop_path = Path(row["crop_path"]).resolve()
            if not crop_path.is_file() or run not in crop_path.parents:
                raise RuntimeError(f"invalid crop path for {drive_id}: {crop_path}")
            with np.load(crop_path, allow_pickle=False) as archive:
                crop = np.asarray(archive["crop"], dtype=np.uint8)
                image_shape = np.asarray(archive["image_shape"], dtype=int)
                crop_stamp = int(archive["capture_stamp_ns"])
                crop_frame = str(archive["source_frame_id"])
            if crop.shape != (3, 96, 96):
                raise RuntimeError(f"unexpected crop shape {crop.shape}")
            if crop_stamp != int(row["capture_stamp_ns"]) or crop_frame != row["source_frame_id"]:
                raise RuntimeError("crop identity differs from measurement row")

            raw = np.asarray([require_float(row, "raw_projected_x"), require_float(row, "raw_projected_y")])
            reference = np.asarray([require_float(row, "reference_x"), require_float(row, "reference_y")])
            basis = ray_basis(camera_xy[row["camera_id"]], raw)
            target_world = reference - raw
            target_ray = basis.T @ target_world
            ray = raw - camera_xy[row["camera_id"]]
            distance = float(np.linalg.norm(ray))
            bearing = math.atan2(ray[1], ray[0])
            height, width = map(float, image_shape)
            x0, y0, x1, y1 = (require_float(row, key) for key in
                              ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
            box_w, box_h = x1 - x0, y1 - y0
            if box_w <= 0.0 or box_h <= 0.0:
                raise RuntimeError("non-positive admitted box")
            score = require_float(row, "detector_score")
            admitted_v2 = (
                score >= float(gate["confidence_threshold"])
                and box_w >= float(gate["min_bbox_width_px"])
                and box_h >= float(gate["min_bbox_height_px"])
                and x0 >= float(gate["min_edge_distance_px"])
                and y0 >= float(gate["min_edge_distance_px"])
                and x1 <= width - float(gate["min_edge_distance_px"])
                and y1 <= height - float(gate["min_edge_distance_px"])
                and np.isfinite(raw).all()
            )
            if not admitted_v2:
                gate_refusals[row["partition"]] += 1
                continue
            one_hot = [float(row["camera_id"] == camera) for camera in CAMERAS]
            geometry = [distance, 1.0 / max(distance, 1e-3), math.cos(bearing), math.sin(bearing)]
            box = geometry + [
                box_w / width, box_h / height, box_w / box_h,
                0.5 * (x0 + x1) / width, y1 / height, score,
            ] + one_hot
            records.append({
                "drive": drive_id,
                "partition": row["partition"],
                "route": row["route"],
                "direction": row["direction"],
                "camera": row["camera_id"],
                "source_frame_id": row["source_frame_id"],
                "capture_stamp_ns": int(row["capture_stamp_ns"]),
                "raw": raw,
                "reference": reference,
                "basis": basis,
                "target_ray": target_ray,
                "geometry": geometry + one_hot,
                "box": box,
                "crop": crop[:, ::2, ::2],
                "range_m": distance,
                "bearing_rad": bearing,
                "stationary": str(row["controller_state"]).startswith("stationary"),
                "reference_speed_mps": require_float(row, "reference_speed_mps"),
            })

    if not records:
        raise RuntimeError("no unsealed admitted measurements")
    manifest = {
        "schema": "reference_controlled_bias_inputs.v2",
        "status": "complete_unsealed_inputs_only",
        "campaign_root": str(campaign_root),
        "campaign_execution_sha256": sha256(execution_path),
        "protocol": str(protocol_path),
        "protocol_sha256": execution["protocol_sha256"],
        "sensor_gate": str(gate_path),
        "sensor_gate_sha256": sha256(gate_path),
        "sensor_gate_id": gate["gate_id"],
        "audit_analysis_permitted": False,
        "audit_drive_ids_not_opened": audit_ids,
        "input_drives": sources,
        "model_input_contract": {
            "B1_camera_constant": ["camera_id", "raw camera ray"],
            "B2_smooth_geometry": ["camera_id", "raw range", "raw bearing"],
            "B3_box_mlp": ["B2 inputs", "bbox geometry", "detector confidence"],
            "B4_rgb_context": ["B3 inputs", "recorded RGB crop"],
            "forbidden": ["route id", "drive id", "timestamp", "route progress", "belief",
                          "innovation", "NIS", "reference pose", "reference speed", "controller state"],
        },
        "counts": {
            "fit_drives": len({row["drive"] for row in records if row["partition"] == "fit"}),
            "development_drives": len({row["drive"] for row in records if row["partition"] == "development"}),
            "fit_measurements": sum(row["partition"] == "fit" for row in records),
            "development_measurements": sum(row["partition"] == "development" for row in records),
            "fit_v1_admitted_refused_by_v2": gate_refusals["fit"],
            "development_v1_admitted_refused_by_v2": gate_refusals["development"],
        },
    }
    atomic_json(output / "input_manifest.json", manifest)

    return {
        "drive": np.asarray([row["drive"] for row in records]),
        "partition": np.asarray([row["partition"] for row in records]),
        "route": np.asarray([row["route"] for row in records]),
        "direction": np.asarray([row["direction"] for row in records]),
        "camera": np.asarray([row["camera"] for row in records]),
        "source_frame_id": np.asarray([row["source_frame_id"] for row in records]),
        "stamp_ns": np.asarray([row["capture_stamp_ns"] for row in records], dtype=np.int64),
        "raw": np.stack([row["raw"] for row in records]),
        "reference": np.stack([row["reference"] for row in records]),
        "basis": np.stack([row["basis"] for row in records]),
        "target_ray": np.stack([row["target_ray"] for row in records]),
        "geometry": np.asarray([row["geometry"] for row in records], dtype=np.float32),
        "box": np.asarray([row["box"] for row in records], dtype=np.float32),
        "crop": np.stack([row["crop"] for row in records]),
        "range_m": np.asarray([row["range_m"] for row in records]),
        "bearing_rad": np.asarray([row["bearing_rad"] for row in records]),
        "stationary": np.asarray([row["stationary"] for row in records], dtype=bool),
        "reference_speed_mps": np.asarray([row["reference_speed_mps"] for row in records]),
    }


class CameraConstant:
    def fit(self, x: np.ndarray, y: np.ndarray, camera: np.ndarray) -> "CameraConstant":
        self.values = {value: y[camera == value].mean(0) for value in sorted(set(camera))}
        return self

    def predict(self, x: np.ndarray, camera: np.ndarray) -> np.ndarray:
        return np.stack([self.values[value] for value in camera])


class PerCameraSmooth:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _basis(x: np.ndarray) -> np.ndarray:
        """Bounded, low-order ray geometry without quadratic range extrapolation."""
        distance, inverse, cosine, sine = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return np.column_stack((
            distance, inverse, cosine, sine,
            distance * cosine, distance * sine,
            cosine * cosine - sine * sine, 2.0 * cosine * sine,
        ))

    def fit(self, x: np.ndarray, y: np.ndarray, camera: np.ndarray) -> "PerCameraSmooth":
        for value in sorted(set(camera)):
            selected = camera == value
            raw = x[selected, :4]
            lower, upper = raw.min(0), raw.max(0)
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(self._basis(raw), y[selected])
            self.models[value] = model
            self.bounds[value] = (lower, upper)
        return self

    def predict(self, x: np.ndarray, camera: np.ndarray) -> np.ndarray:
        result = np.zeros((len(x), 2), dtype=float)
        for value, model in self.models.items():
            selected = camera == value
            if np.any(selected):
                lower, upper = self.bounds[value]
                bounded = np.clip(x[selected, :4], lower, upper)
                result[selected] = model.predict(self._basis(bounded))
        return result


def fit_tabular(model_id: str, data: dict[str, np.ndarray], train: np.ndarray, seed: int) -> Any:
    y, camera = data["target_ray"][train], data["camera"][train]
    if model_id == "B1_camera_constant":
        return CameraConstant().fit(data["geometry"][train], y, camera)
    if model_id == "B2_smooth_geometry":
        return PerCameraSmooth().fit(data["geometry"][train], y, camera)
    if model_id == "B3_box_mlp":
        network = MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="relu", solver="adam", alpha=1e-3,
            batch_size=128, learning_rate_init=1e-3, max_iter=800, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=35, random_state=seed,
        )
        model = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), network), transformer=StandardScaler()
        )
        model.fit(data["box"][train], y)
        return model
    raise ValueError(model_id)


def predict_tabular(model_id: str, model: Any, data: dict[str, np.ndarray], index: np.ndarray) -> np.ndarray:
    if model_id in {"B1_camera_constant", "B2_smooth_geometry"}:
        feature = data["geometry"][index]
        return model.predict(feature, data["camera"][index])
    if model_id == "B3_box_mlp":
        return model.predict(data["box"][index])
    raise ValueError(model_id)


class RGBMean(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + feature_count, 96), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 2),
        )

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.vision(image), feature), dim=1))


class RGBArtifact:
    def __init__(self, model: RGBMean, x_mean: np.ndarray, x_sd: np.ndarray,
                 y_mean: np.ndarray, y_sd: np.ndarray, device: torch.device) -> None:
        self.model, self.x_mean, self.x_sd = model, x_mean, x_sd
        self.y_mean, self.y_sd, self.device = y_mean, y_sd, device

    def predict(self, data: dict[str, np.ndarray], index: np.ndarray, batch: int = 256,
                crops: np.ndarray | None = None) -> np.ndarray:
        result = np.zeros((len(index), 2), dtype=np.float32)
        source = data["crop"] if crops is None else crops
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(index), batch):
                part = index[start:start + batch]
                image = torch.from_numpy(source[part].astype(np.float32) / 255.0).to(self.device)
                feature = torch.from_numpy(((data["box"][part] - self.x_mean) / self.x_sd).astype(np.float32)).to(self.device)
                standardized = self.model(image, feature).cpu().numpy()
                result[start:start + len(part)] = standardized * self.y_sd + self.y_mean
        return result


def fit_rgb(data: dict[str, np.ndarray], train: np.ndarray, *, seed: int, epochs: int,
            batch: int, device: torch.device) -> RGBArtifact:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = data["box"]
    y = data["target_ray"]
    x_mean, x_sd = x[train].mean(0), x[train].std(0).clip(1e-3)
    y_mean, y_sd = y[train].mean(0), y[train].std(0).clip(1e-3)
    feature = torch.from_numpy(((x - x_mean) / x_sd).astype(np.float32))
    target = torch.from_numpy(((y - y_mean) / y_sd).astype(np.float32))
    model = RGBMean(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    generator = np.random.default_rng(seed + 1901)
    for _ in range(epochs):
        model.train()
        order = generator.permutation(train)
        for start in range(0, len(order), batch):
            part = order[start:start + batch]
            image = torch.from_numpy(data["crop"][part].astype(np.float32) / 255.0).to(device)
            predicted = model(image, feature[part].to(device))
            loss = torch.nn.functional.mse_loss(predicted, target[part].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return RGBArtifact(model, x_mean, x_sd, y_mean, y_sd, device)


def correction_to_world(data: dict[str, np.ndarray], prediction: np.ndarray,
                        index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual_ray = prediction - data["target_ray"][index]
    residual_world = np.einsum("nij,nj->ni", data["basis"][index], residual_ray)
    corrected = data["raw"][index] + np.einsum("nij,nj->ni", data["basis"][index], prediction)
    return residual_world, corrected


def group_metrics(data: dict[str, np.ndarray], prediction: np.ndarray, index: np.ndarray) -> dict[str, Any]:
    residual_ray = prediction - data["target_ray"][index]
    residual_world, _ = correction_to_world(data, prediction, index)
    error = np.linalg.norm(residual_world, axis=1)
    drives = data["drive"][index]
    by_drive: dict[str, Any] = {}
    for drive in sorted(set(drives)):
        use = drives == drive
        vector = residual_world[use].mean(0)
        by_drive[drive] = {
            "n": int(use.sum()),
            "mse_m2": float(np.mean(error[use] ** 2)),
            "rmse_cm": float(np.sqrt(np.mean(error[use] ** 2)) * 100.0),
            "median_cm": float(np.median(error[use]) * 100.0),
            "p95_cm": float(np.percentile(error[use], 95) * 100.0),
            "signed_along_cm": float(residual_ray[use, 0].mean() * 100.0),
            "signed_across_cm": float(residual_ray[use, 1].mean() * 100.0),
            "world_bias_x_cm": float(vector[0] * 100.0),
            "world_bias_y_cm": float(vector[1] * 100.0),
            "world_bias_norm_cm": float(np.linalg.norm(vector) * 100.0),
        }
    drive_mse = np.asarray([row["mse_m2"] for row in by_drive.values()])
    drive_bias = np.asarray([row["world_bias_norm_cm"] for row in by_drive.values()])
    result: dict[str, Any] = {
        "n": int(len(index)),
        "drive_count": len(by_drive),
        "equal_drive_rmse_cm": float(np.sqrt(drive_mse.mean()) * 100.0),
        "mean_drive_bias_norm_cm": float(drive_bias.mean()),
        "pooled_median_cm": float(np.median(error) * 100.0),
        "pooled_p95_cm": float(np.percentile(error, 95) * 100.0),
        "equal_drive_signed_along_cm": float(np.mean([r["signed_along_cm"] for r in by_drive.values()])),
        "equal_drive_signed_across_cm": float(np.mean([r["signed_across_cm"] for r in by_drive.values()])),
        "by_drive": by_drive,
    }
    result["per_camera"] = {}
    for camera in CAMERAS:
        local = data["camera"][index] == camera
        if not np.any(local):
            continue
        e = error[local]
        ray = residual_ray[local]
        world = residual_world[local]
        result["per_camera"][camera] = {
            "n": int(local.sum()),
            "rmse_cm": float(np.sqrt(np.mean(e ** 2)) * 100.0),
            "median_cm": float(np.median(e) * 100.0),
            "p95_cm": float(np.percentile(e, 95) * 100.0),
            "signed_along_cm": float(ray[:, 0].mean() * 100.0),
            "signed_across_cm": float(ray[:, 1].mean() * 100.0),
            "world_bias_x_cm": float(world[:, 0].mean() * 100.0),
            "world_bias_y_cm": float(world[:, 1].mean() * 100.0),
        }
    result["stationary_moving"] = {}
    for label, local in (("stationary", data["stationary"][index]),
                         ("moving", ~data["stationary"][index])):
        if np.any(local):
            e = error[local]
            ray = residual_ray[local]
            result["stationary_moving"][label] = {
                "n": int(local.sum()),
                "rmse_cm": float(np.sqrt(np.mean(e ** 2)) * 100.0),
                "median_cm": float(np.median(e) * 100.0),
                "p95_cm": float(np.percentile(e, 95) * 100.0),
                "signed_along_cm": float(ray[:, 0].mean() * 100.0),
                "signed_across_cm": float(ray[:, 1].mean() * 100.0),
            }
    return result


def cross_predictions(data: dict[str, np.ndarray], *, mode: str, seed: int, epochs: int,
                      batch: int, device: torch.device) -> dict[str, np.ndarray]:
    fit = np.flatnonzero(data["partition"] == "fit")
    if mode == "drive":
        groups = sorted(set(data["drive"][fit]))
        masks = [(data["drive"] == group) & (data["partition"] == "fit") for group in groups]
    elif mode == "route":
        groups = sorted(set(data["route"][fit]))
        masks = [(data["route"] == group) & (data["partition"] == "fit") for group in groups]
    else:
        raise ValueError(mode)
    output = {name: np.full((len(data["drive"]), 2), np.nan, dtype=float) for name in MODELS}
    output["B0_raw"][fit] = 0.0
    for fold, (group, mask) in enumerate(zip(groups, masks, strict=True)):
        test = np.flatnonzero(mask)
        train = np.flatnonzero((data["partition"] == "fit") & ~mask)
        print(f"{mode} fold {fold + 1}/{len(groups)} hold {group}: train={len(train)} test={len(test)}", flush=True)
        for model_id in MODELS[1:-1]:
            model = fit_tabular(model_id, data, train, seed + fold)
            output[model_id][test] = predict_tabular(model_id, model, data, test)
        rgb = fit_rgb(data, train, seed=seed + fold, epochs=epochs, batch=batch, device=device)
        output["B4_rgb_context"][test] = rgb.predict(data, test, batch=batch)
    for name in MODELS:
        if not np.isfinite(output[name][fit]).all():
            raise RuntimeError(f"incomplete {mode} predictions for {name}")
    return output


def fit_development(data: dict[str, np.ndarray], output: Path, *, seed: int, epochs: int,
                    batch: int, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fit = np.flatnonzero(data["partition"] == "fit")
    development = np.flatnonzero(data["partition"] == "development")
    prediction = {name: np.full((len(data["drive"]), 2), np.nan, dtype=float) for name in MODELS}
    prediction["B0_raw"][development] = 0.0
    artifacts: dict[str, Any] = {}
    for model_id in MODELS[1:-1]:
        model = fit_tabular(model_id, data, fit, seed)
        prediction[model_id][development] = predict_tabular(model_id, model, data, development)
        artifacts[model_id] = model
    print(f"development B4 fit: train={len(fit)} score={len(development)}", flush=True)
    rgb = fit_rgb(data, fit, seed=seed, epochs=epochs, batch=batch, device=device)
    prediction["B4_rgb_context"][development] = rgb.predict(data, development, batch=batch)
    joblib.dump({name: artifacts[name] for name in artifacts}, output / "tabular_models_fit_only.joblib")
    torch.save({
        "schema": "reference_controlled_rgb_bias.v1",
        "model_id": "B4_rgb_context",
        "state_dict": rgb.model.cpu().state_dict(),
        "x_mean": rgb.x_mean.tolist(), "x_sd": rgb.x_sd.tolist(),
        "y_mean": rgb.y_mean.tolist(), "y_sd": rgb.y_sd.tolist(),
        "crop_shape": [3, 48, 48], "seed": seed, "epochs": epochs,
        "target": "reference minus raw projection in raw camera-ray coordinates",
    }, output / "B4_rgb_fit_only.pt")
    rgb.model.to(device)
    shuffled = data["crop"].copy()
    rng = np.random.default_rng(seed + 913)
    for camera in CAMERAS:
        local = development[data["camera"][development] == camera]
        shuffled[local] = shuffled[rng.permutation(local)]
    prediction["B4_rgb_shuffled_diagnostic"] = np.full((len(data["drive"]), 2), np.nan)
    prediction["B4_rgb_shuffled_diagnostic"][development] = rgb.predict(
        data, development, batch=batch, crops=shuffled
    )
    return prediction, artifacts | {"B4_rgb_context": rgb}


def write_predictions(path: Path, data: dict[str, np.ndarray], predictions: dict[str, np.ndarray],
                      index: np.ndarray, model_names: tuple[str, ...] = MODELS) -> None:
    fields = [
        "drive_id", "partition", "route", "direction", "camera_id", "capture_stamp_ns",
        "source_frame_id", "stationary", "range_m", "raw_x", "raw_y", "reference_x",
        "reference_y",
    ]
    for name in model_names:
        fields += [f"{name}_correction_along_m", f"{name}_correction_across_m",
                   f"{name}_corrected_x", f"{name}_corrected_y",
                   f"{name}_residual_x_m", f"{name}_residual_y_m", f"{name}_error_m"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in index:
            row: dict[str, Any] = {
                "drive_id": data["drive"][i], "partition": data["partition"][i],
                "route": data["route"][i], "direction": data["direction"][i],
                "camera_id": data["camera"][i], "capture_stamp_ns": int(data["stamp_ns"][i]),
                "source_frame_id": data["source_frame_id"][i],
                "stationary": int(data["stationary"][i]), "range_m": data["range_m"][i],
                "raw_x": data["raw"][i, 0], "raw_y": data["raw"][i, 1],
                "reference_x": data["reference"][i, 0], "reference_y": data["reference"][i, 1],
            }
            for name in model_names:
                pred = predictions[name][i]
                world_residual, corrected = correction_to_world(data, pred[None, :], np.asarray([i]))
                row.update({
                    f"{name}_correction_along_m": pred[0],
                    f"{name}_correction_across_m": pred[1],
                    f"{name}_corrected_x": corrected[0, 0],
                    f"{name}_corrected_y": corrected[0, 1],
                    f"{name}_residual_x_m": world_residual[0, 0],
                    f"{name}_residual_y_m": world_residual[0, 1],
                    f"{name}_error_m": np.linalg.norm(world_residual[0]),
                })
            writer.writerow(row)


def select_model(summary: dict[str, Any]) -> dict[str, Any]:
    drive_ids = sorted(summary[MODELS[0]]["by_drive"])
    mse = {name: np.asarray([summary[name]["by_drive"][drive]["mse_m2"] for drive in drive_ids])
           for name in MODELS}
    best = min(MODELS, key=lambda name: float(mse[name].mean()))
    eligible = []
    comparisons = {}
    for name in MODELS:
        difference = mse[name] - mse[best]
        se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
        mean = float(difference.mean())
        within = bool(mean <= se + 1e-15)
        comparisons[name] = {"mean_paired_mse_difference_m2": mean, "paired_se_m2": se,
                             "within_one_paired_se": within}
        if within:
            eligible.append(name)
    selected = min(eligible, key=lambda name: COMPLEXITY[name])
    return {
        "schema": "reference_controlled_bias_selection.v1",
        "status": "bias_candidate_selected_audit_still_sealed",
        "primary_statistic": "mean within-drive MSE over eight development drives",
        "rule": "choose the simplest B0-B4 candidate within one paired standard error of the lowest development-drive MSE",
        "best_mean_model": best,
        "selected_model": selected,
        "complexity_order": list(MODELS),
        "development_drive_ids": drive_ids,
        "comparisons_to_best": comparisons,
        "audit_opened": False,
    }


def save_figures(output: Path, data: dict[str, np.ndarray], predictions: dict[str, np.ndarray],
                 route_predictions: dict[str, np.ndarray], summary: dict[str, Any],
                 route_summary: dict[str, Any], selection: dict[str, Any]) -> None:
    development = np.flatnonzero(data["partition"] == "development")
    selected = selection["selected_model"]
    figures = output / "figures"
    figures.mkdir()

    drive_ids = sorted(set(data["drive"][development]))
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    x = np.arange(len(MODELS))
    for drive in drive_ids:
        y = [summary[name]["by_drive"][drive]["rmse_cm"] for name in MODELS]
        ax.plot(x, y, color="#b8b8b2", linewidth=0.8, alpha=0.75)
        ax.scatter(x, y, color=[COLORS[name] for name in MODELS], s=25, zorder=3)
    means = [summary[name]["equal_drive_rmse_cm"] for name in MODELS]
    ax.plot(x, means, color="#111827", linewidth=2.4, marker="o", label="equal-drive result")
    ax.set_xticks(x, [name.split("_", 1)[0] for name in MODELS])
    ax.set_ylabel("camera-reading RMSE [cm]")
    ax.set_title("Bias correction on eight development drives")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.savefig(figures / "01_paired_drive_rmse.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for axis, component, label in zip(axes, ("signed_along_cm", "signed_across_cm"),
                                      ("along camera ray", "across camera ray"), strict=True):
        positions = np.arange(len(CAMERAS))
        width = 0.38
        for offset, name in ((-width / 2, "B0_raw"), (width / 2, selected)):
            values = [summary[name]["per_camera"].get(camera, {}).get(component, np.nan)
                      for camera in CAMERAS]
            axis.bar(positions + offset, values, width, color=COLORS[name], label=name.split("_", 1)[0])
        axis.axhline(0.0, color="#111827", linewidth=0.8)
        axis.set_xticks(positions, [camera[-1] for camera in CAMERAS])
        axis.set_xlabel("camera")
        axis.set_ylabel(f"signed {label} bias [cm]")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Development-drive camera bias before and after correction")
    fig.savefig(figures / "02_per_camera_signed_bias.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for name in MODELS:
        residual, _ = correction_to_world(data, predictions[name][development], development)
        values = np.sort(np.linalg.norm(residual, axis=1) * 100.0)
        ax.plot(values, (np.arange(len(values)) + 1) / len(values), color=COLORS[name],
                linewidth=1.8, label=name.split("_", 1)[0])
    ax.set_xlabel("camera-reading position error [cm]")
    ax.set_ylabel("empirical cumulative fraction")
    ax.set_xlim(left=0.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(figures / "03_development_error_cdf.png", dpi=200)
    plt.close(fig)

    raw_residual, _ = correction_to_world(data, predictions["B0_raw"][development], development)
    selected_residual, _ = correction_to_world(data, predictions[selected][development], development)
    raw_error = np.linalg.norm(raw_residual, axis=1) * 100.0
    selected_error = np.linalg.norm(selected_residual, axis=1) * 100.0
    ranges = data["range_m"][development]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.scatter(ranges, raw_error, s=7, alpha=0.13, color=COLORS["B0_raw"], label="B0 readings")
    ax.scatter(ranges, selected_error, s=7, alpha=0.18, color=COLORS[selected], label=f"{selected.split('_', 1)[0]} readings")
    edges = np.quantile(ranges, np.linspace(0, 1, 13))
    for values, color, label in ((raw_error, COLORS["B0_raw"], "B0 bin median"),
                                 (selected_error, COLORS[selected], f"{selected.split('_', 1)[0]} bin median")):
        centres, medians = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            use = (ranges >= left) & (ranges <= right)
            if np.any(use):
                centres.append(float(np.median(ranges[use])))
                medians.append(float(np.median(values[use])))
        ax.plot(centres, medians, color=color, linewidth=2.4, label=label)
    ax.set_xlabel("raw camera-to-reading range [m]")
    ax.set_ylabel("camera-reading position error [cm]")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(figures / "04_all_points_by_range.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    labels = ("stationary", "moving")
    x = np.arange(2)
    width = 0.36
    for offset, name in ((-width / 2, "B0_raw"), (width / 2, selected)):
        values = [summary[name]["stationary_moving"][label]["rmse_cm"] for label in labels]
        ax.bar(x + offset, values, width, color=COLORS[name], label=name.split("_", 1)[0])
    ax.set_xticks(x, labels)
    ax.set_ylabel("camera-reading RMSE [cm]")
    ax.set_title("Stationary anchors and moving traverses")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.savefig(figures / "05_stationary_moving.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    labels, raw_values, selected_values = [], [], []
    for route in sorted(set(data["route"][development])):
        for direction in ("forward", "reverse"):
            use = development[(data["route"][development] == route) &
                              (data["direction"][development] == direction) &
                              (~data["stationary"][development])]
            if not len(use):
                continue
            labels.append(f"{route.replace('_', ' ')}\n{direction}")
            raw_values.append(float((predictions["B0_raw"][use] - data["target_ray"][use])[:, 0].mean() * 100.0))
            selected_values.append(float((predictions[selected][use] - data["target_ray"][use])[:, 0].mean() * 100.0))
    x = np.arange(len(labels)); width = 0.36
    ax.bar(x - width / 2, raw_values, width, color=COLORS["B0_raw"], label="B0")
    ax.bar(x + width / 2, selected_values, width, color=COLORS[selected], label=selected.split("_", 1)[0])
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("moving signed along-ray bias [cm]")
    ax.set_title("Forward and reverse development traverses")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.savefig(figures / "06_forward_reverse_bias.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    routes = sorted(set(data["route"][data["partition"] == "fit"]))
    x = np.arange(len(routes)); width = 0.15
    for number, name in enumerate(MODELS):
        values = []
        for route in routes:
            use = np.flatnonzero((data["partition"] == "fit") & (data["route"] == route))
            residual, _ = correction_to_world(data, route_predictions[name][use], use)
            values.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))) * 100.0))
        ax.bar(x + (number - 2) * width, values, width, color=COLORS[name], label=name.split("_", 1)[0])
    ax.set_xticks(x, [route.replace("_", "\n") for route in routes])
    ax.set_ylabel("held-route camera-reading RMSE [cm]")
    ax.set_title("Leave-one-route-out fit-drive diagnostic")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=5, fontsize=8)
    fig.savefig(figures / "07_leave_one_route_out.png", dpi=200)
    plt.close(fig)

    _, corrected_selected = correction_to_world(data, predictions[selected][development], development)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), constrained_layout=True)
    for ax, points, title in ((axes[0], data["raw"][development], "B0 raw readings"),
                              (axes[1], corrected_selected, f"{selected.split('_', 1)[0]} corrected readings")):
        ax.scatter(data["reference"][development, 0], data["reference"][development, 1],
                   s=5, color="#111827", alpha=0.16, label="reference at capture")
        for camera in CAMERAS:
            use = data["camera"][development] == camera
            ax.scatter(points[use, 0], points[use, 1], s=6, alpha=0.28,
                       color=CAMERA_COLORS[camera], label=camera[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
        ax.set_title(title)
        ax.grid(alpha=0.15)
    axes[1].legend(title="camera", ncol=3, fontsize=8)
    fig.savefig(figures / "08_all_development_readings.png", dpi=200)
    plt.close(fig)


def write_report(output: Path, input_manifest: dict[str, Any], summary: dict[str, Any],
                 route_summary: dict[str, Any], selection: dict[str, Any],
                 shuffled: dict[str, Any]) -> None:
    selected = selection["selected_model"]
    lines = [
        "# Bias correction on reference-controlled drives", "",
        "The analysis uses eight fit drives and eight development drives. The audit drives remain sealed.", "",
        "## Development results", "",
        "| Model | Equal-drive RMSE [cm] | Mean drive bias [cm] | Median [cm] | p95 [cm] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        row = summary[name]
        lines.append(
            f"| {name.split('_', 1)[0]} | {row['equal_drive_rmse_cm']:.2f} | "
            f"{row['mean_drive_bias_norm_cm']:.2f} | {row['pooled_median_cm']:.2f} | {row['pooled_p95_cm']:.2f} |"
        )
    lines += [
        "", "## Selection", "",
        f"The lowest development-drive MSE is obtained by {selection['best_mean_model'].split('_', 1)[0]}.",
        f"The one-paired-standard-error rule selects {selected.split('_', 1)[0]}.",
        "", "## Route-held-out diagnostic", "",
        "| Model | Equal-drive RMSE [cm] | Mean drive bias [cm] |",
        "|---|---:|---:|",
    ]
    for name in MODELS:
        row = route_summary[name]
        lines.append(f"| {name.split('_', 1)[0]} | {row['equal_drive_rmse_cm']:.2f} | {row['mean_drive_bias_norm_cm']:.2f} |")
    lines += [
        "", "## RGB association diagnostic", "",
        f"B4 with the correct development crops has equal-drive RMSE {summary['B4_rgb_context']['equal_drive_rmse_cm']:.2f} cm. "
        f"Permuting crops within each camera gives {shuffled['equal_drive_rmse_cm']:.2f} cm.",
        "", "## Scope", "",
        "These are individual camera readings scored against the native reference at each capture timestamp. "
        "Rows within a drive are not independent replicates. R, NIS, fusion and belief are not evaluated here.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260911)
    parser.add_argument("--rgb-epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    campaign_root, output = args.campaign_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data = load_data(campaign_root, output)
    fit = np.flatnonzero(data["partition"] == "fit")
    development = np.flatnonzero(data["partition"] == "development")
    print(f"loaded fit={len(fit)} development={len(development)}", flush=True)

    drive_predictions = cross_predictions(
        data, mode="drive", seed=args.seed + 100, epochs=args.rgb_epochs,
        batch=args.batch, device=device
    )
    write_predictions(output / "fit_leave_one_drive_out_predictions.csv", data,
                      drive_predictions, fit)
    drive_summary = {name: group_metrics(data, drive_predictions[name][fit], fit) for name in MODELS}
    atomic_json(output / "fit_leave_one_drive_out_summary.json", drive_summary)

    route_predictions = cross_predictions(
        data, mode="route", seed=args.seed + 200, epochs=args.rgb_epochs,
        batch=args.batch, device=device
    )
    write_predictions(output / "fit_leave_one_route_out_predictions.csv", data,
                      route_predictions, fit)
    route_summary = {name: group_metrics(data, route_predictions[name][fit], fit) for name in MODELS}
    atomic_json(output / "fit_leave_one_route_out_summary.json", route_summary)

    predictions, _ = fit_development(
        data, output, seed=args.seed, epochs=args.rgb_epochs, batch=args.batch, device=device
    )
    write_predictions(output / "development_predictions.csv", data, predictions, development)
    summary = {name: group_metrics(data, predictions[name][development], development) for name in MODELS}
    shuffled = group_metrics(data, predictions["B4_rgb_shuffled_diagnostic"][development], development)
    summary_payload = {
        "schema": "reference_controlled_bias_results.v1",
        "status": "development_scored_audit_still_sealed",
        "layer": "individual camera reading at capture timestamp",
        "reference": "interpolated native ground-truth pose at capture timestamp",
        "aggregation": "within-drive first; eight development drives weighted equally",
        "models": summary,
        "B4_shuffled_rgb_diagnostic": shuffled,
        "audit_opened": False,
    }
    atomic_json(output / "development_results.json", summary_payload)
    selection = select_model(summary)
    atomic_json(output / "bias_selection.json", selection)
    save_figures(output, data, predictions, route_predictions, summary, route_summary, selection)
    input_manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    write_report(output, input_manifest, summary, route_summary, selection, shuffled)
    atomic_json(output / "completion.json", {
        "schema": "reference_controlled_bias_completion.v1",
        "status": "complete_bias_only_audit_still_sealed",
        "selected_model": selection["selected_model"],
        "source_sha256": sha256(Path(__file__)),
        "artifacts": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
    })
    print(json.dumps({"selected": selection["selected_model"],
                      "best": selection["best_mean_model"],
                      "development": {name: summary[name]["equal_drive_rmse_cm"] for name in MODELS},
                      "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
