"""Forward-only runtime for the commissioned box-MLP visibility residual."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from torch import nn


GRID_SIZE = 16
SUPPORTED_SCHEMA = "commissioned_visibility_sensor_model.v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_verified(entry: dict, *, label: str) -> tuple[Path, bytes]:
    path = Path(str(entry.get("path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = path.read_bytes()
    if _sha256_bytes(data) != str(entry.get("sha256", "")):
        raise ValueError(f"{label} hash differs from the commissioned manifest")
    return path, data


class _VisibilityPatchResidualNet(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.visibility = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.features = nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(24 * 4 * 4 + 32, 96), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(96, 48), nn.ReLU(),
        )
        self.residual = nn.Linear(48, 2)
        self.gate = nn.Linear(48, 1)

    def forward(self, grid, features, base):
        encoded = torch.cat((self.visibility(grid), self.features(features)), dim=1)
        hidden = self.trunk(encoded)
        return base + torch.sigmoid(self.gate(hidden)) * self.residual(hidden)


MEAN_MODEL_NAMES = ("box_mlp_visibility_residual", "M4_visibility_patch_residual")


class CommissionedVisibilitySensorModel:
    """Hash-bound visibility-residual correction and its matched covariance."""

    def __init__(self, manifest_path: str | Path, *, expected_sha256: str | None = None) -> None:
        path = Path(manifest_path).expanduser()
        data = path.read_bytes()
        self.sha256 = _sha256_bytes(data)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("commissioned visibility manifest hash differs from expected identity")
        manifest = json.loads(data)
        if manifest.get("schema") != SUPPORTED_SCHEMA:
            raise ValueError("unsupported commissioned visibility schema")
        if manifest.get("status") != "frozen_before_audit" or manifest.get("audit_accessed") is not False:
            raise ValueError("commissioned visibility model was not frozen before audit")
        # Two registry spellings name the same model: the covariance fit writes
        # "box_mlp_visibility_residual", the 5 Hz bundle rewrites it as
        # "M4_visibility_patch_residual". Both are the box-MLP base plus the
        # gated patch residual, and the file hashes below pin the identity.
        if manifest.get("mean_model") not in MEAN_MODEL_NAMES:
            raise ValueError("runtime requires the selected visibility-residual mean model")
        if manifest.get("runtime_covariance_model") != "R4_image_conditioned_scale":
            raise ValueError("runtime requires the visibility residual's matched covariance")
        self.manifest = manifest
        self.path = path
        self.camera_order = tuple(str(value) for value in manifest["camera_order"])
        self.camera_xy = {
            str(key): np.asarray(value, dtype=float)
            for key, value in manifest["camera_xy_m"].items()
        }
        if set(self.camera_order) != set(self.camera_xy):
            raise ValueError("camera registry and geometry differ")
        self.height, self.width = map(int, manifest["image_shape_hw"])

        _, base_data = _read_verified(manifest["correction_base"], label="box MLP base model")
        self.base_model = joblib.load(io.BytesIO(base_data))
        _, patch_data = _read_verified(manifest["correction_patch"], label="visibility residual model")
        try:
            checkpoint = torch.load(io.BytesIO(patch_data), map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            checkpoint = torch.load(io.BytesIO(patch_data), map_location="cpu")
        if checkpoint.get("schema") != "visibility_patch_residual.v1" or checkpoint.get("grid_size") != GRID_SIZE:
            raise ValueError("visibility-residual checkpoint feature contract is unsupported")
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_sd = np.asarray(checkpoint["feature_sd"], dtype=np.float32)
        if self.feature_mean.shape != self.feature_sd.shape:
            raise ValueError("visibility-residual feature normalization shape mismatch")
        self.patch_model = _VisibilityPatchResidualNet(len(self.feature_mean))
        self.patch_model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.patch_model.eval()

        _, parameter_data = _read_verified(manifest["parameters"], label="covariance parameters")
        with np.load(io.BytesIO(parameter_data), allow_pickle=False) as archive:
            for name in archive.files:
                setattr(self, name, np.asarray(archive[name]))
        if self.spatial_base.shape != (len(self.camera_order), 2, 2):
            raise ValueError("spatial base covariance has the wrong shape")
        self.spatial_spec = manifest["spatial"]
        self.image_spec = manifest["image_scale"]

    def _features(
        self, camera_id: str, raw_xy: Sequence[float], bbox_xyxy: Sequence[float], confidence: float
    ) -> np.ndarray:
        if camera_id not in self.camera_xy:
            raise ValueError("camera is outside the commissioned registry")
        raw = np.asarray(raw_xy, dtype=float)
        box = np.asarray(bbox_xyxy, dtype=float)
        if raw.shape != (2,) or box.shape != (4,) or not np.isfinite(np.r_[raw, box, confidence]).all():
            raise ValueError("runtime correction inputs are malformed")
        dx, dy = raw - self.camera_xy[camera_id]
        distance = math.hypot(float(dx), float(dy))
        if distance <= 1e-9 or box[2] <= box[0] or box[3] <= box[1] or not 0.0 <= confidence <= 1.0:
            raise ValueError("runtime correction inputs are outside support")
        width_fraction = (box[2] - box[0]) / self.width
        height_fraction = (box[3] - box[1]) / self.height
        values = [
            distance, 1.0 / distance, math.sin(math.atan2(dy, dx)), math.cos(math.atan2(dy, dx)),
            width_fraction, height_fraction,
            (box[2] - box[0]) / max(box[3] - box[1], 1e-9),
            0.5 * (box[0] + box[2]) / self.width, box[3] / self.height, confidence,
            *[float(camera_id == value) for value in self.camera_order],
        ]
        return np.asarray(values, dtype=np.float32)

    def correction_ray(
        self, camera_id: str, raw_xy: Sequence[float], bbox_xyxy: Sequence[float],
        confidence: float, visibility_grid: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        features = self._features(camera_id, raw_xy, bbox_xyxy, confidence)
        grid = np.asarray(visibility_grid, dtype=np.float32)
        if grid.size != GRID_SIZE * GRID_SIZE:
            raise ValueError("visibility grid must contain exactly 256 values")
        grid = grid.reshape(1, GRID_SIZE, GRID_SIZE)
        if not np.isfinite(grid).all() or grid.min() < 0.0 or grid.max() > 1.0:
            raise ValueError("visibility grid must be finite and in [0,1]")
        base = np.asarray(self.base_model.predict(features[None])[0], dtype=np.float32)
        with torch.inference_mode():
            prediction = self.patch_model(
                torch.from_numpy(grid[None]),
                torch.from_numpy(((features - self.feature_mean) / self.feature_sd)[None]),
                torch.from_numpy(base[None]),
            )[0].numpy()
        if prediction.shape != (2,) or not np.isfinite(prediction).all():
            raise ValueError("visibility-residual model produced an invalid correction")
        return prediction.astype(float)

    def _spatial_covariance(self, camera_id: str, x: float, y: float, heading: float) -> np.ndarray:
        selected = self.spatial_camera == camera_id
        train = self.spatial_features[selected]
        residual = self.spatial_residual[selected]
        weight = self.spatial_sample_weight[selected]
        distance2 = np.sum((train[:, :2] - np.asarray([x, y])) ** 2, axis=1)
        heading_term = 1.0 - np.cos(float(heading) - train[:, 2])
        local = np.exp(
            -0.5 * distance2 / float(self.spatial_spec["position_scale_m"]) ** 2
            - heading_term / float(self.spatial_spec["heading_scale"]) ** 2
        ) * weight
        camera_index = self.camera_order.index(camera_id)
        prior = float(self.spatial_spec["prior_strength"])
        numerator = np.einsum("n,ni,nj->ij", local, residual, residual)
        covariance = (numerator + prior * self.spatial_base[camera_index]) / (local.sum() + prior)
        return 0.5 * (covariance + covariance.T) + np.eye(2) * 1.0e-6

    def planner_covariance(self, camera_id: str, x: float, y: float, heading: float) -> np.ndarray:
        covariance = self._spatial_covariance(camera_id, x, y, heading)
        covariance *= float(self.spatial_spec["external_calibration_scale"])
        return covariance

    def correct_and_covariance(
        self, camera_id: str, raw_xy: Sequence[float], bbox_xyxy: Sequence[float],
        confidence: float, visibility_grid: Sequence[float] | np.ndarray, heading: float,
    ) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
        correction = self.correction_ray(camera_id, raw_xy, bbox_xyxy, confidence, visibility_grid)
        raw = np.asarray(raw_xy, dtype=float)
        ray = raw - self.camera_xy[camera_id]
        along = ray / np.linalg.norm(ray)
        basis = np.column_stack((along, np.asarray([-along[1], along[0]])))
        corrected = raw + basis @ correction
        covariance_ray = self._spatial_covariance(camera_id, corrected[0], corrected[1], heading)
        grid = np.asarray(visibility_grid, dtype=float).reshape(1, GRID_SIZE, GRID_SIZE)
        visibility_score = 0.5 * float(grid.mean()) + 0.5 * float(grid[:, 3 * GRID_SIZE // 4 :, :].mean())
        features = self._features(camera_id, raw_xy, bbox_xyxy, confidence)
        image_row = np.asarray([
            visibility_score, features[4], features[5], confidence, features[0],
            features[2], features[3], *features[10:],
        ], dtype=float)
        standardized = (image_row - self.image_feature_mean) / self.image_feature_scale
        raw_log_scale = float(np.dot(self.image_ridge_coef, standardized) + self.image_ridge_intercept)
        log_scale = raw_log_scale + float(self.image_spec["internal_log_calibration"])
        scale = float(np.exp(np.clip(
            log_scale,
            math.log(float(self.image_spec["minimum"])),
            math.log(float(self.image_spec["maximum"])),
        )))
        covariance_ray *= scale * float(self.image_spec["external_calibration_scale"])
        covariance_world = basis @ covariance_ray @ basis.T
        covariance_world = 0.5 * (covariance_world + covariance_world.T)
        return (
            (float(corrected[0]), float(corrected[1])),
            ((float(covariance_world[0, 0]), float(covariance_world[0, 1])),
             (float(covariance_world[1, 0]), float(covariance_world[1, 1]))),
        )
