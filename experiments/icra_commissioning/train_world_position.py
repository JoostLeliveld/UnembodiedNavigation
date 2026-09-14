#!/usr/bin/env python3
"""Train and audit a mean-only, image-aware world-position correction.

The frozen YOLO box bottom is first back-projected to the floor.  The network predicts
only the residual to the commanded robot-centre XY, in the camera-ray basis.  Its public
output is therefore a world-frame XY observation.  It never predicts R.

This is grouped development evidence from the existing static characterization capture.
Occlusion severity is used only for training balance and evaluation strata; it is not a
runtime input.  Boxes censored by the image boundary are refused before training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "logs/studies/perception_bayesian_gaussian"
PIXELS = STUDY / "data/pixels/pixels.npz"
CAPTURE = REPO / "logs/perception_datasets/warehouse_v2_bbox_characterization_20260831"
DETECTOR_SOURCE = REPO / "logs/perception_datasets/warehouse_v2_yolo_20260821"
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FEATURE_NAMES = (
    "range_m", "inv_range", "box_w_frac", "box_h_frac", "box_aspect",
    "u_frac", "v_frac", "bearing_cos", "bearing_sin", "confidence",
    *(f"is_{camera}" for camera in CAMERAS),
)
SEVERITY_NAMES = ("severe", "moderate", "mild", "clean")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def severity_index(ratio: np.ndarray) -> np.ndarray:
    """Training-only strata: severe <.6, moderate <.8, mild <.95, else clean."""
    return np.digitize(np.asarray(ratio, dtype=float), (0.6, 0.8, 0.95)).astype(np.int64)


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    ray = np.asarray(raw_xy, dtype=float) - np.asarray(camera_xy, dtype=float)
    norm = float(np.linalg.norm(ray))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("invalid camera-to-reading ray")
    along = ray / norm
    left = np.array([-along[1], along[0]], dtype=float)
    return np.column_stack((along, left))


def detector_image_hashes() -> dict[str, set[str]]:
    hashes = {"train": set(), "val": set()}
    for camera in CAMERAS:
        path = DETECTOR_SOURCE / camera / "label_diagnostics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                role = row.get("split", "")
                value = row.get("image_sha1", "")
                if row.get("accepted") == "1" and role in hashes and value:
                    hashes[role].add(value)
    return hashes


def prepare() -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REPO / "experiments/camera_observation_characterization"))
    from derive_interpretations import camera_models

    source = np.load(PIXELS, allow_pickle=True)
    manifest = json.loads((CAPTURE / "capture_manifest.json").read_text())
    models = camera_models(manifest)
    camera_specs = {entry["camera_id"]: entry for entry in manifest["cameras"]}

    with (CAPTURE / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
        truth_rows = {(row["pose_id"], row["camera_id"]): row for row in csv.DictReader(handle)}
    with (CAPTURE / "bbox_observations.csv").open(newline="", encoding="utf-8") as handle:
        box_rows = {(row["pose_id"], row["camera_id"]): row for row in csv.DictReader(handle)
                    if row["detected"] == "1"}

    n = len(source["split"])
    raw_world = np.full((n, 2), np.nan, dtype=np.float32)
    truth_world = np.full((n, 2), np.nan, dtype=np.float32)
    basis = np.full((n, 2, 2), np.nan, dtype=np.float32)
    features = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    centre_in_frame = np.zeros(n, dtype=bool)

    for i, (pose_id, camera_id) in enumerate(zip(source["pose_id"], source["camera"])):
        camera_id, pose_id = str(camera_id), str(pose_id)
        truth_row = truth_rows[pose_id, camera_id]
        box = box_rows[pose_id, camera_id]
        camera = models[camera_id]
        raw = camera.pixel_to_world(*map(float, source["raw_px"][i]))
        if raw is None:
            continue
        truth = np.array([float(truth_row["robot_x"]), float(truth_row["robot_y"])])
        u_truth, v_truth, visible = camera.world_to_pixel(*truth, 0.0)
        centre_in_frame[i] = bool(visible and 0 <= u_truth < camera.img_width
                                  and 0 <= v_truth < camera.img_height)
        raw = np.asarray(raw, dtype=float)
        B = ray_basis(camera.cam_pos[:2], raw)
        x0, y0, x1, y1 = (float(box[key]) for key in ("x0", "y0", "x1", "y1"))
        width, height = float(camera.img_width), float(camera.img_height)
        distance = float(np.linalg.norm(raw - camera.cam_pos[:2]))
        angle = math.atan2(raw[1] - camera.cam_pos[1], raw[0] - camera.cam_pos[0])
        yaw = float(camera_specs[camera_id]["pose_xyz_rpy"][5])
        relative = angle - yaw
        values = [
            distance, 1.0 / max(distance, 1e-3), (x1-x0)/width, (y1-y0)/height,
            (x1-x0)/max(y1-y0, 1.0), 0.5*(x0+x1)/width, y1/height,
            math.cos(relative), math.sin(relative), float(box["confidence"]),
            *(float(camera_id == candidate) for candidate in CAMERAS),
        ]
        raw_world[i], truth_world[i], basis[i], features[i] = raw, truth, B, values

    target_world = truth_world - raw_world
    target_ray = np.einsum("nji,nj->ni", basis, target_world).astype(np.float32)
    overlap = detector_image_hashes()
    overlap_any = np.isin(source["image_sha1"], list(overlap["train"] | overlap["val"]))
    finite = (np.isfinite(features).all(1) & np.isfinite(target_ray).all(1)
              & np.isfinite(raw_world).all(1))
    eligible = (finite & centre_in_frame & ~source["box_at_edge"].astype(bool)
                & ~overlap_any)

    return {
        "images": source["crops"], "features": features, "raw_world": raw_world,
        "truth_world": truth_world, "target_ray": target_ray, "basis": basis,
        "split": source["split"], "place": source["place"], "camera": source["camera"],
        "severity": severity_index(source["height_ratio"]), "height_ratio": source["height_ratio"],
        "eligible": eligible, "box_at_edge": source["box_at_edge"].astype(bool),
        "centre_in_frame": centre_in_frame, "overlap": overlap_any,
        "image_sha1": source["image_sha1"],
        "split_overlap": np.zeros(n, dtype=bool),
    }


def prepare_capture(capture_path: Path, *, role_override: str | None = None,
                    load_images: bool = True) -> dict[str, np.ndarray]:
    """Build the runtime-input dataset directly from a bbox capture.

    ``load_images=False`` is for residual audits that consume saved predictions and
    need only labels/features. It avoids decoding every full-resolution RGB frame.
    """
    from PIL import Image

    sys.path.insert(0, str(REPO / "experiments/camera_observation_characterization"))
    from derive_interpretations import camera_models

    capture_path = capture_path.expanduser().resolve()
    manifest = json.loads((capture_path / "capture_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"capture is not complete: {capture_path}")
    models = camera_models(manifest)
    camera_specs = {entry["camera_id"]: entry for entry in manifest["cameras"]}
    with (capture_path / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
        capture = {(row["pose_id"], row["camera_id"]): row for row in csv.DictReader(handle)}
    with (capture_path / "bbox_observations.csv").open(newline="", encoding="utf-8") as handle:
        detections = [row for row in csv.DictReader(handle) if row["detected"] == "1"]

    n = len(detections)
    images = np.zeros((n, 3, 96, 96), dtype=np.uint8) if load_images else np.empty((n, 0), dtype=np.uint8)
    raw_world = np.full((n, 2), np.nan, dtype=np.float32)
    truth_world = np.full((n, 2), np.nan, dtype=np.float32)
    basis = np.full((n, 2, 2), np.nan, dtype=np.float32)
    features = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    height_ratio = np.full(n, np.nan, dtype=np.float32)
    centre_in_frame = np.zeros(n, dtype=bool)
    box_at_edge = np.zeros(n, dtype=bool)
    split, place, camera_ids, hashes = [], [], [], []
    cache_path, cache_image = None, None

    for i, box in enumerate(detections):
        key = (box["pose_id"], box["camera_id"])
        row = capture[key]
        camera_id = str(box["camera_id"])
        camera = models[camera_id]
        width, height = int(camera.img_width), int(camera.img_height)
        image_path = capture_path / row["image"]
        if load_images and image_path != cache_path:
            cache_image = Image.open(image_path).convert("RGB")
            cache_path = image_path
        x0, y0, x1, y1 = (float(box[name]) for name in ("x0", "y0", "x1", "y1"))
        box_w, box_h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        if load_images:
            crop = cache_image.crop((
                int(max(0, x0 - 0.5 * box_w)), int(max(0, y0 - 0.5 * box_h)),
                int(min(width, x1 + 0.5 * box_w)), int(min(height, y1 + 0.5 * box_h)),
            ))
            images[i] = np.asarray(crop.resize((96, 96), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)

        raw = camera.pixel_to_world(float(box["u_bbox_bottom"]), float(box["v_bbox_bottom"]))
        truth = np.asarray([float(row["robot_x"]), float(row["robot_y"])], dtype=float)
        if raw is not None:
            raw = np.asarray(raw, dtype=float)
            B = ray_basis(camera.cam_pos[:2], raw)
            distance = float(np.linalg.norm(raw - camera.cam_pos[:2]))
            angle = math.atan2(raw[1] - camera.cam_pos[1], raw[0] - camera.cam_pos[0])
            yaw = float(camera_specs[camera_id]["pose_xyz_rpy"][5])
            relative = angle - yaw
            raw_world[i], truth_world[i], basis[i] = raw, truth, B
            features[i] = [
                distance, 1.0 / max(distance, 1e-3), box_w / width, box_h / height,
                box_w / box_h, 0.5 * (x0 + x1) / width, y1 / height,
                math.cos(relative), math.sin(relative), float(box["confidence"]),
                *(float(camera_id == candidate) for candidate in CAMERAS),
            ]
        u_truth, v_truth, visible = camera.world_to_pixel(*truth, 0.0)
        centre_in_frame[i] = bool(visible and 0 <= u_truth < width and 0 <= v_truth < height)
        box_at_edge[i] = bool(y1 >= height - 0.5 or y0 <= 0.5 or x1 >= width - 0.5 or x0 <= 0.5)
        expected_h = float(row["expected_y1"]) - float(row["expected_y0"])
        height_ratio[i] = box_h / expected_h if expected_h > 1e-6 else np.nan
        label = str(row.get("dataset_split", ""))
        split.append(role_override or label.split(":", 1)[0])
        place.append(f"{capture_path.name}:{row['position_id']}")
        camera_ids.append(camera_id)
        hashes.append(row["image_sha1"])

    target_world = truth_world - raw_world
    target_ray = np.einsum("nji,nj->ni", basis, target_world).astype(np.float32)
    overlap_sets = detector_image_hashes()
    overlap = np.isin(np.asarray(hashes), list(overlap_sets["train"] | overlap_sets["val"]))
    finite = (np.isfinite(features).all(1) & np.isfinite(target_ray).all(1)
              & np.isfinite(raw_world).all(1) & np.isfinite(height_ratio))
    eligible = finite & centre_in_frame & ~box_at_edge & ~overlap
    return {
        "images": images, "features": features, "raw_world": raw_world,
        "truth_world": truth_world, "target_ray": target_ray, "basis": basis,
        "split": np.asarray(split), "place": np.asarray(place),
        "camera": np.asarray(camera_ids), "severity": severity_index(height_ratio),
        "height_ratio": height_ratio, "eligible": eligible,
        "box_at_edge": box_at_edge, "centre_in_frame": centre_in_frame,
        "overlap": overlap, "image_sha1": np.asarray(hashes),
        "split_overlap": np.zeros(n, dtype=bool),
    }


def combine_datasets(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    combined = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    roles_by_hash: dict[str, set[str]] = {}
    for image_hash, role in zip(combined["image_sha1"], combined["split"]):
        if role in {"fit", "validation", "development"}:
            roles_by_hash.setdefault(str(image_hash), set()).add(str(role))
    leaking = {image_hash for image_hash, roles in roles_by_hash.items() if len(roles) > 1}
    combined["split_overlap"] = np.isin(combined["image_sha1"], list(leaking))
    combined["eligible"] &= ~combined["split_overlap"]
    return combined


class WorldResidualNet(nn.Module):
    """Mean-only residual in camera-ray metres; no covariance output."""

    def __init__(self, feature_dim: int, use_image: bool):
        super().__init__()
        self.use_image = bool(use_image)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(96, 96, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim + (96 if use_image else 0), 192), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        value = torch.cat((self.vision(image), feature), 1) if self.use_image else feature
        return self.head(value)


def group_sampling_weights(camera: np.ndarray, severity: np.ndarray,
                           indices: np.ndarray, balanced: bool) -> np.ndarray:
    if not balanced:
        return np.ones(len(indices), dtype=float) / len(indices)
    keys = [(str(camera[i]), int(severity[i])) for i in indices]
    counts = {key: keys.count(key) for key in set(keys)}
    # Inverse square-root balance improves rare-severe exposure without replaying the
    # smallest camera/severity cells as aggressively as full inverse-frequency balance.
    weights = np.array([1.0 / math.sqrt(counts[key]) for key in keys], dtype=float)
    return weights / weights.sum()


def validation_score(error_ray: np.ndarray, camera: np.ndarray,
                     severity: np.ndarray, indices: np.ndarray,
                     bias_weight: float = 0.0) -> float:
    norm = np.linalg.norm(error_ray[indices], axis=1)
    natural = math.sqrt(float(np.mean(norm ** 2)))
    group_rms, group_bias = [], []
    for cam in CAMERAS:
        for level in range(4):
            selected = indices[(camera[indices] == cam) & (severity[indices] == level)]
            if len(selected):
                group_rms.append(math.sqrt(float(np.mean(np.sum(error_ray[selected] ** 2, axis=1)))))
                if len(selected) >= 10:
                    group_bias.append(float(np.linalg.norm(np.mean(error_ray[selected], axis=0))))
    return (0.5 * natural + 0.5 * float(np.mean(group_rms))
            + float(bias_weight) * float(np.mean(group_bias)))


def infer(model: nn.Module, images: np.ndarray, features: torch.Tensor,
          indices: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    prediction = np.full((len(images), 2), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for part in np.array_split(indices, max(1, math.ceil(len(indices) / batch))):
            image = torch.from_numpy(images[part]).to(device=device, dtype=torch.float32) / 255.0
            prediction[part] = model(image, features[part].to(device)).cpu().numpy()
    return prediction


def fit(data: dict[str, np.ndarray], *, use_image: bool, balanced: bool, seed: int,
        epochs: int, patience: int, batch: int, device: torch.device,
        output: Path, name: str, loss_mode: str = "smooth_l1",
        selection_bias_weight: float = 0.0) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = np.flatnonzero(data["eligible"] & (data["split"] == "fit"))
    select = np.flatnonzero(data["eligible"] & (data["split"] == "validation"))
    all_eligible = np.flatnonzero(data["eligible"])
    centre = data["features"][train].mean(0)
    spread = data["features"][train].std(0).clip(1e-3)
    feature = torch.from_numpy(((data["features"] - centre) / spread).astype(np.float32))
    target = torch.from_numpy(data["target_ray"].astype(np.float32))
    probabilities = group_sampling_weights(data["camera"], data["severity"], train, balanced)
    model = WorldResidualNet(feature.shape[1], use_image).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    rng = np.random.default_rng(seed + 1709)
    best, best_epoch, waited = float("inf"), -1, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(epochs):
        model.train()
        order = rng.choice(train, size=len(train), replace=balanced, p=probabilities)
        for part in np.array_split(order, max(1, math.ceil(len(order) / batch))):
            image = torch.from_numpy(data["images"][part]).to(device=device, dtype=torch.float32) / 255.0
            if use_image:
                # Photometric augmentation only: geometry and the world label stay exact.
                gain = torch.empty((len(part), 1, 1, 1), device=device).uniform_(0.85, 1.15)
                bias = torch.empty((len(part), 1, 1, 1), device=device).uniform_(-0.05, 0.05)
                image = (image * gain + bias).clamp(0.0, 1.0)
            predicted = model(image, feature[part].to(device))
            if loss_mode == "mse":
                # Conditional-mean regression is the correct objective when success
                # means a zero-mean residual rather than merely a robust small median.
                loss = F.mse_loss(predicted, target[part].to(device))
            elif loss_mode == "smooth_l1":
                loss = F.smooth_l1_loss(predicted, target[part].to(device), beta=0.10)
            else:
                raise ValueError(f"unsupported loss mode {loss_mode!r}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        predicted = infer(model, data["images"], feature, select, device, batch)
        error = predicted - data["target_ray"]
        score = validation_score(error, data["camera"], data["severity"], select,
                                 bias_weight=selection_bias_weight)
        history.append({"epoch": epoch + 1, "selection_score_m": score})
        if score < best - 1e-5:
            best, best_epoch, waited = score, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                break
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"{name} seed={seed} epoch={epoch+1} selection={score:.5f}m", flush=True)

    model.load_state_dict(best_state)
    prediction = infer(model, data["images"], feature, all_eligible, device, batch)
    payload = {
        "schema": "image_world_residual.v1", "name": name, "seed": seed,
        "use_image": use_image, "balanced": balanced, "feature_names": FEATURE_NAMES,
        "feature_mean": centre.tolist(), "feature_std": spread.tolist(),
        "target": "truth minus raw floor projection, camera-ray along/across metres",
        "public_output": "raw world XY plus rotated predicted residual",
        "predicts_R": False, "best_epoch": best_epoch, "selection_score_m": best,
        "loss_mode": loss_mode, "selection_bias_weight": selection_bias_weight,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(payload, output / f"{name}_seed{seed}.pt")
    (output / f"{name}_seed{seed}_history.json").write_text(json.dumps(history, indent=2))
    return prediction, {k: v for k, v in payload.items() if k != "state_dict"}


def metric(error_world: np.ndarray, mask: np.ndarray, error_ray: np.ndarray) -> dict[str, object]:
    norm = np.linalg.norm(error_world[mask], axis=1)
    ray = error_ray[mask]
    if not len(norm):
        return {"n": 0}
    return {
        "n": int(len(norm)), "median_cm": float(np.median(norm) * 100),
        "rms_cm": float(math.sqrt(np.mean(norm ** 2)) * 100),
        "p90_cm": float(np.percentile(norm, 90) * 100),
        "p95_cm": float(np.percentile(norm, 95) * 100),
        "max_cm": float(np.max(norm) * 100),
        "above_20cm": int(np.sum(norm > 0.2)), "above_50cm": int(np.sum(norm > 0.5)),
        "above_1m": int(np.sum(norm > 1.0)),
        "along_bias_cm": float(np.mean(ray[:, 0]) * 100),
        "across_bias_cm": float(np.mean(ray[:, 1]) * 100),
    }


def score_prediction(data: dict[str, np.ndarray], prediction: np.ndarray,
                     role: str, admitted: np.ndarray | None = None) -> dict[str, object]:
    base = data["eligible"] & (data["split"] == role)
    if admitted is not None:
        base &= admitted
    error_ray = prediction - data["target_ray"]
    error_world = np.einsum("nij,nj->ni", data["basis"], error_ray)
    result = {}
    strata = {
        "all": np.ones(len(base), dtype=bool), "clean": data["severity"] == 3,
        "partial": data["severity"] < 3, "severe": data["severity"] == 0,
    }
    for key, subset in strata.items():
        result[key] = metric(error_world, base & subset, error_ray)
    result["per_camera_partial"] = {
        camera: metric(error_world, base & (data["camera"] == camera)
                       & (data["severity"] < 3), error_ray) for camera in CAMERAS
    }
    result["eligible_before_gate"] = int((data["eligible"] & (data["split"] == role)).sum())
    result["retained"] = int(base.sum())
    result["retained_fraction"] = float(base.sum() / max(result["eligible_before_gate"], 1))
    return result


def choose_simple_gates(data: dict[str, np.ndarray], seeds: np.ndarray,
                        ensemble: np.ndarray) -> dict[str, dict[str, object]]:
    """Select small observable gates on validation only, with partial-retention guards."""
    select = data["eligible"] & (data["split"] == "validation")
    dev = data["eligible"] & (data["split"] == "development")
    disagreement = np.sqrt(np.mean(np.sum((seeds - ensemble[None]) ** 2, axis=2), axis=0))
    magnitude = np.linalg.norm(ensemble, axis=1)
    candidates = {"ensemble_disagreement": disagreement, "correction_magnitude": magnitude}
    chosen = {}
    for name, value in candidates.items():
        best = None
        for quantile in (0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 1.0):
            threshold = float(np.quantile(value[select], quantile))
            keep = value <= threshold
            all_retained = float(np.mean(keep[select]))
            partial = select & (data["severity"] < 3)
            severe = select & (data["severity"] == 0)
            partial_retained = float(np.mean(keep[partial])) if partial.any() else 1.0
            severe_retained = float(np.mean(keep[severe])) if severe.any() else 1.0
            if all_retained < 0.95 or partial_retained < 0.85 or severe_retained < 0.80:
                continue
            score = score_prediction(data, ensemble, "validation", keep)["all"]["rms_cm"]
            candidate = (score, -all_retained, threshold, quantile, keep,
                         partial_retained, severe_retained)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            continue
        _, _, threshold, quantile, keep, partial_retained, severe_retained = best
        chosen[name] = {
            "threshold_m": threshold, "validation_quantile": quantile,
            "validation_partial_retained": partial_retained,
            "validation_severe_retained": severe_retained,
            "validation": score_prediction(data, ensemble, "validation", keep),
            "development": score_prediction(data, ensemble, "development", keep),
            "development_rejected": int(np.sum(dev & ~keep)),
        }
    return chosen


def choose_seed_aggregation(data: dict[str, np.ndarray], stack: np.ndarray,
                            bias_weight: float = 0.0) -> tuple[np.ndarray, dict[str, object]]:
    """Choose mean or coordinate-median ensemble on validation, never development."""
    eligible = np.flatnonzero(data["eligible"])
    validation = np.flatnonzero(data["eligible"] & (data["split"] == "validation"))
    candidates = {}
    for name, values in (
        ("mean", np.mean(stack[:, eligible], axis=0)),
        ("coordinate_median", np.median(stack[:, eligible], axis=0)),
    ):
        prediction = np.full_like(stack[0], np.nan)
        prediction[eligible] = values
        score = validation_score(prediction - data["target_ray"], data["camera"],
                                 data["severity"], validation, bias_weight=bias_weight)
        candidates[name] = (score, prediction)
    selected = min(candidates, key=lambda name: candidates[name][0])
    return candidates[selected][1], {
        "selected": selected,
        "validation_selection_score_m": {
            name: float(candidate[0]) for name, candidate in candidates.items()
        },
    }


def census(data: dict[str, np.ndarray]) -> dict[str, object]:
    out = {
        "total": len(data["split"]), "eligible": int(data["eligible"].sum()),
        "boundary_refused": int(data["box_at_edge"].sum()),
        "centre_outside_image": int((~data["centre_in_frame"]).sum()),
        "detector_overlap_removed": int(data["overlap"].sum()),
        "cross_split_image_overlap_removed": int(data["split_overlap"].sum()),
        "roles": {},
    }
    for role in ("fit", "validation", "development"):
        mask = data["eligible"] & (data["split"] == role)
        out["roles"][role] = {
            "n": int(mask.sum()), "places": int(len(set(data["place"][mask]))),
            "severity": {SEVERITY_NAMES[level]: int(np.sum(mask & (data["severity"] == level)))
                         for level in range(4)},
            "per_camera": {camera: int(np.sum(mask & (data["camera"] == camera)))
                           for camera in CAMERAS},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Optional subset of scalar_balanced, rgb_uniform, rgb_balanced, rgb_balanced_mean.")
    parser.add_argument("--augmentation-capture", type=Path,
                        help="Complete geometry-selected capture carrying fit:/validation: strata.")
    parser.add_argument("--transfer-capture", type=Path,
                        help="Complete spatially separate capture scored only as development.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = prepare()
    parts = [base]
    if args.augmentation_capture:
        parts.append(prepare_capture(args.augmentation_capture))
    if args.transfer_capture:
        base["split"] = np.where(base["split"] == "development",
                                 "historical_development", base["split"])
        parts.append(prepare_capture(args.transfer_capture, role_override="development"))
    data = combine_datasets(parts)
    eligible = np.flatnonzero(data["eligible"])
    raw_prediction = np.zeros_like(data["target_ray"], dtype=np.float32)
    results: dict[str, object] = {
        "schema": "image_world_training.v1", "status": "grouped_static_development",
        "device": str(device), "census": census(data),
        "raw": score_prediction(data, raw_prediction, "development"), "arms": {},
    }
    available = {
        "scalar_balanced": (False, True, "smooth_l1", 0.0),
        "rgb_uniform": (True, False, "smooth_l1", 0.0),
        "rgb_balanced": (True, True, "smooth_l1", 0.0),
        # MSE estimates the conditional mean. Validation additionally penalizes
        # supported camera/severity-group mean magnitude one-for-one in metres.
        "rgb_balanced_mean": (True, True, "mse", 1.0),
    }
    selected_arms = args.arms or ["scalar_balanced", "rgb_uniform", "rgb_balanced"]
    unknown = sorted(set(selected_arms) - set(available))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    configurations = [(name, *available[name]) for name in selected_arms]
    predictions = {}
    for name, use_image, balanced, loss_mode, selection_bias_weight in configurations:
        runs, meta = [], []
        for seed in range(args.seeds):
            predicted, details = fit(data, use_image=use_image, balanced=balanced, seed=seed,
                                     epochs=args.epochs, patience=args.patience,
                                     batch=args.batch, device=device, output=args.output, name=name,
                                     loss_mode=loss_mode,
                                     selection_bias_weight=selection_bias_weight)
            runs.append(predicted)
            meta.append(details)
        stack = np.stack(runs)
        ensemble, aggregation = choose_seed_aggregation(
            data, stack, bias_weight=selection_bias_weight)
        predictions[name] = stack
        results["arms"][name] = {
            "seeds": meta, "validation": score_prediction(data, ensemble, "validation"),
            "development": score_prediction(data, ensemble, "development"),
            "aggregation": aggregation,
        }
        if use_image:
            results["arms"][name]["gates"] = choose_simple_gates(data, stack, ensemble)

    np.savez_compressed(args.output / "predictions.npz", eligible=eligible,
                        target_ray=data["target_ray"], basis=data["basis"],
                        split=data["split"], camera=data["camera"], severity=data["severity"],
                        **predictions)
    results["protocol"] = {
        "target": "commanded robot-centre world XY minus raw YOLO floor back-projection",
        "output": "world-frame robot-centre XY", "internal_parameterization": "ray residual metres",
        "predicts_R": False, "training_balance": "inverse-sqrt camera x severity sampling",
        "selection": "half natural world RMS plus half camera x severity macro RMS",
        "mean_arm_selection": "base selection plus 1.0 x supported camera x severity macro mean magnitude",
        "fixed_refusal": "detected box touches image edge or ground-centre projection outside image",
        "simple_gates": "validation-selected only; >=95% all, >=85% partial, >=80% severe retention",
    }
    results["inputs"] = {
        str(PIXELS.relative_to(REPO)): sha256(PIXELS),
        str((CAPTURE / "capture_manifest.json").relative_to(REPO)): sha256(CAPTURE / "capture_manifest.json"),
        str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
    }
    for capture_path in (args.augmentation_capture, args.transfer_capture):
        if capture_path:
            manifest_path = capture_path.expanduser().resolve() / "capture_manifest.json"
            results["inputs"][str(manifest_path.relative_to(REPO))] = sha256(manifest_path)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
