#!/usr/bin/env python3
"""Train perception-only world-position models on static commissioning data.

Both models start from the frozen YOLO box-bottom ground projection and predict its
correction to the reference robot-centre position in the camera-ray frame.  Their
public mean output is world XY.  The RGB model additionally predicts a positive-
definite observation covariance in the same ray frame.

No belief, prior robot state, heading, expected box, silhouette, map visibility, or
line-of-sight quantity is read by this program.  The final-audit role is never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import yaml


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FEATURE_NAMES = (
    "raw_x_m", "raw_y_m", "raw_range_m", "inverse_raw_range",
    "ray_bearing_cos", "ray_bearing_sin",
    "bbox_width_fraction", "bbox_height_fraction", "bbox_aspect",
    "bbox_bottom_u_fraction", "bbox_bottom_v_fraction", "confidence",
    *(f"is_{camera}" for camera in CAMERAS),
)
IMAGE_SIZE = 96
CONTEXT_FRACTION = 0.5
CHI2_95 = 5.99146454711


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    along = np.asarray(raw_xy, dtype=float) - np.asarray(camera_xy, dtype=float)
    length = float(np.linalg.norm(along))
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError("invalid camera-to-reading ray")
    along /= length
    return np.column_stack((along, np.asarray([-along[1], along[0]], dtype=float)))


def context_crop(image_path: Path, box: np.ndarray) -> np.ndarray:
    """Return RGB plus a valid-pixel channel for a two-box-width context crop."""
    x0, y0, x1, y1 = map(float, box)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    requested = (
        int(math.floor(x0 - CONTEXT_FRACTION * width)),
        int(math.floor(y0 - CONTEXT_FRACTION * height)),
        int(math.ceil(x1 + CONTEXT_FRACTION * width)),
        int(math.ceil(y1 + CONTEXT_FRACTION * height)),
    )
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        valid = Image.new("L", rgb.size, color=255)
        rgb_crop = rgb.crop(requested).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        valid_crop = valid.crop(requested).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    value = np.asarray(rgb_crop, dtype=np.uint8).transpose(2, 0, 1)
    mask = np.asarray(valid_crop, dtype=np.uint8)[None]
    return np.concatenate((value, mask), axis=0)


def load_dataset(inference: Path, gate_path: Path, capture: Path) -> dict[str, np.ndarray]:
    inference_manifest_path = inference / "manifest.json"
    records_path = inference / "records.jsonl"
    capture_manifest_path = capture / "capture_manifest.json"
    capture_index_path = capture / "capture_index.csv"
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if inference_manifest.get("status") != "complete":
        raise RuntimeError("static detector inference is incomplete")
    if inference_manifest.get("authorized_role") != "commissioning_fit":
        raise RuntimeError("inference is not restricted to commissioning_fit")
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("static capture is incomplete")
    if capture_manifest.get("capture_index_sha256") != sha256(capture_index_path):
        raise RuntimeError("capture-index hash mismatch")

    with capture_index_path.open(newline="", encoding="utf-8") as handle:
        capture_rows = [
            row for row in csv.DictReader(handle)
            if row.get("dataset_split") == "commissioning_fit"
        ]
    if len(capture_rows) != 9600:
        raise RuntimeError(f"expected 9,600 commissioning-fit opportunities, got {len(capture_rows)}")
    authorized = {row["image_sha1"] for row in capture_rows}

    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if gate.get("gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("world-position training requires commissioning_sensor_gate_v2")
    image_width = float(gate["image_width_px"])
    image_height = float(gate["image_height_px"])
    edge = float(gate["min_edge_distance_px"])
    camera_xy = {
        item["camera_id"]: np.asarray(item["pose_xyz_rpy"][:2], dtype=float)
        for item in capture_manifest["cameras"]
    }
    if set(camera_xy) != set(CAMERAS):
        raise RuntimeError("capture does not contain the five expected cameras")

    rows: list[dict[str, object]] = []
    opportunities = 0
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            opportunities += 1
            if source.get("image_sha1") not in authorized:
                raise RuntimeError("inference record lies outside commissioning_fit")
            box_value = source.get("best_box_xyxy")
            raw_value = source.get("raw_ground_xy")
            if box_value is None or raw_value is None:
                continue
            box = np.asarray(box_value, dtype=float)
            raw = np.asarray(raw_value, dtype=float)
            if box.shape != (4,) or raw.shape != (2,):
                continue
            if not np.isfinite(box).all() or not np.isfinite(raw).all():
                continue
            box_width = float(box[2] - box[0])
            box_height = float(box[3] - box[1])
            confidence = float(source["raw_best_confidence"])
            admitted = (
                confidence >= float(gate["confidence_threshold"])
                and box_width >= float(gate["min_bbox_width_px"])
                and box_height >= float(gate["min_bbox_height_px"])
                and box[0] >= edge and box[1] >= edge
                and box[2] <= image_width - edge and box[3] <= image_height - edge
            )
            if not admitted:
                continue
            camera = str(source["camera_id"])
            basis = ray_basis(camera_xy[camera], raw)
            ray = raw - camera_xy[camera]
            distance = float(np.linalg.norm(ray))
            bearing = math.atan2(ray[1], ray[0])
            feature = np.asarray([
                raw[0], raw[1], distance, 1.0 / max(distance, 1e-6),
                math.cos(bearing), math.sin(bearing),
                box_width / image_width, box_height / image_height,
                box_width / max(box_height, 1e-6),
                0.5 * float(box[0] + box[2]) / image_width,
                float(box[3]) / image_height, confidence,
                *[float(camera == candidate) for candidate in CAMERAS],
            ], dtype=np.float32)
            truth = np.asarray([source["robot_x"], source["robot_y"]], dtype=np.float32)
            image_path = capture / str(source["image"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            rows.append({
                "pose_id": int(source["pose_id"]),
                "position_id": int(source["position_id"]),
                "block_id": str(source["block_id"]),
                "heading_id": int(source["heading_id"]),
                "camera": camera,
                "image_sha1": str(source["image_sha1"]),
                "feature": feature,
                "image": context_crop(image_path, box),
                "raw": raw.astype(np.float32),
                "truth": truth,
                "basis": basis.astype(np.float32),
                "target": (basis.T @ (truth - raw)).astype(np.float32),
            })
    if opportunities != 9600:
        raise RuntimeError(f"expected 9,600 inference opportunities, got {opportunities}")
    if not rows:
        raise RuntimeError("sensor gate admitted no observations")

    blocks = sorted({str(row["block_id"]) for row in rows})
    fold_by_block = {block: index % 5 for index, block in enumerate(blocks)}
    return {
        "feature": np.stack([row["feature"] for row in rows]),
        "image": np.stack([row["image"] for row in rows]),
        "raw": np.stack([row["raw"] for row in rows]),
        "truth": np.stack([row["truth"] for row in rows]),
        "basis": np.stack([row["basis"] for row in rows]),
        "target": np.stack([row["target"] for row in rows]),
        "camera": np.asarray([row["camera"] for row in rows]),
        "pose_id": np.asarray([row["pose_id"] for row in rows], dtype=int),
        "position_id": np.asarray([row["position_id"] for row in rows], dtype=int),
        "heading_id": np.asarray([row["heading_id"] for row in rows], dtype=int),
        "block_id": np.asarray([row["block_id"] for row in rows]),
        "fold": np.asarray([fold_by_block[str(row["block_id"])] for row in rows], dtype=int),
        "image_sha1": np.asarray([row["image_sha1"] for row in rows]),
        "opportunities": np.asarray([opportunities], dtype=int),
    }


class BoxSpatialMLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class RGBGaussianNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 24, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 72, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(72, 96, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.shared = nn.Sequential(
            nn.Linear(feature_dim + 96, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, 2)
        self.cholesky_head = nn.Linear(64, 3)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.cholesky_head.weight)
        initial = math.log(math.expm1(0.15))
        with torch.no_grad():
            self.cholesky_head.bias[:] = torch.tensor([initial, 0.0, initial])

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(torch.cat((self.encoder(image), feature), dim=1))
        mean = self.mean_head(hidden)
        raw = self.cholesky_head(hidden)
        diagonal_1 = F.softplus(raw[:, 0]) + 1e-3
        diagonal_2 = F.softplus(raw[:, 2]) + 1e-3
        zeros = torch.zeros_like(diagonal_1)
        first = torch.stack((diagonal_1, zeros), dim=1)
        second = torch.stack((raw[:, 1], diagonal_2), dim=1)
        return mean, torch.stack((first, second), dim=1)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    residual = (target - mean).unsqueeze(2)
    whitened = torch.linalg.solve_triangular(factor, residual, upper=False).squeeze(2)
    logdet = 2.0 * (
        torch.log(factor[:, 0, 0]) + torch.log(factor[:, 1, 1])
    )
    return 0.5 * (logdet + torch.sum(whitened * whitened, dim=1))


def fit_model(
    kind: str,
    data: dict[str, np.ndarray],
    indexes: np.ndarray,
    *,
    epochs: int,
    warmup_epochs: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_mean = data["feature"][indexes].mean(axis=0)
    feature_std = data["feature"][indexes].std(axis=0).clip(1e-3)
    normalized = ((data["feature"] - feature_mean) / feature_std).astype(np.float32)
    feature = torch.from_numpy(normalized)
    target = torch.from_numpy(data["target"].astype(np.float32))
    if kind == "box_spatial_mlp":
        model: nn.Module = BoxSpatialMLP(len(FEATURE_NAMES)).to(device)
    elif kind == "rgb_gaussian":
        model = RGBGaussianNet(len(FEATURE_NAMES)).to(device)
    else:
        raise ValueError(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    rng = np.random.default_rng(seed + 991)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(indexes)
        for part in np.array_split(order, max(1, math.ceil(len(order) / batch_size))):
            x = feature[part].to(device)
            y = target[part].to(device)
            if kind == "box_spatial_mlp":
                prediction = model(x)
                loss = F.mse_loss(prediction, y)
            else:
                image = torch.from_numpy(data["image"][part]).to(device=device, dtype=torch.float32)
                image[:, :3] /= 255.0
                image[:, 3:] /= 255.0
                gain = torch.empty((len(part), 1, 1, 1), device=device).uniform_(0.90, 1.10)
                offset = torch.empty((len(part), 1, 1, 1), device=device).uniform_(-0.03, 0.03)
                image[:, :3] = (image[:, :3] * gain + offset).clamp(0.0, 1.0)
                mean, factor = model(image, x)
                if epoch < warmup_epochs:
                    loss = F.mse_loss(mean, y)
                else:
                    loss = gaussian_nll(y, mean, factor).mean() + 0.10 * F.mse_loss(mean, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append(float(np.mean(losses)))
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            print(f"{kind} seed={seed} epoch={epoch + 1}/{epochs} loss={history[-1]:.6f}", flush=True)
    model.eval()
    return model, feature_mean, feature_std, history


def predict(
    kind: str,
    model: nn.Module,
    data: dict[str, np.ndarray],
    indexes: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    means = np.full((len(data["target"]), 2), np.nan, dtype=np.float32)
    covariance = (
        np.full((len(data["target"]), 2, 2), np.nan, dtype=np.float32)
        if kind == "rgb_gaussian" else None
    )
    normalized = ((data["feature"] - feature_mean) / feature_std).astype(np.float32)
    with torch.no_grad():
        for part in np.array_split(indexes, max(1, math.ceil(len(indexes) / batch_size))):
            feature = torch.from_numpy(normalized[part]).to(device)
            if kind == "box_spatial_mlp":
                means[part] = model(feature).cpu().numpy()
            else:
                image = torch.from_numpy(data["image"][part]).to(device=device, dtype=torch.float32)
                image /= 255.0
                mean, factor = model(image, feature)
                means[part] = mean.cpu().numpy()
                covariance[part] = (factor @ factor.transpose(1, 2)).cpu().numpy()
    return means, covariance


def mean_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    error = prediction - target
    norm = np.linalg.norm(error, axis=1)
    return {
        "n": int(len(norm)),
        "mean_cm": float(100.0 * np.mean(norm)),
        "median_cm": float(100.0 * np.median(norm)),
        "rms_cm": float(100.0 * np.sqrt(np.mean(norm ** 2))),
        "p90_cm": float(100.0 * np.quantile(norm, 0.90)),
        "p95_cm": float(100.0 * np.quantile(norm, 0.95)),
        "above_25cm": int(np.sum(norm > 0.25)),
        "signed_bias_ray_cm": (100.0 * np.mean(error, axis=0)).tolist(),
    }


def covariance_summary(prediction: np.ndarray, target: np.ndarray, covariance: np.ndarray) -> dict[str, object]:
    residual = target - prediction
    d2, nll, area = [], [], []
    for error, matrix in zip(residual, covariance, strict=True):
        inverse = np.linalg.inv(matrix)
        distance = float(error @ inverse @ error)
        d2.append(distance)
        nll.append(0.5 * (math.log(max(np.linalg.det(matrix), 1e-18)) + distance))
        area.append(math.pi * CHI2_95 * math.sqrt(max(np.linalg.det(matrix), 0.0)))
    return {
        "mean_nll_without_constant": float(np.mean(nll)),
        "mean_mahalanobis2": float(np.mean(d2)),
        "median_mahalanobis2": float(np.median(d2)),
        "containment_95": float(np.mean(np.asarray(d2) <= CHI2_95)),
        "median_95_ellipse_area_m2": float(np.median(area)),
    }


def checkpoint_payload(
    kind: str,
    model: nn.Module,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    return {
        "schema": "perception_world_position_model.v1",
        "model_kind": kind,
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "image_channels": ["red", "green", "blue", "valid_pixel_mask"] if kind == "rgb_gaussian" else [],
        "image_size": IMAGE_SIZE if kind == "rgb_gaussian" else None,
        "context_fraction": CONTEXT_FRACTION if kind == "rgb_gaussian" else None,
        "target": "reference world XY minus raw box-bottom projection, in camera-ray metres",
        "public_output": "corrected world XY",
        "predicts_covariance": kind == "rgb_gaussian",
        "epochs": epochs,
        "seed": seed,
        "state_dict": model.cpu().state_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-epochs", type=int, default=200)
    parser.add_argument("--rgb-epochs", type=int, default=100)
    parser.add_argument("--rgb-warmup-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    inference = args.inference.resolve()
    gate = args.gate.resolve()
    capture = args.capture.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    torch.set_num_threads(2)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"loading admitted crops on {device}", flush=True)
    data = load_dataset(inference, gate, capture)
    np.savez_compressed(
        staging / "dataset_contract.npz",
        feature=data["feature"], target=data["target"], raw=data["raw"], truth=data["truth"],
        basis=data["basis"], camera=data["camera"], pose_id=data["pose_id"],
        position_id=data["position_id"], heading_id=data["heading_id"],
        block_id=data["block_id"], fold=data["fold"], image_sha1=data["image_sha1"],
    )
    n = len(data["target"])
    outputs: dict[str, dict[str, np.ndarray | None]] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    specifications = {
        "box_spatial_mlp": (args.box_epochs, 0),
        "rgb_gaussian": (args.rgb_epochs, args.rgb_warmup_epochs),
    }
    for kind, (epochs, warmup) in specifications.items():
        oof_mean = np.full((n, 2), np.nan, dtype=np.float32)
        oof_covariance = np.full((n, 2, 2), np.nan, dtype=np.float32) if kind == "rgb_gaussian" else None
        histories[kind] = {}
        for fold in range(5):
            train = np.flatnonzero(data["fold"] != fold)
            test = np.flatnonzero(data["fold"] == fold)
            seed = 20260914 + 100 * fold + (1 if kind == "rgb_gaussian" else 0)
            model, feature_mean, feature_std, history = fit_model(
                kind, data, train, epochs=epochs, warmup_epochs=warmup,
                seed=seed, batch_size=args.batch_size, device=device,
            )
            prediction, covariance = predict(
                kind, model, data, test, feature_mean, feature_std,
                batch_size=args.batch_size, device=device,
            )
            oof_mean[test] = prediction[test]
            if oof_covariance is not None and covariance is not None:
                oof_covariance[test] = covariance[test]
            torch.save(
                checkpoint_payload(kind, model, feature_mean, feature_std, epochs=epochs, seed=seed),
                staging / f"{kind}_fold{fold}.pt",
            )
            histories[kind][str(fold)] = history
        if not np.isfinite(oof_mean).all():
            raise RuntimeError(f"{kind} has incomplete out-of-fold predictions")
        if oof_covariance is not None and not np.isfinite(oof_covariance).all():
            raise RuntimeError("rgb_gaussian has incomplete covariance predictions")
        outputs[kind] = {"mean": oof_mean, "covariance": oof_covariance}

        all_indexes = np.arange(n)
        deployment_seed = 20265914 + (1 if kind == "rgb_gaussian" else 0)
        model, feature_mean, feature_std, history = fit_model(
            kind, data, all_indexes, epochs=epochs, warmup_epochs=warmup,
            seed=deployment_seed, batch_size=args.batch_size, device=device,
        )
        torch.save(
            checkpoint_payload(
                kind, model, feature_mean, feature_std,
                epochs=epochs, seed=deployment_seed,
            ),
            staging / f"{kind}_deployment.pt",
        )
        histories[kind]["deployment"] = history

    reports: dict[str, object] = {}
    for kind, prediction in outputs.items():
        mean = prediction["mean"]
        assert mean is not None
        entry: dict[str, object] = {
            "pooled": mean_summary(mean, data["target"]),
            "by_camera": {
                camera: mean_summary(mean[data["camera"] == camera], data["target"][data["camera"] == camera])
                for camera in CAMERAS
            },
            "by_fold": {
                str(fold): mean_summary(mean[data["fold"] == fold], data["target"][data["fold"] == fold])
                for fold in range(5)
            },
        }
        covariance = prediction["covariance"]
        if covariance is not None:
            entry["covariance"] = covariance_summary(mean, data["target"], covariance)
        reports[kind] = entry

    np.savez_compressed(
        staging / "oof_predictions.npz",
        target_ray=data["target"], basis=data["basis"], raw_world=data["raw"],
        truth_world=data["truth"], fold=data["fold"], camera=data["camera"],
        box_spatial_mean=outputs["box_spatial_mlp"]["mean"],
        rgb_gaussian_mean=outputs["rgb_gaussian"]["mean"],
        rgb_gaussian_covariance=outputs["rgb_gaussian"]["covariance"],
    )
    (staging / "training_history.json").write_text(
        json.dumps(histories, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "schema": "static_perception_world_models.v1",
        "status": "static_spatial_oof_complete_dynamic_validation_pending",
        "device": str(device),
        "population": {
            "role": "commissioning_fit",
            "opportunities": int(data["opportunities"][0]),
            "admitted": n,
            "positions": int(len(set(data["position_id"].tolist()))),
            "spatial_blocks": int(len(set(data["block_id"].tolist()))),
            "folds": 5,
            "final_audit_opened": False,
        },
        "sensor_gate": "commissioning_sensor_gate_v2",
        "features": list(FEATURE_NAMES),
        "image_input": {
            "shape": [4, IMAGE_SIZE, IMAGE_SIZE],
            "channels": ["red", "green", "blue", "valid_pixel_mask"],
            "context_fraction_each_side": CONTEXT_FRACTION,
        },
        "target": "reference robot-centre world XY minus raw box-bottom projection, ray frame metres",
        "public_output": "corrected world XY",
        "forbidden_inputs_absent": [
            "belief", "prior_robot_position", "prior_robot_heading", "predicted_box",
            "expected_box", "robot_hull", "semantic_mask", "line_of_sight",
            "obstacle_height", "route_identity", "innovation", "NIS",
        ],
        "training": {
            "box_epochs": args.box_epochs,
            "rgb_epochs": args.rgb_epochs,
            "rgb_mean_warmup_epochs": args.rgb_warmup_epochs,
            "batch_size": args.batch_size,
            "spatial_resampling": "five complete 3.2 m block folds; fixed epochs; held-out fold never trains",
            "deployment_refit": "each architecture refitted on all admitted commissioning_fit observations",
        },
        "reports": reports,
        "inputs": {
            "inference_manifest": sha256(inference / "manifest.json"),
            "inference_records": sha256(inference / "records.jsonl"),
            "capture_manifest": sha256(capture / "capture_manifest.json"),
            "capture_index": sha256(capture / "capture_index.csv"),
            "gate": sha256(gate),
            "implementation": sha256(Path(__file__)),
        },
    }
    result_path = staging / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {
        path.name: sha256(path) for path in staging.iterdir() if path.is_file()
    }
    (staging / "manifest.json").write_text(
        json.dumps({
            "schema": "static_perception_world_models_manifest.v1",
            "status": "complete",
            "results_sha256": sha256(result_path),
            "artifacts": artifacts,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    print(json.dumps({"status": result["status"], "reports": reports}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
