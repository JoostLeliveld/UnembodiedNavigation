#!/usr/bin/env python3
"""Fit raw-box mean-correction candidates on static commissioning-fit data.

This stage fits only the conditional mean.  It deliberately does not estimate R,
does not open a validation/audit role, and does not use detector-training images.
Every reported fit residual is spatially out of fold by 3.2 m capture block.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    along = raw_xy - camera_xy
    norm = float(np.linalg.norm(along))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("invalid camera ray")
    along /= norm
    return np.column_stack((along, np.asarray([-along[1], along[0]])))


def summary(error: np.ndarray) -> dict:
    norm = np.linalg.norm(error, axis=1)
    return {
        "n": int(len(norm)),
        "median_m": float(np.median(norm)),
        "rms_m": float(np.sqrt(np.mean(norm ** 2))),
        "p90_m": float(np.quantile(norm, 0.90)),
        "p95_m": float(np.quantile(norm, 0.95)),
        "above_0_25_m": int(np.sum(norm > 0.25)),
        "signed_bias_xy_m": np.mean(error, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inference = args.inference.resolve()
    gate_path = args.gate.resolve()
    capture = args.capture.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    inference_manifest_path = inference / "manifest.json"
    records_path = inference / "records.jsonl"
    capture_manifest_path = capture / "capture_manifest.json"
    capture_index_path = capture / "capture_index.csv"
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if inference_manifest.get("status") != "complete":
        raise RuntimeError("static detector inference is incomplete")
    if inference_manifest.get("authorized_role") != "commissioning_fit":
        raise RuntimeError("static source is not the authorized commissioning-fit role")
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("static capture is incomplete")
    if capture_manifest.get("capture_index_sha256") != sha256(capture_index_path):
        raise RuntimeError("static capture index hash mismatch")

    with capture_index_path.open(newline="", encoding="utf-8") as handle:
        capture_rows = [
            row for row in csv.DictReader(handle)
            if row.get("dataset_split") == "commissioning_fit"
        ]
    authorized_hashes = {row["image_sha1"] for row in capture_rows}
    if len(capture_rows) != 9600:
        raise RuntimeError(f"expected 9600 commissioning-fit opportunities, got {len(capture_rows)}")

    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if gate.get("gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("this fit requires the raw-box commissioning gate v2")
    width_px = float(gate["image_width_px"])
    height_px = float(gate["image_height_px"])
    edge = float(gate["min_edge_distance_px"])

    cameras = {
        item["camera_id"]: np.asarray(item["pose_xyz_rpy"][:2], dtype=float)
        for item in capture_manifest["cameras"]
    }
    if set(cameras) != set(CAMERAS):
        raise RuntimeError("capture does not contain the five frozen cameras")

    opportunities = 0
    admitted = []
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            opportunities += 1
            if row.get("image_sha1") not in authorized_hashes:
                raise RuntimeError("inference record is outside commissioning_fit")
            box = row.get("best_box_xyxy")
            raw = row.get("raw_ground_xy")
            if box is None or raw is None:
                continue
            box = np.asarray(box, dtype=float)
            raw = np.asarray(raw, dtype=float)
            if box.shape != (4,) or raw.shape != (2,) or not np.isfinite(box).all() or not np.isfinite(raw).all():
                continue
            box_w, box_h = float(box[2] - box[0]), float(box[3] - box[1])
            confidence = float(row["raw_best_confidence"])
            passes = (
                confidence >= float(gate["confidence_threshold"])
                and box_w >= float(gate["min_bbox_width_px"])
                and box_h >= float(gate["min_bbox_height_px"])
                and box[0] >= edge and box[1] >= edge
                and box[2] <= width_px - edge and box[3] <= height_px - edge
            )
            if not passes:
                continue
            camera = str(row["camera_id"])
            truth = np.asarray([row["robot_x"], row["robot_y"]], dtype=float)
            basis = ray_basis(cameras[camera], raw)
            ray = raw - cameras[camera]
            bearing = math.atan2(ray[1], ray[0])
            feature = [
                float(np.linalg.norm(ray)), 1.0 / max(float(np.linalg.norm(ray)), 1e-6),
                math.cos(bearing), math.sin(bearing),
                box_w / width_px, box_h / height_px, box_w / max(box_h, 1e-6),
                0.5 * float(box[0] + box[2]) / width_px, float(box[3]) / height_px,
                confidence, *[float(camera == candidate) for candidate in CAMERAS],
            ]
            admitted.append({
                "pose_id": int(row["pose_id"]), "position_id": int(row["position_id"]),
                "block_id": str(row["block_id"]), "heading_id": int(row["heading_id"]),
                "camera": camera, "image_sha1": str(row["image_sha1"]),
                "raw": raw, "truth": truth, "basis": basis,
                "target_ray": basis.T @ (truth - raw), "feature": feature,
            })
    if opportunities != 9600:
        raise RuntimeError(f"expected 9600 inference opportunities, got {opportunities}")
    if not admitted:
        raise RuntimeError("raw-box gate admitted no static observations")

    blocks = sorted({row["block_id"] for row in admitted})
    fold_by_block = {block: index % 5 for index, block in enumerate(blocks)}
    fold = np.asarray([fold_by_block[row["block_id"]] for row in admitted], dtype=int)
    camera_id = np.asarray([row["camera"] for row in admitted])
    feature = np.asarray([row["feature"] for row in admitted], dtype=float)
    target = np.asarray([row["target_ray"] for row in admitted], dtype=float)
    basis = np.asarray([row["basis"] for row in admitted], dtype=float)

    names = ("C0_raw", "C1_camera_constant", "C2_ridge_geometry", "C3_box_mlp")
    oof = {name: np.full_like(target, np.nan) for name in names}
    for held_out in range(5):
        train = np.flatnonzero(fold != held_out)
        test = np.flatnonzero(fold == held_out)
        oof["C0_raw"][test] = 0.0
        for camera in CAMERAS:
            source = train[camera_id[train] == camera]
            value = target[source].mean(axis=0) if len(source) else target[train].mean(axis=0)
            oof["C1_camera_constant"][test[camera_id[test] == camera]] = value
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(feature[train], target[train])
        oof["C2_ridge_geometry"][test] = ridge.predict(feature[test])
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                max_iter=200, early_stopping=False, random_state=20260914 + held_out,
            ),
        ).fit(feature[train], target[train])
        oof["C3_box_mlp"][test] = mlp.predict(feature[test])

    reports = {}
    for name in names:
        if not np.isfinite(oof[name]).all():
            raise RuntimeError(f"{name} has missing out-of-fold predictions")
        residual_ray = oof[name] - target
        residual_world = np.einsum("nij,nj->ni", basis, residual_ray)
        reports[name] = {
            "pooled": summary(residual_world),
            "by_camera": {
                camera: summary(residual_world[camera_id == camera]) for camera in CAMERAS
            },
            "folds": {
                str(value): summary(residual_world[fold == value]) for value in range(5)
            },
        }

    deployment = {
        "C0_raw": None,
        "C1_camera_constant": {
            camera: target[camera_id == camera].mean(axis=0) for camera in CAMERAS
        },
        "C2_ridge_geometry": make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(feature, target),
        "C3_box_mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                max_iter=200, early_stopping=False, random_state=20260914,
            ),
        ).fit(feature, target),
    }
    joblib.dump(deployment, output / "mean_candidates.joblib")

    with (output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "pose_id", "position_id", "block_id", "fold", "heading_id", "camera_id",
            "image_sha1", "truth_x", "truth_y", "raw_x", "raw_y",
            *[f"{name}_{axis}" for name in names for axis in ("along_m", "across_m")],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(admitted):
            record = {
                "pose_id": row["pose_id"], "position_id": row["position_id"],
                "block_id": row["block_id"], "fold": int(fold[index]),
                "heading_id": row["heading_id"], "camera_id": row["camera"],
                "image_sha1": row["image_sha1"], "truth_x": row["truth"][0],
                "truth_y": row["truth"][1], "raw_x": row["raw"][0], "raw_y": row["raw"][1],
            }
            for name in names:
                record[f"{name}_along_m"] = float(oof[name][index, 0])
                record[f"{name}_across_m"] = float(oof[name][index, 1])
            writer.writerow(record)

    manifest = {
        "schema": "static_commissioning_mean_candidates.v1",
        "status": "fit_complete_selection_pending_dynamic_validation",
        "scope": "static commissioning_fit only; mean correction only; no R fitted",
        "opportunities": opportunities,
        "admitted": len(admitted),
        "positions": len({row["position_id"] for row in admitted}),
        "spatial_blocks": len(blocks),
        "folds": 5,
        "gate_id": gate["gate_id"],
        "audit_opened": False,
        "candidate_reports": reports,
        "selection": "not selected; compare on independent dynamic development routes",
        "inputs": {
            "inference_manifest": str(inference_manifest_path),
            "inference_manifest_sha256": sha256(inference_manifest_path),
            "records": str(records_path), "records_sha256": sha256(records_path),
            "capture_manifest": str(capture_manifest_path),
            "capture_manifest_sha256": sha256(capture_manifest_path),
            "capture_index_sha256": sha256(capture_index_path),
            "gate": str(gate_path), "gate_sha256": sha256(gate_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "outputs": {
            "models": "mean_candidates.joblib",
            "oof_predictions": "oof_predictions.csv",
        },
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({
        "status": manifest["status"], "admitted": len(admitted),
        "positions": manifest["positions"], "spatial_blocks": len(blocks),
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
