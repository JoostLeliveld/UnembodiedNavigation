#!/usr/bin/env python3
"""Fit and select the frozen Stage-08 camera-availability model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/shared"))
import metrics as M  # noqa: E402

from reliability.projection import camera_model_from_world
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world, segment_occluded


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CLIP = (0.001, 0.999)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            rows.append({
                **source,
                "pose_id": int(source["pose_id"]),
                "position_id": int(source["position_id"]),
                "fold": int(source["fold"]),
                "heading_id": int(source["heading_id"]),
                "robot_x": float(source["robot_x"]),
                "robot_y": float(source["robot_y"]),
                "robot_yaw": float(source["robot_yaw"]),
                "available": int(source["admitted"]),
                "expected_box": tuple(float(source[key]) for key in (
                    "expected_box_x0", "expected_box_y0", "expected_box_x1", "expected_box_y1"
                )),
            })
    if len(rows) != 9600:
        raise RuntimeError(f"expected 9,600 availability opportunities, found {len(rows)}")
    return rows


def build_geometry_features(rows: list[dict], cameras: dict, scene) -> np.ndarray:
    output = []
    for row in rows:
        camera = cameras[row["camera_id"]]
        box = row["expected_box"]
        width, height = box[2] - box[0], box[3] - box[1]
        inclusion = float(
            box[0] >= 0.0 and box[1] >= 0.0
            and box[2] < camera.img_width and box[3] < camera.img_height
        )
        target = np.asarray((row["robot_x"], row["robot_y"], 0.20), dtype=float)
        line_of_sight = float(not segment_occluded(scene.prisms, camera.cam_pos, target))
        distance = math.hypot(
            row["robot_x"] - float(camera.cam_pos[0]),
            row["robot_y"] - float(camera.cam_pos[1]),
        )
        output.append((distance, width, height, inclusion, line_of_sight))
    result = np.asarray(output, dtype=float)
    if result.shape != (len(rows), 5) or not np.isfinite(result).all():
        raise RuntimeError("invalid Q1 geometry features")
    return result


def clipped(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), CLIP[0], CLIP[1])


def fit_logistic(x: np.ndarray, y: np.ndarray, *, c: float) -> tuple[dict, object | None]:
    if len(np.unique(y)) < 2:
        probability = float(np.clip(np.mean(y), *CLIP))
        return {"kind": "constant", "probability": probability}, None
    model = LogisticRegression(
        penalty="l2", C=float(c), solver="lbfgs", tol=1e-9, max_iter=2000,
        random_state=20260910,
    ).fit(x, y)
    return {"kind": "logistic"}, model


def predict_model(description: dict, model, x: np.ndarray) -> np.ndarray:
    if description["kind"] == "constant":
        return np.full(len(x), description["probability"], dtype=float)
    return clipped(model.predict_proba(x)[:, 1])


def q0_fold(rows, y, cameras, train, test):
    prediction = np.empty(len(test), dtype=float)
    parameters = {}
    camera_ids = np.asarray([row["camera_id"] for row in rows])
    for camera in CAMERAS:
        rate = float(np.mean(y[train][camera_ids[train] == camera]))
        rate = float(np.clip(rate, *CLIP))
        prediction[camera_ids[test] == camera] = rate
        parameters[camera] = rate
    return prediction, parameters


def q1_fold(rows, y, geometry, train, test):
    prediction = np.empty(len(test), dtype=float)
    parameters = {}
    camera_ids = np.asarray([row["camera_id"] for row in rows])
    for camera in CAMERAS:
        train_camera = train[camera_ids[train] == camera]
        test_camera_local = np.flatnonzero(camera_ids[test] == camera)
        mean = np.mean(geometry[train_camera, :3], axis=0)
        scale = np.std(geometry[train_camera, :3], axis=0)
        scale[scale < 1e-12] = 1.0
        x_train = np.column_stack((
            (geometry[train_camera, :3] - mean) / scale,
            geometry[train_camera, 3:],
        ))
        x_test = np.column_stack((
            (geometry[test[test_camera_local], :3] - mean) / scale,
            geometry[test[test_camera_local], 3:],
        ))
        description, model = fit_logistic(x_train, y[train_camera], c=1.0)
        prediction[test_camera_local] = predict_model(description, model, x_test)
        parameters[camera] = {
            **description,
            "continuous_mean": mean.tolist(),
            "continuous_scale": scale.tolist(),
            **({} if model is None else {
                "coefficient": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
            }),
        }
    return prediction, parameters


def rbf_matrix(xy: np.ndarray, centres: np.ndarray, length_scale: float) -> np.ndarray:
    squared = np.sum((xy[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return np.exp(-squared / (2.0 * float(length_scale) ** 2))


def q2_fold(rows, y, train, test, *, length_scale: float, c: float):
    prediction = np.empty(len(test), dtype=float)
    parameters = {}
    camera_ids = np.asarray([row["camera_id"] for row in rows])
    xy = np.asarray([[row["robot_x"], row["robot_y"]] for row in rows])
    yaw = np.asarray([row["robot_yaw"] for row in rows])
    position_ids = np.asarray([row["position_id"] for row in rows])
    for camera in CAMERAS:
        train_camera = train[camera_ids[train] == camera]
        test_camera_local = np.flatnonzero(camera_ids[test] == camera)
        centre_indexes = []
        for position_id in sorted(set(position_ids[train_camera].tolist())):
            centre_indexes.append(train_camera[np.flatnonzero(position_ids[train_camera] == position_id)[0]])
        centres = xy[np.asarray(centre_indexes, dtype=int)]
        x_train = np.column_stack((
            rbf_matrix(xy[train_camera], centres, length_scale),
            np.sin(yaw[train_camera]), np.cos(yaw[train_camera]),
        ))
        test_global = test[test_camera_local]
        x_test = np.column_stack((
            rbf_matrix(xy[test_global], centres, length_scale),
            np.sin(yaw[test_global]), np.cos(yaw[test_global]),
        ))
        description, model = fit_logistic(x_train, y[train_camera], c=c)
        prediction[test_camera_local] = predict_model(description, model, x_test)
        parameters[camera] = {
            **description,
            "centres_xy_m": centres.tolist(),
            **({} if model is None else {
                "coefficient": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
            }),
        }
    return prediction, parameters


def cross_validate(rows, y, geometry, family: str, *, length_scale=1.6, c=1.0):
    folds = np.asarray([row["fold"] for row in rows])
    prediction = np.empty(len(rows), dtype=float)
    fold_parameters = {}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        if family == "Q0_camera_constant":
            values, parameters = q0_fold(rows, y, CAMERAS, train, test)
        elif family == "Q1_geometry_logistic":
            values, parameters = q1_fold(rows, y, geometry, train, test)
        elif family == "Q2_rbf_heading_logistic":
            values, parameters = q2_fold(
                rows, y, train, test, length_scale=length_scale, c=c
            )
        else:
            raise ValueError(f"unknown family {family}")
        prediction[test] = values
        fold_parameters[str(fold)] = parameters
    return clipped(prediction), fold_parameters


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    logits = M.logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2 or np.std(logits) < 1e-12:
        return {"intercept": math.nan, "slope": math.nan}
    model = LogisticRegression(
        penalty=None, solver="lbfgs", tol=1e-9, max_iter=2000,
    ).fit(logits, y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def score(rows, y: np.ndarray, p: np.ndarray) -> dict:
    blocks = np.asarray([row["block_id"] for row in rows])
    cameras = np.asarray([row["camera_id"] for row in rows])
    folds = np.asarray([row["fold"] for row in rows])
    block_values = []
    for block in sorted(set(blocks.tolist())):
        selector = blocks == block
        block_values.append(M.brier(y[selector], p[selector]))
    return {
        "pooled_brier": M.brier(y, p),
        "block_macro_brier": float(np.mean(block_values)),
        "pooled_log_loss": M.logloss(y, p, eps=CLIP[0]),
        "pooled_ece_10": M.ece(y, p, bins=10),
        "pooled_auprc": M.auprc(y, p),
        "calibration": calibration(y, p),
        "by_fold": {
            str(fold): {
                "opportunities": int(np.sum(folds == fold)),
                "usable": int(np.sum(y[folds == fold])),
                "brier": M.brier(y[folds == fold], p[folds == fold]),
                "log_loss": M.logloss(y[folds == fold], p[folds == fold], eps=CLIP[0]),
                "ece_10": M.ece(y[folds == fold], p[folds == fold], bins=10),
            }
            for fold in range(5)
        },
        "by_camera": {
            camera: {
                "opportunities": int(np.sum(cameras == camera)),
                "usable": int(np.sum(y[cameras == camera])),
                "brier": M.brier(y[cameras == camera], p[cameras == camera]),
                "log_loss": M.logloss(y[cameras == camera], p[cameras == camera], eps=CLIP[0]),
                "ece_10": M.ece(y[cameras == camera], p[cameras == camera], bins=10),
                "auprc": M.auprc(y[cameras == camera], p[cameras == camera]),
            }
            for camera in CAMERAS
        },
    }


def maximin_positions(rows, indexes: np.ndarray) -> list[int]:
    positions = {}
    for index in indexes:
        row = rows[int(index)]
        positions[row["position_id"]] = np.asarray((row["robot_x"], row["robot_y"]), dtype=float)
    ids = sorted(positions)
    centroid = np.mean(np.asarray([positions[value] for value in ids]), axis=0)
    first = min(ids, key=lambda value: (-float(np.linalg.norm(positions[value] - centroid)), value))
    selected = [first]
    remaining = set(ids) - {first}
    while remaining:
        candidate = min(
            remaining,
            key=lambda value: (
                -min(float(np.linalg.norm(positions[value] - positions[taken])) for taken in selected),
                value,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def deployment_fit(rows, y, geometry, selected, q2_parameters):
    all_indexes = np.arange(len(rows))
    if selected == "Q0_camera_constant":
        _prediction, parameters = q0_fold(rows, y, CAMERAS, all_indexes, all_indexes)
    elif selected == "Q1_geometry_logistic":
        _prediction, parameters = q1_fold(rows, y, geometry, all_indexes, all_indexes)
    else:
        _prediction, parameters = q2_fold(
            rows, y, all_indexes, all_indexes,
            length_scale=q2_parameters["length_scale_m"], c=q2_parameters["C"],
        )
    return parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage06", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_stage06_selection_result_was_read":
        raise RuntimeError("Stage-08 protocol is not frozen")
    pipeline = json.loads((REPO / "experiments/thesis_pipeline_lock/pipeline_lock.json").read_text())
    stage06 = next(stage for stage in pipeline["stages"] if stage["id"] == "06_detector_gate")
    if stage06["status"] != "locked":
        raise RuntimeError("Stage 06 is not locked")
    source = args.stage06.resolve() / "admission_records.csv"
    if sha256(source) != "4599e7c4a529a4560cc54e362cbe27a34eb8872a386c5513c1c05c9c4f04a2a8":
        raise RuntimeError("Stage-06 opportunities differ from the locked population")
    rows = load_rows(source)
    y = np.asarray([row["available"] for row in rows], dtype=int)
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    include_names = dict(zip(CAMERAS, (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )))
    cameras = {
        camera: camera_model_from_world(world, include_name=include)
        for camera, include in include_names.items()
    }
    scene = parse_occlusion_scene_from_world(
        str(world), model_name="warehouse_v2_occluders", geometry_tags=("collision",)
    )
    geometry = build_geometry_features(rows, cameras, scene)

    predictions = {}
    fold_parameters = {}
    predictions["Q0_camera_constant"], fold_parameters["Q0_camera_constant"] = cross_validate(
        rows, y, geometry, "Q0_camera_constant"
    )
    predictions["Q1_geometry_logistic"], fold_parameters["Q1_geometry_logistic"] = cross_validate(
        rows, y, geometry, "Q1_geometry_logistic"
    )
    grid = []
    q2_predictions = {}
    for length_scale in protocol["q2_grid"]["rbf_length_scale_m"]:
        for c in protocol["q2_grid"]["inverse_regularization_C"]:
            key = f"ls{length_scale:g}_C{c:g}"
            values, _parameters = cross_validate(
                rows, y, geometry, "Q2_rbf_heading_logistic",
                length_scale=float(length_scale), c=float(c),
            )
            result = score(rows, y, values)
            q2_predictions[key] = values
            grid.append({
                "key": key,
                "length_scale_m": float(length_scale),
                "C": float(c),
                "block_macro_brier": result["block_macro_brier"],
                "pooled_brier": result["pooled_brier"],
                "pooled_ece_10": result["pooled_ece_10"],
            })
    best_q2 = min(grid, key=lambda value: (value["block_macro_brier"], value["pooled_brier"], value["key"]))
    predictions["Q2_rbf_heading_logistic"] = q2_predictions[best_q2["key"]]
    _, fold_parameters["Q2_rbf_heading_logistic"] = cross_validate(
        rows, y, geometry, "Q2_rbf_heading_logistic",
        length_scale=best_q2["length_scale_m"], c=best_q2["C"],
    )

    reports = {name: score(rows, y, values) for name, values in predictions.items()}
    baseline = reports["Q0_camera_constant"]
    eligible = {}
    for name, report in reports.items():
        camera_gate = all(
            report["by_camera"][camera]["brier"]
            <= baseline["by_camera"][camera]["brier"] + 1e-12
            for camera in CAMERAS
            if report["by_camera"][camera]["opportunities"] >= 100
        )
        eligible[name] = bool(report["pooled_ece_10"] <= 0.10 and camera_gate)
    candidates = [name for name in predictions if eligible[name]]
    if not candidates:
        raise RuntimeError("no availability candidate passed the frozen calibration gate")
    best = min(candidates, key=lambda name: reports[name]["block_macro_brier"])
    margin = 0.005
    order = ("Q0_camera_constant", "Q1_geometry_logistic", "Q2_rbf_heading_logistic")
    selected = next(
        name for name in order
        if eligible[name] and reports[name]["block_macro_brier"] <= reports[best]["block_macro_brier"] + margin
    )

    folds = np.asarray([row["fold"] for row in rows])
    position_ids = np.asarray([row["position_id"] for row in rows])
    size_curve = []
    for requested in protocol["commissioning_size_curve"]["position_counts"]:
        curve_prediction = np.empty(len(rows), dtype=float)
        effective = {}
        for fold in range(5):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            ordering = maximin_positions(rows, train)
            count = min(int(requested), len(ordering))
            chosen = set(ordering[:count])
            subset = train[np.asarray([position_ids[index] in chosen for index in train])]
            effective[str(fold)] = count
            if selected == "Q0_camera_constant":
                values, _ = q0_fold(rows, y, CAMERAS, subset, test)
            elif selected == "Q1_geometry_logistic":
                values, _ = q1_fold(rows, y, geometry, subset, test)
            else:
                values, _ = q2_fold(
                    rows, y, subset, test,
                    length_scale=best_q2["length_scale_m"], c=best_q2["C"],
                )
            curve_prediction[test] = values
        result = score(rows, y, clipped(curve_prediction))
        size_curve.append({
            "requested_positions": int(requested),
            "effective_training_positions_by_fold": effective,
            "block_macro_brier": result["block_macro_brier"],
            "pooled_brier": result["pooled_brier"],
            "pooled_ece_10": result["pooled_ece_10"],
            "pooled_log_loss": result["pooled_log_loss"],
        })

    deployment_parameters = deployment_fit(rows, y, geometry, selected, best_q2)
    artifact = {
        "schema": "thesis_commissioned_availability_model.v1",
        "pipeline_id": "THESIS-FINAL-PIPELINE-V1",
        "final_audit_accessed": False,
        "model": selected,
        "probability_clip": list(CLIP),
        "camera_order": list(CAMERAS),
        "feature_names": [
            "range_m", "projected_hull_width_px", "projected_hull_height_px",
            "full_image_inclusion", "static_line_of_sight",
        ],
        "image_size_px": [1280, 720],
        "line_of_sight": {
            "target_height_m": 0.20,
            "world_model": "warehouse_v2_occluders",
            "geometry_tags": ["collision"],
            "prism_count": len(scene.prisms),
        },
        "q2_hyperparameters": {
            "length_scale_m": best_q2["length_scale_m"], "C": best_q2["C"]
        },
        "parameters": deployment_parameters,
        "training_population": "all 9,600 Stage-06 commissioning opportunities",
        "target": "frozen detector-plus-admission chain supplies a usable measurement",
        "world_sha256": sha256(world),
    }

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"staging directory exists: {staging}")
    staging.mkdir(parents=True)
    report_path = staging / "availability_report.json"
    artifact_path = staging / "availability_model.json"
    prediction_path = staging / "oof_predictions.csv"
    curve_path = staging / "commissioning_size_curve.csv"
    report = {
        "schema": "thesis_stage08_availability_selection.v1",
        "status": "complete",
        "final_audit_accessed": False,
        "opportunities": len(rows),
        "usable": int(np.sum(y)),
        "positions": len(set(position_ids.tolist())),
        "spatial_blocks": len(set(row["block_id"] for row in rows)),
        "candidates": reports,
        "candidate_eligibility": eligible,
        "q2_grid": grid,
        "q2_selected_hyperparameters": best_q2,
        "selected": selected,
        "selection_rule": "simplest eligible candidate within 0.005 block-macro Brier of best eligible",
        "commissioning_size_curve": size_curve,
        "line_of_sight_prisms": len(scene.prisms),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pose_id", "position_id", "block_id", "fold", "camera_id", "available", *predictions))
        for index, row in enumerate(rows):
            writer.writerow((
                row["pose_id"], row["position_id"], row["block_id"], row["fold"],
                row["camera_id"], row["available"],
                *(f"{predictions[name][index]:.12g}" for name in predictions),
            ))
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "requested_positions", "effective_training_positions_by_fold",
            "block_macro_brier", "pooled_brier", "pooled_ece_10", "pooled_log_loss",
        ))
        writer.writeheader()
        for row in size_curve:
            writer.writerow({**row, "effective_training_positions_by_fold": json.dumps(row["effective_training_positions_by_fold"], sort_keys=True)})
    precision, recall, thresholds = precision_recall_curve(y, predictions[selected])
    pr_path = staging / "selected_precision_recall.csv"
    with pr_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("threshold", "precision", "recall"))
        for index in range(len(thresholds)):
            writer.writerow((thresholds[index], precision[index], recall[index]))
        writer.writerow(("", precision[-1], recall[-1]))
    manifest = {
        "schema": "thesis_stage08_run.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_audit_accessed": False,
        "protocol": os.path.relpath(protocol_path, REPO),
        "protocol_sha256": sha256(protocol_path),
        "source_admission_records_sha256": sha256(source),
        "implementation": os.path.relpath(Path(__file__).resolve(), REPO),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "selected": selected,
        "artifacts": {
            path.name: sha256(path)
            for path in (report_path, artifact_path, prediction_path, curve_path, pr_path)
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / ".complete").write_text("complete\n")
    staging.rename(output)
    print(json.dumps({
        "selected": selected,
        "candidate_metrics": {name: {
            "block_macro_brier": value["block_macro_brier"],
            "pooled_brier": value["pooled_brier"],
            "pooled_ece_10": value["pooled_ece_10"],
            "eligible": eligible[name],
        } for name, value in reports.items()},
        "q2": best_q2,
        "size_curve": size_curve,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
