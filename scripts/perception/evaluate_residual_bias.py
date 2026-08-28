#!/usr/bin/env python3
"""Evaluate learned pixel correction against raw and detection-only measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from residual_bias_model import feature_vector, load_artifact


def _camera(record: dict[str, str]):
    from unav_common.camera_model import ObliqueCameraModel

    pose = [float(value) for value in json.loads(record["camera_pose_xyz_rpy"])]
    x, y, z, _roll, pitch, yaw = pose
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    scale = -z / forward[2]
    return ObliqueCameraModel(
        cam_pos=pose[:3], look_at=(x + scale * forward[0], y + scale * forward[1], 0.0),
        img_width=int(record["image_width"]), img_height=int(record["image_height"]),
        fov_h_rad=float(record["fov_h_rad"]),
    )


def _stats(rows: list[dict], prefix: str) -> dict:
    errors = np.asarray([[row[f"{prefix}_error_x_m"], row[f"{prefix}_error_y_m"]] for row in rows])
    pixel = np.asarray([[row[f"{prefix}_pixel_u"], row[f"{prefix}_pixel_v"]] for row in rows])
    magnitude = np.linalg.norm(errors, axis=1)
    mean = errors.mean(axis=0)
    return {
        "n": len(rows),
        "mean_euclidean_error_m": float(magnitude.mean()),
        "median_euclidean_error_m": float(np.median(magnitude)),
        "p95_euclidean_error_m": float(np.percentile(magnitude, 95)),
        "rmse_m": float(np.sqrt(np.mean(magnitude ** 2))),
        "world_bias_x_m": float(mean[0]), "world_bias_y_m": float(mean[1]),
        "world_bias_m": float(np.linalg.norm(mean)),
        "signed_along_camera_bias_m": float(np.mean([row[f"{prefix}_along_m"] for row in rows])),
        "signed_cross_camera_bias_m": float(np.mean([row[f"{prefix}_cross_m"] for row in rows])),
        "pixel_rmse": float(np.sqrt(np.mean(np.sum(pixel ** 2, axis=1)))),
        "pixel_bias_u": float(pixel[:, 0].mean()), "pixel_bias_v": float(pixel[:, 1].mean()),
    }


def _range_group(value: float) -> str:
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 100)):
        if lo <= value < hi: return f"{lo}-{hi}m"
    return "outside"


def _heading_group(value: float) -> str:
    degrees = math.degrees(value) % 360.0
    lo = int(degrees // 45.0) * 45
    return f"{lo}-{lo + 45}deg"


def evaluate(dataset: Path, model_path: Path, output: Path, *, device: str) -> dict:
    import torch

    dataset, model_path, output = dataset.resolve(), model_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True)
    with (dataset / "records.csv").open(newline="", encoding="utf-8") as handle:
        all_test = [row for row in csv.DictReader(handle) if row["residual_split"] == "test"]
    rows = [row for row in all_test if row["detected"] == "1"]
    model, artifact = load_artifact(model_path, device=device)
    x = np.stack([
        feature_vector(row, disable_heading=bool(artifact["disable_heading"])) for row in rows
    ])
    x = (x - np.asarray(artifact["x_mean"], dtype=np.float32)) / np.asarray(artifact["x_std"], dtype=np.float32)
    with torch.no_grad():
        predicted_n = model(torch.from_numpy(x).to(device)).cpu().numpy()
    predicted = predicted_n * np.asarray(artifact["y_std"]) + np.asarray(artifact["y_mean"])

    scored = []
    for record, correction in zip(rows, predicted):
        camera = _camera(record)
        gt = np.asarray((float(record["robot_x"]), float(record["robot_y"])))
        camera_xy = camera.cam_pos[:2]
        bearing = gt - camera_xy; bearing /= np.linalg.norm(bearing)
        cross = np.asarray((-bearing[1], bearing[0]))
        target_u, target_v = float(record["target_floor_u"]), float(record["target_floor_v"])
        raw_u, raw_v = float(record["box_bottom_u"]), float(record["box_bottom_v"])
        learned_u, learned_v = raw_u + correction[0], raw_v + correction[1]
        # Perfect geometry correction applied to the predicted box leaves only the
        # detector-vs-semantic-box pixel error.
        detection_u = target_u + float(record["detector_error_u"])
        detection_v = target_v + float(record["detector_error_v"])
        item = {
            "camera": record["camera"], "robot_x": float(record["robot_x"]),
            "robot_y": float(record["robot_y"]), "robot_yaw": float(record["robot_yaw"]),
            "range_m": float(record["range_m"]),
            "predicted_du": float(correction[0]), "predicted_dv": float(correction[1]),
            "target_du": float(record["target_du"]), "target_dv": float(record["target_dv"]),
        }
        for prefix, u, v in (
            ("raw", raw_u, raw_v), ("learned", learned_u, learned_v),
            ("detection_only", detection_u, detection_v),
        ):
            world = camera.pixel_to_world_at_z(u, v, 0.0)
            if world is None:
                raise RuntimeError(f"invalid back-projection for {record['image']}")
            error = np.asarray(world) - gt
            item.update({
                f"{prefix}_pixel_u": u - target_u, f"{prefix}_pixel_v": v - target_v,
                f"{prefix}_error_x_m": float(error[0]), f"{prefix}_error_y_m": float(error[1]),
                f"{prefix}_along_m": float(error @ bearing), f"{prefix}_cross_m": float(error @ cross),
            })
        scored.append(item)
    arms = {prefix: _stats(scored, prefix) for prefix in ("raw", "learned", "detection_only")}
    groups = {"camera": defaultdict(list), "range": defaultdict(list), "heading": defaultdict(list)}
    for row in scored:
        groups["camera"][row["camera"]].append(row)
        groups["range"][_range_group(row["range_m"])].append(row)
        groups["heading"][_heading_group(row["robot_yaw"])].append(row)
    payload = {
        "status": "provisional_heldout_set_pose_mechanism_evaluation",
        "metric_object": "camera_measurement",
        "reference": "commanded_ground_truth_xy",
        "projection_runtime": "detector_bbox_bottom_centre_floor_ipm_with_learned_pixel_residual",
        "dataset": str(dataset), "model": str(model_path),
        "experimental_unit": "spatially held-out detected RGB image",
        "n_positive_trials": len(all_test), "n_detections": len(scored),
        "detection_rate": len(scored) / len(all_test),
        "online_inputs": [
            "RGB-derived detector box/confidence", "fixed camera calibration",
            *( [] if artifact["disable_heading"] else ["exact commanded heading (oracle online-input study)"] ),
        ],
        "evaluation_only_inputs": ["commanded GT x/y", "semantic mask box for detection-only floor"],
        "arms": arms,
        "detection_only_definition": "perfect semantic geometry correction plus predicted-vs-mask box error",
        "gap_to_detection_only": {
            "learned_minus_detection_only_rmse_m": arms["learned"]["rmse_m"] - arms["detection_only"]["rmse_m"],
            "learned_over_detection_only_rmse": arms["learned"]["rmse_m"] / arms["detection_only"]["rmse_m"],
        },
        "by": {
            axis: {key: {prefix: _stats(values, prefix) for prefix in ("raw", "learned", "detection_only")}
                   for key, values in sorted(mapping.items())}
            for axis, mapping in groups.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0])); writer.writeheader(); writer.writerows(scored)
    report = [
        "# Heading-conditioned detector residual — held-out measurement evaluation", "",
        f"Detected {len(scored)}/{len(all_test)} held-out positive RGB images.", "",
        "| Arm | Mean | Median | p95 | RMSE | Along bias | Cross bias |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("raw", "learned", "detection_only"):
        s = arms[name]
        report.append(
            f"| {name} | {100*s['mean_euclidean_error_m']:.2f} cm | {100*s['median_euclidean_error_m']:.2f} cm | "
            f"{100*s['p95_euclidean_error_m']:.2f} cm | {100*s['rmse_m']:.2f} cm | "
            f"{100*s['signed_along_camera_bias_m']:+.2f} cm | {100*s['signed_cross_camera_bias_m']:+.2f} cm |"
        )
    report += ["", "`detection_only` is an evaluation-only floor using semantic masks; it is not available online."]
    (output / "RESULTS.md").write_text("\n".join(report) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate(args.dataset, args.model, args.out, device=args.device)
    print(json.dumps({"arms": result["arms"], "gap": result["gap_to_detection_only"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
