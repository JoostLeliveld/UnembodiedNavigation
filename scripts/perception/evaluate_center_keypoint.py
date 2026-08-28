#!/usr/bin/env python3
"""Score a trained centre-keypoint model as a camera measurement, not as mAP."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _look_at(pose: list[float]) -> tuple[float, float, float]:
    x, y, z, _roll, pitch, yaw = pose
    forward = (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    )
    scale = -z / forward[2]
    return x + scale * forward[0], y + scale * forward[1], 0.0


def _camera(record: dict[str, str]):
    from unav_common.camera_model import ObliqueCameraModel

    pose = [float(value) for value in json.loads(record["camera_pose_xyz_rpy"])]
    return ObliqueCameraModel(
        cam_pos=pose[:3],
        look_at=_look_at(pose),
        img_width=int(record["image_width"]),
        img_height=int(record["image_height"]),
        fov_h_rad=float(record["fov_h_rad"]),
    )


def _stats(rows: list[dict]) -> dict:
    trials = len(rows)
    detected = [row for row in rows if row.get("detected")]
    output = {"trials": trials, "detections": len(detected),
              "detection_rate": len(detected) / trials if trials else float("nan")}
    if not detected:
        return output
    error = np.asarray([[row["error_x_m"], row["error_y_m"]] for row in detected])
    magnitude = np.linalg.norm(error, axis=1)
    pixel = np.asarray([[row["pixel_du"], row["pixel_dv"]] for row in detected])
    mean = error.mean(axis=0)
    output.update({
        "mean_error_x_m": float(mean[0]),
        "mean_error_y_m": float(mean[1]),
        "bias_m": float(np.linalg.norm(mean)),
        "signed_along_camera_bias_m": float(np.mean([row["along_m"] for row in detected])),
        "signed_cross_camera_bias_m": float(np.mean([row["cross_m"] for row in detected])),
        "median_euclidean_error_m": float(np.median(magnitude)),
        "p95_euclidean_error_m": float(np.percentile(magnitude, 95)),
        "rmse_m": float(np.sqrt(np.mean(magnitude ** 2))),
        "pixel_bias_u": float(pixel[:, 0].mean()),
        "pixel_bias_v": float(pixel[:, 1].mean()),
        "pixel_rmse": float(np.sqrt(np.mean(np.sum(pixel ** 2, axis=1)))),
    })
    return output


def _range_group(value: float) -> str:
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 100)):
        if lo <= value < hi:
            return f"{lo}-{hi}m"
    return "outside"


def _heading_group(value: float) -> str:
    degrees = math.degrees(value) % 360.0
    lo = int(degrees // 60.0) * 60
    return f"{lo}-{lo + 60}deg"


def _semantic_box_centre_baseline(dataset: Path, positives: list[dict[str, str]]) -> list[dict]:
    """Optimistic visible-box baseline using evaluation-only semantic masks.

    This is deliberately stronger and cleaner than a predicted detector box.  It
    isolates the geometric bias caused by using the centre of the *visible* robot
    extent, even when that extent is known perfectly.
    """
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source_root"])
    diagnostics: dict[tuple[str, str], dict[str, str]] = {}
    for camera_dir in sorted(source.glob("camera_*")):
        with (camera_dir / "label_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["accepted"] == "1" and row["sample_kind"] == "positive":
                    diagnostics[(camera_dir.name, row["sample_index"])] = row

    baseline: list[dict] = []
    for record in positives:
        source_row = diagnostics[(record["camera"], record["sample_index"])]
        # Diagnostics maxima are occupied indices, so convert to a half-open box.
        u = 0.5 * (
            float(source_row["mask_bbox_x0"]) + float(source_row["mask_bbox_x1"]) + 1.0
        )
        v = 0.5 * (
            float(source_row["mask_bbox_y0"]) + float(source_row["mask_bbox_y1"]) + 1.0
        )
        camera = _camera(record)
        world = camera.pixel_to_world_at_z(u, v, float(record["centre_z_m"]))
        item = {
            "camera": record["camera"],
            "range_m": float(record["range_m"]),
            "yaw_rad": float(record["robot_yaw"]),
            "visibility": int(record["visibility"]),
            "detected": world is not None,
        }
        if world is not None:
            gt = np.asarray((float(record["robot_x"]), float(record["robot_y"])))
            error = np.asarray(world) - gt
            bearing = gt - camera.cam_pos[:2]
            bearing /= np.linalg.norm(bearing)
            cross = np.asarray((-bearing[1], bearing[0]))
            item.update({
                "pixel_du": u - float(record["centre_u"]),
                "pixel_dv": v - float(record["centre_v"]),
                "error_x_m": float(error[0]),
                "error_y_m": float(error[1]),
                "along_m": float(error @ bearing),
                "cross_m": float(error @ cross),
            })
        baseline.append(item)
    return baseline


def evaluate(
    dataset: Path,
    weights: Path,
    output: Path,
    *,
    imgsz: int,
    conf: float,
    keypoint_conf: float,
    batch: int,
    device: str,
) -> dict:
    dataset = dataset.expanduser().resolve()
    weights = weights.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / "records.csv").open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    positives = [row for row in records if row["positive"] == "1"]
    negatives = [row for row in records if row["positive"] != "1"]
    semantic_box_baseline = _semantic_box_centre_baseline(dataset, positives)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    scored: list[dict] = []
    false_positives = 0
    all_records = positives + negatives
    for start in range(0, len(all_records), int(batch)):
        chunk = all_records[start:start + int(batch)]
        results = model.predict(
            [row["image"] for row in chunk], imgsz=int(imgsz), conf=float(conf),
            batch=int(batch), device=str(device), verbose=False,
        )
        for record, result in zip(chunk, results):
            boxes = result.boxes
            keypoints = result.keypoints
            if record["positive"] != "1":
                if boxes is not None and keypoints is not None and len(boxes) > 0:
                    keypoint_data = keypoints.data.detach().cpu().numpy()
                    false_positives += int(
                        keypoint_data.shape[1] >= 1
                        and keypoint_data.shape[2] >= 3
                        and np.any(keypoint_data[:, 0, 2] >= float(keypoint_conf))
                    )
                continue
            item = {
                "camera": record["camera"],
                "range_m": float(record["range_m"]),
                "yaw_rad": float(record["robot_yaw"]),
                "visibility": int(record["visibility"]),
                "detected": False,
            }
            if boxes is None or keypoints is None or len(boxes) == 0:
                scored.append(item)
                continue
            confidence = boxes.conf.detach().cpu().numpy()
            best = int(np.argmax(confidence))
            data = keypoints.data.detach().cpu().numpy()[best]
            if data.shape[0] < 1 or data.shape[1] < 3:
                scored.append(item)
                continue
            u, v, keypoint_confidence = (
                float(data[0, 0]), float(data[0, 1]), float(data[0, 2])
            )
            if keypoint_confidence < float(keypoint_conf) or not (
                math.isfinite(u) and math.isfinite(v)
            ):
                scored.append(item)
                continue
            camera = _camera(record)
            world = camera.pixel_to_world_at_z(u, v, float(record["centre_z_m"]))
            if world is None:
                scored.append(item)
                continue
            gt = np.asarray((float(record["robot_x"]), float(record["robot_y"])))
            error = np.asarray(world) - gt
            bearing = gt - camera.cam_pos[:2]
            bearing /= np.linalg.norm(bearing)
            cross = np.asarray((-bearing[1], bearing[0]))
            item.update({
                "detected": True,
                "confidence": float(confidence[best]),
                "keypoint_confidence": keypoint_confidence,
                "pred_u": u, "pred_v": v,
                "pixel_du": u - float(record["centre_u"]),
                "pixel_dv": v - float(record["centre_v"]),
                "error_x_m": float(error[0]), "error_y_m": float(error[1]),
                "along_m": float(error @ bearing), "cross_m": float(error @ cross),
            })
            scored.append(item)

    groups: dict[str, dict[str, list[dict]]] = {
        "camera": defaultdict(list), "range": defaultdict(list),
        "heading": defaultdict(list), "visibility": defaultdict(list),
    }
    for row in scored:
        groups["camera"][row["camera"]].append(row)
        groups["range"][_range_group(row["range_m"])].append(row)
        groups["heading"][_heading_group(row["yaw_rad"])].append(row)
        groups["visibility"]["visible" if row["visibility"] == 2 else "amodal"].append(row)
    payload = {
        "status": "provisional_heldout_measurement_evaluation",
        "metric_object": "camera_measurement_from_learned_centre_keypoint",
        "reference": "commanded_ground_truth_xy",
        "projection_runtime": f"keypoint back-projected at z={positives[0]['centre_z_m']} m",
        "dataset": str(dataset),
        "weights": str(weights),
        "experimental_unit": "held-out RGB image",
        "n_positive_trials": len(positives),
        "n_negative_trials": len(negatives),
        "online_inputs": ["RGB image", "fixed camera calibration"],
        "evaluation_only_inputs": [
            "robot_x", "robot_y", "robot_yaw", "target pixel", "visibility label",
            "semantic mask box for the optimistic baseline",
        ],
        "inference": {
            "imgsz": int(imgsz),
            "box_confidence": float(conf),
            "keypoint_confidence": float(keypoint_conf),
            "batch": int(batch),
        },
        "pooled": _stats(scored),
        "oracle_visible_mask_box_centre_baseline": {
            "status": "evaluation_only_optimistic_baseline",
            "warning": "uses ground-truth semantic mask boxes, not detector predictions",
            "pooled": _stats(semantic_box_baseline),
        },
        "false_positive_images": false_positives,
        "by": {axis: {key: _stats(value) for key, value in sorted(mapping.items())}
               for axis, mapping in groups.items()},
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in scored for key in row})
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)

    pooled = payload["pooled"]
    box_baseline = payload["oracle_visible_mask_box_centre_baseline"]["pooled"]
    report = [
        "# Learned robot-centre observation — held-out measurement evaluation",
        "",
        f"- dataset: `{dataset}`",
        f"- weights: `{weights}`",
        f"- unit: {len(positives)} spatially held-out RGB images",
        f"- detections: {pooled['detections']} ({100.0 * pooled['detection_rate']:.1f}%)",
    ]
    if pooled["detections"]:
        report.extend([
            f"- camera-measurement median Euclidean error vs commanded GT: {100*pooled['median_euclidean_error_m']:.2f} cm",
            f"- p95 Euclidean error: {100*pooled['p95_euclidean_error_m']:.2f} cm",
            f"- RMSE: {100*pooled['rmse_m']:.2f} cm",
            f"- pooled signed bias magnitude: {100*pooled['bias_m']:.2f} cm",
            f"- signed camera-bearing bias: along {100*pooled['signed_along_camera_bias_m']:+.2f} cm, cross {100*pooled['signed_cross_camera_bias_m']:+.2f} cm",
        ])
    report.extend([
        f"- false-positive images: {false_positives}/{len(negatives)}",
        "",
        "## Optimistic visible-box-centre baseline",
        "",
        "This baseline uses evaluation-only ground-truth semantic masks, not predicted boxes.",
        f"Its median / p95 / RMSE are {100*box_baseline['median_euclidean_error_m']:.2f} / "
        f"{100*box_baseline['p95_euclidean_error_m']:.2f} / {100*box_baseline['rmse_m']:.2f} cm,",
        f"with pooled bias {100*box_baseline['bias_m']:.2f} cm.",
        "",
        "Ground truth and visibility are used only by this offline evaluator.",
    ])
    (output / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--keypoint-conf", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = evaluate(
        args.dataset, args.weights, args.out, imgsz=args.imgsz, conf=args.conf,
        keypoint_conf=args.keypoint_conf, batch=args.batch, device=args.device,
    )
    print(json.dumps(result["pooled"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
