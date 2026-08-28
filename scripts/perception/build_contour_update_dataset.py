#!/usr/bin/env python3
"""Join frozen RGB-derived contours to candidate-prior rows for a paired ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from contour_update_model import CONTOUR_POINTS, resample_closed_contour


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera(record):
    from unav_common.camera_model import ObliqueCameraModel
    pose = [float(value) for value in json.loads(record["camera_pose_xyz_rpy"])]
    x, y, z, _roll, pitch, yaw = pose
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    scale = -z / forward[2]
    return ObliqueCameraModel(cam_pos=pose[:3],
        look_at=(x + scale * forward[0], y + scale * forward[1], 0.0),
        img_width=int(record["image_width"]), img_height=int(record["image_height"]),
        fov_h_rad=float(record["fov_h_rad"]))


def _projected_hull(camera, x, y, yaw):
    from unav_common.robot_hull import VISUAL_HULL
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = VISUAL_HULL @ rotation.T + np.asarray([x, y, 0.0])
    cam = (world - camera.cam_pos) @ camera.R.T
    cam = cam[cam[:, 2] > 1e-6]
    projected = (camera.K @ cam.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def _bottom(points, band_px=3.0):
    bottom_v = float(points[:, 1].max())
    band = points[points[:, 1] >= bottom_v - band_px]
    return float(band[:, 0].mean()), bottom_v


def build(source: Path, predictions: Path, output: Path, *, min_confidence: float,
          min_mask_area: float) -> dict:
    from reliability.silhouette_observation import equivalent_position_measurement, plausibility_reasons

    source, predictions, output = source.resolve(), predictions.resolve(), output.resolve()
    if output.exists(): raise FileExistsError(f"output exists: {output}")
    staged = output.with_name(output.name + ".incomplete")
    if staged.exists(): raise FileExistsError(f"staging output exists: {staged}")
    staged.mkdir(parents=True)
    with (predictions / "contours.csv").open(newline="", encoding="utf-8") as handle:
        contour_rows = {(row["camera"], row["sample_index"]): row for row in csv.DictReader(handle)}
    with (source / "records.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for base in source_rows:
        prediction = contour_rows[(base["camera"], base["sample_index"])]
        row = dict(base)
        row["seg_detected"] = prediction["detected"]
        row["gate_pass"] = "0"
        if prediction["detected"] != "1":
            row["gate_reasons"] = "missing_predicted_contour"
            rows.append(row); continue
        polygon = np.asarray(json.loads(prediction["polygon_json"]), dtype=float)
        area = float(cv2.contourArea(polygon.astype(np.float32)))
        if len(polygon) < 3 or area < min_mask_area:
            row["gate_reasons"] = "small_predicted_contour"
            rows.append(row); continue
        predicted = resample_closed_contour(polygon, CONTOUR_POINTS)
        camera = _camera(row); prior_x, prior_y = float(row["prior_x"]), float(row["prior_y"])
        yaw = float(row["prior_yaw"])
        expected = resample_closed_contour(_projected_hull(camera, prior_x, prior_y, yaw), CONTOUR_POINTS)
        x1, y1, x2, y2 = (float(prediction[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2"))
        raw = camera.pixel_to_world_at_z(0.5 * (x1 + x2), y2, 0.0)
        analytic = None if raw is None else equivalent_position_measurement(
            raw, ((1.0, 0.0), (0.0, 1.0)), camera, (prior_x, prior_y), yaw)
        reasons = plausibility_reasons((x1, y1, x2, y2), camera, prior_x, prior_y, yaw,
            image_size=(int(row["image_width"]), int(row["image_height"])))
        confidence = float(prediction["confidence"])
        gate_pass = confidence >= min_confidence and not reasons and analytic is not None
        bottom_u, bottom_v = _bottom(polygon)
        detection_only = camera.pixel_to_world_at_z(
            float(row["target_floor_u"]) + bottom_u - float(row["mask_bottom_u"]),
            float(row["target_floor_v"]) + bottom_v - float(row["mask_bottom_v"]), 0.0)
        row.update({
            "box_x1": f"{x1:.10f}", "box_y1": f"{y1:.10f}",
            "box_x2": f"{x2:.10f}", "box_y2": f"{y2:.10f}",
            "box_bottom_u": f"{0.5*(x1+x2):.10f}", "box_bottom_v": f"{y2:.10f}",
            "box_confidence": f"{confidence:.10f}", "contour_area_px": f"{area:.10f}",
            "contour_bottom_u": f"{bottom_u:.10f}", "contour_bottom_v": f"{bottom_v:.10f}",
            "raw_ground_x": "" if raw is None else f"{raw[0]:.10f}",
            "raw_ground_y": "" if raw is None else f"{raw[1]:.10f}",
            "analytic_x": "" if analytic is None else f"{analytic[0][0]:.10f}",
            "analytic_y": "" if analytic is None else f"{analytic[0][1]:.10f}",
            "detection_only_x": "" if detection_only is None else f"{detection_only[0]:.10f}",
            "detection_only_y": "" if detection_only is None else f"{detection_only[1]:.10f}",
            "gate_pass": int(gate_pass and detection_only is not None),
            "gate_reasons": "|".join(reasons),
        })
        for index, (predicted_point, expected_point) in enumerate(zip(predicted, expected)):
            row[f"contour_u_{index:02d}"] = f"{predicted_point[0]:.10f}"
            row[f"contour_v_{index:02d}"] = f"{predicted_point[1]:.10f}"
            row[f"expected_u_{index:02d}"] = f"{expected_point[0]:.10f}"
            row[f"expected_v_{index:02d}"] = f"{expected_point[1]:.10f}"
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    records = staged / "records.csv"
    with records.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    counts = Counter((row["residual_split"], int(row["exact_prior"]), int(row["gate_pass"])) for row in rows)
    payload = {
        "status": "complete_provisional_rgb_contour_update_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_manifest_sha256": _sha256(source / "dataset_manifest.json"),
        "contour_predictions": str(predictions),
        "contour_predictions_manifest_sha256": _sha256(predictions / "manifest.json"),
        "online_inputs": ["RGB-derived YOLO segmentation contour", "candidate prior xy", "exact commanded heading", "camera calibration", "known CAD visual hull"],
        "evaluation_only_inputs": ["commanded GT xy", "simulator semantic mask for detector-only scoring"],
        "admission": {"confidence_min": min_confidence, "mask_area_min_px": min_mask_area,
                      "shape_gate": "runtime hull plausibility gate"},
        "counts": {f"{s}_{'exact' if e else 'perturbed'}_{'kept' if g else 'rejected'}": n
                   for (s, e, g), n in sorted(counts.items())},
        "records_sha256": _sha256(records),
    }
    manifest = staged / "dataset_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (staged / ".complete").write_text(json.dumps({"manifest_sha256": _sha256(manifest)}) + "\n")
    os.replace(staged, output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-mask-area", type=float, default=12.0)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.predictions, args.out,
                           min_confidence=args.min_confidence, min_mask_area=args.min_mask_area),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
