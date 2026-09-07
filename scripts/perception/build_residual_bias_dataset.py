#!/usr/bin/env python3
"""Freeze detector observations and pixel-residual targets for bias learning.

The runtime baseline is the detector box bottom-centre back-projected onto z=0.
The learned target is only the systematic geometry term: projected commanded-GT
(x, y, z=0) minus the ground-truth visible-mask bottom-centre.  It deliberately
excludes predicted-vs-mask detector error, because that stochastic error is what
must remain after correction.  Heading is stored as an online candidate input;
ground-truth position and semantic masks remain training/evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import io
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_center_keypoint_dataset import project_point
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/unav_common'))
from unav_common.capture_integrity import checked_image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera(record: dict[str, str]):
    from unav_common.camera_model import ObliqueCameraModel

    pose = [float(value) for value in json.loads(record["camera_pose_xyz_rpy"])]
    x, y, z, _roll, pitch, yaw = pose
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    scale = -z / forward[2]
    look_at = (x + scale * forward[0], y + scale * forward[1], 0.0)
    return ObliqueCameraModel(
        cam_pos=pose[:3], look_at=look_at,
        img_width=int(record["image_width"]), img_height=int(record["image_height"]),
        fov_h_rad=float(record["fov_h_rad"]),
    )


def _calibration_split(x: str, y: str) -> str:
    key = f"residual-calibration-v1|{float(x):.6f}|{float(y):.6f}".encode()
    return "calibration" if int(hashlib.sha256(key).hexdigest()[:8], 16) % 5 == 0 else "fit"


def _source_diagnostics(source_root: Path, sources: dict) -> dict[tuple[str, str], dict[str, str]]:
    if not sources:
        raise ValueError('centre dataset must declare exact source manifests/diagnostics')
    result = {}
    for camera_name, entry in sources.items():
        for key in ('manifest', 'diagnostics'):
            data = Path(entry[key]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.get(key + '_sha256'):
                raise ValueError(f'changed source {key} for {camera_name}')
            if key == 'diagnostics':
                diagnostics = list(csv.DictReader(io.StringIO(data.decode())))
        for row in diagnostics:
            if row['accepted'] == '1' and row['sample_kind'] == 'positive':
                key = (camera_name, row['sample_index'])
                if key in result:
                    raise ValueError('duplicate source diagnostic identity')
                result[key] = row
    return result


def build(
    centre_dataset: Path, detector: Path, output: Path, *,
    imgsz: int, confidence: float, batch: int, device: str,
) -> dict:
    centre_dataset = centre_dataset.resolve()
    detector = detector.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    staged = output.with_name(output.name + ".incomplete")
    if staged.exists():
        raise FileExistsError(f"staging output exists: {staged}")
    staged.mkdir(parents=True)
    try:
        manifest_bytes = (centre_dataset / 'dataset_manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        completion = json.loads((centre_dataset / '.complete').read_text())
        if completion.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError('centre dataset completion digest mismatch')
        records_bytes = (centre_dataset / 'records.csv').read_bytes()
        if manifest.get('records_sha256') != hashlib.sha256(records_bytes).hexdigest():
            raise ValueError('centre dataset records digest mismatch')
        if int(batch) < 1:
            raise ValueError('inference batch must be positive')
        source_root = Path(manifest["source_root"])
        diagnostics = _source_diagnostics(source_root, manifest.get('sources', {}))
        with io.StringIO(records_bytes.decode(), newline='') as handle:
            # Exclude clipped-centre and background records from the correction task.
            records = [row for row in csv.DictReader(handle) if row["positive"] == "1"]

        from ultralytics import YOLO

        detector_hash = _sha256(detector)
        detector_copy = staged / 'frozen_detector_input.pt'
        detector_copy.write_bytes(detector.read_bytes())
        if _sha256(detector_copy) != detector_hash:
            raise ValueError('detector changed during loading')
        model = YOLO(str(detector_copy))
        output_rows: list[dict[str, object]] = []
        for start in range(0, len(records), int(batch)):
            chunk = records[start:start + int(batch)]
            images = []
            for row in chunk:
                diagnostic = diagnostics[(row['camera'], row['sample_index'])]
                relative = str(Path(row['image']).resolve().relative_to(source_root.resolve()))
                image = checked_image(source_root, dict(image=relative, image_sha1=diagnostic.get('image_sha1')),
                                      size=(int(row['image_width']), int(row['image_height'])))
                row['image_sha1'] = diagnostic['image_sha1']
                images.append(image)
            predictions = model.predict(
                images, imgsz=int(imgsz), conf=float(confidence),
                batch=int(batch), device=str(device), verbose=False,
            )
            predictions = list(predictions)
            if len(predictions) != len(chunk):
                raise RuntimeError('detector result cardinality differs from source image count')
            for record, prediction in zip(chunk, predictions, strict=True):
                row: dict[str, object] = dict(record)
                row["residual_split"] = (
                    "test" if record["split"] == "val"
                    else _calibration_split(record["robot_x"], record["robot_y"])
                )
                boxes = prediction.boxes
                if boxes is None or len(boxes) == 0:
                    row["detected"] = 0
                    row['inference_status'] = 'miss'
                    output_rows.append(row)
                    continue
                confidence_values = boxes.conf.detach().cpu().numpy()
                best = int(np.argmax(confidence_values))
                x1, y1, x2, y2 = boxes.xyxy.detach().cpu().numpy()[best].tolist()
                bottom_u, bottom_v = 0.5 * (x1 + x2), y2
                camera = _camera(record)
                baseline_world = camera.pixel_to_world_at_z(bottom_u, bottom_v, 0.0)
                pose = [float(value) for value in json.loads(record["camera_pose_xyz_rpy"])]
                target_u, target_v, target_inside = project_point(
                    (float(record["robot_x"]), float(record["robot_y"]), 0.0),
                    camera_pose_xyz_rpy=pose,
                    image_width=int(record["image_width"]), image_height=int(record["image_height"]),
                    fov_h_rad=float(record["fov_h_rad"]),
                )
                if baseline_world is None or not target_inside:
                    row["detected"] = 0
                    row['inference_status'] = 'detected_projection_or_target_invalid'
                    output_rows.append(row)
                    continue
                baseline_xy = np.asarray(baseline_world)
                camera_xy = np.asarray(pose[:2])
                bearing = math.atan2(baseline_xy[1] - camera_xy[1], baseline_xy[0] - camera_xy[0])
                source = diagnostics[(record["camera"], record["sample_index"])]
                # Occupied-pixel maxima -> half-open box coordinates.
                mask_bottom_u = 0.5 * (
                    float(source["mask_bbox_x0"]) + float(source["mask_bbox_x1"]) + 1.0
                )
                mask_bottom_v = float(source["mask_bbox_y1"]) + 1.0
                row.update({
                    "detected": 1,
                    'inference_status': 'detected_target_valid',
                    "box_x1": f"{x1:.10f}", "box_y1": f"{y1:.10f}",
                    "box_x2": f"{x2:.10f}", "box_y2": f"{y2:.10f}",
                    "box_bottom_u": f"{bottom_u:.10f}", "box_bottom_v": f"{bottom_v:.10f}",
                    "box_confidence": f"{confidence_values[best]:.10f}",
                    "baseline_world_x": f"{baseline_xy[0]:.10f}",
                    "baseline_world_y": f"{baseline_xy[1]:.10f}",
                    "baseline_range_m": f"{np.linalg.norm(baseline_xy - camera_xy):.10f}",
                    "baseline_bearing_rad": f"{bearing:.10f}",
                    "target_floor_u": f"{target_u:.10f}", "target_floor_v": f"{target_v:.10f}",
                    "target_du": f"{target_u - mask_bottom_u:.10f}",
                    "target_dv": f"{target_v - mask_bottom_v:.10f}",
                    "full_residual_du": f"{target_u - bottom_u:.10f}",
                    "full_residual_dv": f"{target_v - bottom_v:.10f}",
                    "mask_bottom_u": f"{mask_bottom_u:.10f}",
                    "mask_bottom_v": f"{mask_bottom_v:.10f}",
                    "detector_error_u": f"{bottom_u - mask_bottom_u:.10f}",
                    "detector_error_v": f"{bottom_v - mask_bottom_v:.10f}",
                })
                output_rows.append(row)

        detector_copy.unlink()
        if _sha256(detector) != detector_hash:
            raise ValueError('detector source changed during dataset construction')
        if (centre_dataset / 'dataset_manifest.json').read_bytes() != manifest_bytes or (centre_dataset / 'records.csv').read_bytes() != records_bytes:
            raise ValueError('centre dataset changed during residual construction')
        image_splits = {}
        for row in output_rows:
            old = image_splits.setdefault(row['image_sha1'], row['residual_split'])
            if old != row['residual_split']:
                raise ValueError('same detector image appears in different residual fitting roles')
        if len(output_rows) != len(records):
            raise RuntimeError('residual dataset lost input rows')
        fields = sorted({key for row in output_rows for key in row})
        records_out = staged / "records.csv"
        with records_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        counts = Counter((row["residual_split"], int(row["detected"])) for row in output_rows)
        payload = {
            "status": "complete_provisional_residual_dataset",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "metric_object": "camera_measurement_pixel_residual_training_rows",
            "baseline_projection": "detector_bbox_bottom_centre_to_z0_floor",
            "target": "projected commanded-GT floor centre minus semantic-mask bottom-centre",
            "target_excludes": "predicted-vs-semantic-box detector error",
            "online_candidate_inputs": [
                "detector box geometry", "detector confidence", "fixed camera ID/calibration",
                "heading observation (commanded GT is used in this provisional set-pose study)",
            ],
            "evaluation_only_inputs": ["commanded GT x/y", "semantic mask box"],
            "centre_dataset": str(centre_dataset),
            "centre_dataset_manifest_sha256": _sha256(centre_dataset / "dataset_manifest.json"),
            "detector": str(detector), "detector_sha256": detector_hash,
            "detector_inference": {"imgsz": imgsz, "confidence": confidence, "batch": batch},
            "split_contract": "source val is untouched test; source train xy groups hash to fit/calibration",
            "counts": {f"{split}_{'detected' if detected else 'missed'}": count
                       for (split, detected), count in sorted(counts.items())},
            "records_sha256": _sha256(records_out),
        }
        manifest_out = staged / "dataset_manifest.json"
        manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (staged / ".complete").write_text(json.dumps({"manifest_sha256": _sha256(manifest_out)}) + "\n")
        os.replace(staged, output)
        return payload
    except BaseException:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centre-dataset", required=True, type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    print(json.dumps(build(
        args.centre_dataset, args.detector, args.out, imgsz=args.imgsz,
        confidence=args.confidence, batch=args.batch, device=args.device,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
