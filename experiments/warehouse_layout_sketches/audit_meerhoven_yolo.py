#!/usr/bin/env python3
"""Fail-closed offline detector gate for the immutable Meerhoven A--L merge.

Semantic masks and robot poses are evaluation-only here. They never enter the
runtime detector, calibration, GP, manager, or planner paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDEFGHIJKL")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty sample index: {path}")
    return rows


def _label_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for raw in path.read_text(encoding="utf-8").splitlines():
        values = raw.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            continue
        coords = np.asarray([float(item) for item in values[1:]], dtype=float).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0] * width, 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1] * height, 0, height - 1)
        cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
    return mask


def _prediction_masks(result, width: int, height: int) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    if result.masks is not None and result.masks.xy is not None:
        for polygon in result.masks.xy:
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.asarray(polygon, dtype=float)
            if points.shape[0] >= 3:
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
            output.append(mask)
        return output
    if result.boxes is not None:
        for box in result.boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[max(y0, 0):min(y1 + 1, height), max(x0, 0):min(x1 + 1, width)] = 1
            output.append(mask)
    return output


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero((left > 0) | (right > 0)))
    if union == 0:
        return 0.0
    return float(np.count_nonzero((left > 0) & (right > 0))) / float(union)


def _bottom_point(mask: np.ndarray, band_px: int = 3) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    bottom = int(np.max(ys))
    band = xs[ys >= bottom - max(int(band_px), 0)]
    return float(np.mean(band)), float(bottom)


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mask-iou", type=float, default=0.25)
    parser.add_argument("--min-camera-recall", type=float, default=0.75)
    parser.add_argument("--min-range-recall", type=float, default=0.65)
    parser.add_argument("--min-range-samples", type=int, default=8)
    parser.add_argument("--max-median-bottom-error-px", type=float, default=20.0)
    parser.add_argument("--max-p90-bottom-error-px", type=float, default=60.0)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    index_path = dataset / "sample_qualifications.csv"
    completion = dataset / ".complete"
    manifest = dataset / "dataset_manifest.json"
    for required in (model_path, index_path, completion, manifest):
        if not required.is_file():
            raise RuntimeError(f"Required immutable input is missing: {required}")

    all_rows = _read_index(index_path)
    selected = [
        row for row in all_rows
        if (row["sample_kind"] == "positive" and row["split"] == "val")
        or row["sample_kind"] == "negative"
    ]
    if {row["camera_id"] for row in selected} != set(CAMERAS):
        raise RuntimeError("Audit index does not contain every camera A--L")
    images = [str(dataset / row["image"]) for row in selected]

    model = YOLO(str(model_path))
    results = model.predict(
        source=images,
        imgsz=int(args.imgsz),
        conf=float(args.confidence),
        iou=0.45,
        device=(str(args.device) or None),
        batch=max(int(args.batch), 1),
        verbose=False,
        stream=False,
    )
    if len(results) != len(selected):
        raise RuntimeError(f"Prediction count mismatch: {len(results)} != {len(selected)}")

    audit_rows: list[dict[str, object]] = []
    for source, result in zip(selected, results):
        image_path = dataset / source["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        height, width = image.shape[:2]
        predictions = _prediction_masks(result, width, height)
        is_positive = source["sample_kind"] == "positive"
        best_iou = 0.0
        bottom_error = math.nan
        predicted_bottom_u = math.nan
        predicted_bottom_v = math.nan
        hit = False
        if is_positive:
            truth = _label_mask(dataset / source["label"], width, height)
            if not np.any(truth):
                raise RuntimeError(f"Positive sample has an empty label: {source['label']}")
            scores = [_iou(truth, candidate) for candidate in predictions]
            if scores:
                best_index = int(np.argmax(scores))
                best_iou = float(scores[best_index])
                hit = best_iou >= float(args.mask_iou)
                truth_point = _bottom_point(truth)
                predicted_point = _bottom_point(predictions[best_index])
                if hit and truth_point is not None and predicted_point is not None:
                    bottom_error = math.dist(truth_point, predicted_point)
                    predicted_bottom_u, predicted_bottom_v = predicted_point
        false_positive = (not is_positive) and bool(predictions)
        audit_rows.append({
            "sample_id": source["sample_id"],
            "camera_id": source["camera_id"],
            "sample_kind": source["sample_kind"],
            "split": source["split"],
            "range_bin": source["range_bin"],
            "occlusion_state": source["occlusion_state"],
            "localization_qualified": int(_as_bool(source["localization_qualified"])),
            "prediction_count": len(predictions),
            "best_mask_iou": best_iou,
            "hit": int(hit),
            "false_positive": int(false_positive),
            "bottom_error_px": bottom_error,
            "predicted_bottom_u": predicted_bottom_u,
            "predicted_bottom_v": predicted_bottom_v,
            "true_x": float(source["robot_x"]),
            "true_y": float(source["robot_y"]),
        })

    positives = [row for row in audit_rows if row["sample_kind"] == "positive"]
    negatives = [row for row in audit_rows if row["sample_kind"] == "negative"]
    by_camera: dict[str, dict[str, object]] = {}
    range_cells: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    all_localization_errors: list[float] = []
    for camera in CAMERAS:
        rows = [row for row in positives if row["camera_id"] == camera]
        recall = float(np.mean([int(row["hit"]) for row in rows])) if rows else 0.0
        errors = [
            float(row["bottom_error_px"]) for row in rows
            if int(row["localization_qualified"]) and math.isfinite(float(row["bottom_error_px"]))
        ]
        all_localization_errors.extend(errors)
        by_camera[camera] = {
            "validation_opportunities": len(rows),
            "hits": sum(int(row["hit"]) for row in rows),
            "recall": recall,
            "bottom_error_median_px": _quantile(errors, 0.5),
            "bottom_error_p90_px": _quantile(errors, 0.9),
        }
        if recall < float(args.min_camera_recall):
            failures.append(
                f"{camera} recall {recall:.3f} < {float(args.min_camera_recall):.3f}"
            )
        for range_bin in sorted({str(row["range_bin"]) for row in rows}):
            cell = [row for row in rows if row["range_bin"] == range_bin]
            cell_recall = float(np.mean([int(row["hit"]) for row in cell]))
            key = f"{camera}/{range_bin}"
            range_cells[key] = {
                "validation_opportunities": len(cell),
                "hits": sum(int(row["hit"]) for row in cell),
                "recall": cell_recall,
                "gate_applies": len(cell) >= int(args.min_range_samples),
            }
            if len(cell) >= int(args.min_range_samples) and cell_recall < float(args.min_range_recall):
                failures.append(
                    f"{key} recall {cell_recall:.3f} < {float(args.min_range_recall):.3f}"
                )

    median_error = _quantile(all_localization_errors, 0.5)
    p90_error = _quantile(all_localization_errors, 0.9)
    if median_error is None or median_error > float(args.max_median_bottom_error_px):
        failures.append(f"bottom error median {median_error} exceeds gate")
    if p90_error is None or p90_error > float(args.max_p90_bottom_error_px):
        failures.append(f"bottom error p90 {p90_error} exceeds gate")
    false_positives = sum(int(row["false_positive"]) for row in negatives)
    if len(negatives) != len(CAMERAS):
        failures.append(f"background count {len(negatives)} != {len(CAMERAS)}")
    if false_positives:
        failures.append(f"background false positives {false_positives} != 0")

    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "sample_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    projection_dir = output / "evaluation_projection_inputs"
    projection_dir.mkdir()
    projection_fields = ("diag_stamp", "detected", "true_x", "true_y", "obs_u", "obs_v")
    for camera_index, camera in enumerate(CAMERAS):
        rows = [row for row in audit_rows if row["camera_id"] == camera]
        with (projection_dir / f"{camera}_perception.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=projection_fields)
            writer.writeheader()
            for row_index, row in enumerate(rows):
                usable = (
                    row["sample_kind"] == "positive"
                    and int(row["hit"]) == 1
                    and int(row["localization_qualified"]) == 1
                    and math.isfinite(float(row["predicted_bottom_u"]))
                    and math.isfinite(float(row["predicted_bottom_v"]))
                )
                writer.writerow({
                    "diag_stamp": float(camera_index * 1_000_000 + row_index),
                    "detected": int(usable),
                    "true_x": row["true_x"],
                    "true_y": row["true_y"],
                    "obs_u": row["predicted_bottom_u"] if usable else "",
                    "obs_v": row["predicted_bottom_v"] if usable else "",
                })
    summary = {
        "status": "pass" if not failures else "fail",
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "dataset": str(dataset),
        "dataset_manifest_sha256": _sha256(manifest),
        "semantic_truth_role": "evaluation_only",
        "projection_fit_input_role": (
            "evaluation_only spatial-validation detections; calibration fit only, "
            "must be qualified on the separately held-out routes"
        ),
        "parameters": vars(args) | {"model": str(model_path), "dataset": str(dataset), "out": str(output)},
        "camera_results": by_camera,
        "camera_range_results": range_cells,
        "localization": {
            "matched_qualified_samples": len(all_localization_errors),
            "bottom_error_median_px": median_error,
            "bottom_error_p90_px": p90_error,
        },
        "backgrounds": {
            "camera_distinct_frames": len(negatives),
            "false_positives": false_positives,
        },
        "failures": failures,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
