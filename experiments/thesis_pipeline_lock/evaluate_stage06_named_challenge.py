#!/usr/bin/env python3
"""Evaluate the frozen post-selection Camera-D sliver challenge."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from export_detector_dataset import classify
from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import equivalent_position_measurement
from select_stage06_gate import gate_reasons


REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture = args.capture.resolve()
    gate_report_path = args.gate_report.resolve()
    gate_report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    candidate = gate_report["selected"]["candidate"]
    with (capture / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or {row["camera_id"] for row in rows} != {
        f"camera_{letter}" for letter in "ABCDE"
    }:
        raise RuntimeError("Named challenge must contain one synchronized five-camera batch")
    if any(row["dataset_split"] != "stage06_named_diagnostic" for row in rows):
        raise RuntimeError("Unexpected data role in named challenge")

    label_protocol_path = REPO / "experiments/thesis_pipeline_lock/label_dataset_protocol.json"
    label_protocol = json.loads(label_protocol_path.read_text(encoding="utf-8"))
    label_contract = label_protocol["detector_label_contract"]["positive_requires_all"]
    checkpoint = REPO / "logs/thesis_final_pipeline_v1/stage05_detector_training/imgsz960/upper_finetune/weights/best.pt"
    model = YOLO(str(checkpoint))
    images = [capture / row["image"] for row in rows]
    results = model.predict(
        source=[str(path) for path in images], imgsz=960, conf=0.001, iou=0.7,
        max_det=20, device="0", batch=2, workers=2, verbose=False, save=False,
    )
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    evaluated = []
    for row, result in zip(rows, results, strict=True):
        label_class, reference_reasons, mask_box, metrics = classify(
            row, 1280, 720, label_contract
        )
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        scores = result.boxes.conf.detach().cpu().numpy().astype(float)
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        valid = np.isfinite(scores) & np.all(np.isfinite(boxes), axis=1)
        valid &= (boxes[:, 0] >= 0.0) & (boxes[:, 1] >= 0.0)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (classes == 0)
        indexes = np.flatnonzero(valid)
        indexes = indexes[np.argsort(-scores[indexes], kind="stable")]
        box = None if not len(indexes) else [float(value) for value in boxes[int(indexes[0]), :4]]
        confidence = 0.0 if not len(indexes) else float(scores[int(indexes[0])])
        camera = camera_model_from_world(world, include_name=row["camera_model"])
        raw = equivalent = None
        if box is not None:
            raw = camera.pixel_to_world_at_z(0.5 * (box[0] + box[2]), box[3], 0.0)
            if raw is not None:
                converted = equivalent_position_measurement(
                    raw, ((1.0, 0.0), (0.0, 1.0)), camera,
                    (float(row["robot_x"]), float(row["robot_y"])),
                    float(row["robot_yaw"]),
                )
                equivalent = None if converted is None else converted[0]
        expected = [float(row[key]) for key in ("expected_x0", "expected_y0", "expected_x1", "expected_y1")]
        record = {
            "best_box_xyxy": box,
            "raw_best_confidence": confidence,
            "expected_box_xyxy": expected,
            "raw_ground_xy": None if raw is None else list(raw),
            "silhouette_equivalent_xy_exact_prior": None if equivalent is None else list(equivalent),
        }
        reasons = gate_reasons(record, candidate)
        truth = (float(row["robot_x"]), float(row["robot_y"]))
        evaluated.append({
            "camera_id": row["camera_id"],
            "reference_class": label_class,
            "reference_reasons": reference_reasons,
            "semantic_robot_pixels": int(float(row["semantic_robot_pixels"])),
            "reference_metrics": metrics,
            "mask_box_xyxy": None if mask_box is None else list(mask_box),
            "expected_box_xyxy": expected,
            "best_box_xyxy": box,
            "confidence": confidence,
            "gate_admitted": not reasons,
            "gate_reasons": reasons,
            "raw_ground_error_m": None if raw is None else math.hypot(raw[0] - truth[0], raw[1] - truth[1]),
            "equivalent_error_m": None if equivalent is None else math.hypot(equivalent[0] - truth[0], equivalent[1] - truth[1]),
        })

    camera_d = next(record for record in evaluated if record["camera_id"] == "camera_D")
    passed = not camera_d["gate_admitted"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_d = capture / next(row["image"] for row in rows if row["camera_id"] == "camera_D")
    canvas = cv2.imread(str(source_d), cv2.IMREAD_COLOR)
    if canvas is None:
        raise RuntimeError("Could not render Camera-D challenge image")
    for box, color, label in (
        (camera_d["expected_box_xyxy"], (255, 140, 0), "projected hull"),
        (camera_d["mask_box_xyxy"], (0, 165, 255), "visible semantic support"),
        (camera_d["best_box_xyxy"], (0, 0, 255), "YOLO box"),
    ):
        if box is None:
            continue
        p0 = (round(box[0]), round(box[1])); p1 = (round(box[2]), round(box[3]))
        cv2.rectangle(canvas, p0, p1, color, 3)
        cv2.putText(canvas, label, (p0[0], max(24, p0[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    verdict = "REFUSED" if passed else "ADMITTED"
    cv2.rectangle(canvas, (20, 20), (690, 94), (20, 20, 20), -1)
    cv2.putText(canvas, f"Camera D named sliver: {verdict}", (35, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 80) if passed else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, ", ".join(camera_d["gate_reasons"]) or "no refusal reason", (35, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    annotated = output / "camera_D_sliver_annotated.png"
    cv2.imwrite(str(annotated), canvas)

    report = {
        "schema": "thesis_stage06_named_sliver_diagnostic.v1",
        "status": "pass" if passed else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_timing": "post-selection diagnostic; did not alter gate thresholds",
        "capture": str(capture.relative_to(REPO)),
        "capture_manifest_sha256": sha256(capture / "capture_manifest.json"),
        "capture_index_sha256": sha256(capture / "capture_index.csv"),
        "gate_report": str(gate_report_path.relative_to(REPO)),
        "gate_report_sha256": sha256(gate_report_path),
        "checkpoint_sha256": sha256(checkpoint),
        "selected_gate": candidate,
        "camera_D_required_outcome": "not admitted as localization measurement",
        "camera_D_outcome_pass": passed,
        "cameras": evaluated,
        "annotated_image": annotated.name,
        "annotated_image_sha256": sha256(annotated),
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
