#!/usr/bin/env python3
"""Evaluate the sealed Stage-09 audit without fitting or selecting any model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from select_stage06_gate import gate_reasons
from reliability.commissioned_availability import CommissionedAvailabilityModel
from reliability.commissioned_measurement import CommissionedMeasurementModel
from reliability.projection import camera_model_from_world


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CHI2_DF2 = {0.50: 1.3862943611198906, 0.90: 4.605170185988092,
            0.95: 5.991464547107979, 0.99: 9.21034037197618}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def error_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median_m": None, "rms_m": None, "p90_m": None,
                "p95_m": None, "max_m": None, "above_0_25_m": 0,
                "above_0_25_fraction": None}
    array = np.asarray(values, dtype=float)
    above = int(np.sum(array > 0.25))
    return {
        "n": len(values), "median_m": float(np.median(array)),
        "rms_m": float(np.sqrt(np.mean(array ** 2))),
        "p90_m": percentile(values, 0.90), "p95_m": percentile(values, 0.95),
        "max_m": float(np.max(array)), "above_0_25_m": above,
        "above_0_25_fraction": above / len(values),
    }


def probability_metrics(targets: list[int], probabilities: list[float]) -> dict:
    # Import the repository's single canonical implementations, never local copies.
    import sys
    shared = str(REPO / "scripts/shared")
    if shared not in sys.path:
        sys.path.insert(0, shared)
    import metrics as M

    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    return {
        "n": len(targets), "positive": int(np.sum(y)),
        "brier": M.brier(y, p), "logloss": M.logloss(y, p),
        "ece_10": M.ece(y, p, bins=10), "auprc": M.auprc(y, p),
        "mean_prediction": float(np.mean(p)), "observed_rate": float(np.mean(y)),
    }


def detector_metrics(records: list[dict]) -> dict:
    # Match the Stage-05 convention: a positive requires IoU >= 0.5.  A returned
    # but unmatched box on a positive contributes both one FP and one FN.
    eligible = [r for r in records if r["reference_class"] in ("positive", "negative")]
    tp = fp = fn = tn = 0
    for record in eligible:
        positive = record["reference_class"] == "positive"
        returned = bool(record["detector_return_at_0.25"])
        matched = returned and record["iou_with_mask_box"] is not None and record["iou_with_mask_box"] >= 0.5
        if positive and matched:
            tp += 1
        elif positive:
            fn += 1
            fp += int(returned)
        elif returned:
            fp += 1
        else:
            tn += 1
    return {
        "definition": "IoU>=0.5 at confidence>=0.25; ambiguous views excluded",
        "eligible_views": len(eligible), "reference_positive": sum(r["reference_class"] == "positive" for r in eligible),
        "reference_negative": sum(r["reference_class"] == "negative" for r in eligible),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "true_negative_views": tn, "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def subset_metrics(rows: list[dict]) -> dict:
    admitted = [row for row in rows if row["admitted"]]
    true_admitted = sum(row["reference_class"] == "positive" for row in admitted)
    false_admitted = len(admitted) - true_admitted
    positives = sum(row["reference_class"] == "positive" for row in rows)
    errors = [float(row["corrected_error_m"]) for row in admitted if row["corrected_error_m"] is not None]
    d2 = [float(row["mahalanobis_d2"]) for row in admitted if row["mahalanobis_d2"] is not None]
    covariance = {
        "n": len(d2),
        "containment": {
            str(int(level * 100)): (sum(value <= threshold for value in d2) / len(d2) if d2 else None)
            for level, threshold in CHI2_DF2.items()
        },
        "above_chi2_99_fraction": (sum(value > CHI2_DF2[0.99] for value in d2) / len(d2) if d2 else None),
    }
    return {
        "views": len(rows), "reference_positive": positives, "admitted": len(admitted),
        "true_admitted": true_admitted, "false_admitted": false_admitted,
        "precision": true_admitted / len(admitted) if admitted else 1.0,
        "reference_recall": true_admitted / positives if positives else 0.0,
        "corrected_error": error_summary(errors), "covariance": covariance,
        "availability": probability_metrics(
            [int(row["admitted"]) for row in rows],
            [float(row["availability_probability"]) for row in rows],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_final_audit_access":
        raise RuntimeError("release protocol is not frozen")
    inputs = protocol["locked_inputs"]
    inference = args.inference.resolve()
    manifest_path = inference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records_path = inference / manifest["records"]
    if manifest.get("status") != "complete" or manifest.get("authorized_role") != "final_audit":
        raise RuntimeError("inference is not a complete authorized final audit")
    if manifest.get("protocol_sha256") != sha256(protocol_path):
        raise RuntimeError("inference used a different release protocol")
    if manifest.get("records_sha256") != sha256(records_path):
        raise RuntimeError("audit inference records drift")
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if len(records) != int(protocol["final_audit"]["views"]):
        raise RuntimeError("unexpected number of final-audit inference records")

    gate_path = REPO / inputs["gate_report"]
    measurement_path = REPO / inputs["measurement_model"]
    availability_path = REPO / inputs["availability_model"]
    for path, expected in (
        (gate_path, inputs["gate_report_sha256"]),
        (measurement_path, inputs["measurement_model_sha256"]),
        (availability_path, inputs["availability_model_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"locked artifact drift: {path}")
    candidate = json.loads(gate_path.read_text())["selected"]["candidate"]
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    with (REPO / protocol["final_audit"]["master_capture"] / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    model_names = {
        camera: next(row["camera_model"] for row in index_rows if row["camera_id"] == camera)
        for camera in CAMERAS
    }
    cameras = {camera: camera_model_from_world(world, include_name=model_names[camera]) for camera in CAMERAS}
    measurement = CommissionedMeasurementModel(measurement_path, expected_sha256=inputs["measurement_model_sha256"])
    availability = CommissionedAvailabilityModel(
        availability_path, world, expected_sha256=inputs["availability_model_sha256"]
    )

    evaluated: list[dict] = []
    refusal_reasons: Counter[str] = Counter()
    for record in records:
        reasons = gate_reasons(record, candidate)
        refusal_reasons.update(reasons)
        admitted = not reasons
        camera = cameras[record["camera_id"]]
        q = availability.probability(
            record["camera_id"], record["robot_x"], record["robot_y"], record["robot_yaw"], camera
        )
        corrected = covariance = None
        if admitted:
            result = measurement.apply(
                record["camera_id"], record["raw_ground_xy"],
                record["silhouette_equivalent_xy_exact_prior"], record["best_box_xyxy"],
                record["expected_box_xyxy"], record["raw_best_confidence"],
                (record["robot_x"], record["robot_y"]), record["robot_yaw"], camera,
            )
            if result is None:
                raise RuntimeError("Stage-07 runtime refused an observation admitted by the frozen gate")
            corrected, covariance = result
        error = d2 = None
        if corrected is not None:
            residual = np.asarray(corrected, dtype=float) - np.asarray((record["robot_x"], record["robot_y"]), dtype=float)
            matrix = np.asarray(covariance, dtype=float)
            error = float(np.linalg.norm(residual))
            d2 = float(residual @ np.linalg.solve(matrix, residual))
        evaluated.append({
            **record, "gate_reasons": reasons, "admitted": admitted,
            "availability_probability": q,
            "corrected_xy": None if corrected is None else list(corrected),
            "covariance_m2": None if covariance is None else [list(row) for row in covariance],
            "corrected_error_m": error, "mahalanobis_d2": d2,
        })

    detector = detector_metrics(records)
    pooled = subset_metrics(evaluated)
    by_camera = {camera: subset_metrics([row for row in evaluated if row["camera_id"] == camera]) for camera in CAMERAS}
    gates = protocol["release_gates"]
    checks = {
        "detector_precision": detector["precision"] >= gates["detector_precision_positive_vs_negative_min"],
        "detector_recall": detector["recall"] >= gates["detector_recall_positive_vs_negative_min"],
        "localization_admission_precision": pooled["precision"] >= gates["localization_admission_precision_min"],
        "localization_admission_reference_recall": pooled["reference_recall"] >= gates["localization_admission_reference_recall_min"],
        "corrected_planar_p95": pooled["corrected_error"]["p95_m"] <= gates["corrected_planar_p95_m_max"],
        "corrected_above_0_25_fraction": pooled["corrected_error"]["above_0_25_fraction"] <= gates["corrected_above_0_25_fraction_max"],
        "covariance_95_containment_min": pooled["covariance"]["containment"]["95"] >= gates["covariance_95_containment_min"],
        "covariance_95_containment_max": pooled["covariance"]["containment"]["95"] <= gates["covariance_95_containment_max"],
        "mahalanobis_above_99_fraction": pooled["covariance"]["above_chi2_99_fraction"] <= gates["mahalanobis_above_99_fraction_max"],
        "availability_brier": pooled["availability"]["brier"] <= gates["availability_brier_max"],
        "availability_ece_10": pooled["availability"]["ece_10"] <= gates["availability_ece_10_max"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records_output = output.with_name("final_audit_evaluated.jsonl")
    if output.exists() or records_output.exists():
        raise FileExistsError("refusing to overwrite an existing final-audit evaluation")
    with records_output.open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    report = {
        "schema": "thesis_stage09_final_audit_report.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "final_audit_accessed": True,
        "selection_or_fitting_performed": False, "release_pass": all(checks.values()),
        "release_checks": checks, "release_thresholds": gates,
        "detector": detector, "localization_chain": pooled, "by_camera": by_camera,
        "refusal_reason_counts": dict(sorted(refusal_reasons.items())),
        "protocol": str(protocol_path.relative_to(REPO)), "protocol_sha256": sha256(protocol_path),
        "inference_manifest_sha256": sha256(manifest_path), "inference_records_sha256": sha256(records_path),
        "evaluated_records": records_output.name, "evaluated_records_sha256": sha256(records_output),
        "locked_artifact_sha256": {
            "gate": sha256(gate_path), "measurement": measurement.sha256,
            "availability": availability.sha256,
        },
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["release_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
