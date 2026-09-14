#!/usr/bin/env python3
"""Select the Stage-06 localization-admission gate on spatial commissioning blocks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import equivalent_position_measurement
from unav_common.robot_hull import VISUAL_HULL, silhouette_box


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def error_summary(values) -> dict:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "median_m": None, "rms_m": None, "p90_m": None, "p95_m": None, "max_m": None}
    return {
        "n": len(finite),
        "median_m": float(statistics.median(finite)),
        "rms_m": float(math.sqrt(sum(value * value for value in finite) / len(finite))),
        "p90_m": percentile(finite, 0.90),
        "p95_m": percentile(finite, 0.95),
        "max_m": max(finite),
    }


def fold_map(records: list[dict]) -> dict[str, int]:
    blocks = sorted({record["block_id"] for record in records})
    return {block: index % 5 for index, block in enumerate(blocks)}


def candidate_dict(confidence: float, side: int, ratio: float, bottom_fraction: float) -> dict:
    return {
        "confidence_min": float(confidence),
        "minimum_box_side_px": int(side),
        "minimum_shape_ratio": float(ratio),
        "maximum_shape_ratio": float(1.0 / ratio),
        "bottom_shortfall_fraction": float(bottom_fraction),
        "border_margin_px": 2,
    }


def gate_reasons(
    record: dict,
    candidate: dict,
    *,
    predicted_box: list[float] | tuple[float, float, float, float] | None = None,
    equivalent_available: bool | None = None,
) -> list[str]:
    reasons: list[str] = []
    box = record["best_box_xyxy"]
    if box is None:
        return ["no_detection"]
    if float(record["raw_best_confidence"]) < candidate["confidence_min"]:
        reasons.append("confidence")
    width, height = float(box[2] - box[0]), float(box[3] - box[1])
    if width < candidate["minimum_box_side_px"] or height < candidate["minimum_box_side_px"]:
        reasons.append("minimum_box_side")
    margin = float(candidate["border_margin_px"])
    if box[0] < margin or box[1] < margin or box[2] >= 1280.0 - margin or box[3] >= 720.0 - margin:
        reasons.append("touches_border")
    predicted = predicted_box if predicted_box is not None else record["expected_box_xyxy"]
    if predicted is None:
        reasons.append("no_hull_prediction")
    else:
        predicted_width = float(predicted[2] - predicted[0])
        predicted_height = float(predicted[3] - predicted[1])
        if predicted_width <= 0.0 or predicted_height <= 0.0:
            reasons.append("degenerate_hull_prediction")
        else:
            ratios = (width / predicted_width, height / predicted_height)
            if any(
                ratio < candidate["minimum_shape_ratio"]
                or ratio > candidate["maximum_shape_ratio"]
                for ratio in ratios
            ):
                reasons.append("shape_ratio")
            bottom_shortfall = float(predicted[3] - box[3])
            maximum_shortfall = max(6.0, candidate["bottom_shortfall_fraction"] * predicted_height)
            if bottom_shortfall > maximum_shortfall:
                reasons.append("bottom_shortfall")
    if record["raw_ground_xy"] is None:
        reasons.append("projection_unavailable")
    if equivalent_available is None:
        equivalent_available = record["silhouette_equivalent_xy_exact_prior"] is not None
    if not equivalent_available:
        reasons.append("equivalent_measurement_unavailable")
    return reasons


def metrics(records: list[dict], candidate: dict, folds: dict[str, int]) -> tuple[dict, list[tuple[bool, list[str]]]]:
    outcomes = []
    for record in records:
        reasons = gate_reasons(record, candidate)
        outcomes.append((not reasons, reasons))

    def subset_summary(indexes: list[int]) -> dict:
        reference = [records[index]["reference_class"] == "positive" for index in indexes]
        admitted = [outcomes[index][0] for index in indexes]
        tp = sum(keep and target for keep, target in zip(admitted, reference, strict=True))
        fp = sum(keep and not target for keep, target in zip(admitted, reference, strict=True))
        fn = sum(not keep and target for keep, target in zip(admitted, reference, strict=True))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        errors = [
            records[index]["silhouette_equivalent_error_m_exact_prior"]
            for index in indexes if outcomes[index][0]
        ]
        catastrophic = sum(
            error is not None and error > 0.25 for error in errors
        )
        return {
            "views": len(indexes),
            "reference_positive": sum(reference),
            "admitted": sum(admitted),
            "true_admitted": tp,
            "false_admitted": fp,
            "refused_reference_positive": fn,
            "precision": precision,
            "recall": recall,
            "catastrophic_admitted": catastrophic,
            "catastrophic_rate": catastrophic / max(sum(admitted), 1),
            "corrected_error": error_summary(errors),
        }

    all_indexes = list(range(len(records)))
    pooled = subset_summary(all_indexes)
    by_fold = {
        str(fold): subset_summary([
            index for index, record in enumerate(records) if folds[record["block_id"]] == fold
        ])
        for fold in range(5)
    }
    by_camera = {
        camera: subset_summary([
            index for index, record in enumerate(records) if record["camera_id"] == camera
        ])
        for camera in CAMERAS
    }
    zero_indexes = [
        index for index, record in enumerate(records) if int(record["semantic_robot_pixels"]) == 0
    ]
    zero_false = sum(outcomes[index][0] for index in zero_indexes)
    ambiguous_indexes = [
        index for index, record in enumerate(records) if record["reference_class"] == "ambiguous"
    ]
    ambiguous_false = sum(outcomes[index][0] for index in ambiguous_indexes)
    reason_counts = Counter(reason for _keep, reasons in outcomes for reason in reasons)
    return ({
        "candidate": candidate,
        "pooled": pooled,
        "by_fold": by_fold,
        "by_camera": by_camera,
        "minimum_fold_precision": min(value["precision"] for value in by_fold.values()),
        "minimum_fold_recall": min(value["recall"] for value in by_fold.values()),
        "zero_mask_views": len(zero_indexes),
        "false_admitted_zero_mask": zero_false,
        "false_admission_rate_zero_mask": zero_false / max(len(zero_indexes), 1),
        "ambiguous_views": len(ambiguous_indexes),
        "false_admitted_ambiguous": ambiguous_false,
        "refusal_reason_counts": dict(sorted(reason_counts.items())),
    }, outcomes)


def feasible(result: dict, protocol: dict) -> tuple[bool, list[str]]:
    constraints = protocol["feasibility_constraints"]
    failures = []
    if result["pooled"]["precision"] < constraints["pooled_precision_min"]:
        failures.append("pooled_precision")
    if result["minimum_fold_precision"] < constraints["worst_spatial_fold_precision_min"]:
        failures.append("worst_fold_precision")
    if result["false_admission_rate_zero_mask"] > constraints["maximum_false_admission_rate_on_zero-mask_views"]:
        failures.append("zero_mask_false_admission")
    if result["pooled"]["catastrophic_rate"] > constraints["maximum_catastrophic_error_rate_among_admitted"]:
        failures.append("catastrophic_error")
    if result["pooled"]["recall"] < constraints["minimum_pooled_reference_recall"]:
        failures.append("reference_recall")
    return not failures, failures


def ranking(result: dict) -> tuple:
    candidate = result["candidate"]
    error = result["pooled"]["corrected_error"]["p95_m"]
    return (
        result["minimum_fold_recall"],
        result["pooled"]["recall"],
        -result["pooled"]["false_admitted"],
        -result["pooled"]["catastrophic_admitted"],
        -float("inf") if error is None else -error,
        -candidate["confidence_min"],
        -candidate["minimum_box_side_px"],
        -candidate["minimum_shape_ratio"],
        candidate["bottom_shortfall_fraction"],
    )


def sensitivity(records: list[dict], candidate: dict, camera_models: dict) -> dict:
    aggregate = Counter()
    corrected_errors = []
    by_draw = defaultdict(Counter)
    for record in records:
        camera = camera_models[record["camera_id"]]
        raw = record["raw_ground_xy"]
        reference = record["reference_class"] == "positive"
        for draw in range(4):
            key = (
                f"thesis-stage06-prior-sensitivity-v1|{record['pose_id']}|"
                f"{record['camera_id']}|{draw}"
            ).encode("ascii")
            seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            dx, dy = np.clip(rng.normal(0.0, 0.10, size=2), -0.20, 0.20)
            dyaw = float(np.clip(rng.normal(0.0, math.radians(5.0)), -math.radians(10.0), math.radians(10.0)))
            prior = (record["robot_x"] + float(dx), record["robot_y"] + float(dy))
            yaw = record["robot_yaw"] + dyaw
            predicted = silhouette_box(camera, prior[0], prior[1], yaw, VISUAL_HULL)
            equivalent = None
            if raw is not None:
                converted = equivalent_position_measurement(
                    (float(raw[0]), float(raw[1])),
                    ((1.0, 0.0), (0.0, 1.0)),
                    camera,
                    prior,
                    yaw,
                )
                if converted is not None:
                    equivalent = converted[0]
            reasons = gate_reasons(
                record,
                candidate,
                predicted_box=predicted,
                equivalent_available=equivalent is not None,
            )
            admitted = not reasons
            aggregate["opportunities"] += 1
            aggregate["reference_positive"] += int(reference)
            aggregate["admitted"] += int(admitted)
            aggregate["tp"] += int(admitted and reference)
            aggregate["fp"] += int(admitted and not reference)
            by_draw[draw]["reference_positive"] += int(reference)
            by_draw[draw]["admitted"] += int(admitted)
            by_draw[draw]["tp"] += int(admitted and reference)
            by_draw[draw]["fp"] += int(admitted and not reference)
            if admitted and equivalent is not None:
                error = math.hypot(
                    float(equivalent[0]) - record["robot_x"],
                    float(equivalent[1]) - record["robot_y"],
                )
                corrected_errors.append(error)
                aggregate["catastrophic"] += int(error > 0.25)
                by_draw[draw]["catastrophic"] += int(error > 0.25)
    precision = aggregate["tp"] / max(aggregate["tp"] + aggregate["fp"], 1)
    recall = aggregate["tp"] / max(aggregate["reference_positive"], 1)
    catastrophic_rate = aggregate["catastrophic"] / max(aggregate["admitted"], 1)
    return {
        "interpretation": "Deterministic prior-error sensitivity observations; draws are not independent experimental replicates.",
        "counts": dict(aggregate),
        "precision": precision,
        "recall": recall,
        "catastrophic_rate": catastrophic_rate,
        "corrected_error": error_summary(corrected_errors),
        "by_draw": {
            str(draw): {
                **dict(counts),
                "precision": counts["tp"] / max(counts["tp"] + counts["fp"], 1),
                "recall": counts["tp"] / max(counts["reference_positive"], 1),
                "catastrophic_rate": counts["catastrophic"] / max(counts["admitted"], 1),
            }
            for draw, counts in sorted(by_draw.items())
        },
        "advance_pass": precision >= 0.97 and catastrophic_rate <= 0.02,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inference = args.inference.resolve()
    inference_manifest_path = inference / "manifest.json"
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if inference_manifest.get("status") != "complete":
        raise RuntimeError("Commissioning inference is incomplete")
    if inference_manifest["protocol_sha256"] != sha256(protocol_path):
        raise RuntimeError("Stage-06 protocol hash mismatch")
    records_path = inference / inference_manifest["records"]
    if inference_manifest["records_sha256"] != sha256(records_path):
        raise RuntimeError("Commissioning record hash mismatch")
    with records_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != int(protocol["inputs"]["authorized_views"]):
        raise RuntimeError("Commissioning population size mismatch")

    folds = fold_map(records)
    grid = protocol["candidate_grid"]
    candidates = [
        candidate_dict(confidence, side, ratio, bottom)
        for confidence, side, ratio, bottom in itertools.product(
            grid["confidence_min"],
            grid["minimum_box_side_px"],
            grid["minimum_detected_to_predicted_shape_ratio"],
            (0.05, 0.10, 0.15),
        )
    ]
    if len(candidates) != int(grid["candidate_count"]):
        raise RuntimeError("Candidate-grid count drift")
    results = []
    outcome_by_key = {}
    for candidate in candidates:
        result, outcomes = metrics(records, candidate, folds)
        result["feasible"], result["feasibility_failures"] = feasible(result, protocol)
        results.append(result)
        outcome_by_key[json.dumps(candidate, sort_keys=True)] = outcomes
    eligible = [result for result in results if result["feasible"]]
    selected = max(eligible, key=ranking) if eligible else None

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    report = {
        "schema": "thesis_stage06_gate_selection.v1",
        "status": "pass" if selected is not None else "failed_no_feasible_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "population": "commissioning_fit only",
        "experimental_unit": "spatial block; frame-level counts retained only as opportunity accounting",
        "final_audit_accessed": False,
        "protocol": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256(protocol_path),
        "inference_manifest": str(inference_manifest_path.relative_to(REPO)),
        "inference_manifest_sha256": sha256(inference_manifest_path),
        "fold_assignment": folds,
        "candidate_count": len(results),
        "feasible_candidate_count": len(eligible),
        "selected": selected,
        "all_candidates": results,
    }

    if selected is not None:
        world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
        include_names = {
            "camera_A": "external_camera",
            "camera_B": "external_camera_b",
            "camera_C": "external_camera_c",
            "camera_D": "external_camera_d",
            "camera_E": "external_camera_e",
        }
        camera_models = {
            camera: camera_model_from_world(world, include_name=include)
            for camera, include in include_names.items()
        }
        report["prior_sensitivity"] = sensitivity(records, selected["candidate"], camera_models)
        report["advance_to_stage07"] = bool(report["prior_sensitivity"]["advance_pass"])
        outcomes = outcome_by_key[json.dumps(selected["candidate"], sort_keys=True)]
        csv_path = staging / "admission_records.csv"
        fields = [
            "pose_id", "position_id", "position_key", "block_id", "fold", "heading_id",
            "camera_id", "source_batch_id", "reference_class", "reference_reasons",
            "semantic_robot_pixels", "raw_best_confidence", "detector_return_at_0_25",
            "best_box_x0", "best_box_y0", "best_box_x1", "best_box_y1",
            "expected_box_x0", "expected_box_y0", "expected_box_x1", "expected_box_y1",
            "admitted", "gate_reasons", "raw_ground_x", "raw_ground_y",
            "equivalent_x", "equivalent_y", "robot_x", "robot_y", "robot_yaw",
            "camera_range_m", "raw_ground_error_m", "equivalent_error_m",
            "visible_bbox_width_px", "visible_bbox_height_px", "visible_width_ratio",
            "visible_height_ratio", "bottom_edge_gap_px",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record, (admitted, reasons) in zip(records, outcomes, strict=True):
                box = record["best_box_xyxy"] or [None] * 4
                expected = record["expected_box_xyxy"] or [None] * 4
                raw = record["raw_ground_xy"] or [None] * 2
                equivalent = record["silhouette_equivalent_xy_exact_prior"] or [None] * 2
                metrics_ref = record["reference_metrics"]
                writer.writerow({
                    "pose_id": record["pose_id"], "position_id": record["position_id"],
                    "position_key": record["position_key"], "block_id": record["block_id"],
                    "fold": folds[record["block_id"]], "heading_id": record["heading_id"],
                    "camera_id": record["camera_id"], "source_batch_id": record["source_batch_id"],
                    "reference_class": record["reference_class"],
                    "reference_reasons": ";".join(record["reference_reasons"]),
                    "semantic_robot_pixels": record["semantic_robot_pixels"],
                    "raw_best_confidence": record["raw_best_confidence"],
                    "detector_return_at_0_25": int(record["detector_return_at_0.25"]),
                    "best_box_x0": box[0], "best_box_y0": box[1],
                    "best_box_x1": box[2], "best_box_y1": box[3],
                    "expected_box_x0": expected[0], "expected_box_y0": expected[1],
                    "expected_box_x1": expected[2], "expected_box_y1": expected[3],
                    "admitted": int(admitted), "gate_reasons": ";".join(reasons),
                    "raw_ground_x": raw[0], "raw_ground_y": raw[1],
                    "equivalent_x": equivalent[0], "equivalent_y": equivalent[1],
                    "robot_x": record["robot_x"], "robot_y": record["robot_y"],
                    "robot_yaw": record["robot_yaw"], "camera_range_m": record["camera_range_m"],
                    "raw_ground_error_m": record["raw_ground_error_m"],
                    "equivalent_error_m": record["silhouette_equivalent_error_m_exact_prior"],
                    "visible_bbox_width_px": metrics_ref.get("bbox_width_px"),
                    "visible_bbox_height_px": metrics_ref.get("bbox_height_px"),
                    "visible_width_ratio": metrics_ref.get("visible_width_ratio"),
                    "visible_height_ratio": metrics_ref.get("visible_height_ratio"),
                    "bottom_edge_gap_px": metrics_ref.get("bottom_edge_gap_px"),
                })
        report["admission_records"] = "admission_records.csv"
        report["admission_records_sha256"] = sha256(csv_path)
    else:
        report["advance_to_stage07"] = False

    report_path = staging / "gate_selection.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": report["status"],
        "report": "gate_selection.json",
        "report_sha256": sha256(report_path),
        "advance_to_stage07": report["advance_to_stage07"],
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(
        json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    print(json.dumps({
        "status": report["status"],
        "feasible_candidate_count": len(eligible),
        "selected": selected,
        "prior_sensitivity": report.get("prior_sensitivity"),
        "advance_to_stage07": report["advance_to_stage07"],
    }, indent=2, sort_keys=True))
    return 0 if report["advance_to_stage07"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
