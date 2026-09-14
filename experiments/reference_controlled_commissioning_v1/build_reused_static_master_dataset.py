#!/usr/bin/env python3
"""Build an analysis-ready static commissioning master without copying images.

The source master capture is immutable.  This builder joins its capture ledger to the
frozen-detector outputs, applies only the declared v2 sensor gate, and records the current
geometry-only exclusions.  Detector partitions remain present only in a separate catalog.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
CAPTURE = REPO / "logs/thesis_final_pipeline_v1/master_capture"
CAPTURE_INDEX = CAPTURE / "capture_index.csv"
CAPTURE_MANIFEST = CAPTURE / "capture_manifest.json"
COMMISSIONING_INFERENCE = (
    REPO / "logs/thesis_final_pipeline_v1/stage06_detector_gate/commissioning_inference"
)
AUDIT_INFERENCE = (
    REPO / "logs/thesis_final_pipeline_v1/stage09_navigation/final_audit_inference"
)
GATE = REPO / "config/commissioning_sensor_gate_v2.yaml"
GEOMETRY_PROTOCOL = Path(__file__).resolve().parent / "static_incremental_campaign_v1.json"
PIXEL_AUDIT = REPO / "logs/thesis_final_pipeline_v1/stage04_dataset/capture_pixel_audit.json"

DETECTOR_ROLES = {"detector_fit", "detector_validation"}
STATIC_SOURCE_ROLES = {"commissioning_fit", "final_audit"}
CAMERAS = {f"camera_{letter}" for letter in "ABCDE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["pose_id"]), str(row["camera_id"])


def finite_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(math.isfinite(float(item)) for item in value)
    )


def apply_gate(record: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    box = record.get("best_box_xyxy")
    score = float(record.get("raw_best_confidence") or 0.0)
    if not isinstance(box, list) or len(box) != 4:
        return False, ["no_detection"]
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError):
        return False, ["nonfinite_box"]
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        reasons.append("nonfinite_box")
    if score < float(gate["confidence_threshold"]):
        reasons.append("confidence")
    if x1 - x0 < float(gate["min_bbox_width_px"]):
        reasons.append("bbox_width")
    if y1 - y0 < float(gate["min_bbox_height_px"]):
        reasons.append("bbox_height")
    margin = float(gate["min_edge_distance_px"])
    width = float(gate["image_width_px"])
    height = float(gate["image_height_px"])
    if x0 < margin or y0 < margin or x1 > width - margin or y1 > height - margin:
        reasons.append("full_bbox_edge_margin")
    if not finite_pair(record.get("raw_ground_xy")):
        reasons.append("invalid_projection")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO
            / "logs/studies/reference_controlled_commissioning_v1"
            / "reused_static_master_dataset_20260914_v1"
        ),
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing dataset: {output}")
    output.mkdir(parents=True)

    capture_manifest = read_json(CAPTURE_MANIFEST)
    geometry = read_json(GEOMETRY_PROTOCOL)
    pixel_audit = read_json(PIXEL_AUDIT)
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("source master capture is not complete")
    if capture_manifest.get("capture_index_sha256") != sha256(CAPTURE_INDEX):
        raise RuntimeError("source capture index hash does not match its manifest")
    if pixel_audit.get("passed") is not True:
        raise RuntimeError("locked decoded-pixel audit did not pass")
    if gate.get("gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("unexpected sensor gate")

    with CAPTURE_INDEX.open(newline="", encoding="utf-8") as handle:
        capture_rows = list(csv.DictReader(handle))
    if len(capture_rows) != 16000:
        raise RuntimeError(f"expected 16000 source rows, found {len(capture_rows)}")
    capture_by_key = {key(row): row for row in capture_rows}
    if len(capture_by_key) != len(capture_rows):
        raise RuntimeError("source master contains duplicate pose/camera identities")

    commissioning_records = read_jsonl(COMMISSIONING_INFERENCE / "records.jsonl")
    audit_records = read_jsonl(AUDIT_INFERENCE / "records.jsonl")
    inference_records = commissioning_records + audit_records
    inference_by_key = {key(row): row for row in inference_records}
    if len(inference_by_key) != 11200 or len(inference_records) != 11200:
        raise RuntimeError("expected exactly 11200 unique static inference records")

    excluded = geometry["legacy_exclusions"]
    excluded_ids = {int(row["position_id"]) for row in excluded}
    if len(excluded_ids) != 10:
        raise RuntimeError("expected exactly ten geometry-only legacy exclusions")

    detector_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    excluded_opportunities: list[dict[str, Any]] = []
    source_missing: list[str] = []
    for capture_row in capture_rows:
        role = capture_row["dataset_split"]
        image_path = CAPTURE / capture_row["image"]
        mask_path = CAPTURE / capture_row["robot_mask"]
        if not image_path.is_file():
            source_missing.append(str(image_path))
        if not mask_path.is_file():
            source_missing.append(str(mask_path))
        base = dict(capture_row)
        base.update({
            "source_capture_root": str(CAPTURE),
            "source_image_path": str(image_path),
            "source_mask_path": str(mask_path),
            "source_role": role,
        })
        if role in DETECTOR_ROLES:
            base["master_family"] = "detector"
            base["analysis_role"] = role
            detector_rows.append(base)
            continue
        if role not in STATIC_SOURCE_ROLES:
            raise RuntimeError(f"unexpected source role {role}")
        inferred = inference_by_key.get(key(capture_row))
        if inferred is None:
            raise RuntimeError(f"missing inference record for {key(capture_row)}")
        if inferred["image_sha1"] != capture_row["image_sha1"]:
            raise RuntimeError(f"image identity mismatch for {key(capture_row)}")
        admitted, reasons = apply_gate(inferred, gate)
        row = dict(base)
        for name, value in inferred.items():
            if name not in row:
                row[name] = value
        box = inferred.get("best_box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            for name, value in zip(("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"), box):
                row[name] = value
        else:
            for name in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
                row[name] = ""
        ground = inferred.get("raw_ground_xy")
        row["raw_projected_x"] = ground[0] if finite_pair(ground) else ""
        row["raw_projected_y"] = ground[1] if finite_pair(ground) else ""
        row["detector_hit"] = int(bool(inferred.get("detector_return_at_0.25")))
        row["sensor_gate_v2_admitted"] = int(admitted)
        row["sensor_gate_v2_reasons"] = ";".join(reasons)
        row["master_family"] = "static_commissioning"
        row["analysis_role"] = "static_commissioning_development"
        row["original_final_audit_previously_accessed"] = int(role == "final_audit")
        position_id = int(capture_row["position_id"])
        row["geometry_eligible"] = int(position_id not in excluded_ids)
        if position_id in excluded_ids:
            excluded_opportunities.append(row)
        else:
            static_rows.append(row)

    if source_missing:
        raise RuntimeError(f"missing source files, first: {source_missing[:3]}")

    exclusions_by_id = {int(row["position_id"]): row for row in excluded}
    exclusion_rows = [exclusions_by_id[position_id] for position_id in sorted(exclusions_by_id)]
    detector_fields = list(capture_rows[0]) + [
        "source_capture_root", "source_image_path", "source_mask_path", "source_role",
        "master_family", "analysis_role",
    ]
    extra_static_fields = [
        "source_capture_root", "source_image_path", "source_mask_path", "source_role",
        "master_family", "analysis_role", "geometry_eligible",
        "original_final_audit_previously_accessed", "reference_class", "reference_reasons",
        "raw_best_confidence", "candidate_count_above_0.001", "detector_hit",
        "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "raw_projected_x", "raw_projected_y", "raw_ground_error_m",
        "sensor_gate_v2_admitted", "sensor_gate_v2_reasons",
    ]
    static_fields = list(capture_rows[0]) + [
        field for field in extra_static_fields if field not in capture_rows[0]
    ]
    write_csv(output / "detector_partitions.csv", detector_rows, detector_fields)
    write_csv(output / "camera_opportunities.csv", static_rows, static_fields)
    write_csv(output / "excluded_static_opportunities.csv", excluded_opportunities, static_fields)
    exclusion_fields = sorted({field for row in exclusion_rows for field in row})
    write_csv(output / "excluded_static_positions.csv", exclusion_rows, exclusion_fields)
    measurement_rows = [row for row in static_rows if row["sensor_gate_v2_admitted"] == 1]
    write_csv(output / "camera_measurements.csv", measurement_rows, static_fields)

    positions_by_role: dict[str, set[int]] = defaultdict(set)
    for row in capture_rows:
        positions_by_role[row["dataset_split"]].add(int(row["position_id"]))
    static_positions = {int(row["position_id"]) for row in static_rows}
    static_batches: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in static_rows:
        static_batches[(int(row["pose_id"]), int(row["repetition_id"]))].add(row["camera_id"])
    counts = {
        "source_positions": len({int(row["position_id"]) for row in capture_rows}),
        "source_pose_batches": len({int(row["pose_id"]) for row in capture_rows}),
        "source_camera_opportunities": len(capture_rows),
        "detector_positions": sum(len(positions_by_role[role]) for role in DETECTOR_ROLES),
        "detector_camera_opportunities": len(detector_rows),
        "static_source_positions": len(
            positions_by_role["commissioning_fit"] | positions_by_role["final_audit"]
        ),
        "static_valid_positions": len(static_positions),
        "static_excluded_positions": len(excluded_ids),
        "static_valid_pose_batches": len(static_batches),
        "static_camera_opportunities": len(static_rows),
        "static_admitted_measurements_v2": len(measurement_rows),
        "static_detector_hits": sum(int(row["detector_hit"]) for row in static_rows),
    }
    checks = {
        "source_capture_complete": capture_manifest.get("status") == "complete",
        "source_index_hash_matches": capture_manifest.get("capture_index_sha256") == sha256(CAPTURE_INDEX),
        "locked_pixel_audit_passed": pixel_audit.get("passed") is True,
        "all_source_files_exist": not source_missing,
        "source_has_400_positions": counts["source_positions"] == 400,
        "source_has_3200_pose_batches": counts["source_pose_batches"] == 3200,
        "source_has_16000_opportunities": counts["source_camera_opportunities"] == 16000,
        "detector_roles_isolated": len(detector_rows) == 4800,
        "static_has_270_valid_positions": counts["static_valid_positions"] == 270,
        "static_has_2160_pose_batches": counts["static_valid_pose_batches"] == 2160,
        "static_has_10800_opportunities": len(static_rows) == 10800,
        "ten_positions_excluded_by_geometry": len(excluded_ids) == 10 and len(excluded_opportunities) == 400,
        "five_cameras_per_static_batch": all(cameras == CAMERAS for cameras in static_batches.values()),
        "all_static_inference_records_accounted": len(static_rows) + len(excluded_opportunities) == 11200,
    }

    sources = {
        "capture_manifest": {"path": str(CAPTURE_MANIFEST), "sha256": sha256(CAPTURE_MANIFEST)},
        "capture_index": {"path": str(CAPTURE_INDEX), "sha256": sha256(CAPTURE_INDEX)},
        "commissioning_inference_manifest": {
            "path": str(COMMISSIONING_INFERENCE / "manifest.json"),
            "sha256": sha256(COMMISSIONING_INFERENCE / "manifest.json"),
        },
        "commissioning_inference_records": {
            "path": str(COMMISSIONING_INFERENCE / "records.jsonl"),
            "sha256": sha256(COMMISSIONING_INFERENCE / "records.jsonl"),
        },
        "previous_final_audit_inference_manifest": {
            "path": str(AUDIT_INFERENCE / "manifest.json"),
            "sha256": sha256(AUDIT_INFERENCE / "manifest.json"),
        },
        "previous_final_audit_inference_records": {
            "path": str(AUDIT_INFERENCE / "records.jsonl"),
            "sha256": sha256(AUDIT_INFERENCE / "records.jsonl"),
        },
        "geometry_protocol": {"path": str(GEOMETRY_PROTOCOL), "sha256": sha256(GEOMETRY_PROTOCOL)},
        "sensor_gate_v2": {"path": str(GATE), "sha256": sha256(GATE)},
        "locked_pixel_audit": {"path": str(PIXEL_AUDIT), "sha256": sha256(PIXEL_AUDIT)},
    }
    manifest = {
        "schema": "reused_static_commissioning_master.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(checks.values()) else "invalid",
        "purpose": "Analysis-ready static commissioning master reusing the immutable 400-position capture.",
        "storage_policy": "No RGB or mask files are copied; rows retain source paths and pixel hashes.",
        "partition_policy": {
            "detector_fit_and_validation": "isolated in detector_partitions.csv and not used for commissioning fits",
            "static_commissioning": "commissioning_fit plus the previously accessed legacy final_audit, reclassified as development",
            "dynamic_commissioning": "not included; continuous drives remain a separate dataset and independent unit",
            "current_sealed_audit": "none",
        },
        "geometry_policy": {
            "eligible": "complete 0.8 x 0.55 m body clears the current full collision scene by at least 0.05 m at all eight headings",
            "excluded_positions": 10,
            "exclusions_are_outcome_independent": True,
        },
        "gate_policy": {
            "gate_id": gate["gate_id"],
            "uses_only_runtime_observable_detector_box_fields": True,
            "includes_nis": False,
        },
        "sources": sources,
        "outputs": {},
        "counts": counts,
        "checks": checks,
    }
    for name in (
        "detector_partitions.csv", "camera_opportunities.csv", "camera_measurements.csv",
        "excluded_static_opportunities.csv", "excluded_static_positions.csv",
    ):
        manifest["outputs"][name] = {"sha256": sha256(output / name)}
    atomic_json(output / "master_manifest.json", manifest)
    print(json.dumps({"output": str(output), "status": manifest["status"], "counts": counts}, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
