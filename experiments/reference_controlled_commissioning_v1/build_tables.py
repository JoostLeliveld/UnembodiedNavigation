#!/usr/bin/env python3
"""Build synchronized opportunity, measurement, and network-round tables."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml
import numpy as np

from validate_protocol import HERE, REPO
from validate_protocol import validate as _validate_v1
from validate_protocol_v2 import validate as _validate_v2


def validate(protocol_path):
    """Dispatch to the validator matching the protocol's declared schema."""
    import yaml as _yaml

    schema = _yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8")).get(
        "schema_version"
    )
    if schema == "reference_controlled_commissioning.v2":
        return _validate_v2(Path(protocol_path))
    return _validate_v1(Path(protocol_path))


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _lerp_angle(first: float, second: float, fraction: float) -> float:
    delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return (first + fraction * delta + math.pi) % (2.0 * math.pi) - math.pi


def _reference_at(trace: list[dict[str, Any]], stamp_ns: int, max_gap_s: float) -> dict[str, Any] | None:
    stamps = [int(row["stamp_ns"]) for row in trace]
    upper = bisect_right(stamps, int(stamp_ns))
    if upper == 0 or upper >= len(trace):
        return None
    before, after = trace[upper - 1], trace[upper]
    t0, t1 = int(before["stamp_ns"]), int(after["stamp_ns"])
    gap_s = (t1 - t0) * 1e-9
    if gap_s <= 0.0 or gap_s > max_gap_s or not (t0 <= stamp_ns <= t1):
        return None
    fraction = (stamp_ns - t0) / (t1 - t0)
    p0, p1 = before["pose"], after["pose"]
    x = float(p0[0]) + fraction * (float(p1[0]) - float(p0[0]))
    y = float(p0[1]) + fraction * (float(p1[1]) - float(p0[1]))
    yaw = _lerp_angle(float(p0[2]), float(p1[2]), fraction)
    speed = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])) / gap_s
    nearest = before if fraction <= 0.5 else after
    return {
        "reference_x": x,
        "reference_y": y,
        "reference_yaw": yaw,
        "reference_speed_mps": speed,
        "commanded_linear_mps": nearest.get("linear_mps"),
        "commanded_angular_radps": nearest.get("angular_radps"),
        "controller_state": nearest.get("state", ""),
        "reference_bracket_gap_s": gap_s,
    }


def _camera_positions(protocol: dict[str, Any]) -> dict[str, tuple[float, float]]:
    sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common")]
    from reliability.projection import camera_model_from_world

    profile_payload = yaml.safe_load(
        (REPO / protocol["world"]["profiles_path"]).read_text(encoding="utf-8")
    )
    profile = profile_payload["worlds"][protocol["world"]["name"]]
    cameras = {}
    for camera_id, include_name in zip(
        profile["camera_ids"], profile["camera_model_includes"], strict=True
    ):
        model = camera_model_from_world(
            REPO / protocol["world"]["path"], include_name=include_name
        )
        cameras[str(camera_id)] = (float(model.cam_pos[0]), float(model.cam_pos[1]))
    return cameras


def _ray_residual(
    camera_xy: tuple[float, float],
    reference_xy: tuple[float, float],
    residual_xy: tuple[float, float],
) -> tuple[float, float]:
    dx = reference_xy[0] - camera_xy[0]
    dy = reference_xy[1] - camera_xy[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return math.nan, math.nan
    ux, uy = dx / norm, dy / norm
    along = residual_xy[0] * ux + residual_xy[1] * uy
    across = residual_xy[0] * (-uy) + residual_xy[1] * ux
    return along, across


def build_tables(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    preflight = validate(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "drive_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != preflight["protocol_sha256"]:
        raise RuntimeError(
            "drive manifest protocol SHA-256 does not match the requested protocol; "
            "do not relabel or pool a drive collected under another commissioning contract"
        )
    drive = manifest["drive"]
    if drive["partition"] == "audit" and protocol["status"] != "sensor_model_frozen_audit_authorized":
        raise RuntimeError("audit outcomes remain sealed until the complete sensor model is frozen")

    experiment_dirs = sorted((run_dir / "experiment_logs").glob("experiment_*"))
    if len(experiment_dirs) != 1:
        raise RuntimeError("expected exactly one experiment logger directory")
    experiment_dir = experiment_dirs[0]
    opportunity_path = experiment_dir / "camera_opportunities.jsonl"
    manager_path = run_dir / "manager_outcomes.jsonl"
    trace_path = run_dir / "reference_controller_trace.jsonl"
    identity_path = run_dir / "frame_identity.jsonl"
    for path in (opportunity_path, manager_path, trace_path, identity_path):
        if not path.is_file():
            raise RuntimeError(f"missing table source: {path}")

    trace = [row for row in _jsonl(trace_path) if row.get("pose") is not None]
    trace.sort(key=lambda row: int(row["stamp_ns"]))
    identities: dict[tuple[str, int], dict[str, Any]] = {}
    identity_duplicates = 0
    for row in _jsonl(identity_path):
        key = (str(row["camera_id"]), int(row["capture_stamp_ns"]))
        if key in identities:
            identity_duplicates += 1
        identities[key] = row

    mappings: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in _jsonl(manager_path):
        if row.get("status") != "camera_mapping":
            continue
        key = (
            str(row.get("camera_id", "")),
            int(row.get("capture_stamp_ns", -1)),
            str(row.get("source_frame_id", "")),
        )
        if key in mappings:
            raise RuntimeError(f"duplicate manager terminal mapping for {key}")
        mappings[key] = row

    camera_positions = _camera_positions(protocol)
    from reliability.observation_gates import (
        UsableObservationGateConfig,
        evaluate_sensor_gate,
    )

    gate_config = UsableObservationGateConfig.from_yaml(
        str(REPO / protocol["sensor_gate"]["config_path"])
    )
    gate_config.assert_belief_independent()
    seen_frames: set[tuple[str, int, str]] = set()
    opportunities: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    parked_excluded = 0
    by_round: dict[str, list[dict[str, Any]]] = {}
    max_gap = float(protocol["capture"]["maximum_reference_interpolation_gap_s"])
    for delivery in _jsonl(opportunity_path):
        if not delivery.get("valid_contract", False) or delivery.get("duplicate", False):
            continue
        observation = delivery["observation"]
        camera_id = str(observation["camera_id"])
        stamp_ns = int(observation["capture_stamp_ns"])
        source_frame_id = str(observation["source_frame_id"])
        key = (camera_id, stamp_ns, source_frame_id)
        if key in seen_frames:
            raise RuntimeError(f"duplicate physical camera frame in opportunity log: {key}")
        seen_frames.add(key)
        identity = identities.get((camera_id, stamp_ns))
        reference = _reference_at(trace, stamp_ns, max_gap)
        mapping = mappings.get(key)
        bbox = observation.get("bbox_xyxy")
        # A detector hit is any returned candidate box. The frozen confidence
        # threshold belongs to the sensor gate below, so a low-confidence box
        # remains distinguishable from a detector miss.
        box_returned = bbox is not None
        detector_score = (
            float(observation["detector_score"]) if box_returned else None
        )
        mapped = bool(mapping and mapping.get("disposition") == "mapped")
        capture_observation = mapping.get("capture_observation") if mapping else None
        raw_xy = (
            capture_observation.get("xy_m")
            if box_returned and capture_observation
            else None
        )
        sensor_gate = evaluate_sensor_gate({
            "frame_expected": True,
            "frame_received": True,
            "detection_received": box_returned,
            "detector_class": protocol["detector"]["target_class"],
            "detector_confidence": detector_score,
            "bbox_xmin": bbox[0] if bbox else None,
            "bbox_ymin": bbox[1] if bbox else None,
            "bbox_xmax": bbox[2] if bbox else None,
            "bbox_ymax": bbox[3] if bbox else None,
            "projection_valid": bool(mapped and raw_xy),
        }, gate_config)
        row: dict[str, Any] = {
            "drive_id": drive["id"],
            "partition": drive["partition"],
            "route": drive["route"],
            "direction": drive["direction"],
            "source_batch_id": observation["source_batch_id"],
            "capture_stamp_ns": stamp_ns,
            "camera_id": camera_id,
            "source_frame_id": source_frame_id,
            "detector_content_sha256": source_frame_id.rsplit(":", 1)[-1],
            "recorder_image_contract_sha256": (
                identity.get("image_contract_sha256", "") if identity else ""
            ),
            "frame_identity_matched": int(identity is not None),
            "reference_supported": int(reference is not None),
            "yolo_hit": int(box_returned),
            "detector_score": detector_score if detector_score is not None else "",
            "detector_score_raw": (
                float(observation["detector_score_raw"]) if box_returned else ""
            ),
            "bbox_xmin": bbox[0] if bbox else "",
            "bbox_ymin": bbox[1] if bbox else "",
            "bbox_xmax": bbox[2] if bbox else "",
            "bbox_ymax": bbox[3] if bbox else "",
            "capture_support_pass": int(mapped),
            "capture_support_reason": "passed" if mapped else (
                str(mapping.get("reason", "")) if mapping else "missing_manager_mapping"
            ),
            "sensor_gate_id": sensor_gate.gate_id,
            "sensor_gate_config_hash": sensor_gate.gate_config_hash,
            "sensor_gate_admitted": int(sensor_gate.admitted),
            "sensor_gate_reason": sensor_gate.reason,
            "raw_projected_x": raw_xy[0] if raw_xy else "",
            "raw_projected_y": raw_xy[1] if raw_xy else "",
            "crop_path": str(run_dir / "selected_crops" / f"{camera_id}_{stamp_ns:019d}.npz"),
        }
        if reference:
            row.update(reference)
        else:
            row.update({name: "" for name in (
                "reference_x", "reference_y", "reference_yaw", "reference_speed_mps",
                "commanded_linear_mps", "commanded_angular_radps", "controller_state",
                "reference_bracket_gap_s",
            )})
        opportunities.append(row)
        by_round.setdefault(str(observation["source_batch_id"]), []).append(row)

        # The run keeps logging briefly after the route completes, while the robot stands
        # at the final pose.  The renderer is deterministic, so those are repeated copies
        # of one image: including them would let a covariance shrink on duplicates.  They
        # stay in camera_opportunities.csv and are counted in integrity.json.
        parked = str(reference.get("controller_state", "")).startswith("terminal") \
            if reference else False
        if parked:
            parked_excluded += 1
        if sensor_gate.admitted and mapped and raw_xy and reference and not parked:
            residual = (
                float(raw_xy[0]) - float(reference["reference_x"]),
                float(raw_xy[1]) - float(reference["reference_y"]),
            )
            along, across = _ray_residual(
                camera_positions[camera_id],
                (float(reference["reference_x"]), float(reference["reference_y"])),
                residual,
            )
            measurements.append({
                **row,
                "raw_residual_x": residual[0],
                "raw_residual_y": residual[1],
                "raw_residual_along_ray": along,
                "raw_residual_across_ray": across,
                "correction_model_id": "",
                "correction_fold_id": "",
                "corrected_x": "",
                "corrected_y": "",
                "corrected_residual_x": "",
                "corrected_residual_y": "",
                "corrected_residual_along_ray": "",
                "corrected_residual_across_ray": "",
                "R_model_id": "",
                "R_xx_m2": "",
                "R_xy_m2": "",
                "R_yy_m2": "",
                "q_sensor_model_id": "",
                "q_sensor_pred": "",
            })

    camera_ids = list(protocol["detector"]["camera_ids"])
    rounds: list[dict[str, Any]] = []
    complete_rounds = 0
    for source_batch_id, members in sorted(
        by_round.items(), key=lambda item: min(int(row["capture_stamp_ns"]) for row in item[1])
    ):
        present = sorted(row["camera_id"] for row in members)
        complete = present == sorted(camera_ids)
        complete_rounds += int(complete)
        supported = sorted(row["camera_id"] for row in members if row["sensor_gate_admitted"])
        manager_mapped = sorted(
            row["camera_id"] for row in members if row["capture_support_pass"]
        )
        hits = sorted(row["camera_id"] for row in members if row["yolo_hit"])
        referenced = [row for row in members if row["reference_supported"]]
        raw_residuals = {}
        measurement_lookup = {
            (row["camera_id"], row["capture_stamp_ns"]): row for row in measurements
        }
        for row in members:
            measured = measurement_lookup.get((row["camera_id"], row["capture_stamp_ns"]))
            if measured:
                raw_residuals[row["camera_id"]] = [
                    measured["raw_residual_along_ray"], measured["raw_residual_across_ray"]
                ]
        rounds.append({
            "drive_id": drive["id"],
            "partition": drive["partition"],
            "source_batch_id": source_batch_id,
            "capture_stamp_ns": min(int(row["capture_stamp_ns"]) for row in members),
            "complete_five_camera_round": int(complete),
            "present_camera_subset_json": json.dumps(present, separators=(",", ":")),
            "yolo_hit_subset_json": json.dumps(hits, separators=(",", ":")),
            "manager_mapped_subset_json": json.dumps(manager_mapped, separators=(",", ":")),
            "sensor_gate_admitted_subset_json": json.dumps(supported, separators=(",", ":")),
            "supported_camera_subset_json": json.dumps(supported, separators=(",", ":")),
            "n_supported": len(supported),
            "reference_supported": int(len(referenced) == len(members)),
            "reference_x": referenced[0]["reference_x"] if referenced else "",
            "reference_y": referenced[0]["reference_y"] if referenced else "",
            "reference_yaw": referenced[0]["reference_yaw"] if referenced else "",
            "reference_speed_mps": referenced[0]["reference_speed_mps"] if referenced else "",
            "raw_ray_residuals_by_camera_json": json.dumps(raw_residuals, separators=(",", ":")),
            "corrected_ray_residuals_by_camera_json": "",
        })

    output = run_dir / "tables"
    output.mkdir(exist_ok=False)
    opportunity_fields = list(opportunities[0]) if opportunities else []
    measurement_fields = list(measurements[0]) if measurements else opportunity_fields + [
        "raw_residual_x", "raw_residual_y", "raw_residual_along_ray",
        "raw_residual_across_ray", "correction_model_id", "correction_fold_id",
        "corrected_x", "corrected_y", "corrected_residual_x", "corrected_residual_y",
        "corrected_residual_along_ray", "corrected_residual_across_ray", "R_model_id",
        "R_xx_m2", "R_xy_m2", "R_yy_m2", "q_sensor_model_id", "q_sensor_pred",
    ]
    round_fields = list(rounds[0]) if rounds else []
    _write_csv(output / "camera_opportunities.csv", opportunity_fields, opportunities)
    _write_csv(output / "camera_measurements.csv", measurement_fields, measurements)
    _write_csv(output / "network_rounds.csv", round_fields, rounds)

    count = len(opportunities)
    identity_fraction = sum(row["frame_identity_matched"] for row in opportunities) / count if count else 0.0
    reference_fraction = sum(row["reference_supported"] for row in opportunities) / count if count else 0.0
    complete_fraction = complete_rounds / len(rounds) if rounds else 0.0
    crop_required = bool(protocol["capture"].get("save_selected_rgb_crop_npz", False))
    crop_present = 0
    crop_identity_matches = 0
    for row in measurements:
        crop_path = Path(str(row["crop_path"]))
        if not crop_path.is_file():
            continue
        crop_present += 1
        try:
            with np.load(crop_path, allow_pickle=False) as payload:
                stored_source_frame_id = str(payload["source_frame_id"].item())
                stored_stamp_ns = int(payload["capture_stamp_ns"].item())
            crop_identity_matches += int(
                stored_source_frame_id == row["source_frame_id"]
                and stored_stamp_ns == int(row["capture_stamp_ns"])
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
    crop_fraction = crop_present / len(measurements) if measurements else 0.0
    crop_identity_fraction = crop_identity_matches / len(measurements) if measurements else 0.0
    sensor_gate_reason_counts = Counter(row["sensor_gate_reason"] for row in opportunities)
    integrity = {
        "schema": "commissioning_synchronized_tables_integrity.v1",
        "drive_id": drive["id"],
        "partition": drive["partition"],
        "opportunity_rows": count,
        "measurement_rows": len(measurements),
        "parked_admitted_rows_excluded": parked_excluded,
        "network_round_rows": len(rounds),
        "unique_physical_frame_rows": len(seen_frames),
        "frame_identity_match_fraction": identity_fraction,
        "reference_supported_fraction": reference_fraction,
        "complete_five_camera_round_fraction": complete_fraction,
        "selected_crop_required": crop_required,
        "selected_crop_expected_rows": len(measurements),
        "selected_crop_present_rows": crop_present,
        "selected_crop_present_fraction": crop_fraction,
        "selected_crop_identity_match_fraction": crop_identity_fraction,
        "frame_identity_duplicate_count": identity_duplicates,
        "sensor_gate_id": gate_config.gate_id,
        "sensor_gate_config_hash": gate_config.config_hash(),
        "sensor_gate_reason_counts": dict(sorted(sensor_gate_reason_counts.items())),
        "preflight_protocol_sha256": preflight["protocol_sha256"],
        "passed": bool(
            count > 0
            and len(seen_frames) == count
            and identity_duplicates == 0
            and identity_fraction >= float(protocol["capture"]["minimum_frame_identity_match_fraction"])
            and reference_fraction >= 0.99
            and complete_fraction == 1.0
            and (not crop_required or (
                len(measurements) > 0
                and crop_present == len(measurements)
                and crop_identity_matches == len(measurements)
            ))
        ),
    }
    _atomic_json(output / "integrity.json", integrity)
    return integrity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "campaign_v2.yaml")
    args = parser.parse_args()
    report = build_tables(args.run_dir.resolve(), args.protocol.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
