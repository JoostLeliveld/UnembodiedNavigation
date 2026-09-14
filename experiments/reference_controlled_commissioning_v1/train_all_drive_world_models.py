#!/usr/bin/env python3
"""Fit both perception-only world-position models on all completed commissioning drives.

The historical fit/development/audit labels describe the superseded drive-holdout
protocol.  This deployment fit deliberately pools every completed drive.  Independent
navigation drives remain the evaluation unit.  Source campaign directories are read-only.

Only current-frame perception quantities enter either network.  The timestamp-aligned
reference pose supplies the offline target and is never a model input.  Belief, EKF,
innovation, map visibility, obstacle geometry, and route identity are not read as inputs.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common")]

import train_static_world_models as models  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def lerp_angle(first: float, second: float, fraction: float) -> float:
    delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return (first + fraction * delta + math.pi) % (2.0 * math.pi) - math.pi


def reference_at(
    trace: list[dict[str, Any]], stamp_ns: int, max_gap_s: float
) -> dict[str, Any] | None:
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
    nearest = before if fraction <= 0.5 else after
    return {
        "reference_x": float(p0[0]) + fraction * (float(p1[0]) - float(p0[0])),
        "reference_y": float(p0[1]) + fraction * (float(p1[1]) - float(p0[1])),
        "reference_yaw": lerp_angle(float(p0[2]), float(p1[2]), fraction),
        "controller_state": nearest.get("state", ""),
    }


def camera_positions(protocol: dict[str, Any]) -> dict[str, tuple[float, float]]:
    from reliability.projection import camera_model_from_world

    profile_payload = yaml.safe_load(
        (REPO / protocol["world"]["profiles_path"]).read_text(encoding="utf-8")
    )
    profile = profile_payload["worlds"][protocol["world"]["name"]]
    result = {}
    for camera_id, include_name in zip(
        profile["camera_ids"], profile["camera_model_includes"], strict=True
    ):
        model = camera_model_from_world(
            REPO / protocol["world"]["path"], include_name=include_name
        )
        result[str(camera_id)] = (float(model.cam_pos[0]), float(model.cam_pos[1]))
    return result


def source_file(run: Path, pattern: str) -> Path:
    matches = sorted(run.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {run}, found {len(matches)}")
    return matches[0]


def crop_with_valid_channel(path: Path, source_frame_id: str, stamp_ns: int) -> tuple[np.ndarray, bool]:
    with np.load(path, allow_pickle=False) as archive:
        crop = np.asarray(archive["crop"], dtype=np.uint8)
        bbox = np.asarray(archive["bbox_xyxy"], dtype=float)
        image_shape = np.asarray(archive["image_shape"], dtype=int)
        stored_stamp = int(archive["capture_stamp_ns"])
        stored_frame = str(archive["source_frame_id"])
    if crop.shape != (3, models.IMAGE_SIZE, models.IMAGE_SIZE):
        raise RuntimeError(f"unexpected crop shape {crop.shape} at {path}")
    if stored_stamp != stamp_ns or stored_frame != source_frame_id:
        raise RuntimeError(f"crop identity mismatch at {path}")
    x0, y0, x1, y1 = bbox
    height, width = image_shape
    box_width, box_height = x1 - x0, y1 - y0
    context_clipped = bool(
        x0 - models.CONTEXT_FRACTION * box_width < 0
        or y0 - models.CONTEXT_FRACTION * box_height < 0
        or x1 + models.CONTEXT_FRACTION * box_width > width
        or y1 + models.CONTEXT_FRACTION * box_height > height
    )
    # The drive recorder clipped the requested context rectangle before resizing.
    # Pixels in its stored crop are therefore all valid; lost padding cannot be rebuilt.
    valid = np.full((1, models.IMAGE_SIZE, models.IMAGE_SIZE), 255, dtype=np.uint8)
    return np.concatenate((crop, valid), axis=0), context_clipped


def load_all_drives(campaign: Path, gate_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    execution_path = campaign / "campaign_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("commissioning campaign is not complete")
    sources = execution.get("drive_sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("campaign has no drive_sources")

    protocol_path = Path(execution["protocol"]).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    if gate.get("gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("drive training requires commissioning_sensor_gate_v2")
    if gate.get("edge_check_mode") != "full_bbox":
        raise RuntimeError("expected full_bbox sensor gate")
    camera_xy = {
        key: np.asarray(value, dtype=float)
        for key, value in camera_positions(protocol).items()
    }
    if set(camera_xy) != set(models.CAMERAS):
        raise RuntimeError("campaign does not use the expected five cameras")
    max_gap_s = float(protocol["capture"]["maximum_reference_interpolation_gap_s"])
    edge = float(gate["min_edge_distance_px"])

    rows: list[dict[str, Any]] = []
    drive_reports: list[dict[str, Any]] = []
    input_sources: list[dict[str, Any]] = []
    opportunities_total = 0
    rejection_counts: dict[str, int] = {}
    context_clipped_total = 0

    for source in sources:
        drive_id = str(source["drive_id"])
        run = Path(source["source"]).resolve()
        drive_manifest_path = run / "drive_manifest.json"
        manager_path = run / "manager_outcomes.jsonl"
        trace_path = run / "reference_controller_trace.jsonl"
        opportunity_path = source_file(run, "experiment_logs/experiment_*/camera_opportunities.jsonl")
        manifest = json.loads(drive_manifest_path.read_text(encoding="utf-8"))
        drive = manifest["drive"]
        if drive["id"] != drive_id:
            raise RuntimeError(f"drive identity mismatch for {drive_id}")

        trace = [row for row in jsonl(trace_path) if row.get("pose") is not None]
        trace.sort(key=lambda row: int(row["stamp_ns"]))
        mappings: dict[tuple[str, int, str], dict[str, Any]] = {}
        for mapping in jsonl(manager_path):
            if mapping.get("status") != "camera_mapping":
                continue
            key = (
                str(mapping.get("camera_id", "")),
                int(mapping.get("capture_stamp_ns", -1)),
                str(mapping.get("source_frame_id", "")),
            )
            if key in mappings:
                raise RuntimeError(f"duplicate manager mapping for {key}")
            mappings[key] = mapping

        drive_opportunities = 0
        drive_admitted = 0
        drive_clipped = 0
        seen: set[tuple[str, int, str]] = set()
        for delivery in jsonl(opportunity_path):
            if not delivery.get("valid_contract", False) or delivery.get("duplicate", False):
                continue
            observation = delivery["observation"]
            camera = str(observation["camera_id"])
            stamp_ns = int(observation["capture_stamp_ns"])
            source_frame_id = str(observation["source_frame_id"])
            key = (camera, stamp_ns, source_frame_id)
            if key in seen:
                raise RuntimeError(f"duplicate physical frame in {drive_id}: {key}")
            seen.add(key)
            drive_opportunities += 1
            opportunities_total += 1

            bbox_value = observation.get("bbox_xyxy")
            if bbox_value is None:
                rejection_counts["no_detection"] = rejection_counts.get("no_detection", 0) + 1
                continue
            bbox = np.asarray(bbox_value, dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                rejection_counts["invalid_bbox"] = rejection_counts.get("invalid_bbox", 0) + 1
                continue
            mapping = mappings.get(key)
            capture = mapping.get("capture_observation") if mapping else None
            raw_value = capture.get("xy_m") if capture else None
            if not mapping or mapping.get("disposition") != "mapped" or raw_value is None:
                rejection_counts["invalid_projection"] = rejection_counts.get("invalid_projection", 0) + 1
                continue
            raw = np.asarray(raw_value, dtype=float)
            if raw.shape != (2,) or not np.isfinite(raw).all():
                rejection_counts["invalid_projection"] = rejection_counts.get("invalid_projection", 0) + 1
                continue
            reference = reference_at(trace, stamp_ns, max_gap_s)
            if reference is None:
                rejection_counts["reference_unsupported"] = rejection_counts.get("reference_unsupported", 0) + 1
                continue
            if str(reference.get("controller_state", "")).startswith("terminal"):
                rejection_counts["terminal_duplicate"] = rejection_counts.get("terminal_duplicate", 0) + 1
                continue

            crop_path = run / "selected_crops" / f"{camera}_{stamp_ns:019d}.npz"
            if not crop_path.is_file():
                rejection_counts["missing_crop"] = rejection_counts.get("missing_crop", 0) + 1
                continue
            with np.load(crop_path, allow_pickle=False) as archive:
                image_shape = np.asarray(archive["image_shape"], dtype=float)
            height, width = map(float, image_shape)
            x0, y0, x1, y1 = bbox
            box_width, box_height = x1 - x0, y1 - y0
            confidence = float(observation["detector_score"])
            checks = (
                (confidence >= float(gate["confidence_threshold"]), "confidence"),
                (box_width >= float(gate["min_bbox_width_px"]), "box_width"),
                (box_height >= float(gate["min_bbox_height_px"]), "box_height"),
                (x0 >= edge and y0 >= edge and x1 <= width - edge and y1 <= height - edge, "bbox_edge"),
            )
            refusal = next((reason for passed, reason in checks if not passed), None)
            if refusal is not None:
                rejection_counts[refusal] = rejection_counts.get(refusal, 0) + 1
                continue

            basis = models.ray_basis(camera_xy[camera], raw)
            ray = raw - camera_xy[camera]
            distance = float(np.linalg.norm(ray))
            bearing = math.atan2(ray[1], ray[0])
            feature = np.asarray([
                raw[0], raw[1], distance, 1.0 / max(distance, 1e-6),
                math.cos(bearing), math.sin(bearing),
                box_width / width, box_height / height,
                box_width / max(box_height, 1e-6),
                0.5 * (x0 + x1) / width, y1 / height, confidence,
                *[float(camera == candidate) for candidate in models.CAMERAS],
            ], dtype=np.float32)
            truth = np.asarray([reference["reference_x"], reference["reference_y"]], dtype=np.float32)
            image, context_clipped = crop_with_valid_channel(crop_path, source_frame_id, stamp_ns)
            context_clipped_total += int(context_clipped)
            drive_clipped += int(context_clipped)
            rows.append({
                "drive": drive_id,
                "partition": str(drive["partition"]),
                "route": str(drive["route"]),
                "direction": str(drive["direction"]),
                "camera": camera,
                "source_frame_id": source_frame_id,
                "stamp_ns": stamp_ns,
                "feature": feature,
                "image": image,
                "raw": raw.astype(np.float32),
                "truth": truth,
                "basis": basis.astype(np.float32),
                "target": (basis.T @ (truth - raw)).astype(np.float32),
                "stationary": str(reference.get("controller_state", "")).startswith("stationary"),
            })
            drive_admitted += 1

        drive_reports.append({
            "drive_id": drive_id,
            "historical_partition": str(drive["partition"]),
            "route": str(drive["route"]),
            "direction": str(drive["direction"]),
            "opportunities": drive_opportunities,
            "admitted": drive_admitted,
            "context_clipped": drive_clipped,
        })
        input_sources.append({
            "drive_id": drive_id,
            "run_dir": str(run),
            "drive_manifest_sha256": sha256(drive_manifest_path),
            "opportunities_sha256": sha256(opportunity_path),
            "manager_outcomes_sha256": sha256(manager_path),
            "reference_trace_sha256": sha256(trace_path),
        })
        print(f"loaded {drive_id}: opportunities={drive_opportunities} admitted={drive_admitted}", flush=True)

    if not rows:
        raise RuntimeError("current sensor gate admitted no drive observations")
    data = {
        "feature": np.stack([row["feature"] for row in rows]),
        "image": np.stack([row["image"] for row in rows]),
        "raw": np.stack([row["raw"] for row in rows]),
        "truth": np.stack([row["truth"] for row in rows]),
        "basis": np.stack([row["basis"] for row in rows]),
        "target": np.stack([row["target"] for row in rows]),
        "camera": np.asarray([row["camera"] for row in rows]),
        "drive": np.asarray([row["drive"] for row in rows]),
        "partition": np.asarray([row["partition"] for row in rows]),
        "route": np.asarray([row["route"] for row in rows]),
        "direction": np.asarray([row["direction"] for row in rows]),
        "source_frame_id": np.asarray([row["source_frame_id"] for row in rows]),
        "stamp_ns": np.asarray([row["stamp_ns"] for row in rows], dtype=np.int64),
        "stationary": np.asarray([row["stationary"] for row in rows], dtype=bool),
    }
    manifest = {
        "campaign_execution": str(execution_path),
        "campaign_execution_sha256": sha256(execution_path),
        "protocol": str(protocol_path),
        "current_gate": str(gate_path),
        "current_gate_sha256": sha256(gate_path),
        "drive_sources": input_sources,
        "drive_counts": drive_reports,
        "opportunities": opportunities_total,
        "admitted": len(rows),
        "context_clipped": context_clipped_total,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return data, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-epochs", type=int, default=200)
    parser.add_argument("--rgb-epochs", type=int, default=100)
    parser.add_argument("--rgb-warmup-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    print(f"loading all completed commissioning drives on {device}", flush=True)
    data, input_manifest = load_all_drives(args.campaign.resolve(), args.gate.resolve())
    (staging / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        staging / "dataset_contract.npz",
        feature=data["feature"], target=data["target"], raw=data["raw"], truth=data["truth"],
        basis=data["basis"], camera=data["camera"], drive=data["drive"],
        historical_partition=data["partition"], route=data["route"],
        direction=data["direction"], source_frame_id=data["source_frame_id"],
        stamp_ns=data["stamp_ns"], stationary=data["stationary"],
    )

    indexes = np.arange(len(data["target"]))
    reports: dict[str, Any] = {}
    histories: dict[str, list[float]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    specifications = {
        "box_spatial_mlp": (args.box_epochs, 0),
        "rgb_gaussian": (args.rgb_epochs, args.rgb_warmup_epochs),
    }
    for kind, (epochs, warmup) in specifications.items():
        seed = 20267914 + (1 if kind == "rgb_gaussian" else 0)
        model, feature_mean, feature_std, history = models.fit_model(
            kind, data, indexes, epochs=epochs, warmup_epochs=warmup,
            seed=seed, batch_size=args.batch_size, device=device,
        )
        mean, covariance = models.predict(
            kind, model, data, indexes, feature_mean, feature_std,
            batch_size=args.batch_size, device=device,
        )
        checkpoint = models.checkpoint_payload(
            kind, model, feature_mean, feature_std, epochs=epochs, seed=seed,
        )
        if kind == "rgb_gaussian":
            checkpoint["image_preprocessing"] = {
                "context": "half one box width/height on each side",
                "boundary_behavior": "clip context rectangle to the source image before resizing",
                "resize": [models.IMAGE_SIZE, models.IMAGE_SIZE],
                "valid_pixel_mask": "constant one; retained for the shared four-channel architecture",
            }
        torch.save(checkpoint, staging / f"{kind}_deployment.pt")
        report: dict[str, Any] = {
            "label": "training_fit_diagnostic_not_evaluation",
            "pooled": models.mean_summary(mean, data["target"]),
            "by_camera": {
                camera: models.mean_summary(
                    mean[data["camera"] == camera], data["target"][data["camera"] == camera]
                )
                for camera in models.CAMERAS
            },
            "by_drive": {
                drive: models.mean_summary(
                    mean[data["drive"] == drive], data["target"][data["drive"] == drive]
                )
                for drive in sorted(set(data["drive"].tolist()))
            },
        }
        if covariance is not None:
            report["covariance_training_diagnostic"] = models.covariance_summary(
                mean, data["target"], covariance
            )
        reports[kind] = report
        histories[kind] = history
        predictions[kind] = (mean, covariance)

    np.savez_compressed(
        staging / "training_predictions.npz",
        target_ray=data["target"], basis=data["basis"], raw_world=data["raw"],
        truth_world=data["truth"], camera=data["camera"], drive=data["drive"],
        box_spatial_mean=predictions["box_spatial_mlp"][0],
        rgb_gaussian_mean=predictions["rgb_gaussian"][0],
        rgb_gaussian_covariance=predictions["rgb_gaussian"][1],
    )
    (staging / "training_history.json").write_text(
        json.dumps(histories, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "schema": "drive_perception_world_models_all_completed.v1",
        "status": "deployment_fit_complete_navigation_evaluation_pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "population": {
            "completed_drives": len(input_manifest["drive_counts"]),
            "historical_partitions_reclassified_as_training": sorted(set(data["partition"].tolist())),
            "opportunities": input_manifest["opportunities"],
            "admitted": len(indexes),
            "static_or_drive_holdout": "none",
            "evaluation": "independent continuous navigation drives",
        },
        "sensor_gate": "commissioning_sensor_gate_v2",
        "features": list(models.FEATURE_NAMES),
        "image_input": {
            "shape": [4, models.IMAGE_SIZE, models.IMAGE_SIZE],
            "channels": ["red", "green", "blue", "valid_pixel_mask"],
            "context_fraction_each_side": models.CONTEXT_FRACTION,
            "drive_archive_limitation": (
                "stored context rectangles were clipped before resizing; the valid-pixel "
                "channel is therefore constant and lost padding cannot be reconstructed"
            ),
            "context_clipped_records": input_manifest["context_clipped"],
        },
        "target": "reference robot-centre world XY minus raw box-bottom projection, ray frame metres",
        "public_output": "corrected world XY",
        "forbidden_inputs_absent": [
            "belief", "robot_ekf", "prior_robot_position", "prior_robot_heading",
            "route_identity", "timestamp", "reference_pose", "controller_state",
            "innovation", "NIS", "line_of_sight", "obstacle_height", "robot_hull",
        ],
        "reports": reports,
        "reporting_boundary": (
            "All drive accuracy and covariance summaries are in-sample training diagnostics. "
            "They are not navigation evaluation results."
        ),
        "training": {
            "box_epochs": args.box_epochs,
            "rgb_epochs": args.rgb_epochs,
            "rgb_mean_warmup_epochs": args.rgb_warmup_epochs,
            "batch_size": args.batch_size,
            "fit": "all current-gate-admitted observations from all completed commissioning drives",
        },
        "inputs": {
            "input_manifest_sha256": sha256(staging / "input_manifest.json"),
            "campaign_execution_sha256": input_manifest["campaign_execution_sha256"],
            "gate_sha256": input_manifest["current_gate_sha256"],
            "implementation": sha256(Path(__file__)),
            "shared_model_implementation": sha256(Path(models.__file__)),
        },
    }
    result_path = staging / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {path.name: sha256(path) for path in staging.iterdir() if path.is_file()}
    (staging / "manifest.json").write_text(
        json.dumps({"schema": "artifact_manifest.v1", "status": "complete", "artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
