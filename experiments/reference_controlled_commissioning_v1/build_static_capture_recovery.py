#!/usr/bin/env python3
"""Freeze an additive recovery plan for failed static capture transactions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    manifest_path = capture / "capture_manifest.json"
    index_path = capture / "capture_index.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_with_failed_batches":
        raise RuntimeError("recovery planning requires a completed capture with failed batches")
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["pose_id"]), int(row["repetition_id"]))].append(row)
    failures = []
    poses = []
    for (pose_id, repetition_id), batch in sorted(grouped.items()):
        statuses = {row["capture_status"] for row in batch}
        if statuses == {"ok"}:
            continue
        if statuses != {"failed"} or len(batch) != int(manifest["plan"]["camera_count"]):
            raise RuntimeError(f"pose {pose_id} is not an atomic all-camera failed batch")
        source = manifest["pose_plan"][pose_id]
        poses.append({
            "x": source["x"], "y": source["y"], "yaw": source["yaw"],
            "position_id": source["position_id"],
            "position_key": source.get("position_key", ""),
            "x_idx": source["position_id"], "y_idx": source["position_id"],
            "heading_id": source["yaw_idx"],
            "heading_degrees": source.get("heading_degrees"),
            "block_id": source.get("block_id", ""),
            "stratum": source.get("dataset_split", ""),
            "original_pose_id": pose_id,
            "original_repetition_id": repetition_id,
        })
        failures.append({
            "original_pose_id": pose_id,
            "original_repetition_id": repetition_id,
            "position_id": source["position_id"],
            "heading_id": source["yaw_idx"],
            "errors": sorted({row["capture_error"] for row in batch}),
        })
    if len(failures) != int(manifest.get("failed_batches", -1)):
        raise RuntimeError("manifest failure count does not match the capture index")

    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pose_path = out / f"{args.name}_poses.json"
    pose_path.write_text(json.dumps({
        "schema": "static_capture_additive_recovery_pose_plan.v1",
        "source_capture": str(capture),
        "poses": poses,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    amendment_path = out / f"{args.name}_protocol.json"
    amendment_path.write_text(json.dumps({
        "schema": "static_capture_additive_recovery_protocol.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_recovery_capture",
        "source_capture": {
            "path": str(capture),
            "capture_manifest_sha256": sha256(manifest_path),
            "capture_index_sha256": sha256(index_path),
            "status": manifest["status"],
            "failed_batches": manifest["failed_batches"],
        },
        "original_protocol_manifest": manifest.get("protocol_manifest"),
        "recovery_pose_file": {"path": str(pose_path), "sha256": sha256(pose_path)},
        "failed_transactions": failures,
        "recovery_policy": {
            "additive_only": True,
            "original_capture_is_never_rewritten": True,
            "accepted_recovery_must_have_exactly_five_successful_camera_rows_per_transaction": True,
            "scientific_pose_set_is_unchanged": True,
            "wall_clock_timeout_may_be_increased_but_settle_and_freshness_criteria_are_unchanged": True,
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "failed_transactions": len(failures),
        "pose_file": str(pose_path),
        "protocol": str(amendment_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
