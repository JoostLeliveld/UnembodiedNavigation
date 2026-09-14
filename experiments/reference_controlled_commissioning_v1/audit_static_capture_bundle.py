#!/usr/bin/env python3
"""Audit a primary static capture plus an additive failed-transaction recovery."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2


CAMERAS = {f"camera_{letter}" for letter in "ABCDE"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_sha1(image) -> str:
    digest = hashlib.sha1()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def read_rows(root: Path) -> list[dict]:
    with (root / "capture_index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        result[int(row["pose_id"])].append(row)
    return result


def fresh_and_coherent(batch: list[dict], limit_s: float) -> bool:
    if len(batch) != 5 or {row["camera_id"] for row in batch} != CAMERAS:
        return False
    if {row["capture_status"] for row in batch} != {"ok"}:
        return False
    stamps = [int(row["image_stamp_ns"]) for row in batch]
    if max(stamps) - min(stamps) > round(limit_s * 1e9):
        return False
    return all(
        0 < int(row["command_issue_ns"])
        <= int(row["command_ack_ns"])
        <= int(row["settle_barrier_ns"])
        < int(row["image_stamp_ns"])
        for row in batch
    )


def source_hashes_match(manifest: dict, root: Path) -> bool:
    targets = [
        (manifest.get("world_path"), manifest.get("world_sha256")),
        (manifest.get("world_profiles_path"), manifest.get("world_profiles_sha256")),
        (manifest.get("plan", {}).get("pose_file"), manifest.get("plan", {}).get("pose_file_sha256")),
    ]
    protocol = manifest.get("protocol_manifest") or {}
    targets.append((protocol.get("path"), protocol.get("sha256")))
    return all(
        bool(path and expected and Path(path).is_file() and sha256(Path(path)) == expected)
        for path, expected in targets
    ) and sha256(root / "capture_index.csv") == manifest.get("capture_index_sha256")


def verify_images(root: Path, rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        if row["capture_status"] != "ok":
            continue
        path = root / row["image"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image_sha1(image) != row["image_sha1"]:
            problems.append(str(path))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary = args.primary.expanduser().resolve()
    recovery = args.recovery.expanduser().resolve()
    recovery_protocol_path = args.recovery_protocol.expanduser().resolve()
    primary_manifest = json.loads((primary / "capture_manifest.json").read_text(encoding="utf-8"))
    recovery_manifest = json.loads((recovery / "capture_manifest.json").read_text(encoding="utf-8"))
    recovery_protocol = json.loads(recovery_protocol_path.read_text(encoding="utf-8"))
    primary_rows, recovery_rows = read_rows(primary), read_rows(recovery)
    primary_batches, recovery_batches = grouped(primary_rows), grouped(recovery_rows)

    failed_primary = {
        pose_id: batch for pose_id, batch in primary_batches.items()
        if {row["capture_status"] for row in batch} == {"failed"}
    }
    successful_primary = {
        pose_id: batch for pose_id, batch in primary_batches.items()
        if {row["capture_status"] for row in batch} == {"ok"}
    }
    recovery_map = {
        int(item["original_pose_id"]): recovery_batches[index]
        for index, item in enumerate(recovery_protocol["failed_transactions"])
    }
    pose_matches = True
    for original_pose_id, batch in recovery_map.items():
        expected = primary_manifest["pose_plan"][original_pose_id]
        pose_matches &= all(
            int(row["position_id"]) == int(expected["position_id"])
            and int(row["heading_id"]) == int(expected["yaw_idx"])
            and math.isclose(float(row["robot_x"]), float(expected["x"]), abs_tol=1e-12)
            and math.isclose(float(row["robot_y"]), float(expected["y"]), abs_tol=1e-12)
            and math.isclose(float(row["robot_yaw"]), float(expected["yaw"]), abs_tol=1e-12)
            for row in batch
        )

    primary_limit = float(primary_manifest["plan"]["batch_sync_slop_ms"]) / 1000.0
    recovery_limit = float(recovery_manifest["plan"]["batch_sync_slop_ms"]) / 1000.0
    primary_image_problems = verify_images(primary, primary_rows)
    recovery_image_problems = verify_images(recovery, recovery_rows)
    checks = {
        "primary_closed_with_declared_failures": (
            primary_manifest.get("status") == "complete_with_failed_batches"
            and len(failed_primary) == int(primary_manifest.get("failed_batches", -1))
        ),
        "primary_has_complete_pose_accounting": len(primary_batches) == int(primary_manifest["plan"]["pose_count"]),
        "primary_success_batches_fresh_and_coherent": all(
            fresh_and_coherent(batch, primary_limit) for batch in successful_primary.values()
        ),
        "recovery_complete_without_failures": (
            recovery_manifest.get("status") == "complete"
            and int(recovery_manifest.get("failed_batches", -1)) == 0
        ),
        "recovery_batches_fresh_and_coherent": all(
            fresh_and_coherent(batch, recovery_limit) for batch in recovery_batches.values()
        ),
        "recovery_exactly_covers_failed_primary_pose_ids": set(recovery_map) == set(failed_primary),
        "recovery_pose_values_match_primary_plan": bool(pose_matches),
        "combined_successful_pose_count": len(successful_primary) + len(recovery_map) == int(primary_manifest["plan"]["pose_count"]),
        "combined_successful_camera_opportunities": (
            5 * (len(successful_primary) + len(recovery_map))
            == int(primary_manifest["plan"]["planned_rows"])
        ),
        "primary_source_hashes_match": source_hashes_match(primary_manifest, primary),
        "recovery_source_hashes_match": source_hashes_match(recovery_manifest, recovery),
        "recovery_protocol_hash_matches_manifest": (
            sha256(recovery_protocol_path) == recovery_manifest.get("protocol_manifest", {}).get("sha256")
        ),
        "primary_decoded_image_hashes_match": not primary_image_problems,
        "recovery_decoded_image_hashes_match": not recovery_image_problems,
    }
    report = {
        "schema": "static_capture_additive_bundle_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "primary": str(primary),
        "recovery": str(recovery),
        "counts": {
            "planned_pose_batches": int(primary_manifest["plan"]["pose_count"]),
            "primary_successful_pose_batches": len(successful_primary),
            "recovered_pose_batches": len(recovery_map),
            "combined_successful_camera_opportunities": 5 * (len(successful_primary) + len(recovery_map)),
        },
        "failed_primary_pose_ids_replaced": sorted(failed_primary),
        "image_problems": (primary_image_problems + recovery_image_problems)[:50],
        "source_sha256": {
            "primary_manifest": sha256(primary / "capture_manifest.json"),
            "primary_index": sha256(primary / "capture_index.csv"),
            "recovery_manifest": sha256(recovery / "capture_manifest.json"),
            "recovery_index": sha256(recovery / "capture_index.csv"),
            "recovery_protocol": sha256(recovery_protocol_path),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
