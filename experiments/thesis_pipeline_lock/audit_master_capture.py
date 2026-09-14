#!/usr/bin/env python3
"""Gate the locked master capture before any detector labels are exported.

This is an integrity audit, not a model evaluation.  It may verify the sealed
final-audit files and accounting, but it never computes detector metrics or
uses final-audit content for a decision.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CAMERAS = {f"camera_{letter}" for letter in "ABCDE"}
ROLES = {
    "detector_fit": (80, 3200),
    "detector_validation": (40, 1600),
    "commissioning_fit": (240, 9600),
    "final_audit": (40, 1600),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_sha1(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    # Match the capture writer's canonical array identity exactly.  Shape and
    # dtype are part of the digest so differently interpreted pixel buffers do
    # not compare equal merely because their raw bytes happen to match.
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha1()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-pixels",
        action="store_true",
        help="Decode each unique RGB/mask file and verify its recorded SHA-1.",
    )
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    manifest_path = capture / "capture_manifest.json"
    index_path = capture / "capture_index.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    checks: dict[str, bool] = {}
    checks["capture_complete"] = manifest.get("status") == "complete"
    checks["schema_supported"] = manifest.get("schema") == "bbox_characterization_capture.v3"
    checks["transport_training_eligible"] = bool(
        manifest.get("transport", {}).get("training_eligible")
    )
    checks["protocol_identity"] = (
        manifest.get("protocol_manifest", {}).get("sha256") == sha256(protocol_path)
    )
    checks["index_identity"] = (
        manifest.get("capture_index_sha256") == sha256(index_path)
    )

    expected_rows = int(protocol["capture_contract"]["attempted_camera_views"])
    expected_poses = int(protocol["capture_contract"]["robot_pose_count"])
    max_span_s = float(
        protocol["capture_contract"]["synchronized_five_camera_batch_max_span_ms"]
    ) / 1000.0
    checks["exact_row_count"] = len(rows) == expected_rows
    checks["all_rows_ok"] = all(row.get("capture_status") == "ok" for row in rows)
    checks["all_rows_have_rgb_and_mask"] = all(
        row.get("image") and row.get("robot_mask") for row in rows
    )

    batches: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        batches[(row["pose_id"], row["repetition_id"])].append(row)
    checks["exact_pose_count"] = len(batches) == expected_poses
    checks["complete_five_camera_transactions"] = all(
        len(batch) == 5 and {row["camera_id"] for row in batch} == CAMERAS
        for batch in batches.values()
    )
    checks["batch_timestamps_within_contract"] = all(
        float(row["batch_image_span_s"]) <= max_span_s for row in rows
    )

    positions_by_role: dict[str, set[str]] = collections.defaultdict(set)
    views_by_role: collections.Counter[str] = collections.Counter()
    for row in rows:
        role = row["dataset_split"]
        positions_by_role[role].add(row["position_id"])
        views_by_role[role] += 1
    checks["exact_role_names"] = set(positions_by_role) == set(ROLES)
    checks["exact_role_counts"] = all(
        len(positions_by_role[role]) == positions
        and views_by_role[role] == views
        for role, (positions, views) in ROLES.items()
    )
    all_position_memberships = [
        (role, position)
        for role, positions in positions_by_role.items()
        for position in positions
    ]
    checks["positions_belong_to_one_role_only"] = (
        len(all_position_memberships)
        == len({position for _, position in all_position_memberships})
    )

    missing_files: list[str] = []
    for row in rows:
        for field in ("image", "robot_mask"):
            path = capture / row[field]
            if not path.is_file():
                missing_files.append(row[field])
    checks["all_referenced_files_exist"] = not missing_files

    hash_mismatches: list[str] = []
    if args.verify_pixels and not missing_files:
        expected_by_path: dict[str, str] = {}
        for row in rows:
            expected_by_path[row["image"]] = row["image_sha1"]
            expected_by_path[row["robot_mask"]] = row["robot_mask_sha1"]
        for relative, expected in expected_by_path.items():
            if decoded_sha1(capture / relative) != expected:
                hash_mismatches.append(relative)
        checks["decoded_pixel_hashes_match"] = not hash_mismatches

    report = {
        "schema": "thesis_master_capture_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture": str(capture),
        "protocol": str(protocol_path),
        "integrity_only_final_audit_access": True,
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "poses": len(batches),
            "rows": len(rows),
            "positions_by_role": {
                role: len(positions_by_role[role]) for role in sorted(positions_by_role)
            },
            "views_by_role": dict(sorted(views_by_role.items())),
            "unique_rgb_files": len({row["image"] for row in rows if row.get("image")}),
            "unique_mask_files": len(
                {row["robot_mask"] for row in rows if row.get("robot_mask")}
            ),
        },
        "missing_files": missing_files[:100],
        "pixel_hash_mismatches": hash_mismatches[:100],
        "final_audit_policy": (
            "File existence, identity and split accounting only; no model score, threshold, "
            "checkpoint or hyperparameter may be selected from final_audit."
        ),
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
