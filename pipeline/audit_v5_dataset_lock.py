#!/usr/bin/env python3
"""Read-only preflight and prefix/completion audit for the final reference survey."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name("v5_dataset_lock.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", type=Path, default=LOCK)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--json-output", type=Path)
    args = ap.parse_args()

    lock_path = args.lock if args.lock.is_absolute() else REPO / args.lock
    lock = json.loads(lock_path.read_text())
    errors: list[str] = []
    warnings: list[str] = []

    hash_checks: dict[str, dict[str, object]] = {}
    declared = [
        ("world", lock["environment"]["world"], lock["environment"]["world_sha256"]),
        ("camera_profile", lock["environment"]["camera_profile"], lock["environment"]["camera_profile_sha256"]),
        ("detector", lock["detector"]["checkpoint"], lock["detector"]["checkpoint_sha256"]),
        ("plan", lock["sampling"]["plan"], lock["sampling"]["plan_sha256"]),
        ("positions", lock["sampling"]["positions"], lock["sampling"]["positions_sha256"]),
        ("poses", lock["sampling"]["poses"], lock["sampling"]["poses_sha256"]),
        ("capture_script", lock["capture_implementation"]["script"], lock["capture_implementation"]["script_sha256"]),
        ("resume_auditor", lock["capture_implementation"]["resume_auditor"], lock["capture_implementation"]["resume_auditor_sha256"]),
    ]
    for name, rel, expected in declared:
        path = REPO / rel
        actual = sha256(path) if path.is_file() else None
        ok = actual == expected
        hash_checks[name] = {"path": rel, "expected": expected, "actual": actual, "ok": ok}
        if not ok:
            fail(errors, f"{name} hash mismatch: {rel}")

    model_names = {
        "camera_A": "external_camera", "camera_B": "external_camera_b",
        "camera_C": "external_camera_c", "camera_D": "external_camera_d",
        "camera_E": "external_camera_e",
    }
    for camera, expected in lock["environment"]["camera_model_sha256"].items():
        source = REPO / "src/sim/models" / model_names[camera] / "model.sdf"
        overlay = REPO / lock["campaign_root"] / "capture_models" / model_names[camera] / "model.sdf"
        actual = sha256(source) if source.is_file() else None
        overlay_actual = sha256(overlay) if overlay.is_file() else None
        ok = actual == expected and overlay_actual == expected
        hash_checks[f"{camera}_model"] = {
            "source": str(source.relative_to(REPO)), "source_actual": actual,
            "overlay": str(overlay.relative_to(REPO)), "overlay_actual": overlay_actual,
            "expected": expected, "ok": ok,
        }
        if not ok:
            fail(errors, f"{camera} source/capture-overlay calibration mismatch")

    poses = json.loads((REPO / lock["sampling"]["poses"]).read_text())
    by_position: dict[str, list[dict]] = defaultdict(list)
    for row in poses:
        by_position[str(row["position_key"])].append(row)
    role_counts = Counter()
    for position, rows in by_position.items():
        roles = {r["stratum"] for r in rows}
        xy = {(float(r["x"]), float(r["y"])) for r in rows}
        heading_ids = {int(r["yaw_idx"]) for r in rows}
        if len(roles) != 1 or len(xy) != 1:
            fail(errors, f"position {position} crosses a role or coordinate boundary")
        if len(rows) != lock["sampling"]["headings_per_position"] or len(heading_ids) != len(rows):
            fail(errors, f"position {position} does not contain the locked heading set")
        ordered = sorted(float(r["yaw"]) % (2 * math.pi) for r in rows)
        gaps = [(ordered[(i + 1) % len(ordered)] - ordered[i]) % (2 * math.pi)
                for i in range(len(ordered))]
        if any(abs(math.degrees(g) - lock["sampling"]["heading_spacing_deg"]) > 1e-5 for g in gaps):
            fail(errors, f"position {position} headings are not equally spaced")
        role_counts[next(iter(roles))] += 1

    if len(poses) != lock["sampling"]["expected_pose_batches"]:
        fail(errors, "pose-batch count differs from lock")
    if len(by_position) != lock["sampling"]["position_count"]:
        fail(errors, "position count differs from lock")
    if dict(role_counts) != lock["partition"]["roles"]:
        fail(errors, f"role counts differ from lock: {dict(role_counts)}")

    capture_summary = None
    if args.capture:
        capture = args.capture if args.capture.is_absolute() else REPO / args.capture
        manifest_path = capture / "capture_manifest.json"
        index_path = capture / "capture_index.csv"
        if not manifest_path.is_file() or not index_path.is_file():
            fail(errors, f"capture is missing manifest or index: {capture}")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("world_sha256") != lock["environment"]["world_sha256"]:
                fail(errors, "capture manifest world hash differs from campaign lock")
            if manifest.get("world_profiles_sha256") != lock["environment"]["capture_time_camera_profile_sha256"]:
                fail(errors, "capture manifest camera-profile hash differs from campaign lock")
            if manifest.get("plan", {}).get("pose_file_sha256") != lock["sampling"]["poses_sha256"]:
                fail(errors, "capture manifest pose hash differs from campaign lock")
            if int(manifest.get("plan", {}).get("planned_rows", -1)) != lock["sampling"]["expected_camera_opportunities"]:
                fail(errors, "capture manifest opportunity count differs from campaign lock")

            seen: set[tuple[int, int, str]] = set()
            status = Counter()
            roles_by_position: dict[int, set[str]] = defaultdict(set)
            camera_ids = {"camera_A", "camera_B", "camera_C", "camera_D", "camera_E"}
            row_count = 0
            with index_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    row_count += 1
                    key = (int(row["pose_id"]), int(row["repetition_id"]), row["camera_id"])
                    if key in seen:
                        fail(errors, f"duplicate opportunity at row {row_count}: {key}")
                    seen.add(key)
                    if row["camera_id"] not in camera_ids:
                        fail(errors, f"unknown camera at row {row_count}: {row['camera_id']}")
                    pose_id = int(row["pose_id"])
                    if pose_id < 0 or pose_id >= len(poses):
                        fail(errors, f"pose_id out of range at row {row_count}")
                        continue
                    expected = poses[pose_id]
                    if int(row["position_id"]) != int(expected["position_id"]):
                        fail(errors, f"position mismatch at row {row_count}")
                    if row["dataset_split"] != expected["stratum"]:
                        fail(errors, f"partition mismatch at row {row_count}")
                    roles_by_position[int(row["position_id"])].add(row["dataset_split"])
                    status[row["capture_status"]] += 1
            if any(len(v) != 1 for v in roles_by_position.values()):
                fail(errors, "captured rows split at least one physical position across roles")
            expected_rows = lock["sampling"]["expected_camera_opportunities"]
            if row_count > expected_rows:
                fail(errors, "capture contains more opportunities than scheduled")
            complete = row_count == expected_rows and status.get("ok", 0) == expected_rows
            if status.get("failed", 0):
                fail(errors, f"capture contains {status['failed']} infrastructure-failed rows; "
                     "these are not detector misses and must be recaptured")
            if args.require_complete and not complete:
                fail(errors, f"capture incomplete: {row_count}/{expected_rows} rows, statuses={dict(status)}")
            elif not complete:
                warnings.append(f"capture prefix is valid but incomplete: {row_count}/{expected_rows} rows")
            capture_summary = {
                "path": str(capture.relative_to(REPO)), "manifest_status": manifest.get("status"),
                "rows": row_count, "expected_rows": expected_rows, "status_counts": dict(status),
                "complete": complete,
            }

    report = {
        "schema": "reference_position_campaign_audit.v1",
        "lock": str(lock_path.relative_to(REPO)),
        "lock_sha256": sha256(lock_path),
        "ok": not errors,
        "hash_checks": hash_checks,
        "sampling": {"positions": len(by_position), "poses": len(poses), "roles": dict(role_counts)},
        "capture": capture_summary,
        "warnings": warnings,
        "errors": errors,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_output:
        out = args.json_output if args.json_output.is_absolute() else REPO / args.json_output
        out.write_text(text + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
