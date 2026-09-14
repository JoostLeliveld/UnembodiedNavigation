#!/usr/bin/env python3
"""Recoverably remove a contiguous failed tail before capture resume.

The original index and manifest are archived inside capture provenance, and the
attempt ledger remains untouched.  Only all-camera failed batches at the end of
an otherwise coherent prefix are eligible; successful or interior batches are
never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/unav_common"))
from unav_common.capture_integrity import atomic_bytes, atomic_csv, atomic_json, capture_lock  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()

    root = args.capture.expanduser().resolve()
    index_path = root / "capture_index.csv"
    manifest_path = root / "capture_manifest.json"
    attempts_path = root / "capture_attempts.jsonl"
    with capture_lock(root):
        index_bytes = index_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        attempts_bytes = attempts_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("status") != "running":
            raise RuntimeError("tail retry requires a status=running capture")
        with index_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        if not fields or not rows:
            raise RuntimeError("capture index is empty or lacks a header")

        batches: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in rows:
            batches[(int(row["pose_id"]), int(row["repetition_id"]))].append(row)
        ordered = sorted(batches)
        failed = [key for key in ordered if {r["capture_status"] for r in batches[key]} == {"failed"}]
        if not failed:
            raise RuntimeError("capture has no all-camera failed batches to retry")
        first_failed = failed[0]
        expected_tail = ordered[ordered.index(first_failed):]
        if failed != expected_tail:
            raise RuntimeError("failed batches are not one contiguous tail")
        camera_count = int(manifest["plan"]["camera_count"])
        if any(len(batches[key]) != camera_count for key in failed):
            raise RuntimeError("failed tail contains an incomplete camera transaction")
        if any(row.get("image") or row.get("robot_mask") for key in failed for row in batches[key]):
            raise RuntimeError("failed tail unexpectedly references captured files")

        retained = [row for row in rows if (int(row["pose_id"]), int(row["repetition_id"])) < first_failed]
        archive = root / "provenance" / f"retry_failed_tail_pose_{first_failed[0]:06d}_r{first_failed[1]:02d}"
        archive.mkdir(parents=True, exist_ok=False)
        atomic_bytes(archive / "capture_index.before_retry.csv", index_bytes)
        atomic_bytes(archive / "capture_manifest.before_retry.json", manifest_bytes)
        atomic_bytes(archive / "capture_attempts.before_retry.jsonl", attempts_bytes)
        report = {
            "schema": "capture_failed_tail_retry.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "capture": str(root),
            "first_retried_pose_id": first_failed[0],
            "first_retried_repetition_id": first_failed[1],
            "failed_batches_removed_from_active_index": [
                {"pose_id": pose, "repetition_id": repetition}
                for pose, repetition in failed
            ],
            "failed_rows_removed_from_active_index": sum(len(batches[key]) for key in failed),
            "successful_rows_retained": len(retained),
            "evidence_retention": "Original index, manifest and attempt ledger archived; live attempt ledger unchanged.",
            "before_sha256": {
                "capture_index.csv": sha256(index_bytes),
                "capture_manifest.json": sha256(manifest_bytes),
                "capture_attempts.jsonl": sha256(attempts_bytes),
            },
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).read_bytes()),
        }
        atomic_json(archive / "recovery_report.json", report)
        atomic_csv(index_path, retained, fields)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
