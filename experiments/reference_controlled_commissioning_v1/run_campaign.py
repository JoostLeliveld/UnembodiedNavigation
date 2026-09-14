#!/usr/bin/env python3
"""Run the frozen commissioning drives sequentially and fail closed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "campaign.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pilot_passed_frozen_before_collection":
        raise RuntimeError(
            "scientific collection requires status=pilot_passed_frozen_before_collection"
        )
    drives = sorted(protocol["drives"], key=lambda item: int(item["order"]))
    if len(drives) != 24:
        raise RuntimeError(f"expected 24 frozen drives, found {len(drives)}")
    if args.dry_run:
        print("\n".join(f"{int(d['order']):02d} {d['id']}" for d in drives))
        return 0

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    progress_path = output_root / "campaign_execution.json"
    progress = {
        "schema": "reference_controlled_commissioning_execution.v1",
        "protocol": str(protocol_path),
        "output_root": str(output_root),
        "started_wall_unix_s": time.time(),
        "status": "running",
        "completed_drive_ids": [],
        "failed_drive_id": None,
        "current_drive_id": None,
        "audit_analysis_permitted": False,
    }
    atomic_json(progress_path, progress)

    for drive in drives:
        drive_id = str(drive["id"])
        progress["current_drive_id"] = drive_id
        atomic_json(progress_path, progress)
        command = [
            sys.executable,
            str(HERE / "run_drive.py"),
            "--protocol", str(protocol_path),
            "--drive-id", drive_id,
            "--output-root", str(output_root),
        ]
        completed = subprocess.run(command, cwd=REPO, check=False)
        if completed.returncode != 0:
            progress["status"] = "failed_closed"
            progress["failed_drive_id"] = drive_id
            progress["current_drive_id"] = None
            progress["finished_wall_unix_s"] = time.time()
            atomic_json(progress_path, progress)
            return completed.returncode
        progress["completed_drive_ids"].append(drive_id)
        atomic_json(progress_path, progress)

    progress["status"] = "collection_complete_audit_sealed"
    progress["current_drive_id"] = None
    progress["finished_wall_unix_s"] = time.time()
    atomic_json(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
