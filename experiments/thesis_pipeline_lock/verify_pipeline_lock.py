#!/usr/bin/env python3
"""Verify the current locked prefix of the final thesis evidence pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTRY = HERE / "pipeline_lock.json"
ALLOWED_STATUS = {"pending", "locked", "historical", "invalidated"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    failures = []
    stages = registry.get("stages", [])
    by_id = {stage.get("id"): stage for stage in stages}
    if len(by_id) != len(stages):
        failures.append("stage IDs are not unique")
    if [stage.get("id", "")[:2] for stage in stages] != [
        f"{index:02d}" for index in range(1, len(stages) + 1)
    ]:
        failures.append("stages are not in contiguous numeric order")

    for stage in stages:
        stage_id = stage.get("id", "<missing>")
        status = stage.get("status")
        if status not in ALLOWED_STATUS:
            failures.append(f"{stage_id}: invalid status {status!r}")
            continue
        for dependency in stage.get("depends_on", []):
            if dependency not in by_id:
                failures.append(f"{stage_id}: missing dependency {dependency}")
            elif status == "locked" and by_id[dependency].get("status") != "locked":
                failures.append(f"{stage_id}: locked before dependency {dependency}")
        if status == "locked":
            if not stage.get("lock_id") or not stage.get("manifest"):
                failures.append(f"{stage_id}: locked stage lacks lock ID or manifest")
                continue
            path = REPO / stage["manifest"]
            if not path.is_file():
                failures.append(f"{stage_id}: manifest missing: {path}")
            elif digest(path) != stage.get("manifest_sha256"):
                failures.append(f"{stage_id}: manifest hash drift")
        elif any(stage.get(key) is not None for key in (
            "lock_id", "manifest", "manifest_sha256"
        )):
            failures.append(f"{stage_id}: non-locked stage claims a frozen artifact")

    checks = [
        [sys.executable, "experiments/warehouse_v2_sketches/world_freeze.py"],
        [sys.executable, "experiments/thesis_pipeline_lock/robot_target_freeze.py"],
        [sys.executable, "experiments/thesis_pipeline_lock/capture_location_regime.py"],
    ]
    for command in checks:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        if result.returncode:
            failures.append(f"sub-lock failed: {' '.join(command)}\n{result.stdout}{result.stderr}")

    if failures:
        print("THESIS PIPELINE LOCK: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    locked = [stage["id"] for stage in stages if stage["status"] == "locked"]
    pending = [stage["id"] for stage in stages if stage["status"] == "pending"]
    print(f"THESIS PIPELINE LOCK: PASS {registry['pipeline_id']}")
    print(f"  locked: {', '.join(locked)}")
    print(f"  next: {pending[0] if pending else 'complete'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
