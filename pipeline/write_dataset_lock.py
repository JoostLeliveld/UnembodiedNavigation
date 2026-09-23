#!/usr/bin/env python3
"""(Re)write the v8 campaign lock from the v8 index: roles, counts and source hashes.

The index itself (dataset.load_rows) refuses any robot-absent run, so a
lock is only written for a dataset that passed the presence check.

    python3 pipeline/write_dataset_lock.py
"""
from __future__ import annotations

import collections
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline import dataset as v8  # noqa: E402

HERE = REPO / "pipeline"
R = "logs/thesis_final_pipeline_v1/recapture_v8_uniform"


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def main() -> int:
    rows = v8.load_rows()
    work = [r for r in rows if r["stratum"] in ("D_mu", "D_R", "D_dev")]
    audit = [r for r in rows if r["stratum"] == "final_audit"]
    positions: dict[str, dict] = {}
    for r in rows:
        p = positions.setdefault(r["position_key"], {
            "position_key": r["position_key"], "x": round(float(r["robot_x"]), 4),
            "y": round(float(r["robot_y"]), 4), "role": r["stratum"],
            "source": r["capture_source"], "camera_opportunities": 0})
        p["camera_opportunities"] += 1
    table = REPO / R / "capture_positions_v8.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(positions.values()))))
        writer.writeheader()
        writer.writerows(sorted(positions.values(), key=lambda p: p["position_key"]))
    roles = collections.Counter(p["role"] for p in positions.values())

    base = "pipeline/v5_dataset_lock.json"
    lock = copy.deepcopy(json.loads((REPO / base).read_text()))
    lock.update({
        "lock_id": "THESIS-REFERENCE-POSITION-CAMPAIGN-V8-UNIFORM", "status": "locked_before_refit",
        "campaign_root": R, "amendment": "docs/METHOD.md, Amendment 2026-09-23",
        "derived_from": {"lock": base, "sha256": sha(base)},
    })
    lock["sampling"].update({
        "loader": "pipeline/dataset.py",
        "positions": f"{R}/capture_positions_v8.csv", "positions_sha256": sha(f"{R}/capture_positions_v8.csv"),
        "topup_rule": "pipeline/capture/plan_topup.py",
        "robot_absent_list": f"{R}/v5_robot_absent_poses.json",
        "robot_absent_list_sha256": sha(f"{R}/v5_robot_absent_poses.json"),
        **{f"{name}_capture_index_sha256": sha(f"{R}/master_capture_{name}/capture_index.csv")
           for name in ("supplement", "topup", "repair")},
        "position_count": len(positions), "expected_camera_opportunities": len(rows),
        "expected_pose_batches": len(rows) // 5,
    })
    lock["partition"]["roles"] = dict(sorted(roles.items()))
    lock["partition"]["added_position_rule"] = (
        "each added position takes the role of its nearest v5 position; none borders final_audit")
    lock["opportunity_accounting"].update({
        "expected_working_opportunities": len(work),
        "expected_working_unique_images": len({r["image_sha1"] for r in work}),
        "expected_final_audit_opportunities": len(audit),
    })
    out = HERE / "dataset_lock.json"
    out.write_text(json.dumps(lock, indent=1) + "\n")
    print(out.name, dict(sorted(roles.items())), len(rows), len(work), len(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
