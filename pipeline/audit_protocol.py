#!/usr/bin/env python3
"""Freeze the v8 final-audit protocol after the refit.

Copies the v5 protocol and replaces every locked input with the v8 uniform artifacts. The
audit population is the v5 final_audit set, unchanged. The D_dev evaluation manifest is locked too, because it carries the
R_proj sigma_px fitted on D_R. Writing this file is the last step before final_audit is
opened, so it refuses to run while any locked input is missing.

    python3 pipeline/audit_protocol.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "pipeline"
V8 = "logs/thesis_final_pipeline_v1/recapture_v8_uniform"
LOCK = "pipeline/dataset_lock.json"


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def main() -> int:
    protocol = json.loads((HERE / "final_audit_protocol_template.json").read_text())
    lock = json.loads((REPO / LOCK).read_text())
    expected = lock["opportunity_accounting"]["expected_final_audit_opportunities"]
    protocol["population"] = {
        "positions": lock["partition"]["roles"]["final_audit"],
        "opportunities": expected,
        "merge_loader": "pipeline/combined_recapture_v8.py",
    }
    inputs = {
        "campaign_lock": LOCK,
        "gate_config": "config/sensor_gate.yaml",
        "label_protocol": "pipeline/detector/label_protocol.json",
        "detector": lock["detector"]["checkpoint"],
        "runtime_R0": f"{V8}/runtime_r012/R0_global_full.json",
        "runtime_R1": f"{V8}/runtime_r012/R1_per_camera_full.json",
        "runtime_R2": f"{V8}/runtime_r012/R2_spatial_residual.json",
        "planning_M0": f"{V8}/planning_precision/m0_planning_precision.npz",
        "planning_M1": f"{V8}/planning_precision/m1_planning_precision.npz",
        "planning_M2": f"{V8}/planning_precision/m2_planning_precision.npz",
        "ddev_evaluation": f"{V8}/ddev_evaluation/manifest.json",
        "implementation": "pipeline/final_audit.py",
    }
    protocol["locked_inputs"] = {key: {"path": rel, "sha256": sha(rel)}
                                 for key, rel in inputs.items()}
    protocol["amendment"] = "docs/METHOD.md, Amendment 2026-09-23"
    protocol["status"] = "frozen_before_final_audit_access"
    out = HERE / "reference_final_audit_protocol_v8_uniform.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
