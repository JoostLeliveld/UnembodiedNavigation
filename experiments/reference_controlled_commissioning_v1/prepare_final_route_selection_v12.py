#!/usr/bin/env python3
"""Freeze sparse control knots on the original one-second rollout grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v12.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v14.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v13.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v15.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(relative_roots: list[str]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for root in relative_roots
        for path in (REPO / root).rglob("*.py") if path.is_file()
    )
    for path in files:
        relative = str(path.relative_to(REPO)).encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"immutable release already exists: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["study_title"] = "Final commissioned four-arm sparse-knot route selection V13"
    # Preserve the thesis-facing 1 s rollout and 200 s horizon. Only the free
    # optimization variables are sparse: one [v,w] pair for every two steps.
    config["global_horizon"] = 200
    config["global_dt"] = 1.0
    config["camera_network_updates_per_step"] = 5
    config["optimizer_control_block_steps"] = 2
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 15,
        "status": "frozen_before_sparse_knot_route_selection_and_four_arm_pilots",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The two-second integration-grid seed check showed unacceptable corner "
            "discretization and was not optimized. V15 restores the original 200-step, "
            "one-second dynamics, belief, and safety rollout. CasADi optimizes 100 "
            "piecewise-constant two-step control blocks and expands them to all 200 "
            "states before objective accounting and swept collision validation."
        ),
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "purpose": (
            "Select one hash-bound route per task and arm using the launch-resolved "
            "CasADi/L-BFGS-B expected-belief planner with 100 sparse control blocks, "
            "a 200-step one-second rollout, and five camera opportunities per step."
        ),
        "sparse_global_parameterization": {
            "rollout_steps": 200,
            "rollout_step_s": 1.0,
            "physical_horizon_s": 200.0,
            "control_block_steps": 2,
            "free_control_blocks": 100,
            "free_scalar_variables": 200,
            "camera_opportunities_per_rollout_step": 5,
            "camera_opportunities_per_second": 5,
            "maximum_obstacle_cost_sample_spacing_m": 0.20,
            "independent_frozen_route_sweep_spacing_m": 0.04,
        },
    })
    protocol["route_selection_config_sha256"] = sha256(OUTPUT_CONFIG)
    protocol["runtime_estimator_contract"]["camera_network_updates_per_global_step"] = 5
    protocol["selector_sha256"] = sha256(REPO / protocol["selector_path"])
    protocol["launch_resolution"]["sha256"] = sha256(
        REPO / protocol["launch_resolution"]["path"]
    )
    protocol["route_probe"]["sha256"] = sha256(REPO / protocol["route_probe"]["path"])
    protocol["route_freezer"]["sha256"] = sha256(REPO / protocol["route_freezer"]["path"])
    protocol["planner_source_tree"]["sha256"] = tree_sha256(
        protocol["planner_source_tree"]["roots"]
    )
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
