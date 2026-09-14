#!/usr/bin/env python3
"""Freeze quorum-only lost-belief recovery and complete collision accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments/reference_controlled_commissioning_v1"
BASE_CONFIG = ROOT / "final_route_selection_config_v9.yaml"
BASE_PROTOCOL = ROOT / "final_route_selection_protocol_v11.json"
OUTPUT_CONFIG = ROOT / "final_route_selection_config_v10.yaml"
OUTPUT_PROTOCOL = ROOT / "final_route_selection_protocol_v12.json"
REANCHOR_THRESHOLD_M = 0.30


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
    config["study_title"] = "Final commissioned four-arm route selection V10"
    config["state_reanchor_m"] = REANCHOR_THRESHOLD_M
    write_new(OUTPUT_CONFIG, yaml.safe_dump(config, sort_keys=False, width=100000))

    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol.update({
        "schema_version": 12,
        "status": "frozen_after_belief_divergence_and_collision-accounting_diagnostic",
        "supersedes": str(BASE_PROTOCOL.relative_to(REPO)),
        "amendment_reason": (
            "The V9 execution rejected a four-camera fused correction whose own-time "
            "error was 0.008 m, then continued on a diverged belief because runtime "
            "re-anchoring was disabled. The ground-truth chassis subsequently entered "
            "the loose-pallet geometry. Metric re-anchoring is enabled at 0.30 m only "
            "for schema-2 fused events supported by at least two accepted cameras; a "
            "single camera cannot snap the belief. The experiment logger now assigns "
            "all current-world collision prisms to the wall or obstacle audit instead "
            "of filtering them with obsolete model-name prefixes."
        ),
        "failed_execution_evidence": (
            "logs/studies/reference_controlled_commissioning_v1/"
            "final_navigation_pilot_t1_c00_v9"
        ),
        "belief_recovery_policy": {
            "innovation_threshold_m": REANCHOR_THRESHOLD_M,
            "minimum_accepted_cameras": 2,
            "single_camera_reanchor_allowed": False,
            "ordinary_nis_gate": 9.21,
        },
        "route_selection_config": str(OUTPUT_CONFIG.relative_to(REPO)),
        "route_selection_config_sha256": sha256(OUTPUT_CONFIG),
        "execution_logger": {
            "path": "src/experiments/experiments/nodes/experiment_logger.py",
            "sha256": sha256(
                REPO / "src/experiments/experiments/nodes/experiment_logger.py"
            ),
        },
    })
    roots = protocol["planner_source_tree"]["roots"]
    protocol["planner_source_tree"]["sha256"] = tree_sha256(roots)
    protocol["route_freezer"]["sha256"] = sha256(
        REPO / protocol["route_freezer"]["path"]
    )
    write_new(OUTPUT_PROTOCOL, json.dumps(protocol, indent=2) + "\n")
    print(OUTPUT_CONFIG.relative_to(REPO), sha256(OUTPUT_CONFIG))
    print(OUTPUT_PROTOCOL.relative_to(REPO), sha256(OUTPUT_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
