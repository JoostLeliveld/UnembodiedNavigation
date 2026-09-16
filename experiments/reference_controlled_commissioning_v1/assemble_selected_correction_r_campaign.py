#!/usr/bin/env python3
"""Assemble the no-contact U/W route release and one-seed runtime campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import yaml


REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "logs/studies/reference_controlled_commissioning_v1"
COMPONENTS = STUDY / "frozen_routes_no_contact_20260915_v2"
OUTPUT = STUDY / "selected_correction_r_frozen_routes_20260915_v5"
BASE = REPO / "experiments/reference_controlled_commissioning_v1/selected_correction_r_route_selection_v4.yaml"
PROTOCOL = REPO / "experiments/reference_controlled_commissioning_v1/selected_correction_r_route_protocol_v4.json"
CAMPAIGN = REPO / "experiments/reference_controlled_commissioning_v1/selected_correction_r_navigation_campaign_v5.yaml"
TASKS = (
    "thesis09_west_to_east_north",
    "thesis09_east_to_west_south",
    "thesis09_aisle_to_crossaisle",
    "thesis09_south_to_north_central",
)
ARMS = ("U0", "U1", "U2", "U3", "U4", "W0", "W3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if OUTPUT.exists() or CAMPAIGN.exists():
        raise RuntimeError("release outputs are immutable; choose a new version")
    OUTPUT.mkdir(parents=True)
    protocol_hash = sha256(PROTOCOL)
    selected: dict[str, dict[str, dict]] = {}
    components: dict[str, dict[str, str]] = {}
    output_files: dict[str, str] = {}
    for task in TASKS:
        selected[task] = {}
        for arm in ARMS:
            component_dir = COMPONENTS / f"{task}__{arm}"
            manifest_path = component_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or manifest.get("protocol_sha256") != protocol_hash:
                raise RuntimeError(f"invalid component manifest: {manifest_path}")
            entry = dict(manifest["selected"][task][arm])
            source = REPO / entry["preselected_route_source_path"]
            destination = OUTPUT / f"{task}__{arm}.json"
            shutil.copy2(source, destination)
            if sha256(destination) != entry["preselected_route_source_sha256"]:
                raise RuntimeError(f"route source hash mismatch: {source}")
            entry["preselected_route_source_path"] = str(destination.relative_to(REPO))
            entry["preselected_route_source_sha256"] = sha256(destination)
            selected[task][arm] = entry
            output_files[destination.name] = sha256(destination)
            components[f"{task}/{arm}"] = {
                "path": str(manifest_path.relative_to(REPO)), "sha256": sha256(manifest_path)
            }
    merged = {
        "schema_version": 1,
        "kind": "merged_selected_correction_r_route_selection_manifest",
        "status": "complete",
        "protocol_path": str(PROTOCOL.relative_to(REPO)),
        "protocol_sha256": protocol_hash,
        "selected": selected,
        "component_manifests": components,
        "output_files": dict(sorted(output_files.items())),
    }
    identity = json.dumps(merged, sort_keys=True, separators=(",", ":"))
    merged["selection_digest"] = hashlib.sha256(identity.encode()).hexdigest()
    merged_path = OUTPUT / "manifest.json"
    write_json(merged_path, merged)

    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config["study_title"] = "Provisional selected-correction R/q navigation campaign"
    config["study_comparison"] = (
        "One matched seed over U0-U4 and q=1 contrasts W0/W3 on hash-frozen routes."
    )
    config["route_selection_manifest_path"] = str(merged_path.relative_to(REPO))
    config["route_selection_manifest_sha256"] = sha256(merged_path)
    config["global_planner_mode"] = "preselected_route"
    config["camera_network_objective"] = "legacy_pixel_chart"
    config["global_optimizer_multistart"] = False
    config["optimizer_initial_routes_json"] = ""
    config["debug_runtime"] = False
    config["ros_domain_id_base"] = 190
    for arm in ARMS:
        config["conditions"][arm].pop("camera_network_artifact_path", None)
    for task in TASKS:
        routes = {}
        for arm in ARMS:
            entry = selected[task][arm]
            routes[arm] = {
                key: entry[key] for key in (
                    "preselected_route_json", "preselected_route_sha256",
                    "preselected_route_source_path", "preselected_route_source_sha256",
                )
            }
            routes[arm].update({
                "preselected_route_clearance_m": 0.25,
                "preselected_route_endpoint_tolerance_m": 0.0,
                "preselected_route_sample_step_m": 0.04,
            })
        config["tasks"][task] = {
            "conditions": list(ARMS), "seeds": [91500], "preselected_routes": routes
        }
    CAMPAIGN.write_text(yaml.safe_dump(config, sort_keys=False, width=100000), encoding="utf-8")
    print(merged_path, sha256(merged_path))
    print(CAMPAIGN, sha256(CAMPAIGN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
