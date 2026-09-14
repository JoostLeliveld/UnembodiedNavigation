#!/usr/bin/env python3
"""Merge the sixteen immutable V24 single-cell route releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "logs/studies/reference_controlled_commissioning_v1"
PROTOCOL = REPO / "experiments/reference_controlled_commissioning_v1/final_route_selection_protocol_v24.json"
OUTPUT = STUDY / "final_frozen_routes_20260914_v24_complete"
TASKS = (
    ("t1", "thesis09_west_to_east_north"),
    ("t2", "thesis09_east_to_west_south"),
    ("t3", "thesis09_aisle_to_crossaisle"),
    ("t4", "thesis09_south_to_north_central"),
)
ARMS = ("C00", "C01", "C10", "C11")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"immutable release already exists: {OUTPUT}")
    protocol_hash = sha256(PROTOCOL)
    OUTPUT.mkdir(parents=True)
    selected: dict[str, dict[str, dict]] = {}
    output_files: dict[str, str] = {}
    component_manifests: dict[str, dict[str, str]] = {}
    for task_tag, task in TASKS:
        selected[task] = {}
        for arm in ARMS:
            source_dir = STUDY / f"final_frozen_route_20260914_v24_{task_tag}_{arm.lower()}"
            manifest_path = source_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise RuntimeError(f"incomplete component: {manifest_path}")
            if manifest.get("protocol_sha256") != protocol_hash:
                raise RuntimeError(f"protocol mismatch: {manifest_path}")
            entry = dict(manifest["selected"][task][arm])
            source_path = REPO / entry["preselected_route_source_path"]
            filename = f"{task}__{arm}.json"
            destination = OUTPUT / filename
            shutil.copy2(source_path, destination)
            copied_hash = sha256(destination)
            if copied_hash != entry["preselected_route_source_sha256"]:
                raise RuntimeError(f"source hash mismatch: {source_path}")
            entry["preselected_route_source_path"] = str(destination.relative_to(REPO))
            entry["preselected_route_source_sha256"] = copied_hash
            selected[task][arm] = entry
            output_files[filename] = copied_hash
            for suffix in (".png", ".pdf"):
                figure_source = source_dir / f"{task}__{arm}_route{suffix}"
                figure_name = f"{task}__{arm}_route{suffix}"
                shutil.copy2(figure_source, OUTPUT / figure_name)
                output_files[figure_name] = sha256(OUTPUT / figure_name)
            cell = f"{task}/{arm}"
            component_manifests[cell] = {
                "path": str(manifest_path.relative_to(REPO)),
                "sha256": sha256(manifest_path),
            }
    manifest = {
        "schema_version": 1,
        "kind": "merged_expected_belief_route_selection_manifest",
        "status": "complete",
        "protocol_path": str(PROTOCOL.relative_to(REPO)),
        "protocol_sha256": protocol_hash,
        "selected": selected,
        "component_manifests": component_manifests,
        "output_files": dict(sorted(output_files.items())),
    }
    identity = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["selection_digest"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    write_json(OUTPUT / "manifest.json", manifest)
    print(OUTPUT / "manifest.json", sha256(OUTPUT / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
