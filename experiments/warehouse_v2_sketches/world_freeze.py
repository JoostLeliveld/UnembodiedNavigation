#!/usr/bin/env python3
"""Build or verify the reproducible warehouse_v2 geometry freeze manifest."""
from __future__ import annotations

import argparse
import hashlib
import datetime as _dt
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from make_world import CAMERA_MODELS, WORLD_FILES, WORLD_NAMES, build_world  # noqa: E402

MANIFEST_PATH = HERE / "world_freeze_manifest.json"
WORLD_DIR = REPO / "src/sim/gazebo_worlds/worlds"
PROFILES_PATH = REPO / "src/experiments/config/world_profiles.yaml"
TASKS_PATH = REPO / "src/experiments/config/tasks.yaml"
CAMERA_LAYOUT_DECISION = HERE / "camera_layout_decision.json"
SOURCE_PATHS = (
    HERE / "warehouse_v2.py",
    HERE / "make_world.py",
    HERE / "coverage.py",
    HERE / "route_tasks.py",
    HERE / "verify_world.py",
    HERE / "camera_layout_gate.py",
    HERE / "world_freeze.py",
    HERE / "sim_up.sh",
    CAMERA_LAYOUT_DECISION,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(payload.encode("utf-8"))


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def _camera_includes(world: ET.Element) -> dict:
    wanted = set(CAMERA_MODELS.values())
    records = {}
    for include in world.findall("include"):
        name = include.findtext("name", "").strip()
        if name not in wanted:
            continue
        pose = [float(token) for token in include.findtext("pose", "").split()]
        records[name] = {
            "uri": include.findtext("uri", "").strip(),
            "pose_xyz_rpy_rad": pose,
        }
    return dict(sorted(records.items()))


def _model_xml(world: ET.Element, name: str) -> bytes:
    model = world.find(f"model[@name='{name}']")
    return ET.tostring(model) if model is not None else b""


def _world_record(state: str) -> tuple[dict, ET.Element]:
    filename = WORLD_FILES[state]
    path = WORLD_DIR / filename
    payload = path.read_bytes()
    world = ET.fromstring(payload).find("world")
    if world is None:
        raise RuntimeError(f"{filename} has no <world> element")
    contacts = sum(
        1 for sensor in world.iter("sensor") if sensor.get("type") == "contact"
    )
    occluder = world.find("model[@name='warehouse_v2_occluders']")
    occluder_links = len(occluder.findall("link")) if occluder is not None else 0
    generated = build_world(state).encode("utf-8")
    return ({
        "file": _relative(path),
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "internal_world_name": world.get("name"),
        "expected_internal_world_name": WORLD_NAMES[state],
        "generator_matches_file": generated == payload,
        "stock_state": "peak" if state == "A" else "shipout",
        "occluder_links": occluder_links,
        "contact_sensors": contacts,
        "camera_includes": _camera_includes(world),
    }, world)


def build_manifest() -> dict:
    profiles = yaml.safe_load(PROFILES_PATH.read_text())["worlds"]
    tasks = yaml.safe_load(TASKS_PATH.read_text())["tasks"]
    decision = json.loads(CAMERA_LAYOUT_DECISION.read_text())

    world_records = {}
    parsed_worlds = {}
    for state in ("A", "B"):
        record, world = _world_record(state)
        world_records[state] = record
        parsed_worlds[state] = world

    paired_model_hashes = {}
    for model_name in (
        "warehouse_shell",
        "warehouse_v2_rack_frames",
        "warehouse_v2_floor_markings",
    ):
        hashes = {
            state: _sha256(_model_xml(parsed_worlds[state], model_name))
            for state in ("A", "B")
        }
        paired_model_hashes[model_name] = {
            "state_hashes": hashes,
            "identical": hashes["A"] == hashes["B"],
        }

    profile_records = {}
    task_records = {}
    for filename in WORLD_FILES.values():
        profile = profiles[filename]
        world_tasks = tasks[filename]
        profile_records[filename] = {
            "canonical_sha256": _json_sha256(profile),
            "world_name": profile["world_name"],
            "stock_state": profile["stock_state"],
            "evidence_role": profile["evidence_role"],
            "recommended_task": profile["recommended_task"],
            "camera_ids": profile["camera_ids"],
            "camera_model_includes": profile["camera_model_includes"],
            "camera_image_topics": profile["camera_image_topics"],
        }
        task_records[filename] = {
            "canonical_sha256": _json_sha256(world_tasks),
            "task_names": [item["name"] for item in world_tasks],
        }

    return {
        "schema_version": 1,
        "freeze_id": "WHV2-GEOMETRY-V2",
        "camera_layout_decision_id": decision["decision_id"],
        "selected_camera_layout": decision["selected_candidate"],
        "evidence_boundary": "geometry_and_runtime_contract_only_no_detector_data",
        "sources": {
            _relative(path): _file_sha256(path)
            for path in SOURCE_PATHS
        },
        "worlds": world_records,
        "paired_invariants": paired_model_hashes,
        "profiles": profile_records,
        "tasks": task_records,
    }


def _diff(expected, actual, path="manifest") -> list[str]:
    failures = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            failures.append(
                f"{path} keys differ: expected {sorted(expected)}, got {sorted(actual)}"
            )
            return failures
        for key in expected:
            failures.extend(_diff(expected[key], actual[key], f"{path}.{key}"))
    elif isinstance(expected, list) and isinstance(actual, list):
        if expected != actual:
            failures.append(f"{path} list changed")
    elif expected != actual:
        failures.append(f"{path}: expected {expected!r}, got {actual!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print",
        action="store_true",
        help="print the current manifest candidate; does not modify the freeze",
    )
    parser.add_argument(
        "--refreeze",
        metavar="REASON",
        default="",
        help=(
            "REPLACE the frozen manifest with the current world, recording REASON. "
            "Only legitimate for an intentional change made BEFORE any image of this "
            "world has been captured -- never to make a mismatch go away after seeing "
            "a result. The reason is appended to the manifest's own history."
        ),
    )
    args = parser.parse_args()
    actual = build_manifest()
    if args.print:
        print(json.dumps(actual, indent=2, sort_keys=True))
        return 0
    if args.refreeze:
        previous = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
        history = list(previous.get("refreeze_history", []))
        history.append({
            "date": _dt.date.today().isoformat(),
            "reason": str(args.refreeze),
            "replaced_world_sha256": {
                state: record.get("sha256", "")
                for state, record in (previous.get("worlds") or {}).items()
            },
        })
        actual["refreeze_history"] = history
        MANIFEST_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print("WAREHOUSE_V2 FREEZE: REFROZEN")
        print(f"  reason: {args.refreeze}")
        for state, record in actual["worlds"].items():
            print(f"  state {state}: sha256={record['sha256'][:12]}...")
        return 0

    expected = json.loads(MANIFEST_PATH.read_text())
    # refreeze_history records WHY the freeze was replaced. It is provenance about
    # the manifest, not a frozen property of the world, so it is not compared.
    history = expected.pop("refreeze_history", [])
    failures = _diff(expected, actual)
    failures.extend(
        f"world {state} no longer reproduces byte-for-byte from make_world.py"
        for state, record in actual["worlds"].items()
        if not record["generator_matches_file"]
    )
    failures.extend(
        f"paired invariant {name} differs between peak and shipout"
        for name, record in actual["paired_invariants"].items()
        if not record["identical"]
    )
    if failures:
        print("WAREHOUSE_V2 FREEZE: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("WAREHOUSE_V2 FREEZE: PASS")
    print(f"  freeze: {actual['freeze_id']}")
    for entry in history:
        print(f"  refrozen {entry.get('date', '?')}: {str(entry.get('reason', ''))[:96]}")
    for state, record in actual["worlds"].items():
        print(
            f"  state {state}: {record['internal_world_name']} "
            f"sha256={record['sha256'][:12]}... cameras={len(record['camera_includes'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
