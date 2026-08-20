#!/usr/bin/env python3
"""Resolve and validate the prospective CBAF-WV2 campaign manifest.

This script is deliberately ROS-independent.  It does not run an experiment and
does not inspect result folders.  It turns the registered protocol into an exact
schedule, resolves hashes of the current warehouse/method artifacts, and rejects
stale worlds, leakage, missing units, or a modified apparatus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / "protocol.json"
CAMERA_IDS = ("A", "B", "C", "D", "E")
STAGE_IDS = tuple(f"G{i}" for i in range(7))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """The prospective campaign contract is incomplete or has been violated."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"top-level JSON must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ContractError(f"artifact path must be repository-relative: {relative!r}")
    resolved = (REPO / rel).resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as exc:
        raise ContractError(f"artifact escapes repository: {relative!r}") from exc
    return resolved


def _assert_exact_ids(actual: Iterable[str], expected: Iterable[str], label: str) -> None:
    actual_list = list(actual)
    expected_list = list(expected)
    if len(actual_list) != len(set(actual_list)):
        raise ContractError(f"duplicate {label} IDs")
    missing = sorted(set(expected_list) - set(actual_list))
    extra = sorted(set(actual_list) - set(expected_list))
    if missing or extra:
        raise ContractError(
            f"{label} schedule mismatch: missing={missing[:8]} extra={extra[:8]}"
        )


def _session_split(protocol: dict[str, Any], stage: str, session: int) -> str:
    split = protocol["splits"][f"{stage.lower()}_sessions"]
    memberships = [name for name, values in split.items() if session in values]
    if len(memberships) != 1:
        raise ContractError(
            f"{stage} session {session} must occur in exactly one split; got {memberships}"
        )
    return memberships[0]


def _validate_splits(protocol: dict[str, Any]) -> None:
    for stage in ("G1", "G2"):
        split = protocol["splits"].get(f"{stage.lower()}_sessions")
        if not isinstance(split, dict) or set(split) != {"train", "validation", "test"}:
            raise ContractError(f"{stage} must define train/validation/test sessions")
        values = {name: set(items) for name, items in split.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = values[left] & values[right]
            if overlap:
                raise ContractError(f"{stage} split leakage between {left}/{right}: {sorted(overlap)}")
        expected = protocol["sample_size_targets"][stage]["sessions"]
        flattened = set().union(*values.values())
        if flattened != set(range(1, expected + 1)):
            raise ContractError(
                f"{stage} sessions must be exactly 1..{expected}; got {sorted(flattened)}"
            )


def _world_audit(relative: str, model_names: dict[str, str]) -> dict[str, Any]:
    path = _repo_path(relative)
    if not path.is_file():
        raise ContractError(f"missing world artifact: {relative}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ContractError(f"invalid SDF/XML in {relative}: {exc}") from exc
    world = root.find("world") if root.tag != "world" else root
    if world is None:
        raise ContractError(f"no <world> element in {relative}")
    includes: dict[str, str] = {}
    for include in world.findall("include"):
        name = include.findtext("name")
        pose = include.findtext("pose")
        if name:
            includes[name.strip()] = (pose or "").strip()
    missing = [cid for cid, model in model_names.items() if model not in includes]
    if missing:
        raise ContractError(f"{relative} lacks camera models for IDs {missing}")
    camera_poses = {cid: includes[model] for cid, model in model_names.items()}
    if len(set(camera_poses.values())) != len(camera_poses):
        raise ContractError(f"camera poses are not unique in {relative}")
    return {
        "path": relative,
        "sdf_world_name": world.attrib.get("name", ""),
        "camera_poses": camera_poses,
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ContractError("unsupported protocol schema_version")
    if protocol.get("status") not in {"prospective_unfrozen", "prospective_frozen"}:
        raise ContractError("protocol status must remain prospective")
    repo = protocol.get("repository", {})
    model_names = repo.get("camera_model_names", {})
    if tuple(model_names) != CAMERA_IDS:
        raise ContractError(f"camera IDs/order must be exactly {CAMERA_IDS}")
    worlds = repo.get("world_artifacts", [])
    basenames = [Path(item.get("path", "")).name for item in worlds]
    if sorted(basenames) != sorted(repo.get("allowed_world_basenames", [])):
        raise ContractError("world artifacts must be exactly the registered v2 and shipout files")
    forbidden = tuple(repo.get("forbidden_legacy_tokens", []))
    for item in worlds:
        path = item.get("path", "")
        if any(token.lower() in path.lower() for token in forbidden):
            raise ContractError(f"legacy world/dataset token in artifact: {path}")
        _world_audit(path, model_names)
    for relative in repo.get("method_artifacts", []):
        if not _repo_path(relative).is_file():
            raise ContractError(f"missing method artifact: {relative}")
    _validate_splits(protocol)
    stage_ids = [stage.get("id") for stage in protocol.get("stages", [])]
    if tuple(stage_ids) != STAGE_IDS:
        raise ContractError(f"stages must be ordered exactly {STAGE_IDS}")
    baseline_ids = [arm.get("id") for arm in protocol.get("baselines", [])]
    expected_baselines = [
        "B1_best_single",
        "B2_scalar_R",
        "B3_calibrated_R_independent",
        "B4_static_offset",
        "B5_NIS_gating",
        "B6a_covariance_intersection",
        "B6b_split_covariance_intersection",
        "B7_proposed_observability_gated_drift",
        "B8_oracle_exclusion",
    ]
    if baseline_ids != expected_baselines:
        raise ContractError("baseline IDs/order changed from the registered comparison")
    metric = protocol.get("metrics", {}).get("metric_contract", {})
    if metric.get("mean_nees_reference_2d") != 2.0:
        raise ContractError("2D mean NEES reference must be 2.0")
    if metric.get("median_nees_reference_2d") != 1.386:
        raise ContractError("2D median NEES reference must be 1.386")
    measurement = protocol.get("measurement_model", {})
    if measurement.get("dimension") != 4 or len(measurement.get("vector", [])) != 4:
        raise ContractError("measurement model must preserve the full 4D two-keypoint vector")
    availability_text = json.dumps(protocol.get("availability_model", {})).lower()
    if "binary" not in availability_text or "miss" not in availability_text:
        raise ContractError("availability must be a miss-inclusive binary opportunity model")
    if "detector confidence as a gp target" not in availability_text:
        raise ContractError("protocol must explicitly prohibit confidence as the GP target")
    online = set(protocol.get("information_boundary", {}).get("online_allowed", []))
    forbidden_online = set(protocol.get("information_boundary", {}).get("forbidden_online", []))
    if online & forbidden_online:
        raise ContractError(f"online/evaluation leakage in protocol: {sorted(online & forbidden_online)}")
    if any(field.startswith("gt_") or "ground_truth" in field for field in online):
        raise ContractError("GT field declared online")


def expected_schedule(protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build the exact experimental-unit schedule without reading any result data."""
    schedule: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGE_IDS}
    schedule["G0"] = [
        {"unit_id": "g0-primary-world-load", "kind": "apparatus_gate"},
        {"unit_id": "g0-shipout-world-load", "kind": "apparatus_gate"},
        {"unit_id": "g0-generic-a-e-capture", "kind": "commissioning_capture_gate"},
        {"unit_id": "g0-closed-loop-a-e-runtime", "kind": "runtime_gate"},
        {"unit_id": "g0-gt-offline-boundary", "kind": "information_boundary_gate"},
        {"unit_id": "g0-route-menu", "kind": "geometry_gate"},
    ]

    n_g1 = protocol["sample_size_targets"]["G1"]["sessions"]
    for session in range(1, n_g1 + 1):
        schedule["G1"].append({
            "unit_id": f"g1-session-{session:02d}",
            "kind": "independent_capture_session",
            "session": session,
            "split": _session_split(protocol, "G1", session),
            "camera_ids": list(CAMERA_IDS),
        })

    n_g2 = protocol["sample_size_targets"]["G2"]["sessions"]
    for session in range(1, n_g2 + 1):
        schedule["G2"].append({
            "unit_id": f"g2-session-{session:02d}",
            "kind": "miss_inclusive_opportunity_session",
            "session": session,
            "split": _session_split(protocol, "G2", session),
            "camera_ids": list(CAMERA_IDS),
        })

    levels = protocol["drift_model"]["injection_levels"]
    g3_seeds = protocol["sample_size_targets"]["G3"]["seeds_per_injection_cell"]
    for camera in CAMERA_IDS:
        for seed in range(1, g3_seeds + 1):
            schedule["G3"].append({
                "unit_id": f"g3-{camera}-control-seed-{seed:02d}",
                "kind": "no_injection_control_episode",
                "camera_id": camera,
                "seed": seed,
                "split": "development" if seed <= 3 else "test",
            })
        for mode, magnitudes in levels.items():
            for level_index, magnitude in enumerate(magnitudes, start=1):
                for sign in (-1, 1):
                    sign_id = "neg" if sign < 0 else "pos"
                    for seed in range(1, g3_seeds + 1):
                        schedule["G3"].append({
                            "unit_id": (
                                f"g3-{camera}-{mode}-L{level_index}-{sign_id}-seed-{seed:02d}"
                            ),
                            "kind": "physical_extrinsic_injection_episode",
                            "camera_id": camera,
                            "mode": mode,
                            "magnitude": magnitude,
                            "sign": sign,
                            "seed": seed,
                            "split": "development" if seed <= 3 else "test",
                        })

    g4 = protocol["sample_size_targets"]["G4"]
    for session in range(1, g4["capture_sessions"] + 1):
        for route in range(1, g4["route_profiles"] + 1):
            schedule["G4"].append({
                "unit_id": f"g4-session-{session:02d}-route-{route:02d}",
                "kind": "heldout_capture_stream",
                "session": session,
                "route_profile": f"R{route}",
                "split": "test",
                "replay_arms": list(protocol["stage_arms"]["G4"]),
            })

    g5 = protocol["sample_size_targets"]["G5"]
    for task in protocol["task_resolution"]["symbolic_handover_slots"]:
        for seed in range(1, g5["seeds_per_task"] + 1):
            for arm in protocol["stage_arms"]["G5"]:
                schedule["G5"].append({
                    "unit_id": f"g5-{task}-seed-{seed:02d}-{arm}",
                    "kind": "closed_loop_handover_run",
                    "task_slot": task,
                    "seed": seed,
                    "arm": arm,
                    "split": "test",
                    "world_role": "primary_world",
                })

    g6 = protocol["sample_size_targets"]["G6"]
    for world_role in ("primary_world", "shipout_world"):
        for task in protocol["task_resolution"]["symbolic_planner_slots"]:
            for seed in range(1, g6["seeds_per_task"] + 1):
                for arm in protocol["stage_arms"]["G6"]:
                    schedule["G6"].append({
                        "unit_id": f"g6-{world_role}-{task}-seed-{seed:02d}-{arm}",
                        "kind": "fixed_efe_downstream_run",
                        "task_slot": task,
                        "seed": seed,
                        "arm": arm,
                        "split": "test",
                        "world_role": world_role,
                    })
    return schedule


def _artifact_records(protocol_path: Path, protocol: dict[str, Any]) -> list[dict[str, str]]:
    records = [{
        "role": "protocol",
        "path": str(protocol_path.resolve().relative_to(REPO.resolve())),
        "sha256": _sha256(protocol_path),
    }]
    for item in protocol["repository"]["world_artifacts"]:
        path = _repo_path(item["path"])
        records.append({"role": item["role"], "path": item["path"], "sha256": _sha256(path)})
    for relative in protocol["repository"]["method_artifacts"]:
        path = _repo_path(relative)
        records.append({"role": "method_artifact", "path": relative, "sha256": _sha256(path)})
    return records


def build_manifest(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    try:
        protocol_path.relative_to(REPO.resolve())
    except ValueError as exc:
        raise ContractError("protocol must live inside the repository") from exc
    protocol = _load_json(protocol_path)
    validate_protocol(protocol)
    world_audits = [
        _world_audit(item["path"], protocol["repository"]["camera_model_names"])
        | {"role": item["role"]}
        for item in protocol["repository"]["world_artifacts"]
    ]
    schedule = expected_schedule(protocol)
    return {
        "manifest_schema_version": 1,
        "campaign_id": protocol["protocol_id"],
        "status": "planned_no_results",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path.relative_to(REPO.resolve())),
        "protocol_sha256": _sha256(protocol_path),
        "camera_ids": list(CAMERA_IDS),
        "world_audits": world_audits,
        "artifacts": _artifact_records(protocol_path, protocol),
        "data_root": protocol["manifest_policy"]["data_root"],
        "input_datasets": [],
        "legacy_input_datasets": [],
        "online_inputs": list(protocol["information_boundary"]["online_allowed"]),
        "evaluation_only_inputs": list(protocol["information_boundary"]["evaluation_only"]),
        "schedule_counts": {stage: len(units) for stage, units in schedule.items()},
        "schedule": schedule,
        "result_files": [],
    }


def validate_manifest(manifest: dict[str, Any], protocol_path: Path | None = None) -> None:
    if manifest.get("manifest_schema_version") != 1:
        raise ContractError("unsupported manifest schema_version")
    if manifest.get("status") != "planned_no_results":
        raise ContractError("this validator accepts only prospective no-results manifests")
    embedded_protocol = manifest.get("protocol_path")
    if protocol_path is None:
        if not isinstance(embedded_protocol, str):
            raise ContractError("manifest lacks protocol_path")
        protocol_path = _repo_path(embedded_protocol)
    else:
        protocol_path = protocol_path.resolve()
    protocol = _load_json(protocol_path)
    validate_protocol(protocol)
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ContractError("protocol hash mismatch")
    if manifest.get("campaign_id") != protocol.get("protocol_id"):
        raise ContractError("campaign/protocol ID mismatch")
    if tuple(manifest.get("camera_ids", [])) != CAMERA_IDS:
        raise ContractError("manifest camera set/order is not A-E")
    if manifest.get("input_datasets") or manifest.get("legacy_input_datasets"):
        raise ContractError("new campaign may not consume an existing evidence dataset")
    if manifest.get("result_files"):
        raise ContractError("prospective plan may not claim result files")
    data_root = manifest.get("data_root", "")
    if data_root != protocol["manifest_policy"]["data_root"]:
        raise ContractError("campaign data root changed")
    forbidden_tokens = tuple(protocol["repository"]["forbidden_legacy_tokens"])
    declared_paths = [data_root]
    declared_paths += [item.get("path", "") for item in manifest.get("artifacts", [])]
    declared_paths += [item.get("path", "") for item in manifest.get("world_audits", [])]
    for value in declared_paths:
        if any(token.lower() in str(value).lower() for token in forbidden_tokens):
            raise ContractError(f"legacy world/dataset token in manifest path: {value}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractError("manifest has no artifact hashes")
    roles = [item.get("role") for item in artifacts]
    for role in protocol["manifest_policy"]["required_sha256_roles"]:
        if role not in roles:
            raise ContractError(f"manifest lacks required hashed role: {role}")
    expected_paths = {
        str(protocol_path.relative_to(REPO.resolve())),
        *(item["path"] for item in protocol["repository"]["world_artifacts"]),
        *protocol["repository"]["method_artifacts"],
    }
    actual_paths = {item.get("path") for item in artifacts}
    if actual_paths != expected_paths:
        raise ContractError(
            f"hashed artifact set changed: missing={sorted(expected_paths - actual_paths)} "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    for item in artifacts:
        digest = item.get("sha256", "")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ContractError(f"missing/invalid sha256 for {item.get('path')}")
        path = _repo_path(item["path"])
        if not path.is_file():
            raise ContractError(f"hashed artifact disappeared: {item['path']}")
        current = _sha256(path)
        if current != digest:
            raise ContractError(f"artifact hash mismatch: {item['path']}")
    online = set(manifest.get("online_inputs", []))
    evaluation = set(manifest.get("evaluation_only_inputs", []))
    if online & evaluation:
        raise ContractError(f"manifest leaks evaluation fields online: {sorted(online & evaluation)}")
    forbidden_online = set(protocol["information_boundary"]["forbidden_online"])
    if online & forbidden_online:
        raise ContractError(f"forbidden online inputs: {sorted(online & forbidden_online)}")
    expected = expected_schedule(protocol)
    actual = manifest.get("schedule")
    if not isinstance(actual, dict) or tuple(actual) != STAGE_IDS:
        raise ContractError(f"manifest schedule stages/order must be {STAGE_IDS}")
    for stage in STAGE_IDS:
        if not isinstance(actual[stage], list):
            raise ContractError(f"schedule {stage} is not a list")
        _assert_exact_ids(
            (unit.get("unit_id") for unit in actual[stage]),
            (unit["unit_id"] for unit in expected[stage]),
            stage,
        )
        count = manifest.get("schedule_counts", {}).get(stage)
        if count != len(actual[stage]):
            raise ContractError(f"schedule count mismatch for {stage}")
    if manifest.get("schedule_counts") != {stage: len(expected[stage]) for stage in STAGE_IDS}:
        raise ContractError("schedule_counts do not match the registered exact schedule")


def _write_json(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise ContractError(f"refusing to overwrite {path}; pass --force after reviewing the diff")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-protocol", help="validate protocol and current source apparatus")
    plan = sub.add_parser("plan", help="resolve hashes and emit an exact no-results schedule")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--force", action="store_true")
    check = sub.add_parser("validate-manifest", help="reject drift, leakage, or missing units")
    check.add_argument("manifest", type=Path)
    sub.add_parser("summary", help="print planned experimental-unit counts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol = _load_json(args.protocol)
        if args.command == "validate-protocol":
            validate_protocol(protocol)
            print(f"PASS {protocol['protocol_id']}: protocol and current apparatus are valid")
        elif args.command == "plan":
            manifest = build_manifest(args.protocol)
            validate_manifest(manifest, args.protocol)
            _write_json(args.output, manifest, args.force)
            print(f"WROTE {args.output} ({sum(manifest['schedule_counts'].values())} units)")
        elif args.command == "validate-manifest":
            manifest = _load_json(args.manifest)
            validate_manifest(manifest, args.protocol)
            print(f"PASS {args.manifest}: hashes, boundaries, splits, and schedule are intact")
        elif args.command == "summary":
            validate_protocol(protocol)
            schedule = expected_schedule(protocol)
            for stage, units in schedule.items():
                print(f"{stage}: {len(units):4d} independent/scheduled units")
            print(f"TOTAL: {sum(map(len, schedule.values()))}")
        return 0
    except ContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
