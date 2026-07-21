#!/usr/bin/env python3
"""Plan and validate a resumable, immutable D1/D2 collection campaign.

This tool deliberately does not launch ROS, Gazebo, recorders, or route drivers.
It gives a future runner a small safety boundary:

* every passive collection row has a content-derived identity and one run dir;
* the only mutable file is an atomically replaced status ledger;
* a row is complete only while its completion manifest and artifacts validate;
* frozen protocol, study, model, calibration, and runtime config bytes are
  hashed into both the campaign and every completion contract.

The six-field identity is appropriate for D1/D2 physical collection.  D3/D4
rows have additional condition/policy dimensions and intentionally require a
separate replay/closed-loop ledger instead of weakening this identity.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml


REPO = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO / "experiments" / "multicamera_commissioning_bigwarehouse"
DEFAULT_STUDY = STUDY_DIR / "config" / "study.yaml"
DEFAULT_PROTOCOL = STUDY_DIR / "config" / "paper_protocol.yaml"
DEFAULT_LEDGER_NAME = "campaign_ledger.json"
DEFAULT_ATTEMPT_SLOTS = 3
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
RUNTIME_READINESS_ARTIFACT = "runtime_readiness.json"
ROW_FIELDS = (
    "phase",
    "route",
    "lateral_offset_m",
    "speed_mps",
    "repeat",
    "seed",
)
STATUSES = ("planned", "in_progress", "completed", "failed")
EVIDENCE_ROLES = ("fit", "qualification")
SUCCESS_END_REASONS = (
    "goal_reached",
    "goal_reached_stable",
    "route_complete",
    "completed",
)
REQUIRED_ARTIFACTS = (
    RUNTIME_READINESS_ARTIFACT,
    "raw/route_manifest.json",
    "raw/route_completion.json",
    "raw/operational_recording_manifest.json",
    "raw/experiment.csv",
    "raw/camera_A_perception.csv",
    "raw/camera_B_perception.csv",
    "raw/camera_C_perception.csv",
    "raw/camera_D_perception.csv",
    "evaluation_only/evaluation_truth_manifest.json",
    "evaluation_only/ground_truth.csv",
)


class LedgerError(RuntimeError):
    """A campaign contract or on-disk campaign is unsafe or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LedgerError(f"Cannot read YAML input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerError(f"Expected a YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _float_token(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise LedgerError(f"Campaign coordinates and speeds must be finite, got {value!r}")
    if number == 0.0:
        number = 0.0
    return format(number, ".17g")


def canonical_row_tuple(
    *,
    phase: str,
    route: str,
    lateral_offset_m: Any,
    speed_mps: Any,
    repeat: Any,
    seed: Any,
) -> dict[str, Any]:
    """Return the exact, portable representation used to derive a row ID."""

    phase_text = str(phase)
    route_text = str(route)
    if not phase_text or not route_text:
        raise LedgerError("phase and route must be non-empty")
    repeat_value = None if repeat in (None, "") else int(repeat)
    seed_value = None if seed in (None, "") else int(seed)
    if repeat_value is not None and repeat_value < 1:
        raise LedgerError(f"repeat must be at least one, got {repeat!r}")
    return {
        "phase": phase_text,
        "route": route_text,
        # Decimal strings make the tuple independent of YAML/JSON float
        # rendering while retaining the exact configured binary float value.
        "lateral_offset_m": _float_token(lateral_offset_m),
        "speed_mps": _float_token(speed_mps),
        "repeat": repeat_value,
        "seed": seed_value,
    }


def row_id(row_tuple: Mapping[str, Any]) -> str:
    """Derive a stable ID from exactly the declared six-field tuple."""

    if tuple(row_tuple.keys()) != ROW_FIELDS:
        # Mapping insertion order is not part of the digest, but enforcing the
        # shape here prevents an accidental seventh discriminator.
        if set(row_tuple) != set(ROW_FIELDS):
            raise LedgerError(f"Row tuple must contain exactly {ROW_FIELDS}")
    exact = [row_tuple[field] for field in ROW_FIELDS]
    return "row_" + hashlib.sha256(_canonical_json(exact)).hexdigest()


def _collection_rows(
    *,
    phase: str,
    routes: Iterable[str],
    repeats: int,
    offsets: Iterable[float],
    speeds: Iterable[float],
    seed_values: Sequence[int],
) -> Iterator[dict[str, Any]]:
    for route in routes:
        for repeat in range(1, int(repeats) + 1):
            for offset in offsets:
                for speed in speeds:
                    yield canonical_row_tuple(
                        phase=phase,
                        route=str(route),
                        lateral_offset_m=offset,
                        speed_mps=speed,
                        repeat=repeat,
                        seed=seed_values[repeat - 1],
                    )


def expected_evidence_role(phase: str) -> str:
    if phase == "D1_mapping_train":
        return "fit"
    if phase == "D1_mapping_heldout" or phase.startswith("D2_overlap_"):
        return "qualification"
    raise LedgerError(f"No evidence role declared for campaign phase {phase!r}")


def expected_analysis_split(phase: str, repeat: int) -> str:
    """Freeze run-disjoint fit/calibration/validation ownership before data."""

    if phase == "D1_mapping_train":
        mapping = {
            1: "gp_fit",
            2: "gp_fit",
            3: "gp_fit",
            4: "confidence_calibration",
            5: "stacker_validation",
        }
        if repeat not in mapping:
            raise LedgerError(
                "D1 mapping-train repeats must be exactly 1..5 for the frozen analysis split"
            )
        return mapping[repeat]
    if phase == "D1_mapping_heldout":
        return "final_heldout_test"
    if phase.startswith("D2_overlap_"):
        return "overlap_qualification"
    raise LedgerError(f"No analysis split declared for campaign phase {phase!r}")


def build_collection_plan(
    study: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Materialise only the frozen passive D1/D2 collection matrix."""

    known_routes = {
        str(route.get("name"))
        for route in study.get("collection", {}).get("routes", [])
        if isinstance(route, dict) and route.get("name")
    }
    mapping = dict(protocol.get("route_disjoint_mapping") or {})
    overlap = dict(protocol.get("overlap_qualification") or {})
    randomization = dict(protocol.get("randomization") or {})
    try:
        seed_values = [int(value) for value in randomization.get("seed_values", [])]
    except (TypeError, ValueError) as exc:
        raise LedgerError("randomization.seed_values must be integers") from exc
    max_repeats = max(
        int(mapping.get("repeats_per_route", 0)),
        int(overlap.get("repeats_per_route", 0)),
    )
    if len(seed_values) < max_repeats or len(seed_values) != len(set(seed_values)):
        raise LedgerError(
            "randomization.seed_values must provide at least one unique pre-declared "
            "seed per D1/D2 repeat"
        )
    rows = list(
        _collection_rows(
            phase="D1_mapping_train",
            routes=mapping.get("train_routes", []),
            repeats=int(mapping.get("repeats_per_route", 0)),
            offsets=mapping.get("lateral_offsets_m", []),
            speeds=mapping.get("speeds_mps", []),
            seed_values=seed_values,
        )
    )
    rows.extend(
        _collection_rows(
            phase="D1_mapping_heldout",
            routes=mapping.get("heldout_routes", []),
            repeats=int(mapping.get("repeats_per_route", 0)),
            offsets=mapping.get("lateral_offsets_m", []),
            speeds=mapping.get("speeds_mps", []),
            seed_values=seed_values,
        )
    )
    for edge in overlap.get("edges", []):
        if not isinstance(edge, dict) or len(edge.get("cameras", [])) != 2:
            raise LedgerError(f"Malformed D2 overlap edge: {edge!r}")
        cameras = tuple(str(value) for value in edge["cameras"])
        rows.extend(
            _collection_rows(
                phase=f"D2_overlap_{'_'.join(cameras)}",
                routes=[str(edge.get("route", ""))],
                repeats=int(overlap.get("repeats_per_route", 0)),
                offsets=overlap.get("lateral_offsets_m", []),
                speeds=overlap.get("speeds_mps", []),
                seed_values=seed_values,
            )
        )
    if not rows:
        raise LedgerError("The protocol produced no D1/D2 collection rows")
    missing = sorted({row["route"] for row in rows} - known_routes)
    if missing:
        raise LedgerError(f"Protocol routes are absent from study.yaml: {missing}")
    ids = [row_id(row) for row in rows]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise LedgerError(
            "Duplicate six-field campaign tuples produced row IDs: " + ", ".join(duplicate_ids)
        )
    return rows


def _input_record(label: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LedgerError(f"Frozen input {label!r} is not a file: {resolved}")
    return {
        "label": label,
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def build_provenance(
    *,
    study_path: Path,
    protocol_path: Path,
    model_path: Path,
    calibration_path: Path,
    config_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    if not config_paths:
        raise LedgerError("At least one frozen runtime --config is required")
    labels = [f"config:{index}:{path.name}" for index, path in enumerate(config_paths, start=1)]
    if len(labels) != len(set(labels)):
        raise LedgerError("Frozen input labels must be unique")
    return [
        _input_record("study", study_path),
        _input_record("protocol", protocol_path),
        _input_record("model", model_path),
        _input_record("calibration", calibration_path),
        *(
            _input_record(label, path)
            for label, path in zip(labels, config_paths, strict=True)
        ),
    ]


def input_sha256(provenance: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    hashes = {str(item["label"]): str(item["sha256"]) for item in provenance}
    if len(hashes) != len(provenance):
        raise LedgerError("Duplicate frozen-input provenance labels")
    return hashes


def build_method_freeze_provenance(config_paths: Sequence[Path]) -> dict[str, Any]:
    """Snapshot the analysis-declared implementation before a campaign starts.

    A SHA-256 of the analysis YAML alone is insufficient: its `source_files`
    are the actual implementation that turns operational recordings into paper
    evidence.  This record is deliberately compatible with the operational
    recorder's `method_freeze` payload while retaining resolved paths for
    future frozen-input verification.
    """

    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for supplied in config_paths:
        path = supplied.expanduser().resolve()
        payload = _load_yaml(path)
        freeze = payload.get("method_freeze")
        if freeze is not None:
            if not isinstance(freeze, dict):
                raise LedgerError(f"method_freeze must be a mapping in {path}")
            candidates.append((path, payload, freeze))
    if len(candidates) != 1:
        raise LedgerError(
            "Campaign --config inputs must contain exactly one analysis plan with method_freeze"
        )
    analysis_path, analysis, freeze = candidates[0]
    analysis_plan_id = analysis.get("analysis_plan_id")
    if not isinstance(analysis_plan_id, str) or not analysis_plan_id.strip():
        raise LedgerError(f"analysis plan lacks a non-empty analysis_plan_id: {analysis_path}")
    source_files = freeze.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise LedgerError(f"analysis plan lacks method_freeze.source_files: {analysis_path}")
    source_records: list[dict[str, str]] = []
    source_sha256: dict[str, str] = {}
    study_dir = analysis_path.parent.parent
    for value in source_files:
        if not isinstance(value, str) or not value.strip():
            raise LedgerError("method_freeze.source_files entries must be non-empty strings")
        relative = str(value)
        if relative in source_sha256:
            raise LedgerError(f"method_freeze.source_files contains duplicate {relative!r}")
        source = (study_dir / relative).resolve()
        if not source.is_file():
            raise LedgerError(f"frozen method source is not a file: {source}")
        digest = _sha256(source)
        source_sha256[relative] = digest
        source_records.append(
            {"relative": relative, "path": str(source), "sha256": digest}
        )
    snapshot = {
        "analysis_plan_id": analysis_plan_id,
        "analysis_plan_sha256": _sha256(analysis_path),
        "source_sha256": source_sha256,
    }
    core = {
        "schema_version": 1,
        "analysis_plan_path": str(analysis_path),
        "snapshot": snapshot,
        "source_records": source_records,
    }
    return {**core, "sha256": hashlib.sha256(_canonical_json(core)).hexdigest()}


def _method_freeze_provenance_errors(ledger: Mapping[str, Any]) -> list[str]:
    """Validate the stored source snapshot without reading mutable files."""

    provenance = _dict(ledger.get("method_freeze_provenance"))
    snapshot = _dict(provenance.get("snapshot"))
    records = provenance.get("source_records")
    if provenance.get("schema_version") != 1:
        return ["ledger method-freeze provenance schema is invalid"]
    analysis_path = provenance.get("analysis_plan_path")
    if not isinstance(analysis_path, str) or not analysis_path:
        return ["ledger method-freeze provenance lacks analysis_plan_path"]
    if (
        not isinstance(snapshot.get("analysis_plan_id"), str)
        or not snapshot["analysis_plan_id"]
        or not _is_sha256(snapshot.get("analysis_plan_sha256"))
        or not isinstance(snapshot.get("source_sha256"), dict)
        or not snapshot["source_sha256"]
    ):
        return ["ledger method-freeze snapshot is incomplete"]
    if not isinstance(records, list) or not records:
        return ["ledger method-freeze provenance lacks source records"]
    source_hashes = snapshot["source_sha256"]
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return ["ledger method-freeze source record is malformed"]
        relative = record.get("relative")
        path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or not isinstance(path, str)
            or not path
            or not _is_sha256(digest)
            or source_hashes.get(relative) != digest
        ):
            return ["ledger method-freeze source records do not match snapshot hashes"]
        seen.add(relative)
    if set(source_hashes) != seen or not all(_is_sha256(value) for value in source_hashes.values()):
        return ["ledger method-freeze source hash set is invalid"]
    core = {
        "schema_version": provenance.get("schema_version"),
        "analysis_plan_path": analysis_path,
        "snapshot": snapshot,
        "source_records": records,
    }
    if provenance.get("sha256") != hashlib.sha256(_canonical_json(core)).hexdigest():
        return ["ledger method-freeze provenance hash does not match its payload"]
    return []


def method_freeze_snapshot(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recorder-compatible method snapshot from a valid ledger."""

    errors = _method_freeze_provenance_errors(ledger)
    if errors:
        raise LedgerError("; ".join(errors))
    return copy.deepcopy(dict(_dict(ledger["method_freeze_provenance"]["snapshot"])))


def method_freeze_sha256(ledger: Mapping[str, Any]) -> str:
    """Return the immutable digest that recorder preflight propagates."""

    errors = _method_freeze_provenance_errors(ledger)
    if errors:
        raise LedgerError("; ".join(errors))
    return str(ledger["method_freeze_provenance"]["sha256"])


def _plan_sha256(
    rows: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    method_freeze_provenance: Mapping[str, Any],
) -> str:
    contract = {
        "rows": [
            {
                "row_id": row["row_id"],
                "row_tuple": row["row_tuple"],
                "expected_evidence_role": row["expected_evidence_role"],
                "expected_analysis_split": row["expected_analysis_split"],
                "run_dir": row["run_dir"],
                "attempts": row["attempts"],
            }
            for row in rows
        ],
        "input_sha256": input_sha256(provenance),
        "method_freeze_provenance": method_freeze_provenance,
    }
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def build_ledger(
    *,
    study_path: Path,
    protocol_path: Path,
    model_path: Path,
    calibration_path: Path,
    config_paths: Sequence[Path],
    attempt_slots: int = DEFAULT_ATTEMPT_SLOTS,
) -> dict[str, Any]:
    if isinstance(attempt_slots, bool) or int(attempt_slots) < 1:
        raise LedgerError("attempt_slots must be a positive integer")
    attempt_slots = int(attempt_slots)
    study = _load_yaml(study_path.expanduser().resolve())
    protocol = _load_yaml(protocol_path.expanduser().resolve())
    tuples = build_collection_plan(study, protocol)
    provenance = build_provenance(
        study_path=study_path,
        protocol_path=protocol_path,
        model_path=model_path,
        calibration_path=calibration_path,
        config_paths=config_paths,
    )
    method_freeze_provenance = build_method_freeze_provenance(config_paths)
    rows: list[dict[str, Any]] = []
    for exact_tuple in tuples:
        identifier = row_id(exact_tuple)
        attempts = [
            {
                "attempt_id": f"attempt_{index:03d}",
                "run_dir": f"runs/{identifier}/attempt_{index:03d}",
                "ign_partition": f"unav_{identifier[4:16]}_a{index:03d}",
                "ros_domain_id": 10 + (int(identifier[-8:], 16) + index - 1) % 200,
            }
            for index in range(1, attempt_slots + 1)
        ]
        rows.append(
            {
                "row_id": identifier,
                "row_tuple": exact_tuple,
                # Compatibility alias for tooling that displays the first
                # pre-declared slot. Runtime selection always uses attempts.
                "run_dir": attempts[0]["run_dir"],
                "attempts": attempts,
                "expected_evidence_role": expected_evidence_role(
                    str(exact_tuple["phase"])
                ),
                "expected_analysis_split": expected_analysis_split(
                    str(exact_tuple["phase"]), int(exact_tuple["repeat"])
                ),
                "status": "planned",
                "status_reason": "",
                "validation_errors": [],
                "validated_at": None,
            }
        )
    created_at = _utc_now()
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "campaign_kind": "D1_D2_passive_collection",
        "protocol_id": str(protocol.get("protocol_id", "")),
        "study_id": str(study.get("study_id", "")),
        "created_at": created_at,
        "updated_at": created_at,
        "revision": 0,
        "provenance": provenance,
        # The analysis plan itself is one frozen config input, but its declared
        # implementation files also need an eager, immutable snapshot.  Do
        # not defer these hashes to recorder startup: code could change between
        # planning and the first attempted run.
        "method_freeze_provenance": method_freeze_provenance,
        "completion_contract": {
            "manifest": "completion_manifest.json",
            "row_tuple_fields": list(ROW_FIELDS),
            "success_end_reasons": list(SUCCESS_END_REASONS),
            "required_artifacts": list(REQUIRED_ARTIFACTS),
            "runtime_readiness": {
                "artifact": RUNTIME_READINESS_ARTIFACT,
                "mode": "enforce",
                "must_pass": True,
                "must_be_evidence_eligible": True,
                "must_precede_pre_run_route_manifest": True,
                "route_wide_camera_health": {
                    "required": True,
                    "source": "raw camera CSV timing columns",
                    "requirements": [
                        "continuous route coverage",
                        "fresh frame-age fraction",
                        "maximum recorder-wall-time gap",
                    ],
                },
            },
            "method_freeze": {
                "snapshot": method_freeze_provenance["snapshot"],
                "sha256": method_freeze_provenance["sha256"],
            },
            "input_sha256": input_sha256(provenance),
            "attempt_slots": attempt_slots,
        },
        "rows": rows,
    }
    ledger["plan_sha256"] = _plan_sha256(rows, provenance, method_freeze_provenance)
    ledger["summary"] = _summary(rows)
    validate_ledger_contract(ledger)
    return ledger


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("status", "")) for row in rows)
    return {status: int(counts.get(status, 0)) for status in STATUSES} | {"total": len(rows)}


def validate_ledger_contract(ledger: Mapping[str, Any]) -> None:
    if ledger.get("schema_version") != 1:
        raise LedgerError(f"Unsupported ledger schema {ledger.get('schema_version')!r}")
    if ledger.get("campaign_kind") != "D1_D2_passive_collection":
        raise LedgerError("This tool accepts only the D1/D2 passive collection ledger")
    completion_contract = _dict(ledger.get("completion_contract"))
    if tuple(completion_contract.get("required_artifacts", [])) != REQUIRED_ARTIFACTS:
        raise LedgerError("Ledger completion contract lacks the exact required artifact set")
    readiness_contract = _dict(completion_contract.get("runtime_readiness"))
    if (
        readiness_contract.get("artifact") != RUNTIME_READINESS_ARTIFACT
        or readiness_contract.get("mode") != "enforce"
        or readiness_contract.get("must_pass") is not True
        or readiness_contract.get("must_be_evidence_eligible") is not True
        or readiness_contract.get("must_precede_pre_run_route_manifest") is not True
        or _dict(readiness_contract.get("route_wide_camera_health")).get("required")
        is not True
    ):
        raise LedgerError("Ledger runtime-readiness completion contract is incomplete")
    method_freeze_errors = _method_freeze_provenance_errors(ledger)
    if method_freeze_errors:
        raise LedgerError("; ".join(method_freeze_errors))
    method_freeze_contract = _dict(completion_contract.get("method_freeze"))
    if (
        method_freeze_contract.get("snapshot") != method_freeze_snapshot(ledger)
        or method_freeze_contract.get("sha256") != method_freeze_sha256(ledger)
    ):
        raise LedgerError("Ledger method-freeze completion contract is incomplete")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not rows:
        raise LedgerError("Ledger rows must be a non-empty list")
    ids: list[str] = []
    dirs: list[str] = []
    tuples: list[bytes] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LedgerError(f"Ledger row {index} is not an object")
        exact_tuple = row.get("row_tuple")
        if not isinstance(exact_tuple, dict) or set(exact_tuple) != set(ROW_FIELDS):
            raise LedgerError(f"Ledger row {index} has a malformed row_tuple")
        expected_id = row_id(exact_tuple)
        if row.get("row_id") != expected_id:
            raise LedgerError(f"Ledger row {index} ID is not derived from its exact tuple")
        expected_attempts = [
            {
                "attempt_id": f"attempt_{attempt_index:03d}",
                "run_dir": f"runs/{expected_id}/attempt_{attempt_index:03d}",
                "ign_partition": f"unav_{expected_id[4:16]}_a{attempt_index:03d}",
                "ros_domain_id": 10
                + (int(expected_id[-8:], 16) + attempt_index - 1) % 200,
            }
            for attempt_index in range(
                1, int(ledger.get("completion_contract", {}).get("attempt_slots", 0)) + 1
            )
        ]
        if not expected_attempts or row.get("attempts") != expected_attempts:
            raise LedgerError(f"Ledger row {expected_id} has malformed attempt slots")
        expected_dir = expected_attempts[0]["run_dir"]
        if row.get("run_dir") != expected_dir:
            raise LedgerError(f"Ledger row {expected_id} must use run dir {expected_dir}")
        if row.get("status") not in STATUSES:
            raise LedgerError(f"Ledger row {expected_id} has invalid status {row.get('status')!r}")
        if row.get("expected_evidence_role") != expected_evidence_role(
            str(exact_tuple["phase"])
        ):
            raise LedgerError(f"Ledger row {expected_id} has the wrong pre-declared evidence role")
        if row.get("expected_analysis_split") != expected_analysis_split(
            str(exact_tuple["phase"]), int(exact_tuple["repeat"])
        ):
            raise LedgerError(f"Ledger row {expected_id} has the wrong pre-declared analysis split")
        ids.append(expected_id)
        dirs.append(expected_dir)
        tuples.append(_canonical_json([exact_tuple[field] for field in ROW_FIELDS]))
    for name, values in (("row ID", ids), ("run dir", dirs), ("six-field tuple", tuples)):
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        if duplicates:
            raise LedgerError(f"Ledger contains duplicate {name}s")
    provenance = ledger.get("provenance")
    if not isinstance(provenance, list) or len(provenance) < 5:
        raise LedgerError("Ledger must hash study, protocol, model, calibration, and config inputs")
    expected_plan = _plan_sha256(
        rows, provenance, _dict(ledger.get("method_freeze_provenance"))
    )
    if ledger.get("plan_sha256") != expected_plan:
        raise LedgerError("Ledger plan_sha256 does not match rows and frozen inputs")


def verify_frozen_inputs(ledger: Mapping[str, Any]) -> None:
    for record in ledger["provenance"]:
        path = Path(str(record["path"]))
        if not path.is_file():
            raise LedgerError(f"Frozen input disappeared: {record['label']} -> {path}")
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise LedgerError(
                f"Frozen input changed: {record['label']} expected {record['sha256']}, got {actual}"
            )
    for record in _dict(ledger.get("method_freeze_provenance")).get(
        "source_records", []
    ):
        if not isinstance(record, dict):
            # validate_ledger_contract normally catches this first.  Keep this
            # explicit fail-closed branch for direct callers.
            raise LedgerError("Frozen method source record is malformed")
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise LedgerError(
                f"Frozen method source disappeared: {record.get('relative')} -> {path}"
            )
        actual = _sha256(path)
        if actual != record.get("sha256"):
            raise LedgerError(
                "Frozen method source changed: "
                f"{record.get('relative')} expected {record.get('sha256')}, got {actual}"
            )


def active_campaign_attempts(
    campaign_root: Path,
    ledger: Mapping[str, Any],
    *,
    exclude_run_dir: str | None = None,
) -> list[dict[str, str]]:
    """Return non-terminal declared attempt directories across a campaign.

    ``ROS_DOMAIN_ID`` is deliberately a bounded integer namespace.  The
    ledger does not pretend its recycled values are leases: a campaign may
    reserve/run exactly one attempt directory at a time.  The three recorder
    processes belonging to that *same* attempt are allowed to preflight
    together, hence the optional exclusion.
    """

    root = campaign_root.expanduser().resolve()
    active: list[dict[str, str]] = []
    for row in ledger.get("rows", []):
        if not isinstance(row, dict):
            continue
        for attempt in row.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            relative = attempt.get("run_dir")
            if not isinstance(relative, str) or not relative:
                continue
            if relative == exclude_run_dir:
                continue
            run_dir = root / relative
            # A broken symlink is still a conflicting/unsafe occupancy.
            if not run_dir.exists() and not run_dir.is_symlink():
                continue
            completion = run_dir / "completion_manifest.json"
            failure = run_dir / "failure_manifest.json"
            if completion.exists() or failure.exists():
                continue
            active.append(
                {
                    "row_id": str(row.get("row_id", "")),
                    "attempt_id": str(attempt.get("attempt_id", "")),
                    "run_dir": relative,
                }
            )
    return active


def recorder_preflight_contract(
    *,
    ledger_path: Path,
    plan_row_id: str,
    attempt_id: str,
    seed: int,
    evidence_role: str,
    analysis_split: str,
    output_dir: Path,
    artifact_subdir: str,
    expected_inputs: Mapping[str, Path],
    frozen_config_paths: Sequence[Path],
    transport_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail before ROS startup when recorder inputs differ from the campaign."""

    path = ledger_path.expanduser().resolve()
    ledger = _load_json(path)
    validate_ledger_contract(ledger)
    verify_frozen_inputs(ledger)
    matches = [row for row in ledger["rows"] if row.get("row_id") == plan_row_id]
    if len(matches) != 1:
        raise LedgerError(f"Campaign contains no unique row {plan_row_id!r}")
    row = matches[0]
    if row["row_tuple"].get("seed") != seed:
        raise LedgerError(
            f"Recorder seed {seed} does not match frozen row seed {row['row_tuple'].get('seed')}"
        )
    if row.get("expected_evidence_role") != evidence_role:
        raise LedgerError(
            f"Recorder evidence role {evidence_role!r} does not match row role "
            f"{row.get('expected_evidence_role')!r}"
        )
    if row.get("expected_analysis_split") != analysis_split:
        raise LedgerError(
            f"Recorder analysis split {analysis_split!r} does not match row split "
            f"{row.get('expected_analysis_split')!r}"
        )
    attempt_matches = [
        (index, attempt)
        for index, attempt in enumerate(row["attempts"])
        if attempt.get("attempt_id") == attempt_id
    ]
    if len(attempt_matches) != 1:
        raise LedgerError(f"Row contains no unique attempt slot {attempt_id!r}")
    attempt_index, attempt = attempt_matches[0]
    expected_transport = {
        "ROS_LOCALHOST_ONLY": "1",
        "IGN_IP": "127.0.0.1",
        "GZ_IP": "127.0.0.1",
        "IGN_PARTITION": str(attempt["ign_partition"]),
        "ROS_DOMAIN_ID": str(attempt["ros_domain_id"]),
    }
    actual_environment = os.environ if transport_environment is None else transport_environment
    transport_mismatches = [
        name
        for name, expected_value in expected_transport.items()
        if actual_environment.get(name) != expected_value
    ]
    if transport_mismatches:
        raise LedgerError(
            "Transport environment differs from attempt contract: "
            + ", ".join(transport_mismatches)
        )
    for prior in row["attempts"][:attempt_index]:
        prior_dir = path.parent / str(prior["run_dir"])
        failure_path = prior_dir / "failure_manifest.json"
        completion_path = prior_dir / "completion_manifest.json"
        if completion_path.exists():
            raise LedgerError("A prior attempt already completed this campaign row")
        if not failure_path.is_file():
            raise LedgerError(
                f"Prior slot {prior['attempt_id']} must be explicitly quarantined before {attempt_id}"
            )
        failure = _load_json(failure_path)
        if (
            failure.get("row_id") != row["row_id"]
            or failure.get("attempt_id") != prior["attempt_id"]
            or not str(failure.get("reason", "")).strip()
        ):
            raise LedgerError(f"Prior failure contract is invalid: {failure_path}")
    for later in row["attempts"][attempt_index + 1 :]:
        if (path.parent / str(later["run_dir"])).exists():
            raise LedgerError("Later attempt slot exists before the selected attempt")
    attempt_dir = path.parent / str(attempt["run_dir"])
    if (attempt_dir / "completion_manifest.json").exists() or (
        attempt_dir / "failure_manifest.json"
    ).exists():
        raise LedgerError(f"Attempt slot {attempt_id} is already terminal")
    expected_output = (attempt_dir / str(artifact_subdir)).resolve()
    if output_dir.expanduser().resolve() != expected_output:
        raise LedgerError(
            f"Recorder output must be the ledger row's {artifact_subdir}/ directory: "
            f"{expected_output}"
        )
    frozen = input_sha256(ledger["provenance"])
    for label, input_path in expected_inputs.items():
        resolved = input_path.expanduser().resolve()
        if not resolved.is_file() or _sha256(resolved) != frozen.get(label):
            raise LedgerError(f"Recorder {label} input differs from frozen campaign bytes")
    expected_configs = sorted(
        digest for label, digest in frozen.items() if label.startswith("config:")
    )
    actual_configs = sorted(
        _sha256(config_path.expanduser().resolve())
        for config_path in frozen_config_paths
        if config_path.expanduser().resolve().is_file()
    )
    if len(actual_configs) != len(frozen_config_paths) or actual_configs != expected_configs:
        raise LedgerError("Recorder --frozen-config set differs from campaign --config inputs")
    # Reserve this declared directory while holding the campaign lock.  Merely
    # checking for a directory before recorder startup has a race: two rows
    # could both see an idle campaign and launch with colliding recycled ROS
    # domains.  The empty directory is an intentional in-progress reservation;
    # later preflights for the same attempt are permitted.
    root = path.parent
    with _campaign_lock(root):
        current = _load_json(path)
        validate_ledger_contract(current)
        verify_frozen_inputs(current)
        if current.get("plan_sha256") != ledger.get("plan_sha256"):
            raise LedgerError("Campaign ledger changed during recorder preflight")
        conflicts = active_campaign_attempts(
            root,
            current,
            exclude_run_dir=str(attempt["run_dir"]),
        )
        if conflicts:
            labels = ", ".join(
                f"{item['row_id']}/{item['attempt_id']}" for item in conflicts
            )
            raise LedgerError(
                "another campaign attempt is active; only one attempt may use the "
                f"shared ROS domain namespace at a time ({labels})"
            )
        if attempt_dir.is_symlink():
            raise LedgerError(f"Attempt directory is unsafe symlink: {attempt_dir}")
        attempt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ledger_path": str(path),
        "plan_sha256": str(ledger["plan_sha256"]),
        "row_id": str(row["row_id"]),
        "attempt_id": str(attempt_id),
        "row_tuple": dict(row["row_tuple"]),
        "expected_evidence_role": str(row["expected_evidence_role"]),
        "expected_analysis_split": str(row["expected_analysis_split"]),
        "transport_environment": expected_transport,
        "input_sha256": frozen,
        # Both the readiness barrier and operational/evaluation recorders
        # persist this exact plan-time snapshot.  This prevents a later
        # re-hash of mutable analysis code from silently changing the method
        # used to create paper evidence.
        "method_freeze": method_freeze_snapshot(ledger),
        "method_freeze_sha256": method_freeze_sha256(ledger),
    }


@contextmanager
def _campaign_lock(campaign_root: Path) -> Iterator[None]:
    campaign_root.mkdir(parents=True, exist_ok=True)
    lock_path = campaign_root / ".campaign_ledger.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fsynced JSON file without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _attempt_artifact_hashes(attempt_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(attempt_dir)): _sha256(path)
        for path in sorted(attempt_dir.rglob("*"))
        if path.is_file() and path.name != "failure_manifest.json"
    }


def quarantine_attempt(
    campaign_root: Path,
    *,
    row_id_value: str,
    attempt_id: str,
    reason: str,
) -> dict[str, Any]:
    """Seal a failed/crashed attempt so the next pre-declared slot can run."""

    root = campaign_root.expanduser().resolve()
    if not reason.strip():
        raise LedgerError("Attempt quarantine reason must be non-empty")
    with _campaign_lock(root):
        ledger_path = root / DEFAULT_LEDGER_NAME
        ledger = _load_json(ledger_path)
        validate_ledger_contract(ledger)
        verify_frozen_inputs(ledger)
        row = next(
            (item for item in ledger["rows"] if item["row_id"] == row_id_value), None
        )
        if row is None:
            raise LedgerError(f"Unknown campaign row {row_id_value!r}")
        attempt = next(
            (item for item in row["attempts"] if item["attempt_id"] == attempt_id), None
        )
        if attempt is None:
            raise LedgerError(f"Unknown attempt slot {attempt_id!r}")
        attempt_dir = root / str(attempt["run_dir"])
        if not attempt_dir.is_dir() or attempt_dir.is_symlink():
            raise LedgerError(f"Attempt directory is absent or unsafe: {attempt_dir}")
        completion_path = attempt_dir / "completion_manifest.json"
        failure_path = attempt_dir / "failure_manifest.json"
        if completion_path.exists() or failure_path.exists():
            raise LedgerError("Attempt is already terminal and cannot be relabelled")
        payload = {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "row_id": row["row_id"],
            "attempt_id": attempt["attempt_id"],
            "row_tuple": row["row_tuple"],
            "plan_sha256": ledger["plan_sha256"],
            "input_sha256": input_sha256(ledger["provenance"]),
            "reason": reason.strip(),
            "preserved_artifacts_sha256": _attempt_artifact_hashes(attempt_dir),
        }
        try:
            _atomic_write_json_new(failure_path, payload)
        except FileExistsError as exc:
            raise LedgerError(f"Attempt was quarantined concurrently: {failure_path}") from exc
    return payload


def initialize_campaign(campaign_root: Path, desired: Mapping[str, Any]) -> dict[str, Any]:
    """Create a ledger once, or safely resume the identical campaign."""

    root = campaign_root.expanduser().resolve()
    ledger_path = root / DEFAULT_LEDGER_NAME
    with _campaign_lock(root):
        if ledger_path.exists():
            existing = _load_json(ledger_path)
            validate_ledger_contract(existing)
            if existing["plan_sha256"] != desired["plan_sha256"]:
                raise LedgerError(
                    "Campaign root already belongs to a different plan or frozen input set; "
                    "choose a new root instead of overwriting it"
                )
            verify_frozen_inputs(existing)
            return existing
        # A missing ledger must never adopt pre-existing expected output dirs:
        # that could relabel data from a different campaign.
        collisions = [
            str(attempt["run_dir"])
            for row in desired["rows"]
            for attempt in row["attempts"]
            if (root / str(attempt["run_dir"])).exists()
        ]
        if collisions:
            raise LedgerError(
                "Expected run directories predate the ledger; refusing to adopt/overwrite: "
                + ", ".join(collisions[:5])
            )
        _atomic_write_json(ledger_path, desired)
        return copy.deepcopy(dict(desired))


def _csv_data_rows(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or not any(cell.strip() for cell in header):
                return 0
            return sum(1 for row in reader if any(cell.strip() for cell in row))
    except (OSError, csv.Error):
        return 0


def _csv_finite_column(path: Path, column: str) -> list[float] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            if column not in (rows.fieldnames or []):
                return None
            values: list[float] = []
            for row in rows:
                value = float(row[column])
                if not math.isfinite(value):
                    return None
                values.append(value)
            return values
    except (OSError, csv.Error, KeyError, TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    """Return a plain mapping view for fail-closed malformed-manifest checks."""

    return value if isinstance(value, dict) else {}


def _nested_sha256(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value.get("sha256") if isinstance(value, dict) else None


def _manifest_config_hashes(payload: Mapping[str, Any]) -> list[str] | None:
    records = payload.get("frozen_configs")
    if not isinstance(records, list) or not records:
        return None
    hashes: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            return None
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return None
        hashes.append(digest)
    return hashes


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_sha256(value: Any) -> str:
    """Hash a structured, already-recorded contract deterministically."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc_epoch(value: Any) -> float | None:
    """Parse an explicit UTC timestamp without accepting naive local time."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    epoch = parsed.timestamp()
    return epoch if math.isfinite(epoch) else None


def _attempt_id_for_run_dir(row: Mapping[str, Any], run_dir: Path) -> str | None:
    matches = [
        attempt
        for attempt in row.get("attempts", [])
        if (run_dir.parents[2] / str(attempt.get("run_dir", ""))).resolve()
        == run_dir.resolve()
    ]
    if len(matches) != 1:
        return None
    attempt_id = matches[0].get("attempt_id")
    return str(attempt_id) if isinstance(attempt_id, str) and attempt_id else None


def _runtime_readiness_report_errors(
    report: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any], float | None]:
    """Validate the immutable startup barrier used by an evidence attempt.

    The report is generated before the route starts.  Its startup check is not
    sufficient evidence by itself; :func:`build_route_camera_health` separately
    verifies every camera throughout the recorded route interval.
    """

    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("runtime readiness schema_version must be 1")
    if report.get("mode") != "enforce":
        errors.append("runtime readiness must use enforce mode")
    if report.get("pass") is not True:
        errors.append("runtime readiness did not pass")
    if report.get("evidence_eligible") is not True:
        errors.append("runtime readiness is not evidence-eligible")
    if report.get("age_enforced") is not True:
        errors.append("runtime readiness did not enforce frame age")
    if report.get("would_pass_age_enforcement") is not True:
        errors.append("runtime readiness did not pass its age-enforcement gate")
    if report.get("timed_out") is not False:
        errors.append("runtime readiness timed out or omits timed_out=false")
    if report.get("interrupted") is not False:
        errors.append("runtime readiness was interrupted or omits interrupted=false")
    failures = report.get("failures")
    if not isinstance(failures, list) or failures:
        errors.append("runtime readiness reports unresolved failures")

    created_epoch = _utc_epoch(report.get("created_utc"))
    if created_epoch is None:
        errors.append("runtime readiness created_utc is missing, naive, or invalid")

    thresholds = _dict(report.get("thresholds"))
    max_frame_age_s = _finite_float(thresholds.get("max_frame_age_s"))
    max_stream_wall_age_s = _finite_float(thresholds.get("max_stream_wall_age_s"))
    min_fresh_fraction = _finite_float(thresholds.get("min_fresh_fraction"))
    if max_frame_age_s is None or max_frame_age_s <= 0.0:
        errors.append("runtime readiness max_frame_age_s threshold is invalid")
    if max_stream_wall_age_s is None or max_stream_wall_age_s <= 0.0:
        errors.append("runtime readiness max_stream_wall_age_s threshold is invalid")
    if (
        min_fresh_fraction is None
        or min_fresh_fraction < 0.0
        or min_fresh_fraction > 1.0
    ):
        errors.append("runtime readiness min_fresh_fraction threshold is invalid")

    cameras = _dict(report.get("cameras"))
    age_gates = _dict(report.get("age_gate_by_camera"))
    if set(cameras) != set(CAMERAS):
        errors.append("runtime readiness does not report exactly the four campaign cameras")
    if set(age_gates) != set(CAMERAS) or any(
        age_gates.get(camera) is not True for camera in CAMERAS
    ):
        errors.append("runtime readiness camera age gates are incomplete or failed")
    max_diag_stamp_gap_s_by_camera: dict[str, float] = {}
    if max_frame_age_s is not None and min_fresh_fraction is not None:
        for camera in CAMERAS:
            summary = _dict(cameras.get(camera))
            frame_age = _dict(summary.get("frame_age_s"))
            window_count = _finite_float(summary.get("summary_window_count"))
            finite_count = _finite_float(frame_age.get("finite_count"))
            fresh_fraction = _finite_float(frame_age.get("fresh_fraction"))
            reported_threshold = _finite_float(frame_age.get("threshold_s"))
            sim_hz = _finite_float(summary.get("sim_hz"))
            if (
                window_count is None
                or window_count < 1.0
                or finite_count != window_count
                or fresh_fraction is None
                or fresh_fraction < min_fresh_fraction
                or reported_threshold is None
                or abs(reported_threshold - max_frame_age_s) > 1.0e-9
            ):
                errors.append(
                    f"runtime readiness {camera} lacks a fully fresh enforced sample window"
                )
            if sim_hz is None or sim_hz <= 0.0:
                errors.append(
                    f"runtime readiness {camera} lacks a positive simulation-time output rate"
                )
            else:
                # Permit at most one missing nominal frame.  This gives a
                # route-wide diagnostic-stamp gap an explicit, per-camera
                # threshold derived from the enforced startup observation.
                max_diag_stamp_gap_s_by_camera[camera] = 2.0 / sim_hz

    normalized_thresholds = {
        "max_frame_age_s": max_frame_age_s,
        "max_stream_wall_age_s": max_stream_wall_age_s,
        "min_fresh_fraction": min_fresh_fraction,
        "max_diag_stamp_gap_s_by_camera": max_diag_stamp_gap_s_by_camera,
    }
    return sorted(set(errors)), normalized_thresholds, created_epoch


def _camera_route_samples(path: Path) -> tuple[list[dict[str, float]] | None, list[str]]:
    """Read the timing columns needed for route-wide liveness evidence."""

    required = (
        "diag_stamp",
        "recorder_wall_elapsed_s",
        "frame_age_at_publish_s",
    )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not set(required).issubset(reader.fieldnames or []):
                return None, [
                    f"{path.name} lacks route-health timing columns: {', '.join(required)}"
                ]
            samples: list[dict[str, float]] = []
            previous_stamp: float | None = None
            previous_wall: float | None = None
            for row_number, row in enumerate(reader, start=2):
                try:
                    stamp = float(row["diag_stamp"])
                    wall = float(row["recorder_wall_elapsed_s"])
                    age = float(row["frame_age_at_publish_s"])
                except (KeyError, TypeError, ValueError) as exc:
                    return None, [
                        f"{path.name} has an invalid route-health value at CSV row {row_number}: {exc}"
                    ]
                if not all(math.isfinite(value) for value in (stamp, wall, age)) or age < 0.0:
                    return None, [
                        f"{path.name} has non-finite or negative route-health timing at CSV row {row_number}"
                    ]
                if previous_stamp is not None and stamp <= previous_stamp:
                    return None, [
                        f"{path.name} has duplicate or out-of-order diagnostic stamps"
                    ]
                if previous_wall is not None and wall <= previous_wall:
                    return None, [
                        f"{path.name} has duplicate or out-of-order recorder wall times"
                    ]
                samples.append(
                    {
                        "diag_stamp": stamp,
                        "recorder_wall_elapsed_s": wall,
                        "frame_age_at_publish_s": age,
                    }
                )
                previous_stamp = stamp
                previous_wall = wall
    except (OSError, csv.Error) as exc:
        return None, [f"cannot read {path.name} for route-health evidence: {exc}"]
    if not samples:
        return None, [f"{path.name} has no route-health samples"]
    return samples, []


def build_route_camera_health(
    run_dir: Path,
    *,
    route_completion: Mapping[str, Any],
    operational: Mapping[str, Any],
    readiness_thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Derive a replayable, route-wide camera health contract from raw CSVs.

    Startup readiness proves that all streams were healthy before motion.  This
    payload proves that each camera remained live, fresh, ordered, and bounded
    in *recorder wall-time gap* through the completed simulation-time route.
    """

    errors: list[str] = []
    route_start = _finite_float(route_completion.get("route_started_sim_time_s"))
    route_end = _finite_float(route_completion.get("route_completed_sim_time_s"))
    max_frame_age_s = _finite_float(readiness_thresholds.get("max_frame_age_s"))
    min_fresh_fraction = _finite_float(readiness_thresholds.get("min_fresh_fraction"))
    max_wall_gap_s = _finite_float(readiness_thresholds.get("max_stream_wall_age_s"))
    max_diag_gap_by_camera = _dict(
        readiness_thresholds.get("max_diag_stamp_gap_s_by_camera")
    )
    recorder_fresh_age_s = _finite_float(operational.get("fresh_age_s"))
    if route_start is None or route_end is None or route_end <= route_start:
        errors.append("route completion lacks a non-empty simulation-time interval for camera health")
    if max_frame_age_s is None or max_frame_age_s <= 0.0:
        errors.append("route camera-health frame-age threshold is invalid")
    if min_fresh_fraction is None or not 0.0 <= min_fresh_fraction <= 1.0:
        errors.append("route camera-health fresh-fraction threshold is invalid")
    if max_wall_gap_s is None or max_wall_gap_s <= 0.0:
        errors.append("route camera-health wall-gap threshold is invalid")
    if recorder_fresh_age_s is None or recorder_fresh_age_s <= 0.0:
        errors.append("operational recorder fresh-age contract is invalid for route health")
    elif max_frame_age_s is not None and abs(recorder_fresh_age_s - max_frame_age_s) > 1.0e-9:
        errors.append("operational recorder fresh_age_s differs from readiness frame-age threshold")

    source_hashes: dict[str, str] = {}
    per_camera: dict[str, Any] = {}
    boundary_tolerance_s = recorder_fresh_age_s
    for camera in CAMERAS:
        path = run_dir / f"raw/{camera}_perception.csv"
        if path.is_file():
            source_hashes[path.name] = _sha256(path)
        samples, sample_errors = _camera_route_samples(path)
        camera_errors = list(sample_errors)
        result: dict[str, Any] = {
            "source_csv": f"raw/{camera}_perception.csv",
            "row_count": 0 if samples is None else len(samples),
            "route_window_sample_count": 0,
            "route_interior_sample_count": 0,
            "route_start_covered": False,
            "route_end_covered": False,
            "diag_stamp_range_s": {"first": None, "last": None},
            "recorder_wall_elapsed_range_s": {"first": None, "last": None},
            "max_diag_stamp_gap_s": None,
            "max_allowed_diag_stamp_gap_s": _finite_float(
                max_diag_gap_by_camera.get(camera)
            ),
            "max_recorder_wall_gap_s": None,
            "frame_age_s": {
                "finite_count": 0,
                "fresh_count": 0,
                "fresh_fraction": 0.0,
                "max": None,
            },
            "pass": False,
        }
        if (
            samples is not None
            and route_start is not None
            and route_end is not None
            and boundary_tolerance_s is not None
            and max_frame_age_s is not None
            and min_fresh_fraction is not None
            and max_wall_gap_s is not None
        ):
            interior = [
                sample
                for sample in samples
                if route_start <= sample["diag_stamp"] <= route_end
            ]
            max_diag_gap_s = _finite_float(max_diag_gap_by_camera.get(camera))
            result.update(
                {
                    # All coverage and gap checks intentionally use only
                    # samples inside the route.  Pre/post-route rows must not
                    # conceal a camera outage while the robot was moving.
                    "route_window_sample_count": len(interior),
                    "route_interior_sample_count": len(interior),
                    "route_start_covered": bool(interior)
                    and interior[0]["diag_stamp"] - route_start
                    <= boundary_tolerance_s,
                    "route_end_covered": bool(interior)
                    and route_end - interior[-1]["diag_stamp"]
                    <= boundary_tolerance_s,
                    "diag_stamp_range_s": {
                        "first": interior[0]["diag_stamp"] if interior else None,
                        "last": interior[-1]["diag_stamp"] if interior else None,
                    },
                    "recorder_wall_elapsed_range_s": {
                        "first": (
                            interior[0]["recorder_wall_elapsed_s"] if interior else None
                        ),
                        "last": (
                            interior[-1]["recorder_wall_elapsed_s"] if interior else None
                        ),
                    },
                }
            )
            if len(interior) >= 2:
                # Include both route boundaries in the simulation-time gap:
                # this catches a missing first/last in-route observation even
                # when a pre/post sample exists just outside the interval.
                result["max_diag_stamp_gap_s"] = max(
                    [interior[0]["diag_stamp"] - route_start]
                    + [
                        right["diag_stamp"] - left["diag_stamp"]
                        for left, right in zip(interior, interior[1:])
                    ]
                    + [route_end - interior[-1]["diag_stamp"]]
                )
                result["max_recorder_wall_gap_s"] = max(
                    right["recorder_wall_elapsed_s"] - left["recorder_wall_elapsed_s"]
                    for left, right in zip(interior, interior[1:])
                )
            if len(interior) < 2:
                camera_errors.append(
                    f"{camera} has fewer than two camera observations inside the route interval"
                )
            if not result["route_start_covered"] or not result["route_end_covered"]:
                camera_errors.append(f"{camera} does not cover both ends of the completed route")
            observed_diag_gap = _finite_float(result["max_diag_stamp_gap_s"])
            if max_diag_gap_s is None or max_diag_gap_s <= 0.0:
                camera_errors.append(
                    f"{camera} lacks an enforced maximum diagnostic-stamp gap"
                )
            elif observed_diag_gap is None or observed_diag_gap > max_diag_gap_s + 1.0e-9:
                camera_errors.append(
                    f"{camera} route-wide diagnostic-stamp gap exceeds {max_diag_gap_s:.6g}s"
                )
            observed_wall_gap = _finite_float(result["max_recorder_wall_gap_s"])
            if observed_wall_gap is None or observed_wall_gap > max_wall_gap_s + 1.0e-9:
                camera_errors.append(
                    f"{camera} route-wide recorder wall gap exceeds {max_wall_gap_s:.6g}s"
                )
            ages = [sample["frame_age_at_publish_s"] for sample in interior]
            fresh_count = sum(age <= max_frame_age_s for age in ages)
            fresh_fraction = fresh_count / len(ages) if ages else 0.0
            result["frame_age_s"] = {
                "finite_count": len(ages),
                "fresh_count": fresh_count,
                "fresh_fraction": fresh_fraction,
                "max": max(ages) if ages else None,
            }
            if not ages or fresh_fraction < min_fresh_fraction:
                camera_errors.append(
                    f"{camera} route-wide frame freshness is below {min_fresh_fraction:.6g}"
                )
        result["pass"] = not camera_errors
        result["errors"] = sorted(set(camera_errors))
        per_camera[camera] = result
        errors.extend(camera_errors)

    health = {
        "schema_version": 1,
        "route_interval_sim_s": {"start": route_start, "end": route_end},
        "thresholds": {
            "max_frame_age_s": max_frame_age_s,
            "min_fresh_fraction": min_fresh_fraction,
            "max_recorder_wall_gap_s": max_wall_gap_s,
            "max_diag_stamp_gap_s_by_camera": {
                camera: _finite_float(max_diag_gap_by_camera.get(camera))
                for camera in CAMERAS
            },
            "route_boundary_tolerance_s": boundary_tolerance_s,
            "minimum_route_interior_samples": 2,
        },
        "source_camera_csv_sha256": source_hashes,
        "cameras": per_camera,
        "pass": not errors,
    }
    return health, sorted(set(errors))


def _observed_detector_contract_errors(
    readiness: Mapping[str, Any],
    operational: Mapping[str, Any],
    frozen: Mapping[str, str],
) -> tuple[str | None, list[str]]:
    """Independently cross-check the live detector contract on both artifacts.

    The readiness tool validates the retained detector publication, and the
    recorder validates it against its frozen CLI/config/model expectation.  The
    completion validator repeats the non-ROS checks so a later manifest edit
    cannot relabel an observation stream from another detector deployment.
    """

    errors: list[str] = []
    readiness_contract = _dict(readiness.get("detector_runtime_contract"))
    if readiness_contract.get("validated") is not True:
        errors.append("runtime readiness lacks a validated detector runtime contract")
    contract = readiness_contract.get("contract")
    reported_digest = readiness_contract.get("contract_sha256")
    if not isinstance(contract, dict):
        errors.append("runtime readiness detector runtime contract is missing")
        return None, errors
    if not _is_sha256(reported_digest):
        errors.append("runtime readiness detector runtime contract hash is invalid")
        return None, errors
    contract_digest = contract.get("contract_sha256")
    if contract_digest != reported_digest:
        errors.append("runtime readiness detector runtime contract self-hash disagrees")
    without_digest = dict(contract)
    without_digest.pop("contract_sha256", None)
    if _canonical_sha256(without_digest) != reported_digest:
        errors.append("runtime readiness detector runtime contract self-hash is invalid")
    if contract.get("model_sha256") != frozen.get("model"):
        errors.append(
            "runtime readiness detector runtime contract model hash does not match frozen campaign model"
        )
    if contract.get("runtime_mode") != "batched_four_camera":
        errors.append("runtime readiness detector runtime contract is not batched four-camera")
    if contract.get("executable") != "batched_four_camera_yolo_node":
        errors.append("runtime readiness detector runtime contract executable is invalid")
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes or not all(
        _is_sha256(value) for value in source_hashes.values()
    ):
        errors.append("runtime readiness detector runtime contract source hashes are invalid")

    detector_runtime = _dict(operational.get("detector_runtime"))
    if detector_runtime.get("model_sha256") != contract.get("model_sha256"):
        errors.append(
            "operational detector model hash differs from readiness detector runtime contract"
        )
    if detector_runtime.get("observed_contract") != contract:
        errors.append(
            "operational observed detector runtime contract differs from readiness"
        )
    if detector_runtime.get("observed_contract_sha256") != reported_digest:
        errors.append(
            "operational observed detector runtime contract hash differs from readiness"
        )
    observed_contract = detector_runtime.get("observed_contract")
    if isinstance(observed_contract, dict):
        observed_without_digest = dict(observed_contract)
        observed_without_digest.pop("contract_sha256", None)
        if _canonical_sha256(observed_without_digest) != reported_digest:
            errors.append("operational observed detector runtime contract self-hash is invalid")
    requirement = _dict(detector_runtime.get("observed_contract_requirement"))
    if requirement.get("state") != "armed":
        errors.append("operational recorder did not arm on the detector runtime contract")
    if requirement.get("expected_contract_sha256") != reported_digest:
        errors.append(
            "operational expected detector runtime contract hash differs from readiness"
        )
    if requirement.get("expected_contract") != contract:
        errors.append(
            "operational expected detector runtime contract differs from readiness"
        )
    return str(reported_digest), sorted(set(errors))


def _runtime_readiness_binding_errors(
    readiness: Mapping[str, Any],
    operational: Mapping[str, Any],
    ledger: Mapping[str, Any],
    row: Mapping[str, Any],
    expected_attempt_id: str | None,
) -> list[str]:
    """Cross-bind the pre-route report to the one campaign attempt.

    The readiness barrier is published before the recorder begins, so merely
    seeing a passing report is insufficient.  Its immutable identity,
    transport, provenance and method snapshot must be the same ones later
    carried by the operational manifest and declared campaign row.
    """

    errors: list[str] = []
    frozen = input_sha256(ledger["provenance"])
    expected_method_freeze = method_freeze_snapshot(ledger)
    expected_method_freeze_sha256 = method_freeze_sha256(ledger)
    attempt = next(
        (
            item
            for item in row.get("attempts", [])
            if item.get("attempt_id") == expected_attempt_id
        ),
        None,
    )
    if not isinstance(attempt, dict):
        errors.append("runtime readiness cannot resolve the selected campaign attempt")
        expected_transport: dict[str, str] = {}
    else:
        expected_transport = {
            "ROS_LOCALHOST_ONLY": "1",
            "IGN_IP": "127.0.0.1",
            "GZ_IP": "127.0.0.1",
            "IGN_PARTITION": str(attempt.get("ign_partition", "")),
            "ROS_DOMAIN_ID": str(attempt.get("ros_domain_id", "")),
        }

    expected_identity = {
        "plan_row_id": row.get("row_id"),
        "attempt_id": expected_attempt_id,
        "analysis_split": row.get("expected_analysis_split"),
        "seed": _dict(row.get("row_tuple")).get("seed"),
        "evidence_role": row.get("expected_evidence_role"),
    }
    for field, expected in expected_identity.items():
        if readiness.get(field) != expected:
            errors.append(f"runtime readiness {field} does not match the campaign row")
    readiness_seed = readiness.get("seed")
    if isinstance(readiness_seed, bool) or not isinstance(readiness_seed, int):
        errors.append("runtime readiness seed must be an integer")
    readiness_run_id = readiness.get("run_id")
    if not isinstance(readiness_run_id, str) or not readiness_run_id.strip():
        errors.append("runtime readiness run_id is missing")

    campaign = _dict(readiness.get("campaign_contract"))
    expected_campaign = {
        "plan_sha256": ledger.get("plan_sha256"),
        "row_id": row.get("row_id"),
        "row_tuple": row.get("row_tuple"),
        "attempt_id": expected_attempt_id,
        "expected_evidence_role": row.get("expected_evidence_role"),
        "expected_analysis_split": row.get("expected_analysis_split"),
        "input_sha256": frozen,
        "method_freeze": expected_method_freeze,
        "method_freeze_sha256": expected_method_freeze_sha256,
        "transport_environment": expected_transport,
    }
    for field, expected in expected_campaign.items():
        if campaign.get(field) != expected:
            errors.append(
                f"runtime readiness campaign contract {field} does not match the campaign"
            )
    if readiness.get("transport_environment") != expected_transport:
        errors.append("runtime readiness transport environment does not match attempt contract")
    if readiness.get("method_freeze") != expected_method_freeze:
        errors.append("runtime readiness method-freeze does not match campaign snapshot")
    if readiness.get("method_freeze_sha256") != expected_method_freeze_sha256:
        errors.append("runtime readiness method-freeze hash does not match campaign snapshot")

    expected_config_hashes = sorted(
        digest for label, digest in frozen.items() if label.startswith("config:")
    )
    readiness_config_hashes = _manifest_config_hashes(readiness)
    if (
        readiness_config_hashes is None
        or sorted(readiness_config_hashes) != expected_config_hashes
    ):
        errors.append("runtime readiness frozen config hashes do not match campaign configs")
    readiness_provenance = _dict(readiness.get("provenance"))
    expected_provenance_hashes = {
        "study": frozen.get("study"),
        "protocol": frozen.get("protocol"),
        "model": frozen.get("model"),
        "calibration": frozen.get("calibration"),
        "analysis_plan": expected_method_freeze.get("analysis_plan_sha256"),
    }
    for field, expected in expected_provenance_hashes.items():
        if _nested_sha256(readiness_provenance, field) != expected:
            errors.append(
                f"runtime readiness provenance {field} hash does not match campaign"
            )

    # Match the later recorder on every overlapping binding.  The two
    # manifests use intentionally different provenance labels for a few
    # paths, so compare those explicitly rather than incorrectly requiring
    # their whole mappings to have identical keys.
    for field in (
        "run_id",
        "plan_row_id",
        "attempt_id",
        "analysis_split",
        "seed",
        "evidence_role",
    ):
        if readiness.get(field) != operational.get(field):
            errors.append(f"runtime readiness {field} differs from operational recorder")
    operational_campaign = _dict(operational.get("campaign_contract"))
    if campaign != operational_campaign:
        errors.append("runtime readiness campaign contract differs from operational recorder")
    if readiness.get("transport_environment") != operational.get("transport_environment"):
        errors.append("runtime readiness transport environment differs from operational recorder")
    if readiness.get("frozen_configs") != operational.get("frozen_configs"):
        errors.append("runtime readiness frozen configs differ from operational recorder")
    if readiness.get("method_freeze") != operational.get("method_freeze"):
        errors.append("runtime readiness method-freeze differs from operational recorder")
    if readiness.get("method_freeze_sha256") != operational.get("method_freeze_sha256"):
        errors.append("runtime readiness method-freeze hash differs from operational recorder")
    operational_provenance = _dict(operational.get("provenance"))
    for readiness_key, operational_key in (
        ("study", "study_config"),
        ("protocol", "protocol"),
        ("analysis_plan", "analysis_plan"),
        ("model", "detector_model"),
        ("calibration", "projection_calibration"),
    ):
        if readiness_provenance.get(readiness_key) != operational_provenance.get(
            operational_key
        ):
            errors.append(
                "runtime readiness provenance "
                f"{readiness_key} differs from operational recorder"
            )
    return sorted(set(errors))


def runtime_readiness_completion_evidence(
    run_dir: Path,
    ledger: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Build the expected readiness binding and route-health payload.

    This is intentionally pure: the finalizer embeds its result in the
    immutable completion manifest, while validation recomputes it from raw
    artifacts on every ledger refresh.
    """

    errors: list[str] = []
    paths = {
        "readiness": run_dir / RUNTIME_READINESS_ARTIFACT,
        "route_manifest": run_dir / "raw/route_manifest.json",
        "route_completion": run_dir / "raw/route_completion.json",
        "operational": run_dir / "raw/operational_recording_manifest.json",
    }
    payloads: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        try:
            payloads[label] = _load_json(path)
        except LedgerError as exc:
            errors.append(str(exc))

    readiness = payloads.get("readiness", {})
    route_manifest = payloads.get("route_manifest", {})
    route_completion = payloads.get("route_completion", {})
    operational = payloads.get("operational", {})
    readiness_errors, thresholds, readiness_epoch = _runtime_readiness_report_errors(
        readiness
    )
    errors.extend(readiness_errors)

    route_manifest_epoch = _finite_float(route_manifest.get("created_wall_time_s"))
    if route_manifest_epoch is None or route_manifest_epoch <= 0.0:
        errors.append("pre-run route manifest lacks a valid created_wall_time_s")
    elif readiness_epoch is not None and readiness_epoch >= route_manifest_epoch:
        errors.append("runtime readiness timestamp is not before the pre-run route manifest")

    recorder_started_epoch = _utc_epoch(operational.get("started_utc"))
    if recorder_started_epoch is None:
        errors.append("operational recorder started_utc is missing, naive, or invalid")
    elif readiness_epoch is not None and readiness_epoch > recorder_started_epoch:
        errors.append("runtime readiness timestamp is after operational recorder startup")

    expected_attempt_id = _attempt_id_for_run_dir(row, run_dir)
    if expected_attempt_id is None:
        errors.append("runtime readiness cannot identify this pre-declared attempt slot")
    errors.extend(
        _runtime_readiness_binding_errors(
            readiness, operational, ledger, row, expected_attempt_id
        )
    )
    detector_runtime = _dict(operational.get("detector_runtime"))
    detector_model_sha256 = detector_runtime.get("model_sha256")
    frozen = input_sha256(ledger["provenance"])
    expected_config_hashes = sorted(
        digest for label, digest in frozen.items() if label.startswith("config:")
    )
    operational_config_hashes = _manifest_config_hashes(operational)
    if detector_model_sha256 != frozen.get("model"):
        errors.append("runtime readiness detector model hash does not match frozen campaign model")
    if operational_config_hashes is None or sorted(operational_config_hashes) != expected_config_hashes:
        errors.append("runtime readiness detector config hashes do not match frozen campaign configs")
    topology = _dict(detector_runtime.get("topology"))
    if not topology:
        errors.append("runtime readiness detector runtime topology is missing")
    detector_contract_sha256, detector_contract_errors = (
        _observed_detector_contract_errors(readiness, operational, frozen)
    )
    errors.extend(detector_contract_errors)
    readiness_wall_age_s = _finite_float(thresholds.get("max_stream_wall_age_s"))
    watchdog = _dict(operational.get("stream_liveness_watchdog"))
    watchdog_wall_age_s = _finite_float(watchdog.get("max_stream_wall_age_s"))
    if watchdog_wall_age_s is None or watchdog_wall_age_s <= 0.0:
        errors.append("operational stream-liveness watchdog threshold is missing or invalid")
    elif (
        readiness_wall_age_s is None
        or abs(watchdog_wall_age_s - readiness_wall_age_s) > 1.0e-9
    ):
        errors.append(
            "operational stream-liveness watchdog threshold differs from runtime readiness"
        )

    binding: dict[str, Any] | None = None
    if readiness and paths["readiness"].is_file():
        binding = {
            "artifact": RUNTIME_READINESS_ARTIFACT,
            "sha256": _sha256(paths["readiness"]),
            "created_utc": readiness.get("created_utc"),
            "binding": {
                "plan_sha256": ledger.get("plan_sha256"),
                "row_id": row.get("row_id"),
                "attempt_id": expected_attempt_id,
                "input_sha256": frozen,
                "detector_model_sha256": detector_model_sha256,
                "detector_runtime_sha256": _canonical_sha256(detector_runtime),
                "detector_topology_sha256": _canonical_sha256(topology),
                "detector_runtime_contract_sha256": detector_contract_sha256,
                "frozen_config_sha256": operational_config_hashes,
                "campaign_contract_sha256": _canonical_sha256(
                    _dict(readiness.get("campaign_contract"))
                ),
                "method_freeze_sha256": readiness.get("method_freeze_sha256"),
                "stream_liveness_watchdog": {
                    "max_stream_wall_age_s": watchdog_wall_age_s,
                },
            },
        }

    health: dict[str, Any] | None = None
    if route_completion and operational:
        health, health_errors = build_route_camera_health(
            run_dir,
            route_completion=route_completion,
            operational=operational,
            readiness_thresholds=thresholds,
        )
        errors.extend(health_errors)
    return binding, health, sorted(set(errors))


def _artifact_errors(run_dir: Path, completion: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    declared = completion.get("artifacts_sha256")
    if not isinstance(declared, dict):
        return ["completion manifest lacks artifacts_sha256 mapping"]
    for relative in REQUIRED_ARTIFACTS:
        path = run_dir / relative
        if not path.is_file():
            errors.append(f"missing required artifact {relative}")
            continue
        expected_hash = declared.get(relative)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"completion manifest lacks a valid hash for {relative}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            errors.append(f"artifact hash mismatch for {relative}")
    return errors


def _semantic_artifact_errors(
    run_dir: Path, row: Mapping[str, Any], ledger: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    exact = row["row_tuple"]
    matching_attempts = [
        attempt
        for attempt in row["attempts"]
        if (run_dir.parents[2] / str(attempt["run_dir"])).resolve()
        == run_dir.resolve()
    ]
    expected_attempt_id = (
        str(matching_attempts[0]["attempt_id"]) if len(matching_attempts) == 1 else ""
    )
    if not expected_attempt_id:
        errors.append("run directory is not a pre-declared attempt slot")
    frozen = input_sha256(ledger["provenance"])
    frozen_config_hashes = sorted(
        digest for label, digest in frozen.items() if label.startswith("config:")
    )
    expected_method_freeze = method_freeze_snapshot(ledger)
    expected_method_freeze_sha256 = method_freeze_sha256(ledger)
    route_manifest_path = run_dir / "raw/route_manifest.json"
    route_completion_path = run_dir / "raw/route_completion.json"
    try:
        route_manifest = _load_json(route_manifest_path)
    except LedgerError as exc:
        errors.append(str(exc))
        route_manifest = {}
    if route_manifest:
        if route_manifest.get("route") != exact["route"]:
            errors.append("route manifest route does not match campaign tuple")
        try:
            if _float_token(route_manifest.get("lateral_offset_m")) != exact["lateral_offset_m"]:
                errors.append("route manifest lateral offset does not match campaign tuple")
        except (LedgerError, TypeError, ValueError):
            errors.append("route manifest lateral offset is missing or invalid")
        try:
            if _float_token(route_manifest.get("speed_mps")) != exact["speed_mps"]:
                errors.append("route manifest speed does not match campaign tuple")
        except (LedgerError, TypeError, ValueError):
            errors.append("route manifest speed is missing or invalid")
        if route_manifest.get("contains_ground_truth") is not False:
            errors.append("route manifest does not explicitly exclude ground truth")
        expected_study_hash = frozen["study"]
        if route_manifest.get("study_config_sha256") != expected_study_hash:
            errors.append("route manifest study hash does not match frozen study input")
    try:
        route_completion = _load_json(route_completion_path)
    except LedgerError as exc:
        errors.append(str(exc))
        route_completion = {}
    if route_completion:
        if route_completion.get("schema_version") != "bigwarehouse_route_completion_v1":
            errors.append("route completion schema is not bigwarehouse_route_completion_v1")
        if route_completion.get("status") != "completed":
            errors.append("route completion status is not completed")
        if route_completion.get("route_complete") is not True:
            errors.append("route completion route_complete must be true")
        if route_completion.get("contains_ground_truth") is not False:
            errors.append("route completion does not explicitly exclude ground truth")
        if route_completion.get("end_reason") not in SUCCESS_END_REASONS:
            errors.append("route completion end_reason is not successful")
        identity = route_completion.get("route_identity")
        if not isinstance(identity, dict):
            errors.append("route completion lacks route_identity")
            identity = {}
        if identity.get("route") != exact["route"]:
            errors.append("route completion route does not match campaign tuple")
        try:
            if _float_token(identity.get("lateral_offset_m")) != exact["lateral_offset_m"]:
                errors.append("route completion lateral offset does not match campaign tuple")
        except (LedgerError, TypeError, ValueError):
            errors.append("route completion lateral offset is missing or invalid")
        try:
            if _float_token(identity.get("speed_mps")) != exact["speed_mps"]:
                errors.append("route completion speed does not match campaign tuple")
        except (LedgerError, TypeError, ValueError):
            errors.append("route completion speed is missing or invalid")
        if identity.get("study_config_sha256") != frozen["study"]:
            errors.append("route completion study hash does not match frozen study input")
        if route_manifest_path.is_file():
            actual_route_manifest_hash = _sha256(route_manifest_path)
            if identity.get("route_manifest_sha256") != actual_route_manifest_hash:
                errors.append("route completion does not hash the recorded route manifest")
        try:
            waypoint_count = int(route_completion.get("waypoint_count"))
            waypoints_reached = int(route_completion.get("waypoints_reached"))
        except (TypeError, ValueError):
            errors.append("route completion waypoint counts are missing or invalid")
        else:
            if waypoint_count < 1 or waypoints_reached != waypoint_count:
                errors.append("route completion did not reach every waypoint")
        safety_limits = _dict(route_completion.get("safety_limits"))
        try:
            stop_count = int(route_completion.get("terminal_stop_publish_count"))
            required_stop_count = int(safety_limits.get("terminal_stop_publish_count"))
        except (TypeError, ValueError):
            errors.append("route completion terminal-stop counts are missing or invalid")
        else:
            if required_stop_count < 2 or stop_count < required_stop_count:
                errors.append("route completion terminal-stop burst is incomplete")
    try:
        operational = _load_json(run_dir / "raw/operational_recording_manifest.json")
    except LedgerError as exc:
        errors.append(str(exc))
        operational = {}
    if operational and operational.get("contains_ground_truth") is not False:
        errors.append("operational manifest does not explicitly exclude ground truth")
    if operational:
        if operational.get("status") != "completed":
            errors.append("operational recorder manifest is not completed")
        if operational.get("stop_reason") != "route_completion_manifest":
            errors.append("operational recorder was not stopped by route completion")
        if operational.get("plan_row_id") != row["row_id"]:
            errors.append("operational recorder plan_row_id does not match campaign row")
        if operational.get("attempt_id") != expected_attempt_id:
            errors.append("operational recorder attempt_id does not match attempt slot")
        detector_hash = _dict(operational.get("detector_runtime")).get("model_sha256")
        projection_hash = _dict(operational.get("projection_calibration")).get("sha256")
        if detector_hash != frozen.get("model"):
            errors.append("operational detector model hash does not match frozen campaign model")
        if projection_hash != frozen.get("calibration"):
            errors.append("operational projection hash does not match frozen campaign calibration")
        detector_runtime = _dict(operational.get("detector_runtime"))
        topology = _dict(detector_runtime.get("topology"))
        mode = topology.get("runtime_mode")
        common_topology = {
            "model_format": "native_ultralytics",
            "camera_order": list(CAMERAS),
        }
        if any(topology.get(key) != value for key, value in common_topology.items()):
            errors.append("operational detector topology lacks the frozen model format/camera order")
        if mode == "batched_four_camera":
            expected_batched = {
                "executable": "batched_four_camera_yolo_node",
                "launch_switch_yolo_batched_four_camera": True,
                "model_instances": 1,
                "batch_size": 4,
                "shared_device": "0",
                "cpu_threads": 2,
                "interop_threads": 1,
                "opencv_threads": 1,
                "max_batch_stamp_skew_s": 0.10,
                "max_pending_wall_s": 0.50,
                "frame_policy": "strict_new_unique_stamp_latest_only_no_reuse",
                "inference_timing_semantics": (
                    "full_batch_wall_ms_repeated_per_camera_result"
                ),
                "fault_policy": (
                    "fatal_process_exit_and_launch_shutdown_no_synthetic_miss"
                ),
                "synchronization_miss_policy": (
                    "no_output_detected_by_liveness_gate"
                ),
            }
            if any(topology.get(key) != value for key, value in expected_batched.items()):
                errors.append("operational batched-detector topology differs from frozen contract")
        elif mode == "separate_processes":
            expected_separate = {
                "executable": "yolo_robot_detector_node",
                "launch_switch_yolo_batched_four_camera": False,
                "model_instances": 4,
                "instances": 4,
                "batch_size": 1,
            }
            devices = topology.get("devices")
            if (
                any(topology.get(key) != value for key, value in expected_separate.items())
                or not isinstance(devices, dict)
                or set(devices) != set(CAMERAS)
            ):
                errors.append("operational separate-detector topology is incomplete")
        else:
            errors.append("operational detector runtime mode is missing or invalid")
        operational_provenance = _dict(operational.get("provenance"))
        if _nested_sha256(operational_provenance, "study_config") != frozen.get("study"):
            errors.append("operational recorder study hash does not match frozen campaign study")
        if _nested_sha256(operational_provenance, "protocol") != frozen.get("protocol"):
            errors.append("operational recorder protocol hash does not match frozen campaign protocol")
        operational_configs = _manifest_config_hashes(operational)
        if operational_configs is None or sorted(operational_configs) != frozen_config_hashes:
            errors.append("operational recorder config hashes do not match frozen campaign configs")
        operational_campaign = _dict(operational.get("campaign_contract"))
        if (
            operational_campaign.get("plan_sha256") != ledger.get("plan_sha256")
            or operational_campaign.get("row_id") != row.get("row_id")
            or operational_campaign.get("row_tuple") != row.get("row_tuple")
            or operational_campaign.get("attempt_id") != expected_attempt_id
            or operational_campaign.get("expected_evidence_role")
            != row.get("expected_evidence_role")
            or operational_campaign.get("expected_analysis_split")
            != row.get("expected_analysis_split")
            or operational_campaign.get("input_sha256") != frozen
        ):
            errors.append("operational recorder campaign preflight contract is missing or mismatched")
        if operational.get("transport_environment") != operational_campaign.get(
            "transport_environment"
        ):
            errors.append("operational recorder transport environment is missing or mismatched")
        if operational.get("method_freeze") != expected_method_freeze:
            errors.append("operational recorder method-freeze does not match campaign snapshot")
        if (
            operational_campaign.get("method_freeze") != expected_method_freeze
            or operational_campaign.get("method_freeze_sha256")
            != expected_method_freeze_sha256
        ):
            errors.append(
                "operational recorder campaign method-freeze binding is missing or mismatched"
            )
        minimums = _dict(operational.get("minimum_stream_rows"))
        try:
            minimum_camera_rows = int(minimums.get("per_camera"))
            minimum_odom_rows = int(minimums.get("odometry"))
        except (TypeError, ValueError):
            minimum_camera_rows = -1
            minimum_odom_rows = -1
            errors.append("operational recorder minimum stream-row contract is missing")
        if minimum_camera_rows < 1 or minimum_odom_rows < 1:
            errors.append("operational recorder minimum stream-row gates must be positive")
        camera_rows = _dict(operational.get("camera_rows"))
        camera_stamps: dict[str, list[float]] = {}
        for camera in CAMERAS:
            camera_csv = run_dir / f"raw/{camera}_perception.csv"
            actual_rows = _csv_data_rows(camera_csv)
            try:
                declared_rows = int(camera_rows.get(camera))
            except (TypeError, ValueError):
                declared_rows = -1
            if declared_rows != actual_rows:
                errors.append(f"{camera} manifest row count does not match its CSV")
            if declared_rows < minimum_camera_rows:
                errors.append(f"{camera} did not meet the declared recorder row minimum")
            stamps = _csv_finite_column(camera_csv, "diag_stamp")
            if (
                stamps is None
                or len(stamps) != actual_rows
                or any(right <= left for left, right in zip(stamps, stamps[1:]))
            ):
                errors.append(f"{camera} timestamps are missing, non-finite, duplicate, or out of order")
            else:
                camera_stamps[camera] = stamps
        actual_odom_rows = _csv_data_rows(run_dir / "raw/experiment.csv")
        try:
            declared_odom_rows = int(operational.get("odometry_rows"))
        except (TypeError, ValueError):
            declared_odom_rows = -1
        if declared_odom_rows != actual_odom_rows:
            errors.append("odometry manifest row count does not match its CSV")
        if declared_odom_rows < minimum_odom_rows:
            errors.append("odometry did not meet the declared recorder row minimum")
        odom_stamps = _csv_finite_column(run_dir / "raw/experiment.csv", "stamp")
        if (
            odom_stamps is None
            or len(odom_stamps) != actual_odom_rows
            or any(right <= left for left, right in zip(odom_stamps, odom_stamps[1:]))
        ):
            errors.append("odometry timestamps are missing, non-finite, duplicate, or out of order")

        if mode == "batched_four_camera" and set(camera_stamps) == set(CAMERAS):
            if len({len(values) for values in camera_stamps.values()}) != 1:
                errors.append("batched detector camera CSV row counts are not equal")
            else:
                for batch in zip(*(camera_stamps[camera] for camera in CAMERAS), strict=True):
                    if max(batch) - min(batch) > 0.10 + 1.0e-9:
                        errors.append("batched detector CSV stamps exceed frozen skew bound")
                        break

        route_start = _finite_float(route_completion.get("route_started_sim_time_s"))
        route_end = _finite_float(route_completion.get("route_completed_sim_time_s"))
        fresh_age = _finite_float(operational.get("fresh_age_s"))
        camera_ranges = _dict(operational.get("camera_stamp_ranges_s"))
        if route_start is None or route_end is None or route_end < route_start:
            errors.append("route completion lacks a valid simulation-time interval")
        elif fresh_age is None or fresh_age < 0.0:
            errors.append("operational recorder fresh-age contract is invalid")
        else:
            for camera in CAMERAS:
                interval = _dict(camera_ranges.get(camera))
                first = _finite_float(interval.get("first"))
                last = _finite_float(interval.get("last"))
                if first is None or last is None or last < first:
                    errors.append(f"{camera} manifest stamp range is missing or invalid")
                elif first > route_start + fresh_age or last < route_end - fresh_age:
                    errors.append(f"{camera} did not cover the completed route interval")
            odom_interval = _dict(operational.get("odometry_stamp_range_s"))
            odom_first = _finite_float(odom_interval.get("first"))
            odom_last = _finite_float(odom_interval.get("last"))
            odom_freshness = _finite_float(
                _dict(route_completion.get("safety_limits")).get(
                    "odom_freshness_timeout_s"
                )
            )
            if odom_freshness is None or odom_freshness < 0.0:
                errors.append("route completion lacks an odometry freshness contract")
            elif (
                odom_first is None
                or odom_last is None
                or odom_last < odom_first
                or odom_first > route_start + odom_freshness
                or odom_last < route_end - odom_freshness
            ):
                errors.append("odometry did not cover the completed route interval")
        completion_reference = operational.get("route_completion_manifest")
        if not isinstance(completion_reference, dict):
            errors.append("operational recorder lacks route completion provenance")
        elif route_completion_path.is_file() and completion_reference.get("sha256") != _sha256(
            route_completion_path
        ):
            errors.append("operational recorder route completion hash does not match driver artifact")
    try:
        evaluation = _load_json(run_dir / "evaluation_only/evaluation_truth_manifest.json")
    except LedgerError as exc:
        errors.append(str(exc))
        evaluation = {}
    if evaluation:
        if evaluation.get("status") != "completed":
            errors.append("truth recorder manifest is not completed")
        if evaluation.get("stop_reason") != "route_completion_manifest":
            errors.append("truth recorder was not stopped by route completion")
        if evaluation.get("plan_row_id") != row["row_id"]:
            errors.append("truth recorder plan_row_id does not match campaign row")
        if evaluation.get("attempt_id") != expected_attempt_id:
            errors.append("truth recorder attempt_id does not match attempt slot")
        if evaluation.get("evaluation_only") is not True:
            errors.append("truth manifest is not marked evaluation_only")
        if evaluation.get("contains_ground_truth") is not True:
            errors.append("truth manifest does not explicitly contain ground truth")
        evaluation_provenance = _dict(evaluation.get("provenance"))
        if _nested_sha256(evaluation_provenance, "study_config") != frozen.get("study"):
            errors.append("truth recorder study hash does not match frozen campaign study")
        if _nested_sha256(evaluation_provenance, "protocol") != frozen.get("protocol"):
            errors.append("truth recorder protocol hash does not match frozen campaign protocol")
        evaluation_configs = _manifest_config_hashes(evaluation)
        if evaluation_configs is None or sorted(evaluation_configs) != frozen_config_hashes:
            errors.append("truth recorder config hashes do not match frozen campaign configs")
        evaluation_campaign = _dict(evaluation.get("campaign_contract"))
        if (
            evaluation_campaign.get("plan_sha256") != ledger.get("plan_sha256")
            or evaluation_campaign.get("row_id") != row.get("row_id")
            or evaluation_campaign.get("row_tuple") != row.get("row_tuple")
            or evaluation_campaign.get("attempt_id") != expected_attempt_id
            or evaluation_campaign.get("expected_evidence_role")
            != row.get("expected_evidence_role")
            or evaluation_campaign.get("expected_analysis_split")
            != row.get("expected_analysis_split")
            or evaluation_campaign.get("input_sha256") != frozen
        ):
            errors.append("truth recorder campaign preflight contract is missing or mismatched")
        if evaluation.get("transport_environment") != evaluation_campaign.get(
            "transport_environment"
        ):
            errors.append("truth recorder transport environment is missing or mismatched")
        if (
            evaluation_campaign.get("method_freeze") != expected_method_freeze
            or evaluation_campaign.get("method_freeze_sha256")
            != expected_method_freeze_sha256
        ):
            errors.append("truth recorder campaign method-freeze binding is missing or mismatched")
        if _nested_sha256(evaluation_provenance, "analysis_plan") not in frozen_config_hashes:
            errors.append("truth recorder analysis-plan hash is not frozen by the campaign")
        actual_truth_rows = _csv_data_rows(run_dir / "evaluation_only/ground_truth.csv")
        try:
            declared_truth_rows = int(evaluation.get("sample_count"))
            minimum_truth_rows = int(evaluation.get("minimum_samples"))
        except (TypeError, ValueError):
            declared_truth_rows = -1
            minimum_truth_rows = -1
        if declared_truth_rows != actual_truth_rows:
            errors.append("truth manifest sample count does not match its CSV")
        if minimum_truth_rows < 1 or declared_truth_rows < minimum_truth_rows:
            errors.append("truth recorder did not meet a positive declared sample minimum")
        truth_stamps = _csv_finite_column(
            run_dir / "evaluation_only/ground_truth.csv", "stamp"
        )
        if (
            truth_stamps is None
            or len(truth_stamps) != actual_truth_rows
            or any(right <= left for left, right in zip(truth_stamps, truth_stamps[1:]))
        ):
            errors.append("truth timestamps are missing, non-finite, duplicate, or out of order")
        completion_reference = evaluation.get("route_completion_manifest")
        if not isinstance(completion_reference, dict):
            errors.append("truth recorder lacks route completion provenance")
        elif route_completion_path.is_file() and completion_reference.get("sha256") != _sha256(
            route_completion_path
        ):
            errors.append("truth recorder route completion hash does not match driver artifact")
    if operational and evaluation:
        operational_run_id = operational.get("run_id")
        evaluation_run_id = evaluation.get("run_id")
        if not isinstance(operational_run_id, str) or not operational_run_id.strip():
            errors.append("operational recorder run_id is missing")
        if operational_run_id != evaluation_run_id:
            errors.append("operational and truth recorder run_id values disagree")
        if operational.get("attempt_id") != evaluation.get("attempt_id"):
            errors.append("operational and truth recorder attempt_id values disagree")
        if operational.get("analysis_split") != evaluation.get("analysis_split"):
            errors.append("operational and truth recorder analysis splits disagree")
        operational_seed = operational.get("seed")
        evaluation_seed = evaluation.get("seed")
        if (
            isinstance(operational_seed, bool)
            or not isinstance(operational_seed, int)
            or isinstance(evaluation_seed, bool)
            or not isinstance(evaluation_seed, int)
        ):
            errors.append("recorder seed values must be integers")
        elif operational_seed != evaluation_seed:
            errors.append("operational and truth recorder seed values disagree")
        operational_role = operational.get("evidence_role")
        evaluation_role = evaluation.get("evidence_role")
        if operational_role not in {"fit", "qualification", "diagnostic"}:
            errors.append("operational recorder evidence role is invalid")
        if operational_role != evaluation_role:
            errors.append("operational and truth recorder evidence roles disagree")
        if operational_role != row.get("expected_evidence_role"):
            errors.append("recorder evidence role does not match the pre-declared campaign phase")
        if operational.get("analysis_split") != row.get("expected_analysis_split"):
            errors.append("recorder analysis split does not match the pre-declared campaign row")
        row_seed = exact.get("seed")
        if operational_seed != row_seed:
            errors.append("recorder seed does not match campaign tuple")
    csv_requirements = {
        "raw/experiment.csv": "operational odometry",
        "evaluation_only/ground_truth.csv": "evaluation truth",
        **{
            f"raw/{camera}_perception.csv": f"{camera} operational"
            for camera in CAMERAS
        },
    }
    for relative, label in csv_requirements.items():
        if _csv_data_rows(run_dir / relative) <= 0:
            errors.append(f"{label} CSV has zero data rows")
    return errors


def validate_completion_payload(
    run_dir: Path,
    ledger: Mapping[str, Any],
    row: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> list[str]:
    """Validate a proposed or published completion payload independently."""

    errors: list[str] = []
    matching_attempts = [
        attempt
        for attempt in row["attempts"]
        if (run_dir.parents[2] / str(attempt["run_dir"])).resolve()
        == run_dir.resolve()
    ]
    expected_attempt_id = (
        str(matching_attempts[0]["attempt_id"]) if len(matching_attempts) == 1 else ""
    )
    if completion.get("schema_version") != 1:
        errors.append("completion manifest schema_version must be 1")
    if completion.get("row_id") != row["row_id"]:
        errors.append("completion manifest row_id does not match expected row")
    if completion.get("attempt_id") != expected_attempt_id:
        errors.append("completion manifest attempt_id does not match attempt slot")
    if completion.get("row_tuple") != row["row_tuple"]:
        errors.append("completion manifest does not contain the exact six-field tuple")
    if completion.get("input_sha256") != input_sha256(ledger["provenance"]):
        errors.append("completion manifest frozen-input hashes do not match campaign")
    if completion.get("method_freeze") != method_freeze_snapshot(ledger):
        errors.append("completion manifest method-freeze snapshot does not match campaign")
    if completion.get("method_freeze_sha256") != method_freeze_sha256(ledger):
        errors.append("completion manifest method-freeze hash does not match campaign")
    expected_readiness, expected_route_health, readiness_errors = (
        runtime_readiness_completion_evidence(run_dir, ledger, row)
    )
    errors.extend(readiness_errors)
    if expected_readiness is None:
        errors.append("completion manifest lacks reconstructable runtime readiness evidence")
    elif completion.get("runtime_readiness") != expected_readiness:
        errors.append(
            "completion manifest runtime readiness binding does not match immutable artifacts"
        )
    if expected_route_health is None:
        errors.append("completion manifest lacks reconstructable route-wide camera health evidence")
    elif completion.get("route_camera_health") != expected_route_health:
        errors.append(
            "completion manifest route-wide camera health does not match immutable raw streams"
        )
    route_completion = completion.get("route_completion")
    if not isinstance(route_completion, dict):
        errors.append("completion manifest lacks route_completion")
    else:
        if route_completion.get("route_complete") is not True:
            errors.append("route_completion.route_complete must be true")
        reason = route_completion.get("end_reason")
        if reason not in SUCCESS_END_REASONS:
            errors.append(f"route completion end_reason is not successful: {reason!r}")
        route_completion_path = run_dir / "raw/route_completion.json"
        if route_completion_path.is_file():
            try:
                recorded_route_completion = _load_json(route_completion_path)
            except LedgerError as exc:
                errors.append(str(exc))
            else:
                if route_completion != recorded_route_completion:
                    errors.append(
                        "completion manifest route_completion does not equal driver artifact"
                    )
    run_identity = completion.get("run_identity")
    if not isinstance(run_identity, dict):
        errors.append("completion manifest lacks run_identity")
    else:
        try:
            operational = _load_json(run_dir / "raw/operational_recording_manifest.json")
            evaluation = _load_json(
                run_dir / "evaluation_only/evaluation_truth_manifest.json"
            )
        except LedgerError as exc:
            errors.append(str(exc))
        else:
            expected_run_identity = {
                "run_id": operational.get("run_id"),
                "plan_row_id": operational.get("plan_row_id"),
                "attempt_id": operational.get("attempt_id"),
                "seed": operational.get("seed"),
                "truth_run_id": evaluation.get("run_id"),
                "truth_plan_row_id": evaluation.get("plan_row_id"),
                "truth_attempt_id": evaluation.get("attempt_id"),
                "truth_seed": evaluation.get("seed"),
            }
            if run_identity != expected_run_identity:
                errors.append("completion manifest run_identity does not match recorder manifests")
    errors.extend(_artifact_errors(run_dir, completion))
    if not errors or all(not error.startswith("missing required artifact") for error in errors):
        # Semantic checks give useful independent failures even when a declared
        # artifact hash is wrong, but only after all required paths exist.
        errors.extend(_semantic_artifact_errors(run_dir, row, ledger))
    return sorted(set(errors))


def validate_completed_attempt(
    campaign_root: Path,
    ledger: Mapping[str, Any],
    row: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> list[str]:
    """Strictly validate one pre-declared attempt's completion contract."""

    run_dir = campaign_root / str(attempt["run_dir"])
    completion_path = run_dir / "completion_manifest.json"
    try:
        completion = _load_json(completion_path)
    except LedgerError as exc:
        return [str(exc)]
    return validate_completion_payload(run_dir, ledger, row, completion)


def validate_completed_row(
    campaign_root: Path, ledger: Mapping[str, Any], row: Mapping[str, Any]
) -> list[str]:
    """Compatibility wrapper validating the row's selected or first attempt."""

    selected = row.get("selected_attempt_id")
    attempt = next(
        (
            item
            for item in row["attempts"]
            if selected is None or item["attempt_id"] == selected
        ),
        row["attempts"][0],
    )
    return validate_completed_attempt(campaign_root, ledger, row, attempt)


def _disk_duplicate_errors(campaign_root: Path, ledger: Mapping[str, Any]) -> dict[str, list[str]]:
    expected = {
        str(row["row_id"]): {
            (campaign_root / str(attempt["run_dir"]) / "completion_manifest.json").resolve()
            for attempt in row["attempts"]
        }
        for row in ledger["rows"]
    }
    locations: dict[str, list[Path]] = defaultdict(list)
    for path in campaign_root.glob("**/completion_manifest.json"):
        try:
            payload = _load_json(path)
        except LedgerError:
            continue
        identifier = payload.get("row_id")
        if isinstance(identifier, str):
            locations[identifier].append(path.resolve())
    errors: dict[str, list[str]] = defaultdict(list)
    for identifier, paths in locations.items():
        if identifier not in expected:
            continue
        expected_paths = expected[identifier]
        unique = sorted(set(paths))
        if len(unique) > 1:
            errors[identifier].append(
                "duplicate completion manifests for row: " + ", ".join(str(path) for path in unique)
            )
        if any(path not in expected_paths for path in unique):
            errors[identifier].append("completion manifest exists outside pre-declared attempt slots")
    return errors


def evaluate_campaign(campaign_root: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Return fresh row states without mutating the supplied ledger."""

    validate_ledger_contract(ledger)
    verify_frozen_inputs(ledger)
    duplicate_errors = _disk_duplicate_errors(campaign_root, ledger)
    evaluated = copy.deepcopy(dict(ledger))
    now = _utc_now()
    for row in evaluated["rows"]:
        errors = list(duplicate_errors.get(row["row_id"], []))
        attempt_states: list[dict[str, Any]] = []
        terminal_or_active_seen = False
        completed_attempts: list[Mapping[str, Any]] = []
        in_progress_attempts: list[Mapping[str, Any]] = []
        failed_attempts: list[Mapping[str, Any]] = []
        for attempt in row["attempts"]:
            run_dir = campaign_root / str(attempt["run_dir"])
            completion_path = run_dir / "completion_manifest.json"
            failure_path = run_dir / "failure_manifest.json"
            attempt_errors: list[str] = []
            if run_dir.is_symlink():
                attempt_status = "invalid"
                attempt_errors.append("attempt directory must not be a symlink")
            elif not run_dir.exists():
                attempt_status = "planned"
            elif completion_path.exists() and failure_path.exists():
                attempt_status = "invalid"
                attempt_errors.append("attempt has both completion and failure manifests")
            elif completion_path.exists():
                attempt_errors.extend(
                    validate_completed_attempt(campaign_root, evaluated, row, attempt)
                )
                attempt_status = "completed" if not attempt_errors else "invalid"
            elif failure_path.exists():
                try:
                    failure = _load_json(failure_path)
                except LedgerError as exc:
                    attempt_errors.append(str(exc))
                else:
                    if failure.get("schema_version") != 1:
                        attempt_errors.append("failure manifest schema_version must be 1")
                    if failure.get("row_id") != row["row_id"]:
                        attempt_errors.append("failure manifest row_id mismatch")
                    if failure.get("attempt_id") != attempt["attempt_id"]:
                        attempt_errors.append("failure manifest attempt_id mismatch")
                    if failure.get("row_tuple") != row["row_tuple"]:
                        attempt_errors.append("failure manifest row tuple mismatch")
                    if failure.get("plan_sha256") != evaluated.get("plan_sha256"):
                        attempt_errors.append("failure manifest plan hash mismatch")
                    if failure.get("input_sha256") != input_sha256(
                        evaluated["provenance"]
                    ):
                        attempt_errors.append("failure manifest frozen-input hashes mismatch")
                    if not str(failure.get("reason", "")).strip():
                        attempt_errors.append("failure manifest reason is empty")
                    declared_failed_artifacts = failure.get(
                        "preserved_artifacts_sha256"
                    )
                    actual_failed_artifacts = _attempt_artifact_hashes(run_dir)
                    if declared_failed_artifacts != actual_failed_artifacts:
                        attempt_errors.append("quarantined attempt artifacts changed after sealing")
                attempt_status = "failed" if not attempt_errors else "invalid"
            else:
                attempt_status = "in_progress"

            if attempt_status != "planned":
                if terminal_or_active_seen and attempt_status != "planned":
                    # Multiple failed attempts are the sole legal sequence;
                    # every other multi-slot occupancy is out of order.
                    previous_statuses = [item["status"] for item in attempt_states]
                    if not all(value == "failed" for value in previous_statuses):
                        attempt_errors.append("attempt slot used out of order")
                        attempt_status = "invalid"
                terminal_or_active_seen = True
            elif terminal_or_active_seen and any(
                item["status"] in {"completed", "in_progress", "invalid"}
                for item in attempt_states
            ):
                # Later planned slots after an active/completed attempt are normal.
                pass

            state = {
                **attempt,
                "status": attempt_status,
                "validation_errors": sorted(set(attempt_errors)),
            }
            attempt_states.append(state)
            errors.extend(attempt_errors)
            if attempt_status == "completed":
                completed_attempts.append(attempt)
            elif attempt_status == "in_progress":
                in_progress_attempts.append(attempt)
            elif attempt_status == "failed":
                failed_attempts.append(attempt)

        # A later occupied slot may only follow explicitly failed slots.
        occupied_indices = [
            index for index, state in enumerate(attempt_states) if state["status"] != "planned"
        ]
        if occupied_indices and occupied_indices != list(range(max(occupied_indices) + 1)):
            errors.append("attempt slots were not used sequentially")
        if len(completed_attempts) > 1:
            errors.append("more than one validated completion exists for the row")
        if errors:
            status = "failed"
            reason = "attempt validation failed"
        elif len(completed_attempts) == 1:
            status = "completed"
            reason = "exactly one attempt completion and all artifacts validated"
        elif len(in_progress_attempts) == 1:
            status = "in_progress"
            reason = "one immutable attempt is active"
        elif len(failed_attempts) == len(row["attempts"]):
            status = "failed"
            reason = "all pre-declared attempt slots are quarantined"
        else:
            status = "planned"
            reason = "next attempt slot is available"
        row["status"] = status
        row["status_reason"] = reason
        row["validation_errors"] = sorted(set(errors))
        row["attempt_states"] = attempt_states
        row["selected_attempt_id"] = (
            completed_attempts[0]["attempt_id"] if len(completed_attempts) == 1 else None
        )
        row["selected_run_dir"] = (
            completed_attempts[0]["run_dir"] if len(completed_attempts) == 1 else None
        )
        next_planned = next(
            (state for state in attempt_states if state["status"] == "planned"), None
        )
        row["next_attempt_id"] = (
            next_planned["attempt_id"] if status == "planned" and next_planned else None
        )
        row["validated_at"] = now

    # Domain IDs are deterministic but intentionally recycled after 200
    # values.  Treating them as concurrent leases would let a large campaign
    # cross-talk.  Revalidate the physical attempt directories as a global
    # invariant, not merely one row at a time.
    active_attempts = active_campaign_attempts(campaign_root, evaluated)
    if len(active_attempts) > 1:
        labels = ", ".join(
            f"{item['row_id']}/{item['attempt_id']}" for item in active_attempts
        )
        violation = (
            "more than one campaign attempt is active; ROS domain leases are not "
            f"exclusive ({labels})"
        )
        occupied = {(item["row_id"], item["attempt_id"]) for item in active_attempts}
        for row in evaluated["rows"]:
            involved = [
                state
                for state in row.get("attempt_states", [])
                if (str(row.get("row_id", "")), str(state.get("attempt_id", "")))
                in occupied
            ]
            if not involved:
                continue
            for state in involved:
                state["status"] = "invalid"
                state["validation_errors"] = sorted(
                    set(list(state.get("validation_errors", [])) + [violation])
                )
            row["status"] = "failed"
            row["status_reason"] = "campaign-wide active-attempt safety violation"
            row["validation_errors"] = sorted(
                set(list(row.get("validation_errors", [])) + [violation])
            )
            row["selected_attempt_id"] = None
            row["selected_run_dir"] = None
            row["next_attempt_id"] = None
    evaluated["summary"] = _summary(evaluated["rows"])
    return evaluated


def refresh_campaign(campaign_root: Path) -> dict[str, Any]:
    """Revalidate every row and atomically persist the resulting statuses."""

    root = campaign_root.expanduser().resolve()
    ledger_path = root / DEFAULT_LEDGER_NAME
    with _campaign_lock(root):
        if not ledger_path.is_file():
            raise LedgerError(f"No campaign ledger at {ledger_path}; run plan first")
        current = _load_json(ledger_path)
        evaluated = evaluate_campaign(root, current)
        evaluated["revision"] = int(current.get("revision", 0)) + 1
        evaluated["updated_at"] = _utc_now()
        _atomic_write_json(ledger_path, evaluated)
        return evaluated


def runnable_rows(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rows a runner may execute after a fresh refresh.

    Completed is the only skipped state.  Failed rows retain their immutable
    run dirs and therefore require a new campaign root or explicit future retry
    design; this foundation never overwrites them.
    """

    return [copy.deepcopy(row) for row in ledger["rows"] if row["status"] != "completed"]


def _report(ledger: Mapping[str, Any], *, include_rows: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "campaign_kind": ledger["campaign_kind"],
        "protocol_id": ledger["protocol_id"],
        "study_id": ledger["study_id"],
        "plan_sha256": ledger["plan_sha256"],
        "revision": ledger["revision"],
        "summary": ledger["summary"],
    }
    if include_rows:
        report["rows"] = [
            {
                "row_id": row["row_id"],
                "row_tuple": row["row_tuple"],
                "expected_evidence_role": row["expected_evidence_role"],
                "expected_analysis_split": row["expected_analysis_split"],
                "run_dir": row["run_dir"],
                "attempts": row["attempts"],
                "attempt_states": row.get("attempt_states", []),
                "selected_attempt_id": row.get("selected_attempt_id"),
                "selected_run_dir": row.get("selected_run_dir"),
                "next_attempt_id": row.get("next_attempt_id"),
                "status": row["status"],
                "status_reason": row["status_reason"],
                "validation_errors": row["validation_errors"],
            }
            for row in ledger["rows"]
        ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="create or resume an identical campaign")
    plan_parser.add_argument("--campaign-root", type=Path, required=True)
    plan_parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    plan_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    plan_parser.add_argument("--model", type=Path, required=True)
    plan_parser.add_argument("--calibration", type=Path, required=True)
    plan_parser.add_argument(
        "--attempt-slots",
        type=int,
        default=DEFAULT_ATTEMPT_SLOTS,
        help="Pre-declared immutable attempt slots per row (default: 3).",
    )
    plan_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="additional frozen runtime config; repeat for every input",
    )

    for command in ("validate", "status"):
        command_parser = subparsers.add_parser(command, help=f"{command} campaign rows")
        command_parser.add_argument("--campaign-root", type=Path, required=True)
        command_parser.add_argument(
            "--rows",
            action="store_true",
            help="include all row-level states and validation errors",
        )

    quarantine_parser = subparsers.add_parser(
        "quarantine", help="immutably seal one failed attempt and release the next slot"
    )
    quarantine_parser.add_argument("--campaign-root", type=Path, required=True)
    quarantine_parser.add_argument("--row-id", required=True)
    quarantine_parser.add_argument("--attempt-id", required=True)
    quarantine_parser.add_argument("--reason", required=True)

    args = parser.parse_args()
    try:
        if args.command == "plan":
            desired = build_ledger(
                study_path=args.study,
                protocol_path=args.protocol,
                model_path=args.model,
                calibration_path=args.calibration,
                config_paths=args.config,
                attempt_slots=args.attempt_slots,
            )
            initialize_campaign(args.campaign_root, desired)
            ledger = refresh_campaign(args.campaign_root)
            include_rows = False
        elif args.command in {"validate", "status"}:
            # Both commands refresh.  "status" is concise, while "validate"
            # has a distinct name for automation and a non-zero invalid-row
            # exit status.
            ledger = refresh_campaign(args.campaign_root)
            include_rows = bool(args.rows or args.command == "validate")
        else:
            payload = quarantine_attempt(
                args.campaign_root,
                row_id_value=str(args.row_id),
                attempt_id=str(args.attempt_id),
                reason=str(args.reason),
            )
            ledger = refresh_campaign(args.campaign_root)
            include_rows = True
        print(json.dumps(_report(ledger, include_rows=include_rows), indent=2, sort_keys=True))
        if args.command == "validate" and ledger["summary"]["failed"]:
            return 2
        return 0
    except LedgerError as exc:
        parser.exit(2, f"campaign ledger error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
