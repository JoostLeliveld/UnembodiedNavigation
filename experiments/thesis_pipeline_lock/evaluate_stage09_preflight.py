#!/usr/bin/env python3
"""Evaluate the frozen Stage-09 runtime preflight against its release gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ALIGNED_PATH = REPO / "experiments/fusion_on_fixed_routes/aligned.py"
TERMINAL_ACK_STATUS_BY_COMPONENT = {
    "planner": "stop_command_published",
    "camera_manager": "correction_stream_quiescent",
    "detector": "outcome_stream_quiescent",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligned():
    spec = importlib.util.spec_from_file_location("stage09_aligned", ALIGNED_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def percentile(values: np.ndarray, probability: float) -> float | None:
    return float(np.quantile(values, probability)) if values.size else None


def nested_values(value, key: str):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from nested_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from nested_values(child, key)


def bootstrap_verified(manager_journal: Path) -> tuple[bool, list[dict]]:
    evidence = []
    with manager_journal.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for item in nested_values(record, "bootstrap_evidence"):
                if isinstance(item, dict):
                    evidence.append(item)
    unique = []
    seen = set()
    for item in evidence:
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"))
        if encoded not in seen:
            seen.add(encoded)
            unique.append(item)
    passed = any(
        item.get("prior_used") is True
        and item.get("support_count") == 2
        and item.get("required_support_count") == 2
        and len(item.get("camera_ids", ())) == 1
        for item in unique
    )
    return passed, unique


def resolved_preflight_gate(protocol_path: Path, protocol: dict) -> dict:
    """Resolve the frozen gate, including an explicitly hashed parent fragment."""
    gate = protocol.get("preflight_release_gate")
    if not isinstance(gate, dict):
        raise RuntimeError("protocol has no preflight release gate")
    if "quantitative" in gate:
        return gate
    inherited = gate.get("inherits")
    if not isinstance(inherited, str) or "#" not in inherited:
        raise RuntimeError("preflight release gate has no quantitative thresholds")
    relative, fragment = inherited.split("#", 1)
    parent_path = (REPO / relative).resolve()
    try:
        parent_path.relative_to(REPO)
    except ValueError as exc:
        raise RuntimeError("inherited preflight gate is outside the repository") from exc
    supersedes = protocol.get("supersedes", {})
    if supersedes.get("path") != relative:
        raise RuntimeError("inherited preflight gate is not the frozen superseded protocol")
    if sha256(parent_path) != supersedes.get("sha256"):
        raise RuntimeError("inherited preflight-gate protocol bytes changed")
    value = json.loads(parent_path.read_text())
    for key in filter(None, fragment.split("/")):
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"missing inherited protocol fragment: {fragment}")
        value = value[key]
    if not isinstance(value, dict) or "quantitative" not in value:
        raise RuntimeError("inherited preflight gate is malformed")
    merged = dict(value)
    merged.update({key: item for key, item in gate.items() if key != "inherits"})
    merged["inherited_from"] = inherited
    return merged


def final_journal_marker(path: Path) -> dict | None:
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    return records[-1] if records and isinstance(records[-1], dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if not str(protocol.get("status", "")).startswith("frozen_before_runtime_preflight_"):
        raise RuntimeError("protocol is not a frozen runtime-preflight release protocol")
    campaign_root = args.campaign_root.resolve()
    execution = protocol["execution"]
    expected_root = (REPO / execution["runtime_preflight_log_root"]).resolve()
    if campaign_root != expected_root:
        raise RuntimeError("preflight root differs from the frozen protocol")
    config = (REPO / execution["runtime_preflight_config_path"]).resolve()
    if sha256(config) != execution["runtime_preflight_config_sha256"]:
        raise RuntimeError("preflight configuration bytes changed")

    campaign_log_path = campaign_root / "campaign_log.json"
    campaign_log = json.loads(campaign_log_path.read_text())
    expected_key = "thesis09_west_to_east_north__P1__seed899"
    if set(campaign_log) != {expected_key}:
        raise RuntimeError("campaign ledger differs from the one frozen preflight cell")
    entry = campaign_log[expected_key]
    run = Path(str(entry.get("run_dir", ""))).resolve()
    attempt = Path(str(entry.get("run_log_dir", ""))).resolve()
    if not run.is_dir() or not attempt.is_dir() or run.parent != attempt:
        raise RuntimeError("selected attempt has no identified run directory")
    verdict_path = attempt / "attempt_evidence_verdict.json"
    verdict = json.loads(verdict_path.read_text())
    manifest_path = run / "run_manifest.json"
    summary_path = run / "run_summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())

    aligned = load_aligned()
    ledger_validation_error = None
    try:
        aligned.validate_run_ledger(run)
    except Exception as exc:  # Preserve a formal failing report for malformed evidence.
        ledger_validation_error = f"{type(exc).__name__}: {exc}"
    start, stop = aligned.mission_interval(run)
    reference_gap = 0.15
    if ledger_validation_error is None:
        belief = aligned.aligned_error_cm(
            run, "belief", max_reference_gap_s=reference_gap
        )
        identity_mask = (
            aligned.latest_revision_mask(
                belief["stamp"], belief["revision"], belief["epoch"]
            )
            if belief.get("estimate_selection")
            == "highest_anchor_revision_per_state_timestamp"
            else aligned.landed_mask(belief["stamp"])
        )
        belief_mask = (
            identity_mask
            & belief["have"]
            & belief["reference_supported"]
            & (belief["stamp"] >= start)
            & (belief["stamp"] <= stop)
            & np.isfinite(belief["aligned_cm"])
        )
        belief_m = belief["aligned_cm"][belief_mask] / 100.0
        fused = [
            row for row in aligned.fused_answers(
                run, max_reference_gap_s=reference_gap
            )
            if start <= row["fused_stamp"] <= stop
        ]
        fused_m = np.asarray([row["error_cm"] / 100.0 for row in fused], dtype=float)
        accounting = aligned.correction_accounting(run)
    else:
        belief = {}
        belief_m = np.asarray([], dtype=float)
        fused_m = np.asarray([], dtype=float)
        accounting = {
            "accepted_updates": 0,
            "rejected": 0,
            "longest_correction_gap_s": None,
        }
    fresh_total = accounting["accepted_updates"] + accounting["rejected"]
    accepted_fraction = (
        accounting["accepted_updates"] / fresh_total if fresh_total else 0.0
    )
    quantitative = {
        "belief_samples": int(belief_m.size),
        "belief_median_m": percentile(belief_m, 0.50),
        "belief_p95_m": percentile(belief_m, 0.95),
        "belief_rmse_m": (
            float(np.sqrt(np.mean(belief_m ** 2))) if belief_m.size else None
        ),
        "belief_max_m": float(np.max(belief_m)) if belief_m.size else None,
        "fused_corrections": int(fused_m.size),
        "fused_correction_median_m": percentile(fused_m, 0.50),
        "fused_correction_p95_m": percentile(fused_m, 0.95),
        "fused_correction_above_0_25_fraction": (
            float(np.mean(fused_m > 0.25)) if fused_m.size else None
        ),
        "accepted_fraction_of_fresh_corrections": accepted_fraction,
        "longest_accepted_correction_gap_s": accounting["longest_correction_gap_s"],
        "correction_accounting": accounting,
    }
    release_gate = resolved_preflight_gate(protocol_path, protocol)
    thresholds = release_gate["quantitative"]
    bootstrap_ok, bootstrap_evidence = bootstrap_verified(
        attempt / "manager_outcomes.jsonl"
    )
    request_id = summary.get("terminal_stop_request_id")
    request = summary.get("terminal_stop_request")
    acknowledgements = summary.get("terminal_stop_acknowledgements")
    exact_acknowledgements = bool(
        isinstance(request_id, str)
        and request_id
        and isinstance(request, dict)
        and request.get("request_id") == request_id
        and isinstance(acknowledgements, dict)
        and set(acknowledgements) == set(TERMINAL_ACK_STATUS_BY_COMPONENT)
        and all(
            isinstance(acknowledgements.get(component), dict)
            and acknowledgements[component].get("status") == status
            for component, status in TERMINAL_ACK_STATUS_BY_COMPONENT.items()
        )
    )
    manager_marker = final_journal_marker(attempt / "manager_outcomes.jsonl")
    detector_marker = final_journal_marker(attempt / "detector_outcomes.jsonl")
    checks = {
        "run_ledger_valid": ledger_validation_error is None,
        "bootstrap_prior_plus_camera": bootstrap_ok,
        "goal_reached": entry.get("outcome") == "goal_reached",
        "no_collision": not bool(summary.get("collision_any")),
        "attempt_evidence_complete": verdict.get("complete") is True,
        "summary_evidence_complete": summary.get("evidence_complete") is True,
        "correction_assimilation_verified": (
            verdict.get("correction_assimilation_verified") is True
        ),
        "manifest_local_use_obs_risk": manifest.get("local_use_obs_risk") is False,
        "manifest_no_diagnostic_odometry_localization": (
            manifest.get("use_diagnostic_odom_localization") is False
        ),
        "schema9_belief_identity": (
            int(manifest.get("logging_schema_version", 0)) >= 9
            and (run / "belief_predictions.jsonl").is_file()
            and belief.get("estimate_selection")
            == "highest_anchor_revision_per_state_timestamp"
        ),
        "manifest_heading_method": manifest.get("heading_update_mode") == "camera_xy_only",
        "manifest_no_reanchor": math.isclose(float(manifest.get("state_reanchor_m", math.nan)), 0.0),
        "manifest_config_hash": manifest.get("campaign_config_sha256") == sha256(config),
        "belief_samples_present": bool(belief_m.size),
        "fused_corrections_present": bool(fused_m.size),
        "physical_goal_distance": bool(
            summary.get("final_goal_distance_reference") == "ground_truth"
            and float(summary.get("final_goal_distance", math.inf)) <= 0.35
        ),
        "terminal_stop_verified": summary.get("terminal_stop_verified") is True,
        "producer_quiescence_acknowledged": (
            summary.get("producer_quiescence_acknowledged") is True
        ),
        "terminal_acknowledgement_identities_and_statuses": exact_acknowledgements,
        "manager_journal_terminal_request_identity": bool(
            manager_marker
            and manager_marker.get("status") == "session_stopped"
            and manager_marker.get("terminal_stop_request_id") == request_id
        ),
        "detector_journal_terminal_request_identity": bool(
            detector_marker
            and detector_marker.get("status") == "session_stopped"
            and detector_marker.get("terminal_stop_request_id") == request_id
        ),
        "belief_prediction_ledger_valid": (
            int(summary.get("belief_prediction_count", 0)) > 0
            and int(summary.get("belief_prediction_invalid_count", -1)) == 0
        ),
        "fused_correction_p95": bool(
            fused_m.size
            and quantitative["fused_correction_p95_m"]
            <= thresholds["fused_correction_p95_m_max"]
        ),
        "fused_correction_above_0_25_fraction": bool(
            fused_m.size
            and quantitative["fused_correction_above_0_25_fraction"]
            <= thresholds["fused_correction_above_0_25_fraction_max"]
        ),
        "belief_p95": bool(
            belief_m.size
            and quantitative["belief_p95_m"] <= thresholds["belief_p95_m_max"]
        ),
        "belief_max": bool(
            belief_m.size
            and quantitative["belief_max_m"] <= thresholds["belief_max_m_max"]
        ),
        "accepted_fraction": (
            accepted_fraction >= thresholds["accepted_fraction_of_fresh_corrections_min"]
        ),
        "longest_accepted_correction_gap": (
            accounting["longest_correction_gap_s"] is not None
            and accounting["longest_correction_gap_s"]
            <= thresholds["longest_accepted_correction_gap_s_max"]
        ),
    }
    report = {
        "schema": "thesis_stage09_runtime_preflight_report.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "release_final_campaign": all(checks.values()),
        "selection_or_fitting_performed": False,
        "checks": checks,
        "thresholds": thresholds,
        "release_gate": release_gate,
        "quantitative": quantitative,
        "ledger_validation_error": ledger_validation_error,
        "bootstrap_evidence": bootstrap_evidence,
        "terminal_identity": {
            "request_id": request_id,
            "acknowledgements": acknowledgements,
            "manager_final_marker": manager_marker,
            "detector_final_marker": detector_marker,
        },
        "terminal": {
            "campaign_outcome": entry.get("outcome"),
            "completion_reason": entry.get("completion_reason"),
            "summary_completion_reason": summary.get("completion_reason"),
            "collision_any": summary.get("collision_any"),
            "final_goal_distance_m": summary.get("final_goal_distance"),
            "final_goal_distance_reference": summary.get("final_goal_distance_reference"),
        },
        "sources": {
            str(protocol_path.relative_to(REPO)): sha256(protocol_path),
            str(config.relative_to(REPO)): sha256(config),
            str(campaign_log_path.relative_to(REPO)): sha256(campaign_log_path),
            str(verdict_path.relative_to(REPO)): sha256(verdict_path),
            str(manifest_path.relative_to(REPO)): sha256(manifest_path),
            str(summary_path.relative_to(REPO)): sha256(summary_path),
            str(ALIGNED_PATH.relative_to(REPO)): sha256(ALIGNED_PATH),
            str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__).resolve()),
        },
        "run": str(run.relative_to(REPO)),
        "analysis_interval": {"start_s": start, "stop_s": stop},
        "reference_max_gap_s": reference_gap,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite a preflight report")
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["release_final_campaign"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
