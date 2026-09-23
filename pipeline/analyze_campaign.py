#!/usr/bin/env python3
"""Analyze the exact frozen Stage-09 navigation campaign and render thesis figures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis_stage09_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPO / "pipeline"),
    str(REPO / "src/unav_common"),
]
import aligned
from unav_common.occlusion_geometry import scene_from_json
from unav_common.rectangular_footprint import RectangularFootprint


REQUIRED_BASE = (
    "run_manifest.json", "run_summary.json", "experiment.csv",
    "fusion_observations.csv", "correction_assimilations.csv",
    "camera_opportunities.jsonl", "global_plan.csv", "global_waypoints.csv",
    "global_plan_meta.json", "belief_predictions.jsonl",
)
PRESELECTED_ROUTE_ARTIFACTS = ("preselected_route.json",)
REQUIRED_ATTEMPT = (
    "attempt_evidence_verdict.json",
    "manager_outcomes.jsonl",
    "detector_outcomes.jsonl",
)
TERMINAL_ACK_STATUS_BY_COMPONENT = {
    "planner": "stop_command_published",
    "camera_manager": "correction_stream_quiescent",
    "detector": "outcome_stream_quiescent",
}
COLORS = {
    "P0": "#657789", "P1": "#c57a2a",
    "global_intact": "#657789", "global_removal": "#9aa5af",
    "per_camera_intact": "#c57a2a", "per_camera_removal": "#dfad72",
    "spatial_intact": "#3f7f5f", "spatial_removal": "#86ae98",
}
TASK_LABELS = {
    "thesis09_parallel_aisles_west": "Western parallel aisles",
    "thesis09_parallel_aisles_central": "Central parallel aisles",
    "thesis09_west_to_east_north": "West to east",
    "thesis09_east_to_west_south": "East to west",
    "thesis09_aisle_to_crossaisle": "Aisle to cross-aisle (null route)",
    "thesis10_camera_a_western_dock_detour": "Camera A western dock detour",
    "thesis10_camera_b_cross_warehouse_detour": "Camera B cross-warehouse detour",
    "thesis10_camera_c_inner_warehouse_detour": "Camera C inner-warehouse detour",
    "thesis10_camera_e_eastern_detour": "Camera E eastern detour",
    "thesis10_camera_e_long_cross_warehouse_detour": "Camera E long cross-warehouse detour",
}
STRICT_GOAL_DISTANCE_M = 0.30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stored_artifact(path: Path) -> Path | None:
    """Resolve a campaign artifact, including its losslessly compressed form."""
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".zst")
    return compressed if compressed.is_file() else None


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def required_base_artifacts(run: Path) -> tuple[str, ...]:
    """Require route evidence appropriate to the planner mode used by the run."""
    manifest_path = run / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return REQUIRED_BASE
    if manifest.get("global_planner_mode") == "preselected_route":
        return REQUIRED_BASE + PRESELECTED_ROUTE_ARTIFACTS
    return REQUIRED_BASE


def final_jsonl_record(path: Path) -> dict | None:
    stored = stored_artifact(path)
    if stored is None:
        return None
    try:
        if stored.suffix == ".zst":
            text = subprocess.run(
                ["zstd", "-q", "-d", "-c", str(stored)],
                check=True, capture_output=True, text=True,
            ).stdout
        else:
            text = stored.read_text()
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError):
        return None
    return records[-1] if records and isinstance(records[-1], dict) else None


def terminal_identity_valid(summary: dict, attempt: Path) -> tuple[bool, dict]:
    request_id = summary.get("terminal_stop_request_id")
    request = summary.get("terminal_stop_request")
    acknowledgements = summary.get("terminal_stop_acknowledgements")
    manager_marker = final_jsonl_record(attempt / "manager_outcomes.jsonl")
    detector_marker = final_jsonl_record(attempt / "detector_outcomes.jsonl")
    checks = {
        "request_identity": bool(
            isinstance(request_id, str)
            and request_id
            and isinstance(request, dict)
            and request.get("request_id") == request_id
        ),
        "exact_acknowledgements": bool(
            isinstance(acknowledgements, dict)
            and set(acknowledgements) == set(TERMINAL_ACK_STATUS_BY_COMPONENT)
            and all(
                isinstance(acknowledgements.get(component), dict)
                and acknowledgements[component].get("status") == status
                for component, status in TERMINAL_ACK_STATUS_BY_COMPONENT.items()
            )
        ),
        "manager_final_marker": bool(
            manager_marker
            and manager_marker.get("status") == "session_stopped"
            and manager_marker.get("terminal_stop_request_id") == request_id
        ),
        "detector_final_marker": bool(
            detector_marker
            and detector_marker.get("status") == "session_stopped"
            and detector_marker.get("terminal_stop_request_id") == request_id
        ),
    }
    return all(checks.values()), checks


def condition_ids(protocol: dict) -> list[str]:
    """Return the experiment conditions in their predeclared order.

    ``arms`` remains accepted so sealed historical analyses can still be read,
    but new protocols use the less ambiguous ``conditions`` key.
    """
    design = protocol["design"]
    values = design.get("conditions", design.get("arms"))
    if not isinstance(values, list) or not values:
        raise RuntimeError("protocol design must define nonempty conditions")
    return [str(value) for value in values]


def primary_pair(protocol: dict) -> tuple[str, str]:
    comparison = protocol.get("comparisons", {}).get("primary_pair", {})
    baseline = comparison.get("baseline")
    treatment = comparison.get("treatment")
    if baseline and treatment:
        return str(baseline), str(treatment)
    conditions = condition_ids(protocol)
    if {"global_intact", "global_removal"}.issubset(conditions):
        return "global_intact", "global_removal"
    if {"P0", "P1"}.issubset(conditions):
        return "P0", "P1"
    raise RuntimeError("protocol does not declare a primary matched comparison")


def comparison_pairs(protocol: dict) -> list[tuple[str, str]]:
    conditions = set(condition_ids(protocol))
    canonical = [(f"{model}_intact", f"{model}_removal")
                 for model in ("global", "per_camera", "spatial")]
    selected = [pair for pair in canonical if set(pair).issubset(conditions)]
    return selected or [primary_pair(protocol)]


def expected_cells(protocol: dict) -> list[tuple[str, str, int]]:
    design = protocol["design"]
    return [
        (task, condition, seed)
        for task in design["tasks"]
        for condition in condition_ids(protocol)
        for seed in design["matched_seeds"]
    ]


def freeze_selection(protocol_path: Path, campaign_root: Path, output: Path) -> list[dict]:
    protocol = json.loads(protocol_path.read_text())
    config_path = (REPO / protocol["campaign_config_path"]).resolve()
    if sha256(config_path) != protocol["campaign_config_sha256"]:
        raise RuntimeError("frozen campaign config changed")
    expected = expected_cells(protocol)
    if len(expected) != int(protocol["design"]["expected_cells"]):
        raise RuntimeError("protocol cell count is internally inconsistent")
    campaign_log_path = campaign_root / "campaign_log.json"
    ledger = json.loads(campaign_log_path.read_text())
    expected_keys = {f"{task}__{arm}__seed{seed}" for task, arm, seed in expected}
    if set(ledger) != expected_keys:
        missing = sorted(expected_keys - set(ledger))
        extra = sorted(set(ledger) - expected_keys)
        raise RuntimeError(f"campaign ledger differs from protocol; missing={missing}, extra={extra}")
    entries = []
    seen_runs = set()
    for task, condition, seed in expected:
        key = f"{task}__{condition}__seed{seed}"
        event = ledger[key]
        if not event.get("finished_at"):
            raise RuntimeError(f"campaign is incomplete: {key}")
        entry = {
            "key": key, "task": task, "condition": condition,
            # Historical loaders and aligned result files used the name arm.
            "arm": condition, "seed": seed,
            "campaign_outcome": event.get("outcome"),
            "campaign_completion_reason": event.get("completion_reason"),
            "attempt_id": event.get("attempt_id"),
            "event": event,
        }
        run_value = str(event.get("run_dir", "") or "")
        attempt_value = str(event.get("run_log_dir", "") or "")
        if attempt_value:
            attempt = Path(attempt_value).resolve()
            entry["attempt"] = str(attempt.relative_to(REPO))
            entry["attempt_files"] = {}
            for name in REQUIRED_ATTEMPT:
                stored = stored_artifact(attempt / name)
                if stored is not None:
                    entry["attempt_files"][name] = sha256(stored)
            entry["attempt_missing"] = [
                name for name in REQUIRED_ATTEMPT
                if stored_artifact(attempt / name) is None
            ]
        if run_value:
            run = Path(run_value).resolve()
            if run in seen_runs:
                raise RuntimeError("one run directory was selected for multiple cells")
            seen_runs.add(run)
            entry["run"] = str(run.relative_to(REPO))
            required = aligned.required_artifacts(run, required_base_artifacts(run))
            entry["files"] = {}
            for name in required:
                stored = stored_artifact(run / name)
                if stored is not None:
                    entry["files"][name] = sha256(stored)
            entry["missing"] = [
                name for name in required if stored_artifact(run / name) is None
            ]
        entries.append(entry)
    selection = {
        "schema": "thesis_stage09_navigation_selection.v1",
        "status": "frozen_complete_campaign_selection",
        "protocol": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256(protocol_path),
        "config_sha256": sha256(config_path),
        "campaign_log": str(campaign_log_path.relative_to(REPO)),
        "campaign_log_sha256": sha256(campaign_log_path),
        "runs": entries,
    }
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise RuntimeError("frozen Stage-09 selection changed")
    else:
        write_json(selection_path, selection)
    return entries


def exact_config_checks(
    manifest: dict, config: dict, entry: dict, campaign_config_sha256: str
) -> None:
    condition = entry["condition"]
    if condition in config.get("conditions", {}):
        condition_cfg = config["conditions"][condition]
        task_override = (
            config.get("tasks", {}).get(entry["task"], {}).get(
                "condition_overrides", {}
            ).get(condition, {}) or {}
        )
        expected_runtime = [
            value.strip() for value in str(
                task_override.get(
                    "manager_camera_ids",
                    condition_cfg.get("manager_camera_ids", config["manager_camera_ids"]),
                )
            ).split(",") if value.strip()
        ]
        expected_planning = [
            value.strip() for value in str(
                task_override.get(
                    "camera_network_active_camera_ids",
                    condition_cfg["camera_network_active_camera_ids"],
                )
            ).split(",") if value.strip()
        ]
        if manifest.get("campaign_config_sha256") != campaign_config_sha256:
            raise RuntimeError(f"{entry['key']}: campaign config hash mismatch")
        if manifest.get("manager_camera_ids") not in (
            ",".join(expected_runtime), expected_runtime
        ):
            raise RuntimeError(f"{entry['key']}: runtime camera set mismatch")
        if manifest.get("camera_network_active_camera_ids") != expected_planning:
            raise RuntimeError(f"{entry['key']}: planning camera set mismatch")
        if manifest.get("manager_availability_model_path"):
            raise RuntimeError(f"{entry['key']}: active run unexpectedly used q")
        artifact = (REPO / condition_cfg["camera_network_artifact_path"]).resolve()
        if manifest.get("camera_network_artifact_sha256") != sha256(artifact):
            raise RuntimeError(f"{entry['key']}: planning-information artifact mismatch")
        return

    # Read-only validation for the superseded sealed P0/P1 campaign.
    expected = {
        "campaign_config_sha256": campaign_config_sha256,
        "heading_update_mode": "camera_xy_only",
        "state_reanchor_m": 0.0,
        "pixel_correction_nis_threshold": 9.21,
        "yolo_imgsz": 960,
        "yolo_model_sha256": "1e99afb5361a7c2599cb3f2863ebb04fd6f65bc1ee23996b989c9bba8b004f13",
        "manager_stage07_measurement_model_sha256": "249106c706b5e6dd3b10ab75abcfa54604c16b2fa987f38a5c84af58a2da9f4a",
        "manager_availability_model_sha256": "3dfa83ad0f9d8838bd6b5085e33c20fe89eb8c4334fa17d1d797bad69b1510e7",
        "manager_fusion_rule": "independent",
        "use_command_noise": True,
        "use_encoder_noise": True,
        "local_use_obs_risk": False,
        "use_diagnostic_odom_localization": False,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, abs_tol=1e-12, rel_tol=0):
                raise RuntimeError(f"{entry['key']}: manifest {key}={actual!r}, expected {value!r}")
        elif actual != value:
            raise RuntimeError(f"{entry['key']}: manifest {key}={actual!r}, expected {value!r}")
    route = config["tasks"][entry["task"]]["preselected_routes"][entry["arm"]]
    if manifest.get("preselected_route_sha256") != route["preselected_route_sha256"]:
        raise RuntimeError(f"{entry['key']}: wrong frozen route")


def finite_summary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"{prefix}_n": 0, f"{prefix}_median": None, f"{prefix}_p95": None,
                f"{prefix}_rmse": None, f"{prefix}_max": None}
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_max": float(np.max(values)),
    }


def analyze_run(
    entry: dict, config: dict, reference_gap: float, campaign_config_sha256: str
):
    base = {
        "key": entry["key"], "task": entry["task"],
        "condition": entry["condition"], "arm": entry["arm"],
        "seed": entry["seed"], "campaign_outcome": entry["campaign_outcome"],
        "campaign_completion_reason": entry["campaign_completion_reason"],
    }
    if ("run" not in entry or entry.get("missing") or "attempt" not in entry
            or entry.get("attempt_missing")):
        return {**base, "evidence_valid": False, "strict_success": False,
                "analysis_status": "infrastructure_invalid",
                "reason": {
                    "run_missing": entry.get("missing", []),
                    "attempt_missing": entry.get("attempt_missing", []),
                }}, None
    selected_run = REPO / entry["run"]
    run, manifest, summary = aligned.verify_frozen_entry(
        entry, required_base_artifacts(selected_run), minimum_schema=9, repo=REPO
    )
    exact_config_checks(manifest, config, entry, campaign_config_sha256)
    attempt = REPO / entry["attempt"]
    verdict = json.loads((attempt / "attempt_evidence_verdict.json").read_text())
    terminal_identity_ok, terminal_identity_checks = terminal_identity_valid(summary, attempt)
    evidence_valid = bool(
        verdict.get("complete") is True
        and summary.get("evidence_complete") is True
        and summary.get("data_files_closed") is True
        and summary.get("valid_run") is True
        and terminal_identity_ok
    )
    strict_success = bool(
        entry["campaign_outcome"] == "goal_reached"
        and evidence_valid
        and summary.get("completed") is True
        and not summary.get("collision_any")
        and summary.get("final_goal_distance_reference") == "ground_truth"
        and float(summary.get("final_goal_distance", math.inf)) <= STRICT_GOAL_DISTANCE_M
        and summary.get("terminal_stop_verified") is True
        and summary.get("producer_quiescence_acknowledged") is True
    )
    result = {
        **base, "run": entry["run"], "evidence_valid": evidence_valid,
        "evidence_reason": verdict.get("completion_reason"),
        "terminal_identity_valid": terminal_identity_ok,
        "terminal_identity_checks": terminal_identity_checks,
        "strict_success": strict_success,
        "collision": bool(summary.get("collision_any")),
        "collision_reason": summary.get("collision_reason"),
        "summary_completion_reason": summary.get("completion_reason"),
        "final_goal_distance_m": summary.get("final_goal_distance"),
        "final_goal_distance_reference": summary.get("final_goal_distance_reference"),
        "path_length_m": summary.get("path_length_m"),
    }
    if not evidence_valid:
        result.update(analysis_status="evidence_invalid")
        return result, None
    ledger = aligned.validate_run_ledger(run)
    plan_meta = json.loads((run / "global_plan_meta.json").read_text())
    result.update(
        selected_route_source=plan_meta.get("selected_source"),
        planned_total_cost=plan_meta.get("total_cost"),
        planned_risk_cost=plan_meta.get("risk_cost"),
        planned_ambiguity_cost=plan_meta.get("ambiguity_cost"),
        planned_obstacle_cost=plan_meta.get("obstacle_cost"),
        planned_rollout_valid=plan_meta.get("rollout_valid"),
        planned_terminal_goal_distance_m=plan_meta.get("terminal_goal_distance_pred"),
    )
    start, stop = aligned.mission_interval(run)
    table = aligned.rows(run)
    truth = aligned.truth_series(run, table, max_reference_gap_s=reference_gap)
    belief = aligned.aligned_error_cm(
        run, "belief", table, max_reference_gap_s=reference_gap
    )
    identity_mask = aligned.landed_mask(belief["stamp"])
    use = (
        identity_mask
        & belief["have"] & belief["reference_supported"]
        & (belief["stamp"] >= start) & (belief["stamp"] <= stop)
        & np.isfinite(belief["aligned_cm"])
    )
    errors_m = belief["aligned_cm"][use] / 100.0
    result.update(finite_summary(errors_m, "belief_error_m"))
    covariance_rows = np.asarray([
        [[f(row, "planner_cov_x"), f(row, "planner_cov_xy")],
         [f(row, "planner_cov_xy"), f(row, "planner_cov_y")]]
        for row in table
    ], dtype=float)
    covariances = covariance_rows[use]
    residuals = np.column_stack([
        belief["x"][use] - belief["gt_x"][use],
        belief["y"][use] - belief["gt_y"][use],
    ])
    nees = aligned.nees(residuals, covariances)
    result.update(
        planar_nees_median=float(np.median(nees)) if nees.size else None,
        planar_95_ellipse_containment=float(np.mean(nees <= aligned.CHI2_95_2D)) if nees.size else None,
    )
    truth_yaw = truth.yaw_at(belief["stamp"][use])
    estimate_yaw = np.asarray(
        [f(row, "planner_belief_yaw") for row in table], dtype=float
    )[use]
    yaw = np.abs(np.arctan2(np.sin(estimate_yaw - truth_yaw), np.cos(estimate_yaw - truth_yaw)))
    result.update(finite_summary(np.rad2deg(yaw), "belief_yaw_error_deg"))
    fused = [row for row in aligned.fused_answers(run, max_reference_gap_s=reference_gap)
             if start <= row["fused_stamp"] <= stop]
    fused_errors_m = np.asarray([row["error_cm"] / 100.0 for row in fused])
    result.update(finite_summary(fused_errors_m, "fused_error_m"))
    result["fused_above_0_25_fraction"] = (
        float(np.mean(fused_errors_m > 0.25)) if fused_errors_m.size else None
    )
    accounting = aligned.correction_accounting(run)
    result.update(
        correction_published=accounting["mission_outcomes"],
        correction_accepted=accounting["accepted_updates"],
        correction_rejected=accounting["rejected"],
        correction_dropped=accounting["dropped"],
        correction_dropped_fraction=accounting["correction_dropped_fraction"],
        longest_correction_gap_s=accounting["longest_correction_gap_s"],
        accepted_fraction_of_fresh=(
            accounting["accepted_updates"] /
            (accounting["accepted_updates"] + accounting["rejected"])
            if accounting["accepted_updates"] + accounting["rejected"] else None
        ),
    )
    opportunities, duplicates = aligned.camera_opportunities(run)
    opportunities = [row for row in opportunities if start <= float(row["timestamp_s"]) <= stop]
    result.update(
        camera_opportunities=len(opportunities),
        detector_returns=sum(bool(row["detection_valid"]) for row in opportunities),
        detector_return_fraction=(
            sum(bool(row["detection_valid"]) for row in opportunities) / len(opportunities)
            if opportunities else None
        ),
        duplicate_opportunity_deliveries=duplicates,
        analysis_status="scored",
        reference_max_gap_s=reference_gap,
        reference_source=truth.source,
        belief_estimate_selection="first_row_per_distinct_belief_timestamp",
        duration_sim_s=stop - start,
        correction_ledger_valid=ledger.valid,
    )
    gt = (truth.t >= start) & (truth.t <= stop)
    collision_scene = scene_from_json(manifest["collision_geometry_json"])
    footprint = RectangularFootprint(
        collision_scene.prisms,
        float(manifest["robot_length_m"]),
        float(manifest["robot_width_m"]),
    )
    clearances = np.asarray([
        footprint.clearance((x, y, yaw))
        for x, y, yaw in zip(truth.x[gt], truth.y[gt], truth.yaw[gt], strict=True)
    ], dtype=float)
    result["minimum_body_clearance_m"] = (
        float(np.min(clearances)) if clearances.size else None
    )
    curve = {
        "time": belief["stamp"][use] - start,
        "error_m": errors_m,
        "belief_x": belief["x"][use], "belief_y": belief["y"][use],
        "gt_x": truth.x[gt], "gt_y": truth.y[gt],
    }
    with (run / "global_plan.csv").open(newline="", encoding="utf-8") as handle:
        planned = list(csv.DictReader(handle))
    curve["plan_x"] = np.asarray([f(row, "x") for row in planned], dtype=float)
    curve["plan_y"] = np.asarray([f(row, "y") for row in planned], dtype=float)
    return result, curve


def paired_results(results: list[dict], protocol: dict) -> list[dict]:
    lookup = {(row["task"], row["condition"], row["seed"]): row for row in results}
    fields = (
        "strict_success", "collision", "duration_sim_s", "path_length_m",
        "belief_error_m_rmse", "belief_error_m_p95", "longest_correction_gap_s",
        "accepted_fraction_of_fresh", "fused_error_m_p95", "minimum_body_clearance_m",
    )
    pairs = []
    for baseline, treatment in comparison_pairs(protocol):
        for task in protocol["design"]["tasks"]:
            for seed in protocol["design"]["matched_seeds"]:
                a = lookup[(task, baseline, seed)]
                b = lookup[(task, treatment, seed)]
                row = {
                    "task": task, "seed": seed,
                    "baseline_condition": baseline, "treatment_condition": treatment,
                    f"{baseline}_outcome": a["campaign_outcome"],
                    f"{treatment}_outcome": b["campaign_outcome"],
                    f"{baseline}_selected_route_source": a.get("selected_route_source"),
                    f"{treatment}_selected_route_source": b.get("selected_route_source"),
                    "route_changed": (
                        a.get("selected_route_source") != b.get("selected_route_source")
                        if a.get("selected_route_source") is not None
                        and b.get("selected_route_source") is not None else None
                    ),
                }
                for field in fields:
                    value_a, value_b = a.get(field), b.get(field)
                    row[f"{baseline}_{field}"] = value_a
                    row[f"{treatment}_{field}"] = value_b
                    row[f"difference_{treatment}_minus_{baseline}_{field}"] = (
                        float(value_b) - float(value_a)
                        if value_a is not None and value_b is not None else None
                    )
                pairs.append(row)
    return pairs


def group_summary(results: list[dict], protocol: dict) -> dict:
    groups = {}
    for task in protocol["design"]["tasks"]:
        groups[task] = {}
        for condition in condition_ids(protocol):
            rows = [
                row for row in results
                if row["task"] == task and row["condition"] == condition
            ]
            groups[task][condition] = {
                "runs": len(rows),
                "strict_successes": sum(bool(row["strict_success"]) for row in rows),
                "collisions": sum(bool(row.get("collision")) for row in rows),
                "outcomes": dict(Counter(row["campaign_outcome"] for row in rows)),
                "evidence_invalid": sum(not bool(row.get("evidence_valid")) for row in rows),
                "selected_routes": dict(Counter(
                    row.get("selected_route_source") for row in rows
                    if row.get("selected_route_source") is not None
                )),
                "median_run_belief_rmse_m": (
                    float(np.median([row["belief_error_m_rmse"] for row in rows
                                     if row.get("belief_error_m_rmse") is not None]))
                    if any(row.get("belief_error_m_rmse") is not None for row in rows) else None
                ),
                "median_run_longest_correction_gap_s": (
                    float(np.median([row["longest_correction_gap_s"] for row in rows
                                     if row.get("longest_correction_gap_s") is not None]))
                    if any(row.get("longest_correction_gap_s") is not None for row in rows) else None
                ),
            }
    return groups


def condition_summary(results: list[dict], protocol: dict) -> dict:
    """Aggregate the two tasks without treating time samples as replicates."""
    output = {}
    metric_fields = (
        "belief_error_m_rmse", "planar_nees_median",
        "planar_95_ellipse_containment", "path_length_m", "duration_sim_s",
        "longest_correction_gap_s", "minimum_body_clearance_m",
    )
    for condition in condition_ids(protocol):
        rows = [row for row in results if row["condition"] == condition]
        item = {
            "runs": len(rows),
            "strict_successes": sum(bool(row.get("strict_success")) for row in rows),
            "collisions": sum(bool(row.get("collision")) for row in rows),
            "evidence_invalid": sum(not bool(row.get("evidence_valid")) for row in rows),
            "outcomes": dict(Counter(row.get("campaign_outcome") for row in rows)),
            "selected_routes": dict(Counter(
                row.get("selected_route_source") for row in rows
                if row.get("selected_route_source") is not None
            )),
        }
        for field in metric_fields:
            values = np.asarray(
                [row[field] for row in rows if row.get(field) is not None], dtype=float
            )
            item[f"median_run_{field}"] = (
                float(np.median(values)) if values.size else None
            )
        output[condition] = item
    return output


def paired_difference_summary(pairs: list[dict], protocol: dict) -> dict:
    """Retain individual matched-seed differences and only summarize those values."""
    def summarize(rows: list[dict], baseline: str, treatment: str) -> dict:
        prefix = f"difference_{treatment}_minus_{baseline}_"
        fields = sorted({key.removeprefix(prefix) for row in rows for key in row
                         if key.startswith(prefix)})
        result = {}
        for field in fields:
            key = f"{prefix}{field}"
            individual = [
                {"task": row["task"], "seed": row["seed"], "difference": row.get(key)}
                for row in rows
            ]
            values = np.asarray(
                [item["difference"] for item in individual if item["difference"] is not None],
                dtype=float,
            )
            result[field] = {
                "planned_pairs": len(rows),
                "available_pairs": int(values.size),
                "missing_pairs": len(rows) - int(values.size),
                "individual": individual,
                "median": float(np.median(values)) if values.size else None,
                "min": float(np.min(values)) if values.size else None,
                "max": float(np.max(values)) if values.size else None,
            }
        return result

    output = {}
    for baseline, treatment in comparison_pairs(protocol):
        rows = [row for row in pairs if row["baseline_condition"] == baseline
                and row["treatment_condition"] == treatment]
        output[f"{treatment}_minus_{baseline}"] = {
            "by_task": {
                task: summarize([row for row in rows if row["task"] == task],
                                baseline, treatment)
                for task in protocol["design"]["tasks"]
            },
            "overall": summarize(rows, baseline, treatment),
        }
    return output


def route_change_summary(pairs: list[dict]) -> dict:
    output = {}
    for baseline, treatment in sorted({
            (row["baseline_condition"], row["treatment_condition"])
            for row in pairs}):
        rows = [row for row in pairs if row["baseline_condition"] == baseline
                and row["treatment_condition"] == treatment]
        available = [row for row in rows if row.get("route_changed") is not None]
        output[f"{treatment}_minus_{baseline}"] = {
            "planned_pairs": len(rows),
            "available_pairs": len(available),
            "changed_pairs": sum(bool(row["route_changed"]) for row in available),
            "by_task": {
                task: sum(bool(row["route_changed"]) for row in available
                          if row["task"] == task)
                for task in sorted({row["task"] for row in rows})
            },
        }
    return output


def map_background(ax, collision_json: str) -> None:
    for prism in scene_from_json(collision_json).prisms:
        if prism.zmin <= 0.55 and prism.zmax >= 0.0:
            ax.add_patch(Rectangle(
                (prism.xmin, prism.ymin), prism.xmax - prism.xmin,
                prism.ymax - prism.ymin, facecolor="#d7dcdf", edgecolor="#aab2b7",
                linewidth=0.25, zorder=0,
            ))
    ax.set(xlim=(-12.3, 12.3), ylim=(-10.3, 10.3), aspect="equal",
           xlabel="World x [m]", ylabel="World y [m]")
    ax.grid(alpha=0.12)


def save_figure(fig, output: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_figures(results, curves, pairs, config, protocol, output):
    plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    first = next(row for row in results if row.get("run"))
    collision_json = json.loads((REPO / first["run"] / "run_manifest.json").read_text())["collision_geometry_json"]
    tasks = list(config["tasks"])
    conditions = condition_ids(protocol)
    baseline, treatment = primary_pair(protocol)
    fig, axes = plt.subplots(
        len(tasks), len(conditions),
        figsize=(4.3 * len(conditions), 3.2 * len(tasks)),
        squeeze=False, constrained_layout=True,
    )
    for i, task in enumerate(tasks):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]
            map_background(ax, collision_json)
            route_cfg = config["tasks"][task].get("preselected_routes", {}).get(condition)
            if route_cfg:
                route = np.asarray(json.loads(route_cfg["preselected_route_json"]))
                ax.plot(route[:, 0], route[:, 1], "--", color="#2e3740", lw=1.2,
                        label="Frozen route")
            for seed in config["tasks"][task]["seeds"]:
                curve = curves.get((task, condition, seed))
                if curve is not None:
                    ax.plot(curve["plan_x"], curve["plan_y"], "--",
                            color="#2e3740", alpha=0.30, lw=0.65)
                    ax.plot(curve["gt_x"], curve["gt_y"],
                            color=COLORS.get(condition, "#657789"), alpha=0.55, lw=0.8)
            score = sum(
                row["strict_success"] for row in results
                if row["task"] == task and row["condition"] == condition
            )
            planned = sum(
                1 for row in results
                if row["task"] == task and row["condition"] == condition
            )
            ax.set_title(
                f"{TASK_LABELS.get(task, task)} — {condition}: "
                f"{score}/{planned} strict successes"
            )
            if j:
                ax.set_ylabel("")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "stage09_routes_and_trajectories")

    fig, axes = plt.subplots(
        len(tasks), len(conditions),
        figsize=(4.3 * len(conditions), 2.5 * len(tasks)),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    for i, task in enumerate(tasks):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]
            for seed in config["tasks"][task]["seeds"]:
                curve = curves.get((task, condition, seed))
                if curve is not None:
                    ax.plot(curve["time"], 100 * curve["error_m"], lw=0.85, alpha=0.72, label=str(seed))
            ax.axhline(25, color="#9e3d32", ls="--", lw=0.8, label="25 cm method gate")
            ax.set(title=f"{TASK_LABELS.get(task, task)} — {condition}",
                   xlabel="Simulation time [s]", ylabel="Own-time belief error [cm]")
            ax.grid(alpha=0.18)
            if j:
                ax.set_ylabel("")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7)
    save_figure(fig, output, "stage09_belief_error")

    primary = set(protocol["design"]["primary_route_choice_tasks"])
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), constrained_layout=True)
    metrics = (("belief_error_m_rmse", "Belief RMSE [cm]", 100.0),
               ("longest_correction_gap_s", "Longest correction gap [s]", 1.0),
               ("duration_sim_s", "Duration [s]", 1.0))
    for ax, (field, label, scale) in zip(axes, metrics):
        for task_i, task in enumerate(sorted(primary)):
            selected = [row for row in pairs if row["task"] == task]
            for k, row in enumerate(selected):
                a, b = row.get(f"{baseline}_{field}"), row.get(f"{treatment}_{field}")
                if a is None or b is None:
                    continue
                x0, x1 = task_i * 3, task_i * 3 + 1
                ax.plot([x0, x1], [scale*a, scale*b], color="#a8afb4", lw=0.8)
                ax.scatter([x0, x1], [scale*a, scale*b],
                           c=[COLORS.get(baseline), COLORS.get(treatment)], s=18)
        ax.set_xticks([0, 1, 3, 4], [baseline, treatment, baseline, treatment])
        ax.set_xlabel("Central task                 Western task")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)
    save_figure(fig, output, "stage09_paired_run_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    for task_i, task in enumerate(tasks):
        for condition_i, condition in enumerate(conditions):
            rows = [
                r for r in results
                if r["task"] == task and r["condition"] == condition
            ]
            x = task_i * (len(conditions) + 1) + condition_i
            fractions = [r["accepted_fraction_of_fresh"] for r in rows if r.get("accepted_fraction_of_fresh") is not None]
            gaps = [r["longest_correction_gap_s"] for r in rows if r.get("longest_correction_gap_s") is not None]
            axes[0].scatter(np.full(len(fractions), x), fractions,
                            color=COLORS.get(condition, "#657789"), s=19)
            axes[1].scatter(np.full(len(gaps), x), gaps,
                            color=COLORS.get(condition, "#657789"), s=19)
    ticks = [
        task_i * (len(conditions) + 1) + condition_i
        for task_i in range(len(tasks)) for condition_i in range(len(conditions))
    ]
    labels = conditions * len(tasks)
    for ax in axes:
        ax.set_xticks(ticks, labels)
        ax.set_xlabel("Western task                         Central task")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Accepted fraction of fresh corrections")
    axes[1].set_ylabel("Longest accepted-correction gap [s]")
    save_figure(fig, output, "stage09_correction_integrity")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if not str(protocol.get("status", "")).startswith("frozen_before_final_campaign_"):
        raise RuntimeError("analysis protocol is not frozen for a final campaign")
    execution_protocol = (REPO / protocol["execution_protocol_path"]).resolve()
    if sha256(execution_protocol) != protocol["execution_protocol_sha256"]:
        raise RuntimeError("frozen execution protocol changed")
    implementation = protocol["analysis_implementation"]
    if sha256(Path(__file__).resolve()) != implementation["sha256"]:
        raise RuntimeError("frozen Stage-09 analyzer changed")
    if sha256(Path(aligned.__file__).resolve()) != implementation["aligned_loader_sha256"]:
        raise RuntimeError("frozen aligned-metrics loader changed")
    campaign_root = (REPO / protocol["campaign_root"]).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = (REPO / protocol["campaign_config_path"]).resolve()
    config = yaml.safe_load(config_path.read_text())
    entries = freeze_selection(protocol_path, campaign_root, output)
    analysis_identity = {
        "schema": "thesis_stage09_navigation_analysis_identity.v1",
        "protocol_sha256": sha256(protocol_path),
        "selection_sha256": sha256(output / "selection.json"),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "aligned_loader_sha256": sha256(Path(aligned.__file__)),
    }
    identity_path = output / "analysis_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != analysis_identity:
        raise RuntimeError("analysis identity changed; use a new output directory")
    if not identity_path.exists():
        write_json(identity_path, analysis_identity)
    results, curves = [], {}
    reference_gap = float(protocol["time_and_reference"]["max_reference_gap_s"])
    for entry in entries:
        result, curve = analyze_run(
            entry, config, reference_gap, protocol["campaign_config_sha256"]
        )
        results.append(result)
        curves[(entry["task"], entry["arm"], entry["seed"])] = curve
    pairs = paired_results(results, protocol)
    summary = {
        "schema": "thesis_stage09_navigation_summary.v1",
        "analysis_identity_sha256": sha256(identity_path),
        "group_summary": group_summary(results, protocol),
        "condition_summary": condition_summary(results, protocol),
        "paired_difference_summary": paired_difference_summary(pairs, protocol),
        "route_change_summary": route_change_summary(pairs),
        "campaign_accounting": {
            "planned_runs": len(expected_cells(protocol)),
            "selected_runs": len(results),
            "evidence_valid_runs": sum(bool(row.get("evidence_valid")) for row in results),
            "strict_successes": sum(bool(row.get("strict_success")) for row in results),
            "collisions": sum(bool(row.get("collision")) for row in results),
            "campaign_outcomes": dict(Counter(row["campaign_outcome"] for row in results)),
        },
        "primary_route_choice_tasks": protocol["design"]["primary_route_choice_tasks"],
        "null_route_choice_task": protocol["design"]["null_route_choice_task"],
        "commissioning_refused_task": protocol["design"]["commissioning_refused_task"],
        "interpretation_scope": protocol["comparisons"]["scope"],
    }
    owned = ["run_results.json", "run_results.csv", "paired_results.json",
             "paired_results.csv", "summary.json"]
    if any((output / name).exists() for name in owned):
        raise FileExistsError("refusing to overwrite Stage-09 analysis outputs")
    write_json(output / "run_results.json", results)
    write_csv(output / "run_results.csv", results)
    write_json(output / "paired_results.json", pairs)
    write_csv(output / "paired_results.csv", pairs)
    write_json(output / "summary.json", summary)
    render_figures(results, curves, pairs, config, protocol, output)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
