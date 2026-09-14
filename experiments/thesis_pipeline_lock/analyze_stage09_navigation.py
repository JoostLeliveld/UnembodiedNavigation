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


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "experiments/fusion_on_fixed_routes"),
    str(REPO / "src/unav_common"),
]
import aligned
from unav_common.occlusion_geometry import scene_from_json


REQUIRED_BASE = (
    "run_manifest.json", "run_summary.json", "experiment.csv",
    "fusion_observations.csv", "correction_assimilations.csv",
    "camera_opportunities.jsonl", "global_plan.csv", "global_waypoints.csv",
    "global_plan_meta.json", "preselected_route.json",
    "belief_predictions.jsonl",
)
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
COLORS = {"P0": "#657789", "P1": "#c57a2a"}
TASK_LABELS = {
    "thesis09_west_to_east_north": "West to east",
    "thesis09_east_to_west_south": "East to west",
    "thesis09_aisle_to_crossaisle": "Aisle to cross-aisle (null route)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def final_jsonl_record(path: Path) -> dict | None:
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
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


def expected_cells(protocol: dict) -> list[tuple[str, str, int]]:
    design = protocol["design"]
    return [
        (task, arm, seed)
        for task in design["tasks"]
        for arm in design["arms"]
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
    for task, arm, seed in expected:
        key = f"{task}__{arm}__seed{seed}"
        event = ledger[key]
        if not event.get("finished_at"):
            raise RuntimeError(f"campaign is incomplete: {key}")
        entry = {
            "key": key, "task": task, "arm": arm, "seed": seed,
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
            entry["attempt_files"] = {
                name: sha256(attempt / name)
                for name in REQUIRED_ATTEMPT if (attempt / name).is_file()
            }
            entry["attempt_missing"] = [
                name for name in REQUIRED_ATTEMPT if not (attempt / name).is_file()
            ]
        if run_value:
            run = Path(run_value).resolve()
            if run in seen_runs:
                raise RuntimeError("one run directory was selected for multiple cells")
            seen_runs.add(run)
            entry["run"] = str(run.relative_to(REPO))
            required = aligned.required_artifacts(run, REQUIRED_BASE)
            entry["files"] = {
                name: sha256(run / name) for name in required if (run / name).is_file()
            }
            entry["missing"] = [name for name in required if not (run / name).is_file()]
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
        "key": entry["key"], "task": entry["task"], "arm": entry["arm"],
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
    run, manifest, summary = aligned.verify_frozen_entry(
        entry, REQUIRED_BASE, minimum_schema=9, repo=REPO
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
        and float(summary.get("final_goal_distance", math.inf)) <= 0.35
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
    start, stop = aligned.mission_interval(run)
    table = aligned.rows(run)
    truth = aligned.truth_series(run, table, max_reference_gap_s=reference_gap)
    belief = aligned.aligned_error_cm(
        run, "belief", table, max_reference_gap_s=reference_gap
    )
    identity_mask = (
        aligned.latest_revision_mask(
            belief["stamp"], belief["revision"], belief["epoch"]
        )
        if belief.get("estimate_selection")
        == "highest_anchor_revision_per_state_timestamp"
        else aligned.landed_mask(belief["stamp"])
    )
    use = (
        identity_mask
        & belief["have"] & belief["reference_supported"]
        & (belief["stamp"] >= start) & (belief["stamp"] <= stop)
        & np.isfinite(belief["aligned_cm"])
    )
    errors_m = belief["aligned_cm"][use] / 100.0
    result.update(finite_summary(errors_m, "belief_error_m"))
    covariances = np.asarray(belief["covariance"], dtype=float)[use, :2, :2]
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
    estimate_yaw = np.asarray(belief["yaw"], dtype=float)[use]
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
        belief_estimate_selection=belief.get("estimate_selection"),
        duration_sim_s=stop - start,
        correction_ledger_valid=ledger.valid,
    )
    gt = (truth.t >= start) & (truth.t <= stop)
    curve = {
        "time": belief["stamp"][use] - start,
        "error_m": errors_m,
        "belief_x": belief["x"][use], "belief_y": belief["y"][use],
        "gt_x": truth.x[gt], "gt_y": truth.y[gt],
    }
    return result, curve


def paired_results(results: list[dict], protocol: dict) -> list[dict]:
    lookup = {(row["task"], row["arm"], row["seed"]): row for row in results}
    fields = (
        "strict_success", "collision", "duration_sim_s", "path_length_m",
        "belief_error_m_rmse", "belief_error_m_p95", "longest_correction_gap_s",
        "accepted_fraction_of_fresh", "fused_error_m_p95",
    )
    pairs = []
    for task in protocol["design"]["tasks"]:
        for seed in protocol["design"]["matched_seeds"]:
            p0, p1 = lookup[(task, "P0", seed)], lookup[(task, "P1", seed)]
            row = {"task": task, "seed": seed, "P0_outcome": p0["campaign_outcome"],
                   "P1_outcome": p1["campaign_outcome"]}
            for field in fields:
                a, b = p0.get(field), p1.get(field)
                row[f"P0_{field}"] = a
                row[f"P1_{field}"] = b
                row[f"difference_P1_minus_P0_{field}"] = (
                    float(b) - float(a) if a is not None and b is not None else None
                )
            pairs.append(row)
    return pairs


def group_summary(results: list[dict], protocol: dict) -> dict:
    groups = {}
    for task in protocol["design"]["tasks"]:
        groups[task] = {}
        for arm in protocol["design"]["arms"]:
            rows = [row for row in results if row["task"] == task and row["arm"] == arm]
            groups[task][arm] = {
                "runs": len(rows),
                "strict_successes": sum(bool(row["strict_success"]) for row in rows),
                "collisions": sum(bool(row.get("collision")) for row in rows),
                "outcomes": dict(Counter(row["campaign_outcome"] for row in rows)),
                "evidence_invalid": sum(not bool(row.get("evidence_valid")) for row in rows),
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


def paired_difference_summary(pairs: list[dict], protocol: dict) -> dict:
    """Retain individual matched-seed differences and only summarize those values."""
    fields = sorted({
        key.removeprefix("difference_P1_minus_P0_")
        for row in pairs for key in row
        if key.startswith("difference_P1_minus_P0_")
    })

    def summarize(rows: list[dict]) -> dict:
        result = {}
        for field in fields:
            key = f"difference_P1_minus_P0_{field}"
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

    primary = set(protocol["design"]["primary_route_choice_tasks"])
    null_task = protocol["design"]["null_route_choice_task"]
    return {
        "by_task": {
            task: summarize([row for row in pairs if row["task"] == task])
            for task in protocol["design"]["tasks"]
        },
        "by_predeclared_stratum": {
            "primary_route_choice": summarize(
                [row for row in pairs if row["task"] in primary]
            ),
            "implementation_null": summarize(
                [row for row in pairs if row["task"] == null_task]
            ),
        },
    }


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
    arms = ("P0", "P1")
    fig, axes = plt.subplots(len(tasks), 2, figsize=(9.4, 10.2), constrained_layout=True)
    for i, task in enumerate(tasks):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            map_background(ax, collision_json)
            route = np.asarray(json.loads(config["tasks"][task]["preselected_routes"][arm]["preselected_route_json"]))
            ax.plot(route[:, 0], route[:, 1], "--", color="#2e3740", lw=1.2, label="Frozen route")
            for seed in config["tasks"][task]["seeds"]:
                curve = curves.get((task, arm, seed))
                if curve is not None:
                    ax.plot(curve["gt_x"], curve["gt_y"], color=COLORS[arm], alpha=0.55, lw=0.8)
            ax.plot(route[0, 0], route[0, 1], "ko", ms=3)
            ax.plot(route[-1, 0], route[-1, 1], "k*", ms=7)
            score = sum(row["strict_success"] for row in results if row["task"] == task and row["arm"] == arm)
            ax.set_title(f"{TASK_LABELS[task]} — {arm}: {score}/5 strict successes")
            if j:
                ax.set_ylabel("")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "stage09_routes_and_trajectories")

    fig, axes = plt.subplots(len(tasks), 2, figsize=(9.4, 7.7), sharey=True, constrained_layout=True)
    for i, task in enumerate(tasks):
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            for seed in config["tasks"][task]["seeds"]:
                curve = curves.get((task, arm, seed))
                if curve is not None:
                    ax.plot(curve["time"], 100 * curve["error_m"], lw=0.85, alpha=0.72, label=str(seed))
            ax.axhline(25, color="#9e3d32", ls="--", lw=0.8, label="25 cm method gate")
            ax.set(title=f"{TASK_LABELS[task]} — {arm}", xlabel="Simulation time [s]", ylabel="Own-time belief error [cm]")
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
                a, b = row.get(f"P0_{field}"), row.get(f"P1_{field}")
                if a is None or b is None:
                    continue
                x0, x1 = task_i * 3, task_i * 3 + 1
                ax.plot([x0, x1], [scale*a, scale*b], color="#a8afb4", lw=0.8)
                ax.scatter([x0, x1], [scale*a, scale*b], c=[COLORS["P0"], COLORS["P1"]], s=18)
        ax.set_xticks([0, 1, 3, 4], ["P0", "P1", "P0", "P1"])
        ax.set_xlabel("East→west                 West→east")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)
    save_figure(fig, output, "stage09_paired_run_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    for task_i, task in enumerate(tasks):
        for arm_i, arm in enumerate(arms):
            rows = [r for r in results if r["task"] == task and r["arm"] == arm]
            x = task_i * 3 + arm_i
            fractions = [r["accepted_fraction_of_fresh"] for r in rows if r.get("accepted_fraction_of_fresh") is not None]
            gaps = [r["longest_correction_gap_s"] for r in rows if r.get("longest_correction_gap_s") is not None]
            axes[0].scatter(np.full(len(fractions), x), fractions, color=COLORS[arm], s=19)
            axes[1].scatter(np.full(len(gaps), x), gaps, color=COLORS[arm], s=19)
    ticks = [0, 1, 3, 4, 6, 7]
    labels = ["P0", "P1"] * 3
    for ax in axes:
        ax.set_xticks(ticks, labels)
        ax.set_xlabel("West→east        East→west          Null route")
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
        "paired_difference_summary": paired_difference_summary(pairs, protocol),
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
