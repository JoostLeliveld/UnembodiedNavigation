#!/usr/bin/env python3
"""Audit and summarize the exact 4 x 2 final navigation factorial campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/final_factorial_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "experiments/fusion_on_fixed_routes"),
    str(REPO / "src/unav_common"),
]
import aligned


ARMS = ("C00", "C01", "C10", "C11")
TASKS = (
    "thesis09_west_to_east_north",
    "thesis09_east_to_west_south",
    "thesis09_aisle_to_crossaisle",
    "thesis09_south_to_north_central",
)
TASK_LABELS = {
    "thesis09_west_to_east_north": "West to east",
    "thesis09_east_to_west_south": "East to west",
    "thesis09_aisle_to_crossaisle": "Aisle to cross-aisle",
    "thesis09_south_to_north_central": "South to north",
}
COLORS = {"C00": "#66788a", "C01": "#c27a32", "C10": "#278277", "C11": "#875f9f"}
REQUIRED_BASE = (
    "run_manifest.json", "run_summary.json", "experiment.csv",
    "fusion_observations.csv", "correction_assimilations.csv",
    "camera_opportunities.jsonl", "global_plan.csv", "global_waypoints.csv",
    "global_plan_meta.json", "preselected_route.json",
    "belief_predictions.jsonl",
)
REQUIRED_ATTEMPT = (
    "attempt_evidence_verdict.json", "manager_outcomes.jsonl", "detector_outcomes.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_summary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"{prefix}_n": 0, f"{prefix}_median": None,
                f"{prefix}_p95": None, f"{prefix}_rmse": None,
                f"{prefix}_max": None}
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_max": float(np.max(values)),
    }


def expected_cells(config: dict) -> list[tuple[str, str, int]]:
    return [
        (task, arm, int(seed))
        for task in TASKS
        for arm in ARMS
        for seed in config["tasks"][task]["seeds"]
    ]


def freeze_selection(config_path: Path, campaign_root: Path, output: Path) -> list[dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = expected_cells(config)
    if len(expected) != 80:
        raise RuntimeError(f"expected 80 cells, found {len(expected)}")
    ledger_path = campaign_root / "campaign_log.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    expected_keys = {f"{task}__{arm}__seed{seed}" for task, arm, seed in expected}
    if set(ledger) != expected_keys:
        raise RuntimeError(
            f"campaign ledger mismatch: missing={sorted(expected_keys-set(ledger))}, "
            f"extra={sorted(set(ledger)-expected_keys)}"
        )
    entries, seen = [], set()
    for task, arm, seed in expected:
        key = f"{task}__{arm}__seed{seed}"
        event = ledger[key]
        if not event.get("finished_at") or event.get("outcome") is None:
            raise RuntimeError(f"campaign is incomplete: {key}")
        entry = {
            "key": key, "task": task, "arm": arm, "seed": seed,
            "campaign_outcome": event.get("outcome"),
            "campaign_completion_reason": event.get("completion_reason"),
            "attempt_id": event.get("attempt_id"), "event": event,
        }
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
        run_value = str(event.get("run_dir", "") or "")
        if run_value:
            run = Path(run_value).resolve()
            if run in seen:
                raise RuntimeError("one run directory was selected for multiple cells")
            seen.add(run)
            entry["run"] = str(run.relative_to(REPO))
            required = aligned.required_artifacts(run, REQUIRED_BASE)
            entry["files"] = {
                name: sha256(run / name) for name in required if (run / name).is_file()
            }
            entry["missing"] = [name for name in required if not (run / name).is_file()]
        entries.append(entry)
    selection = {
        "schema": "final_factorial_navigation_selection.v1",
        "status": "frozen_complete_campaign_selection",
        "campaign_config": str(config_path.relative_to(REPO)),
        "campaign_config_sha256": sha256(config_path),
        "campaign_log": str(ledger_path.relative_to(REPO)),
        "campaign_log_sha256": sha256(ledger_path),
        "runs": entries,
    }
    write_json(output / "selection.json", selection)
    return entries


def exact_config_checks(manifest: dict, config: dict, entry: dict, config_hash: str) -> None:
    expected = {
        "campaign_config_sha256": config_hash,
        "heading_update_mode": "camera_xy_only",
        "state_reanchor_m": 0.0,
        "pixel_correction_nis_threshold": 9.21,
        "yolo_imgsz": 960,
        "yolo_model_sha256": "1e99afb5361a7c2599cb3f2863ebb04fd6f65bc1ee23996b989c9bba8b004f13",
        "manager_visibility_sensor_model_sha256": config["manager_visibility_sensor_model_expected_sha256"],
        "manager_availability_model_expected_sha256": config["manager_availability_model_expected_sha256"],
        "manager_fusion_rule": "independent",
        "manager_decision_rate_hz": 5.0,
        "camera_network_updates_per_step": 1,
        "optimizer_control_block_steps": 2,
        "use_command_noise": True,
        "use_encoder_noise": True,
        "local_controller_type": "ff_fb",
        "global_planner_mode": "preselected_route",
        "v_max": 1.0,
        "use_diagnostic_odom_localization": False,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if isinstance(value, float):
            good = actual is not None and math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12)
        else:
            good = actual == value
        if not good:
            raise RuntimeError(f"{entry['key']}: {key}={actual!r}, expected {value!r}")
    route = config["tasks"][entry["task"]]["preselected_routes"][entry["arm"]]
    if manifest.get("preselected_route_sha256") != route["preselected_route_sha256"]:
        raise RuntimeError(f"{entry['key']}: wrong frozen route")


def analyze_run(entry: dict, config: dict, reference_gap: float, config_hash: str):
    base = {
        "key": entry["key"], "task": entry["task"], "arm": entry["arm"],
        "seed": entry["seed"], "campaign_outcome": entry["campaign_outcome"],
        "campaign_completion_reason": entry["campaign_completion_reason"],
    }
    if ("run" not in entry or entry.get("missing") or "attempt" not in entry
            or entry.get("attempt_missing")):
        return {**base, "evidence_valid": False, "strict_success": False,
                "analysis_status": "infrastructure_invalid"}, None
    run, manifest, summary = aligned.verify_frozen_entry(
        entry, REQUIRED_BASE, minimum_schema=9, repo=REPO
    )
    exact_config_checks(manifest, config, entry, config_hash)
    verdict = json.loads(
        (REPO / entry["attempt"] / "attempt_evidence_verdict.json").read_text(encoding="utf-8")
    )
    evidence_valid = bool(
        verdict.get("complete") is True
        and entry["event"].get("attempt_evidence_complete") is True
        and entry["event"].get("route_artifact_verified") is True
        and entry["event"].get("correction_assimilation_verified") is True
        and entry["event"].get("detector_journal_verified") is True
        and entry["event"].get("manager_journal_verified") is True
        and summary.get("evidence_complete") is True
        and summary.get("data_files_closed") is True
        and summary.get("valid_run") is True
        and summary.get("terminal_stop_verified") is True
        and summary.get("producer_quiescence_acknowledged") is True
    )
    final_distance = summary.get("final_goal_distance")
    strict_success = bool(
        entry["campaign_outcome"] == "goal_reached" and evidence_valid
        and summary.get("completed") is True and not summary.get("collision_any")
        and summary.get("final_goal_distance_reference") == "ground_truth"
        and final_distance is not None and float(final_distance) <= 0.35
    )
    result = {
        **base, "run": entry["run"], "evidence_valid": evidence_valid,
        "strict_success": strict_success, "collision": bool(summary.get("collision_any")),
        "collision_reason": summary.get("collision_reason", ""),
        "final_goal_distance_m": final_distance,
        "final_goal_distance_reference": summary.get("final_goal_distance_reference"),
        "path_length_m": summary.get("path_length_m"),
    }
    if not evidence_valid:
        result["analysis_status"] = "evidence_invalid"
        return result, None
    ledger = aligned.validate_run_ledger(run)
    start, stop = aligned.mission_interval(run)
    table = aligned.rows(run)
    truth = aligned.truth_series(run, table, max_reference_gap_s=reference_gap)
    belief = aligned.aligned_error_cm(
        run, "belief", table, max_reference_gap_s=reference_gap
    )
    identity = aligned.latest_revision_mask(
        belief["stamp"], belief["revision"], belief["epoch"]
    )
    use = (
        identity & belief["have"] & belief["reference_supported"]
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
        planar_nees_mean=float(np.mean(nees)) if nees.size else None,
        planar_95_ellipse_containment=float(np.mean(nees <= aligned.CHI2_95_2D)) if nees.size else None,
    )
    accounting = aligned.correction_accounting(run)
    result.update(
        correction_published=accounting["mission_outcomes"],
        correction_accepted=accounting["accepted_updates"],
        correction_rejected=accounting["rejected"],
        correction_dropped=accounting["dropped"],
        correction_dropped_fraction=accounting["correction_dropped_fraction"],
        longest_correction_gap_s=accounting["longest_correction_gap_s"],
        accepted_fraction=(accounting["accepted_updates"] / accounting["mission_outcomes"]
                           if accounting["mission_outcomes"] else None),
    )
    opportunities, duplicates = aligned.camera_opportunities(run)
    opportunities = [o for o in opportunities if start <= float(o["timestamp_s"]) <= stop]
    result.update(
        duration_sim_s=stop-start,
        camera_opportunities=len(opportunities),
        detector_returns=sum(bool(o["detection_valid"]) for o in opportunities),
        detector_return_fraction=(sum(bool(o["detection_valid"]) for o in opportunities)/len(opportunities)
                                  if opportunities else None),
        duplicate_opportunity_deliveries=duplicates,
        reference_source=truth.source,
        reference_max_gap_s=reference_gap,
        correction_ledger_valid=ledger.valid,
        analysis_status="scored",
    )
    gt = (truth.t >= start) & (truth.t <= stop)
    curve = {
        "time": belief["stamp"][use]-start, "error_m": errors_m,
        "gt_x": truth.x[gt], "gt_y": truth.y[gt],
    }
    return result, curve


def summarize_values(values) -> dict:
    x = np.asarray([v for v in values if v is not None], dtype=float)
    return {
        "n": int(x.size), "mean": float(np.mean(x)) if x.size else None,
        "median": float(np.median(x)) if x.size else None,
        "min": float(np.min(x)) if x.size else None,
        "max": float(np.max(x)) if x.size else None,
    }


def cell_summaries(results: list[dict]) -> list[dict]:
    rows = []
    for task in TASKS:
        for arm in ARMS:
            group = [r for r in results if r["task"] == task and r["arm"] == arm]
            scored = [r for r in group if r.get("analysis_status") == "scored"]
            row = {
                "task": task, "arm": arm, "runs": len(group),
                "evidence_valid_runs": sum(bool(r.get("evidence_valid")) for r in group),
                "strict_successes": sum(bool(r.get("strict_success")) for r in group),
                "collisions": sum(bool(r.get("collision")) for r in group),
                "outcomes": json.dumps(dict(Counter(r["campaign_outcome"] for r in group)), sort_keys=True),
            }
            for field in ("duration_sim_s", "path_length_m", "belief_error_m_rmse",
                          "belief_error_m_p95", "planar_95_ellipse_containment",
                          "planar_nees_median", "correction_dropped_fraction",
                          "longest_correction_gap_s"):
                stats = summarize_values([r.get(field) for r in scored])
                row[f"{field}_run_mean"] = stats["mean"]
                row[f"{field}_run_median"] = stats["median"]
            rows.append(row)
    return rows


def descriptive_bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float] | None:
    if len(values) < 2:
        return None
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def factorial_effects(results: list[dict]) -> dict:
    lookup = {(r["task"], r["arm"], r["seed"]): r for r in results}
    metrics = (
        "strict_success", "collision", "duration_sim_s", "path_length_m",
        "belief_error_m_rmse", "belief_error_m_p95",
        "planar_95_ellipse_containment", "correction_dropped_fraction",
        "longest_correction_gap_s",
    )
    blocks = []
    for task in TASKS:
        seeds = sorted({r["seed"] for r in results if r["task"] == task})
        for seed in seeds:
            arms = {arm: lookup[(task, arm, seed)] for arm in ARMS}
            block = {"task": task, "seed": seed,
                     "outcomes": {arm: arms[arm]["campaign_outcome"] for arm in ARMS}}
            for metric in metrics:
                values = {arm: arms[arm].get(metric) for arm in ARMS}
                if all(value is not None for value in values.values()):
                    c00, c01, c10, c11 = (float(values[a]) for a in ARMS)
                    block[metric] = {
                        "availability_main": 0.5*((c10+c11)-(c00+c01)),
                        "covariance_main": 0.5*((c01+c11)-(c00+c10)),
                        "interaction": c11-c10-c01+c00,
                    }
                else:
                    block[metric] = None
            blocks.append(block)
    rng = np.random.default_rng(91524)
    summary = {}
    for metric in metrics:
        summary[metric] = {}
        for effect in ("availability_main", "covariance_main", "interaction"):
            values = np.asarray([
                block[metric][effect] for block in blocks if block[metric] is not None
            ], dtype=float)
            summary[metric][effect] = {
                **summarize_values(values.tolist()),
                "descriptive_block_bootstrap95_of_mean": descriptive_bootstrap(values, rng),
            }
    return {
        "coding": {
            "availability_main": "0.5[(C10+C11)-(C00+C01)]",
            "covariance_main": "0.5[(C01+C11)-(C00+C10)]",
            "interaction": "C11-C10-C01+C00",
        },
        "replication_unit": "matched task-seed block",
        "blocks": blocks, "summary": summary,
        "uncertainty": "descriptive resampling of the 20 task-seed blocks; not an asymptotic significance test",
    }


def route_summary(config: dict, route_manifest: dict) -> dict:
    by_task = {}
    for task in TASKS:
        hashes = {arm: config["tasks"][task]["preselected_routes"][arm]["preselected_route_sha256"]
                  for arm in ARMS}
        entries = route_manifest["selected"][task]
        by_task[task] = {
            "hashes": hashes,
            "unique_route_hashes": len(set(hashes.values())),
            "same_route_all_arms": len(set(hashes.values())) == 1,
            "route_length_m": {arm: entries[arm]["route_length_m"] for arm in ARMS},
            "minimum_static_body_clearance_m": {
                arm: entries[arm]["minimum_static_body_clearance_m"] for arm in ARMS
            },
            "selected_candidate": {arm: entries[arm]["candidate"] for arm in ARMS},
            "selection_cost": {arm: entries[arm]["total_cost"] for arm in ARMS},
        }
    return {"by_task": by_task,
            "all_tasks_same_route_across_arms": all(v["same_route_all_arms"] for v in by_task.values())}


def render_figures(results: list[dict], cells: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    metrics = (
        ("strict_successes", "Strict successes [of 5]", 1.0),
        ("belief_error_m_rmse_run_mean", "Mean run-level belief RMSE [cm]", 100.0),
        ("planar_95_ellipse_containment_run_mean", "Mean run-level 95% containment", 1.0),
    )
    x = np.arange(len(TASKS)); width = 0.19
    for ax, (field, label, scale) in zip(axes, metrics):
        for index, arm in enumerate(ARMS):
            values = [next(r[field] for r in cells if r["task"] == task and r["arm"] == arm)
                      for task in TASKS]
            values = [np.nan if value is None else scale*value for value in values]
            ax.bar(x + (index-1.5)*width, values, width, color=COLORS[arm], label=arm)
        ax.set_xticks(x, ["W→E", "E→W", "Aisle", "S→N"])
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylim(0, 5.25); axes[0].legend(frameon=False, ncol=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"final_factorial_summary.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        for arm in ARMS:
            for row in [r for r in results if r["task"] == task and r["arm"] == arm]:
                run = row.get("run")
                if not run:
                    continue
                table = aligned.rows(REPO / run)
                x = np.asarray([float(r["gt_x"]) for r in table if r.get("gt_available") == "1.0"])
                y = np.asarray([float(r["gt_y"]) for r in table if r.get("gt_available") == "1.0"])
                ax.plot(x, y, color=COLORS[arm], lw=0.75, alpha=0.48)
        ax.set_title(TASK_LABELS[task]); ax.set_aspect("equal"); ax.grid(alpha=0.15)
        ax.set_xlabel("World x [m]"); ax.set_ylabel("World y [m]")
    for arm in ARMS:
        axes.flat[0].plot([], [], color=COLORS[arm], label=arm)
    axes.flat[0].legend(frameon=False, ncol=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"final_trajectories.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def report_markdown(accounting: dict, cells: list[dict], routes: dict, effects: dict) -> str:
    lines = [
        "# Final 5 Hz navigation campaign", "",
        "## Result", "",
        f"The campaign contains {accounting['planned_runs']} planned runs. "
        f"{accounting['evidence_valid_runs']} runs passed the evidence checks and "
        f"{accounting['strict_successes']} met the strict physical success criterion. "
        f"The ledger retains {accounting['collisions']} collisions and "
        f"{accounting['infrastructure_invalid_runs']} infrastructure-invalid cells.", "",
    ]
    if routes["all_tasks_same_route_across_arms"]:
        lines += [
            "All four planner arms selected the same route within each task. The final "
            "factorial therefore found no route-choice change at these four starts and goals. "
            "Paired execution differences measure repeated-run variation on the same routes, "
            "not a route-mediated planner benefit.", "",
        ]
    lines += [
        "## Per-task results", "",
        "| Task | Arm | Strict success | Outcomes | Belief RMSE | 95% containment | Longest correction gap |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in cells:
        rmse = row["belief_error_m_rmse_run_mean"]
        coverage = row["planar_95_ellipse_containment_run_mean"]
        gap = row["longest_correction_gap_s_run_mean"]
        lines.append(
            f"| {TASK_LABELS[row['task']]} | {row['arm']} | {row['strict_successes']}/{row['runs']} | "
            f"{row['outcomes']} | {100*rmse:.1f} cm | {100*coverage:.1f}% | {gap:.2f} s |"
            if None not in (rmse, coverage, gap) else
            f"| {TASK_LABELS[row['task']]} | {row['arm']} | {row['strict_successes']}/{row['runs']} | "
            f"{row['outcomes']} | unscoreable | unscoreable | unscoreable |"
        )
    lines += ["", "Each continuous statistic was computed per run before aggregation. Belief "
              "errors use the planner prediction timestamp and the highest recorded anchor "
              "revision at each state timestamp.", "", "## Factorial contrasts", "",
              "Effects use the 20 matched task-seed blocks. Positive accuracy or duration "
              "effects are worse. Positive success effects are better.", "",
              "| Metric | Availability main effect | Covariance main effect | Interaction |",
              "|---|---:|---:|---:|"]
    labels = (("strict_success", "Strict success", 1.0, ""),
              ("belief_error_m_rmse", "Belief RMSE", 100.0, " cm"),
              ("duration_sim_s", "Duration", 1.0, " s"),
              ("longest_correction_gap_s", "Longest correction gap", 1.0, " s"))
    for key, label, scale, unit in labels:
        values = effects["summary"][key]
        fields = []
        for effect in ("availability_main", "covariance_main", "interaction"):
            item = values[effect]
            fields.append("unavailable" if item["mean"] is None else f"{scale*item['mean']:+.3f}{unit}")
        lines.append(f"| {label} | {fields[0]} | {fields[1]} | {fields[2]} |")
    lines += [
        "", "Bootstrap intervals in `factorial_effects.json` are descriptive block-resampling "
        "intervals, not significance tests.", "", "## Scope", "",
        "The detector and camera manager run at 5 Hz. The commissioned per-frame covariance "
        "is inflated by five, and route selection uses one effective camera update per "
        "one-second planning step. Residual temporal dependence remains a limitation. The "
        "four tasks and five seeds do not establish arbitrary-warehouse generalization or "
        "nominal uncertainty calibration.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-gap-s", type=float, default=0.15)
    args = parser.parse_args()
    config_path, campaign_root = args.config.resolve(), args.campaign.resolve()
    route_path, output = args.route_manifest.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("analysis outputs are immutable; choose an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entries = freeze_selection(config_path, campaign_root, output)
    results, curves = [], {}
    for entry in entries:
        result, curve = analyze_run(entry, config, args.reference_gap_s, sha256(config_path))
        results.append(result); curves[(entry["task"], entry["arm"], entry["seed"])] = curve
    cells = cell_summaries(results)
    effects = factorial_effects(results)
    routes = route_summary(config, json.loads(route_path.read_text(encoding="utf-8")))
    accounting = {
        "planned_runs": 80, "selected_runs": len(results),
        "evidence_valid_runs": sum(bool(r.get("evidence_valid")) for r in results),
        "strict_successes": sum(bool(r.get("strict_success")) for r in results),
        "collisions": sum(bool(r.get("collision")) for r in results),
        "infrastructure_invalid_runs": sum(r["campaign_outcome"] == "infra_invalid" for r in results),
        "outcomes": dict(Counter(r["campaign_outcome"] for r in results)),
    }
    protocol = {
        "schema": "final_factorial_navigation_analysis.v1",
        "campaign_config": str(config_path.relative_to(REPO)),
        "campaign_config_sha256": sha256(config_path),
        "campaign_root": str(campaign_root.relative_to(REPO)),
        "route_manifest": str(route_path.relative_to(REPO)),
        "route_manifest_sha256": sha256(route_path),
        "analysis_source": str(Path(__file__).resolve().relative_to(REPO)),
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
        "aligned_loader": str(Path(aligned.__file__).resolve().relative_to(REPO)),
        "aligned_loader_sha256": sha256(Path(aligned.__file__).resolve()),
        "reference_gap_s": args.reference_gap_s,
        "analysis_timing_disclosure": (
            "The analysis implementation was finalized during campaign execution. "
            "Metrics and evidence rules inherit the pre-existing localization contract "
            "and Stage-09 analyzer; all results are reported descriptively."
        ),
    }
    write_json(output / "analysis_protocol.json", protocol)
    write_json(output / "run_results.json", results)
    write_csv(output / "run_results.csv", results)
    write_json(output / "cell_summary.json", cells)
    write_csv(output / "cell_summary.csv", cells)
    write_json(output / "factorial_effects.json", effects)
    write_json(output / "route_summary.json", routes)
    summary = {"schema": "final_factorial_navigation_summary.v1",
               "accounting": accounting, "cells": cells,
               "factorial_effects": effects["summary"], "routes": routes,
               "analysis_protocol_sha256": sha256(output / "analysis_protocol.json"),
               "selection_sha256": sha256(output / "selection.json")}
    write_json(output / "summary.json", summary)
    (output / "RESULTS.md").write_text(
        report_markdown(accounting, cells, routes, effects), encoding="utf-8"
    )
    render_figures(results, cells, output)
    owned = sorted(path for path in output.iterdir() if path.name != "manifest.json")
    manifest = {"schema": "final_factorial_navigation_outputs.v1",
                "files": {path.name: sha256(path) for path in owned}}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(accounting, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
