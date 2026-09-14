#!/usr/bin/env python3
"""Score and plot the non-inferential, one-seed Stage-09 sanity pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis_stage09_sanity_mpl")
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


CAMPAIGN = REPO / "logs/thesis_final_pipeline_v1/stage09_navigation/sanity_pilot_v3"
CONFIG = REPO / "experiments/thesis_pipeline_lock/stage09_sanity_pilot_v2.yaml"
OUTPUT = REPO / "logs/thesis_final_pipeline_v1/stage10_reporting/navigation_sanity_v3"
COLORS = {"P0": "#657789", "P1": "#2a78d6", "P2": "#d0792a"}
LABELS = {"P0": "HULL", "P1": "MLP", "P2": "FULL"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def map_background(ax, collision_json: str) -> None:
    for prism in scene_from_json(collision_json).prisms:
        if prism.zmin <= 0.55 and prism.zmax >= 0.0:
            ax.add_patch(Rectangle(
                (prism.xmin, prism.ymin), prism.xmax - prism.xmin,
                prism.ymax - prism.ymin, facecolor="#d9dfe2",
                edgecolor="#a7b0b5", linewidth=0.22, zorder=0,
            ))
    ax.set(xlim=(-12.3, 12.3), ylim=(-10.3, 10.3), aspect="equal",
           xlabel="World x [m]", ylabel="World y [m]")
    ax.grid(alpha=0.10, linewidth=0.4)


def score_run(arm: str, event: dict) -> tuple[dict, dict]:
    run = Path(event["run_dir"])
    attempt = Path(event["run_log_dir"])
    summary = json.loads((run / "run_summary.json").read_text())
    verdict = json.loads((attempt / "attempt_evidence_verdict.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    start, stop = aligned.mission_interval(run)
    table = aligned.rows(run)
    truth = aligned.truth_series(run, table, max_reference_gap_s=0.15)
    gt_use = (truth.t >= start) & (truth.t <= stop)

    belief = aligned.aligned_error_cm(run, "belief", table, max_reference_gap_s=0.15)
    if belief.get("estimate_selection") == "highest_anchor_revision_per_state_timestamp":
        identity = aligned.latest_revision_mask(
            belief["stamp"], belief["revision"], belief["epoch"]
        )
    else:
        identity = aligned.landed_mask(belief["stamp"])
    use = (
        identity & belief["have"] & belief["reference_supported"]
        & (belief["stamp"] >= start) & (belief["stamp"] <= stop)
        & np.isfinite(belief["aligned_cm"])
    )
    error_m = np.asarray(belief["aligned_cm"][use], dtype=float) / 100.0

    experiment = read_csv(run / "experiment.csv")
    command = np.asarray([
        abs(float(row["cmd_v"])) for row in experiment
        if row.get("stamp") and start <= float(row["stamp"]) <= stop
        and row.get("cmd_v") not in (None, "", "nan", "NaN")
    ])
    accounting = aligned.correction_accounting(run)
    opportunities, duplicates = aligned.camera_opportunities(run)
    opportunities = [
        row for row in opportunities
        if start <= float(row["timestamp_s"]) <= stop
    ]
    fused = [
        row for row in aligned.fused_answers(run, max_reference_gap_s=0.15)
        if start <= row["fused_stamp"] <= stop
    ]
    fused_error = np.asarray([row["error_cm"] for row in fused], dtype=float) / 100.0
    assimilations = read_csv(run / "correction_assimilations.csv")
    accepted_stamps = [
        float(row["apply_stamp"]) for row in assimilations
        if str(row.get("accepted", "")).lower() in ("1", "true")
        and start <= float(row["apply_stamp"]) <= stop
    ]
    last_accepted_stamp = max(accepted_stamps) if accepted_stamps else math.nan
    fused_after_lockout = np.asarray([
        row["error_cm"] / 100.0 for row in fused
        if row["fused_stamp"] > last_accepted_stamp
    ], dtype=float)
    evidence_valid = bool(
        event.get("attempt_evidence_complete") is True
        and verdict.get("complete") is True
        and summary.get("evidence_complete") is True
        and summary.get("valid_run") is True
    )
    result = {
        "arm": arm,
        "label": LABELS[arm],
        "outcome": event["outcome"],
        "collision_reason": summary.get("collision_reason") or None,
        "evidence_valid": evidence_valid,
        "duration_sim_s": float(stop - start),
        "path_length_m": float(summary["path_length_m"]),
        "final_goal_distance_m": float(summary["final_goal_distance"]),
        "peak_command_speed_mps": float(np.max(command)) if command.size else None,
        "belief_rmse_m": float(np.sqrt(np.mean(error_m ** 2))) if error_m.size else None,
        "belief_p95_m": float(np.quantile(error_m, 0.95)) if error_m.size else None,
        "belief_max_m": float(np.max(error_m)) if error_m.size else None,
        "camera_opportunities": len(opportunities),
        "detector_returns": sum(bool(row["detection_valid"]) for row in opportunities),
        "corrections_published": int(accounting["mission_outcomes"]),
        "corrections_accepted": int(accounting["accepted_updates"]),
        "corrections_rejected": int(accounting["rejected"]),
        "longest_correction_gap_s": float(accounting["longest_correction_gap_s"]),
        "last_accepted_update_s": float(last_accepted_stamp - start),
        "terminal_update_gap_s": float(stop - last_accepted_stamp),
        "fused_error_p95_m": float(np.quantile(fused_error, 0.95)) if fused_error.size else None,
        "post_lockout_fused_n": int(fused_after_lockout.size),
        "post_lockout_fused_p95_m": (
            float(np.quantile(fused_after_lockout, 0.95))
            if fused_after_lockout.size else None
        ),
        "post_lockout_fused_max_m": (
            float(np.max(fused_after_lockout)) if fused_after_lockout.size else None
        ),
        "duplicate_opportunity_deliveries": int(duplicates),
        "run_dir": str(run.relative_to(REPO)),
        "run_summary_sha256": sha256(run / "run_summary.json"),
    }
    curve = {
        "time_s": np.asarray(belief["stamp"][use], dtype=float) - start,
        "belief_error_m": error_m,
        "gt_x": np.asarray(truth.x[gt_use], dtype=float),
        "gt_y": np.asarray(truth.y[gt_use], dtype=float),
        "terminal_x": float(truth.x[gt_use][-1]),
        "terminal_y": float(truth.y[gt_use][-1]),
        "last_accepted_update_s": float(last_accepted_stamp - start),
        "assimilation_time_s": np.asarray([
            float(row["apply_stamp"]) - start for row in assimilations
            if start <= float(row["apply_stamp"]) <= stop
            and row.get("nis") not in (None, "", "nan", "NaN")
        ], dtype=float),
        "assimilation_nis": np.asarray([
            float(row["nis"]) for row in assimilations
            if start <= float(row["apply_stamp"]) <= stop
            and row.get("nis") not in (None, "", "nan", "NaN")
        ], dtype=float),
        "assimilation_accepted": np.asarray([
            str(row.get("accepted", "")).lower() in ("1", "true")
            for row in assimilations
            if start <= float(row["apply_stamp"]) <= stop
            and row.get("nis") not in (None, "", "nan", "NaN")
        ], dtype=bool),
        "fused_time_s": np.asarray([row["fused_stamp"] - start for row in fused], dtype=float),
        "fused_error_m": fused_error,
        "collision_json": manifest["collision_geometry_json"],
    }
    return result, curve


def save(fig, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    ledger = json.loads((CAMPAIGN / "campaign_log.json").read_text())
    config = yaml.safe_load(CONFIG.read_text())
    expected = {f"thesis09_west_to_east_north__{arm}__seed900" for arm in COLORS}
    if set(ledger) != expected:
        raise RuntimeError("campaign ledger is not the exact three-cell pilot")
    results, curves = [], {}
    for arm in COLORS:
        event = ledger[f"thesis09_west_to_east_north__{arm}__seed900"]
        if not event.get("finished_at"):
            raise RuntimeError(f"incomplete campaign cell {arm}")
        result, curve = score_run(arm, event)
        results.append(result)
        curves[arm] = curve

    plt.rcParams.update({"font.size": 8.2, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(10.6, 6.05), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=(1.0, 0.82))
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]

    ax = axes[0]
    map_background(ax, curves["P0"]["collision_json"])
    for arm in COLORS:
        route = np.asarray(json.loads(
            config["tasks"]["thesis09_west_to_east_north"]
            ["preselected_routes"][arm]["preselected_route_json"]
        ), dtype=float)
        ax.plot(route[:, 0], route[:, 1], "--", color=COLORS[arm], lw=0.9,
                alpha=0.72)
        curve = curves[arm]
        ax.plot(curve["gt_x"], curve["gt_y"], color=COLORS[arm], lw=1.7,
                label=LABELS[arm])
        marker = "X" if results[list(COLORS).index(arm)]["outcome"] == "collision" else "s"
        ax.scatter(curve["terminal_x"], curve["terminal_y"], marker=marker,
                   s=45, color=COLORS[arm], edgecolor="white", linewidth=0.5, zorder=5)
    ax.scatter(-7.9, -8.7, marker="o", s=22, color="#20272c", zorder=6)
    ax.scatter(10.6, 6.5, marker="*", s=75, color="#20272c", zorder=6)
    ax.set_title("Frozen routes and recorded motion")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    for arm in COLORS:
        ax.plot(curves[arm]["time_s"], 100 * curves[arm]["belief_error_m"],
                color=COLORS[arm], lw=1.25, label=LABELS[arm])
        ax.axvline(curves[arm]["last_accepted_update_s"], color=COLORS[arm],
                   ls=":", lw=0.8, alpha=0.8)
    ax.axhline(25, color="#9e3d32", ls="--", lw=0.8, label="25 cm diagnostic line")
    ax.set(xlabel="Simulation time [s]", ylabel="Own-time belief error [cm]",
           title="Localization during the pilot")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.axis("off")
    rows = []
    for result in results:
        rows.append([
            result["label"], result["outcome"], f'{result["path_length_m"]:.1f}',
            f'{result["peak_command_speed_mps"]:.2f}',
            f'{100 * result["belief_rmse_m"]:.1f}',
            str(result["corrections_accepted"]),
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["Arm", "End", "Path\n[m]", "Peak $v$\n[m/s]", "Belief\nRMSE [cm]", "Accepted\nupdates"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.16, 0.20, 0.13, 0.15, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.55)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#c9ced1")
        if row == 0:
            cell.set_facecolor("#edf0f2")
            cell.set_text_props(weight="bold")
    ax.set_title("One seed: diagnostic only", pad=3)
    gaps = ", ".join(
        f'{result["label"]} {result["terminal_update_gap_s"]:.1f} s'
        for result in results
    )
    ax.text(0.5, 0.08, "All ledgers valid; 0/3 reached the goal.\n"
            f"Terminal no-update gaps: {gaps}.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
            color="#7f3028")

    ax = axes[3]
    for arm in COLORS:
        curve = curves[arm]
        accepted = curve["assimilation_accepted"]
        ax.scatter(curve["assimilation_time_s"][accepted],
                   curve["assimilation_nis"][accepted], s=17, marker="o",
                   color=COLORS[arm], alpha=0.82, label=f"{LABELS[arm]} accepted")
        ax.scatter(curve["assimilation_time_s"][~accepted],
                   curve["assimilation_nis"][~accepted], s=20, marker="x",
                   color=COLORS[arm], alpha=0.88, label=f"{LABELS[arm]} rejected")
    ax.axhline(9.21, color="#9e3d32", ls="--", lw=0.9, label="NIS gate 9.21")
    ax.set(xlabel="Simulation time [s]", ylabel="Planar innovation NIS",
           title="Correct updates become self-locked out")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=6.4, ncol=2)

    ax = axes[4]
    for arm in COLORS:
        curve = curves[arm]
        ax.plot(curve["fused_time_s"], 100 * curve["fused_error_m"], ".-",
                color=COLORS[arm], lw=0.75, ms=2.5, alpha=0.85,
                label=LABELS[arm])
        ax.axvline(curve["last_accepted_update_s"], color=COLORS[arm],
                   ls=":", lw=0.8, alpha=0.8)
    ax.axhline(25, color="#9e3d32", ls="--", lw=0.8)
    ax.set(xlabel="Simulation time [s]", ylabel="Fused measurement error [cm]",
           title="Camera fusion stays accurate after lockout")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("1.0 m/s commissioning-stack runtime sanity pilot (seed 900)", fontsize=11)
    save(fig, "stage09_sanity_navigation")

    report = {
        "schema": "thesis_stage09_sanity_pilot.v1",
        "status": "diagnostic_not_inferential",
        "campaign_log": str((CAMPAIGN / "campaign_log.json").relative_to(REPO)),
        "campaign_log_sha256": sha256(CAMPAIGN / "campaign_log.json"),
        "config": str(CONFIG.relative_to(REPO)),
        "config_sha256": sha256(CONFIG),
        "claim_boundary": "one task, one seed; route/controller commissioning check only",
        "results": results,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
