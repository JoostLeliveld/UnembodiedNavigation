#!/usr/bin/env python3
"""Plot the longest accepted-correction gap in one frozen spatial-removal run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.fusion_on_fixed_routes.aligned import (  # noqa: E402
    assimilations,
    camera_opportunities,
    mission_interval,
    rows,
    truth_series,
    validate_run_ledger,
)


def runtime_events(run: Path) -> list[dict]:
    path = run / "runtime_event_deliveries.jsonl.zst"
    text = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [json.loads(line) for line in text.splitlines()]


def f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()

    ledger = validate_run_ledger(run)
    start, stop = mission_interval(run)
    accepted = sorted((
        a for a in assimilations(run)
        if start <= a["apply_stamp"] <= stop
        and a["source_batch_id"] in ledger.accepted_update_ids
    ), key=lambda a: a["apply_stamp"])
    stamps = np.array([start, *[a["apply_stamp"] for a in accepted], stop])
    i = int(np.argmax(np.diff(stamps)))
    gap_start, gap_stop = stamps[i], stamps[i + 1]

    table = rows(run)
    truth = truth_series(run, table, max_reference_gap_s=0.2)
    tt = np.array([f(r, "gt_stamp") for r in table])
    keep = np.isfinite(tt) & (tt >= start) & (tt <= stop)
    tx = np.array([f(r, "gt_x") for r in table])[keep]
    ty = np.array([f(r, "gt_y") for r in table])[keep]
    tt = tt[keep]
    gap_path = (tt >= gap_start) & (tt <= gap_stop)

    bt = np.array([f(r, "planner_belief_stamp") for r in table])
    be = np.array([f(r, "belief_error_gt_m") for r in table])
    bs = np.sqrt(np.maximum(0.0, np.array([
        (f(r, "planner_cov_x") + f(r, "planner_cov_y")) / 2 for r in table
    ])))
    belief_keep = np.isfinite(bt) & (bt >= start) & (bt <= stop)

    opportunities, _ = camera_opportunities(run)
    cameras = ["camera_A", "camera_B", "camera_C", "camera_D", "camera_E"]
    camera_y = {camera: n for n, camera in enumerate(cameras)}
    valid = {camera: [] for camera in cameras}
    misses = {camera: [] for camera in cameras}
    for obs in opportunities:
        stamp = float(obs["timestamp_s"])
        if not start <= stamp <= stop:
            continue
        target = valid if obs["detection_valid"] else misses
        target[obs["camera_id"]].append(stamp)

    mapped = {camera: [] for camera in cameras}
    refused = {camera: [] for camera in cameras}
    reasons: dict[str, int] = {}
    for delivery in runtime_events(run):
        if delivery.get("topic") != "/reliability/camera_manager/batch_outcome":
            continue
        event = delivery.get("parsed_payload", {})
        camera = event.get("camera_id")
        capture_ns = event.get("capture_stamp_ns")
        if event.get("status") != "camera_mapping" or camera not in camera_y or capture_ns is None:
            continue
        stamp = int(capture_ns) / 1e9
        if event.get("disposition") == "mapped":
            mapped[camera].append(stamp)
        elif event.get("disposition") == "refused":
            refused[camera].append(stamp)
            if gap_start <= stamp <= gap_stop:
                reason = str(event.get("reason", "unspecified"))
                reasons[reason] = reasons.get(reason, 0) + 1

    fig = plt.figure(figsize=(10.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1.05, 1.0])
    ax_route = fig.add_subplot(grid[:, 0])
    ax_events = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[1, 1], sharex=ax_events)

    ax_route.plot(tx, ty, color="#9aa0a6", lw=2.2, label="Driven path")
    ax_route.plot(tx[gap_path], ty[gap_path], color="#d62728", lw=4.0,
                  label=f"Longest gap, {gap_stop-gap_start:.2f} s")
    gx, gy = truth.at([gap_start, gap_stop])
    ax_route.scatter(gx, gy, s=52, c=["#d62728", "#2ca02c"], zorder=5)
    ax_route.annotate("last correction", (gx[0], gy[0]), xytext=(-55, -22),
                      textcoords="offset points", arrowprops={"arrowstyle": "->"}, fontsize=8)
    ax_route.annotate("next correction", (gx[1], gy[1]), xytext=(12, 18),
                      textcoords="offset points", arrowprops={"arrowstyle": "->"}, fontsize=8)
    ax_route.set(xlabel="x [m]", ylabel="y [m]", title="Where the gap occurred")
    ax_route.set_aspect("equal", adjustable="box")
    ax_route.grid(alpha=0.25)
    ax_route.legend(loc="lower right", fontsize=8)

    for camera in cameras:
        y = camera_y[camera]
        ax_events.scatter(misses[camera], np.full(len(misses[camera]), y), s=7,
                          color="#d9d9d9", marker="|", linewidths=0.8)
        ax_events.scatter(valid[camera], np.full(len(valid[camera]), y), s=17,
                          color="#1f77b4", marker="o", linewidths=0, label="Detector hit" if y == 0 else None)
        ax_events.scatter(refused[camera], np.full(len(refused[camera]), y), s=18,
                          color="#e67e22", marker="x", linewidths=0.8, label="Mapping refused" if y == 0 else None)
        ax_events.scatter(mapped[camera], np.full(len(mapped[camera]), y), s=18,
                          facecolor="none", edgecolor="#2ca02c", marker="o", linewidths=0.8,
                          label="Mapped measurement" if y == 0 else None)
    ax_events.axvspan(gap_start, gap_stop, color="#d62728", alpha=0.10)
    camera_labels = [c.replace("camera_", "Camera ") for c in cameras]
    camera_labels[1] += "  (removed)"
    ax_events.set_yticks(range(len(cameras)), camera_labels)
    ax_events.set_title("Every frame and its runtime disposition")
    ax_events.grid(axis="x", alpha=0.25)
    ax_events.legend(ncol=3, loc="upper left", fontsize=7)
    ax_events.text(
        0.99, 0.05,
        f"Camera frames  5.0 Hz each\nManager timer  4.0 Hz\nGap refusals  {reasons.get('sensor_gate:CLIPPED_OR_EDGE', 0)} edge-clipped, {reasons.get('detector_miss', 0)} misses",
        transform=ax_events.transAxes, ha="right", va="bottom", fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )

    ax_error.plot(bt[belief_keep], be[belief_keep], color="#542788", lw=1.5, label="Belief position error")
    ax_error.plot(bt[belief_keep], bs[belief_keep], color="#fdae61", lw=1.2, label="Stated planar std.")
    ax_error.scatter([a["apply_stamp"] for a in accepted], np.zeros(len(accepted)),
                     s=14, color="#2ca02c", marker="|", label="Accepted correction")
    ax_error.axvspan(gap_start, gap_stop, color="#d62728", alpha=0.10)
    ax_error.set(xlabel="Simulation time [s]", ylabel="Distance [m]",
                 title="Estimator behaviour during the same interval")
    ax_error.grid(alpha=0.25)
    ax_error.legend(loc="upper left", fontsize=8)
    ax_error.set_xlim(start, stop)

    fig.suptitle("Representative spatial-removal run  |  west task, seed 91504", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(json.dumps({
        "gap_start_s": gap_start,
        "gap_stop_s": gap_stop,
        "gap_duration_s": gap_stop - gap_start,
        "gap_start_xy_m": [float(gx[0]), float(gy[0])],
        "gap_stop_xy_m": [float(gx[1]), float(gy[1])],
        "gap_refusal_reasons": reasons,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
