#!/usr/bin/env python3
"""Render the reference-controlled commissioning drives and sensor-gate outcomes.

Only fit and development tables are opened. Audit-drive paths remain sealed. The map
draws every distinct reference sample once; the outcome bars count every camera
opportunity and replay the versioned deterministic sensor gate from recorded fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/commissioning_drives_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "src/reliability"),
    str(REPO / "src/unav_common"),
    str(REPO / "experiments/warehouse_v2_sketches"),
]

from reliability.observation_gates import (  # noqa: E402
    UsableObservationGateConfig,
    evaluate_sensor_gate,
)
import warehouse_v2 as warehouse  # noqa: E402


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
ROUTE_COLORS = {
    "west_spine": "#31688e",
    "central_spine": "#35b779",
    "south_cross": "#e6842a",
    "north_cross": "#8e5aa9",
    "settle_lap1": "#31688e",
    "settle_lap2": "#35b779",
    "settle_lap3": "#e6842a",
    "settle_lap4": "#8e5aa9",
    "settle_lap5": "#2a9d8f",
    "settle_lap6": "#c65d3b",
}
ROUTE_LABELS = {
    "west_spine": "west spine",
    "central_spine": "central spine",
    "south_cross": "south cross-aisle",
    "north_cross": "north cross-aisle",
    "settle_lap1": "fit lap 1",
    "settle_lap2": "fit lap 2",
    "settle_lap3": "development lap 1",
    "settle_lap4": "development lap 2",
    "settle_lap5": "audit lap 1",
    "settle_lap6": "audit lap 2",
}
OUTCOME_COLORS = {
    "admitted": "#238e83",
    "refused": "#d8862c",
    "miss": "#8a9095",
}
SURFACE = "#ffffff"
INK = "#1c252b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_unsealed_tables(campaign: Path) -> tuple[list[dict[str, str]], dict]:
    execution_path = campaign / "campaign_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("campaign is not complete with its audit partition sealed")
    if execution.get("incomplete_note"):
        raise RuntimeError("campaign is explicitly marked incomplete")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("expected audit_analysis_permitted=false")

    selected_run_dirs = execution.get("selected_run_dirs")
    if selected_run_dirs is None:
        drive_sources = execution.get("drive_sources")
        if drive_sources is not None:
            selected_run_dirs = {
                source["drive_id"]: source["source"]
                for source in drive_sources
            }
        else:
            selected_run_dirs = {
                drive_id: str(campaign / drive_id)
                for drive_id in execution.get("completed_drive_ids", [])
            }

    rows: list[dict[str, str]] = []
    sources = []
    audit_ids = []
    for drive_id, run_text in sorted(selected_run_dirs.items()):
        if drive_id.startswith("audit_"):
            audit_ids.append(drive_id)
            continue
        if not drive_id.startswith(("fit_", "development_")):
            raise RuntimeError(f"unexpected drive id {drive_id}")
        run = Path(run_text).resolve()
        if "audit" in run.name or any(part.startswith("audit_") for part in run.parts):
            raise RuntimeError(f"refusing audit path {run}")
        table = run / "tables/camera_opportunities.csv"
        with table.open(newline="", encoding="utf-8") as handle:
            members = list(csv.DictReader(handle))
        rows.extend(members)
        sources.append({
            "drive_id": drive_id,
            "partition": members[0]["partition"],
            "path": str(table),
            "sha256": sha256(table),
            "rows": len(members),
        })
    return rows, {
        "campaign_execution": str(execution_path),
        "campaign_execution_sha256": sha256(execution_path),
        "audit_analysis_permitted": False,
        "audit_drive_ids_not_opened": sorted(audit_ids),
        "input_tables": sources,
    }


def gate_outcome(row: dict[str, str], gate: UsableObservationGateConfig) -> str:
    if row["yolo_hit"] != "1":
        return "miss"
    box = None
    try:
        box = [float(row[name]) for name in
               ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")]
    except (TypeError, ValueError):
        pass
    result = evaluate_sensor_gate({
        "frame_expected": True,
        "frame_received": True,
        "detection_received": True,
        "detector_class": "robot",
        "detector_confidence": row["detector_score"],
        "bbox_xmin": box[0] if box else None,
        "bbox_ymin": box[1] if box else None,
        "bbox_xmax": box[2] if box else None,
        "bbox_ymax": box[3] if box else None,
        "projection_valid": bool(row["raw_projected_x"] and row["raw_projected_y"]),
    }, gate)
    return "admitted" if result.admitted else "refused"


def distinct_reference_traces(rows: list[dict[str, str]]) -> dict[str, list[dict]]:
    samples: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        if not row["reference_x"] or not row["reference_y"]:
            continue
        stamp = int(row["capture_stamp_ns"])
        samples[row["drive_id"]][stamp] = {
            "stamp": stamp,
            "x": float(row["reference_x"]),
            "y": float(row["reference_y"]),
            "route": row["route"],
            "partition": row["partition"],
            "direction": row["direction"],
        }
    return {
        drive: [by_stamp[key] for key in sorted(by_stamp)]
        for drive, by_stamp in sorted(samples.items())
    }


def draw_warehouse(ax) -> None:
    layout = warehouse.build()
    ax.add_patch(Rectangle(
        (-12, -10), 24, 20, facecolor=SURFACE, edgecolor="#62696d",
        linewidth=1.0, zorder=0,
    ))
    for zone in layout.zones:
        ax.add_patch(Rectangle(
            (zone.xmin, zone.ymin), zone.xmax - zone.xmin, zone.ymax - zone.ymin,
            facecolor="#e3e4e2", edgecolor="#c2c5c3", linewidth=0.45, zorder=1,
        ))
    for camera in layout.cameras:
        angle = np.deg2rad(camera.yaw_deg)
        ax.scatter(camera.x, camera.y, marker="s", s=23, color="#26343c",
                   edgecolor="white", linewidth=0.55, zorder=8)
        ax.annotate(
            "", xy=(camera.x + 1.15 * np.cos(angle), camera.y + 1.15 * np.sin(angle)),
            xytext=(camera.x, camera.y),
            arrowprops=dict(arrowstyle="-|>", color="#26343c", lw=0.8,
                            shrinkA=3, shrinkB=0), zorder=7,
        )
        ax.text(camera.x, camera.y + 0.58, camera.name, fontsize=7.2,
                fontweight="bold", color="#26343c", ha="center", va="center", zorder=9)
    ax.set(xlim=(-12.4, 12.4), ylim=(-10.4, 10.4), aspect="equal",
           xlabel="world $x$ [m]", ylabel="world $y$ [m]")
    ax.tick_params(labelsize=7.5, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_map(ax, traces: dict[str, list[dict]], *, panel_label: bool = True) -> int:
    draw_warehouse(ax)
    sample_count = 0
    plotted_routes: list[str] = []
    for _drive_id, trace in traces.items():
        if not trace:
            continue
        route = trace[0]["route"]
        if route not in plotted_routes:
            plotted_routes.append(route)
        xy = np.asarray([[row["x"], row["y"]] for row in trace])
        sample_count += len(trace)
        color = ROUTE_COLORS[route]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.75,
                alpha=0.72, zorder=4)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.7, color=color, alpha=0.50,
                   linewidths=0, zorder=5)
        if len(xy) >= 3:
            index = min(max(len(xy) // 2, 1), len(xy) - 2)
            delta = xy[index + 1] - xy[index - 1]
            norm = float(np.linalg.norm(delta))
            if norm > 1e-9:
                delta = 0.58 * delta / norm
                ax.annotate(
                    "", xy=xy[index] + 0.5 * delta, xytext=xy[index] - 0.5 * delta,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.65,
                                    mutation_scale=6), zorder=6,
                )
    title = "(a) Reference-controlled drive samples" if panel_label else "Reference-controlled drive samples"
    ax.set_title(title, fontsize=9.2, loc="left")
    route_handles = [
        Line2D([], [], color=color, lw=1.8, label=ROUTE_LABELS[route])
        for route in plotted_routes
        for color in (ROUTE_COLORS[route],)
    ]
    camera_handle = Line2D([], [], color="#26343c", marker="s", ms=4.5, lw=0,
                           label="fixed camera")
    ax.legend(handles=route_handles + [camera_handle], frameon=False,
              fontsize=6.6, ncol=1, loc="center right", handlelength=1.9,
              labelspacing=0.35)
    return sample_count


def plot_outcomes(ax, rows: list[dict[str, str]], gate: UsableObservationGateConfig) -> dict:
    counts = {camera: Counter() for camera in CAMERAS}
    for row in rows:
        counts[row["camera_id"]][gate_outcome(row, gate)] += 1
    y = np.arange(len(CAMERAS))
    left = np.zeros(len(CAMERAS))
    totals = np.asarray([sum(counts[camera].values()) for camera in CAMERAS], dtype=float)
    for outcome in ("admitted", "refused", "miss"):
        values = 100.0 * np.asarray([counts[camera][outcome] for camera in CAMERAS]) / totals
        ax.barh(y, values, left=left, height=0.58, color=OUTCOME_COLORS[outcome],
                edgecolor="white", linewidth=0.45, label=outcome)
        for index, (start, value) in enumerate(zip(left, values, strict=True)):
            if value >= 8.0:
                ax.text(start + 0.5 * value, index, f"{value:.0f}%", ha="center",
                        va="center", fontsize=6.8,
                        color="white" if outcome != "refused" else INK)
        left += values
    ax.set_yticks(y, labels=[f"Camera {camera[-1]}" for camera in CAMERAS])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="camera opportunities [%]")
    ax.set_title("(b) Outcome of every opportunity", fontsize=9.2, loc="left")
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.31), columnspacing=0.8, handlelength=1.1)
    return {camera: dict(counts[camera]) for camera in CAMERAS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, provenance = read_unsealed_tables(args.campaign.resolve())
    gate = UsableObservationGateConfig.from_yaml(str(args.gate.resolve()))
    gate.assert_belief_independent()
    traces = distinct_reference_traces(rows)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#8b9194",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": "#4f575b",
        "ytick.color": "#4f575b",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 3.55), gridspec_kw={"width_ratios": [1.52, 1.0]}
    )
    sample_count = plot_map(axes[0], traces)
    outcomes = plot_outcomes(axes[1], rows, gate)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.19, wspace=0.22)

    stem = "commissioning_drive_survey"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight",
                    pad_inches=0.025)
    plt.close(fig)

    map_fig, map_ax = plt.subplots(figsize=(3.55, 3.25))
    plot_map(map_ax, traces, panel_label=False)
    map_fig.subplots_adjust(left=0.17, right=0.99, top=0.91, bottom=0.15)
    map_stem = "commissioning_drive_routes_current"
    for suffix in ("pdf", "svg", "png"):
        map_fig.savefig(output / f"{map_stem}.{suffix}", dpi=300,
                        bbox_inches="tight", pad_inches=0.025)
    plt.close(map_fig)

    total = Counter()
    for camera_counts in outcomes.values():
        total.update(camera_counts)
    report = {
        "schema": "reference_controlled_commissioning_drive_figure.v1",
        "audit_analysis_permitted": False,
        "gate_id": gate.gate_id,
        "gate_config": str(args.gate.resolve()),
        "gate_config_sha256": sha256(args.gate.resolve()),
        "complete_drives": len(traces),
        "distinct_reference_samples": sample_count,
        "camera_opportunities": len(rows),
        "outcomes": dict(total),
        "outcomes_by_camera": outcomes,
        "provenance": provenance,
    }
    (output / f"{stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
