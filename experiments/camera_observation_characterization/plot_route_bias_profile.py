#!/usr/bin/env python3
"""Show camera-reading bias along a declared route without inventing a time series.

The characterization capture teleported over a warehouse grid. This script projects
spatially held-out grid positions onto a frozen route, selects the captured heading nearest
the local direction of travel, and plots camera-reading errors against route distance. It is
a virtual route profile, not a dynamic drive and not evidence about temporal filter error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
DEFAULT_ROUTE = REPO / "experiments/fusion_on_fixed_routes/routes/fusion_network_traverse.json"
for rel in ("experiments/deck_figures", "experiments/camera_observation_characterization"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
import plot_gate_sensitivity as G  # noqa: E402

METHODS = (
    ("raw", "Raw box → floor", "no correction"),
    ("fixed", "Fixed 30.9 cm", "one radial constant"),
    ("learned", "Learned linear", "box-only correction"),
    ("nn", "Neural net", "box-only correction"),
)
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FOLDER = "07_along_a_virtual_route"
ROUTE_TOLERANCE_M = 0.50
STATION_DEDUP_M = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def project_to_polyline(xy: np.ndarray, points: np.ndarray):
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    best = None
    for index, (start, delta, length) in enumerate(zip(points[:-1], segments, lengths)):
        fraction = float(np.clip(np.dot(xy - start, delta) / (length * length), 0.0, 1.0))
        projected = start + fraction * delta
        candidate = {
            "distance_m": float(np.linalg.norm(xy - projected)),
            "station_m": float(cumulative[index] + fraction * length),
            "segment_index": index,
            "projected_xy": projected,
            "tangent": delta / length,
        }
        if best is None or candidate["distance_m"] < best["distance_m"]:
            best = candidate
    return best


def load_route(path: Path) -> tuple[dict, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("polyline_canonical_json")
    if not isinstance(raw, str):
        raise RuntimeError(f"{path} lacks polyline_canonical_json")
    points = np.asarray(json.loads(raw), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise RuntimeError(f"Invalid route polyline in {path}")
    return payload, points


def choose_stations(rows: list[dict[str, str]], route: np.ndarray) -> list[dict]:
    by_position: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_position[row["position_id"]].append(row)

    candidates = []
    for position_id, group in by_position.items():
        first = group[0]
        if first["split"] != "test":
            continue
        projection = project_to_polyline(
            np.asarray([float(first["robot_x"]), float(first["robot_y"])]), route
        )
        if projection["distance_m"] > ROUTE_TOLERANCE_M:
            continue
        target_yaw = math.atan2(projection["tangent"][1], projection["tangent"][0])
        heading_yaws = {row["heading_id"]: float(row["robot_yaw"]) for row in group}
        heading_id = min(
            heading_yaws,
            key=lambda key: abs(wrap(heading_yaws[key] - target_yaw)),
        )
        selected = [row for row in group if row["heading_id"] == heading_id]
        if {row["camera_id"] for row in selected} != set(CAMERAS):
            raise RuntimeError(f"Incomplete camera set at position {position_id}")
        candidates.append({
            **projection,
            "position_id": position_id,
            "position_xy": np.asarray([float(first["robot_x"]), float(first["robot_y"])]),
            "heading_id": heading_id,
            "captured_yaw": heading_yaws[heading_id],
            "heading_difference_deg": math.degrees(abs(wrap(heading_yaws[heading_id] - target_yaw))),
            "rows": selected,
        })

    # At diagonal segments, two neighbouring grid rows can project to effectively the same
    # station. Keep the one physically closest to the route so each x location is counted once.
    kept: list[dict] = []
    for item in sorted(candidates, key=lambda value: value["station_m"]):
        if kept and item["station_m"] - kept[-1]["station_m"] < STATION_DEDUP_M:
            if item["distance_m"] < kept[-1]["distance_m"]:
                kept[-1] = item
        else:
            kept.append(item)
    if not kept:
        raise RuntimeError("No held-out characterization positions lie near the route")
    return kept


def route_component(row: dict[str, str], method: str, tangent: np.ndarray,
                    component: str) -> float:
    residual = np.asarray([float(row[f"{method}_dx"]), float(row[f"{method}_dy"])])
    if component == "forward":
        return float(np.dot(residual, tangent))
    normal_left = np.asarray([-tangent[1], tangent[0]])
    return float(np.dot(residual, normal_left))


def admitted_rows(station: dict, predicate) -> list[dict[str, str]]:
    return [row for row in station["rows"] if predicate(row)]


def station_series(stations: list[dict], predicate, method: str, component: str):
    """Every returned reading at each station, tagged with whether the gate admits it."""
    result = []
    for station in stations:
        points = []
        for row in station["rows"]:
            if row[f"{method}_valid"] != "1":
                continue
            points.append((
                row["camera_id"],
                route_component(row, method, station["tangent"], component),
                bool(predicate(row)),
            ))
        result.append((station["station_m"], points))
    return result


def broken_median(ax, series, *, admitted_only: bool, colour: str, lw: float,
                  linestyle: str, zorder: float) -> None:
    x, y = [], []
    previous = None
    for station, points in series:
        chosen = [value for _camera, value, admitted in points
                  if admitted or not admitted_only]
        if not chosen:
            if x and not math.isnan(x[-1]):
                x.append(math.nan)
                y.append(math.nan)
            previous = None
            continue
        if previous is not None and station - previous > 1.6:
            x.append(math.nan)
            y.append(math.nan)
        x.append(station)
        y.append(float(np.median(chosen)))
        previous = station
    ax.plot(x, y, color=colour, lw=lw, linestyle=linestyle, marker="o", ms=4.5,
            zorder=zorder)


def draw_profile(
    stations: list[dict],
    route: np.ndarray,
    route_meta: dict,
    predicate,
    gate_name: str,
    filename: str,
    out: Path,
) -> dict:
    """One profile showing every returned reading and which ones the gate keeps."""
    method_series = {
        (method, component): station_series(stations, predicate, method, component)
        for method, _title, _subtitle in METHODS
        for component in ("forward", "left")
    }
    absolute = np.asarray([
        abs(value)
        for series in method_series.values()
        for _station, points in series
        for _camera, value, _admitted in points
    ])
    limit = max(0.35, min(0.80, float(np.quantile(absolute, 0.985)))) if absolute.size else 0.5
    route_length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    counts = [len(admitted_rows(station, predicate)) for station in stations]
    all_counts = [
        sum(row["raw_valid"] == "1" for row in station["rows"]) for station in stations
    ]
    readings = int(sum(counts))
    all_readings = int(sum(all_counts))
    opportunities = len(stations) * len(CAMERAS)

    columns = len(METHODS)
    fig = plt.figure(figsize=(5.2 * columns, 15.0), constrained_layout=True)
    grid = fig.add_gridspec(3, columns, height_ratios=(1.12, 1.0, 1.0))
    map_ax = fig.add_subplot(grid[0, :columns - 1])
    note_ax = fig.add_subplot(grid[0, columns - 1])
    axes = np.asarray([
        [fig.add_subplot(grid[1, col]) for col in range(columns)],
        [fig.add_subplot(grid[2, col]) for col in range(columns)],
    ])

    D.draw_warehouse(map_ax, D.layout(), show_cameras=True, camera_labels=True, rack_alpha=0.72)
    map_ax.plot(route[:, 0], route[:, 1], color=D.ROBOT, lw=4.2, zorder=6)
    map_ax.scatter([item["position_xy"][0] for item in stations],
                   [item["position_xy"][1] for item in stations],
                   s=45, facecolors="white", edgecolors=D.ROBOT, linewidths=1.5, zorder=7)
    map_ax.plot(route[0, 0], route[0, 1], marker="o", ms=12, mfc="white", mec=D.ROBOT,
                mew=3, zorder=8)
    map_ax.plot(route[-1, 0], route[-1, 1], marker="*", ms=18, color=D.ROBOT,
                mec="white", mew=1.2, zorder=8)
    map_ax.set_title("Frozen route and held-out field samples", fontsize=15.5,
                     fontweight="bold")

    note_ax.axis("off")
    note_ax.text(0.0, 0.98, "How to read this virtual drive", va="top", fontsize=16,
                 fontweight="bold")
    note_ax.text(
        0.0,
        0.86,
        f"Route: {route_meta['task'].replace('_', ' ')} ({route_length:.1f} m)\n"
        f"Field stations: {len(stations)} held-out positions\n"
        f"Readings returned: {all_readings}/{opportunities} camera opportunities\n"
        f"Still admitted after the gate: {readings}/{opportunities}\n"
        f"Median cameras per station after the gate: {np.median(counts):.0f}\n\n"
        "x is distance along the route, not time. At each\n"
        "position we use the captured heading nearest the\n"
        "local direction of travel.\n\n"
        "Filled dots are readings the candidate gate keeps.\n"
        "Hollow dots are readings it rejects — they are drawn\n"
        "so the cost of the gate is visible, not hidden.\n"
        "Black line is the station median of what survives;\n"
        "grey dashed line is the median of everything.\n\n"
        "Positive forward error lands ahead of the robot.\n"
        "Positive sideways error lands left of travel.\n\n"
        "This exposes spatial bias and camera handovers. It\n"
        "does not reproduce motion blur, timing or filter error.\n"
        f"Triangles at \u00b1{limit:.2f} m mark larger off-scale errors.",
        va="top",
        fontsize=11.4,
        linespacing=1.34,
    )

    clipped = 0
    for col, (method, method_title, method_subtitle) in enumerate(METHODS):
        for row_index, (component, y_label) in enumerate((
            ("forward", "Signed error along travel (m)\npositive = reading lands ahead"),
            ("left", "Signed error across travel (m)\npositive = reading lands left"),
        )):
            ax = axes[row_index, col]
            series = method_series[(method, component)]
            for station, points in series:
                for camera_id, value, admitted in points:
                    shown = float(np.clip(value, -limit, limit))
                    clipped += int(shown != value)
                    letter = camera_id[-1]
                    marker = "^" if value > limit else ("v" if value < -limit else "o")
                    if admitted:
                        ax.scatter(station, shown, s=36, marker=marker,
                                   color=D.CAM_COLOUR[letter], alpha=0.80,
                                   edgecolors="white", linewidths=0.35, zorder=4)
                    else:
                        ax.scatter(station, shown, s=36, marker=marker,
                                   facecolors="none", edgecolors=D.CAM_COLOUR[letter],
                                   linewidths=1.15, alpha=0.65, zorder=3)
            broken_median(ax, series, admitted_only=False, colour=D.MUTED, lw=1.6,
                          linestyle="--", zorder=5)
            broken_median(ax, series, admitted_only=True, colour=D.INK, lw=2.2,
                          linestyle="-", zorder=6)
            ax.axhline(0.0, color=D.MUTED, lw=1.2, linestyle=":", zorder=2)
            ax.set_xlim(0.0, route_length)
            ax.set_ylim(-limit, limit)
            ax.grid(color="#e4e2dc", lw=0.8)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(y_label)
            if row_index == 0:
                ax.set_title(f"{method_title}\n{method_subtitle}", fontsize=14,
                             fontweight="bold")
            else:
                ax.set_xlabel("Distance travelled along frozen route (m)")

    camera_handles = [
        Line2D([0], [0], marker="o", lw=0, ms=8, color=D.CAM_COLOUR[letter],
               label=f"Camera {letter}") for letter in "ABCDE"
    ] + [
        Line2D([0], [0], marker="o", lw=0, ms=8, mfc="none", mec=D.MUTED, mew=1.3,
               label="rejected by the gate"),
        Line2D([0], [0], color=D.INK, marker="o", ms=5, lw=2.2,
               label="median of admitted readings"),
        Line2D([0], [0], color=D.MUTED, lw=1.6, linestyle="--",
               label="median of all returned readings"),
    ]
    fig.legend(handles=camera_handles, loc="lower center", ncol=8, frameon=True,
               fontsize=11.0, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Where does box-projection bias appear during a warehouse traverse?\n"
        f"Virtual route profile from held-out field samples — {gate_name}",
        fontsize=19,
        fontweight="bold",
    )
    save_path = out / FOLDER / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")
    return {
        "gate_name": gate_name,
        "stations": len(stations),
        "camera_opportunities": opportunities,
        "camera_readings_returned": all_readings,
        "camera_readings_admitted": readings,
        "fraction_returned": all_readings / opportunities,
        "fraction_admitted": readings / opportunities,
        "zero_camera_stations_after_gate": int(sum(count == 0 for count in counts)),
        "median_cameras_per_station_after_gate": float(np.median(counts)),
        "plot_limit_m": limit,
        "clipped_points_across_all_panels": clipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    route_path = args.route.expanduser().resolve()
    table = capture / "bias_update_interpretations.csv"
    bias_manifest_path = capture / "bias_update_interpretations_manifest.json"
    capture_manifest_path = capture / "capture_manifest.json"
    for path in (table, bias_manifest_path, capture_manifest_path, route_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")
    bias_manifest = json.loads(bias_manifest_path.read_text(encoding="utf-8"))
    if sha256(table) != bias_manifest.get("bias_update_interpretations_sha256"):
        raise RuntimeError("bias update table no longer matches its manifest")

    route_meta, route = load_route(route_path)
    every = list(csv.DictReader(table.open(encoding="utf-8")))
    stations = choose_stations(every, route)
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    camera_xy = {
        item["camera_id"]: tuple(float(value) for value in item["pose_xyz_rpy"][:2])
        for item in capture_manifest["cameras"]
    }
    gate_defs = {gate_id: predicate for gate_id, _label, predicate in G.gates(camera_xy)}

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    target = out / FOLDER
    if target.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {target}; pass --overwrite to refresh")
    target.mkdir(parents=True, exist_ok=True)

    reports = {
        "all_returns_with_candidate_gate": draw_profile(
            stations, route, route_meta, gate_defs["candidate_edge_range"],
            "all selected YOLO returns, marked by whether the candidate "
            "edge + raw-range gate admits them",
            "18_virtual_route_bias.png", out,
        ),
    }
    manifest = {
        "status": "complete",
        "schema": "bbox_virtual_route_bias_profile.v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": str(table),
        "source_table_sha256": sha256(table),
        "route_source": str(route_path),
        "route_source_sha256": sha256(route_path),
        "route_task": route_meta.get("task"),
        "route_polyline_sha256": route_meta.get("polyline_sha256"),
        "route_length_m": float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1))),
        "methods": [method for method, _title, _subtitle in METHODS],
        "selection": {
            "split": "held-out checkerboard tiles only",
            "maximum_position_to_route_distance_m": ROUTE_TOLERANCE_M,
            "near_duplicate_station_distance_m": STATION_DEDUP_M,
            "heading_rule": "captured heading nearest local route tangent",
            "stations": [
                {
                    "position_id": station["position_id"],
                    "station_m": station["station_m"],
                    "position_to_route_distance_m": station["distance_m"],
                    "heading_id": station["heading_id"],
                    "heading_difference_deg": station["heading_difference_deg"],
                }
                for station in stations
            ],
        },
        "interpretation": (
            "spatial route profile reconstructed from a teleport field capture; not a timed "
            "drive, not repeated-sampling evidence, and not belief/filter error"
        ),
        "ground_truth_firewall": (
            "candidate gate uses only confidence, selected pixel, raw back-projection and "
            "camera calibration; commanded pose is used offline to select route-near held-out "
            "samples and score the retained camera readings"
        ),
        "model_refit_after_route_selection_or_gating": False,
        "reports": reports,
        "figures": [str(path.relative_to(target)) for path in sorted(target.glob("*.png"))],
    }
    manifest_path = target / "route_profile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
