#!/usr/bin/env python3
"""Build and audit the frozen paper campaign for the four-camera warehouse.

The command has two jobs:

* materialise the exact collection/replay matrix from ``paper_protocol.yaml``;
* inspect a supplied operational run root and report which evidence gates are
  actually earned, without reading simulation truth.

It intentionally does not start Gazebo or tune a policy.  Collection remains a
three-terminal operation so that every passive route has an explicit spawn,
recorder, and route manifest.  The generated checklist makes that process
repeatable, while the audit prevents a sparse pilot from being presented as a
qualified campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# The workspace is intentionally sandboxed; keep Matplotlib's cache outside
# the repo and away from an unwritable user configuration directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/unav_paper_campaign_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO / "experiments" / "multicamera_commissioning_bigwarehouse"
DEFAULT_STUDY = STUDY_DIR / "config" / "study.yaml"
DEFAULT_PROTOCOL = STUDY_DIR / "config" / "paper_protocol.yaml"
DEFAULT_OUTPUT = REPO / "logs" / "studies" / "multicamera_commissioning_bigwarehouse" / "paper_protocol_v1"
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return payload


def _route_index(study: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(route["name"]): dict(route)
        for route in study.get("collection", {}).get("routes", [])
    }


def _collection_rows(
    *,
    phase: str,
    routes: Iterable[str],
    repeats: int,
    offsets: Iterable[float],
    speeds: Iterable[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        for repeat in range(1, int(repeats) + 1):
            for offset in offsets:
                for speed in speeds:
                    rows.append(
                        {
                            "phase": phase,
                            "route": str(route),
                            "repeat": repeat,
                            "lateral_offset_m": float(offset),
                            "speed_mps": float(speed),
                            "seed": "",
                            "condition": "nominal",
                        }
                    )
    return rows


def build_plan(study: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the complete collection and paired-replay contract."""

    routes = _route_index(study)
    mapping = dict(protocol["route_disjoint_mapping"])
    overlap = dict(protocol["overlap_qualification"])
    rows = _collection_rows(
        phase="D1_mapping_train",
        routes=mapping["train_routes"],
        repeats=int(mapping["repeats_per_route"]),
        offsets=mapping["lateral_offsets_m"],
        speeds=mapping["speeds_mps"],
    )
    rows.extend(
        _collection_rows(
            phase="D1_mapping_heldout",
            routes=mapping["heldout_routes"],
            repeats=int(mapping["repeats_per_route"]),
            offsets=mapping["lateral_offsets_m"],
            speeds=mapping["speeds_mps"],
        )
    )
    for edge in overlap["edges"]:
        rows.extend(
            _collection_rows(
                phase=f"D2_overlap_{'_'.join(edge['cameras'])}",
                routes=[str(edge["route"])],
                repeats=int(overlap["repeats_per_route"]),
                offsets=overlap["lateral_offsets_m"],
                speeds=overlap["speeds_mps"],
            )
        )

    for task in protocol["navigation_tasks"]:
        for condition in protocol["randomization"]["conditions"]:
            for seed in protocol["randomization"]["seed_values"]:
                rows.append(
                    {
                        "phase": "D3_paired_replay",
                        "route": str(task["route"]),
                        "repeat": "",
                        "lateral_offset_m": "",
                        "speed_mps": "",
                        "seed": int(seed),
                        "condition": str(condition),
                        "task_id": str(task["id"]),
                        "purpose": str(task["purpose"]),
                    }
                )

    closed_loop = dict(protocol.get("closed_loop_confirmation", {}))
    task_by_id = {str(task["id"]): task for task in protocol["navigation_tasks"]}
    for task_id in closed_loop.get("tasks", []):
        task = task_by_id.get(str(task_id))
        if task is None:
            raise RuntimeError(f"Closed-loop confirmation references unknown task {task_id!r}")
        for policy in closed_loop.get("policies", []):
            for seed in range(int(closed_loop.get("paired_seeds", 0))):
                rows.append(
                    {
                        "phase": "D4_closed_loop_confirmation",
                        "route": str(task["route"]),
                        "repeat": "",
                        "lateral_offset_m": "",
                        "speed_mps": "",
                        "seed": seed,
                        "condition": str(closed_loop.get("condition", "nominal")),
                        "task_id": str(task_id),
                        "purpose": str(task["purpose"]),
                        "policy": str(policy),
                    }
                )

    missing = sorted({str(row["route"]) for row in rows}.difference(routes))
    if missing:
        raise RuntimeError(f"Protocol references routes absent from study.yaml: {missing}")
    return rows


def _finite(value: str | float | int | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _camera_rows(path: Path) -> list[tuple[float, float, float]]:
    """Return operationally valid projected measurements from one camera CSV."""

    if not path.is_file():
        return []
    records = []
    for row in _read_rows(path):
        if _finite(row.get("detected")) < 0.5:
            continue
        stamp = _finite(row.get("diag_stamp"))
        x = _finite(row.get("pred_world_x"))
        y = _finite(row.get("pred_world_y"))
        if all(math.isfinite(value) for value in (stamp, x, y)):
            records.append((stamp, x, y))
    return sorted(records)


def _synchronised_pairs(
    a_rows: list[tuple[float, float, float]],
    b_rows: list[tuple[float, float, float]],
    *,
    max_dt_s: float,
) -> list[float]:
    """Greedily pair nearest unused observations within the frozen time gate."""

    pairs: list[float] = []
    used_b: set[int] = set()
    for stamp_a, x_a, y_a in a_rows:
        best: tuple[float, int] | None = None
        for index, (stamp_b, x_b, y_b) in enumerate(b_rows):
            if index in used_b:
                continue
            dt = abs(stamp_a - stamp_b)
            if dt > max_dt_s:
                continue
            candidate = (dt, index)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            continue
        index = best[1]
        used_b.add(index)
        _, x_b, y_b = b_rows[index]
        pairs.append(math.hypot(x_a - x_b, y_a - y_b))
    return pairs


def discover_runs(run_root: Path) -> list[dict[str, Any]]:
    """Discover only manifest-backed passive recordings under a run root."""

    if not run_root.exists():
        return []
    discovered = []
    for manifest_path in sorted(run_root.glob("**/raw/route_manifest.json")):
        raw_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        route = str(manifest.get("route", ""))
        experiment = raw_dir / "experiment.csv"
        camera_counts = {
            camera: len(_read_rows(raw_dir / f"{camera}_perception.csv"))
            if (raw_dir / f"{camera}_perception.csv").is_file()
            else 0
            for camera in CAMERAS
        }
        covariance_rows = 0
        total_odom_rows = 0
        if experiment.is_file():
            for row in _read_rows(experiment):
                total_odom_rows += 1
                xx = _finite(row.get("odom_noisy_cov_xx"))
                xy = _finite(row.get("odom_noisy_cov_xy"))
                yy = _finite(row.get("odom_noisy_cov_yy"))
                if math.isfinite(xx) and math.isfinite(xy) and math.isfinite(yy) and xx > 0 and yy > 0 and xx * yy > xy * xy:
                    covariance_rows += 1
        truth_path = raw_dir.parent / "evaluation_only" / "ground_truth.csv"
        truth_rows = len(_read_rows(truth_path)) if truth_path.is_file() else 0
        discovered.append(
            {
                "run_id": raw_dir.parent.name,
                "raw_dir": raw_dir,
                "route": route,
                "offset_m": _finite(manifest.get("lateral_offset_m")),
                "speed_mps": _finite(manifest.get("speed_mps")),
                "camera_rows": camera_counts,
                "usable": experiment.is_file() and sum(camera_counts.values()) > 0,
                "covariance_rows": covariance_rows,
                "total_odom_rows": total_odom_rows,
                "truth_rows": truth_rows,
            }
        )
    return discovered


def qualification_status(
    study: dict[str, Any], protocol: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Audit coverage and D2 evidence from operational data only."""

    plan = build_plan(study, protocol)
    collection_rows = [
        row
        for row in plan
        if row["phase"].startswith("D1_") or row["phase"].startswith("D2_")
    ]
    expected_by_phase_route = Counter((row["phase"], row["route"]) for row in collection_rows)
    usable = [run for run in runs if run["usable"]]
    actual_by_route = Counter(str(run["route"]) for run in usable)
    planned_coverage = [
        {
            "phase": phase,
            "route": route,
            "planned_runs": int(planned),
            "usable_runs": int(actual_by_route[route]),
            "complete": bool(actual_by_route[route] >= planned),
        }
        for (phase, route), planned in sorted(expected_by_phase_route.items())
    ]

    overlap = dict(protocol["overlap_qualification"])
    edge_summaries = []
    for edge in overlap["edges"]:
        camera_a, camera_b = (str(value) for value in edge["cameras"])
        distances: list[float] = []
        for run in usable:
            if run["route"] != str(edge["route"]):
                continue
            raw_dir = Path(run["raw_dir"])
            distances.extend(
                _synchronised_pairs(
                    _camera_rows(raw_dir / f"{camera_a}_perception.csv"),
                    _camera_rows(raw_dir / f"{camera_b}_perception.csv"),
                    max_dt_s=float(overlap["max_time_delta_s"]),
                )
            )
        pair_count = len(distances)
        outlier_rate = (
            float(sum(value > float(overlap["max_disagreement_m"]) for value in distances) / pair_count)
            if pair_count
            else math.nan
        )
        passed = bool(
            pair_count >= int(overlap["min_pairs_per_edge"])
            and math.isfinite(outlier_rate)
            and outlier_rate <= float(overlap["max_outlier_rate"])
        )
        edge_summaries.append(
            {
                "camera_a": camera_a,
                "camera_b": camera_b,
                "route": str(edge["route"]),
                "pair_count": pair_count,
                "mean_disagreement_m": None if not distances else float(np.mean(distances)),
                "max_disagreement_m": None if not distances else float(np.max(distances)),
                "outlier_rate": None if not math.isfinite(outlier_rate) else outlier_rate,
                "pass_gate": passed,
            }
        )

    total_odom = sum(int(run["total_odom_rows"]) for run in usable)
    valid_covariance = sum(int(run["covariance_rows"]) for run in usable)
    truth_rows = sum(int(run["truth_rows"]) for run in usable)
    covariance_coverage = valid_covariance / total_odom if total_odom else math.nan
    mapping_routes = set(protocol["route_disjoint_mapping"]["train_routes"])
    mapping_routes.update(protocol["route_disjoint_mapping"]["heldout_routes"])
    return {
        "protocol_id": str(protocol["protocol_id"]),
        "evaluation_world": str(protocol["evaluation_world"]),
        "two_world_rule": bool(protocol["two_world_rule"]),
        "run_count": len(runs),
        "usable_run_count": len(usable),
        "route_counts": dict(sorted(actual_by_route.items())),
        "planned_coverage": planned_coverage,
        "mapping_route_disjointness": {
            "train_routes": list(protocol["route_disjoint_mapping"]["train_routes"]),
            "heldout_routes": list(protocol["route_disjoint_mapping"]["heldout_routes"]),
            "disjoint": not bool(
                set(protocol["route_disjoint_mapping"]["train_routes"])
                & set(protocol["route_disjoint_mapping"]["heldout_routes"])
            ),
            "observed_mapping_routes": sorted(mapping_routes.intersection(actual_by_route)),
        },
        "overlap_edges": edge_summaries,
        "covariance_logging": {
            "valid_rows": valid_covariance,
            "total_rows": total_odom,
            "coverage_fraction": None if not math.isfinite(covariance_coverage) else covariance_coverage,
            "ready_for_calibration": bool(total_odom and covariance_coverage >= 0.99),
        },
        "evaluation_truth": {
            "samples": truth_rows,
            "runs_with_truth": sum(int(run["truth_rows"] > 0) for run in usable),
            "ready_for_paired_replay": bool(usable and all(int(run["truth_rows"]) > 0 for run in usable)),
        },
        "gates": {
            "mapping_collection_complete": all(
                item["complete"]
                for item in planned_coverage
                if item["phase"].startswith("D1_")
            ),
            "overlap_complete": bool(edge_summaries) and all(item["pass_gate"] for item in edge_summaries),
            "covariance_logged": bool(total_odom and covariance_coverage >= 0.99),
            "evaluation_truth_logged": bool(usable and all(int(run["truth_rows"]) > 0 for run in usable)),
            "closed_loop_permitted": False,
            "closed_loop_reason": "Frozen policy release requires mapping, D2, covariance-calibration, and matched replay evidence; this audit does not infer those from a sparse pilot.",
        },
    }


def _save_route_figure(study: dict[str, Any], protocol: dict[str, Any], runs: list[dict[str, Any]], path: Path) -> None:
    routes = _route_index(study)
    mapping = dict(protocol["route_disjoint_mapping"])
    colors = {"train": "#1976d2", "heldout": "#ef6c00", "overlap": "#2e7d32"}
    short_labels = {
        "camera_A_south_west_pass": "A support",
        "camera_B_north_west_pass": "B support",
        "camera_C_south_east_pass": "C support",
        "camera_D_north_east_pass": "D support",
        "south_to_north_handover": "S→N handover",
        "north_to_south_handover": "N→S handover",
        "south_pair_overlap": "south overlap",
        "north_pair_overlap": "north overlap",
        "central_overlap_sweep": "central sweep",
    }
    camera_mounts = {
        "A": (-6.0, -10.0),
        "B": (-6.0, 10.0),
        "C": (6.0, -10.0),
        "D": (6.0, 10.0),
    }
    fig, ax = plt.subplots(figsize=(13.2, 7.9), constrained_layout=True)
    ax.set_facecolor("#f7f9fc")
    ax.add_patch(
        plt.Rectangle(
            (-12.25, -10.25),
            24.5,
            20.5,
            fill=False,
            linewidth=2.0,
            edgecolor="#455a64",
            zorder=0,
        )
    )
    ax.axvspan(-10.0, -3.0, color="#eceff1", alpha=0.70, zorder=-1, label="rack / occlusion zone")
    ax.axvspan(3.0, 10.0, color="#eceff1", alpha=0.70, zorder=-1)
    for camera_id, (x, y) in camera_mounts.items():
        aim_y = -2.5 if camera_id in {"A", "C"} else 2.5
        ax.scatter(x, y, marker="^", s=105, color="#5e35b1", edgecolors="white", linewidths=0.8, zorder=7)
        ax.annotate(
            "",
            xy=(0.0, aim_y),
            xytext=(x, y),
            arrowprops={"arrowstyle": "->", "color": "#9575cd", "alpha": 0.58, "linewidth": 1.6},
            zorder=2,
        )
        ax.text(x, y + (0.7 if y < 0 else -0.9), f"camera {camera_id}", ha="center", fontsize=9, weight="bold")
    for name, route in routes.items():
        start = route["start"]
        goal = route["goal"]
        kind = "overlap"
        if name in mapping["train_routes"]:
            kind = "train"
        elif name in mapping["heldout_routes"]:
            kind = "heldout"
        ax.plot(
            [float(start["x"]), float(goal["x"])],
            [float(start["y"]), float(goal["y"])],
            color=colors[kind],
            linewidth=3.2,
            solid_capstyle="round",
            label={"train": "D1 GP training routes", "heldout": "D1 held-out handover routes", "overlap": "D2 overlap routes"}[kind],
        )
        ax.text(
            (float(start["x"]) + float(goal["x"])) / 2.0 + 0.12,
            (float(start["y"]) + float(goal["y"])) / 2.0 + 0.12,
            short_labels[name],
            fontsize=8.5,
            weight="medium",
        )
    for run in runs:
        route = routes.get(str(run["route"]))
        if route is None:
            continue
        start = route["start"]
        goal = route["goal"]
        ax.scatter(
            [float(start["x"]), float(goal["x"])],
            [float(start["y"]), float(goal["y"])],
            s=18,
            color="#212121" if run["usable"] else "#b71c1c",
            zorder=5,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=True)
    ax.set_title("Frozen four-camera paper protocol: route-disjoint mapping and overlap evidence", weight="bold")
    ax.set_xlabel("warehouse x [m]")
    ax.set_ylabel("warehouse y [m]")
    ax.set_xlim(-12.9, 12.9)
    ax.set_ylim(-10.9, 10.9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_qualification_figure(status: dict[str, Any], path: Path) -> None:
    coverage = status["planned_coverage"]
    labels = [f"{row['phase'].replace('D1_', '').replace('D2_', '')}\n{row['route']}" for row in coverage]
    planned = np.asarray([row["planned_runs"] for row in coverage], dtype=float)
    observed = np.asarray([row["usable_runs"] for row in coverage], dtype=float)
    fig, ax = plt.subplots(figsize=(max(11.0, len(labels) * 1.25), 6.2), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, planned, color="#d8e5f3", label="planned runs")
    ax.bar(x, observed, color="#1565c0", label="usable recorded runs")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("run count")
    ax.set_title("Collection qualification: recorded evidence versus frozen protocol", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    cov = status["covariance_logging"]
    ax.text(
        0.99,
        0.97,
        f"Recorded covariance rows: {cov['valid_rows']}/{cov['total_rows']}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#90a4ae"},
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_route_disjoint_validation(validation_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for camera in CAMERAS:
        path = validation_root / camera / "route_disjoint_validation.csv"
        if not path.is_file():
            continue
        for row in _read_rows(path):
            row["camera_id"] = camera
            rows.append(row)
    return rows


def _save_route_disjoint_figure(rows: list[dict[str, str]], path: Path) -> None:
    """Render held-out Brier/ECE matrices without pooling camera evidence."""

    modes = [
        "constant_train_mean",
        "prior_only",
        "naive",
        "uncertainty_weighted",
        "belief_spread",
        "expected_kernel",
    ]
    by_key = {(str(row["camera_id"]), str(row["mode"])): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0), constrained_layout=True)
    labels = [item.replace("_", "\n") for item in modes]
    for axis, metric, title, cmap in (
        (axes[0], "brier", "Held-out Brier score ↓", "Blues_r"),
        (axes[1], "ece", "Held-out calibration error ↓", "Purples_r"),
    ):
        matrix = np.full((len(CAMERAS), len(modes)), np.nan, dtype=float)
        for i, camera in enumerate(CAMERAS):
            for j, mode in enumerate(modes):
                matrix[i, j] = _finite(by_key.get((camera, mode), {}).get(metric))
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_xticks(np.arange(len(modes)), labels, fontsize=8)
        axis.set_yticks(np.arange(len(CAMERAS)), CAMERAS)
        axis.set_title(title, weight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, shrink=0.86)
    fig.suptitle("Route-disjoint per-camera GP validation — complete held-out route", weight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "phase",
        "route",
        "repeat",
        "lateral_offset_m",
        "speed_mps",
        "seed",
        "condition",
        "task_id",
        "purpose",
        "policy",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_talking_points(path: Path, protocol: dict[str, Any], status: dict[str, Any]) -> None:
    overlap = status["overlap_edges"]
    overlap_lines = "\n".join(
        f"- {row['camera_a']}↔{row['camera_b']}: {row['pair_count']} synchronized operational pairs; "
        f"D2 {'PASS' if row['pass_gate'] else 'NOT YET QUALIFIED'}."
        for row in overlap
    )
    text = f"""# Paper-campaign talking points

## What is frozen

- Evaluation uses `{protocol['evaluation_world']}`; method development remains in `{protocol['method_development_world']}`.
- Per-camera GP posteriors are trained only on D1 training routes and evaluated on route-disjoint handover routes.
- The pooled GP remains diagnostic-only. Runtime selection receives one posterior per camera.

## How to read the figures

- `01_protocol_route_coverage.png` separates blue training routes, orange held-out handover routes, and green overlap qualification routes. Black endpoints mark usable recorded passes.
- `02_collection_qualification.png` reports collection progress against the frozen matrix. It is deliberately a progress figure, not a performance figure.
- `03_route_disjoint_gp_validation.png`, when supplied with GP validation outputs, reports Brier and calibration per camera without pooling camera evidence. Treat it as pilot-only unless the frozen D1 matrix is complete.

## Gate status

- Usable recordings: {status['usable_run_count']} of {status['run_count']} manifest-backed runs.
- Propagated covariance present in {status['covariance_logging']['valid_rows']}/{status['covariance_logging']['total_rows']} logged odometry rows.
- Evaluation-only truth: {status['evaluation_truth']['samples']} samples across {status['evaluation_truth']['runs_with_truth']} usable runs.
{overlap_lines or '- No overlap recording is available yet.'}
- Closed-loop handover remains disabled until mapping, D2 overlap, covariance calibration, and paired replay gates all pass.
"""
    path.write_text(text, encoding="utf-8")


def _write_results(path: Path, status: dict[str, Any], validation_rows: list[dict[str, str]]) -> None:
    """Write a claim-safe summary beside figures, including negative evidence."""

    lines = [
        "# Paper campaign status",
        "",
        "## Release decision",
        "",
        "**Not eligible for a learned-map, fusion, or closed-loop claim.**",
        "",
        f"- Mapping matrix complete: `{status['gates']['mapping_collection_complete']}`.",
        f"- All D2 edges qualified: `{status['gates']['overlap_complete']}`.",
        f"- Propagated covariance logged: `{status['gates']['covariance_logged']}`.",
        f"- Evaluation-only truth logged: `{status['gates']['evaluation_truth_logged']}`.",
        "",
        "## Current operational pilot",
        "",
        f"- {status['usable_run_count']} usable recordings out of {status['run_count']} manifest-backed recordings.",
        f"- {status['covariance_logging']['valid_rows']}/{status['covariance_logging']['total_rows']} odometry rows contain propagated covariance (the existing pilot predates this implementation).",
        f"- {status['evaluation_truth']['samples']} evaluation-only truth samples are available for paired replay.",
        "",
    ]
    if validation_rows:
        lines.extend(
            [
                "## Route-disjoint GP pilot",
                "",
                "The table uses complete held-out runs, but only 12–13 held-out detector observations per camera. It is an end-to-end smoke result, not a paper comparison.",
                "",
                "| Camera | Best Brier mode | Brier | Held-out events |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for camera in CAMERAS:
            candidates = [row for row in validation_rows if row.get("camera_id") == camera]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: _finite(row.get("brier")))
            lines.append(
                f"| {camera} | {best['mode']} | {_finite(best.get('brier')):.3f} | {best['heldout_events']} |"
            )
        lines.extend(
            [
                "",
                "Several learned variants underperform a constant or day-zero prior on this tiny split. That is the expected reason to collect the frozen D1 matrix before selecting a method.",
                "",
            ]
        )
    lines.extend(
        [
            "## Next evidence-producing action",
            "",
            "Collect the D1 training routes with the new covariance and separate evaluation-only truth recorder, then collect the held-out handovers and each D2 edge to its configured 30-pair minimum. Re-run this folder after every batch; do not alter the protocol based on held-out outcomes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_collection_checklist(path: Path, study: dict[str, Any], protocol: dict[str, Any]) -> None:
    text = f"""# Frozen collection checklist

Use one row from `campaign_plan.csv` at a time.  Do not change a threshold, GP hyperparameter, or route after reviewing evaluation-only truth.

1. Launch `warehouse_full4cam_commissioning.launch.py` at the selected route's documented spawn pose, setting the paired encoder-noise seed.
2. Start `record_operational_logs.py` with that same documented spawn pose.  Verify its manifest says `contains_ground_truth: false` and that the propagated covariance columns are populated.
3. Start `record_evaluation_truth.py` separately under `evaluation_only/`; it is a passive evaluator and must never be an operational topic.
4. Run `drive_study_route.py` with the plan row's route, offset, and speed.  Keep the emitted `route_manifest.json` beside the raw CSVs.
5. Use `attach_evaluation_truth.py` to create evaluation-only copies, then export each completed run with `reliability_tools export-multicamera`.
6. Build per-camera GP inputs from D1 training runs.  Fit the four posteriors separately, then run the route-disjoint held-out report.
7. Qualify every D2 edge at ≥{protocol['overlap_qualification']['min_pairs_per_edge']} synchronized pairs before allowing its fusion condition.
8. Run all replay policies on exactly the same exported frames and paired seeds.  Do not include any oracle policy in the operational comparison.

`paper_campaign.py` may be re-run at any time to update the audit figures and gate status.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=None, help="Operational run root to audit; omit for an empty-plan audit.")
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Root containing camera_A...camera_D/route_disjoint_validation.csv outputs.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    study_path = args.study.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    allowed_root = (REPO / "logs" / "studies" / "multicamera_commissioning_bigwarehouse").resolve()
    if allowed_root not in out_dir.parents and out_dir != allowed_root:
        raise RuntimeError(f"--out-dir must stay under {allowed_root}")
    study = _load_yaml(study_path)
    protocol = _load_yaml(protocol_path)
    plan = build_plan(study, protocol)
    runs = discover_runs(args.run_root.expanduser().resolve()) if args.run_root else []
    status = qualification_status(study, protocol, runs)

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "campaign_plan.csv", plan)
    (out_dir / "qualification_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_route_figure(study, protocol, runs, figures / "01_protocol_route_coverage.png")
    _save_qualification_figure(status, figures / "02_collection_qualification.png")
    validation_rows = (
        _load_route_disjoint_validation(args.validation_root.expanduser().resolve())
        if args.validation_root
        else []
    )
    if validation_rows:
        _save_route_disjoint_figure(validation_rows, figures / "03_route_disjoint_gp_validation.png")
    _write_talking_points(out_dir / "TALKING_POINTS.md", protocol, status)
    _write_results(out_dir / "RESULTS.md", status, validation_rows)
    _write_collection_checklist(out_dir / "COLLECTION_CHECKLIST.md", study, protocol)

    print(json.dumps({"out_dir": str(out_dir), "status": status["gates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
