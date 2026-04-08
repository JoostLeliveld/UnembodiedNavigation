#!/usr/bin/env python3
"""Summarize logged runs for the thesis comparison pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from json import JSONDecodeError
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, JSONDecodeError, TypeError, ValueError):
        return {}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_float(row, key) for row in rows], dtype=float)


def _finite_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _finite_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0 if finite.size == 1 else float("nan")
    return float(np.std(finite, ddof=1))


def _polyline_length(xs: np.ndarray, ys: np.ndarray) -> float:
    finite = np.isfinite(xs) & np.isfinite(ys)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    pts = np.column_stack([xs[finite], ys[finite]])
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def _run_summary(run_dir: Path) -> dict[str, object] | None:
    manifest = _load_json(run_dir / "run_manifest.json")
    experiment_rows = _load_csv(run_dir / "experiment.csv")
    if not experiment_rows:
        return None

    perception_rows = _load_csv(run_dir / "perception.csv")
    final_row = experiment_rows[-1]
    goal_dist = _series(experiment_rows, "goal_dist")
    plan_time_ms = _series(experiment_rows, "plan_time_ms")
    solve_time_ms = _series(experiment_rows, "solve_time_ms")
    p_vis_plan = _series(experiment_rows, "p_vis_plan")
    detected = _series(perception_rows, "detected")
    stamps = _series(experiment_rows, "stamp")
    state_x = _series(experiment_rows, "x")
    state_y = _series(experiment_rows, "y")
    state_pos_error = _series(perception_rows, "state_pos_error")
    state_yaw_error_deg = _series(perception_rows, "state_yaw_error_deg")

    method = str(manifest.get("method") or manifest.get("planner") or "unknown")
    success_radius_m = float(manifest.get("goal_success_radius", 0.35) or 0.35)
    finite_goal_dist = goal_dist[np.isfinite(goal_dist)]
    min_goal_dist = float(np.min(finite_goal_dist)) if finite_goal_dist.size else float("nan")
    run_duration_s = float(np.nanmax(stamps) - np.nanmin(stamps)) if np.any(np.isfinite(stamps)) else float("nan")
    success = bool(np.isfinite(min_goal_dist) and min_goal_dist <= success_radius_m)
    return {
        "run_dir": str(run_dir),
        "method": method,
        "planner": str(manifest.get("planner", method)),
        "world": str(manifest.get("world", "")),
        "task": str(manifest.get("task", "")),
        "seed": int(manifest.get("seed", 0) or 0),
        "success": int(success),
        "success_radius_m": success_radius_m,
        "final_goal_dist": _float(final_row, "goal_dist"),
        "min_goal_dist": min_goal_dist,
        "run_duration_s": run_duration_s,
        "executed_path_length_m": _polyline_length(state_x, state_y),
        "mean_plan_time_ms": _finite_mean(plan_time_ms),
        "mean_solve_time_ms": _finite_mean(solve_time_ms),
        "mean_plan_visibility": _finite_mean(p_vis_plan),
        "detection_rate": _finite_mean(detected),
        "mean_state_pos_error": _finite_mean(state_pos_error),
        "mean_state_yaw_error_deg": _finite_mean(state_yaw_error_deg),
    }


def _group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["world"]), str(row["task"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (method, world, task), group in sorted(grouped.items()):
        def _mean(key: str) -> float:
            values = np.asarray([float(item[key]) for item in group], dtype=float)
            return float(np.nanmean(values)) if values.size else float("nan")

        summary.append({
            "method": method,
            "world": world,
            "task": task,
            "n_runs": len(group),
            "success_rate": _mean("success"),
            "mean_final_goal_dist": _mean("final_goal_dist"),
            "std_final_goal_dist": _finite_std(np.asarray([float(item["final_goal_dist"]) for item in group], dtype=float)),
            "mean_min_goal_dist": _mean("min_goal_dist"),
            "mean_run_duration_s": _mean("run_duration_s"),
            "mean_executed_path_length_m": _mean("executed_path_length_m"),
            "mean_plan_time_ms": _mean("mean_plan_time_ms"),
            "mean_solve_time_ms": _mean("mean_solve_time_ms"),
            "mean_plan_visibility": _mean("mean_plan_visibility"),
            "mean_detection_rate": _mean("detection_rate"),
            "mean_state_pos_error": _mean("mean_state_pos_error"),
            "mean_state_yaw_error_deg": _mean("mean_state_yaw_error_deg"),
        })
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="?", default=Path("logs/experiments"))
    parser.add_argument("--method", action="append", dest="methods", default=None, help="Filter to one or more methods")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/evaluation"))
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_methods = {
        name.strip() for name in (args.methods or ["efe1", "visibility_unaware_baseline"]) if name.strip()
    }
    run_dirs = sorted(path for path in run_root.glob("experiment_*") if path.is_dir())
    run_rows = []
    for run_dir in run_dirs:
        row = _run_summary(run_dir)
        if row is None:
            continue
        if requested_methods and str(row["method"]) not in requested_methods:
            continue
        run_rows.append(row)

    grouped_rows = _group_summary(run_rows)
    _write_csv(output_dir / "run_summary.csv", run_rows)
    _write_csv(output_dir / "group_summary.csv", grouped_rows)
    with (output_dir / "group_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(grouped_rows, handle, indent=2)

    print(f"Wrote {output_dir / 'run_summary.csv'}")
    print(f"Wrote {output_dir / 'group_summary.csv'}")
    print(f"Wrote {output_dir / 'group_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
