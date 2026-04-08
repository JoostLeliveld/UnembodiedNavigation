#!/usr/bin/env python3
"""Summarize logged runs for the thesis comparison pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _run_summary(run_dir: Path) -> dict[str, object] | None:
    manifest = _load_json(run_dir / "run_manifest.json")
    experiment_rows = _load_csv(run_dir / "experiment.csv")
    if not experiment_rows:
        return None

    perception_rows = _load_csv(run_dir / "perception.csv")
    final_row = experiment_rows[-1]
    goal_dist = np.asarray([_float(row, "goal_dist") for row in experiment_rows], dtype=float)
    plan_time_ms = np.asarray([_float(row, "plan_time_ms") for row in experiment_rows], dtype=float)
    solve_time_ms = np.asarray([_float(row, "solve_time_ms") for row in experiment_rows], dtype=float)
    p_vis_plan = np.asarray([_float(row, "p_vis_plan") for row in experiment_rows], dtype=float)
    detected = np.asarray([_float(row, "detected") for row in perception_rows], dtype=float)

    method = str(manifest.get("method") or manifest.get("planner") or "unknown")
    return {
        "run_dir": str(run_dir),
        "method": method,
        "planner": str(manifest.get("planner", method)),
        "world": str(manifest.get("world", "")),
        "task": str(manifest.get("task", "")),
        "seed": int(manifest.get("seed", 0) or 0),
        "final_goal_dist": _float(final_row, "goal_dist"),
        "min_goal_dist": float(np.nanmin(goal_dist)) if goal_dist.size else float("nan"),
        "mean_plan_time_ms": float(np.nanmean(plan_time_ms)) if plan_time_ms.size else float("nan"),
        "mean_solve_time_ms": float(np.nanmean(solve_time_ms)) if solve_time_ms.size else float("nan"),
        "mean_plan_visibility": float(np.nanmean(p_vis_plan)) if np.any(np.isfinite(p_vis_plan)) else float("nan"),
        "detection_rate": float(np.nanmean(detected)) if detected.size else float("nan"),
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
            "mean_final_goal_dist": _mean("final_goal_dist"),
            "mean_min_goal_dist": _mean("min_goal_dist"),
            "mean_plan_time_ms": _mean("mean_plan_time_ms"),
            "mean_solve_time_ms": _mean("mean_solve_time_ms"),
            "mean_plan_visibility": _mean("mean_plan_visibility"),
            "mean_detection_rate": _mean("detection_rate"),
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
