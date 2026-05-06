#!/usr/bin/env python3
"""Plot Monte Carlo trajectory bundles and run-to-run deviation.

Input is one or more run_model_selection.py grid logs. The expected comparison
shape is two cells, C1_constant_R and C2_<selected_label>, but the plotter will
group any labels present in the logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

from plot_individual_model_selection_runs import (
    GOAL_SUCCESS_RADIUS_M,
    RHO_LOW_THRESHOLD,
    _clean_success,
    _draw_world,
    _is_true,
    _load_csv_rows,
    _load_gp,
    _load_json,
    _mean_csv_value,
    _pf,
    _query_rho,
)


DEFAULT_REFERENCE_GP = Path("logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz")
DEFAULT_OUT_DIR = Path("logs/visibility_comparison/mc_trajectory_comparison")


PLANNER_CONDITION = {
    "constant_R_efe": "C1",
    "visibility_aware_efe": "C2",
    "risk_only_ablation": "C3",
}

COLORS = {
    "C1": "#111827",
    "C2": "#7c3aed",
    "C3": "#ea580c",
}


@dataclass
class RunBundle:
    task: str
    label: str
    planner: str
    condition: str
    seed: str
    run_dir: Path
    outcome: str
    entry: dict[str, Any]
    xs: np.ndarray
    ys: np.ndarray
    clean_success: bool
    low_rho_exposure: float
    mean_rho: float
    path_length_m: float
    minimum_goal_distance: float
    elapsed_after_first_cmd_s: float


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "na"


def _run_dir_from_entry(entry: dict[str, Any]) -> Path | None:
    run_dir = str(entry.get("run_dir", "") or "")
    if not run_dir:
        return None
    path = Path(run_dir).expanduser()
    if (path / "run_summary.json").is_file() or (path / "experiment.csv").is_file():
        return path
    candidates = sorted(path.rglob("run_summary.json")) if path.is_dir() else []
    return candidates[-1].parent if candidates else path


def _truth_points(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if _pf(row, "truth_available") < 0.5:
            continue
        x = _pf(row, "truth_x")
        y = _pf(row, "truth_y")
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _finite_median(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else math.nan


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _load_records(grid_logs: list[Path], gp, only_tasks: set[str]) -> tuple[list[RunBundle], list[str]]:
    records: list[RunBundle] = []
    skipped: list[str] = []
    for grid_log in grid_logs:
        log = _load_json(grid_log)
        if not isinstance(log, dict):
            skipped.append(f"{grid_log}: not a JSON mapping")
            continue
        for key, entry in log.items():
            if not isinstance(entry, dict):
                continue
            merged = entry.get("merged_config") if isinstance(entry.get("merged_config"), dict) else {}
            task = str(merged.get("task", ""))
            if only_tasks and task not in only_tasks:
                continue
            run_dir = _run_dir_from_entry(entry)
            if run_dir is None:
                skipped.append(f"{grid_log}:{key}: missing run_dir")
                continue
            summary_path = run_dir / "run_summary.json"
            experiment_path = run_dir / "experiment.csv"
            if not summary_path.is_file() or not experiment_path.is_file():
                skipped.append(f"{grid_log}:{key}: missing run files at {run_dir}")
                continue
            rows = _load_csv_rows(experiment_path)
            summary = _load_json(summary_path)
            xs, ys = _truth_points(rows)
            if xs.size < 2:
                skipped.append(f"{grid_log}:{key}: not enough truth points")
                continue
            rhos = np.asarray([_query_rho(gp, x, y) for x, y in zip(xs, ys)], dtype=float)
            finite = rhos[np.isfinite(rhos)]
            outcome = str(entry.get("outcome", "") or "")
            planner = str(merged.get("planner", ""))
            condition = PLANNER_CONDITION.get(planner, planner or "unknown")
            records.append(
                RunBundle(
                    task=task,
                    label=str(entry.get("label", "")),
                    planner=planner,
                    condition=condition,
                    seed=str(entry.get("seed", "")),
                    run_dir=run_dir,
                    outcome=outcome,
                    entry=entry,
                    xs=xs,
                    ys=ys,
                    clean_success=_clean_success(outcome, summary),
                    low_rho_exposure=float(np.mean(finite < RHO_LOW_THRESHOLD)) if finite.size else math.nan,
                    mean_rho=float(np.mean(finite)) if finite.size else math.nan,
                    path_length_m=_as_float(summary.get("path_length_m", math.nan)),
                    minimum_goal_distance=_as_float(summary.get("minimum_goal_distance", math.nan)),
                    elapsed_after_first_cmd_s=_as_float(summary.get("elapsed_after_first_cmd_s", math.nan)),
                )
            )
    return records, skipped


def _resample_path(xs: np.ndarray, ys: np.ndarray, n: int = 200) -> np.ndarray:
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if xs.size < 2:
        return np.full((n, 2), np.nan)
    ds = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.column_stack([np.full(n, xs[0]), np.full(n, ys[0])])
    q = s / total
    target = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(target, q, xs), np.interp(target, q, ys)])


def _cov_ellipse(points: np.ndarray, *, nsigma: float = 1.0) -> tuple[float, float, float] | None:
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 2:
        return None
    cov = np.cov(points.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
    return 2.0 * nsigma * math.sqrt(float(vals[0])), 2.0 * nsigma * math.sqrt(float(vals[1])), angle


def _group_key(record: RunBundle) -> str:
    if record.condition in {"C1", "C2", "C3"}:
        return f"{record.condition} {record.label}".strip()
    return record.label or record.condition or "unknown"


def _condition_from_group(group: str) -> str:
    return group.split()[0] if group.split() else group


def _plot_task(task: str, runs: list[RunBundle], gp, out_dir: Path, formats: set[str]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    if not runs:
        return out_rows

    manifest = _load_json(runs[0].run_dir / "run_manifest.json")
    groups: dict[str, list[RunBundle]] = {}
    for run in runs:
        groups.setdefault(_group_key(run), []).append(run)

    fig = plt.figure(figsize=(10.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    dev_ax = fig.add_subplot(gs[1, 0])

    im = _draw_world(ax, gp, manifest)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("nominal raw-GP rho_plan")

    goal_x = math.nan
    goal_y = math.nan
    first_rows = _load_csv_rows(runs[0].run_dir / "experiment.csv")
    if first_rows:
        goal_x = _mean_csv_value(first_rows[:10], "goal_x")
        goal_y = _mean_csv_value(first_rows[:10], "goal_y")
    if math.isfinite(goal_x) and math.isfinite(goal_y):
        ax.add_patch(Circle((goal_x, goal_y), GOAL_SUCCESS_RADIUS_M, fill=False, edgecolor="#ef4444",
                            linewidth=1.8, linestyle="--", zorder=15))
        ax.scatter([goal_x], [goal_y], marker="*", s=260, facecolor="#ef4444",
                   edgecolors="black", linewidths=1.0, zorder=16, label="goal")

    q = np.linspace(0.0, 1.0, 200)
    for group_idx, (group, group_runs) in enumerate(sorted(groups.items())):
        condition = _condition_from_group(group)
        color = COLORS.get(condition, f"C{group_idx}")
        clean_n = sum(1 for run in group_runs if run.clean_success)
        total_n = len(group_runs)
        med_low = _finite_median([run.low_rho_exposure for run in group_runs])
        med_path = _finite_median([run.path_length_m for run in group_runs])
        label = f"{group} ({clean_n}/{total_n}, low {med_low:.2f}, L {med_path:.2f}m)"

        resampled: list[np.ndarray] = []
        for run in group_runs:
            alpha = 0.24 if run.clean_success else 0.12
            linestyle = "-" if run.clean_success else ":"
            ax.plot(run.xs, run.ys, color=color, alpha=alpha, linewidth=1.2,
                    linestyle=linestyle, zorder=8 if run.clean_success else 5)
            resampled.append(_resample_path(run.xs, run.ys, n=q.size))

        stack = np.stack(resampled, axis=0)
        mean_path = np.nanmean(stack, axis=0)
        ax.plot(mean_path[:, 0], mean_path[:, 1], color=color, linewidth=3.1,
                solid_capstyle="round", zorder=12, label=label)

        if group_runs:
            ax.scatter([group_runs[0].xs[0]], [group_runs[0].ys[0]], marker="o", s=110,
                       facecolor="#22c55e", edgecolors="black", linewidths=1.0, zorder=16)

        deviations = np.sqrt(np.nanmean(np.sum((stack - mean_path[None, :, :]) ** 2, axis=2), axis=0))
        dev_ax.plot(q, deviations, color=color, linewidth=2.0, label=group)

        for idx in np.linspace(18, q.size - 18, 7).astype(int):
            pts = stack[:, idx, :]
            center = mean_path[idx]
            ellipse = _cov_ellipse(pts, nsigma=1.0)
            if ellipse is None or not np.all(np.isfinite(center)):
                continue
            width, height, angle = ellipse
            ax.add_patch(Ellipse(
                xy=(float(center[0]), float(center[1])),
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                alpha=0.13,
                linewidth=1.0,
                zorder=10,
            ))

        for run in group_runs:
            out_rows.append({
                "task": task,
                "group": group,
                "label": run.label,
                "planner": run.planner,
                "condition": run.condition,
                "seed": run.seed,
                "outcome": run.outcome,
                "clean_success": run.clean_success,
                "reference_low_rho_exposure": run.low_rho_exposure,
                "mean_rho_reference": run.mean_rho,
                "path_length_m": run.path_length_m,
                "minimum_goal_distance": run.minimum_goal_distance,
                "elapsed_after_first_cmd_s": run.elapsed_after_first_cmd_s,
                "run_dir": str(run.run_dir),
            })

    ax.set_title(f"Monte Carlo trajectories: {task}", fontsize=12)
    ax.legend(loc="lower left", fontsize=8.2, framealpha=0.92, facecolor="white", edgecolor="black")
    dev_ax.set_xlabel("normalized path progress")
    dev_ax.set_ylabel("run deviation (m)")
    dev_ax.grid(True, alpha=0.25)
    dev_ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    out_base = out_dir / f"mc_trajectories_{_safe_name(task)}"
    if "png" in formats:
        fig.savefig(out_base.with_suffix(".png"), dpi=190)
    if "pdf" in formats:
        fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    return out_rows


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task", "group", "label", "planner", "condition", "seed", "outcome",
        "clean_success", "reference_low_rho_exposure", "mean_rho_reference",
        "path_length_m", "minimum_goal_distance", "elapsed_after_first_cmd_s", "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, bool):
                    out[field] = "1" if value else "0"
                elif isinstance(value, float):
                    out[field] = "" if not math.isfinite(value) else f"{value:.6f}"
                else:
                    out[field] = value
            writer.writerow(out)


def _write_summary(path: Path, rows: list[dict[str, Any]], skipped: list[str]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["group"])), []).append(row)

    lines = [
        "# Monte Carlo Trajectory Comparison",
        "",
        "| task | group | clean success | median low-rho | median path m | median min-goal m |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (task, group), group_rows in sorted(grouped.items()):
        clean_n = sum(1 for row in group_rows if _is_true(row.get("clean_success")))
        n = len(group_rows)
        med_low = _finite_median([float(row["reference_low_rho_exposure"]) for row in group_rows])
        med_path = _finite_median([float(row["path_length_m"]) for row in group_rows])
        med_goal = _finite_median([float(row["minimum_goal_distance"]) for row in group_rows])
        lines.append(f"| {task} | {group} | {clean_n}/{n} | {med_low:.3f} | {med_path:.2f} | {med_goal:.3f} |")

    lines += ["", "Figures are written as `mc_trajectories_<task>.png/.pdf` when requested."]
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- {item}" for item in skipped[:80]]
        if len(skipped) > 80:
            lines.append(f"- ... {len(skipped) - 80} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-log", action="append", required=True, help="grid_log.json from MC comparison; repeatable")
    parser.add_argument("--reference-gp", default=str(DEFAULT_REFERENCE_GP))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--only-task", action="append", default=[])
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()}
    invalid = formats.difference({"png", "pdf"})
    if invalid:
        raise SystemExit(f"Unsupported format(s): {sorted(invalid)}")
    if not formats:
        formats = {"png"}

    grid_logs = [Path(p).expanduser() for p in args.grid_log]
    reference_gp = Path(args.reference_gp).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reference_gp.is_file():
        raise SystemExit(f"Reference GP not found: {reference_gp}")
    for path in grid_logs:
        if not path.is_file():
            raise SystemExit(f"Grid log not found: {path}")

    gp = _load_gp(reference_gp)
    records, skipped = _load_records(grid_logs, gp, set(args.only_task))
    if not records:
        raise SystemExit("No complete runs found to plot")

    all_rows: list[dict[str, Any]] = []
    for task in sorted({record.task for record in records}):
        task_runs = [record for record in records if record.task == task]
        all_rows.extend(_plot_task(task, task_runs, gp, out_dir, formats))

    _write_metrics(out_dir / "mc_run_metrics.csv", all_rows)
    _write_summary(out_dir / "mc_summary.md", all_rows, skipped)

    print(f"Plotted {len(all_rows)} runs into {out_dir}")
    print(f"Wrote metrics: {out_dir / 'mc_run_metrics.csv'}")
    print(f"Wrote summary: {out_dir / 'mc_summary.md'}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete log entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
