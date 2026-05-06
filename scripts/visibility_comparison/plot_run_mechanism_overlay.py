#!/usr/bin/env python3
"""Single-run mechanism overlay plot for visibility-comparison experiments.

The figure is meant to answer the simple visual question:
"What did the detector see, where were fresh belief updates applied, and how
did truth and belief evolve over the visibility field?"
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from plot_individual_model_selection_runs import (
    GOAL_SUCCESS_RADIUS_M,
    RHO_LOW_THRESHOLD,
    _draw_belief_with_uncertainty,
    _draw_truth,
    _draw_world,
    _load_csv_rows,
    _load_gp,
    _load_json,
    _mean_csv_value,
    _pf,
)

FRESH_UPDATE_MAX_AGE_S = 0.50


def _load_plan_samples(path: Path) -> list[dict[str, str]]:
    p = path / "plan_samples.csv"
    if not p.is_file():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _truth_points(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        if _pf(row, "truth_available") < 0.5:
            continue
        x = _pf(row, "truth_x")
        y = _pf(row, "truth_y")
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _belief_with_updates(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts, xs, ys, sxx, sxy, syy, fresh = [], [], [], [], [], [], []
    seen_stamp = None
    for row in rows:
        if _pf(row, "planner_belief_available") < 0.5:
            continue
        b_stamp = _pf(row, "planner_belief_stamp")
        if not math.isfinite(b_stamp):
            continue
        if seen_stamp is not None and b_stamp == seen_stamp:
            continue
        seen_stamp = b_stamp
        x = _pf(row, "planner_belief_x")
        y = _pf(row, "planner_belief_y")
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        age = _pf(row, "planner_pixel_correction_age_s")
        applied = _pf(row, "planner_pixel_correction_available")
        ts.append(_pf(row, "stamp"))
        xs.append(x)
        ys.append(y)
        sxx.append(_pf(row, "planner_cov_x"))
        sxy.append(_pf(row, "planner_cov_xy"))
        syy.append(_pf(row, "planner_cov_y"))
        is_fresh = (
            math.isfinite(applied)
            and applied >= 0.5
            and math.isfinite(age)
            and age <= FRESH_UPDATE_MAX_AGE_S
        )
        fresh.append(1.0 if is_fresh else 0.0)
    return (
        np.asarray(ts, dtype=float),
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(sxx, dtype=float),
        np.asarray(sxy, dtype=float),
        np.asarray(syy, dtype=float),
        np.asarray(fresh, dtype=float),
    )


def _perception_points(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        detected = _pf(row, "detected") >= 0.5 or _pf(row, "yolo_detected_after_threshold") >= 0.5
        x = _pf(row, "pred_world_x")
        y = _pf(row, "pred_world_y")
        if detected and math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _group_plan_snapshots(plan_rows: list[dict[str, str]]) -> list[np.ndarray]:
    groups: dict[float, list[tuple[int, float, float]]] = {}
    for row in plan_rows:
        stamp = _pf(row, "plan_stamp")
        idx = _pf(row, "point_idx")
        x = _pf(row, "x")
        y = _pf(row, "y")
        if not (math.isfinite(stamp) and math.isfinite(idx) and math.isfinite(x) and math.isfinite(y)):
            continue
        groups.setdefault(stamp, []).append((int(idx), float(x), float(y)))
    ordered = []
    for stamp in sorted(groups.keys()):
        pts = [(x, y) for _, x, y in sorted(groups[stamp], key=lambda item: item[0])]
        if len(pts) >= 2:
            ordered.append(np.asarray(pts, dtype=float))
    return ordered


def _draw_plan_horizon_snapshots(ax, plans: list[np.ndarray], *, max_snapshots: int = 6) -> None:
    if not plans:
        return
    idx = np.linspace(0, len(plans) - 1, min(max_snapshots, len(plans))).astype(int)
    first = True
    for i in idx:
        pts = plans[int(i)]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#dc5f4b",
            linewidth=1.1,
            alpha=0.32,
            zorder=5,
            label="planned horizon" if first else None,
        )
        first = False


def _format_float(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if math.isfinite(v) else "nan"


def plot_run(run_dir: Path, gp_path: Path, out_dir: Path) -> Path:
    rows = _load_csv_rows(run_dir / "experiment.csv")
    if not rows:
        raise SystemExit(f"No experiment.csv found in {run_dir}")
    perception_rows = _load_csv_rows(run_dir / "perception.csv")
    manifest = _load_json(run_dir / "run_manifest.json")
    summary = _load_json(run_dir / "run_summary.json")
    plans = _group_plan_snapshots(_load_plan_samples(run_dir))
    gp = _load_gp(gp_path)

    tx, ty = _truth_points(rows)
    bt, bx, by, bsxx, bsxy, bsyy, fresh = _belief_with_updates(rows)
    px, py = _perception_points(perception_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    label = run_dir.parent.name if run_dir.parent.name.startswith("seed") else run_dir.name
    stem = f"{run_dir.parent.parent.name if run_dir.parent.parent.exists() else 'run'}_{label}_mechanism_overlay"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    fig = plt.figure(figsize=(8.5, 7.8))
    ax = fig.add_subplot(1, 1, 1)
    im = _draw_world(ax, gp, manifest)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("nominal raw-GP rho_plan")

    _draw_plan_horizon_snapshots(ax, plans)
    if bx.size:
        _draw_belief_with_uncertainty(ax, bx, by, bsxx, bsxy, bsyy)
        upd = fresh >= 0.5
        if np.any(upd):
            ax.scatter(
                bx[upd], by[upd],
                s=22, facecolor="white", edgecolor="#2563eb", linewidths=0.9,
                zorder=10, alpha=0.95, label="fresh camera update",
            )
    if tx.size:
        _draw_truth(ax, tx, ty)
    if px.size:
        step = max(1, int(math.ceil(px.size / 120)))
        ax.scatter(
            px[::step], py[::step],
            c="#ff9f43", s=15, alpha=0.88,
            edgecolors="black", linewidths=0.35,
            zorder=9, label="YOLO world point",
        )

    if tx.size:
        ax.scatter(
            [tx[0]], [ty[0]],
            marker="o", s=170,
            facecolor="#22c55e", edgecolors="black", linewidths=1.4,
            zorder=12, label="start",
        )
        ax.scatter(
            [tx[-1]], [ty[-1]],
            marker="X", s=120,
            facecolor="#f59e0b", edgecolors="black", linewidths=1.0,
            zorder=12, label="truth end",
        )

    goal_x = _mean_csv_value(rows[:10], "goal_x")
    goal_y = _mean_csv_value(rows[:10], "goal_y")
    if math.isfinite(goal_x) and math.isfinite(goal_y):
        ax.add_patch(Circle(
            (goal_x, goal_y), GOAL_SUCCESS_RADIUS_M,
            fill=False, edgecolor="#ef4444", linewidth=2.0, linestyle="--",
            zorder=11,
        ))
        ax.scatter(
            [goal_x], [goal_y],
            marker="*", s=320,
            facecolor="#ef4444", edgecolors="black", linewidths=1.2,
            zorder=13, label="goal",
        )

    low_rho = np.asarray([
        gp.interp([[float(y), float(x)]])[0] if math.isfinite(x) and math.isfinite(y) else math.nan
        for x, y in zip(tx, ty)
    ], dtype=float)
    finite_low = low_rho[np.isfinite(low_rho)]
    low_exposure = float(np.mean(finite_low < RHO_LOW_THRESHOLD)) if finite_low.size else math.nan

    info = [
        f"fresh updates: {int(np.sum(fresh >= 0.5))}",
        f"detections: {int(px.size)}",
        f"low-rho exposure: {_format_float(low_exposure)}",
        f"mean belief err: {_format_float(float(summary.get('mean_truth_belief_error_m', math.nan)))} m",
        f"min goal: {_format_float(float(summary.get('minimum_goal_distance', math.nan)))} m",
        f"elapsed: {_format_float(float(summary.get('elapsed_after_first_cmd_s', math.nan)), 1)} s",
    ]
    ax.text(
        0.015,
        0.985,
        "\n".join(info),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.6,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.82, "pad": 4.0},
        zorder=20,
    )

    title_bits = [
        str(manifest.get("task", run_dir.parent.parent.name if run_dir.parent.parent.exists() else "task")),
        str(manifest.get("planner", "planner")),
        str(run_dir.parent.name),
        str(summary.get("completion_reason", summary.get("outcome", ""))),
    ]
    ax.set_title(" | ".join(title_bits), fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(
            seen.values(), seen.keys(),
            loc="lower left", fontsize=8.1,
            framealpha=0.93, facecolor="white", edgecolor="black",
        )

    fig.tight_layout()
    fig.savefig(png_path, dpi=190)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Experiment run directory containing experiment.csv")
    parser.add_argument(
        "--reference-gp",
        default="logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz",
        help="Reference GP artifact for the background rho_plan field",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/visibility_comparison/paper_appendix_mechanism_overlays_v1",
        help="Output directory for the rendered figure",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    gp_path = Path(args.reference_gp).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if not gp_path.is_file():
        raise SystemExit(f"Reference GP not found: {gp_path}")
    png_path = plot_run(run_dir, gp_path, out_dir)
    print(png_path)


if __name__ == "__main__":
    main()
