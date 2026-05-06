#!/usr/bin/env python3
"""Create a paper-style diagnostic figure for a visibility-unaware C1 run."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

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
    _draw_world,
    _load_csv_rows,
    _load_gp,
    _load_json,
    _mean_csv_value,
    _pf,
    _query_rho,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "na"


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _path_progress(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.asarray([], dtype=float)
    ds = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.zeros_like(s)
    return s / total


def _cov_ellipse_params(cov_xx: float, cov_xy: float, cov_yy: float, *, nsigma: float = 3.0):
    cov = np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
    width = 2.0 * nsigma * math.sqrt(float(vals[0]))
    height = 2.0 * nsigma * math.sqrt(float(vals[1]))
    return width, height, angle


def _load_plans(path: Path) -> dict[float, np.ndarray]:
    grouped: dict[float, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[float(row["plan_stamp"])].append(
                (int(row["point_idx"]), float(row["x"]), float(row["y"]))
            )
    return {
        stamp: np.asarray([(x, y) for _, x, y in sorted(points)], dtype=float)
        for stamp, points in grouped.items()
        if points
    }


def _nearest_row_index(stamps: np.ndarray, target_stamp: float) -> int:
    finite = np.isfinite(stamps)
    if not finite.any():
        return 0
    idxs = np.nonzero(finite)[0]
    best = idxs[int(np.argmin(np.abs(stamps[finite] - target_stamp)))]
    return int(best)


def _latest_plan_before(plans: dict[float, np.ndarray], stamp: float) -> tuple[float, np.ndarray] | None:
    candidates = [s for s in plans if s <= stamp]
    if not candidates:
        return None
    s = max(candidates)
    return s, plans[s]


def _recent_detection_points(perception_rows: list[dict[str, str]], target_stamp: float, window_s: float = 2.0):
    pts = []
    hits = misses = applied = 0
    for row in perception_rows:
        stamp = _pf(row, "log_stamp")
        if not math.isfinite(stamp) or stamp < target_stamp - window_s or stamp > target_stamp:
            continue
        detected = _pf(row, "yolo_detected_after_threshold") > 0.5
        pix_ok = _pf(row, "pixel_pose_available") > 0.5
        x = _pf(row, "pred_world_x")
        y = _pf(row, "pred_world_y")
        if detected:
            hits += 1
        else:
            misses += 1
        if pix_ok:
            applied += 1
        if detected and math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
    return np.asarray(pts, dtype=float), hits, misses, applied


def _select_snapshot_stamps(
    rows: list[dict[str, str]],
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    progress: np.ndarray,
    rhos: np.ndarray,
    first_cmd_stamp: float,
    early_offset_s: float,
) -> tuple[float, float]:
    stamps = np.asarray([_pf(r, "stamp") for r in rows], dtype=float)
    early_target = first_cmd_stamp + early_offset_s
    early_stamp = float(stamps[_nearest_row_index(stamps, early_target)])

    best_idx = None
    best_age = -math.inf
    for idx, row in enumerate(rows):
        tx = _pf(row, "truth_x")
        ty = _pf(row, "truth_y")
        if not (_pf(row, "truth_available") > 0.5 and math.isfinite(tx) and math.isfinite(ty)):
            continue
        nearest = int(np.argmin((truth_x - tx) ** 2 + (truth_y - ty) ** 2))
        in_low = bool(math.isfinite(rhos[nearest]) and rhos[nearest] < RHO_LOW_THRESHOLD)
        age = _pf(row, "planner_pixel_correction_age_s")
        if in_low and math.isfinite(age) and age > best_age:
            best_age = age
            best_idx = idx
    if best_idx is None:
        ages = np.asarray([_pf(r, "planner_pixel_correction_age_s") for r in rows], dtype=float)
        finite = np.isfinite(ages)
        late_stamp = float(stamps[np.argmax(np.where(finite, ages, -1.0))]) if finite.any() else early_stamp
    else:
        late_stamp = float(stamps[best_idx])
    return early_stamp, late_stamp


def _snapshot_annotation(row: dict[str, str], hits: int, misses: int, applied: int) -> str:
    return "\n".join(
        [
            f"belief err: {_pf(row, 'truth_belief_error_m'):.3f} m",
            f"state err: {_pf(row, 'truth_state_error_m'):.3f} m",
            f"belief age: {_pf(row, 'belief_age_s'):.2f} s",
            f"corr age: {_pf(row, 'planner_pixel_correction_age_s'):.2f} s",
            f"meas avail: {int(_pf(row, 'measurement_available') > 0.5)}",
            f"hits/miss: {hits}/{misses}",
            f"pixel poses: {applied}",
            f"p_vis_eff: {_pf(row, 'p_vis_plan_eff'):.2f}",
        ]
    )


def _plot_snapshot(
    ax,
    *,
    title: str,
    stamp: float,
    run_rows: list[dict[str, str]],
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    plans: dict[float, np.ndarray],
    perception_rows: list[dict[str, str]],
    gp,
    manifest,
):
    im = _draw_world(ax, gp, manifest)
    row = run_rows[_nearest_row_index(np.asarray([_pf(r, "stamp") for r in run_rows], dtype=float), stamp)]
    idx = _nearest_row_index(np.asarray([_pf(r, "stamp") for r in run_rows], dtype=float), stamp)
    current_truth = np.asarray([_pf(row, "truth_x"), _pf(row, "truth_y")], dtype=float)
    current_belief = np.asarray([_pf(row, "planner_belief_x"), _pf(row, "planner_belief_y")], dtype=float)
    plan_item = _latest_plan_before(plans, stamp)
    recent_pts, hits, misses, applied = _recent_detection_points(perception_rows, stamp)

    # recent truth tail
    tail = np.column_stack([truth_x, truth_y])
    truth_stamp_arr = np.asarray([_pf(r, "stamp") for r in run_rows if _pf(r, "truth_available") > 0.5], dtype=float)
    if truth_stamp_arr.size:
        tail_mask = truth_stamp_arr <= stamp
        ax.plot(tail[tail_mask, 0], tail[tail_mask, 1], color="#111827", linewidth=2.2, zorder=12, label="truth")
    if np.all(np.isfinite(current_truth)):
        ax.scatter([current_truth[0]], [current_truth[1]], s=80, facecolor="#111827", edgecolors="white", linewidths=0.8, zorder=13)
    if np.all(np.isfinite(current_belief)):
        ax.scatter([current_belief[0]], [current_belief[1]], s=95, facecolor="#7c3aed", edgecolors="white", linewidths=0.8, zorder=14, label="belief")

    if recent_pts.size:
        ax.scatter(recent_pts[:, 0], recent_pts[:, 1], s=18, facecolor="#f59e0b", edgecolors="black",
                   linewidths=0.3, alpha=0.9, zorder=11, label="recent detections")

    if plan_item is not None:
        _, pts = plan_item
        ax.plot(pts[:, 0], pts[:, 1], color="#dc5f4b", linewidth=2.0, zorder=10, label="current horizon")

    ellipse = _cov_ellipse_params(_pf(row, "planner_cov_x"), _pf(row, "planner_cov_xy"), _pf(row, "planner_cov_y"), nsigma=3.0)
    if ellipse is not None and np.all(np.isfinite(current_belief)):
        width, height, angle = ellipse
        ax.add_patch(Ellipse(
            xy=(float(current_belief[0]), float(current_belief[1])),
            width=width,
            height=height,
            angle=angle,
            facecolor="#a855f7",
            edgecolor="#7c3aed",
            alpha=0.25,
            linewidth=1.2,
            zorder=9,
        ))

    goal_x = _mean_csv_value(run_rows[:10], "goal_x")
    goal_y = _mean_csv_value(run_rows[:10], "goal_y")
    if math.isfinite(goal_x) and math.isfinite(goal_y):
        ax.scatter([goal_x], [goal_y], marker="o", s=85, facecolor="#ef4444", edgecolors="black", linewidths=0.8, zorder=15)
        ax.add_patch(Circle((goal_x, goal_y), GOAL_SUCCESS_RADIUS_M, fill=False, edgecolor="#ef4444", linestyle="--", linewidth=1.2))

    ax.text(
        0.03,
        0.97,
        _snapshot_annotation(row, hits, misses, applied),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#d1d5db"),
    )
    ax.set_title(title, fontsize=12)
    return im


def _timeline_plot(ax, rows: list[dict[str, str]], first_cmd_stamp: float):
    stamps = np.asarray([_pf(r, "stamp") for r in rows], dtype=float)
    t = stamps - first_cmd_stamp
    belief_err = np.asarray([_pf(r, "truth_belief_error_m") for r in rows], dtype=float)
    corr_age = np.asarray([_pf(r, "planner_pixel_correction_age_s") for r in rows], dtype=float)
    meas_avail = np.asarray([_pf(r, "measurement_available") for r in rows], dtype=float)
    applied = np.asarray([_pf(r, "planner_pixel_correction_available") for r in rows], dtype=float)
    p_vis_eff = np.asarray([_pf(r, "p_vis_plan_eff") for r in rows], dtype=float)
    low_frac = np.asarray([_pf(r, "fraction_horizon_low_pvis") for r in rows], dtype=float)

    finite = np.isfinite(t)
    ax.plot(t[finite], belief_err[finite], color="#111827", linewidth=2.0, label="truth-belief error (m)")
    ax.plot(t[finite], corr_age[finite], color="#7c3aed", linewidth=1.8, label="pixel-correction age (s)")
    ax.plot(t[finite], p_vis_eff[finite], color="#0ea5e9", linewidth=1.4, label="p_vis_plan_eff")
    ax.plot(t[finite], low_frac[finite], color="#dc5f4b", linewidth=1.4, label="fraction horizon low-pvis")

    hit_t = t[(meas_avail > 0.5) & np.isfinite(t)]
    miss_t = t[(meas_avail <= 0.5) & np.isfinite(t)]
    if hit_t.size:
        ax.scatter(hit_t, np.full_like(hit_t, -0.12), s=14, color="#16a34a", marker="|", label="meas available", zorder=10)
    if miss_t.size:
        ax.scatter(miss_t, np.full_like(miss_t, -0.22), s=14, color="#b45309", marker="|", label="meas unavailable", zorder=10)
    corr_t = t[(applied > 0.5) & np.isfinite(t)]
    if corr_t.size:
        ax.scatter(corr_t, np.full_like(corr_t, -0.32), s=10, color="#2563eb", marker="o", label="pixel correction diag", zorder=10)

    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle=":")
    ax.set_xlabel("time after first command (s)")
    ax.set_ylabel("diagnostic values")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.92)


def _plot_run(run_dir: Path, gp, out_dir: Path, formats: set[str], early_offset_s: float) -> Path:
    summary = _load_json(run_dir / "run_summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    rows = _load_csv_rows(run_dir / "experiment.csv")
    perception_rows = _load_csv_rows(run_dir / "perception.csv")
    plans = _load_plans(run_dir / "plan_samples.csv")
    if not rows:
        raise RuntimeError(f"No experiment rows in {run_dir}")

    truth_rows = [r for r in rows if _pf(r, "truth_available") > 0.5 and math.isfinite(_pf(r, "truth_x")) and math.isfinite(_pf(r, "truth_y"))]
    truth_x = np.asarray([_pf(r, "truth_x") for r in truth_rows], dtype=float)
    truth_y = np.asarray([_pf(r, "truth_y") for r in truth_rows], dtype=float)
    progress = _path_progress(truth_x, truth_y)
    rhos = np.asarray([_query_rho(gp, x, y) for x, y in zip(truth_x, truth_y)], dtype=float)
    first_cmd_stamp = float(summary.get("first_cmd_stamp", math.nan))
    if not math.isfinite(first_cmd_stamp):
        first_cmd_stamp = float(_pf(rows[0], "stamp"))

    early_stamp, late_stamp = _select_snapshot_stamps(
        rows, truth_x, truth_y, progress, rhos, first_cmd_stamp, early_offset_s
    )

    fig = plt.figure(figsize=(15.8, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.25, 1.15], hspace=0.16, wspace=0.1)
    ax_setup = fig.add_subplot(gs[0, 0])
    ax_early = fig.add_subplot(gs[0, 1])
    ax_late = fig.add_subplot(gs[0, 2])
    ax_timeline = fig.add_subplot(gs[1, :])

    im = _draw_world(ax_setup, gp, manifest)
    cbar = fig.colorbar(im, ax=[ax_setup, ax_early, ax_late], fraction=0.022, pad=0.02)
    cbar.set_label("nominal raw-GP rho_plan")
    goal_x = _mean_csv_value(rows[:10], "goal_x")
    goal_y = _mean_csv_value(rows[:10], "goal_y")
    if math.isfinite(goal_x) and math.isfinite(goal_y):
        ax_setup.scatter([goal_x], [goal_y], marker="o", s=95, facecolor="#ef4444", edgecolors="black", linewidths=0.8, zorder=16)
        ax_setup.add_patch(Circle((goal_x, goal_y), GOAL_SUCCESS_RADIUS_M, fill=False, edgecolor="#ef4444", linestyle="--", linewidth=1.2))
    ax_setup.scatter([truth_x[0]], [truth_y[0]], s=110, facecolor="#22c55e", edgecolors="black", linewidths=0.8, zorder=16)
    ax_setup.plot(truth_x, truth_y, color="#111827", linewidth=2.7, zorder=14, label="truth path")
    ax_setup.set_title("(a) setup and full truth path", fontsize=12)
    ax_setup.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    _plot_snapshot(
        ax_early,
        title=f"(b) early direct-rollout\n t={early_stamp-first_cmd_stamp:.1f}s",
        stamp=early_stamp,
        run_rows=rows,
        truth_x=truth_x,
        truth_y=truth_y,
        plans=plans,
        perception_rows=perception_rows,
        gp=gp,
        manifest=manifest,
    )
    _plot_snapshot(
        ax_late,
        title=f"(c) stale camera-update interval\n t={late_stamp-first_cmd_stamp:.1f}s",
        stamp=late_stamp,
        run_rows=rows,
        truth_x=truth_x,
        truth_y=truth_y,
        plans=plans,
        perception_rows=perception_rows,
        gp=gp,
        manifest=manifest,
    )

    _timeline_plot(ax_timeline, rows, first_cmd_stamp)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / _safe_name(f"{run_dir.parent.parent.name}_{run_dir.parent.name}_visibility_unaware_diagnostic")
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=190 if fmt == "png" else None)
    plt.close(fig)
    return out_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-gp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--early-offset-s", type=float, default=2.0)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    gp = _load_gp(Path(args.reference_gp).expanduser())
    formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()} or {"png"}
    out = _plot_run(
        Path(args.run_dir).expanduser(),
        gp,
        Path(args.out_dir).expanduser(),
        formats,
        early_offset_s=float(args.early_offset_s),
    )
    print(f"Wrote visibility-unaware diagnostic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
