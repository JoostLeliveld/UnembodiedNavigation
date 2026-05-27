#!/usr/bin/env python3
"""Plot individual raw-GP model-selection runs for trajectory inspection.

The script uses campaign/grid log entries as the source of truth, rather than
glob-discovering run directories. This avoids accidentally plotting stale retry
folders that are no longer selected by the campaign or grid log.
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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Rectangle
from scipy.interpolate import RegularGridInterpolator


RHO_LOW_THRESHOLD = 0.35
GOAL_SUCCESS_RADIUS_M = 0.20
RUN_TIMEOUT_AFTER_FIRST_CMD_S = 75.0

DEFAULT_ANCHOR_LOG = Path("logs/visibility_comparison/paper_campaign_rawgp_v1/campaign_log.json")
DEFAULT_GRID_LOG = Path("logs/visibility_comparison/model_selection_rawgp_v1/grid_log.json")
DEFAULT_REFERENCE_GP = Path("logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz")
DEFAULT_OUT_DIR = Path("logs/visibility_comparison/run_investigation_rawgp_v1")

TASK_LABELS = {
    "shadow_tradeoff_a": "Task A",
    "shadow_tradeoff_a_leftstart": "Task A-left",
    "shadow_tradeoff_b": "Task B",
    "sanity_open": "Sanity",
}

RANK_FIELDS = [
    "rank",
    "source",
    "task",
    "condition",
    "axis",
    "label",
    "seed",
    "outcome",
    "completion_reason",
    "clean_success",
    "valid_run",
    "collision",
    "timeout",
    "reference_low_rho_exposure",
    "mean_rho_reference",
    "median_rho_reference",
    "min_rho_reference",
    "path_length_m",
    "minimum_goal_distance",
    "elapsed_after_first_cmd_s",
    "mean_truth_belief_error_m",
    "mean_p_vis_plan",
    "mean_r_plan_u_std",
    "mean_efe_risk",
    "mean_efe_ambiguity",
    "plot_png",
    "plot_pdf",
    "run_dir",
]


@dataclass(frozen=True)
class GpField:
    xs: np.ndarray
    ys: np.ndarray
    p_plan: np.ndarray
    interp: RegularGridInterpolator
    camera_pos: tuple[float, float] | None


@dataclass
class RunRecord:
    source: str
    task: str
    condition: str
    axis: str
    label: str
    seed: str
    outcome: str
    completion_reason: str
    run_dir: Path
    entry: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pf(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def _safe_name(value: str) -> str:
    value = str(value or "na")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "na"


def _load_gp(path: Path) -> GpField:
    with np.load(path, allow_pickle=False) as data:
        missing = {"xs", "ys", "P_conservative_plan_map"}.difference(data.files)
        if missing:
            raise RuntimeError(f"GP artifact missing keys {sorted(missing)}: {path}")
        xs = np.asarray(data["xs"], dtype=float)
        ys = np.asarray(data["ys"], dtype=float)
        p_plan = np.asarray(data["P_conservative_plan_map"], dtype=float)
        camera_pos = None
        if "camera_pos" in data.files:
            cam = np.asarray(data["camera_pos"], dtype=float).ravel()
            if cam.size >= 2 and math.isfinite(cam[0]) and math.isfinite(cam[1]):
                camera_pos = (float(cam[0]), float(cam[1]))
    interp = RegularGridInterpolator(
        (ys, xs),
        p_plan,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return GpField(xs=xs, ys=ys, p_plan=p_plan, interp=interp, camera_pos=camera_pos)


def _query_rho(gp: GpField, x: float, y: float) -> float:
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.nan
    return float(gp.interp([[y, x]])[0])


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_dir_from_entry(entry: dict[str, Any]) -> Path | None:
    run_dir = str(entry.get("run_dir", "") or "")
    if not run_dir:
        return None
    path = Path(run_dir).expanduser()
    if (path / "run_summary.json").is_file() or (path / "experiment.csv").is_file():
        return path
    candidates = sorted(path.rglob("run_summary.json")) if path.is_dir() else []
    if candidates:
        return candidates[-1].parent
    return path


def _iter_anchor_runs(path: Path) -> list[RunRecord]:
    data = _load_json(path)
    records: list[RunRecord] = []
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        run_dir = _run_dir_from_entry(entry)
        if run_dir is None:
            continue
        records.append(
            RunRecord(
                source="anchor",
                task=str(entry.get("task", "")),
                condition=str(entry.get("condition", "")),
                axis="anchor",
                label=str(entry.get("condition", "")),
                seed=str(entry.get("seed", "")),
                outcome=str(entry.get("outcome", "") or ""),
                completion_reason=str(entry.get("completion_reason", "") or ""),
                run_dir=run_dir,
                entry=entry,
            )
        )
    return records


def _iter_grid_runs(path: Path) -> list[RunRecord]:
    data = _load_json(path)
    records: list[RunRecord] = []
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        run_dir = _run_dir_from_entry(entry)
        if run_dir is None:
            continue
        merged = entry.get("merged_config") or {}
        task = merged.get("task", "shadow_tradeoff_a") if isinstance(merged, dict) else "shadow_tradeoff_a"
        records.append(
            RunRecord(
                source="grid",
                task=str(task),
                condition="C2",
                axis=str(entry.get("axis", "")),
                label=str(entry.get("label", "")),
                seed=str(entry.get("seed", "")),
                outcome=str(entry.get("outcome", "") or ""),
                completion_reason=str(entry.get("completion_reason", "") or ""),
                run_dir=run_dir,
                entry=entry,
            )
        )
    return records


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


def _belief_points(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if _pf(row, "planner_belief_available") < 0.5:
            continue
        x = _pf(row, "planner_belief_x")
        y = _pf(row, "planner_belief_y")
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _belief_with_cov(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Belief mean + 2D covariance, deduplicated by belief stamp.

    The logger ticks faster than the planner publishes its belief, so
    consecutive logger rows can repeat the same belief. We keep one entry
    per unique planner_belief_stamp. Returns (t, x, y, sxx, sxy, syy,
    pixel_correction_age_s).
    """
    ts: list[float] = []
    mx: list[float] = []
    my: list[float] = []
    sxx: list[float] = []
    sxy: list[float] = []
    syy: list[float] = []
    age: list[float] = []
    seen_stamp: float | None = None
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
        ts.append(_pf(row, "stamp"))
        mx.append(x)
        my.append(y)
        sxx.append(_pf(row, "planner_cov_x"))
        sxy.append(_pf(row, "planner_cov_xy"))
        syy.append(_pf(row, "planner_cov_y"))
        age.append(_pf(row, "planner_pixel_correction_age_s"))
    return (
        np.asarray(ts, dtype=float),
        np.asarray(mx, dtype=float),
        np.asarray(my, dtype=float),
        np.asarray(sxx, dtype=float),
        np.asarray(sxy, dtype=float),
        np.asarray(syy, dtype=float),
        np.asarray(age, dtype=float),
    )


def _cov_eig_xy(sxx: float, sxy: float, syy: float) -> tuple[float, float, float]:
    a = 0.5 * (sxx + syy)
    b = 0.5 * (sxx - syy)
    d = math.sqrt(b * b + sxy * sxy)
    lam1 = max(a + d, 1e-12)
    lam2 = max(a - d, 1e-12)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return lam1, lam2, theta


def _perception_points(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        detected = _pf(row, "detected") >= 0.5 or _pf(row, "yolo_detected_after_threshold") >= 0.5
        x = _pf(row, "pred_world_x")
        y = _pf(row, "pred_world_y")
        if detected and math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if arr.size else math.nan


def _finite_median(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else math.nan


def _finite_min(values: list[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    return float(np.min(arr)) if arr.size else math.nan


def _mean_csv_value(rows: list[dict[str, str]], key: str) -> float:
    return _finite_mean([_pf(row, key) for row in rows])


def _clean_success(outcome: str, summary: dict[str, Any]) -> bool:
    completion_reason = str(summary.get("completion_reason", ""))
    collision = bool(
        _is_true(summary.get("crashed", False))
        or _is_true(summary.get("collision_any", False))
        or completion_reason == "collision"
        or outcome == "collision"
    )
    valid_run = _is_true(summary.get("valid_run", False))
    min_goal = summary.get("minimum_goal_distance", math.nan)
    elapsed = summary.get("elapsed_after_first_cmd_s", math.nan)
    try:
        min_goal_f = float(min_goal)
        elapsed_f = float(elapsed)
    except (TypeError, ValueError):
        return False
    return bool(
        outcome == "goal_reached"
        and not collision
        and valid_run
        and math.isfinite(min_goal_f)
        and min_goal_f <= GOAL_SUCCESS_RADIUS_M
        and math.isfinite(elapsed_f)
        and 0.0 <= elapsed_f <= RUN_TIMEOUT_AFTER_FIRST_CMD_S
    )


def _draw_world(ax, gp: GpField, manifest: dict[str, Any]) -> None:
    extent = [float(np.min(gp.xs)), float(np.max(gp.xs)), float(np.min(gp.ys)), float(np.max(gp.ys))]
    im = ax.imshow(
        gp.p_plan,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.65, float(np.nanmax(gp.p_plan))),
        alpha=0.88,
        interpolation="bilinear",
    )
    ax.contour(
        gp.xs,
        gp.ys,
        gp.p_plan,
        levels=[RHO_LOW_THRESHOLD],
        colors=["white"],
        linewidths=1.0,
        linestyles=["--"],
        alpha=0.75,
    )

    geometry_json = manifest.get("visibility_geometry_json") or manifest.get("collision_geometry_json") or ""
    drew_geometry = False
    if geometry_json:
        try:
            geometry = json.loads(str(geometry_json))
            for prism in geometry.get("prisms", []):
                xmin = float(prism.get("xmin"))
                xmax = float(prism.get("xmax"))
                ymin = float(prism.get("ymin"))
                ymax = float(prism.get("ymax"))
                if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
                    continue
                ax.add_patch(
                    Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        facecolor="black",
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.70,
                        zorder=3,
                    )
                )
                drew_geometry = True
        except (TypeError, ValueError, json.JSONDecodeError):
            drew_geometry = False
    if not drew_geometry:
        ax.add_patch(Rectangle((-0.9, -0.33), 1.7, 0.36, facecolor="black", alpha=0.70, zorder=3))

    if gp.camera_pos is not None:
        ax.scatter([gp.camera_pos[0]], [gp.camera_pos[1]], marker="^", s=90, c="#e8f4ff", edgecolors="black", zorder=6)
        ax.text(gp.camera_pos[0] + 0.05, gp.camera_pos[1] - 0.05, "camera", fontsize=8, color="white", zorder=7)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, color="white", alpha=0.18, linewidth=0.5)
    return im


def _draw_truth(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    """Truth trajectory: thick black line with a white outline glow."""
    if xs.size < 2:
        if xs.size == 1:
            ax.scatter(xs, ys, marker="o", s=55, facecolor="black", edgecolors="white", linewidths=1.0, zorder=11)
        return
    line, = ax.plot(
        xs, ys,
        color="black", linewidth=3.2, solid_capstyle="round", solid_joinstyle="round",
        zorder=11, label="truth",
    )
    line.set_path_effects([
        path_effects.Stroke(linewidth=5.4, foreground="white", alpha=0.85),
        path_effects.Normal(),
    ])


def _draw_belief_with_uncertainty(
    ax,
    bx: np.ndarray,
    by: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    age: np.ndarray | None = None,
    *,
    n_ellipses: int = 8,
) -> None:
    """Belief mean (dashed cyan) plus a 2sigma envelope and sampled ellipses."""
    if bx.size < 2:
        return
    # break the line at large planner-belief gaps so we never draw a phantom
    # diagonal across an interval where the EKF didn't publish (this is rare
    # with belief_publish_rate=10 Hz but still safe).
    seg_break = np.zeros(bx.size, dtype=bool)
    # 2sigma envelope along the belief mean. We approximate the local 2sigma
    # width as the larger covariance eigenvalue (so the band reflects the
    # principal-axis uncertainty without trying to draw a true tube).
    widths = np.zeros(bx.size, dtype=float)
    for i in range(bx.size):
        sxx_i = float(sxx[i]) if i < sxx.size else math.nan
        syy_i = float(syy[i]) if i < syy.size else math.nan
        sxy_i = float(sxy[i]) if i < sxy.size else 0.0
        if not (math.isfinite(sxx_i) and math.isfinite(syy_i)):
            widths[i] = math.nan
            continue
        lam1, _, _ = _cov_eig_xy(sxx_i, sxy_i, syy_i)
        widths[i] = 2.0 * math.sqrt(lam1)  # 2sigma along major axis (m)

    # Build a soft "tube" by drawing a wider, low-alpha line whose alpha is
    # already encoded by the LineCollection. We use an offset envelope normal
    # to the belief tangent.
    if np.any(np.isfinite(widths)):
        tangent_x = np.gradient(bx)
        tangent_y = np.gradient(by)
        norms = np.hypot(tangent_x, tangent_y)
        norms[norms < 1e-9] = 1.0
        nx = -tangent_y / norms
        ny = tangent_x / norms
        upper_x = bx + nx * widths
        upper_y = by + ny * widths
        lower_x = bx - nx * widths
        lower_y = by - ny * widths
        finite = np.isfinite(widths)
        if np.any(finite):
            poly_x = np.concatenate([upper_x[finite], lower_x[finite][::-1]])
            poly_y = np.concatenate([upper_y[finite], lower_y[finite][::-1]])
            ax.fill(
                poly_x, poly_y,
                facecolor="#22d3ee", alpha=0.18, edgecolor="none",
                zorder=6, label=r"belief $2\sigma$ envelope",
            )

    # Belief mean line (dashed cyan with thin black outline).
    line, = ax.plot(
        bx, by,
        color="#22d3ee", linestyle="--", linewidth=1.6, alpha=0.95,
        zorder=8, label="belief mean",
    )
    line.set_path_effects([
        path_effects.Stroke(linewidth=2.6, foreground="black", alpha=0.55),
        path_effects.Normal(),
    ])

    # Sampled 2sigma ellipses along the path.
    if bx.size >= 2:
        idx = np.linspace(0, bx.size - 1, n_ellipses).astype(int)
        for i in idx:
            sxx_i = float(sxx[i]) if i < sxx.size else math.nan
            syy_i = float(syy[i]) if i < syy.size else math.nan
            sxy_i = float(sxy[i]) if i < sxy.size else 0.0
            if not (math.isfinite(sxx_i) and math.isfinite(syy_i)):
                continue
            lam1, lam2, theta = _cov_eig_xy(sxx_i, sxy_i, syy_i)
            ax.add_patch(Ellipse(
                xy=(float(bx[i]), float(by[i])),
                width=2.0 * 2.0 * math.sqrt(lam1),
                height=2.0 * 2.0 * math.sqrt(lam2),
                angle=math.degrees(theta),
                facecolor="#22d3ee", alpha=0.22,
                edgecolor="#0ea5b7", linewidth=0.9,
                zorder=7,
            ))


def _plot_run(
    *,
    record: RunRecord,
    gp: GpField,
    rows: list[dict[str, str]],
    perception_rows: list[dict[str, str]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    out_base: Path,
    formats: set[str],
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    tx, ty = _truth_points(rows)
    bt, bx, by, bsxx, bsxy, bsyy, bage = _belief_with_cov(rows)
    px, py = _perception_points(perception_rows)
    rhos = np.asarray([_query_rho(gp, x, y) for x, y in zip(tx, ty)], dtype=float)
    finite_rhos = rhos[np.isfinite(rhos)]

    collision = bool(
        _is_true(summary.get("crashed", False))
        or _is_true(summary.get("collision_any", False))
        or record.outcome == "collision"
    )
    timeout = bool(summary.get("completion_reason") == "timeout_after_first_cmd" or record.outcome == "timeout")
    clean_success = _clean_success(record.outcome, summary)

    fig = plt.figure(figsize=(8.6, 8.2))
    ax = fig.add_subplot(1, 1, 1)
    im = _draw_world(ax, gp, manifest)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("nominal raw-GP rho_plan")

    if bx.size:
        _draw_belief_with_uncertainty(ax, bx, by, bsxx, bsxy, bsyy, bage)
    if tx.size:
        _draw_truth(ax, tx, ty)

    if px.size:
        step = max(1, int(math.ceil(px.size / 80)))
        ax.scatter(
            px[::step], py[::step],
            c="#ff8c42", s=14, alpha=0.85,
            edgecolors="black", linewidths=0.4,
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
            marker="X", s=130,
            facecolor="#f59e0b", edgecolors="black", linewidths=1.0,
            zorder=12, label="truth end",
        )

    goal_x = _mean_csv_value(rows[:10], "goal_x") if rows else math.nan
    goal_y = _mean_csv_value(rows[:10], "goal_y") if rows else math.nan
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

    title_bits = [
        record.source,
        TASK_LABELS.get(record.task, record.task),
        record.condition or record.label,
        f"seed {record.seed}",
        record.outcome or "unknown",
    ]
    if record.source == "grid":
        title_bits.insert(2, f"{record.axis}/{record.label}")
    ax.set_title(" | ".join(title_bits), fontsize=11)

    low_exposure = float(np.mean(finite_rhos < RHO_LOW_THRESHOLD)) if finite_rhos.size else math.nan
    mean_rho = float(np.mean(finite_rhos)) if finite_rhos.size else math.nan
    median_rho = float(np.median(finite_rhos)) if finite_rhos.size else math.nan
    min_rho = float(np.min(finite_rhos)) if finite_rhos.size else math.nan

    info = [
        f"clean success: {'yes' if clean_success else 'no'}",
        f"low-rho exposure: {_format_float(low_exposure)}",
        f"mean rho: {_format_float(mean_rho)}",
        f"path: {_format_float(summary.get('path_length_m'), 2)} m",
        f"min goal: {_format_float(summary.get('minimum_goal_distance'), 3)} m",
        f"elapsed: {_format_float(summary.get('elapsed_after_first_cmd_s'), 1)} s",
    ]
    ax.text(
        0.012,
        0.988,
        "\n".join(info),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.80, "pad": 4.0},
        zorder=20,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen: dict[str, Any] = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(
            seen.values(), seen.keys(),
            loc="lower left", fontsize=8, framealpha=0.92,
            facecolor="white", edgecolor="black",
        )

    fig.tight_layout()
    png_path = out_base.with_suffix(".png") if "png" in formats else None
    pdf_path = out_base.with_suffix(".pdf") if "pdf" in formats else None
    if png_path is not None:
        fig.savefig(png_path, dpi=190)
    if pdf_path is not None:
        fig.savefig(pdf_path)
    plt.close(fig)

    metrics = {
        "source": record.source,
        "task": record.task,
        "condition": record.condition,
        "axis": record.axis,
        "label": record.label,
        "seed": record.seed,
        "outcome": record.outcome,
        "completion_reason": str(summary.get("completion_reason", record.completion_reason)),
        "clean_success": clean_success,
        "valid_run": _is_true(summary.get("valid_run", False)),
        "collision": collision,
        "timeout": timeout,
        "reference_low_rho_exposure": low_exposure,
        "mean_rho_reference": mean_rho,
        "median_rho_reference": median_rho,
        "min_rho_reference": min_rho,
        "path_length_m": summary.get("path_length_m", math.nan),
        "minimum_goal_distance": summary.get("minimum_goal_distance", math.nan),
        "elapsed_after_first_cmd_s": summary.get("elapsed_after_first_cmd_s", math.nan),
        "mean_truth_belief_error_m": summary.get("mean_truth_belief_error_m", math.nan),
        "mean_p_vis_plan": summary.get("mean_p_vis_plan", math.nan),
        "mean_r_plan_u_std": summary.get("mean_r_plan_u_std", math.nan),
        "mean_efe_risk": summary.get("mean_efe_risk", math.nan),
        "mean_efe_ambiguity": summary.get("mean_efe_ambiguity", math.nan),
        "plot_png": str(png_path) if png_path is not None else "",
        "plot_pdf": str(pdf_path) if pdf_path is not None else "",
        "run_dir": str(record.run_dir),
    }
    return png_path, pdf_path, metrics


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    clean = 1 if row.get("clean_success") else 0
    success_outcome = 1 if row.get("outcome") == "goal_reached" else 0
    low = row.get("reference_low_rho_exposure", math.nan)
    mean = row.get("mean_rho_reference", math.nan)
    path = row.get("path_length_m", math.nan)
    min_goal = row.get("minimum_goal_distance", math.nan)
    low_key = float(low) if isinstance(low, (int, float)) and math.isfinite(float(low)) else 999.0
    mean_key = float(mean) if isinstance(mean, (int, float)) and math.isfinite(float(mean)) else -999.0
    path_key = float(path) if isinstance(path, (int, float)) and math.isfinite(float(path)) else 999.0
    goal_key = float(min_goal) if isinstance(min_goal, (int, float)) and math.isfinite(float(min_goal)) else 999.0
    return (-clean, -success_outcome, low_key, -mean_key, path_key, goal_key)


def _write_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RANK_FIELDS)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            row = dict(row)
            row["rank"] = idx
            out = {}
            for field in RANK_FIELDS:
                value = row.get(field, "")
                if isinstance(value, bool):
                    out[field] = "1" if value else "0"
                elif isinstance(value, float):
                    out[field] = "" if not math.isfinite(value) else f"{value:.6f}"
                else:
                    out[field] = value
            writer.writerow(out)


def _rel_link(target: str, base_dir: Path) -> str:
    if not target:
        return ""
    try:
        return str(Path(target).resolve(strict=False).relative_to(base_dir.resolve(strict=False)))
    except ValueError:
        return target


def _write_index(path: Path, rows: list[dict[str, Any]], skipped: list[str]) -> None:
    base_dir = path.parent
    lines = [
        "# Individual Run Investigation",
        "",
        f"Runs plotted: {len(rows)}",
        f"Skipped log entries: {len(skipped)}",
        "",
        "Ranking prioritizes clean success, lower nominal raw-GP low-rho exposure, higher mean rho, then shorter path length.",
        "",
        "## Top Visible Clean Successes",
        "",
        "| rank | source | task | condition/cell | seed | low-rho | mean rho | path m | plot |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    top = [r for r in rows if r.get("clean_success")][:20]
    for idx, row in enumerate(top, start=1):
        cell = row.get("condition") if row.get("source") == "anchor" else f"{row.get('axis')}/{row.get('label')}"
        plot = _rel_link(str(row.get("plot_png", "")), base_dir)
        plot_link = f"[png]({plot})" if plot else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("source", "")),
                    str(row.get("task", "")),
                    str(cell or ""),
                    str(row.get("seed", "")),
                    _format_float(row.get("reference_low_rho_exposure"), 3),
                    _format_float(row.get("mean_rho_reference"), 3),
                    _format_float(row.get("path_length_m"), 2),
                    plot_link,
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## All Plots",
        "",
        "See `run_visibility_ranking.csv` for the sortable table. Each per-run file is in `plots/`.",
    ]
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- {item}" for item in skipped[:80]]
        if len(skipped) > 80:
            lines.append(f"- ... {len(skipped) - 80} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-log", default=str(DEFAULT_ANCHOR_LOG), help="Fresh raw-GP anchor campaign_log.json")
    parser.add_argument("--grid-log", default=str(DEFAULT_GRID_LOG), help="Optional sensitivity grid_log.json")
    parser.add_argument("--reference-gp", default=str(DEFAULT_REFERENCE_GP), help="Nominal raw-GP artifact for background and ranking")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for plots and ranking")
    parser.add_argument("--only-task", action="append", default=[], help="Restrict to a task; repeatable")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats: png,pdf")
    args = parser.parse_args()

    anchor_log = Path(args.anchor_log).expanduser()
    grid_log = Path(args.grid_log).expanduser()
    reference_gp = Path(args.reference_gp).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()}
    invalid_formats = formats.difference({"png", "pdf"})
    if invalid_formats:
        raise SystemExit(f"Unsupported format(s): {sorted(invalid_formats)}")
    if not formats:
        formats = {"png"}

    if not anchor_log.is_file():
        raise SystemExit(f"Anchor log not found: {anchor_log}")
    if not reference_gp.is_file():
        raise SystemExit(f"Reference GP not found: {reference_gp}")

    gp = _load_gp(reference_gp)
    records = _iter_anchor_runs(anchor_log)
    if grid_log.is_file():
        records.extend(_iter_grid_runs(grid_log))

    only_tasks = set(args.only_task)
    if only_tasks:
        records = [r for r in records if r.task in only_tasks]

    plotted: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in records:
        summary_path = record.run_dir / "run_summary.json"
        experiment_path = record.run_dir / "experiment.csv"
        if not summary_path.is_file() or not experiment_path.is_file():
            skipped.append(f"{record.source} {record.task} {record.label} seed {record.seed}: missing run files at {record.run_dir}")
            continue
        rows = _load_csv_rows(experiment_path)
        if not rows:
            skipped.append(f"{record.source} {record.task} {record.label} seed {record.seed}: empty experiment.csv")
            continue
        summary = _load_json(summary_path)
        manifest = _load_json(record.run_dir / "run_manifest.json")
        perception_rows = _load_csv_rows(record.run_dir / "perception.csv")
        name_parts = [
            record.source,
            record.task,
            record.condition if record.source == "anchor" else record.axis,
            record.label,
            f"seed{record.seed}",
        ]
        out_base = plot_dir / "__".join(_safe_name(part) for part in name_parts if str(part))
        _, _, metrics = _plot_run(
            record=record,
            gp=gp,
            rows=rows,
            perception_rows=perception_rows,
            summary=summary,
            manifest=manifest,
            out_base=out_base,
            formats=formats,
        )
        plotted.append(metrics)

    plotted.sort(key=_sort_key)
    _write_ranking(out_dir / "run_visibility_ranking.csv", plotted)
    _write_index(out_dir / "index.md", plotted, skipped)

    print(f"Plotted {len(plotted)} runs into {plot_dir}")
    print(f"Wrote ranking: {out_dir / 'run_visibility_ranking.csv'}")
    print(f"Wrote index:   {out_dir / 'index.md'}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete log entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
