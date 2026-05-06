#!/usr/bin/env python3
"""Plot unique correction-event diagnostics for a single run.

This focuses on the semantics that were muddy in the mechanism overlay:
- unique planner correction events (by correction stamp)
- time since last correction
- belief age used for planning
- truth-belief error
- measurement_available and planner_pixel_correction_available flags
"""

from __future__ import annotations

import argparse
import math
import os
import re
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

PIXEL_TIMEOUT_S = 0.5


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "na"


def _truth_points(rows):
    xs, ys, ts = [], [], []
    for row in rows:
        if _pf(row, "truth_available") < 0.5:
            continue
        x = _pf(row, "truth_x")
        y = _pf(row, "truth_y")
        t = _pf(row, "stamp")
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(t):
            xs.append(x)
            ys.append(y)
            ts.append(t)
    return np.asarray(ts, dtype=float), np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _belief_with_cov(rows):
    ts, xs, ys, sxx, sxy, syy = [], [], [], [], [], []
    seen = None
    for row in rows:
        if _pf(row, "planner_belief_available") < 0.5:
            continue
        bstamp = _pf(row, "planner_belief_stamp")
        if not math.isfinite(bstamp):
            continue
        if seen is not None and bstamp == seen:
            continue
        seen = bstamp
        x = _pf(row, "planner_belief_x")
        y = _pf(row, "planner_belief_y")
        if math.isfinite(x) and math.isfinite(y):
            ts.append(_pf(row, "stamp"))
            xs.append(x)
            ys.append(y)
            sxx.append(_pf(row, "planner_cov_x"))
            sxy.append(_pf(row, "planner_cov_xy"))
            syy.append(_pf(row, "planner_cov_y"))
    return (
        np.asarray(ts, dtype=float),
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(sxx, dtype=float),
        np.asarray(sxy, dtype=float),
        np.asarray(syy, dtype=float),
    )


def _perception_points(rows):
    xs, ys = [], []
    for row in rows:
        detected = _pf(row, "detected") >= 0.5 or _pf(row, "yolo_detected_after_threshold") >= 0.5
        x = _pf(row, "pred_world_x")
        y = _pf(row, "pred_world_y")
        if detected and math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _nearest_truth_xy(truth_t: np.ndarray, truth_x: np.ndarray, truth_y: np.ndarray, stamp: float):
    if truth_t.size == 0 or not math.isfinite(stamp):
        return math.nan, math.nan
    idx = int(np.argmin(np.abs(truth_t - stamp)))
    return float(truth_x[idx]), float(truth_y[idx])


def _nearest_progress(truth_t: np.ndarray, progress: np.ndarray, stamp: float) -> float:
    if truth_t.size == 0 or progress.size == 0 or not math.isfinite(stamp):
        return math.nan
    idx = int(np.argmin(np.abs(truth_t - stamp)))
    return float(progress[idx])


def _extract_unique_correction_events(rows, truth_t, truth_x, truth_y):
    events = []
    seen_stamps = set()
    for row in rows:
        avail = _pf(row, "planner_pixel_correction_available")
        corr_stamp = _pf(row, "planner_pixel_correction_stamp")
        if not (math.isfinite(avail) and avail >= 0.5 and math.isfinite(corr_stamp)):
            continue
        # Deduplicate by the correction stamp itself.
        key = round(corr_stamp, 6)
        if key in seen_stamps:
            continue
        seen_stamps.add(key)
        x = _pf(row, "pixel_corr_next_x")
        y = _pf(row, "pixel_corr_next_y")
        if not (math.isfinite(x) and math.isfinite(y)):
            x, y = _nearest_truth_xy(truth_t, truth_x, truth_y, corr_stamp)
        events.append({
            "corr_stamp": float(corr_stamp),
            "log_stamp": float(_pf(row, "stamp")),
            "age": float(_pf(row, "planner_pixel_correction_age_s")),
            "belief_age": float(_pf(row, "belief_age_s")),
            "meas_avail": float(_pf(row, "measurement_available")),
            "truth_belief_error": float(_pf(row, "truth_belief_error_m")),
            "x": float(x),
            "y": float(y),
        })
    events.sort(key=lambda e: e["corr_stamp"])
    return events


def _path_progress(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.asarray([], dtype=float)
    ds = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    return s / total if total > 1e-9 else np.zeros_like(s)


def _low_rho_spans(progress: np.ndarray, truth_x: np.ndarray, truth_y: np.ndarray, gp) -> list[tuple[float, float]]:
    if progress.size == 0:
        return []
    rhos = np.asarray([gp.interp([[float(y), float(x)]])[0] for x, y in zip(truth_x, truth_y)], dtype=float)
    low = np.isfinite(rhos) & (rhos < RHO_LOW_THRESHOLD)
    spans = []
    start = None
    for i, flag in enumerate(low):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            spans.append((float(progress[start]), float(progress[i - 1])))
            start = None
    if start is not None:
        spans.append((float(progress[start]), float(progress[-1])))
    return spans


def plot_run(run_dir: Path, gp_path: Path, out_dir: Path) -> tuple[Path, Path]:
    rows = _load_csv_rows(run_dir / "experiment.csv")
    if not rows:
        raise RuntimeError(f"No experiment.csv in {run_dir}")
    manifest = _load_json(run_dir / "run_manifest.json")
    summary = _load_json(run_dir / "run_summary.json")
    perception_rows = _load_csv_rows(run_dir / "perception.csv")
    gp = _load_gp(gp_path)

    truth_t, truth_x, truth_y = _truth_points(rows)
    bt, bx, by, bsxx, bsxy, bsyy = _belief_with_cov(rows)
    px, py = _perception_points(perception_rows)
    events = _extract_unique_correction_events(rows, truth_t, truth_x, truth_y)

    first_cmd = float(summary.get("first_cmd_stamp", math.nan))
    if not math.isfinite(first_cmd):
        first_cmd = float(truth_t[0]) if truth_t.size else 0.0

    stamps = np.asarray([_pf(r, "stamp") for r in rows], dtype=float)
    t = stamps - first_cmd
    corr_age = np.asarray([_pf(r, "planner_pixel_correction_age_s") for r in rows], dtype=float)
    belief_age = np.asarray([_pf(r, "belief_age_s") for r in rows], dtype=float)
    belief_err = np.asarray([_pf(r, "truth_belief_error_m") for r in rows], dtype=float)
    meas_avail = np.asarray([_pf(r, "measurement_available") for r in rows], dtype=float)
    corr_avail = np.asarray([_pf(r, "planner_pixel_correction_available") for r in rows], dtype=float)

    progress = _path_progress(truth_x, truth_y)
    low_rho_spans = _low_rho_spans(progress, truth_x, truth_y, gp)

    fig = plt.figure(figsize=(11.8, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.9, 1.1, 1.0, 1.0], hspace=0.08)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_age = fig.add_subplot(gs[2, 0], sharex=ax_err)
    ax_flags = fig.add_subplot(gs[3, 0], sharex=ax_err)

    im = _draw_world(ax_world, gp, manifest)
    cbar = fig.colorbar(im, ax=ax_world, fraction=0.036, pad=0.02)
    cbar.set_label("nominal raw-GP rho_plan")
    if bx.size:
        _draw_belief_with_uncertainty(ax_world, bx, by, bsxx, bsxy, bsyy)
    if truth_x.size:
        _draw_truth(ax_world, truth_x, truth_y)
    if px.size:
        step = max(1, int(math.ceil(px.size / 120)))
        ax_world.scatter(
            px[::step], py[::step],
            c="#ff9f43", s=14, alpha=0.85,
            edgecolors="black", linewidths=0.35,
            zorder=9, label="YOLO world point",
        )
    if events:
        ex = np.asarray([e["x"] for e in events], dtype=float)
        ey = np.asarray([e["y"] for e in events], dtype=float)
        sc = ax_world.scatter(
            ex, ey,
            c=np.arange(len(events)), cmap="coolwarm",
            s=42, edgecolors="white", linewidths=0.8,
            zorder=13, label="unique correction event",
        )
        for i, e in enumerate(events):
            if i in (0, len(events) - 1):
                ax_world.text(e["x"] + 0.03, e["y"] + 0.03, f"{i+1}", fontsize=7.5, color="white", zorder=14)

    if truth_x.size:
        ax_world.scatter([truth_x[0]], [truth_y[0]], s=170, marker="o", facecolor="#22c55e",
                         edgecolors="black", linewidths=1.4, zorder=15, label="start")
        ax_world.scatter([truth_x[-1]], [truth_y[-1]], s=120, marker="X", facecolor="#f59e0b",
                         edgecolors="black", linewidths=1.0, zorder=15, label="truth end")
    goal_x = _mean_csv_value(rows[:10], "goal_x")
    goal_y = _mean_csv_value(rows[:10], "goal_y")
    if math.isfinite(goal_x) and math.isfinite(goal_y):
        ax_world.add_patch(Circle((goal_x, goal_y), GOAL_SUCCESS_RADIUS_M, fill=False,
                                  edgecolor="#ef4444", linestyle="--", linewidth=1.6, zorder=14))
        ax_world.scatter([goal_x], [goal_y], marker="*", s=300, facecolor="#ef4444",
                         edgecolors="black", linewidths=1.2, zorder=15, label="goal")

    info = [
        f"unique correction events: {len(events)}",
        f"max corr age: {np.nanmax(corr_age):.2f} s" if np.isfinite(corr_age).any() else "max corr age: n/a",
        f"mean belief err: {float(summary.get('mean_truth_belief_error_m', math.nan)):.3f} m",
        f"max belief err: {np.nanmax(belief_err):.3f} m" if np.isfinite(belief_err).any() else "max belief err: n/a",
    ]
    ax_world.text(
        0.015, 0.985, "\n".join(info),
        transform=ax_world.transAxes, va="top", ha="left", fontsize=8.6,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.82, "pad": 4.0},
        zorder=20,
    )
    title = f"{manifest.get('task', run_dir.parent.parent.name)} | {manifest.get('planner', '')} | {run_dir.parent.name} | correction-event diagnostics"
    ax_world.set_title(title, fontsize=12)
    handles, labels = ax_world.get_legend_handles_labels()
    if handles:
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax_world.legend(seen.values(), seen.keys(), loc="lower left", fontsize=8.1,
                        framealpha=0.93, facecolor="white", edgecolor="black")

    if truth_x.size:
        # map each logger row to nearest truth progress
        row_prog = []
        for row in rows:
            tx = _pf(row, "truth_x")
            ty = _pf(row, "truth_y")
            if not (_pf(row, "truth_available") > 0.5 and math.isfinite(tx) and math.isfinite(ty)):
                row_prog.append(math.nan)
                continue
            idx = int(np.argmin((truth_x - tx) ** 2 + (truth_y - ty) ** 2))
            row_prog.append(float(progress[idx]))
        row_prog = np.asarray(row_prog, dtype=float)
        for a, b in low_rho_spans:
            ax_err.axvspan(a, b, color="#fde68a", alpha=0.30, linewidth=0)
        finite = np.isfinite(row_prog) & np.isfinite(belief_err)
        ax_err.plot(row_prog[finite], belief_err[finite], color="#111827", linewidth=1.8, label="truth-belief error")
        for e in events:
            event_prog = _nearest_progress(truth_t, progress, e["corr_stamp"])
            if math.isfinite(event_prog):
                ax_err.axvline(event_prog, color="#2563eb", alpha=0.22, linewidth=0.9)
        ax_err.set_ylabel("belief err (m)")
        ax_err.grid(True, alpha=0.25)
        ax_err.legend(loc="upper left", fontsize=8.4, framealpha=0.92)

    finite_corr = np.isfinite(t) & np.isfinite(corr_age)
    finite_bel = np.isfinite(t) & np.isfinite(belief_age)
    ax_age.plot(t[finite_corr], corr_age[finite_corr], color="#7c3aed", linewidth=1.7, label="since last correction")
    ax_age.plot(t[finite_bel], belief_age[finite_bel], color="#0f172a", linewidth=1.2, linestyle="--", label="belief age used for planning")
    ax_age.axhline(PIXEL_TIMEOUT_S, color="#ef4444", linestyle=":", linewidth=1.2, label="pixel timeout 0.5 s")
    event_t = np.asarray([e["corr_stamp"] - first_cmd for e in events], dtype=float)
    if event_t.size:
        ax_age.scatter(event_t, np.zeros_like(event_t), marker="|", s=120, color="#2563eb", alpha=0.95,
                       label="unique correction event")
    ax_age.set_ylabel("age (s)")
    ax_age.grid(True, alpha=0.25)
    ax_age.legend(loc="upper right", fontsize=8.0, framealpha=0.92, ncol=2)

    finite_flags = np.isfinite(t)
    ax_flags.step(t[finite_flags], np.where(np.isfinite(meas_avail[finite_flags]), meas_avail[finite_flags], 0.0),
                  where="post", color="#16a34a", linewidth=1.5, label="measurement_available")
    ax_flags.step(t[finite_flags], np.where(np.isfinite(corr_avail[finite_flags]), corr_avail[finite_flags], 0.0) + 0.08,
                  where="post", color="#f59e0b", linewidth=1.3, label="planner_pixel_correction_available (+0.08)")
    if event_t.size:
        ax_flags.scatter(event_t, np.full_like(event_t, 1.15), marker="v", s=24, color="#2563eb", label="unique correction event")
    ax_flags.set_ylim(-0.05, 1.25)
    ax_flags.set_ylabel("flags")
    ax_flags.set_xlabel("time after first command (s)")
    ax_flags.grid(True, alpha=0.25)
    ax_flags.legend(loc="upper right", fontsize=8.0, framealpha=0.92, ncol=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / _safe_name(f"{run_dir.parent.parent.name}_{run_dir.parent.name}_correction_events")
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-gp", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    png, pdf = plot_run(
        Path(args.run_dir).expanduser().resolve(),
        Path(args.reference_gp).expanduser().resolve(),
        Path(args.out_dir).expanduser().resolve(),
    )
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
