#!/usr/bin/env python3
"""Plot per-run command-noise and localization diagnostics."""

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


def _low_rho_spans(progress: np.ndarray, rhos: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    if progress.size == 0:
        return spans
    low = np.isfinite(rhos) & (rhos < threshold)
    start = None
    for idx, is_low in enumerate(low):
        if is_low and start is None:
            start = idx
        elif (not is_low) and start is not None:
            spans.append((float(progress[start]), float(progress[idx - 1])))
            start = None
    if start is not None:
        spans.append((float(progress[start]), float(progress[-1])))
    return spans


def _plot_run(run_dir: Path, gp, out_dir: Path, formats: set[str]) -> Path:
    summary = _load_json(run_dir / "run_summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    rows = _load_csv_rows(run_dir / "experiment.csv")
    if not rows:
        raise RuntimeError(f"No experiment.csv rows in {run_dir}")

    truth_x = np.asarray([_pf(r, "truth_x") for r in rows], dtype=float)
    truth_y = np.asarray([_pf(r, "truth_y") for r in rows], dtype=float)
    truth_ok = np.asarray([_pf(r, "truth_available") > 0.5 for r in rows], dtype=bool)
    truth_x = truth_x[truth_ok]
    truth_y = truth_y[truth_ok]
    if truth_x.size < 2:
        raise RuntimeError(f"Not enough truth points in {run_dir}")

    progress = _path_progress(truth_x, truth_y)
    rhos = np.asarray([_query_rho(gp, x, y) for x, y in zip(truth_x, truth_y)], dtype=float)

    all_progress = []
    belief_err = []
    state_err = []
    for row in rows:
        tx = _pf(row, "truth_x")
        ty = _pf(row, "truth_y")
        if not (_pf(row, "truth_available") > 0.5 and math.isfinite(tx) and math.isfinite(ty)):
            continue
        # Map each row to nearest truth sample for plotting against path progress.
        idx = int(np.argmin((truth_x - tx) ** 2 + (truth_y - ty) ** 2))
        all_progress.append(float(progress[idx]))
        belief_err.append(_pf(row, "truth_belief_error_m"))
        state_err.append(_pf(row, "truth_state_error_m"))
    all_progress_arr = np.asarray(all_progress, dtype=float)
    belief_err_arr = np.asarray(belief_err, dtype=float)
    state_err_arr = np.asarray(state_err, dtype=float)

    stamps = np.asarray([_pf(r, "stamp") for r in rows], dtype=float)
    first_cmd = float(summary.get("first_cmd_stamp", math.nan))
    if not math.isfinite(first_cmd):
        finite_stamps = stamps[np.isfinite(stamps)]
        first_cmd = float(finite_stamps[0]) if finite_stamps.size else 0.0
    time_s = stamps - first_cmd

    cmd_raw_v = np.asarray([_pf(r, "cmd_raw_v") for r in rows], dtype=float)
    cmd_v = np.asarray([_pf(r, "cmd_v") for r in rows], dtype=float)
    cmd_raw_w = np.asarray([_pf(r, "cmd_raw_w") for r in rows], dtype=float)
    cmd_w = np.asarray([_pf(r, "cmd_w") for r in rows], dtype=float)
    cmd_add_v = np.asarray([_pf(r, "cmd_noise_linear_additive") for r in rows], dtype=float)
    cmd_add_w = np.asarray([_pf(r, "cmd_noise_angular_additive") for r in rows], dtype=float)
    cmd_err_v = np.asarray([_pf(r, "cmd_noise_v_error") for r in rows], dtype=float)
    cmd_err_w = np.asarray([_pf(r, "cmd_noise_w_error") for r in rows], dtype=float)

    low_rho_spans = _low_rho_spans(progress, rhos, RHO_LOW_THRESHOLD)

    fig = plt.figure(figsize=(11.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 1.2, 1.1, 1.1], hspace=0.08)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[2, 0], sharex=ax_v)
    ax_err = fig.add_subplot(gs[3, 0])

    im = _draw_world(ax_world, gp, manifest)
    cbar = fig.colorbar(im, ax=ax_world, fraction=0.035, pad=0.02)
    cbar.set_label("nominal raw-GP rho_plan")
    goal_x = _mean_csv_value(rows[:10], "goal_x")
    goal_y = _mean_csv_value(rows[:10], "goal_y")
    if math.isfinite(goal_x) and math.isfinite(goal_y):
        ax_world.scatter([goal_x], [goal_y], marker="*", s=280, facecolor="#ef4444",
                         edgecolors="black", linewidths=1.0, zorder=15)
        ax_world.add_patch(plt.Circle((goal_x, goal_y), GOAL_SUCCESS_RADIUS_M, fill=False,
                                      edgecolor="#ef4444", linestyle="--", linewidth=1.6, zorder=14))
    ax_world.scatter([truth_x[0]], [truth_y[0]], s=120, facecolor="#22c55e",
                     edgecolors="black", linewidths=1.0, zorder=15)
    ax_world.plot(truth_x, truth_y, color="#111827", linewidth=3.0, zorder=12, label="truth")
    ax_world.set_title(
        f"{run_dir.parent.parent.name} | {run_dir.parent.name} | {summary.get('completion_reason', '')}",
        fontsize=12,
    )
    ax_world.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    mask_v = _finite_mask(time_s, cmd_raw_v, cmd_v, cmd_add_v, cmd_err_v)
    ax_v.plot(time_s[mask_v], cmd_raw_v[mask_v], color="#6b7280", linewidth=1.4, label="raw v")
    ax_v.plot(time_s[mask_v], cmd_v[mask_v], color="#111827", linewidth=1.8, label="noisy v")
    ax_v.plot(time_s[mask_v], cmd_add_v[mask_v], color="#0ea5e9", linewidth=1.0, alpha=0.9, label="linear additive")
    ax_v.plot(time_s[mask_v], cmd_err_v[mask_v], color="#ef4444", linewidth=1.0, alpha=0.9, label="v error")
    ax_v.axhline(0.22, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax_v.set_ylabel("linear")
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.92)

    mask_w = _finite_mask(time_s, cmd_raw_w, cmd_w, cmd_add_w, cmd_err_w)
    ax_w.plot(time_s[mask_w], cmd_raw_w[mask_w], color="#6b7280", linewidth=1.4, label="raw w")
    ax_w.plot(time_s[mask_w], cmd_w[mask_w], color="#7c3aed", linewidth=1.8, label="noisy w")
    ax_w.plot(time_s[mask_w], cmd_add_w[mask_w], color="#0ea5e9", linewidth=1.0, alpha=0.9, label="angular additive")
    ax_w.plot(time_s[mask_w], cmd_err_w[mask_w], color="#ef4444", linewidth=1.0, alpha=0.9, label="w error")
    ax_w.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax_w.set_ylabel("angular")
    ax_w.set_xlabel("time after first command (s)")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.92)

    for start, end in low_rho_spans:
        ax_err.axvspan(start, end, color="#fde68a", alpha=0.35, linewidth=0)
    mask_err = _finite_mask(all_progress_arr, belief_err_arr)
    ax_err.plot(all_progress_arr[mask_err], belief_err_arr[mask_err], color="#111827",
                linewidth=1.9, label="truth-belief error")
    mask_state = _finite_mask(all_progress_arr, state_err_arr)
    ax_err.plot(all_progress_arr[mask_state], state_err_arr[mask_state], color="#7c3aed",
                linewidth=1.6, alpha=0.9, label="truth-state error")
    ax_err.set_xlabel("normalized path progress")
    ax_err.set_ylabel("error (m)")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / _safe_name(f"{run_dir.parent.parent.name}_{run_dir.parent.name}")
    if "png" in formats:
        fig.savefig(out_base.with_suffix(".png"), dpi=190)
    if "pdf" in formats:
        fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    return out_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Experiment run directory")
    parser.add_argument("--reference-gp", required=True, help="Reference GP artifact")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    formats = {s.strip().lower() for s in args.formats.split(",") if s.strip()}
    if not formats:
        formats = {"png"}
    gp = _load_gp(Path(args.reference_gp).expanduser())
    out_dir = Path(args.out_dir).expanduser()
    for run_dir_str in args.run_dir:
        _plot_run(Path(run_dir_str).expanduser(), gp, out_dir, formats)
    print(f"Wrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
