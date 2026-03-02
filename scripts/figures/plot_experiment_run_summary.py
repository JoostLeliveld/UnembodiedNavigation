#!/usr/bin/env python3
"""Create a paper-friendly summary figure for one experiment run directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    vals: List[float] = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def _load_manifest(run_dir: Path) -> Dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_plan_snapshots(run_dir: Path) -> List[Dict[str, np.ndarray]]:
    path = run_dir / "plan_samples.csv"
    if not path.exists():
        return []
    snapshots: Dict[float, List[List[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stamp = float(row["plan_stamp"])
                idx = int(float(row["point_idx"]))
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            pts = snapshots.setdefault(stamp, [])
            while len(pts) <= idx:
                pts.append([float("nan"), float("nan")])
            pts[idx] = [x, y]
    out: List[Dict[str, np.ndarray]] = []
    for stamp in sorted(snapshots.keys()):
        pts = np.asarray(snapshots[stamp], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        out.append({"stamp": float(stamp), "x": pts[:, 0], "y": pts[:, 1]})
    return out


def _first_goal_index(goal_x: np.ndarray, goal_y: np.ndarray) -> int:
    mask = np.isfinite(goal_x) & np.isfinite(goal_y) & ((np.abs(goal_x) + np.abs(goal_y)) > 1e-12)
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else 0


def _nanmin_safe(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.min(finite)) if finite.size else float("nan")


def _nansafe(arr: np.ndarray, idx: int) -> float:
    if idx < 0 or idx >= arr.size:
        return float("nan")
    val = arr[idx]
    return float(val) if np.isfinite(val) else float("nan")


def build_figure(run_dir: Path, output: Optional[Path], success_threshold: float) -> Path:
    csv_path = run_dir / "experiment.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = _load_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")

    manifest = _load_manifest(run_dir)

    stamp = _to_float(rows, "stamp")
    t = stamp - stamp[0]
    x = _to_float(rows, "x")
    y = _to_float(rows, "y")
    yaw = _to_float(rows, "yaw")
    cmd_v = _to_float(rows, "cmd_v")
    cmd_w = _to_float(rows, "cmd_w")
    goal_x = _to_float(rows, "goal_x")
    goal_y = _to_float(rows, "goal_y")
    goal_dist = _to_float(rows, "goal_dist")
    plan_points = _to_float(rows, "plan_points")
    efe_total = _to_float(rows, "efe_total")
    efe_risk = _to_float(rows, "efe_risk")
    efe_ambiguity = _to_float(rows, "efe_ambiguity")
    efe_control = _to_float(rows, "efe_control")

    goal_idx = _first_goal_index(goal_x, goal_y)
    t_goal = t[goal_idx:]
    goal_dist_valid = goal_dist[goal_idx:]
    cmd_v_valid = cmd_v[goal_idx:]
    cmd_w_valid = cmd_w[goal_idx:]

    # Filter out pre-plan zeros for EFE metrics.
    efe_mask = np.isfinite(efe_total) & (
        (np.abs(efe_total) > 1e-12)
        | (np.abs(efe_risk) > 1e-12)
        | (np.abs(efe_ambiguity) > 1e-12)
        | (plan_points > 0)
    )

    title_bits = [
        str(manifest.get("planner", "unknown")).upper(),
        str(manifest.get("state_source", "unknown")),
        str(manifest.get("perception_backend", "unknown")),
        str(manifest.get("task", run_dir.name)),
    ]
    title = " | ".join(title_bits)

    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_goal = fig.add_subplot(gs[0, 1])
    ax_cmd = fig.add_subplot(gs[1, 0])
    ax_efe = fig.add_subplot(gs[1, 1])

    # Trajectory panel
    ax_xy.plot(x, y, color="#0b3c5d", lw=2.0, label="trajectory")
    ax_xy.scatter([x[0]], [y[0]], c="#2ca02c", s=60, marker="o", label="start", zorder=3)
    ax_xy.scatter([x[-1]], [y[-1]], c="#d62728", s=70, marker="x", label="end", zorder=3)
    gx = _nansafe(goal_x, goal_idx)
    gy = _nansafe(goal_y, goal_idx)
    if np.isfinite(gx) and np.isfinite(gy):
        ax_xy.scatter([gx], [gy], c="#ff7f0e", s=110, marker="*", label="goal", zorder=4)
        goal_circle = plt.Circle((gx, gy), success_threshold, color="#ff7f0e", fill=False, ls="--", lw=1.5, alpha=0.8)
        ax_xy.add_patch(goal_circle)
    # Show a few heading ticks for interpretability.
    heading_stride = max(1, len(x) // 20)
    idxs = np.arange(0, len(x), heading_stride, dtype=int)
    ax_xy.quiver(
        x[idxs],
        y[idxs],
        np.cos(yaw[idxs]),
        np.sin(yaw[idxs]),
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.003,
        color="#0b3c5d",
        alpha=0.35,
    )
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Trajectory in map_bev")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best", fontsize=9, frameon=True)

    # Goal distance panel
    ax_goal.plot(t_goal, goal_dist_valid, color="#1f77b4", lw=2.0)
    ax_goal.axhline(success_threshold, color="#ff7f0e", ls="--", lw=1.5, label=f"success radius = {success_threshold:.2f} m")
    min_goal = _nanmin_safe(goal_dist_valid)
    if np.isfinite(min_goal):
        ax_goal.axhline(min_goal, color="#2ca02c", ls=":", lw=1.2, label=f"min goal dist = {min_goal:.3f} m")
    ax_goal.set_xlabel("time since first goal [s]")
    ax_goal.set_ylabel("goal distance [m]")
    ax_goal.set_title("Goal distance")
    ax_goal.grid(True, alpha=0.25)
    ax_goal.legend(loc="best", fontsize=9, frameon=True)

    # Controls panel
    ax_cmd.plot(t_goal, cmd_v_valid, color="#2ca02c", lw=1.8, label="cmd_v [m/s]")
    ax_cmd.set_xlabel("time since first goal [s]")
    ax_cmd.set_ylabel("cmd_v [m/s]", color="#2ca02c")
    ax_cmd.tick_params(axis="y", labelcolor="#2ca02c")
    ax_cmd.grid(True, alpha=0.25)
    ax_cmd2 = ax_cmd.twinx()
    ax_cmd2.plot(t_goal, cmd_w_valid, color="#9467bd", lw=1.5, label="cmd_w [rad/s]")
    ax_cmd2.set_ylabel("cmd_w [rad/s]", color="#9467bd")
    ax_cmd2.tick_params(axis="y", labelcolor="#9467bd")
    ax_cmd.set_title("Control commands")
    # Combined legend
    lines = ax_cmd.get_lines() + ax_cmd2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax_cmd.legend(lines, labels, loc="best", fontsize=9, frameon=True)

    # EFE decomposition panel
    if np.any(efe_mask):
        ax_efe.plot(t[efe_mask], efe_total[efe_mask], color="#111111", lw=2.0, label="total")
        ax_efe.plot(t[efe_mask], efe_risk[efe_mask], color="#1f77b4", lw=1.6, label="risk")
        ax_efe.plot(t[efe_mask], efe_ambiguity[efe_mask], color="#d62728", lw=1.6, label="ambiguity")
        ax_efe.plot(t[efe_mask], efe_control[efe_mask], color="#2ca02c", lw=1.4, label="control")
    ax_efe.set_xlabel("time since start [s]")
    ax_efe.set_ylabel("EFE terms")
    ax_efe.set_title("EFE objective decomposition")
    ax_efe.grid(True, alpha=0.25)
    if np.any(efe_mask):
        ax_efe.legend(loc="best", fontsize=9, frameon=True)

    # Figure title and compact summary
    end_goal_dist = _nansafe(goal_dist, len(goal_dist) - 1)
    summary = (
        f"rows={len(rows)} | duration={t[-1]:.1f}s | "
        f"end=({x[-1]:.2f},{y[-1]:.2f}) | "
        f"min_goal_dist={min_goal:.3f}m | end_goal_dist={end_goal_dist:.3f}m"
    )
    fig.suptitle(f"{title}\n{run_dir.name} | {summary}", fontsize=14, y=1.01)

    if output is None:
        output = run_dir / "paper_behavior_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    pdf_out = output.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return output


def build_gif(
    run_dir: Path,
    output: Optional[Path],
    success_threshold: float,
    fps: int,
    max_frames: int,
) -> Path:
    csv_path = run_dir / "experiment.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    rows = _load_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    manifest = _load_manifest(run_dir)
    plan_snapshots = _load_plan_snapshots(run_dir)

    stamp = _to_float(rows, "stamp")
    t = stamp - stamp[0]
    x = _to_float(rows, "x")
    y = _to_float(rows, "y")
    yaw = _to_float(rows, "yaw")
    goal_x = _to_float(rows, "goal_x")
    goal_y = _to_float(rows, "goal_y")
    goal_idx = _first_goal_index(goal_x, goal_y)
    gx = _nansafe(goal_x, goal_idx)
    gy = _nansafe(goal_y, goal_idx)

    n = len(rows)
    if n <= 1:
        raise RuntimeError("Need at least 2 rows for animation")
    frame_count = min(max_frames, n)
    frame_idxs = np.linspace(0, n - 1, frame_count, dtype=int)
    frame_idxs = np.unique(frame_idxs)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = " | ".join(
        [
            str(manifest.get("planner", "unknown")).upper(),
            str(manifest.get("state_source", "unknown")),
            str(manifest.get("perception_backend", "unknown")),
            str(manifest.get("task", run_dir.name)),
        ]
    )
    ax.set_title(title)

    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size and finite_y.size:
        xpad = max(0.2, 0.05 * float(np.ptp(finite_x)))
        ypad = max(0.2, 0.05 * float(np.ptp(finite_y)))
        xmin = float(np.min(finite_x)) - xpad
        xmax = float(np.max(finite_x)) + xpad
        ymin = float(np.min(finite_y)) - ypad
        ymax = float(np.max(finite_y)) + ypad
        if np.isfinite(gx) and np.isfinite(gy):
            xmin = min(xmin, gx - success_threshold - 0.1)
            xmax = max(xmax, gx + success_threshold + 0.1)
            ymin = min(ymin, gy - success_threshold - 0.1)
            ymax = max(ymax, gy + success_threshold + 0.1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if np.isfinite(gx) and np.isfinite(gy):
        ax.scatter([gx], [gy], c="#ff7f0e", s=120, marker="*", label="goal", zorder=5)
        ax.add_patch(
            plt.Circle((gx, gy), success_threshold, color="#ff7f0e", fill=False, ls="--", lw=1.5, alpha=0.7)
        )

    ax.scatter([x[0]], [y[0]], c="#2ca02c", s=50, marker="o", label="start", zorder=5)
    trail_line, = ax.plot([], [], color="#0b3c5d", lw=2.0, label="inferred state")
    current_pt = ax.scatter([], [], c="#d62728", s=55, marker="o", zorder=6, label="current")
    heading_quiver = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1.0, color="#d62728")
    plan_line, = ax.plot([], [], color="#17becf", lw=1.8, alpha=0.9, label="latest plan")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.legend(loc="best", fontsize=9, frameon=True)

    plan_stamps = np.asarray([float(s["stamp"]) for s in plan_snapshots], dtype=float) if plan_snapshots else np.array([])

    def _latest_plan(abs_stamp: float) -> Optional[Dict[str, np.ndarray]]:
        if plan_stamps.size == 0:
            return None
        idxs = np.flatnonzero(plan_stamps <= abs_stamp + 1e-9)
        if idxs.size == 0:
            return None
        return plan_snapshots[int(idxs[-1])]

    def _update(frame_i: int):
        idx = int(frame_idxs[frame_i])
        trail_line.set_data(x[: idx + 1], y[: idx + 1])
        current_pt.set_offsets(np.array([[x[idx], y[idx]]], dtype=float))
        heading_quiver.set_offsets(np.array([[x[idx], y[idx]]], dtype=float))
        heading_quiver.set_UVC(np.array([math.cos(float(yaw[idx]))]), np.array([math.sin(float(yaw[idx]))]))
        plan = _latest_plan(float(stamp[idx]))
        if plan is None:
            plan_line.set_data([], [])
        else:
            plan_line.set_data(plan["x"], plan["y"])
        time_text.set_text(f"t={t[idx]:.1f}s")
        return trail_line, current_pt, heading_quiver, plan_line, time_text

    anim = FuncAnimation(fig, _update, frames=len(frame_idxs), interval=max(1, int(1000 / max(fps, 1))), blit=False)

    if output is None:
        output = run_dir / "paper_behavior_animation.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a paper-friendly summary for one experiment run.")
    parser.add_argument("--run-dir", required=True, help="Path to run dir containing experiment.csv")
    parser.add_argument("--output", default="", help="Output PNG path (PDF is also written next to it)")
    parser.add_argument("--success-threshold", type=float, default=0.35)
    parser.add_argument("--gif", action="store_true", help="Also write a GIF animation (uses plan_samples.csv if present).")
    parser.add_argument("--gif-output", default="", help="Output GIF path (default: run_dir/paper_behavior_animation.gif)")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=240)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve() if args.output else None
    out_path = build_figure(run_dir, output, success_threshold=float(args.success_threshold))
    print(f"wrote_png={out_path}")
    print(f"wrote_pdf={out_path.with_suffix('.pdf')}")
    if args.gif:
        gif_output = Path(args.gif_output).resolve() if args.gif_output else None
        gif_path = build_gif(
            run_dir,
            gif_output,
            success_threshold=float(args.success_threshold),
            fps=int(args.gif_fps),
            max_frames=int(args.gif_max_frames),
        )
        print(f"wrote_gif={gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
