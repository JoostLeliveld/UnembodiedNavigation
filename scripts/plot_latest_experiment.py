#!/usr/bin/env python3
"""
Plot informative summaries for the most recent experiment log.

Outputs two figures:
1) Trajectory with uncertainty overlays
2) Time-series diagnostics (uncertainty, goal distance, commands, EFE metrics, heading)
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot latest experiment log summary.")
    parser.add_argument(
        "--path",
        default="",
        help="Path to a specific experiment CSV (default: newest in logs/experiments/).",
    )
    parser.add_argument(
        "--out",
        default="plots/experiments",
        help="Output directory for plots (default: plots/experiments).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: save only).",
    )
    parser.add_argument(
        "--ellipse-step",
        type=int,
        default=10,
        help="Stride for uncertainty ellipses along the trajectory (default: 10).",
    )
    parser.add_argument(
        "--ellipse-sigma",
        type=float,
        default=2.0,
        help="Sigma multiplier for ellipses (default: 2.0).",
    )
    return parser.parse_args()


def _find_latest_csv() -> str:
    candidates = glob.glob("logs/experiments/experiment_*.csv")
    if not candidates:
        raise FileNotFoundError("No experiment logs found in logs/experiments/")
    return max(candidates, key=os.path.getmtime)


def _to_float_series(series):
    return series.astype(float, errors="ignore")


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _safe_sqrt(x: float) -> float:
    return math.sqrt(x) if x > 0.0 else 0.0


def _has_cols(df, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def main() -> int:
    args = _parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    csv_path = args.path or _find_latest_csv()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    # Coerce numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["stamp"])

    t = df["stamp"].to_numpy()
    t = t - t[0]

    # Required basics
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    yaw = df["yaw"].to_numpy() if "yaw" in df.columns else np.zeros_like(x)

    cov_x = df["cov_x"].to_numpy() if "cov_x" in df.columns else np.zeros_like(x)
    cov_y = df["cov_y"].to_numpy() if "cov_y" in df.columns else np.zeros_like(x)
    cov_yaw = df["cov_yaw"].to_numpy() if "cov_yaw" in df.columns else np.zeros_like(x)

    cmd_v = df["cmd_v"].to_numpy() if "cmd_v" in df.columns else np.zeros_like(x)
    cmd_w = df["cmd_w"].to_numpy() if "cmd_w" in df.columns else np.zeros_like(x)

    goal_x = df["goal_x"].to_numpy() if "goal_x" in df.columns else np.zeros_like(x)
    goal_y = df["goal_y"].to_numpy() if "goal_y" in df.columns else np.zeros_like(x)
    goal_dist = df["goal_dist"].to_numpy() if "goal_dist" in df.columns else np.zeros_like(x)

    plan_points = df["plan_points"].to_numpy() if "plan_points" in df.columns else np.zeros_like(x)
    plan_length = df["plan_length"].to_numpy() if "plan_length" in df.columns else np.zeros_like(x)

    efe_cols = ["efe_total", "efe_risk", "efe_ambiguity", "efe_control", "efe_boundary"]
    has_efe = _has_cols(df, efe_cols)
    if has_efe:
        efe_total = df["efe_total"].to_numpy()
        efe_risk = df["efe_risk"].to_numpy()
        efe_amb = df["efe_ambiguity"].to_numpy()
        efe_ctrl = df["efe_control"].to_numpy()
        efe_bound = df["efe_boundary"].to_numpy()

    # Derived quantities
    sigma_pos = np.sqrt(np.maximum(cov_x, 0.0) + np.maximum(cov_y, 0.0))
    sigma_yaw = np.sqrt(np.maximum(cov_yaw, 0.0))

    angle_to_goal = np.zeros_like(x)
    yaw_error = np.zeros_like(x)
    if _has_cols(df, ["goal_x", "goal_y"]):
        for i in range(len(x)):
            dx = goal_x[i] - x[i]
            dy = goal_y[i] - y[i]
            angle_to_goal[i] = math.atan2(dy, dx)
            yaw_error[i] = _wrap_angle(angle_to_goal[i] - yaw[i])

    # Figure 1: Trajectory + uncertainty
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    sc = ax1.scatter(x, y, c=sigma_pos, s=12, cmap="viridis", label="Trajectory (colored by σ_pos)")
    ax1.plot(x, y, alpha=0.3)
    ax1.scatter([x[0]], [y[0]], c="green", s=60, marker="o", label="Start")
    if np.any(goal_x != 0.0) or np.any(goal_y != 0.0):
        ax1.scatter([goal_x[-1]], [goal_y[-1]], c="red", s=60, marker="*", label="Goal")
    ax1.set_title(f"Trajectory with Position Uncertainty ({prefix})")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.axis("equal")
    cb = fig1.colorbar(sc, ax=ax1, shrink=0.8)
    cb.set_label("σ_pos (m)")

    # Uncertainty ellipses (axis-aligned, using diag cov only)
    if args.ellipse_step > 0:
        for i in range(0, len(x), args.ellipse_step):
            ex = args.ellipse_sigma * _safe_sqrt(cov_x[i])
            ey = args.ellipse_sigma * _safe_sqrt(cov_y[i])
            if ex == 0.0 and ey == 0.0:
                continue
            ell = Ellipse((x[i], y[i]), width=2 * ex, height=2 * ey, alpha=0.15, color="orange")
            ax1.add_patch(ell)

    # Heading quiver (sparse)
    step = max(1, len(x) // 30)
    ax1.quiver(
        x[::step],
        y[::step],
        np.cos(yaw[::step]),
        np.sin(yaw[::step]),
        color="black",
        alpha=0.3,
        scale=20,
        width=0.003,
    )
    ax1.legend(loc="best")
    traj_out = os.path.join(args.out, f"{prefix}_trajectory.png")
    fig1.tight_layout()
    fig1.savefig(traj_out, dpi=150)

    # Figure 2: Time series diagnostics
    fig2, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    # Uncertainty
    axes[0].plot(t, np.sqrt(np.maximum(cov_x, 0.0)), label="σ_x")
    axes[0].plot(t, np.sqrt(np.maximum(cov_y, 0.0)), label="σ_y")
    axes[0].plot(t, sigma_yaw, label="σ_yaw (rad)")
    axes[0].plot(t, sigma_pos, label="σ_pos")
    axes[0].set_title("Uncertainty (Std Dev)")
    axes[0].set_ylabel("std dev")
    axes[0].legend(loc="best")

    # Goal distance & plan length
    axes[1].plot(t, goal_dist, label="goal_dist")
    ax1b = axes[1].twinx()
    ax1b.plot(t, plan_length, color="tab:orange", label="plan_length")
    axes[1].set_title("Goal Distance & Plan Length")
    axes[1].set_ylabel("goal_dist (m)")
    ax1b.set_ylabel("plan_length (m)")

    # Commands
    axes[2].plot(t, cmd_v, label="cmd_v")
    axes[2].plot(t, cmd_w, label="cmd_w")
    axes[2].set_title("Control Commands")
    axes[2].set_ylabel("cmd")
    axes[2].legend(loc="best")

    # EFE metrics
    if has_efe:
        axes[3].plot(t, efe_total, label="efe_total")
        axes[3].plot(t, efe_risk, label="risk")
        axes[3].plot(t, efe_amb, label="ambiguity")
        axes[3].plot(t, efe_ctrl, label="control")
        axes[3].plot(t, efe_bound, label="boundary")
        axes[3].set_title("EFE Components")
        axes[3].set_ylabel("value")
        axes[3].legend(loc="best")
    else:
        axes[3].text(0.5, 0.5, "EFE metrics not in log", ha="center", va="center")
        axes[3].set_title("EFE Components")

    # Heading vs goal bearing
    axes[4].plot(t, yaw, label="yaw")
    axes[4].plot(t, angle_to_goal, label="angle_to_goal")
    axes[4].plot(t, yaw_error, label="yaw_error")
    axes[4].set_title("Heading vs Goal Bearing")
    axes[4].set_ylabel("rad")
    axes[4].legend(loc="best")

    # Plan points
    axes[5].plot(t, plan_points, label="plan_points")
    axes[5].set_title("Plan Points")
    axes[5].set_ylabel("count")
    axes[5].set_xlabel("time (s)")
    axes[5].legend(loc="best")

    for ax in axes[:-1]:
        ax.set_xlabel("time (s)")

    fig2.tight_layout()
    ts_out = os.path.join(args.out, f"{prefix}_timeseries.png")
    fig2.savefig(ts_out, dpi=150)

    print(f"Saved plots:\n  {traj_out}\n  {ts_out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
