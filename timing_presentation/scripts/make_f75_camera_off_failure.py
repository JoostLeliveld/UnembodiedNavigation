#!/usr/bin/env python3
"""F75: why camera-off fails while F73 C1 can still finish."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "timing_presentation/figures/F75"

RUNS = {
    "F73 C1: direct route, correction ON": ROOT
    / "logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1"
    / "probe_a4_boxside_north_to_a3top/C1/seed0/experiment_20260604_144614",
    "F73 C2: visible route, correction ON": ROOT
    / "logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1"
    / "probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_144802",
    "Run A: same C2 route, correction OFF": ROOT
    / "logs/visibility_comparison/ablation_corrOFF"
    / "probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_155318",
}


def first_cmd_stamp(exp: pd.DataFrame) -> float:
    cmd = exp.get("cmd_v", 0).abs() + exp.get("cmd_w", 0).abs()
    moving = exp.loc[cmd > 1e-3]
    return float(moving["stamp"].iloc[0] if len(moving) else exp["stamp"].iloc[0])


def load_run(path: Path) -> dict:
    exp = pd.read_csv(path / "experiment.csv")
    perc = pd.read_csv(path / "perception.csv") if (path / "perception.csv").exists() else pd.DataFrame()
    summary = json.loads((path / "run_summary.json").read_text())
    t0 = first_cmd_stamp(exp)
    return {
        "path": path,
        "exp": exp.loc[exp["stamp"] >= t0].copy(),
        "perc": perc.loc[perc["log_stamp"] >= t0].copy() if len(perc) and "log_stamp" in perc else perc,
        "summary": summary,
        "t0": t0,
    }


def rel_time(df: pd.DataFrame, t0: float, col: str = "stamp") -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) - float(t0)


def draw_world(ax):
    for x in [-0.25, 1.8, 3.85]:
        for y in [-0.85, 2.2]:
            ax.add_patch(
                patches.Rectangle((x, y), 0.55, 2.05, facecolor="#d9d9d9", edgecolor="#333", lw=0.8)
            )
    for xy, wh in [((0.45, -2.05), (2.85, 0.75)), ((0.45, 1.35), (2.85, 0.85)), ((2.85, -1.95), (0.75, 3.8))]:
        ax.add_patch(patches.Rectangle(xy, *wh, facecolor="#22c55e", alpha=0.10, edgecolor="none"))
    ax.scatter([3.35], [-1.55], s=90, color="#22c55e", edgecolor="black", zorder=5, label="start")
    ax.scatter([1.0], [1.75], s=130, color="#facc15", marker="*", edgecolor="black", zorder=5, label="goal")
    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(-2.25, 4.55)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load_run(path) for name, path in RUNS.items()}
    colors = {
        "F73 C1: direct route, correction ON": "#2563eb",
        "F73 C2: visible route, correction ON": "#ef4444",
        "Run A: same C2 route, correction OFF": "#111827",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    draw_world(ax)
    for name, d in data.items():
        exp = d["exp"]
        color = colors[name]
        ax.plot(exp["truth_x"], exp["truth_y"], color=color, lw=2.5, label=name)
        ax.plot(exp["planner_belief_x"], exp["planner_belief_y"], color=color, lw=1.1, ls="--", alpha=0.55)
        if str(d["summary"].get("completion_reason", "")) == "collision":
            ax.scatter([exp["truth_x"].iloc[-1]], [exp["truth_y"].iloc[-1]], marker="X", s=120, color=color, edgecolor="white", zorder=6)
    ax.set_title("Map: truth paths (solid) and planner belief (dashed)")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[0, 1]
    for name, d in data.items():
        exp = d["exp"]
        ax.plot(rel_time(exp, d["t0"]), exp["truth_belief_error_m"], color=colors[name], lw=1.8, label=name)
    ax.set_title("Planner truth-belief error")
    ax.set_xlabel("time after first command [s]")
    ax.set_ylabel("error [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for name, d in data.items():
        exp = d["exp"]
        if "min_obstacle_distance_m" in exp:
            ax.plot(rel_time(exp, d["t0"]), exp["min_obstacle_distance_m"], color=colors[name], lw=1.8, label=name)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_title("Truth obstacle clearance (collision when <= 0)")
    ax.set_xlabel("time after first command [s]")
    ax.set_ylabel("clearance [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for name, d in data.items():
        perc = d["perc"]
        if len(perc):
            t = rel_time(perc, d["t0"], col="log_stamp")
            det = pd.to_numeric(perc["yolo_detected_after_threshold"], errors="coerce")
            score = pd.to_numeric(perc["yolo_score_selected"], errors="coerce")
            ax.plot(t, score, color=colors[name], lw=1.4, label=f"{name} score")
            if "pixel_pose_age_s" in perc:
                ax.plot(t, pd.to_numeric(perc["pixel_pose_age_s"], errors="coerce"), color=colors[name], lw=0.9, ls=":", alpha=0.8)
            miss_t = t[det.to_numpy(dtype=float) < 0.5]
            if len(miss_t):
                ax.scatter(miss_t, np.zeros_like(miss_t) - 0.03, color=colors[name], s=10, marker="x")
    ax.set_title("YOLO score; dotted lines are pixel-pose age")
    ax.set_xlabel("time after first command [s]")
    ax.set_ylabel("score / age [s]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6)

    fig.suptitle("F75 - Why camera-off fails while C1 can still finish", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "F75_camera_off_failure.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "F75_camera_off_failure.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# F75 Camera-Off Failure Comparison",
        "",
        "This figure compares F73 C1/C2 with Run A, where C2 used the same route-choice setup but `use_pixel_correction:=False`.",
        "",
        "## Files",
        "",
        f"- Figure: `{OUT / 'F75_camera_off_failure.png'}`",
        f"- PDF: `{OUT / 'F75_camera_off_failure.pdf'}`",
        "",
        "## Runtime-Only Summary",
        "",
    ]
    for name, d in data.items():
        exp = d["exp"]
        perc = d["perc"]
        det_rate = float("nan")
        if len(perc):
            det_rate = float((pd.to_numeric(perc["yolo_detected_after_threshold"], errors="coerce") >= 0.5).mean())
        lines.extend([
            f"### {name}",
            f"- Run: `{d['path']}`",
            f"- Outcome: `{d['summary'].get('completion_reason')}`",
            f"- Path length: `{d['summary'].get('path_length_m'):.3f} m`",
            f"- Min goal distance: `{d['summary'].get('minimum_goal_distance'):.3f} m`",
            f"- Min obstacle distance: `{d['summary'].get('min_obstacle_distance_m'):.3f} m`",
            f"- Truth-belief error mean/median/max: `{exp['truth_belief_error_m'].mean():.3f}` / `{exp['truth_belief_error_m'].median():.3f}` / `{exp['truth_belief_error_m'].max():.3f} m`",
            f"- YOLO detection rate: `{det_rate:.3f}`",
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "C1 is not equivalent to camera-off. C1 still has pixel correction enabled and gets enough camera updates before and after the weak-visibility segment to keep the planner belief bounded. Its direct route has a blackout, but not a complete removal of camera corrections.",
        "",
        "Run A disables pixel correction entirely. The robot still has YOLO detections in the logs, but the planner belief cannot use them; it relies on noisy dead reckoning. Belief error grows to multiple metres and the truth trajectory collides before reaching the goal.",
        "",
        "So the correct distinction is: C1 tests a camera-poor route with correction still available when detections exist; Run A tests no camera correction at all. The latter is much harsher and fails because drift accumulates globally, not only in one occluded segment.",
    ])
    (OUT / "F75_camera_off_failure.md").write_text("\n".join(lines))
    print(OUT / "F75_camera_off_failure.png")
    print(OUT / "F75_camera_off_failure.md")


if __name__ == "__main__":
    main()
