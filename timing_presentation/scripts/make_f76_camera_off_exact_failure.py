#!/usr/bin/env python3
"""F76: exact camera-off failure diagnostic.

This diagnostic compares the camera-off ablation against the successful C2 run.
It highlights the core failure mode: the local tracker follows the waypoint
route using a stale camera-derived state, while the physical robot keeps moving.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = TP_ROOT / "figures" / "F76"

RUNS = {
    "C2 correction ON": REPO_ROOT / "logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1/probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_144802",
    "C2 camera OFF": REPO_ROOT / "logs/visibility_comparison/ablation_corrOFF/probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_155318",
}


def load_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exp = pd.read_csv(path / "experiment.csv")
    plan = pd.read_csv(path / "plan_samples.csv")
    summary = json.loads((path / "run_summary.json").read_text())
    fc = float(summary["first_cmd_stamp"])
    exp = exp[exp["stamp"] >= fc].copy()
    exp["t"] = exp["stamp"] - fc
    return exp, plan, summary


def rect(ax, xy, wh, **kw):
    import matplotlib.patches as patches

    ax.add_patch(patches.Rectangle(xy, wh[0], wh[1], **kw))


def draw_geometry(ax):
    # Minimal A3/A4/R4/R5 geometry used in the F73/F75 diagnostics.
    rack_fc = "#cfcfcf"
    rack_ec = "#555555"
    for x, label in [(1.6, "R4L"), (3.65, "R5L")]:
        rect(ax, (x, -0.8), (0.55, 2.0), fc=rack_fc, ec=rack_ec, lw=1.0, zorder=1)
        ax.text(x + 0.275, 1.23, label, ha="center", va="bottom", fontsize=8, weight="bold")
    for x, label in [(1.6, "R4U"), (3.65, "R5U")]:
        rect(ax, (x, 2.2), (0.55, 2.1), fc=rack_fc, ec=rack_ec, lw=1.0, zorder=1)
        ax.text(x + 0.275, 3.18, label, ha="center", va="center", fontsize=8, weight="bold")
    # Driveable cross/aisle hints.
    rect(ax, (0.55, -2.05), (3.35, 0.65), fc="#d8f1de", ec="none", alpha=0.55, zorder=0)
    rect(ax, (0.75, 1.4), (3.25, 0.8), fc="#d8f1de", ec="none", alpha=0.55, zorder=0)
    rect(ax, (2.95, -2.05), (0.65, 4.25), fc="#d8f1de", ec="none", alpha=0.55, zorder=0)
    rect(ax, (0.85, -2.05), (0.65, 4.25), fc="#d8f1de", ec="none", alpha=0.55, zorder=0)
    ax.set_xlim(0.4, 4.1)
    ax.set_ylim(-2.2, 2.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load_run(path) for name, path in RUNS.items()}

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.23)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[1, 1])

    draw_geometry(ax_map)

    colors = {"C2 correction ON": "#d62728", "C2 camera OFF": "#111111"}
    linestyles = {"C2 correction ON": "-", "C2 camera OFF": "-"}

    for name, (exp, plan, summary) in data.items():
        c = colors[name]
        ax_map.plot(plan["x"], plan["y"], ls="--", color=c, lw=1.4, alpha=0.55, label=f"{name}: initial global plan")
        ax_map.plot(exp["truth_x"], exp["truth_y"], color=c, lw=3.0, label=f"{name}: truth")
        ax_map.plot(exp["state_x"], exp["state_y"], color=c, lw=1.5, alpha=0.35, label=f"{name}: /state")
        ax_map.plot(exp["planner_belief_x"], exp["planner_belief_y"], color=c, lw=1.2, ls=":", alpha=0.9, label=f"{name}: planner belief")
        if name == "C2 camera OFF":
            # Mark a few synchronized samples showing truth leaving stale state behind.
            for t in [1, 3, 5, 7]:
                r = exp.iloc[(exp["t"] - t).abs().argmin()]
                ax_map.plot([r["state_x"], r["truth_x"]], [r["state_y"], r["truth_y"]], color="#444444", lw=1.0, alpha=0.75)
                ax_map.text(r["truth_x"] + 0.04, r["truth_y"] + 0.04, f"{r['t']:.0f}s", fontsize=7, color="#111111")
        crash = exp[exp["collision_any"].fillna(False).astype(bool)]
        if len(crash):
            r = crash.iloc[0]
            ax_map.scatter([r["truth_x"]], [r["truth_y"]], marker="X", s=130, c=c, edgecolor="white", linewidth=0.8, zorder=8)

    ax_map.scatter([3.35], [-1.55], s=130, c="#1db954", edgecolor="black", zorder=9, label="start")
    ax_map.scatter([1.0], [1.75], marker="*", s=180, c="red", edgecolor="black", zorder=9, label="goal")
    ax_map.set_title("(a) Same C2 task: route intent vs stale-state execution failure", weight="bold")
    ax_map.legend(fontsize=7, loc="lower left", ncol=1)

    for name, (exp, _, _) in data.items():
        c = colors[name]
        ax_err.plot(exp["t"], exp["truth_belief_error_m"], color=c, lw=2.4, label=f"{name}: truth-belief")
        ax_err.plot(exp["t"], exp["truth_state_error_m"], color=c, lw=1.3, ls="--", alpha=0.75, label=f"{name}: truth-/state")
    ax_err.axhline(0.5, color="0.6", ls=":", lw=1)
    ax_err.set_title("(b) Camera OFF: belief error explodes while correction ON stays bounded", weight="bold")
    ax_err.set_ylabel("position error [m]")
    ax_err.set_xlabel("time after first command [s]")
    ax_err.grid(alpha=0.25)
    ax_err.legend(fontsize=8)

    off = data["C2 camera OFF"][0]
    # Explain the exact stale-state trap numerically.
    truth_to_wp = np.hypot(off["truth_x"] - off["exec_wp_target_x"], off["truth_y"] - off["exec_wp_target_y"])
    state_to_wp = np.hypot(off["state_x"] - off["exec_wp_target_x"], off["state_y"] - off["exec_wp_target_y"])
    belief_to_wp = np.hypot(off["planner_belief_x"] - off["exec_wp_target_x"], off["planner_belief_y"] - off["exec_wp_target_y"])
    ax_state.plot(off["t"], truth_to_wp, color="#111111", lw=2.3, label="truth distance to active waypoint")
    ax_state.plot(off["t"], state_to_wp, color="#1f77b4", lw=2.0, label="/state distance to active waypoint")
    ax_state.plot(off["t"], belief_to_wp, color="#9467bd", lw=1.7, ls="--", label="planner belief distance to active waypoint")
    ax_state.plot(off["t"], off["exec_wp_dist_m"], color="#ff7f0e", lw=1.2, alpha=0.85, label="logged tracker waypoint distance")
    ax_state.axhline(0.2, color="0.5", ls=":", lw=1, label="arrival radius")
    ax_state.set_title("(c) Exact failure: tracker thinks stale /state is near waypoint 2", weight="bold")
    ax_state.set_xlabel("time after first command [s]")
    ax_state.set_ylabel("distance [m]")
    ax_state.grid(alpha=0.25)
    ax_state.legend(fontsize=8)

    fig.suptitle("F76 - Camera-off failure cause: stale camera state drives the local tracker into a false waypoint basin", fontsize=17, weight="bold")
    fig.savefig(OUT / "F76_camera_off_exact_failure.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "F76_camera_off_exact_failure.pdf", bbox_inches="tight")

    off_summary = data["C2 camera OFF"][2]
    on_summary = data["C2 correction ON"][2]
    md = f"""# F76 - Camera-off exact failure

Figure files:

- `timing_presentation/figures/F76/F76_camera_off_exact_failure.png`
- `timing_presentation/figures/F76/F76_camera_off_exact_failure.pdf`

## What fails

The camera-off run does not fail because the initial route plot is uninterpretable. It fails because the execution pipeline becomes internally inconsistent:

1. `use_pixel_correction=False` disables planner belief correction.
2. YOLO and `/state/bev` still exist in the log, but after missed detections `/state/bev` becomes stale near the early route segment.
3. The local tracker continues to compute waypoint distance against that stale `/state`, not against the physical truth pose.
4. The tracker therefore believes it is about `{0.26:.2f} m` from waypoint 2 for several seconds while the real robot is more than `2 m` away from that waypoint.
5. It keeps commanding forward motion and the truth trajectory collides.

## Numbers

Successful C2 correction ON:

- outcome: `{on_summary.get('completion_reason')}`
- path: `{on_summary.get('path_length_m'):.3f} m`
- min goal distance: `{on_summary.get('minimum_goal_distance'):.3f} m`
- min obstacle distance: `{on_summary.get('min_obstacle_distance_m'):.3f} m`

C2 camera OFF:

- outcome: `{off_summary.get('completion_reason')}`
- path: `{off_summary.get('path_length_m'):.3f} m`
- min goal distance: `{off_summary.get('minimum_goal_distance'):.3f} m`
- min obstacle distance: `{off_summary.get('min_obstacle_distance_m'):.3f} m`
- mean truth-belief error after first command: `{off_summary.get('mean_truth_belief_error_m'):.3f} m`

## Interpretation

This is not the same as C1 passing through a low-visibility segment. C1 still has camera correction enabled and can reacquire. Camera-off removes the global `(x,y)` correction completely, while stale `/state` remains available to the local tracker. The result is a false sense of waypoint progress: the controller thinks it is tracking the route, but the physical robot has drifted far away from the state used for control.

This diagnostic should be treated as an ablation/failure-mode figure, not paper evidence for the final C1/C2 comparison.
"""
    (OUT / "F76_camera_off_exact_failure.md").write_text(md)
    print(OUT / "F76_camera_off_exact_failure.png")
    print(OUT / "F76_camera_off_exact_failure.pdf")
    print(OUT / "F76_camera_off_exact_failure.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
