#!/usr/bin/env python3
"""F26: Gazebo smoke diagnostic for F24 R01 — margin + local-solve fix.

F25 crashed immediately (geometry penetration) due to nogo_safe_distance=0.13 m
being too small for ~0.28 m truth-state error, and local_optimizer_maxiter=60
producing ~2 s solve times. F26 fixes both: nogo_safe_distance=0.30, maxiter=25.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Rectangle


ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
RUN_ROOT = ROOT / "logs/visibility_comparison/f26_r01_gazebo_smoke_v1/F24_R01_a4_lower_to_a3_mid"
WORLD_PROFILE = ROOT / "src/experiments/config/world_profiles.yaml"
OUT_DIR = ROOT / "timing_presentation/figures/F26"
WORLD = "warehouse_aws.world.sdf"


def load_run(condition: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    run_dir = next((RUN_ROOT / condition / "seed1").glob("experiment_*"))
    exp = pd.read_csv(run_dir / "experiment.csv")
    plans = pd.read_csv(run_dir / "plan_samples.csv")
    summary = json.loads((run_dir / "run_summary.json").read_text())
    return exp, plans, summary, run_dir


def load_regions() -> list[dict]:
    data = yaml.safe_load(WORLD_PROFILE.read_text())
    return data["worlds"][WORLD]["known_2d_regions"]


def draw_regions(ax, regions: list[dict]) -> None:
    first_drive = True
    first_forbid = True
    for r in regions:
        x = float(r["xmin"])
        y = float(r["ymin"])
        w = float(r["xmax"]) - x
        h = float(r["ymax"]) - y
        kind = r.get("type", "")
        if kind == "traversable":
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="#79c779",
                    edgecolor="#149447",
                    alpha=0.18,
                    linewidth=1.0,
                    label="known driveable floor" if first_drive else None,
                )
            )
            first_drive = False
        elif "non_driveable" in kind:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="#f28b82",
                    edgecolor="#d93025",
                    alpha=0.22,
                    linewidth=1.0,
                    label="known forbidden / staging" if first_forbid else None,
                )
            )
            first_forbid = False


def plot_plan_group(ax, plans: pd.DataFrame, stamp: float, color: str, label: str) -> None:
    g = plans[plans["plan_stamp"] == stamp].sort_values("point_idx")
    if len(g) > 1:
        ax.plot(g["x"], g["y"], color=color, lw=1.6, ls="--", alpha=0.9, label=label)


def first_after_first_cmd_plans(plans: pd.DataFrame, first_cmd: float) -> tuple[float, float]:
    stamps = sorted(float(s) for s in plans["plan_stamp"].dropna().unique())
    after = [s for s in stamps if s >= first_cmd - 1e-6]
    return (after[0] if after else stamps[0], after[-1] if after else stamps[-1])


def draw_condition(ax, label: str, exp: pd.DataFrame, plans: pd.DataFrame, summary: dict, regions: list[dict]) -> None:
    draw_regions(ax, regions)
    first_stamp, last_stamp = first_after_first_cmd_plans(plans, float(summary["first_cmd_stamp"]))
    plot_plan_group(ax, plans, first_stamp, "#7b61ff", f"first local plan t={first_stamp:.1f}s")
    plot_plan_group(ax, plans, last_stamp, "#ff9800", f"last local plan t={last_stamp:.1f}s")
    ax.plot(exp["truth_x"], exp["truth_y"], color="#111827", lw=2.2, label="truth")
    ax.plot(exp["planner_belief_x"], exp["planner_belief_y"], color="#ef4444", lw=1.4, alpha=0.85, label="planner belief")
    ax.scatter(exp["truth_x"].iloc[0], exp["truth_y"].iloc[0], s=75, c="#16a34a", zorder=6, label="start")
    ax.scatter(exp["goal_x"].dropna().iloc[-1], exp["goal_y"].dropna().iloc[-1], s=130, c="#111827", marker="*", zorder=6, label="goal")
    ax.scatter(exp["truth_x"].iloc[-1], exp["truth_y"].iloc[-1], s=120, c="#dc2626", marker="X", zorder=7, label="collision")
    ax.set_title(
        f"{label}: {summary['completion_reason']}, path={summary['path_length_m']:.1f} m, "
        f"min d_goal={summary['minimum_goal_distance']:.2f} m"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, 4.1)
    ax.set_ylim(-3.0, 2.3)
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=7)


def draw_timeseries(ax, label: str, exp: pd.DataFrame, summary: dict) -> None:
    t0 = float(summary["first_cmd_stamp"])
    t = exp["stamp"] - t0
    ax.plot(t, exp["goal_dist"], color="#111827", lw=1.8, label="goal distance")
    ax.plot(t, exp["truth_state_error_m"], color="#2563eb", lw=1.3, label="truth-state err")
    ax.plot(t, exp["truth_belief_error_m"], color="#ef4444", lw=1.3, label="truth-belief err")
    ax.plot(t, exp["min_obstacle_distance_m"], color="#f97316", lw=1.4, label="obs clearance")
    ax.axhline(0.0, color="#dc2626", lw=1.2, ls="--", alpha=0.7, label="forbidden boundary (0 m)")
    if summary.get("first_crash_stamp") and not pd.isna(summary["first_crash_stamp"]):
        ax.axvline(float(summary["first_crash_stamp"]) - t0, color="#dc2626", lw=1.2, ls=":", label="collision")
    ax.set_title(label)
    ax.set_xlabel("time after first command [s]")
    ax.set_ylabel("m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_regions()
    c1, c1_plan, c1_sum, c1_dir = load_run("C1")
    c2, c2_plan, c2_sum, c2_dir = load_run("C2")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    draw_condition(ax1, "C1 constant-R", c1, c1_plan, c1_sum, regions)
    draw_condition(ax2, "C2 GP visibility-aware", c2, c2_plan, c2_sum, regions)
    draw_timeseries(ax3, "C1 runtime traces", c1, c1_sum)
    draw_timeseries(ax4, "C2 runtime traces", c2, c2_sum)

    fig.suptitle(
        "F26 - R01 Gazebo smoke: margin + local-solve fix (nogo_safe_dist=0.30, local_maxiter=25)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = OUT_DIR / "F26_r01_gazebo_smoke.png"
    pdf = OUT_DIR / "F26_r01_gazebo_smoke.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)

    md = OUT_DIR / "F26_r01_gazebo_smoke.md"
    md.write_text(
        "\n".join(
            [
                "# F26 - R01 Gazebo Smoke Diagnostic",
                "",
                f"Source C1 run: `{c1_dir}`",
                f"Source C2 run: `{c2_dir}`",
                "",
                "Config changes vs F25: nogo_safe_distance 0.13->0.30, local_optimizer_maxiter 60->25.",
                "",
                "| condition | outcome | path m | min goal m | mean truth-state err m | min obstacle margin m | mean solve ms | mean p_vis_plan |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| C1 | {c1_sum['completion_reason']} | {c1_sum['path_length_m']:.2f} | {c1_sum['minimum_goal_distance']:.2f} | {c1_sum['mean_truth_state_error_m']:.3f} | {c1_sum['min_obstacle_distance_m']:.3f} | {c1_sum['mean_solve_time_ms']:.0f} | {c1_sum['mean_p_vis_plan']:.2f} |",
                f"| C2 | {c2_sum['completion_reason']} | {c2_sum['path_length_m']:.2f} | {c2_sum['minimum_goal_distance']:.2f} | {c2_sum['mean_truth_state_error_m']:.3f} | {c2_sum['min_obstacle_distance_m']:.3f} | {c2_sum['mean_solve_time_ms']:.0f} | {c2_sum['mean_p_vis_plan']:.2f} |",
                "",
                "Success criteria: min_obstacle_distance_m >= 0 for both, goal distance decreasing, mean solve < 500 ms.",
                "",
                f"Figure: `{png}`",
                f"PDF: `{pdf}`",
                "",
            ]
        )
    )
    print(png)
    print(pdf)
    print(md)


if __name__ == "__main__":
    main()
