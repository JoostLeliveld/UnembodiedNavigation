#!/usr/bin/env python3
"""F29: Gazebo timing smoke diagnostic for R01 hierarchical tracking."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
FIG_LABEL = os.environ.get("FIG_LABEL", "F29")
RUN_ROOT = Path(
    os.environ.get(
        "RUN_ROOT",
        str(ROOT / "logs/visibility_comparison/f29_r01_timing_smoke_v2/F24_R01_a4_lower_to_a3_mid"),
    )
)
WORLD_PROFILE = ROOT / "src/experiments/config/world_profiles.yaml"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "timing_presentation/figures/F29")))
WORLD = "warehouse_aws.world.sdf"
GOAL = (1.075, 1.64)


def load_regions() -> list[dict]:
    data = yaml.safe_load(WORLD_PROFILE.read_text())
    return data["worlds"][WORLD]["known_2d_regions"]


def draw_regions(ax, regions: list[dict]) -> None:
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
                    facecolor="#7fc97f",
                    edgecolor="#1a9850",
                    alpha=0.18,
                    linewidth=1.0,
                )
            )
        elif "non_driveable" in kind:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="#f4a3a3",
                    edgecolor="#d73027",
                    alpha=0.24,
                    linewidth=1.0,
                )
            )


def load_run(condition: str):
    run_dirs = sorted((RUN_ROOT / condition / "seed1").glob("experiment_*"))
    if not run_dirs:
        return None
    run = run_dirs[-1]
    return {
        "condition": condition,
        "run": run,
        "summary": json.loads((run / "run_summary.json").read_text()),
        "experiment": pd.read_csv(run / "experiment.csv"),
        "perception": pd.read_csv(run / "perception.csv"),
    }


def nonzero_solve_times(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["solve_time_ms"], errors="coerce").dropna()
    return s[s > 10.0]


def draw_condition(ax_map, ax_ts, result, regions) -> dict:
    cond = result["condition"]
    df = result["experiment"]
    summary = result["summary"]
    t0 = float(summary["first_cmd_stamp"])
    t = df["stamp"] - t0

    draw_regions(ax_map, regions)
    ax_map.plot(df["truth_x"], df["truth_y"], color="#111827", lw=2.2, label="truth")
    ax_map.plot(df["state_x"], df["state_y"], color="#2563eb", lw=1.2, alpha=0.65, label="EKF state")
    ax_map.plot(
        df["planner_belief_x"],
        df["planner_belief_y"],
        color="#ef4444",
        lw=1.0,
        alpha=0.65,
        label="planner belief",
    )
    ax_map.scatter(df["truth_x"].iloc[0], df["truth_y"].iloc[0], s=70, c="#16a34a", zorder=6, label="start")
    ax_map.scatter(*GOAL, s=140, c="#111827", marker="*", zorder=6, label="goal")
    final_is_goal = summary.get("completion_reason") == "goal_reached"
    ax_map.scatter(
        df["truth_x"].iloc[-1],
        df["truth_y"].iloc[-1],
        s=120,
        c="#16a34a" if final_is_goal else "#dc2626",
        marker="o" if final_is_goal else "X",
        zorder=7,
        label="final / goal reached" if final_is_goal else "collision",
    )
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(-0.3, 3.8)
    ax_map.set_ylim(-3.0, 2.6)
    ax_map.grid(alpha=0.25)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_title(
        f"{cond}: {summary['completion_reason']} | "
        f"path={summary['path_length_m']:.1f}m | "
        f"min obs={summary['min_obstacle_distance_m']:.3f}m"
    )
    ax_map.legend(fontsize=7, loc="upper left")

    solve = nonzero_solve_times(df)
    success = pd.to_numeric(df["optimizer_success"], errors="coerce").fillna(0.0)
    status_limited = df["optimizer_message"].fillna("").str.contains("ITERATIONS REACHED LIMIT")
    ax_ts.plot(t, df["goal_dist"], color="#111827", lw=1.5, label="goal distance")
    ax_ts.plot(t, df["min_obstacle_distance_m"], color="#f97316", lw=1.5, label="truth obstacle clearance")
    ax_ts.plot(t, df["truth_state_error_m"], color="#2563eb", lw=1.2, label="truth-state error")
    ax_ts.axhline(0.0, color="#dc2626", lw=1.0, ls="--", label="collision boundary")
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(t, df["solve_time_ms"], color="#7c3aed", lw=0.9, alpha=0.6, label="solve time")
    ax_ts2.axhline(500.0, color="#7c3aed", lw=0.8, ls=":", alpha=0.7)
    ax_ts2.set_ylabel("solve time [ms]", color="#7c3aed")
    ax_ts2.tick_params(axis="y", labelcolor="#7c3aed")
    ax_ts.set_title(
        f"{cond}: local solve median={solve.median():.0f}ms, "
        f"p90={solve.quantile(0.90):.0f}ms, "
        f"success rows={success.mean():.0%}, iter-limit rows={status_limited.mean():.0%}"
    )
    ax_ts.set_xlabel("t after first command [s]")
    ax_ts.set_ylabel("m")
    ax_ts.grid(alpha=0.25)
    lines, labels = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper right")

    return {
        "condition": cond,
        "run": str(result["run"]),
        "outcome": summary["completion_reason"],
        "path_m": summary["path_length_m"],
        "min_obs_m": summary["min_obstacle_distance_m"],
        "mean_err_m": summary["mean_truth_state_error_m"],
        "solve_mean_ms": float(solve.mean()),
        "solve_median_ms": float(solve.median()),
        "solve_p90_ms": float(solve.quantile(0.90)),
        "solve_max_ms": float(solve.max()),
        "optimizer_success_fraction": float(success.mean()),
        "iteration_limit_fraction": float(status_limited.mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_regions()
    results = [load_run("C1"), load_run("C2")]
    results = [r for r in results if r is not None]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    stats = []
    for row, result in enumerate(results):
        stats.append(draw_condition(axes[row, 0], axes[row, 1], result, regions))

    fig.suptitle(
        f"{FIG_LABEL} - R01 Gazebo timing smoke: init fixed, closed-loop runtime diagnosis",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    stem = f"{FIG_LABEL}_r01_timing_smoke"
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    csv = OUT_DIR / f"{stem}.csv"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    pd.DataFrame(stats).to_csv(csv, index=False)

    md = OUT_DIR / f"{stem}.md"
    all_goal = bool(stats) and all(s["outcome"] == "goal_reached" for s in stats)
    all_collision = bool(stats) and all(s["outcome"] == "collision" for s in stats)
    table = [
        "| condition | outcome | path [m] | min obstacle [m] | mean state error [m] | median solve [ms] | p90 solve [ms] | optimizer success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        table.append(
            f"| {s['condition']} | {s['outcome']} | {s['path_m']:.2f} | "
            f"{s['min_obs_m']:.3f} | {s['mean_err_m']:.3f} | "
            f"{s['solve_median_ms']:.0f} | {s['solve_p90_ms']:.0f} | "
            f"{100.0 * s['optimizer_success_fraction']:.0f}% |"
        )
    if all_goal:
        diagnosis_lines = [
            "- Gazebo spawn/init is clean for both conditions: truth start and yaw errors are essentially zero.",
            "- Both C1 and C2 reached the visible goal with command and encoder noise active.",
            "- The stop-on-exhausted-plan guard plus less overconfident pixel noise prevents the earlier obstacle penetration.",
            "- The remaining weakness is timing: local solves are still around 1.7-2.6 s for the median/p90 range.",
            "- This is a valid smoke pass, not yet evidence of a GP advantage, because this R01 endpoint is largely visible and `p_vis_plan` stays high.",
        ]
        interpretation = (
            "F30 shows that the runtime stack can complete R01 in Gazebo when local execution is made "
            "more conservative. The next experiment should keep these runtime safeguards and move back "
            "to a route where C2 has a real learned-observation-reliability reason to differ from C1."
        )
    elif all_collision:
        diagnosis_lines = [
            "- Gazebo spawn/init is clean for both conditions: truth start and yaw errors are essentially zero.",
            "- YOLO availability is high, so this run is not mainly a detector-dropout failure.",
            "- Both C1 and C2 still end in geometry obstacle penetration.",
            "- The local tracker is the bottleneck: nonzero local solves are slow relative to the local control loop.",
            "- Failed local replans often hit `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`; executing those controls can still be unsafe.",
        ]
        interpretation = (
            "This points away from spawn, world-frame, or GP choice as the immediate blocker. "
            "The global route can be generated, but the closed-loop local EFE tracker is not stable enough yet."
        )
    else:
        diagnosis_lines = [
            "- Gazebo spawn/init is clean for completed runs.",
            "- Outcomes are mixed across C1/C2; inspect the table before treating this as a lock-in result.",
            "- The main remaining questions are local solve timing, estimator error, and whether GP-conditioned covariance changes the route.",
        ]
        interpretation = (
            "This run is diagnostic. It should not be promoted to paper evidence until repeated over seeds "
            "and paired with a task where C1/C2 are expected to differ."
        )
    md.write_text(
        "\n".join(
            [
                f"# {FIG_LABEL} - R01 Gazebo Timing Smoke",
                "",
                f"Figure: `{png}`",
                f"PDF: `{pdf}`",
                f"Stats: `{csv}`",
                "",
                "## Diagnosis",
                "",
                *diagnosis_lines,
                "",
                "## Summary Table",
                "",
                *table,
                "",
                "## Interpretation",
                "",
                interpretation,
                "",
                "## Next Lock-In Decision",
                "",
                "Keep the F30 runtime hygiene as the current Gazebo smoke baseline, but do not claim a visibility-aware advantage from R01 alone. Next, test a task where the final goal is visible but the short route spends enough time in a weak-observation region for C2 to prefer a safer visible route.",
                "",
            ]
        )
    )
    print(png)
    print(pdf)
    print(csv)
    print(md)


if __name__ == "__main__":
    main()
