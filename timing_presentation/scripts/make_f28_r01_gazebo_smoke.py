#!/usr/bin/env python3
"""F28: Gazebo smoke diagnostic for F24 R01 — FIX A: separate local nogo margin + timeout.

F25: nogo_safe_distance=0.13, both conditions crashed (obstacle penetration, 4 cm).
F26: nogo_safe_distance=0.30, global solve stalled (0 plans in 133 s).
F27: nogo_safe_distance=0.20, local_optimizer_maxiter=25.
     C1: wall crash (homography belief-y outliers). C2: global solve timed out (>260 s).
F28: FIX A — local_nogo_safe_distance=0.13 (decouples local tracker from global margin 0.20),
     run_timeout_after_first_cmd_s=200 (budget after ~46 s C2 global solve).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
RUN_ROOT = ROOT / "logs/visibility_comparison/f28_r01_gazebo_smoke_v1/F24_R01_a4_lower_to_a3_mid"
WORLD_PROFILE = ROOT / "src/experiments/config/world_profiles.yaml"
OUT_DIR = ROOT / "timing_presentation/figures/F28"
WORLD = "warehouse_aws.world.sdf"


def load_run(condition: str):
    cond_dir = RUN_ROOT / condition / "seed1"
    exp_dirs = list(cond_dir.glob("experiment_*"))
    if not exp_dirs:
        return None
    run_dir = exp_dirs[0]
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if not summary.get("completed") or summary.get("completion_reason") == "interrupted":
        return None
    exp = pd.read_csv(run_dir / "experiment.csv")
    return exp, summary, run_dir


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
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#79c779", edgecolor="#149447",
                                   alpha=0.18, linewidth=1.0))
        elif "non_driveable" in kind:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#f28b82", edgecolor="#d93025",
                                   alpha=0.22, linewidth=1.0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_regions()

    c1_result = load_run("C1")
    c2_result = load_run("C2")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_bev, ax_belief, ax_ts, ax_solve = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if c1_result is not None:
        c1_exp, c1_sum, c1_dir = c1_result
        t0 = float(c1_sum["first_cmd_stamp"])
        t = c1_exp["stamp"] - t0

        # BEV
        draw_regions(ax_bev, regions)
        ax_bev.plot(c1_exp["truth_x"], c1_exp["truth_y"],
                    color="#111827", lw=2.2, label="truth")
        ax_bev.plot(c1_exp["planner_belief_x"],
                    c1_exp["planner_belief_y"].clip(-6, 6),
                    color="#ef4444", lw=1.2, alpha=0.7, label="belief y (clipped ±6 m)")
        ax_bev.scatter(c1_exp["truth_x"].iloc[0], c1_exp["truth_y"].iloc[0],
                       s=80, c="#16a34a", zorder=6, label="start")
        ax_bev.scatter(1.075, 1.64, s=150, c="#111827", marker="*", zorder=6, label="goal")
        ax_bev.scatter(c1_exp["truth_x"].iloc[-1], c1_exp["truth_y"].iloc[-1],
                       s=130, c="#dc2626", marker="X", zorder=7, label="wall crash")
        ax_bev.axhline(4.92, color="#dc2626", lw=1.5, ls="--", alpha=0.6,
                       label="north wall (y=4.92)")
        ax_bev.set_xlim(-0.5, 4.1)
        ax_bev.set_ylim(-3.2, 5.5)
        ax_bev.set_aspect("equal", adjustable="box")
        ax_bev.set_title(
            f"C1 BEV — wall crash at t={c1_sum['elapsed_after_first_cmd_s']:.1f}s, "
            f"path={c1_sum['path_length_m']:.1f}m\n"
            f"min_obs={c1_sum['min_obstacle_distance_m']:.3f}m (+ve ✓)  "
            f"crash: {c1_sum['collision_reason']}"
        )
        ax_bev.legend(fontsize=7, loc="upper left")
        ax_bev.set_xlabel("x [m]")
        ax_bev.set_ylabel("y [m]")
        ax_bev.grid(alpha=0.25)

        # Belief y divergence
        ax_belief.plot(t, c1_exp["truth_y"], color="#111827", lw=2, label="truth y")
        ax_belief.plot(t, c1_exp["planner_belief_y"].clip(-6, 6),
                       color="#ef4444", lw=1.2, alpha=0.85,
                       label="belief y (clipped ±6 m)")
        ax_belief.axhline(4.92, color="#dc2626", lw=1.2, ls="--", label="north wall (4.92 m)")
        crash_t = float(c1_sum["first_crash_stamp"]) - t0
        ax_belief.axvline(crash_t, color="#dc2626", lw=1.0, ls=":",
                          label=f"crash t={crash_t:.1f}s")
        ax_belief.set_title(
            "Y-axis divergence: homography outliers snap belief to wrong y\n"
            "(planner_belief_y oscillates 0.3 ↔ 14+ while truth_y → 4.8)"
        )
        ax_belief.set_xlabel("t after first cmd [s]")
        ax_belief.set_ylabel("y [m]")
        ax_belief.legend(fontsize=7)
        ax_belief.grid(alpha=0.25)

        # Error and clearance timeseries
        ax_ts.plot(t, c1_exp["truth_state_error_m"],
                   color="#2563eb", lw=1.5, label="truth-state err (m)")
        ax_ts.plot(t, c1_exp["truth_belief_error_m"],
                   color="#7c3aed", lw=1.2, alpha=0.8, label="truth-belief err (m)")
        ax_ts.plot(t, c1_exp["min_obstacle_distance_m"],
                   color="#f97316", lw=1.4, label="obstacle clearance (m)")
        ax_ts.axhline(0.0, color="#dc2626", lw=1.0, ls="--", alpha=0.6,
                      label="forbidden boundary (0 m)")
        ax_ts.axvline(crash_t, color="#dc2626", lw=1.0, ls=":", label="crash")
        ax_ts.set_title("C1 errors and obstacle clearance")
        ax_ts.set_xlabel("t after first cmd [s]")
        ax_ts.set_ylabel("m")
        ax_ts.legend(fontsize=7)
        ax_ts.grid(alpha=0.25)

        # Solve time distribution
        solve_times = c1_exp["solve_time_ms"].dropna()
        solve_times = solve_times[solve_times > 10]
        ax_solve.hist(solve_times, bins=20, color="#2563eb", alpha=0.7, edgecolor="white")
        ax_solve.axvline(solve_times.mean(), color="#dc2626", lw=1.5, ls="--",
                         label=f"mean {solve_times.mean():.0f} ms")
        ax_solve.axvline(solve_times.median(), color="#7c3aed", lw=1.2, ls="-.",
                         label=f"median {solve_times.median():.0f} ms")
        ax_solve.axvline(250, color="#f97316", lw=1.2, ls=":",
                         label="4 Hz budget (250 ms)")
        ax_solve.set_title(
            f"C1 local solve times (n={len(solve_times)}, maxiter=25)\n"
            f"F27 mean was 730 ms"
        )
        ax_solve.set_xlabel("solve time [ms]")
        ax_solve.set_ylabel("count")
        ax_solve.legend(fontsize=7)
        ax_solve.grid(alpha=0.25)
    else:
        for ax in [ax_bev, ax_belief, ax_ts, ax_solve]:
            ax.text(0.5, 0.5, "C1: no completed run",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)

    fig.suptitle(
        "F28 — FIX A: local_nogo_safe_dist=0.13, run_timeout=200s | "
        "C1: wall crash (homography belief-y outliers, pending FIX B) | C2: pending local tracker fix",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png = OUT_DIR / "F28_r01_gazebo_smoke.png"
    pdf = OUT_DIR / "F28_r01_gazebo_smoke.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(png)
    print(pdf)

    # Write markdown note
    c1_note = "no completed run"
    if c1_result is not None:
        c1_exp, c1_sum, _ = c1_result
        solve_times = c1_exp["solve_time_ms"].dropna()
        solve_times = solve_times[solve_times > 10]
        c1_note = (
            f"collision ({c1_sum['collision_reason']}), "
            f"path={c1_sum['path_length_m']:.2f}m, "
            f"elapsed={c1_sum['elapsed_after_first_cmd_s']:.1f}s, "
            f"min_obstacle={c1_sum['min_obstacle_distance_m']:.3f}m, "
            f"mean_solve={solve_times.mean():.0f}ms, "
            f"truth_state_err={c1_sum['mean_truth_state_error_m']:.3f}m"
        )

    c2_note = "no completed run"
    if c2_result is not None:
        c2_exp, c2_sum, _ = c2_result
        c2_note = (
            f"outcome={c2_sum.get('completion_reason','?')}, "
            f"path={c2_sum.get('path_length_m', float('nan')):.2f}m, "
            f"elapsed={c2_sum.get('elapsed_after_first_cmd_s', float('nan')):.1f}s, "
            f"min_obstacle={c2_sum.get('min_obstacle_distance_m', float('nan')):.3f}m"
        )

    md = OUT_DIR / "F28_r01_gazebo_smoke.md"
    md.write_text("\n".join([
        "# F28 - R01 Gazebo Smoke Diagnostic",
        "",
        "Config: `scripts/visibility_comparison/aws_f28_r01_gazebo_smoke_config.yaml`",
        "Changes vs F27: `local_nogo_safe_distance=0.13` (FIX A), `run_timeout_after_first_cmd_s 160->200`.",
        "",
        "## Results",
        "",
        "| condition | outcome | note |",
        "|---|---|---|",
        f"| C1 | {c1_note} |",
        f"| C2 | {c2_note} |",
        "",
        "## Fix Applied (FIX A)",
        "",
        "- `local_nogo_safe_distance: 0.13` — local tracker uses tighter margin (0.13 m) vs global planner (0.20 m).",
        "  Global planner plans safe routes with 0.20 m margin; local tracker can now re-execute those waypoints.",
        "- `run_timeout_after_first_cmd_s: 200` — allows C2 local tracking after ~46 s global solve + Gazebo overhead.",
        "",
        "## Outstanding issues (FIX B pending)",
        "",
        "### C1: Belief-y divergence (homography outliers)",
        "As the robot approaches y≈4.5 (near north wall at 4.92), `planner_belief_y` oscillates",
        "between ~0.3 m and 14+ m. Root causes: (1) outer walls not in known_2d_regions no-go layer,",
        "(2) homography back-projection gives invalid y near the northern camera boundary.",
        "FIX B: add world-bounds validity gate in pixel correction node.",
        "",
        f"Figure: `{png}`",
        f"PDF: `{pdf}`",
        "",
    ]))
    print(md)


if __name__ == "__main__":
    main()
