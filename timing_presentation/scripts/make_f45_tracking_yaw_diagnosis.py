#!/usr/bin/env python3
"""F45: controlled tracking/yaw/timing diagnostic for the B1 Gazebo smoke.

This figure reads the F45 logs only. It does not run Gazebo or change any
runtime configuration. The intent is to diagnose the first controlled iteration
after wiring local tracking yaw and waypoint telemetry into the logger.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
LOG_ROOT = ROOT / "logs/visibility_comparison/f45_b1_tracking_yaw_v2"
OUT_DIR = ROOT / "timing_presentation/figures/F45"
TASK = "F31_b1_apron_a3_mid"

COL = {"C1": "#2563eb", "C2": "#dc2626"}
LIGHT = {"C1": "#93c5fd", "C2": "#fca5a5"}

RACKS = [
    ("R1", -4.30, -3.75, -0.80, 1.25),
    ("R2", -2.25, -1.70, -0.80, 1.25),
    ("R3", -0.20, 0.35, -0.80, 1.25),
    ("R4", 1.75, 2.30, -0.80, 1.25),
    ("R5", 3.85, 4.40, -0.80, 1.25),
    ("R1", -4.30, -3.75, 2.20, 4.25),
    ("R2", -2.25, -1.70, 2.20, 4.25),
    ("R3", -0.20, 0.35, 2.20, 4.25),
    ("R4", 1.75, 2.30, 2.20, 4.25),
    ("R5", 3.85, 4.40, 2.20, 4.25),
]

DRIVEABLE_REGIONS = [
    (-4.55, 4.80, 4.05, 4.80),
    (-4.55, 4.80, 1.45, 2.50),
    (-4.55, 4.80, -3.45, -2.55),
    (-4.55, -3.65, -2.80, 4.45),
    (-2.55, -1.45, -2.80, 4.45),
    (-0.50, 0.65, -2.80, 4.45),
    (1.45, 2.60, -2.80, 4.45),
    (3.55, 4.70, -2.80, 4.45),
]


@dataclass
class Run:
    cond: str
    exp_dir: Path
    exp: pd.DataFrame
    plan: pd.DataFrame
    perception: pd.DataFrame
    summary: dict


def read_run(cond: str) -> Run:
    seed_dir = LOG_ROOT / TASK / cond / "seed1"
    exp_dirs = sorted(seed_dir.glob("experiment_*"))
    if not exp_dirs:
        raise FileNotFoundError(seed_dir)
    exp_dir = exp_dirs[-1]
    exp = pd.read_csv(exp_dir / "experiment.csv")
    plan = pd.read_csv(exp_dir / "plan_samples.csv")
    perception = pd.read_csv(exp_dir / "perception.csv")
    summary = json.loads((exp_dir / "run_summary.json").read_text())
    return Run(cond, exp_dir, exp, plan, perception, summary)


def finite_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def first_cmd_stamp(exp: pd.DataFrame) -> float:
    if "cmd_v" not in exp:
        return float(exp["stamp"].iloc[0])
    moving = exp[pd.to_numeric(exp["cmd_v"], errors="coerce").abs() > 0.01]
    if len(moving):
        return float(moving["stamp"].iloc[0])
    return float(exp["stamp"].iloc[0])


def rel_t(run: Run, col: str = "stamp") -> np.ndarray:
    return pd.to_numeric(run.exp[col], errors="coerce").to_numpy() - first_cmd_stamp(run.exp)


def angle_abs_diff(a: pd.Series, b: pd.Series) -> np.ndarray:
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return np.abs(np.arctan2(np.sin(aa - bb), np.cos(aa - bb)))


def first_plan(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return plan
    stamp = plan["plan_stamp"].min()
    return plan[plan["plan_stamp"] == stamp].sort_values("point_idx")


def local_endpoints(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return plan
    idx = plan.groupby("plan_stamp")["point_idx"].idxmax()
    return plan.loc[idx].sort_values("plan_stamp")


def draw_layout(ax):
    for xmin, xmax, ymin, ymax in DRIVEABLE_REGIONS:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fc="#22c55e",
                ec="#15803d",
                alpha=0.10,
                lw=0.8,
                zorder=0,
            )
        )
    for name, xmin, xmax, ymin, ymax in RACKS:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fc="#f59e0b",
                ec="#92400e",
                alpha=0.28,
                lw=1.0,
                zorder=1,
            )
        )
        if name in {"R4", "R5"}:
            ax.text((xmin + xmax) / 2, ymax + 0.03, name, ha="center", va="bottom", fontsize=7)


def draw_map(ax, runs: list[Run]):
    draw_layout(ax)
    for run in runs:
        exp = run.exp
        plan0 = first_plan(run.plan)
        ends = local_endpoints(run.plan)
        ax.plot(
            plan0["x"],
            plan0["y"],
            "--",
            color=LIGHT[run.cond],
            lw=1.5,
            alpha=0.9,
            label=f"{run.cond} first global plan",
            zorder=2,
        )
        if len(ends) > 2:
            ax.scatter(
                ends["x"],
                ends["y"],
                s=10,
                color=COL[run.cond],
                alpha=0.25,
                zorder=3,
                label=f"{run.cond} local endpoints",
            )
        pts = exp[["truth_x", "truth_y"]].dropna()
        ax.plot(
            pts["truth_x"],
            pts["truth_y"],
            color=COL[run.cond],
            lw=2.3,
            label=f"{run.cond} truth",
            zorder=4,
        )
        if len(pts):
            ax.scatter(
                pts["truth_x"].iloc[-1],
                pts["truth_y"].iloc[-1],
                marker="X",
                s=130,
                color=COL[run.cond],
                edgecolor="white",
                lw=0.8,
                zorder=5,
            )
    exp0 = runs[0].exp
    ax.scatter(exp0["truth_x"].iloc[0], exp0["truth_y"].iloc[0], s=110, color="#16a34a", zorder=6, label="start")
    ax.scatter(exp0["goal_x"].dropna().iloc[-1], exp0["goal_y"].dropna().iloc[-1], s=170, color="#111827", marker="*", zorder=6, label="goal")
    ax.set_title("(a) First route, local endpoints, executed trajectory")
    ax.set_xlim(0.2, 4.8)
    ax.set_ylim(-3.6, 2.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=7, ncol=1)


def summarize(run: Run) -> dict:
    exp = run.exp
    solve = finite_series(exp, "solve_time_ms")
    solve = solve[solve > 10]
    nfev = finite_series(exp, "optimizer_nfev")
    nfev = nfev[nfev > 0]
    msg = exp.get("optimizer_message", pd.Series(dtype=object)).astype(str)
    maxiter_frac = float(msg.str.contains("ITERATIONS REACHED LIMIT", na=False).mean()) if len(msg) else math.nan
    yaw_truth_belief = angle_abs_diff(exp.get("truth_yaw", pd.Series(dtype=float)), exp.get("planner_belief_yaw", pd.Series(dtype=float)))
    exec_yaw = finite_series(exp, "exec_yaw_error").abs()
    meas = finite_series(exp, "measurement_available")
    stale = finite_series(exp, "planner_belief_age_s")
    pixel_age = finite_series(exp, "planner_pixel_correction_age_s")
    perc_det = finite_series(run.perception, "detected")
    yolo_ms = finite_series(run.perception, "yolo_inference_ms")
    total_latency = finite_series(run.perception, "detector_total_latency_s")
    return {
        "outcome": run.summary.get("completion_reason"),
        "path_m": float(run.summary.get("path_length_m", math.nan) or math.nan),
        "min_goal_m": float(run.summary.get("minimum_goal_distance", math.nan) or math.nan),
        "min_obs_m": float(run.summary.get("min_obstacle_distance_m", math.nan) or math.nan),
        "mean_truth_state_error_m": float(run.summary.get("mean_truth_state_error_m", math.nan) or math.nan),
        "solve_mean_ms": float(solve.drop_duplicates().mean()) if len(solve) else math.nan,
        "solve_p90_ms": float(solve.drop_duplicates().quantile(0.9)) if len(solve) else math.nan,
        "nfev_mean": float(nfev.mean()) if len(nfev) else math.nan,
        "ms_per_eval": float(solve.drop_duplicates().mean() / nfev.mean()) if len(solve) and len(nfev) else math.nan,
        "maxiter_frac": maxiter_frac,
        "yaw_truth_belief_mean": float(np.nanmean(yaw_truth_belief)) if len(yaw_truth_belief) else math.nan,
        "yaw_truth_belief_p90": float(np.nanpercentile(yaw_truth_belief, 90)) if len(yaw_truth_belief) else math.nan,
        "exec_yaw_mean": float(exec_yaw.mean()) if len(exec_yaw) else math.nan,
        "exec_yaw_p90": float(exec_yaw.quantile(0.9)) if len(exec_yaw) else math.nan,
        "measurement_frac": float(meas.mean()) if len(meas) else math.nan,
        "belief_age_p90_s": float(stale.quantile(0.9)) if len(stale) else math.nan,
        "belief_age_max_s": float(stale.max()) if len(stale) else math.nan,
        "pixel_age_p90_s": float(pixel_age.quantile(0.9)) if len(pixel_age) else math.nan,
        "pixel_age_max_s": float(pixel_age.max()) if len(pixel_age) else math.nan,
        "detected_frac": float(perc_det.mean()) if len(perc_det) else math.nan,
        "yolo_mean_ms": float(yolo_ms.mean()) if len(yolo_ms) else math.nan,
        "detector_latency_p90_s": float(total_latency.quantile(0.9)) if len(total_latency) else math.nan,
    }


def plot_timeseries(ax, run: Run, fields: list[tuple[str, str]], title: str, ylabel: str):
    t = rel_t(run)
    for field, label in fields:
        if field not in run.exp:
            continue
        y = finite_series(run.exp, field).to_numpy()
        ax.plot(t, y, lw=1.2, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time after first command [s]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")


def make_figure(runs: list[Run], stats: dict[str, dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.45, 1.0, 1.0])

    ax_map = fig.add_subplot(gs[:, 0])
    draw_map(ax_map, runs)

    ax_goal = fig.add_subplot(gs[0, 1])
    for run in runs:
        plot_t = rel_t(run)
        ax_goal.plot(plot_t, finite_series(run.exp, "goal_dist"), color=COL[run.cond], lw=1.6, label=f"{run.cond} goal dist")
        ax_goal.plot(plot_t, finite_series(run.exp, "min_obstacle_distance_m"), color=COL[run.cond], lw=1.0, ls="--", alpha=0.75, label=f"{run.cond} obs margin")
    ax_goal.axhline(0.25, color="#111827", ls=":", lw=1.0, label="goal radius")
    ax_goal.axhline(0.0, color="#dc2626", ls=":", lw=1.0, label="collision")
    ax_goal.set_title("(b) Goal progress and obstacle margin")
    ax_goal.set_xlabel("time after first command [s]")
    ax_goal.set_ylabel("m")
    ax_goal.grid(True, alpha=0.25)
    ax_goal.legend(fontsize=7, ncol=2)

    ax_solve = fig.add_subplot(gs[0, 2])
    for run in runs:
        t = rel_t(run)
        solve = finite_series(run.exp, "solve_time_ms")
        success = finite_series(run.exp, "optimizer_success")
        ax_solve.plot(t, solve, color=COL[run.cond], lw=1.0, alpha=0.85, label=f"{run.cond} solve")
        bad = success == 0
        ax_solve.scatter(t[bad], solve[bad], color=COL[run.cond], s=8, alpha=0.35, marker="x")
    ax_solve.set_title("(c) Local solve time (x = non-success)")
    ax_solve.set_xlabel("time after first command [s]")
    ax_solve.set_ylabel("solve time [ms]")
    ax_solve.grid(True, alpha=0.25)
    ax_solve.legend(fontsize=7)

    ax_wp = fig.add_subplot(gs[1, 1])
    for run in runs:
        t = rel_t(run)
        ax_wp.step(t, finite_series(run.exp, "exec_wp_idx"), where="post", color=COL[run.cond], lw=1.2, label=f"{run.cond} wp idx")
        ax_wp.plot(t, finite_series(run.exp, "exec_wp_dist_m"), color=COL[run.cond], ls="--", lw=1.0, alpha=0.8, label=f"{run.cond} wp dist")
    ax_wp.axhline(0.35, color="#111827", ls=":", lw=1.0, label="arrival radius")
    ax_wp.set_title("(d) Waypoint progress")
    ax_wp.set_xlabel("time after first command [s]")
    ax_wp.set_ylabel("idx / distance [m]")
    ax_wp.grid(True, alpha=0.25)
    ax_wp.legend(fontsize=7, ncol=2)

    ax_yaw = fig.add_subplot(gs[1, 2])
    for run in runs:
        t = rel_t(run)
        exec_err = finite_series(run.exp, "exec_yaw_error").abs()
        truth_state = finite_series(run.exp, "yaw_error_truth_state_rad").abs()
        truth_belief = finite_series(run.exp, "yaw_error_truth_belief_rad").abs()
        ax_yaw.plot(t, exec_err, color=COL[run.cond], lw=1.3, label=f"{run.cond} tracking yaw err")
        ax_yaw.plot(t, truth_belief, color=COL[run.cond], lw=0.9, ls="--", alpha=0.7, label=f"{run.cond} truth-belief yaw")
        ax_yaw.plot(t, truth_state, color=COL[run.cond], lw=0.7, ls=":", alpha=0.7, label=f"{run.cond} truth-state yaw")
    ax_yaw.set_title("(e) Heading/yaw errors")
    ax_yaw.set_xlabel("time after first command [s]")
    ax_yaw.set_ylabel("abs yaw error [rad]")
    ax_yaw.grid(True, alpha=0.25)
    ax_yaw.legend(fontsize=7, ncol=2)

    ax_perc = fig.add_subplot(gs[2, 1])
    for run in runs:
        t = rel_t(run)
        ax_perc.plot(t, finite_series(run.exp, "planner_pixel_correction_age_s"), color=COL[run.cond], lw=1.2, label=f"{run.cond} pixel age")
        ax_perc.plot(t, finite_series(run.exp, "planner_belief_age_s"), color=COL[run.cond], lw=0.8, ls=":", label=f"{run.cond} belief age")
        ax_perc.plot(t, finite_series(run.exp, "truth_state_error_m"), color=COL[run.cond], lw=1.0, ls="--", label=f"{run.cond} state err")
        if "measurement_available" in run.exp:
            miss = finite_series(run.exp, "measurement_available") < 0.5
            ax_perc.scatter(t[miss], np.full(miss.sum(), -0.08), s=8, color=COL[run.cond], marker="|", label=f"{run.cond} no meas")
    ax_perc.axhline(1.25, color="#111827", ls=":", lw=1.0, label="pixel timeout")
    ax_perc.set_title("(f) Perception freshness and localization")
    ax_perc.set_xlabel("time after first command [s]")
    ax_perc.set_ylabel("s / m")
    ax_perc.grid(True, alpha=0.25)
    ax_perc.legend(fontsize=7, ncol=2)

    ax_text = fig.add_subplot(gs[2, 2])
    ax_text.axis("off")
    lines = ["F45 controlled tracking-yaw smoke", ""]
    for cond in ["C1", "C2"]:
        s = stats[cond]
        lines.extend(
            [
                f"{cond}: {s['outcome']}, path={s['path_m']:.2f}m, min_goal={s['min_goal_m']:.2f}m",
                f"  err={s['mean_truth_state_error_m']:.3f}m, min_obs={s['min_obs_m']:.3f}m",
                f"  solve μ/p90={s['solve_mean_ms']:.0f}/{s['solve_p90_ms']:.0f} ms, {s['ms_per_eval']:.0f} ms/eval",
                f"  maxiter/non-success rows={100*s['maxiter_frac']:.0f}%",
                f"  yaw track μ/p90={s['exec_yaw_mean']:.2f}/{s['exec_yaw_p90']:.2f} rad",
                f"  pixel age p90/max={s['pixel_age_p90_s']:.2f}/{s['pixel_age_max_s']:.2f}s",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation:",
            "1. Spawn yaw is clean; yaw problems are runtime/tracking, not init.",
            "2. Global plans still choose the expected route families.",
            "3. Local H15/maxiter20 is too budget-starved: frequent maxiter hits.",
            "4. C1 additionally suffers stale camera belief before collision.",
            "5. C2 localizes better but still collides: local tracking/clearance remains failing.",
        ]
    )
    ax_text.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)

    fig.suptitle("F45 - Tracking yaw + local solver timing diagnostic", fontsize=16, fontweight="bold")
    png = OUT_DIR / "F45_tracking_yaw_diagnosis.png"
    pdf = OUT_DIR / "F45_tracking_yaw_diagnosis.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def write_note(stats: dict[str, dict], png: Path, pdf: Path):
    md = OUT_DIR / "F45_tracking_yaw_diagnosis.md"
    rows = []
    for cond in ["C1", "C2"]:
        s = stats[cond]
        rows.append(
            "| {cond} | {outcome} | {path:.2f} | {goal:.2f} | {obs:.3f} | {err:.3f} | {solve:.0f} | {p90:.0f} | {mpe:.0f} | {yaw:.2f} | {age:.2f} |".format(
                cond=cond,
                outcome=s["outcome"],
                path=s["path_m"],
                goal=s["min_goal_m"],
                obs=s["min_obs_m"],
                err=s["mean_truth_state_error_m"],
                solve=s["solve_mean_ms"],
                p90=s["solve_p90_ms"],
                mpe=s["ms_per_eval"],
                yaw=s["exec_yaw_p90"],
                age=s["pixel_age_max_s"],
            )
        )
    text = f"""# F45 Tracking/Yaw Diagnostic

Generated from existing run root:
`logs/visibility_comparison/f45_b1_tracking_yaw_v2`.

Files:
- PNG: `{png.relative_to(ROOT)}`
- PDF: `{pdf.relative_to(ROOT)}`
- Generic dashboard: `timing_presentation/figures/F45/F45_dashboard.png`

## Configuration Tested

F45 is a controlled Gazebo smoke after wiring the local-tracking parameters into
the runtime and logger:

- global route: `H=80`, multistart route candidates enabled for both C1 and C2.
- local tracker: EFE local controller, `H=15`, `local_optimizer_maxiter=20`.
- `local_tracking_use_odom_yaw=true`.
- `latency_compensate_plan_handoff=true`.
- `cmd_publish_rate=10 Hz`.
- command noise and encoder noise remain enabled.

## Run Summary

| condition | outcome | path [m] | min goal [m] | min obs [m] | mean state err [m] | solve mean [ms] | solve p90 [ms] | ms/eval | yaw p90 [rad] | pixel-age max [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Diagnosis

This is not a successful behavior run. It is useful because it separates three
failure modes that were previously mixed together.

1. **Initial yaw/spawn is not the root cause here.** Both C1 and C2 start with a
   clean frame sanity check (`truth_start_yaw_error=0`). If yaw becomes bad, it
   happens during runtime tracking/perception rather than at Gazebo spawn.

2. **The global route choice remains mostly sensible.** The first route and local
   endpoint overlays show the route family chosen before execution. The weird
   behavior still appears during tracking and replanning, not because the first
   global planner has no idea where to go.

3. **Local EFE is under-budgeted in F45.** `H=15, maxiter=20` reduced horizon and
   should have helped timing, but the logs repeatedly report
   `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`. That means the local controller
   is often executing solver-returned sequences that have not actually converged.
   This can explain jagged heading corrections and poor waypoint tracking.

4. **C1 has a perception freshness failure before collision.** The C1 console log
   reported stale pixel belief ages of multiple seconds and an implausible
   correction gap of about 7.5 s before belief reset. That makes C1 a localization
   failure case, not a clean local-controller-only case.

5. **C2 is the more revealing controller failure.** C2 keeps lower state error
   than C1 but still collides. When localization is comparatively good and the
   route is plausible, the remaining failure is local tracking/clearance/timing:
   waypoint following plus local barrier optimization is not robust enough yet.

## Next Iteration

Do not change maximum speed or the route story first. The next controlled
iteration should keep the same task and route candidates, but make the local
tracker numerically honest:

- try `local_horizon=20` again with `local_optimizer_maxiter=35` or `40`;
- keep `local_tracking_use_odom_yaw=true`;
- keep the new waypoint/yaw diagnostics;
- compare optimizer success fraction and ms/eval against F45;
- only after local convergence improves, test whether the remaining problem is
  waypoint acceptance radius or driveable-region clearance.
"""
    md.write_text(text)
    return md


def main() -> int:
    runs = [read_run("C1"), read_run("C2")]
    stats = {run.cond: summarize(run) for run in runs}
    png, pdf = make_figure(runs, stats)
    md = write_note(stats, png, pdf)
    print(png)
    print(pdf)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
