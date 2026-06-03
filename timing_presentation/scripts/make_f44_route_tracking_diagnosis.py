#!/usr/bin/env python3
"""F44: compare initial route choice against execution/tracking diagnostics.

This script reads existing F34/F35/F37/F43 logs only. It does not run Gazebo,
change planner settings, or recapture perception/GP artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
LOG_ROOT = ROOT / "logs/visibility_comparison"
OUT_DIR = ROOT / "timing_presentation/figures/F44"
TASK = "F31_b1_apron_a3_mid"

CASES = [
    ("F34", "f34_b1_route_choice_v2"),
    ("F35", "f35_b1_route_choice_v1"),
    ("F37", "f37_b1_route_choice_v3"),
    ("F43", "f43_b1_timing_architecture_v2"),
]

COL = {"C1": "#2563eb", "C2": "#dc2626"}
LIGHT = {"C1": "#93c5fd", "C2": "#fca5a5"}

# Focus geometry for the B1 route-choice task. These are explanatory overlays,
# not a new source of runtime truth.
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
class RunData:
    fig: str
    log_name: str
    cond: str
    exp_dir: Path
    exp: pd.DataFrame
    plan: pd.DataFrame
    perception: pd.DataFrame | None
    summary: dict


def finite(v, default=np.nan) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    return f if np.isfinite(f) else default


def unwrap_angle(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(vals)
    if mask.sum() < 2:
        return vals
    out = vals.copy()
    out[mask] = np.unwrap(out[mask])
    return out


def load_run(fig: str, log_name: str, cond: str) -> RunData:
    seed_dir = LOG_ROOT / log_name / TASK / cond / "seed1"
    candidates = sorted(seed_dir.glob("experiment_*"))
    if not candidates:
        raise FileNotFoundError(f"No experiments found under {seed_dir}")

    for exp_dir in reversed(candidates):
        exp_csv = exp_dir / "experiment.csv"
        plan_csv = exp_dir / "plan_samples.csv"
        if exp_csv.exists() and plan_csv.exists():
            exp = pd.read_csv(exp_csv)
            plan = pd.read_csv(plan_csv)
            if len(exp) > 10 and len(plan) > 2:
                perception = (
                    pd.read_csv(exp_dir / "perception.csv")
                    if (exp_dir / "perception.csv").exists()
                    else None
                )
                summary = (
                    json.loads((exp_dir / "run_summary.json").read_text())
                    if (exp_dir / "run_summary.json").exists()
                    else {}
                )
                return RunData(fig, log_name, cond, exp_dir, exp, plan, perception, summary)
    raise FileNotFoundError(f"No usable run found under {seed_dir}")


def t_rel(run: RunData) -> pd.Series:
    t0 = finite(run.summary.get("first_cmd_stamp"), finite(run.exp["stamp"].iloc[0], 0.0))
    return run.exp["stamp"] - t0


def perception_t_rel(run: RunData) -> pd.Series | None:
    if run.perception is None or "diag_stamp" not in run.perception:
        return None
    t0 = finite(run.summary.get("first_cmd_stamp"), finite(run.exp["stamp"].iloc[0], 0.0))
    return run.perception["diag_stamp"] - t0


def first_plan(run: RunData) -> pd.DataFrame:
    for stamp in sorted(run.plan["plan_stamp"].dropna().unique()):
        group = run.plan[run.plan["plan_stamp"] == stamp].sort_values("point_idx")
        if len(group) > 2:
            return group
    return run.plan.sort_values(["plan_stamp", "point_idx"]).head(0)


def last_plan(run: RunData) -> pd.DataFrame:
    for stamp in sorted(run.plan["plan_stamp"].dropna().unique(), reverse=True):
        group = run.plan[run.plan["plan_stamp"] == stamp].sort_values("point_idx")
        if len(group) > 2:
            return group
    return run.plan.sort_values(["plan_stamp", "point_idx"]).head(0)


def plan_length(plan: pd.DataFrame) -> float:
    if plan is None or len(plan) < 2:
        return np.nan
    x = plan["x"].to_numpy(dtype=float)
    y = plan["y"].to_numpy(dtype=float)
    return float(np.nansum(np.hypot(np.diff(x), np.diff(y))))


def plan_endpoints(run: RunData) -> pd.DataFrame:
    rows: list[dict] = []
    for stamp, group in run.plan.groupby("plan_stamp", sort=True):
        group = group.sort_values("point_idx")
        if len(group) < 2:
            continue
        end = group.iloc[-1]
        rows.append(
            {
                "plan_stamp": finite(stamp),
                "x": finite(end["x"]),
                "y": finite(end["y"]),
                "length": plan_length(group),
            }
        )
    return pd.DataFrame(rows)


def waypoint_samples(plan: pd.DataFrame, spacing: float = 0.60) -> tuple[np.ndarray, np.ndarray]:
    if plan is None or len(plan) < 2:
        return np.array([]), np.array([])
    xs = plan["x"].to_numpy(dtype=float)
    ys = plan["y"].to_numpy(dtype=float)
    pts = [(xs[0], ys[0])]
    since = 0.0
    for i in range(1, len(xs)):
        step = float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        since += step
        if since >= spacing:
            pts.append((xs[i], ys[i]))
            since = 0.0
    if pts[-1] != (xs[-1], ys[-1]):
        pts.append((xs[-1], ys[-1]))
    arr = np.asarray(pts)
    return arr[:, 0], arr[:, 1]


def path_final(run: RunData) -> tuple[float, float]:
    xy = run.exp[["truth_x", "truth_y"]].dropna()
    if len(xy) == 0:
        return np.nan, np.nan
    return float(xy.iloc[-1]["truth_x"]), float(xy.iloc[-1]["truth_y"])


def goal_xy(run: RunData) -> tuple[float, float]:
    gx = run.exp["goal_x"].dropna() if "goal_x" in run.exp else pd.Series(dtype=float)
    gy = run.exp["goal_y"].dropna() if "goal_y" in run.exp else pd.Series(dtype=float)
    if len(gx) and len(gy):
        return float(gx.iloc[-1]), float(gy.iloc[-1])
    return 1.0, 1.75


def draw_static_map(ax) -> None:
    for xmin, xmax, ymin, ymax in DRIVEABLE_REGIONS:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="#22c55e",
                edgecolor="#16a34a",
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
                facecolor="#d1d5db",
                edgecolor="#6b7280",
                alpha=0.65,
                lw=0.8,
                zorder=1,
            )
        )
    ax.set_xlim(0.2, 4.7)
    ax.set_ylim(-2.4, 2.5)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.18)


def draw_map(ax, runs: dict[str, RunData]) -> None:
    draw_static_map(ax)
    for cond, run in runs.items():
        p0 = first_plan(run)
        p_last = last_plan(run)
        if len(p0):
            ax.plot(
                p0["x"],
                p0["y"],
                ls="--",
                lw=1.8,
                color=COL[cond],
                alpha=0.75,
                label=f"{cond} first global",
                zorder=4,
            )
            wx, wy = waypoint_samples(p0)
            ax.scatter(wx, wy, s=15, color=COL[cond], alpha=0.45, zorder=5)
        if len(p_last):
            ax.plot(
                p_last["x"],
                p_last["y"],
                ls="-.",
                lw=1.0,
                color=LIGHT[cond],
                alpha=0.70,
                label=f"{cond} last local",
                zorder=3,
            )
        endpoints = plan_endpoints(run)
        if len(endpoints):
            ax.scatter(
                endpoints["x"],
                endpoints["y"],
                s=8,
                color=COL[cond],
                alpha=0.28,
                zorder=3,
            )
        truth = run.exp[["truth_x", "truth_y"]].dropna()
        if len(truth):
            ax.plot(
                truth["truth_x"],
                truth["truth_y"],
                color=COL[cond],
                lw=2.4,
                alpha=0.95,
                label=f"{cond} executed",
                zorder=6,
            )
            marker = "X" if run.summary.get("completion_reason") == "collision" else "o"
            ax.scatter(
                truth.iloc[-1]["truth_x"],
                truth.iloc[-1]["truth_y"],
                color=COL[cond],
                edgecolor="white",
                lw=0.8,
                s=90,
                marker=marker,
                zorder=8,
            )
    first_run = next(iter(runs.values()))
    start = first_run.exp[["truth_x", "truth_y"]].dropna().iloc[0]
    gx, gy = goal_xy(first_run)
    ax.scatter(start["truth_x"], start["truth_y"], color="#111827", s=80, marker="o", zorder=9)
    ax.scatter(gx, gy, color="#111827", s=140, marker="*", zorder=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def draw_yaw(ax, runs: dict[str, RunData]) -> None:
    for cond, run in runs.items():
        t = t_rel(run)
        if "yaw_error_truth_state_rad" in run.exp:
            y = pd.to_numeric(run.exp["yaw_error_truth_state_rad"], errors="coerce").abs()
            ax.plot(t, y, color=COL[cond], lw=1.5, label=f"{cond} |truth-state yaw|")
        if "yaw_error_truth_belief_rad" in run.exp:
            yb = pd.to_numeric(run.exp["yaw_error_truth_belief_rad"], errors="coerce").abs()
            ax.plot(t, yb, color=COL[cond], lw=0.9, ls="--", alpha=0.75, label=f"{cond} |truth-belief yaw|")
    ax.axhline(math.pi / 2, color="#9ca3af", lw=0.8, ls=":", label="pi/2")
    ax.set_ylabel("yaw error [rad]")
    ax.set_ylim(-0.05, 3.35)
    ax.grid(alpha=0.20)


def draw_perception(ax, runs: dict[str, RunData]) -> None:
    for cond, run in runs.items():
        t = t_rel(run)
        if "truth_state_error_m" in run.exp:
            err = pd.to_numeric(run.exp["truth_state_error_m"], errors="coerce")
            ax.plot(t, err, color=COL[cond], lw=1.5, label=f"{cond} localization error")
        if "planner_pixel_correction_age_s" in run.exp:
            age = pd.to_numeric(run.exp["planner_pixel_correction_age_s"], errors="coerce")
            ax.plot(t, age, color=LIGHT[cond], lw=1.0, ls="--", label=f"{cond} pixel age")

        if run.perception is not None and "detected" in run.perception:
            pt = perception_t_rel(run)
            miss = run.perception[pd.to_numeric(run.perception["detected"], errors="coerce").fillna(0) < 0.5]
            if pt is not None and len(miss):
                mt = pt.loc[miss.index]
                ax.vlines(mt, ymin=0.0, ymax=0.08, color=COL[cond], alpha=0.35, lw=0.5)
    ax.set_ylabel("error / age [m or s]")
    ax.grid(alpha=0.20)


def draw_execution(ax, runs: dict[str, RunData]) -> None:
    ax2 = ax.twinx()
    any_exec = False
    for cond, run in runs.items():
        t = t_rel(run)
        if "cmd_v" in run.exp:
            ax.plot(t, run.exp["cmd_v"], color=COL[cond], lw=1.25, label=f"{cond} cmd_v")
        if "cmd_w" in run.exp:
            ax.plot(t, run.exp["cmd_w"], color=COL[cond], lw=0.85, ls="--", alpha=0.70, label=f"{cond} cmd_w")
        if "exec_control_index" in run.exp:
            idx = pd.to_numeric(run.exp["exec_control_index"], errors="coerce")
            if idx.notna().sum() > 0:
                any_exec = True
                ax2.plot(t, idx, color=LIGHT[cond], lw=1.0, alpha=0.80, label=f"{cond} exec idx")
    if not any_exec:
        ax.text(
            0.5,
            0.85,
            "exec_control_index not logged\n(F34-F37 predate command-tick diagnostics)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )
    ax.set_ylabel("command")
    ax2.set_ylabel("exec_control_index")
    ax.grid(alpha=0.20)


def heading_counts(run: RunData) -> str:
    if "heading_source" not in run.exp:
        return "not logged"
    counts = run.exp["heading_source"].fillna("NA").astype(str).value_counts().head(3)
    return ", ".join(f"{k}:{int(v)}" for k, v in counts.items())


def run_metrics(run: RunData) -> dict:
    p0 = first_plan(run)
    plast = last_plan(run)
    endpoints = plan_endpoints(run)
    final_x, final_y = path_final(run)
    gx, gy = goal_xy(run)

    detected_frac = np.nan
    pixel_age_max = np.nan
    loc_mean = np.nan
    loc_max = np.nan
    if run.perception is not None:
        if "detected" in run.perception:
            detected_frac = float(pd.to_numeric(run.perception["detected"], errors="coerce").mean())
        if "pixel_pose_age_s" in run.perception:
            pixel_age_max = float(pd.to_numeric(run.perception["pixel_pose_age_s"], errors="coerce").max())
        if "localization_error_m" in run.perception:
            loc = pd.to_numeric(run.perception["localization_error_m"], errors="coerce")
            loc_mean = float(loc.mean())
            loc_max = float(loc.max())

    yaw_state = pd.to_numeric(run.exp.get("yaw_error_truth_state_rad", pd.Series(dtype=float)), errors="coerce").abs()
    yaw_belief = pd.to_numeric(run.exp.get("yaw_error_truth_belief_rad", pd.Series(dtype=float)), errors="coerce").abs()
    exec_max = np.nan
    if "exec_control_index" in run.exp:
        idx = pd.to_numeric(run.exp["exec_control_index"], errors="coerce")
        exec_max = float(idx.max()) if idx.notna().sum() else np.nan

    first_end = (np.nan, np.nan)
    last_end = (np.nan, np.nan)
    if len(p0):
        first_end = (finite(p0.iloc[-1]["x"]), finite(p0.iloc[-1]["y"]))
    if len(plast):
        last_end = (finite(plast.iloc[-1]["x"]), finite(plast.iloc[-1]["y"]))

    return {
        "fig": run.fig,
        "cond": run.cond,
        "outcome": run.summary.get("completion_reason", "unknown"),
        "path_m": finite(run.summary.get("path_length_m")),
        "min_goal_m": finite(run.summary.get("minimum_goal_distance")),
        "first_plan_m": plan_length(p0),
        "first_end_x": first_end[0],
        "first_end_y": first_end[1],
        "last_end_x": last_end[0],
        "last_end_y": last_end[1],
        "final_x": final_x,
        "final_y": final_y,
        "goal_x": gx,
        "goal_y": gy,
        "num_local_plans": len(endpoints),
        "detected_frac": detected_frac,
        "pixel_age_max_s": pixel_age_max,
        "loc_mean_m": loc_mean,
        "loc_max_m": loc_max,
        "yaw_state_mean_rad": float(yaw_state.mean()) if yaw_state.notna().sum() else np.nan,
        "yaw_state_p90_rad": float(yaw_state.quantile(0.90)) if yaw_state.notna().sum() else np.nan,
        "yaw_state_max_rad": float(yaw_state.max()) if yaw_state.notna().sum() else np.nan,
        "yaw_belief_mean_rad": float(yaw_belief.mean()) if yaw_belief.notna().sum() else np.nan,
        "exec_control_index_max": exec_max,
        "heading_sources": heading_counts(run),
        "exp_dir": str(run.exp_dir.relative_to(ROOT)),
    }


def fmt(v: float, nd: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def make_note(metrics: list[dict]) -> str:
    rows = []
    for m in metrics:
        rows.append(
            "| {fig} | {cond} | {outcome} | {path} | {goal} | {first} | ({fex},{fey}) | ({lex},{ley}) | ({fx},{fy}) | {det} | {age} | {loc} | {yaw} | {execidx} |".format(
                fig=m["fig"],
                cond=m["cond"],
                outcome=m["outcome"],
                path=fmt(m["path_m"], 2),
                goal=fmt(m["min_goal_m"], 2),
                first=fmt(m["first_plan_m"], 2),
                fex=fmt(m["first_end_x"], 2),
                fey=fmt(m["first_end_y"], 2),
                lex=fmt(m["last_end_x"], 2),
                ley=fmt(m["last_end_y"], 2),
                fx=fmt(m["final_x"], 2),
                fy=fmt(m["final_y"], 2),
                det=fmt(m["detected_frac"], 2),
                age=fmt(m["pixel_age_max_s"], 2),
                loc=f"{fmt(m['loc_mean_m'], 2)}/{fmt(m['loc_max_m'], 2)}",
                yaw=f"{fmt(m['yaw_state_mean_rad'], 2)}/{fmt(m['yaw_state_p90_rad'], 2)}/{fmt(m['yaw_state_max_rad'], 2)}",
                execidx=fmt(m["exec_control_index_max"], 0),
            )
        )

    heading_lines = [
        f"- {m['fig']} {m['cond']}: `{m['heading_sources']}`"
        for m in metrics
    ]
    source_lines = [
        f"- {m['fig']} {m['cond']}: `{m['exp_dir']}`"
        for m in metrics
    ]

    return "\n".join(
        [
            "# F44 Route-Tracking Root-Cause Diagnostic",
            "",
            "This figure reads the existing F34/F35/F37/F43 logs only. It is a diagnostic artifact, not a new Gazebo run.",
            "",
            "## Files",
            "",
            "- Figure PNG: `timing_presentation/figures/F44/F44_route_tracking_diagnosis.png`",
            "- Figure PDF: `timing_presentation/figures/F44/F44_route_tracking_diagnosis.pdf`",
            "",
            "## Summary Table",
            "",
            "| Fig | Cond | Outcome | path m | min goal m | first plan m | first end | last local end | final truth | detect frac | max pixel age s | loc mean/max m | yaw state mean/p90/max rad | exec idx max |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: |",
            *rows,
            "",
            "## What Did The Initial Planner Choose?",
            "",
            "The first global plans are mostly sensible. Across F34/F35/F37/F43, C1's first plan is consistently the shorter upper/direct candidate, while C2's first plan is consistently the longer lower-visible sweep. That means the most confusing path shapes are not primarily caused by the initial route-choice optimization choosing random routes.",
            "",
            "## Where Did Execution Diverge?",
            "",
            "The divergence appears during local execution/replanning. In successful C2 runs such as F35, the last local endpoint remains near the true goal and the executed path follows the lower visible sweep. In F43, the initial C2 plan still points toward the lower visible route, but the later local endpoint collapses back near the lower aisle/apron while the truth path never progresses north. This makes F43 a tracking/runtime failure, not a clean visibility-vs-shortest-path comparison.",
            "",
            "## Was Yaw Wrong At The Divergence?",
            "",
            "Yaw is a plausible contributor but not the whole explanation. Current YOLO-seg runs do not use visual heading as a paper-facing measurement; the state estimator mostly uses odometry heading or held previous heading. F43 C2 shows much larger state-yaw error than F35 C2, so heading handling should be inspected before claiming the route behavior is planner-optimal.",
            "",
            "Heading source counts:",
            *heading_lines,
            "",
            "## Was Perception Stale Or Missing?",
            "",
            "F35 shows the desired visibility/localization story: C1 loses detections and localization quality, while C2 stays visually locked and reaches the goal. F43 does not show the same mechanism: perception availability and localization error are good for both conditions, yet both crash. Therefore F43 should be treated as a runtime/local-tracking diagnostic rather than visibility-method evidence.",
            "",
            "## Did F43 Fail For A Different Reason Than F35?",
            "",
            "Yes. F35 is mainly a perception/localization contrast: C1 takes the risky route and loses visual updates; C2 stays observable and succeeds. F43 is mainly a tracking/timing/collision issue: both conditions remain localized, but the local plan endpoints and executed paths disagree with the initial global route.",
            "",
            "## Next Fix Target",
            "",
            "The next fix should target local waypoint tracking, heading-state consistency, no-go clearance during local execution, and command/update timing. YOLO heading should be a future controlled ablation, not an immediate assumption, because the current segmentation setup provides `(x,y)` camera localization while heading comes from odometry/held previous state.",
            "",
            "## Source Runs",
            "",
            *source_lines,
            "",
        ]
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs: dict[str, dict[str, RunData]] = {}
    metrics: list[dict] = []
    for fig, log_name in CASES:
        all_runs[fig] = {}
        for cond in ("C1", "C2"):
            run = load_run(fig, log_name, cond)
            all_runs[fig][cond] = run
            metrics.append(run_metrics(run))

    fig, axes = plt.subplots(
        nrows=len(CASES),
        ncols=4,
        figsize=(24, 20),
        constrained_layout=False,
    )
    fig.suptitle(
        "F44 - Route-choice vs local tracking vs heading/perception",
        fontsize=20,
        fontweight="bold",
        y=0.992,
    )

    for row, (case_fig, _) in enumerate(CASES):
        runs = all_runs[case_fig]
        ax_map, ax_yaw, ax_perc, ax_exec = axes[row]

        draw_map(ax_map, runs)
        c1 = runs["C1"].summary.get("completion_reason", "unknown")
        c2 = runs["C2"].summary.get("completion_reason", "unknown")
        ax_map.set_title(f"{case_fig}: route + local endpoint evolution\nC1={c1}, C2={c2}", fontweight="bold")
        if row == 0:
            ax_map.legend(
                handles=[
                    Line2D([0], [0], color=COL["C1"], lw=2.4, label="C1 executed"),
                    Line2D([0], [0], color=COL["C2"], lw=2.4, label="C2 executed"),
                    Line2D([0], [0], color=COL["C1"], lw=1.8, ls="--", label="first global plan"),
                    Line2D([0], [0], color=LIGHT["C1"], lw=1.0, ls="-.", label="last local plan"),
                    Line2D([0], [0], color="#111827", marker="*", lw=0, label="goal"),
                    Line2D([0], [0], color="#111827", marker="o", lw=0, label="start"),
                ],
                fontsize=8,
                loc="lower left",
                framealpha=0.86,
            )

        draw_yaw(ax_yaw, runs)
        ax_yaw.set_title("Yaw consistency")
        if row == len(CASES) - 1:
            ax_yaw.set_xlabel("time after first command [s]")
        if row == 0:
            ax_yaw.legend(fontsize=7, loc="upper right", framealpha=0.85)

        draw_perception(ax_perc, runs)
        ax_perc.set_title("Perception age + localization error")
        if row == len(CASES) - 1:
            ax_perc.set_xlabel("time after first command [s]")
        if row == 0:
            ax_perc.legend(fontsize=7, loc="upper right", framealpha=0.85)

        draw_execution(ax_exec, runs)
        ax_exec.set_title("Commands + command-tape index")
        if row == len(CASES) - 1:
            ax_exec.set_xlabel("time after first command [s]")
        if row == 0:
            ax_exec.legend(fontsize=7, loc="upper left", framealpha=0.85)

    fig.text(
        0.5,
        0.006,
        "Dashed plans are the first global solve. Pale dash-dot plans are the last logged local solve. "
        "Endpoint dots show where local replans ended over time.",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.975])

    png = OUT_DIR / "F44_route_tracking_diagnosis.png"
    pdf = OUT_DIR / "F44_route_tracking_diagnosis.pdf"
    md = OUT_DIR / "F44_route_tracking_diagnosis.md"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    md.write_text(make_note(metrics), encoding="utf-8")
    print(png)
    print(pdf)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
