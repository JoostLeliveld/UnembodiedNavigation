#!/usr/bin/env python3
"""F15 - failure-chain diagnostic for a visibility-aware timing run.

The original F14 run directory was later cleaned up, while the image survived.
This script therefore accepts an explicit --run-dir and otherwise falls back to
the best preserved collision run in timing_presentation/runs/gazebo.

The figure is intentionally diagnostic, not paper evidence by itself:
it separates map geometry, perception/visibility, belief error, obstacle
clearance, and EFE terms so failure causes are easier to inspect.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / "timing_presentation" / "figures"
DEFAULT_RUN_CANDIDATES = [
    REPO / "timing_presentation/runs/gazebo/hier_v1/experiment_20260527_164918",
    REPO / "timing_presentation/runs/gazebo/H80_ms1_av/experiment_20260527_114852",
]

FALLBACK_DRIVEABLE_RECTS = [
    {"xmin": -4.85, "xmax": 4.85, "ymin": -3.45, "ymax": -2.55, "label": "apron"},
    {"xmin": -4.85, "xmax": 4.85, "ymin": 1.45, "ymax": 2.55, "label": "mid cross"},
    {"xmin": -4.85, "xmax": 4.85, "ymin": 4.28, "ymax": 4.80, "label": "upper cross"},
    {"xmin": -4.525, "xmax": -3.675, "ymin": -2.85, "ymax": 4.80, "label": "A1"},
    {"xmin": -2.525, "xmax": -1.675, "ymin": -2.85, "ymax": 4.80, "label": "A2"},
    {"xmin": 0.625, "xmax": 1.525, "ymin": -2.85, "ymax": 4.80, "label": "A3"},
    {"xmin": 2.625, "xmax": 3.525, "ymin": -2.85, "ymax": 4.80, "label": "A4"},
]


def _finite_series(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _first_cmd_stamp(df: pd.DataFrame, summary: dict) -> float:
    value = summary.get("first_cmd_stamp")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    cmd_v = _finite_series(df, "cmd_v", 0.0).abs()
    cmd_w = _finite_series(df, "cmd_w", 0.0).abs()
    active = df[(cmd_v > 0.01) | (cmd_w > 0.10)]
    if len(active):
        return float(active.iloc[0]["stamp"])
    return float(df["stamp"].dropna().iloc[0])


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _decode_geometry(text: object) -> list[dict]:
    if not text:
        return []
    if isinstance(text, dict):
        return list(text.get("prisms", []) or [])
    try:
        return list(json.loads(str(text)).get("prisms", []) or [])
    except json.JSONDecodeError:
        return []


def _rects_from_driveable_manifest(manifest: dict) -> list[dict]:
    prisms = _decode_geometry(manifest.get("driveable_geometry_json", ""))
    rects = []
    for prism in prisms:
        try:
            rects.append({
                "xmin": float(prism["xmin"]),
                "xmax": float(prism["xmax"]),
                "ymin": float(prism["ymin"]),
                "ymax": float(prism["ymax"]),
                "label": str(prism.get("name", "")),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rects or FALLBACK_DRIVEABLE_RECTS


def _collision_prisms(manifest: dict) -> list[dict]:
    return _decode_geometry(manifest.get("collision_geometry_json", ""))


def _choose_run_dir(raw: str | None) -> Path:
    if raw:
        run_dir = Path(raw).expanduser().resolve()
        if not (run_dir / "experiment.csv").exists():
            raise FileNotFoundError(f"Missing experiment.csv in {run_dir}")
        return run_dir
    for candidate in DEFAULT_RUN_CANDIDATES:
        if (candidate / "experiment.csv").exists():
            return candidate
    raise FileNotFoundError("No default F15 run candidate exists; pass --run-dir.")


def _add_cov_ellipse(ax, row: pd.Series, *, n_sigma: float = 2.0) -> None:
    try:
        cx = float(row["planner_belief_x"])
        cy = float(row["planner_belief_y"])
        cov_xx = float(row["planner_cov_x"])
        cov_xy = float(row.get("planner_cov_xy", 0.0))
        cov_yy = float(row["planner_cov_y"])
    except (KeyError, TypeError, ValueError):
        return
    if not all(math.isfinite(v) for v in (cx, cy, cov_xx, cov_xy, cov_yy)):
        return
    S = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (S + S.T))
    eigvals = np.clip(eigvals, 1e-8, None)
    angle = math.degrees(math.atan2(float(eigvecs[1, 1]), float(eigvecs[0, 1])))
    width, height = 2.0 * n_sigma * np.sqrt(eigvals)
    ax.add_patch(Ellipse(
        (cx, cy),
        float(width),
        float(height),
        angle=angle,
        facecolor="#ef444420",
        edgecolor="#dc2626",
        linewidth=0.7,
        zorder=4,
    ))


def _group_plan_samples(plan_path: Path) -> dict[str, pd.DataFrame]:
    if not plan_path.exists():
        return {}
    plan = pd.read_csv(plan_path)
    if not {"plan_stamp", "point_idx", "x", "y"}.issubset(plan.columns):
        return {}
    groups = {
        str(stamp): group.sort_values("point_idx")
        for stamp, group in plan.groupby("plan_stamp")
        if len(group) >= 2
    }
    return groups


def _nearest_plan(groups: dict[str, pd.DataFrame], stamp: float, *, prefer_after: bool) -> pd.DataFrame | None:
    if not groups:
        return None
    candidates = []
    for raw_stamp, group in groups.items():
        try:
            plan_stamp = float(raw_stamp)
        except ValueError:
            continue
        if prefer_after and plan_stamp < stamp:
            continue
        if (not prefer_after) and plan_stamp > stamp:
            continue
        candidates.append((abs(plan_stamp - stamp), len(group), plan_stamp, group))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[0][3]


def _longest_plan(groups: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if not groups:
        return None
    return max(groups.values(), key=len)


def _event_times(df: pd.DataFrame, perc: pd.DataFrame | None, first_cmd: float, summary: dict) -> dict[str, float]:
    events: dict[str, float] = {}
    t = _finite_series(df, "t")

    pvis = _finite_series(df, "p_vis_plan")
    if pvis.notna().any():
        events["min visibility"] = float(t.loc[pvis.idxmin()])

    clearance = _finite_series(df, "min_obstacle_distance_m").replace([np.inf, -np.inf], np.nan)
    if clearance.notna().any():
        events["min clearance"] = float(t.loc[clearance.idxmin()])
        hit = df[(clearance <= 0.0) & t.notna()]
        if len(hit):
            events["clearance <= 0"] = float(hit.iloc[0]["t"])

    belief = _finite_series(df, "truth_belief_error_m")
    if belief.notna().any():
        threshold = max(0.5, float(np.nanmedian(belief)) + float(np.nanstd(belief)))
        drift = df[(belief > threshold) & t.notna()]
        if len(drift):
            events[f"belief err > {threshold:.2f} m"] = float(drift.iloc[0]["t"])

    crash_stamp = summary.get("first_crash_stamp")
    if isinstance(crash_stamp, (int, float)) and math.isfinite(float(crash_stamp)):
        events["collision"] = float(crash_stamp) - first_cmd
    elif "collision_any" in df.columns:
        crash = df[_finite_series(df, "collision_any", 0.0) > 0.5]
        if len(crash):
            events["collision"] = float(crash.iloc[0]["t"])

    if perc is not None and "detected" in perc.columns:
        stamp_col = "diag_stamp" if "diag_stamp" in perc.columns else "log_stamp"
        if stamp_col in perc.columns:
            pp = perc.copy()
            pp["t"] = pd.to_numeric(pp[stamp_col], errors="coerce") - first_cmd
            pp["detected_f"] = pd.to_numeric(pp["detected"], errors="coerce").fillna(0.0)
            misses = pp[(pp["t"] >= -0.5) & (pp["detected_f"] < 0.5)]
            if len(misses):
                events["YOLO miss starts"] = float(misses.iloc[0]["t"])
    return events


def _add_event_lines(ax, events: dict[str, float], *, labels: bool = False) -> None:
    colors = {
        "YOLO miss starts": "#dc2626",
        "min visibility": "#7e22ce",
        "belief": "#111827",
        "min clearance": "#f97316",
        "clearance <= 0": "#b91c1c",
        "collision": "#991b1b",
    }
    ylim = ax.get_ylim()
    for label, et in events.items():
        if not math.isfinite(et):
            continue
        key = "belief" if label.startswith("belief") else label
        color = colors.get(key, "#6b7280")
        ax.axvline(et, color=color, linestyle="--", linewidth=1.0, alpha=0.55, zorder=0)
        if labels:
            ax.text(
                et,
                ylim[1],
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color=color,
                alpha=0.95,
            )


def _cost_shares(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["efe_risk", "efe_ambiguity", "efe_obstacle", "efe_control"]
    values = pd.DataFrame({col: _finite_series(df, col, 0.0).clip(lower=0.0) for col in cols})
    denom = values.sum(axis=1).replace(0.0, np.nan)
    return values.divide(denom, axis=0).fillna(0.0)


def _fill_summary_from_csv(summary: dict, df: pd.DataFrame) -> dict:
    """Add lightweight diagnostics when a timeout leaves run_summary sparse."""
    out = dict(summary)
    t = _finite_series(df, "t")
    truth_x = _finite_series(df, "truth_x")
    truth_y = _finite_series(df, "truth_y")
    goal_x = _finite_series(df, "goal_x")
    goal_y = _finite_series(df, "goal_y")

    finite_truth = truth_x.notna() & truth_y.notna()
    if "elapsed_after_first_cmd_s" not in out and t.notna().any():
        out["elapsed_after_first_cmd_s"] = float(t.max())
    if "path_length_m" not in out and finite_truth.sum() >= 2:
        xy = np.column_stack([truth_x[finite_truth].to_numpy(), truth_y[finite_truth].to_numpy()])
        out["path_length_m"] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if "minimum_goal_distance" not in out and finite_truth.any() and goal_x.notna().any() and goal_y.notna().any():
        gx = float(goal_x.dropna().iloc[-1])
        gy = float(goal_y.dropna().iloc[-1])
        dist = np.hypot(truth_x[finite_truth].to_numpy() - gx, truth_y[finite_truth].to_numpy() - gy)
        out["minimum_goal_distance"] = float(np.nanmin(dist))
    if "crashed" not in out and "collision_any" in df.columns:
        out["crashed"] = bool((_finite_series(df, "collision_any", 0.0) > 0.5).any())
    return out


def _write_note(path: Path, run_dir: Path, summary: dict, events: dict[str, float], output_files: Iterable[Path]) -> None:
    lines = [
        "# F15 Failure-Chain Diagnostic",
        "",
        f"Run: `{run_dir}`",
        "",
        "This diagnostic separates geometry, perception, belief error, obstacle margin, and EFE cost terms.",
        "It should be treated as timing/debug evidence, not as a final paper result by itself.",
        "",
        "## Run Summary",
        "",
        f"- completion: `{summary.get('completion_reason', 'unknown')}`",
        f"- crashed: `{summary.get('crashed', 'unknown')}`",
        f"- min goal distance: `{summary.get('minimum_goal_distance', 'unknown')}`",
        f"- path length: `{summary.get('path_length_m', 'unknown')}`",
        f"- elapsed after first command: `{summary.get('elapsed_after_first_cmd_s', 'unknown')}`",
        "",
        "## Event Times",
        "",
    ]
    for label, et in events.items():
        lines.append(f"- {label}: `{et:.2f} s after first command`")
    lines.extend(["", "## Files", ""])
    for out in output_files:
        lines.append(f"- `{out}`")
    path.write_text("\n".join(lines) + "\n")


def make_figure(run_dir: Path, out_dir: Path, prefix: str) -> list[Path]:
    exp_path = run_dir / "experiment.csv"
    if not exp_path.exists():
        raise FileNotFoundError(exp_path)

    df = pd.read_csv(exp_path)
    if "stamp" not in df.columns:
        raise RuntimeError(f"{exp_path} has no stamp column")
    summary = _load_json(run_dir / "run_summary.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    perc = pd.read_csv(run_dir / "perception.csv") if (run_dir / "perception.csv").exists() else None
    plans = _group_plan_samples(run_dir / "plan_samples.csv")

    first_cmd = _first_cmd_stamp(df, summary)
    df = df.copy()
    df["t"] = pd.to_numeric(df["stamp"], errors="coerce") - first_cmd
    df = df[df["t"] >= -0.5].copy()
    if not len(df):
        raise RuntimeError("No experiment rows after first command window.")
    summary = _fill_summary_from_csv(summary, df)

    truth_x = _finite_series(df, "truth_x")
    truth_y = _finite_series(df, "truth_y")
    belief_x = _finite_series(df, "planner_belief_x")
    belief_y = _finite_series(df, "planner_belief_y")
    state_x = _finite_series(df, "state_x")
    state_y = _finite_series(df, "state_y")
    goal_x = _finite_series(df, "goal_x")
    goal_y = _finite_series(df, "goal_y")

    events = _event_times(df, perc, first_cmd, summary)
    driveable = _rects_from_driveable_manifest(manifest)
    collisions = _collision_prisms(manifest)

    initial_plan = _longest_plan(plans)
    crash_or_end_stamp = summary.get("first_crash_stamp")
    if not isinstance(crash_or_end_stamp, (int, float)) or not math.isfinite(float(crash_or_end_stamp)):
        crash_or_end_stamp = float(df["stamp"].dropna().iloc[-1])
    latest_plan = _nearest_plan(plans, float(crash_or_end_stamp), prefer_after=False)

    title_run = run_dir.parent.name + "/" + run_dir.name
    completion = str(summary.get("completion_reason", "unknown"))
    min_goal = summary.get("minimum_goal_distance")
    path_len = summary.get("path_length_m")
    elapsed = summary.get("elapsed_after_first_cmd_s")
    min_goal_s = f"{float(min_goal):.2f} m" if isinstance(min_goal, (int, float)) and math.isfinite(float(min_goal)) else "n/a"
    path_s = f"{float(path_len):.1f} m" if isinstance(path_len, (int, float)) and math.isfinite(float(path_len)) else "n/a"
    elapsed_s = f"{float(elapsed):.1f} s" if isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)) else "n/a"

    fig = plt.figure(figsize=(17.5, 10.5), facecolor="white")
    fig.suptitle(
        "F15 - Failure-chain diagnostic\n"
        f"{title_run} | outcome={completion} | min goal={min_goal_s} | path={path_s} | time={elapsed_s}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.18, 1.0],
        hspace=0.34,
        wspace=0.22,
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.075,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])
    ax_cost = fig.add_subplot(gs[2, 1])

    # Map: known driveable floor and collision geometry.
    for rect in driveable:
        width = rect["xmax"] - rect["xmin"]
        height = rect["ymax"] - rect["ymin"]
        ax_map.add_patch(mpatches.Rectangle(
            (rect["xmin"], rect["ymin"]),
            width,
            height,
            facecolor="#dcfce7",
            edgecolor="#16a34a",
            linewidth=0.9,
            alpha=0.36,
            zorder=0,
        ))

    for prism in collisions:
        try:
            xmin, xmax = float(prism["xmin"]), float(prism["xmax"])
            ymin, ymax = float(prism["ymin"]), float(prism["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(prism.get("name", "")).lower()
        is_stack = "stack" in name or "box" in name or "occluder" in name
        ax_map.add_patch(mpatches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#f97316" if is_stack else "#9ca3af",
            edgecolor="#b91c1c" if is_stack else "#374151",
            linewidth=1.0 if is_stack else 0.6,
            alpha=0.48 if is_stack else 0.32,
            zorder=1,
        ))

    finite_truth = truth_x.notna() & truth_y.notna()
    finite_belief = belief_x.notna() & belief_y.notna()
    finite_state = state_x.notna() & state_y.notna()
    if finite_truth.any():
        ax_map.plot(truth_x[finite_truth], truth_y[finite_truth], color="#111827", lw=2.5, label="truth", zorder=7)
    if finite_belief.any():
        ax_map.plot(belief_x[finite_belief], belief_y[finite_belief], color="#ef4444", lw=1.4, alpha=0.85, label="planner belief", zorder=6)
    if finite_state.any():
        ax_map.plot(state_x[finite_state], state_y[finite_state], color="#2563eb", lw=1.0, ls="--", alpha=0.70, label="state estimate", zorder=5)

    cov_rows = df[df["planner_belief_x"].notna() & df["planner_cov_x"].notna()]
    if len(cov_rows):
        step = max(1, len(cov_rows) // 12)
        for _, row in cov_rows.iloc[::step].iterrows():
            _add_cov_ellipse(ax_map, row)

    if initial_plan is not None:
        ax_map.plot(initial_plan["x"], initial_plan["y"], color="#7c3aed", lw=1.4, ls=":", label="initial planned rollout", zorder=4)
    if latest_plan is not None and initial_plan is not latest_plan:
        ax_map.plot(latest_plan["x"], latest_plan["y"], color="#f59e0b", lw=1.4, ls="-.", label="latest planned rollout", zorder=4)

    if finite_truth.any():
        ax_map.scatter([truth_x[finite_truth].iloc[0]], [truth_y[finite_truth].iloc[0]], s=90, color="#16a34a", zorder=10, label="start")
        ax_map.scatter([truth_x[finite_truth].iloc[-1]], [truth_y[finite_truth].iloc[-1]], s=90, color="#111827", marker="s", zorder=10, label="end")
    if goal_x.notna().any() and goal_y.notna().any():
        ax_map.scatter([goal_x.dropna().iloc[-1]], [goal_y.dropna().iloc[-1]], s=170, color="#facc15", edgecolor="#111827", marker="*", zorder=11, label="goal")
    collision_rows = df[_finite_series(df, "collision_any", 0.0) > 0.5]
    if len(collision_rows):
        row = collision_rows.iloc[0]
        ax_map.scatter([row["truth_x"]], [row["truth_y"]], s=160, marker="X", color="#dc2626", edgecolor="#111827", zorder=12, label="collision")

    ax_map.set_title("(a) Map: trajectory, belief, plans, collision geometry", fontsize=11, fontweight="bold")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-5.8, 5.8)
    ax_map.set_ylim(-5.25, 5.25)
    ax_map.grid(True, alpha=0.25)
    ax_map.legend(loc="upper left", fontsize=7, framealpha=0.88)

    t = _finite_series(df, "t")
    t_min = max(-0.5, float(t.min()) - 0.1)
    t_max = float(t.max()) + 0.3

    # Visibility/perception panel.
    pvis = _finite_series(df, "p_vis_plan")
    rstd = _finite_series(df, "r_plan_u_std")
    ax_vis.plot(t, pvis, color="#7e22ce", lw=1.9, label="p_vis_plan")
    ax_vis.set_ylabel("p_vis_plan", color="#7e22ce")
    ax_vis.tick_params(axis="y", labelcolor="#7e22ce")
    ax_vis.set_ylim(-0.05, 1.05)
    ax_vis_r = ax_vis.twinx()
    ax_vis_r.plot(t, rstd, color="#f97316", lw=1.4, label="R_plan std [px]")
    ax_vis_r.set_ylabel("R_plan std [px]", color="#f97316")
    ax_vis_r.tick_params(axis="y", labelcolor="#f97316")
    if perc is not None and "detected" in perc.columns:
        stamp_col = "diag_stamp" if "diag_stamp" in perc.columns else "log_stamp"
        if stamp_col in perc.columns:
            pp = perc.copy()
            pp["t"] = pd.to_numeric(pp[stamp_col], errors="coerce") - first_cmd
            det = pd.to_numeric(pp["detected"], errors="coerce").fillna(0.0)
            pp = pp[(pp["t"] >= t_min) & (pp["t"] <= t_max)]
            det = det.loc[pp.index]
            ax_vis.scatter(pp.loc[det > 0.5, "t"], np.full(int((det > 0.5).sum()), 0.02), marker="|", s=45, color="#16a34a", label="YOLO detect")
            ax_vis.scatter(pp.loc[det <= 0.5, "t"], np.full(int((det <= 0.5).sum()), 0.08), marker="|", s=45, color="#dc2626", label="YOLO miss")
    _add_event_lines(ax_vis, events, labels=True)
    ax_vis.set_xlim(t_min, t_max)
    ax_vis.set_title("(b) Perception/visibility events", fontsize=11, fontweight="bold")
    ax_vis.set_xlabel("time after first command [s]")
    ax_vis.grid(True, alpha=0.25)
    lines, labels = ax_vis.get_legend_handles_labels()
    lines2, labels2 = ax_vis_r.get_legend_handles_labels()
    ax_vis.legend(lines + lines2, labels + labels2, loc="lower left", fontsize=7, framealpha=0.85)

    # Belief and obstacle margin.
    belief_err = _finite_series(df, "truth_belief_error_m")
    state_err = _finite_series(df, "truth_state_error_m")
    clearance = _finite_series(df, "min_obstacle_distance_m").replace([np.inf, -np.inf], np.nan)
    ax_err.plot(t, belief_err, color="#111827", lw=2.0, label="truth-belief error")
    ax_err.plot(t, state_err, color="#2563eb", lw=1.2, alpha=0.75, label="truth-state error")
    if "planner_cov_x" in df.columns and "planner_cov_y" in df.columns:
        sigma_xy = 2.0 * np.sqrt(
            np.maximum(0.0, _finite_series(df, "planner_cov_x")) +
            np.maximum(0.0, _finite_series(df, "planner_cov_y"))
        )
        ax_err.fill_between(t, 0.0, sigma_xy, color="#111827", alpha=0.08, label="2-sigma xy scale")
    ax_clear = ax_err.twinx()
    ax_clear.plot(t, clearance, color="#f97316", lw=1.5, label="obstacle clearance")
    ax_clear.axhline(0.0, color="#dc2626", lw=1.0, ls="--")
    ax_clear.set_ylabel("clearance [m]", color="#f97316")
    ax_clear.tick_params(axis="y", labelcolor="#f97316")
    _add_event_lines(ax_err, events)
    ax_err.set_xlim(t_min, t_max)
    ax_err.set_xlabel("time after first command [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.set_title("(c) Belief drift and obstacle margin", fontsize=11, fontweight="bold")
    ax_err.grid(True, alpha=0.25)
    lines, labels = ax_err.get_legend_handles_labels()
    lines2, labels2 = ax_clear.get_legend_handles_labels()
    ax_err.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=7, framealpha=0.85)

    # EFE cost shares and command saturation.
    shares = _cost_shares(df)
    ax_cost.stackplot(
        t,
        shares["efe_risk"],
        shares["efe_ambiguity"],
        shares["efe_obstacle"],
        shares["efe_control"],
        labels=["risk", "ambiguity", "obstacle/no-go", "control"],
        colors=["#ef4444", "#3b82f6", "#f97316", "#94a3b8"],
        alpha=0.72,
    )
    total = _finite_series(df, "efe_total")
    if total.notna().any() and float(total.max()) > 0.0:
        total_scaled = total / float(total.max())
        ax_cost.plot(t, total_scaled, color="#111827", lw=1.4, label="total EFE / max")
    ax_cmd = ax_cost.twinx()
    v = _finite_series(df, "cmd_v", 0.0)
    w = _finite_series(df, "cmd_w", 0.0)
    ax_cmd.plot(t, v, color="#047857", lw=1.1, alpha=0.8, label="v cmd")
    ax_cmd.plot(t, w, color="#6d28d9", lw=1.0, alpha=0.75, label="w cmd")
    ax_cmd.set_ylabel("command", color="#374151")
    ax_cmd.tick_params(axis="y", labelcolor="#374151")
    _add_event_lines(ax_cost, events)
    ax_cost.set_ylim(0, 1.0)
    ax_cost.set_xlim(t_min, t_max)
    ax_cost.set_xlabel("time after first command [s]")
    ax_cost.set_ylabel("cost share")
    ax_cost.set_title("(d) EFE composition and commands", fontsize=11, fontweight="bold")
    ax_cost.grid(True, alpha=0.25)
    lines, labels = ax_cost.get_legend_handles_labels()
    lines2, labels2 = ax_cmd.get_legend_handles_labels()
    ax_cost.legend(lines + lines2, labels + labels2, loc="upper left", ncol=3, fontsize=7, framealpha=0.85)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{prefix}.png"
    pdf = out_dir / f"{prefix}.pdf"
    note = out_dir / f"{prefix}.md"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_note(note, run_dir, summary, events, [png, pdf, note])
    return [png, pdf, note]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="", help="Run directory containing experiment.csv.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output figure directory.")
    parser.add_argument("--prefix", default="F15_failure_chain", help="Output file prefix.")
    args = parser.parse_args()

    run_dir = _choose_run_dir(args.run_dir)
    outputs = make_figure(run_dir, Path(args.out_dir), args.prefix)
    for out in outputs:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
