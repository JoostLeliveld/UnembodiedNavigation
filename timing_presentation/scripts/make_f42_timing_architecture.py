#!/usr/bin/env python3
"""Create F42 timing-architecture diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs/visibility_comparison/f42_b1_timing_architecture_v1"
OFFLINE_CSV = ROOT / "logs/visibility_comparison/f42_planner_only_timing/initial_rollout_sweep.csv"
OUT_DIR = ROOT / "timing_presentation/figures/F42"


def _read_run(condition: str) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    matches = sorted(LOG_ROOT.glob(f"*/{condition}/seed*/experiment_*/experiment.csv"))
    if not matches:
        raise FileNotFoundError(f"No experiment.csv found for {condition} under {LOG_ROOT}")
    exp_path = matches[-1]
    exp = pd.read_csv(exp_path)
    perc_path = exp_path.with_name("perception.csv")
    perc = pd.read_csv(perc_path) if perc_path.exists() else None
    summary_path = exp_path.with_name("run_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return exp, perc, summary


def _time(exp: pd.DataFrame) -> pd.Series:
    if "stamp" not in exp or exp["stamp"].dropna().empty:
        return pd.Series(np.arange(len(exp)), index=exp.index, dtype=float)
    return exp["stamp"] - float(exp["stamp"].dropna().iloc[0])


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _solve_stats(exp: pd.DataFrame) -> tuple[float, float, float]:
    if "solve_time_ms" not in exp:
        return (np.nan, np.nan, np.nan)
    values = _finite(exp["solve_time_ms"])
    values = values[values > 10].drop_duplicates()
    if values.empty:
        return (np.nan, np.nan, np.nan)
    return (float(values.mean()), float(values.median()), float(values.max()))


def _latency_stats(exp: pd.DataFrame, perc: pd.DataFrame | None) -> dict[str, float]:
    stats: dict[str, float] = {}
    for col in [
        "planner_pixel_correction_age_s",
        "active_plan_age_s",
        "active_control_index",
        "latency_skip_s",
    ]:
        if col in exp:
            v = _finite(exp[col])
            stats[col] = float(v.mean()) if len(v) else np.nan
            stats[f"{col}_max"] = float(v.max()) if len(v) else np.nan
    if perc is not None:
        for col in ["yolo_inference_ms", "detector_callback_ms", "detector_total_latency_s"]:
            if col in perc:
                v = _finite(perc[col])
                stats[col] = float(v.mean()) if len(v) else np.nan
                stats[f"{col}_max"] = float(v.max()) if len(v) else np.nan
    return stats


def _offline_summary() -> pd.DataFrame:
    if not OFFLINE_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(OFFLINE_CSV)
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(["condition", "horizon"], as_index=False)
        .agg(
            solve_mean_ms=("solve_time_ms", "mean"),
            solve_max_ms=("solve_time_ms", "max"),
            terminal_goal_mean_m=("terminal_goal_distance_pred", "mean"),
        )
        .sort_values(["condition", "horizon"])
    )
    return grouped


def _plot() -> tuple[dict[str, dict], pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    for condition in ("C1", "C2"):
        exp, perc, summary = _read_run(condition)
        runs[condition] = {
            "exp": exp,
            "perc": perc,
            "summary": summary,
            "solve_stats": _solve_stats(exp),
            "latency_stats": _latency_stats(exp, perc),
        }

    offline = _offline_summary()

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.42, wspace=0.25)
    fig.suptitle(
        "F42 - timing architecture: perception is not the only bottleneck",
        fontsize=19,
        fontweight="bold",
    )

    colors = {"C1": "#2563eb", "C2": "#dc2626"}

    ax = fig.add_subplot(gs[0, 0])
    if not offline.empty:
        for condition in ("C1", "C2"):
            sub = offline[offline["condition"] == condition]
            ax.plot(
                sub["horizon"],
                sub["solve_mean_ms"] / 1000.0,
                marker="o",
                lw=2.5,
                color=colors[condition],
                label=f"{condition} planner-only",
            )
        ax.set_xticks(sorted(offline["horizon"].unique()))
    for x in (20, 80):
        ax.axvline(x, color="0.85", lw=0.8)
    ax.set_title("(a) Offline solve cost grows before Gazebo enters")
    ax.set_xlabel("horizon")
    ax.set_ylabel("mean initial solve time [s]")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    labels = []
    means = []
    maxes = []
    for condition in ("C1", "C2"):
        mean, _, max_v = runs[condition]["solve_stats"]
        labels.append(condition)
        means.append(mean / 1000.0)
        maxes.append(max_v / 1000.0)
    xpos = np.arange(len(labels))
    ax.bar(xpos, means, color=[colors[c] for c in labels], alpha=0.75, label="mean local solve")
    ax.scatter(xpos, maxes, color="black", marker="x", s=90, label="max local solve")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.2, label="1 Hz replanning period")
    ax.set_xticks(xpos, labels)
    ax.set_title("(b) Local solves are slower than the 1 Hz replan budget")
    ax.set_ylabel("solve time [s]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    for condition in ("C1", "C2"):
        exp = runs[condition]["exp"]
        t = _time(exp)
        if "active_control_index" in exp:
            ax.step(
                t,
                pd.to_numeric(exp["active_control_index"], errors="coerce"),
                where="post",
                color=colors[condition],
                lw=1.8,
                label=f"{condition} active control index",
            )
    ax.set_title("(c) Planner-result samples occur at handoff time")
    ax.set_xlabel("time after first log row [s]")
    ax.set_ylabel("active control index")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    for condition in ("C1", "C2"):
        exp = runs[condition]["exp"]
        t = _time(exp)
        if "active_plan_age_s" in exp:
            ax.plot(
                t,
                pd.to_numeric(exp["active_plan_age_s"], errors="coerce"),
                color=colors[condition],
                lw=1.7,
                label=f"{condition} active plan age",
            )
        if "planner_pixel_correction_age_s" in exp:
            ax.plot(
                t,
                pd.to_numeric(exp["planner_pixel_correction_age_s"], errors="coerce"),
                color=colors[condition],
                lw=1.0,
                alpha=0.35,
                ls="--",
                label=f"{condition} pixel correction age",
            )
    ax.set_title("(d) Plans are fresh, but measurements can be stale")
    ax.set_xlabel("time after first log row [s]")
    ax.set_ylabel("age [s]")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    for condition in ("C1", "C2"):
        perc = runs[condition]["perc"]
        if perc is None or "diag_stamp" not in perc:
            continue
        t = perc["log_stamp"] - float(perc["log_stamp"].dropna().iloc[0])
        if "yolo_inference_ms" in perc:
            ax.plot(
                t,
                pd.to_numeric(perc["yolo_inference_ms"], errors="coerce"),
                color=colors[condition],
                lw=1.2,
                alpha=0.75,
                label=f"{condition} YOLO inference",
            )
        if "detector_total_latency_s" in perc:
            ax.plot(
                t,
                1000.0 * pd.to_numeric(perc["detector_total_latency_s"], errors="coerce"),
                color=colors[condition],
                lw=1.0,
                alpha=0.35,
                ls="--",
                label=f"{condition} detector total latency",
            )
    ax.set_title("(e) YOLO is nontrivial, but not the dominant multi-second cost")
    ax.set_xlabel("time after first perception row [s]")
    ax.set_ylabel("latency [ms]")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    lines = [
        "F42 diagnosis",
        "",
        "Gazebo outcomes:",
    ]
    for condition in ("C1", "C2"):
        summary = runs[condition]["summary"]
        mean, median, max_v = runs[condition]["solve_stats"]
        stats = runs[condition]["latency_stats"]
        lines.extend(
            [
                f"{condition}: {summary.get('completion_reason', '?')}, "
                f"path={summary.get('path_length_m', 0) or 0:.2f} m, "
                f"min goal={summary.get('minimum_goal_distance', 99) or 99:.2f} m",
                f"  local solve mean/median/max = {mean:.0f}/{median:.0f}/{max_v:.0f} ms",
                f"  active index mean/max = {stats.get('active_control_index', np.nan):.2f}/"
                f"{stats.get('active_control_index_max', np.nan):.0f}",
                f"  YOLO inference mean/max = {stats.get('yolo_inference_ms', np.nan):.0f}/"
                f"{stats.get('yolo_inference_ms_max', np.nan):.0f} ms",
                f"  detector latency mean/max = {stats.get('detector_total_latency_s', np.nan):.2f}/"
                f"{stats.get('detector_total_latency_s_max', np.nan):.2f} s",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- The GPU/perception path is alive, but adds ~0.6 s end-to-end latency.",
            "- The optimizer is already multi-second offline; Gazebo makes it worse.",
            "- F42 logs active_control_index only at planner-result publication, so it is a handoff diagnostic.",
            "- Continuous command-tape diagnostics were added after F42 for the next run.",
        ]
    )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"F42_timing_architecture.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return runs, offline


def _write_note(runs: dict[str, dict], offline: pd.DataFrame) -> None:
    lines = [
        "# F42 - Timing Architecture Diagnosis",
        "",
        "This figure checks whether the current instability is mainly caused by small tuning changes, a weak GPU, or the runtime architecture.",
        "",
        "## Files",
        "",
        f"- Dashboard: `{OUT_DIR / 'F42_dashboard.png'}`",
        f"- Timing plot: `{OUT_DIR / 'F42_timing_architecture.png'}`",
        f"- Planner-only sweep: `{ROOT / 'logs/visibility_comparison/f42_planner_only_timing/initial_rollout_sweep.png'}`",
        f"- Gazebo log root: `{LOG_ROOT}`",
        "",
        "## Key Numbers",
        "",
    ]
    if not offline.empty:
        lines.append("Planner-only initial solves:")
        for row in offline.itertuples(index=False):
            lines.append(
                f"- {row.condition} H{int(row.horizon)}: "
                f"{row.solve_mean_ms / 1000.0:.1f} s mean solve, "
                f"{row.terminal_goal_mean_m:.2f} m mean terminal goal distance."
            )
        lines.append("")

    for condition in ("C1", "C2"):
        summary = runs[condition]["summary"]
        mean, median, max_v = runs[condition]["solve_stats"]
        stats = runs[condition]["latency_stats"]
        lines.extend(
            [
                f"{condition} Gazebo run:",
                f"- completion: `{summary.get('completion_reason', '?')}`",
                f"- path: `{summary.get('path_length_m', 0) or 0:.2f} m`, minimum goal distance: `{summary.get('minimum_goal_distance', 99) or 99:.2f} m`",
                f"- local solve mean / median / max: `{mean:.0f} / {median:.0f} / {max_v:.0f} ms`",
                f"- active control index mean / max: `{stats.get('active_control_index', np.nan):.2f} / {stats.get('active_control_index_max', np.nan):.0f}`",
                f"- YOLO inference mean / max: `{stats.get('yolo_inference_ms', np.nan):.0f} / {stats.get('yolo_inference_ms_max', np.nan):.0f} ms`",
                f"- detector total latency mean / max: `{stats.get('detector_total_latency_s', np.nan):.2f} / {stats.get('detector_total_latency_s_max', np.nan):.2f} s`",
                "",
            ]
        )

    lines.extend(
        [
            "## Conclusion",
            "",
            "The workstation is part of the wall-time problem, but it is not the only problem. YOLO is running with CUDA and typical inference is on the order of 0.07-0.11 s, while local EFE solves are multi-second and H80 initial solves are tens of seconds in the full Gazebo stack.",
            "",
            "F42 also exposed an instrumentation gap: the existing active-control fields are sampled when a planner result is published, not at every command tick. Continuous command-tape diagnostics have now been added for the next run, so we can distinguish genuine tape resets from normal handoff-time samples.",
            "",
            "Next fix should target architecture: reduce synchronous local EFE replanning, preserve command-tape phase across replans, and only replace an active tape when the new plan is fresh and meaningfully better. The next smoke run should use the new `exec_*` columns to verify whether the controller actually advances through the tape.",
        ]
    )
    (OUT_DIR / "F42_timing_architecture.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    runs, offline = _plot()
    _write_note(runs, offline)
    print(OUT_DIR / "F42_timing_architecture.png")
    print(OUT_DIR / "F42_timing_architecture.pdf")
    print(OUT_DIR / "F42_timing_architecture.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
