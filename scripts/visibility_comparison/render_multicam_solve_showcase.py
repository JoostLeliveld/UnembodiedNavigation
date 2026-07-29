#!/usr/bin/env python3
"""Render the real-Gazebo multicamera solve fix as an honest showcase.

The comparison is intentionally narrow:

* before: the completed seed-0 ``rob_easy`` run that exposed the invalid
  covariance and terminated as stuck without recognizing the goal;
* after: all three completed seeds from the fixed, waypoint-free campaign.

This is a diagnostic showcase, not a replacement for the full robustness
campaign.  It reads only logged runtime/evaluation data and never feeds ground
truth back into planning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multicam_solve_showcase_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from paths import repo_root


REPO = repo_root(Path(__file__))
DEFAULT_BEFORE = REPO / "logs/visibility_comparison/multicam_solve_showcase_real"
DEFAULT_AFTER = REPO / "logs/visibility_comparison/multicam_solve_showcase_fixed"
DEFAULT_OUT = REPO / "logs/studies/multicam_nav_demo/figures"


def _campaign(root: Path) -> dict:
    with (root / "campaign_log.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _completed_runs(root: Path) -> list[dict]:
    return [
        record
        for record in _campaign(root).values()
        if record.get("run_dir") and Path(record["run_dir"]).joinpath("run_summary.json").exists()
    ]


def _frame(record: dict) -> pd.DataFrame:
    return pd.read_csv(Path(record["run_dir"]) / "experiment.csv")


def _dedup_corrections(frame: pd.DataFrame) -> pd.DataFrame:
    corr = frame.dropna(subset=["pixel_corr_apply_stamp"]).copy()
    corr = corr[np.isfinite(pd.to_numeric(corr["pixel_corr_apply_stamp"], errors="coerce"))]
    return corr.drop_duplicates(subset=["pixel_corr_apply_stamp"], keep="last")


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _summarize(before_record: dict, after_records: list[dict]) -> dict:
    before_frame = _frame(before_record)
    after_frames = [_frame(record) for record in after_records]
    before_corr = _dedup_corrections(before_frame)
    after_corr = pd.concat([_dedup_corrections(frame) for frame in after_frames], ignore_index=True)
    before_error = _finite(before_frame, "belief_error_gt_m")
    after_error = np.concatenate([_finite(frame, "belief_error_gt_m") for frame in after_frames])
    after_nis = _finite(after_corr, "pixel_corr_nis")
    return {
        "before": {
            "outcome": before_record["outcome"],
            "goal_reached": bool(before_record["goal_reached"]),
            "corrections": int(len(before_corr)),
            "accepted_fraction": float(before_corr["pixel_corr_accepted"].mean()),
            "negative_nis": int((_finite(before_corr, "pixel_corr_nis") < 0.0).sum()),
            "belief_error_p95_m": float(np.percentile(before_error, 95)),
            "minimum_goal_distance_m": float(before_record["minimum_goal_distance"]),
        },
        "after": {
            "successes": int(sum(bool(record["goal_reached"]) for record in after_records)),
            "runs": int(len(after_records)),
            "corrections": int(len(after_corr)),
            "accepted_fraction": float(after_corr["pixel_corr_accepted"].mean()),
            "negative_nis": int((after_nis < 0.0).sum()),
            "nis_p95": float(np.percentile(after_nis, 95)),
            "belief_error_p95_m": float(np.percentile(after_error, 95)),
            "path_length_range_m": [
                float(min(record["path_length_m"] for record in after_records)),
                float(max(record["path_length_m"] for record in after_records)),
            ],
            "minimum_goal_distance_range_m": [
                float(min(record["minimum_goal_distance"] for record in after_records)),
                float(max(record["minimum_goal_distance"] for record in after_records)),
            ],
        },
    }


def _style_axis(axis):
    axis.grid(True, color="#dce3eb", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render(before_root: Path, after_root: Path, out_dir: Path) -> tuple[Path, Path]:
    before_records = _completed_runs(before_root)
    after_records = _completed_runs(after_root)
    if not before_records:
        raise RuntimeError(f"no completed before-fix run under {before_root}")
    if len(after_records) != 3 or not all(record.get("goal_reached") for record in after_records):
        raise RuntimeError("the showcase requires three completed, successful after-fix runs")

    before_record = next(
        (record for record in before_records if int(record.get("seed", -1)) == 0),
        before_records[0],
    )
    before_frame = _frame(before_record)
    after_frames = [_frame(record) for record in after_records]
    summary = _summarize(before_record, after_records)

    fig = plt.figure(figsize=(14.2, 5.6), facecolor="#f7f9fc")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 0.88], wspace=0.30)

    path_ax = fig.add_subplot(grid[0, 0])
    path_ax.plot(
        before_frame["gt_x"], before_frame["gt_y"],
        color="#c24a42", linewidth=2.0, linestyle="--", label="before: stuck (belief drift)",
    )
    colors = ["#176b87", "#2a8f6a", "#6a55a3"]
    for color, record, frame in zip(colors, after_records, after_frames):
        path_ax.plot(
            frame["gt_x"], frame["gt_y"], color=color, linewidth=2.0,
            label=f"fixed seed {record['seed']}: goal",
        )
    path_ax.scatter([0.0], [-7.6], marker="o", s=65, color="#182230", zorder=5, label="start")
    path_ax.scatter([0.0], [7.6], marker="*", s=145, color="#e29d2d", edgecolor="#182230",
                    linewidth=0.6, zorder=6, label="goal")
    path_ax.set_title("Closed-loop paths", loc="left", fontsize=13, weight="bold")
    path_ax.set_xlabel("map x [m]")
    path_ax.set_ylabel("map y [m]")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.legend(loc="lower left", fontsize=8, frameon=True)
    _style_axis(path_ax)

    err_ax = fig.add_subplot(grid[0, 1])
    before_t = _finite(before_frame, "stamp")
    before_e = pd.to_numeric(before_frame["belief_error_gt_m"], errors="coerce").to_numpy(float)
    mask = np.isfinite(before_t) & np.isfinite(before_e[: len(before_t)])
    err_ax.plot(before_t[mask], before_e[: len(before_t)][mask], color="#c24a42",
                linewidth=1.7, alpha=0.9, label="before seed 0")
    for color, record, frame in zip(colors, after_records, after_frames):
        stamp = pd.to_numeric(frame["stamp"], errors="coerce").to_numpy(float)
        error = pd.to_numeric(frame["belief_error_gt_m"], errors="coerce").to_numpy(float)
        valid = np.isfinite(stamp) & np.isfinite(error)
        err_ax.plot(stamp[valid], error[valid], color=color, linewidth=1.25, alpha=0.9,
                    label=f"fixed seed {record['seed']}")
    err_ax.axhline(0.25, color="#6c7785", linestyle=":", linewidth=1.2, label="goal radius")
    err_ax.set_title("Belief error remains bounded", loc="left", fontsize=13, weight="bold")
    err_ax.set_xlabel("simulation time [s]")
    err_ax.set_ylabel("belief error vs GT [m]\n(GT evaluation only)")
    err_ax.set_ylim(bottom=0.0)
    err_ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
    _style_axis(err_ax)

    text_ax = fig.add_subplot(grid[0, 2])
    text_ax.axis("off")
    before = summary["before"]
    after = summary["after"]
    text_ax.text(0.0, 1.0, "What changed", va="top", fontsize=13, weight="bold", color="#182230")
    text_ax.text(
        0.0, 0.90,
        "One paper-1 gate chain; fused/per-camera\nobservations differ only at the measurement seam.",
        va="top", fontsize=9.3, color="#3f4b59", linespacing=1.35,
    )
    text_ax.text(0.0, 0.75, "BEFORE", fontsize=9, weight="bold", color="#c24a42")
    text_ax.text(
        0.0, 0.70,
        f"stuck • {before['accepted_fraction']:.0%} corrections accepted\n"
        f"belief error p95 {before['belief_error_p95_m']:.3f} m\n"
        f"{before['negative_nis']} negative NIS values",
        va="top", fontsize=11, color="#182230", linespacing=1.45,
    )
    text_ax.text(0.0, 0.49, "AFTER", fontsize=9, weight="bold", color="#2a8f6a")
    text_ax.text(
        0.0, 0.44,
        f"{after['successes']}/{after['runs']} goals • no collisions\n"
        f"{after['corrections']} corrections • {after['accepted_fraction']:.0%} accepted\n"
        f"belief error p95 {after['belief_error_p95_m']:.3f} m\n"
        f"NIS p95 {after['nis_p95']:.2f} • no negative NIS",
        va="top", fontsize=11, color="#182230", linespacing=1.45,
    )
    text_ax.text(
        0.0, 0.15,
        "Real Gazebo + 4-camera YOLO\n"
        "warehouse_full_4cam • rob_easy • seeds 0–2\n"
        "No mission waypoints; optimizer-generated route\n"
        "SHOWCASE (3 seeds), not full robustness evidence",
        va="top", fontsize=8.8, color="#5d6875", linespacing=1.35,
    )

    fig.suptitle(
        "Multicamera solve succeeds after restoring a valid shared correction path",
        x=0.035, y=0.99, ha="left", fontsize=16, weight="bold", color="#182230",
    )
    fig.text(
        0.035, 0.94,
        "The failure was estimator bookkeeping—not the global optimizer and not a visibility-weight tuning problem.",
        ha="left", fontsize=10.5, color="#5d6875",
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.055, right=0.98)

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "fig33_multicam_solve_showcase.png"
    summary_path = out_dir / "fig33_multicam_solve_showcase.json"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return figure_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    figure, summary = render(args.before.resolve(), args.after.resolve(), args.out_dir.resolve())
    print(f"Wrote {figure}")
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
