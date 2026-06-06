#!/usr/bin/env python3
"""F74 YOLO/perception audit for the F73 boxside route-choice run.

The F73 run did not save raw camera frames.  This script therefore reconstructs
image-space "screenshots" from logged YOLO bounding boxes, selected pixels, and
mask-bottom diagnostics.  Those reconstructions are deliberately labelled as
logs, not RGB images.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, patches


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1"
TASK = "probe_a4_boxside_north_to_a3top"
OUT = ROOT / "timing_presentation/figures/F74"


def _run_dir(cond: str) -> Path:
    matches = sorted((RUN_ROOT / TASK / cond / "seed0").glob("experiment_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory for {cond} under {RUN_ROOT}")
    return matches[-1]


def _read(cond: str) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    rd = _run_dir(cond)
    exp = pd.read_csv(rd / "experiment.csv")
    perc = pd.read_csv(rd / "perception.csv")
    summary = json.loads((rd / "run_summary.json").read_text())
    manifest_path = rd / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary["_manifest"] = manifest
    return rd, exp, perc, summary


def _first_command_stamp(exp: pd.DataFrame) -> float:
    cmd = exp.get("cmd_v", pd.Series(np.zeros(len(exp)))).abs() + exp.get("cmd_w", pd.Series(np.zeros(len(exp)))).abs()
    moving = exp.loc[cmd > 1e-3]
    if moving.empty:
        return float(exp["stamp"].iloc[0])
    return float(moving["stamp"].iloc[0])


def _runtime_exp(exp: pd.DataFrame) -> pd.DataFrame:
    t0 = _first_command_stamp(exp)
    return exp.loc[exp["stamp"] >= t0].copy()


def _runtime_perc(perc: pd.DataFrame, t0: float) -> pd.DataFrame:
    stamp_col = "log_stamp" if "log_stamp" in perc.columns else "diag_stamp"
    return perc.loc[perc[stamp_col] >= t0].copy()


def _t(series: pd.Series, t0: float) -> np.ndarray:
    return series.to_numpy(dtype=float) - t0


def _f(df: pd.DataFrame, col: str, default=np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _stat(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "n/a"
    return f"mean={x.mean():.3f}, p50={x.quantile(0.50):.3f}, p95={x.quantile(0.95):.3f}, max={x.max():.3f}"


def _rack_rects():
    # Approximate AWS rack geometry for the F73 view.
    rects = []
    xs = [-0.25, 1.8, 3.85]
    for x in xs:
        rects.append((x, -0.85, 0.55, 2.05))
        rects.append((x, 2.2, 0.55, 2.05))
    return rects


def _draw_map(ax):
    for x, y, w, h in _rack_rects():
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#d9d9d9", edgecolor="#333333", lw=0.8, zorder=0))
    # Pale driveable bands, just enough context.
    for xy, wh in [((0.45, -2.05), (2.85, 0.75)), ((0.45, 1.35), (2.85, 0.85)), ((2.85, -1.95), (0.75, 3.8))]:
        ax.add_patch(patches.Rectangle(xy, *wh, facecolor="#2fb56d", edgecolor="none", alpha=0.10, zorder=-2))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2.25, 4.55)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22, lw=0.7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _nearest_perc_rows(perc: pd.DataFrame, selectors: list[tuple[str, pd.Series]]) -> list[tuple[str, pd.Series]]:
    rows = []
    used = set()
    for label, mask_or_scores in selectors:
        if isinstance(mask_or_scores, pd.Series) and mask_or_scores.dtype == bool:
            cand = perc.loc[mask_or_scores]
            if cand.empty:
                continue
            idx = cand.index[0]
        else:
            scores = pd.to_numeric(mask_or_scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
            scores = scores.dropna()
            if scores.empty:
                continue
            idx = scores.idxmax()
        if idx in used:
            continue
        used.add(idx)
        rows.append((label, perc.loc[idx]))
    return rows


def _draw_image_row(ax, row: pd.Series, title: str):
    width, height = 1280.0, 720.0
    ax.set_facecolor("#f7f7f7")
    ax.add_patch(patches.Rectangle((0, 0), width, height, facecolor="#f7f7f7", edgecolor="#333333", lw=1.0))
    x0, y0, x1, y1 = [float(row.get(c, math.nan)) for c in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")]
    detected = float(row.get("yolo_detected_after_threshold", row.get("detected", 0.0))) >= 0.5
    color = "#16a34a" if detected else "#dc2626"
    if all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=2.0))
    u, v = float(row.get("obs_u", math.nan)), float(row.get("obs_v", math.nan))
    if math.isfinite(u) and math.isfinite(v):
        ax.scatter([u], [v], s=60, marker="x", color="#111827", lw=2, label="selected bbox-bottom")
    mu, mv = float(row.get("mask_bottom_u", math.nan)), float(row.get("mask_bottom_v", math.nan))
    if math.isfinite(mu) and math.isfinite(mv):
        ax.scatter([mu], [mv], s=35, marker="o", facecolor="none", edgecolor="#2563eb", lw=1.5, label="mask-bottom")
    score = float(row.get("yolo_score_selected", row.get("yolo_selected_score", math.nan)))
    err = float(row.get("localization_error_calibrated_m", math.nan))
    age = float(row.get("pixel_pose_age_s", math.nan))
    px = float(row.get("pred_world_x_calibrated", math.nan))
    py = float(row.get("pred_world_y_calibrated", math.nan))
    tx = float(row.get("true_x", math.nan))
    ty = float(row.get("true_y", math.nan))
    ax.text(
        12, 28,
        f"{title}\nscore={score:.3f}  detected={int(detected)}  pixel_age={age:.2f}s\n"
        f"truth=({tx:.2f},{ty:.2f})  BEV=({px:.2f},{py:.2f})  err={err:.2f}m",
        ha="left", va="top", fontsize=7.5,
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="none", pad=2.0),
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def make_summary_figure(data: dict[str, dict]):
    fig = plt.figure(figsize=(14.0, 10.0))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.1, 0.85, 0.85], hspace=0.36, wspace=0.22)

    ax_map = fig.add_subplot(gs[0, :])
    _draw_map(ax_map)
    for cond, d in data.items():
        exp = d["exp_rt"]
        perc = d["perc_rt"]
        color = "#2563eb" if cond == "C1" else "#ef4444"
        ax_map.plot(exp["truth_x"], exp["truth_y"], color=color, lw=2.3, label=f"{cond} truth")
        ax_map.plot(exp["planner_belief_x"], exp["planner_belief_y"], color=color, lw=1.1, alpha=0.38, ls="--", label=f"{cond} belief")
        detected = _f(perc, "yolo_detected_after_threshold") >= 0.5
        ax_map.scatter(_f(perc.loc[detected], "pred_world_x_calibrated"), _f(perc.loc[detected], "pred_world_y_calibrated"),
                       s=18, color=color, alpha=0.58, marker="o", label=f"{cond} fresh YOLO BEV")
        ax_map.scatter(_f(perc.loc[~detected], "pred_world_x_calibrated"), _f(perc.loc[~detected], "pred_world_y_calibrated"),
                       s=22, color=color, alpha=0.42, marker="x", label=f"{cond} stale/miss BEV")
    ax_map.scatter([3.35], [-1.55], s=80, c="#22c55e", edgecolor="black", zorder=5, label="start")
    ax_map.scatter([1.0], [1.75], s=110, c="#facc15", marker="*", edgecolor="black", zorder=5, label="goal")
    ax_map.set_title("F74 YOLO audit: truth, planner belief, and projected image detections")
    ax_map.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.90)

    for i, (cond, d) in enumerate(data.items()):
        exp, perc, t0 = d["exp_rt"], d["perc_rt"], d["t0"]
        ax = fig.add_subplot(gs[1, i])
        tt = _t(perc["log_stamp"], t0)
        ax.plot(tt, _f(perc, "localization_error_calibrated_m"), color="#111827", lw=1.4, label="YOLO BEV error")
        ax.plot(tt, _f(perc, "state_pos_error"), color="#f97316", lw=1.1, label="/state error")
        ax.plot(_t(exp["stamp"], t0), _f(exp, "truth_belief_error_m"), color="#7c3aed", lw=1.2, label="truth-belief error")
        det = _f(perc, "yolo_detected_after_threshold")
        ax.fill_between(tt, 0, np.nanmax(_f(perc, "localization_error_calibrated_m")) if len(perc) else 1,
                        where=det < 0.5, color="#ef4444", alpha=0.11, step="mid", label="YOLO below threshold")
        ax.set_title(f"{cond}: perception and belief error while driving")
        ax.set_xlabel("time after first command [s]")
        ax.set_ylabel("error [m]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

        ax2 = fig.add_subplot(gs[2, i])
        ax2.plot(tt, _f(perc, "yolo_score_selected"), color="#16a34a", lw=1.4, label="YOLO selected score")
        ax2.plot(tt, _f(perc, "pixel_pose_age_s"), color="#0f172a", lw=1.0, label="pixel pose age [s]")
        ax2.plot(_t(exp["stamp"], t0), _f(exp, "planner_pixel_correction_age_s"), color="#f59e0b", lw=1.0, label="planner correction age [s]")
        acc = _f(exp, "pixel_corr_accepted")
        rej_t = _t(exp.loc[acc < 0.5, "stamp"], t0) if "pixel_corr_accepted" in exp else np.array([])
        if len(rej_t):
            ax2.vlines(rej_t, 0, 1, color="#dc2626", alpha=0.22, lw=0.8, label="correction rejected")
        ax2.set_title(f"{cond}: detection confidence, freshness, correction rejects")
        ax2.set_xlabel("time after first command [s]")
        ax2.set_ylabel("score / age")
        ax2.set_ylim(-0.04, max(1.05, np.nanmax(_f(perc, "pixel_pose_age_s")) * 1.1 if len(perc) else 1.0))
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=7)

    fig.suptitle("F74 - Does YOLO help or hurt? Detection freshness audit (runtime rows only)", fontsize=15, weight="bold")
    fig.savefig(OUT / "F74_yolo_audit_summary.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "F74_yolo_audit_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def make_image_space_figure(data: dict[str, dict]):
    selections = []
    for cond, d in data.items():
        p = d["perc_rt"].copy()
        det = pd.to_numeric(p["yolo_detected_after_threshold"], errors="coerce") >= 0.5
        err = pd.to_numeric(p["localization_error_calibrated_m"], errors="coerce")
        score = pd.to_numeric(p["yolo_score_selected"], errors="coerce")
        if cond == "C1":
            selectors = [
                (f"{cond} best detection", det & (score > 0.7)),
                (f"{cond} worst BEV error", err),
                (f"{cond} low-score stale/miss", (~det) & (err > err.quantile(0.80))),
            ]
        else:
            selectors = [
                (f"{cond} typical detection", det & (err < err.quantile(0.55))),
                (f"{cond} worst BEV error", err),
                (f"{cond} rare miss", ~det),
            ]
        selections.extend([(cond, label, row) for label, row in _nearest_perc_rows(p, selectors)])

    n = len(selections)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.0, 3.25 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (_cond, label, row) in zip(axes, selections):
        _draw_image_row(ax, row, label)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "F74 image-space detection reconstructions (logged bbox/pixel, not saved RGB frames)",
        fontsize=14, weight="bold",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.savefig(OUT / "F74_yolo_image_space_reconstruction.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "F74_yolo_image_space_reconstruction.pdf", bbox_inches="tight")
    plt.close(fig)


def make_correction_figure(data: dict[str, dict]):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex="col")
    for col, (cond, d) in enumerate(data.items()):
        exp, t0 = d["exp_rt"], d["t0"]
        tt = _t(exp["stamp"], t0)
        pred_err = np.hypot(_f(exp, "pixel_corr_pred_x") - _f(exp, "truth_x"), _f(exp, "pixel_corr_pred_y") - _f(exp, "truth_y"))
        next_err = np.hypot(_f(exp, "pixel_corr_next_x") - _f(exp, "truth_x"), _f(exp, "pixel_corr_next_y") - _f(exp, "truth_y"))
        accepted = _f(exp, "pixel_corr_accepted")
        axes[0, col].plot(tt, pred_err, color="#64748b", lw=1.0, label="before correction")
        axes[0, col].plot(tt, next_err, color="#7c3aed", lw=1.2, label="after correction")
        if len(tt):
            axes[0, col].fill_between(tt, 0, np.nanmax([np.nanmax(pred_err), np.nanmax(next_err)]),
                                      where=accepted < 0.5, color="#ef4444", alpha=0.10, step="mid", label="rejected/latest reject")
        axes[0, col].set_title(f"{cond}: correction effect relative to truth")
        axes[0, col].set_ylabel("error [m]")
        axes[0, col].grid(True, alpha=0.25)
        axes[0, col].legend(fontsize=7)

        delta = pred_err - next_err
        axes[1, col].plot(tt, delta, color="#0f766e", lw=1.0, label="positive = correction helped")
        axes[1, col].axhline(0.0, color="black", lw=0.8)
        axes[1, col].plot(tt, _f(exp, "pixel_corr_xy_update_norm_m"), color="#f97316", lw=1.0, alpha=0.8, label="update norm [m]")
        axes[1, col].set_xlabel("time after first command [s]")
        axes[1, col].set_ylabel("m")
        axes[1, col].grid(True, alpha=0.25)
        axes[1, col].legend(fontsize=7)
    fig.suptitle("F74 pixel-correction audit: are camera updates improving the planner belief?", fontsize=14, weight="bold")
    fig.savefig(OUT / "F74_pixel_correction_effect.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "F74_pixel_correction_effect.pdf", bbox_inches="tight")
    plt.close(fig)


def _condition_summary(cond: str, d: dict) -> str:
    perc = d["perc_rt"]
    exp = d["exp_rt"]
    detected = pd.to_numeric(perc["yolo_detected_after_threshold"], errors="coerce") >= 0.5
    accepted = pd.to_numeric(exp["pixel_corr_accepted"], errors="coerce")
    rejected = accepted < 0.5
    reject_reasons = exp.loc[rejected, "pixel_corr_reject_reason"].value_counts(dropna=False).to_dict() if "pixel_corr_reject_reason" in exp else {}
    # Use the runtime pixel timeout scale, not an arbitrary camera-frame
    # freshness threshold.  New logs contain pixel_pose_fresh explicitly.
    timeout = float(d["summary"].get("_manifest", {}).get("pixel_timeout_s", 1.25) or 1.25)
    if "pixel_pose_fresh" in perc.columns:
        fresh = pd.to_numeric(perc["pixel_pose_fresh"], errors="coerce") >= 0.5
    else:
        fresh = pd.to_numeric(perc["pixel_pose_age_s"], errors="coerce") <= timeout
    return (
        f"### {cond}\n"
        f"- Run directory: `{d['run_dir']}`\n"
        f"- Outcome: `{d['summary'].get('completion_reason', d['summary'].get('outcome', 'unknown'))}`; "
        f"path `{d['summary'].get('path_length_m', float('nan')):.3f} m`; "
        f"min goal `{d['summary'].get('minimum_goal_distance', float('nan')):.3f} m`.\n"
        f"- Runtime perception rows: `{len(perc)}` after first command.\n"
        f"- YOLO detection rate: `{detected.mean():.3f}` (`{int(detected.sum())}/{len(detected)}`).\n"
        f"- Pixel-pose within runtime timeout (`age <= {timeout:.2f} s`): `{fresh.mean():.3f}`. "
        f"Important: `pixel_pose_available=1` can still mean latest stale pose exists.\n"
        f"- YOLO selected score: {_stat(perc['yolo_score_selected'])}.\n"
        f"- YOLO BEV localization error: {_stat(perc['localization_error_calibrated_m'])} m.\n"
        f"- `/state` position error: {_stat(perc['state_pos_error'])} m.\n"
        f"- Planner truth-belief error: {_stat(exp['truth_belief_error_m'])} m.\n"
        f"- YOLO latency: {_stat(perc['yolo_latency_s'])} s; inference: {_stat(perc['yolo_inference_ms'])} ms.\n"
        f"- Latest pixel-correction diagnostic accepted flag rate: `{(accepted >= 0.5).mean():.3f}`; "
        f"latest reject reasons: `{reject_reasons}`. This is a sampled diagnostic stream, not a count of unique correction events.\n"
    )


def write_note(data: dict[str, dict]):
    c1, c2 = data["C1"], data["C2"]
    md = f"""# F74 YOLO / State / Belief Audit

This audit uses the F73 boxside route-choice run:

- Log root: `{RUN_ROOT}`
- Task: `{TASK}`
- Config: `scripts/visibility_comparison/aws_probe_boxside_north_route_choice_config.yaml`
- Output folder: `{OUT}`

All metrics below exclude launch/global-solve idle time and start at the first non-trivial command.

## Figures

- `F74_yolo_audit_summary.png`: BEV truth/belief/projected detections plus time-series of detection, state, and belief error.
- `F74_yolo_image_space_reconstruction.png`: reconstructed image-space YOLO boxes and selected pixels from logged diagnostics. These are **not raw RGB screenshots** because F73 did not save camera frames.
- `F74_pixel_correction_effect.png`: whether pixel corrections improved or worsened the planner belief relative to truth.

{_condition_summary("C1 constant-R", c1)}

{_condition_summary("C2 GP-aware", c2)}

## What Is Going Wrong?

The high `/state` error is not evidence that every fresh YOLO detection is bad. It is mostly a
freshness/blackout problem:

1. The detector publishes `/perception/pixel_pose` only when `yolo_detected_after_threshold=1`.
2. The logger records `pixel_pose_available=1` whenever a latest pixel pose exists, even if the current frame is a miss.
3. During C1's weak-visibility segment, many frames are below threshold. The latest pixel/state can remain stale while the robot keeps moving.
4. That stale or low-confidence camera state is then a poor representation of the robot's current truth pose.

So the right reading is:

- C2: YOLO is useful. It stays in a visible route, detections remain high-confidence, and belief error stays small.
- C1: YOLO becomes unavailable/unreliable in the chosen route. The state stream can look available while effectively stale, and pixel corrections are neutral or slightly harmful. This is exactly the failure mode the visibility-aware planner should avoid.

## Does YOLO Add Something?

Yes, but only when the robot remains in regions where detections are fresh and geometrically valid.
For C2, the detector provides dense, high-score updates and keeps the planner belief close to truth.
For C1, the detector blackout makes `/state` misleading unless freshness is handled explicitly.

## Implementation Notes

- Raw camera frames were not stored in this run; future perception audits should enable optional image snapshots for selected frames.
- `frame_age_at_publish_s` is not reliable in this log because it mixes time bases. Prefer `yolo_latency_s`, `yolo_inference_ms`, and explicit pixel/correction ages.
- Paper/result metrics should distinguish:
  - fresh YOLO detection error,
  - stale latest-state error,
  - planner truth-belief error,
  - and final task outcome.

## Implemented Follow-Up For Future Runs

This audit motivated an explicit freshness field for camera-derived state:

- `experiment.csv` now logs `state_age_s` and `state_fresh`.
- `perception.csv` now logs `pixel_pose_fresh`.
- `run_manifest.json` records `pixel_timeout_s`.
- The planner no longer resets belief from stale `/state/bev` after an implausible delayed correction.

Future dashboards should use these fields to separate fresh camera-state error from stale latest-state error.

Do not remove `dt_implausible`; it is a stale-correction guard. The issue is how the system behaves after a rejected or missing update.
"""
    (OUT / "F74_yolo_audit.md").write_text(md)


def write_row_tables(data: dict[str, dict]):
    rows = []
    for cond, d in data.items():
        p = d["perc_rt"].copy()
        p.insert(0, "condition", cond)
        p.insert(1, "time_after_first_command_s", p["log_stamp"].astype(float) - float(d["t0"]))
        keep = [
            "condition", "time_after_first_command_s",
            "detected", "yolo_detected_after_threshold", "yolo_score_selected",
            "pixel_pose_available", "pixel_pose_age_s", "pixel_pose_fresh",
            "true_x", "true_y", "state_x", "state_y", "state_age_s", "state_fresh", "state_pos_error",
            "pred_world_x_calibrated", "pred_world_y_calibrated", "localization_error_calibrated_m",
            "obs_u", "obs_v", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
            "mask_bottom_u", "mask_bottom_v", "mask_area_px", "yolo_inference_ms", "yolo_latency_s",
        ]
        rows.append(p[[c for c in keep if c in p.columns]])
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "F74_yolo_runtime_rows.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for cond in ("C1", "C2"):
        rd, exp, perc, summary = _read(cond)
        t0 = _first_command_stamp(exp)
        data[cond] = {
            "run_dir": rd,
            "exp": exp,
            "perc": perc,
            "summary": summary,
            "t0": t0,
            "exp_rt": _runtime_exp(exp),
            "perc_rt": _runtime_perc(perc, t0),
        }

    make_summary_figure(data)
    make_image_space_figure(data)
    make_correction_figure(data)
    write_row_tables(data)
    write_note(data)
    print(OUT / "F74_yolo_audit_summary.png")
    print(OUT / "F74_yolo_image_space_reconstruction.png")
    print(OUT / "F74_pixel_correction_effect.png")
    print(OUT / "F74_yolo_runtime_rows.csv")
    print(OUT / "F74_yolo_audit.md")


if __name__ == "__main__":
    main()
