#!/usr/bin/env python3
"""D4 demo — per-camera projection bias, made visible (four-panel grammar).

Follows the fixed demo grammar in
``research_story/DEMO_LAYER_PLAN_2026-07-16.md``: P1 the problem happening,
P2 the mechanism overlaid, P3 with-vs-without paired, P4 the verdict anchored
to the frozen gate. Every projected point here is RECOMPUTED from raw
``obs_u/obs_v`` pixels via ``reliability.projection`` (never read from a CSV's
``pred_world_x/y``, which may already carry a stale or partial calibration) —
this is the same discipline ``fit_projection_calibration.py`` uses, so the
figure and the fit it is illustrating cannot silently disagree.

Inputs: the two GT-attached commissioning runs
(``gt_validation_smoke_20260716``, ``gt_validation_smoke2_20260716``,
``evaluation_inputs/*_perception.csv`` — GT-eval-only, never a model input)
and the frozen ``projection_calibration_v2/projection_calibration.json``.
Zero Gazebo dependency; re-renders in seconds from committed data.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for relative in ("src/reliability", "src/unav_common"):
    location = str(REPO / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

from reliability.contracts import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world,
    load_projection_calibration,
    project_observation_to_world,
)

STUDY = REPO / "logs/studies/multicamera_commissioning_bigwarehouse"
RUN_1 = STUDY / "gt_validation_smoke_20260716"
RUN_2 = STUDY / "gt_validation_smoke2_20260716"
CALIBRATION = STUDY / "projection_calibration_v2/projection_calibration.json"
WORLD_SDF = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
OUT = STUDY / "demos/d4_camera_method"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
COLORS = {
    "camera_A": "#2f80ed",
    "camera_B": "#21a366",
    "camera_C": "#8d53c7",
    "camera_D": "#ed8a25",
}
DISAGREEMENT_GATE_M = 0.30
# TB3 Burger nominal footprint width (documented spec, not remeasured from
# this sim's exact mesh) — used only as a visual scale anchor.
ROBOT_FOOTPRINT_W_M = 0.178
CAMERA_MARKER_XY = {
    "camera_A": (-6.0, -9.0),
    "camera_B": (-6.0, 9.0),
    "camera_C": (6.0, -9.0),
    "camera_D": (6.0, 9.0),
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _models() -> dict[str, object]:
    return {
        camera_id: camera_model_from_world(WORLD_SDF, include_name=include)
        for camera_id, include in MODEL_INCLUDES.items()
    }


def _raw_and_corrected(
    row: dict[str, str],
    camera_id: str,
    model,
    calibration: dict[str, dict[str, float]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if row.get("detected") != "1" or not row.get("obs_u"):
        return None, None
    observation = CameraObservation(
        camera_id=camera_id,
        timestamp_s=_f(row, "diag_stamp"),
        pixel_uv=(_f(row, "obs_u"), _f(row, "obs_v")),
        detection_valid=True,
    )
    raw = project_observation_to_world(observation, model, contact_z_m=0.05)
    cal = calibration.get(camera_id, {"intercept_m": 0.0, "slope_per_m": 0.0})
    corrected = project_observation_to_world(
        observation,
        model,
        contact_z_m=0.05,
        along_bearing_offset_m=cal["intercept_m"],
        along_bearing_slope_per_m=cal["slope_per_m"],
    )
    return raw, corrected


def _draw_map_base(axis, *, show_camera_markers: bool = True) -> None:
    axis.add_patch(Rectangle((-12.25, -10.25), 24.5, 20.5, fill=False, edgecolor="#3a4a5b", linewidth=1.0, clip_on=True))
    if show_camera_markers:
        for camera_id, (x, y) in CAMERA_MARKER_XY.items():
            axis.scatter(x, y, s=26, color=COLORS[camera_id], edgecolors="#182534", linewidths=0.5, zorder=5, clip_on=True)
            # Text is NOT clipped by axis limits by default in matplotlib — an
            # off-screen marker's label would otherwise float in the figure
            # margin outside the axes box. clip_on=True keeps it honestly gone
            # when the view is zoomed past the camera mounts (as in P1).
            axis.text(x, y + (0.5 if y < 0 else -0.65), camera_id[-1], ha="center", va="center",
                      fontsize=7.5, weight="bold", clip_on=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("warehouse x [m]", fontsize=8)
    axis.set_ylabel("warehouse y [m]", fontsize=8)
    axis.grid(color="#dce3eb", linewidth=0.6)


def render_p1_problem_happening() -> Path:
    """P1: same robot pose, four cameras report four different positions."""

    rows_by_camera = {cam: _rows(RUN_2 / "evaluation_inputs" / f"{cam}_perception.csv") for cam in CAMERAS}
    models = _models()

    # t=58.0s in the central_overlap_sweep run (run 2) is a genuine
    # simultaneous B/C/D detection of the same robot pose — found by scanning
    # for the densest cross-camera overlap rather than assumed.
    target_stamp = 58.0
    points: dict[str, tuple[float, float]] = {}
    truth: tuple[float, float] | None = None
    for camera_id, rows in rows_by_camera.items():
        model = models[camera_id]
        detected_rows = [row for row in rows if row.get("detected") == "1" and row.get("true_x")]
        if not detected_rows:
            continue
        best = min(detected_rows, key=lambda row: abs(_f(row, "diag_stamp") - target_stamp))
        if abs(_f(best, "diag_stamp") - target_stamp) > 0.3:
            continue
        raw, _ = _raw_and_corrected(best, camera_id, model, {})
        if raw is None:
            continue
        points[camera_id] = raw
        if best.get("true_x"):
            truth = (_f(best, "true_x"), _f(best, "true_y"))

    # This panel deliberately zooms in past the camera mounts (which sit at
    # +-6, +-9) to the robot-scale cluster of disagreement, so the mount
    # markers/labels are correctly out of frame here.
    fig, axis = plt.subplots(figsize=(7.4, 6.9), dpi=170)
    _draw_map_base(axis, show_camera_markers=False)
    if truth is not None:
        axis.scatter(*truth, s=170, marker="*", color="#17212f", zorder=8, label="ground truth (eval-only)")
    for camera_id, (x, y) in points.items():
        axis.scatter(x, y, s=90, color=COLORS[camera_id], edgecolors="white", linewidths=1.0, zorder=7,
                     label=f"{camera_id} reported")
    if len(points) >= 2:
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        spread_m = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        axis.text(
            0.03, 0.03,
            f"same robot, same instant (t={target_stamp:.1f}s) — cameras disagree by up to {spread_m * 100:.0f} cm",
            transform=axis.transAxes, fontsize=9.5, weight="bold", color="#c81919",
        )
    axis.set_xlim(-1.0, 1.5)
    axis.set_ylim(-1.3, 1.0)
    axis.set_title("P1 — the problem: one robot, several reported positions", fontsize=13, weight="bold")
    axis.legend(loc="upper right", fontsize=8, frameon=True)
    output = OUT / "p1_problem_happening.png"
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def _bias_arrows_panel(axis, run_dir: Path, calibration: dict[str, dict[str, float]], title: str, stride: int = 3) -> tuple[list[float], list[float]]:
    models = _models()
    _draw_map_base(axis, show_camera_markers=False)
    all_errors: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for camera_id in CAMERAS:
        source = run_dir / "evaluation_inputs" / f"{camera_id}_perception.csv"
        if not source.exists():
            continue
        model = models[camera_id]
        rows = [row for row in _rows(source) if row.get("detected") == "1" and row.get("true_x")]
        for row in rows[::stride]:
            _, corrected = _raw_and_corrected(row, camera_id, model, calibration)
            if corrected is None:
                continue
            tx, ty = _f(row, "true_x"), _f(row, "true_y")
            error = math.hypot(corrected[0] - tx, corrected[1] - ty)
            all_errors.append(error)
            xs.extend([tx, corrected[0]])
            ys.extend([ty, corrected[1]])
            axis.add_patch(
                FancyArrow(
                    tx, ty, corrected[0] - tx, corrected[1] - ty,
                    width=0.006, head_width=0.05, head_length=0.05,
                    color=COLORS[camera_id], alpha=0.6, length_includes_head=True, zorder=6,
                    clip_on=True,
                )
            )
    axis.set_title(title, fontsize=11, weight="bold")
    return all_errors, [xs, ys] if xs else [[], []]


def render_p2_p3_bias_before_after() -> Path:
    """P2/P3 combined: raw bias arrows vs after the frozen v2 calibration."""

    calibration = load_projection_calibration(CALIBRATION)
    # Run 2 (central_overlap_sweep) is used here, not run 1: run 1 predates
    # the GPU-OOM fix and has zero camera_A rows, and its route is a thin
    # near-vertical corridor that would crop out most cameras' detections
    # under any single fixed extent. Run 2 has all four cameras active along
    # one shared aisle sweep, so this panel can show all of them at once.
    # constrained_layout fights the fixed-equal-aspect axes here (matplotlib
    # will shrink the axes boxes to nothing trying to also fit a long
    # suptitle) — lay out manually and crop with bbox_inches="tight" instead.
    # The overlap-sweep data is wide and shallow (~4 m x ~1 m); a short
    # figure avoids the dead white band a tall figure leaves above/below the
    # equal-aspect-shrunk axes boxes.
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 3.6), dpi=170)
    raw_errors, raw_extent = _bias_arrows_panel(axes[0], RUN_2, {}, "P2 — raw: arrow = reported → truth (all pull toward their own camera)")
    corrected_errors, corr_extent = _bias_arrows_panel(axes[1], RUN_2, calibration, "P3 — after projection_calibration_v2 (frozen, commissioning-time constants)")

    all_xs = raw_extent[0] + corr_extent[0]
    all_ys = raw_extent[1] + corr_extent[1]
    if all_xs and all_ys:
        pad = 0.4
        xlim = (min(all_xs) - pad, max(all_xs) + pad)
        ylim = (min(all_ys) - pad, max(all_ys) + pad)
        for axis in axes:
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[c], markersize=8, label=c) for c in CAMERAS]
    axes[1].legend(handles=handles, loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    if raw_errors and corrected_errors:
        fig.suptitle(
            f"Mean arrow length {sum(raw_errors) / len(raw_errors) * 100:.0f} cm → "
            f"{sum(corrected_errors) / len(corrected_errors) * 100:.0f} cm "
            "(camera C keeps a residual cross-bearing pull near the central pillar — an occlusion effect, not projection)",
            fontsize=10, color="#536273", y=0.98,
        )
    output = OUT / "p2_p3_bias_before_after.png"
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def render_p4_verdict_anchored() -> Path:
    """P4: C-D synchronized disagreement before/after, gate + robot-scale anchor."""

    calibration = load_projection_calibration(CALIBRATION)
    models = _models()

    def sync_pairs(run_dir: Path, calibrated: bool) -> list[float]:
        cal = calibration if calibrated else {}
        points: dict[str, dict[float, tuple[float, float]]] = {}
        for camera_id in ("camera_C", "camera_D"):
            source = run_dir / "evaluation_inputs" / f"{camera_id}_perception.csv"
            if not source.exists():
                continue
            model = models[camera_id]
            cam_points = {}
            for row in _rows(source):
                if row.get("detected") != "1":
                    continue
                _, point = _raw_and_corrected(row, camera_id, model, cal)
                if point is None:
                    continue
                cam_points[round(_f(row, "diag_stamp"), 1)] = point
            points[camera_id] = cam_points
        if "camera_C" not in points or "camera_D" not in points:
            return []
        common = sorted(set(points["camera_C"]) & set(points["camera_D"]))
        return [
            math.hypot(
                points["camera_C"][t][0] - points["camera_D"][t][0],
                points["camera_C"][t][1] - points["camera_D"][t][1],
            )
            for t in common
        ]

    fig, axis = plt.subplots(figsize=(9.4, 5.6), dpi=170)
    for run_idx, (label, run_dir) in enumerate((("run 1", RUN_1), ("run 2", RUN_2))):
        raw = sync_pairs(run_dir, calibrated=False)
        corrected = sync_pairs(run_dir, calibrated=True)
        x_raw = run_idx * 3 + 0.0
        x_corr = run_idx * 3 + 1.0
        if raw:
            axis.scatter([x_raw] * len(raw), raw, s=18, color="#c81919", alpha=0.45, label="raw" if run_idx == 0 else None)
            axis.scatter([x_raw], [sum(raw) / len(raw)], s=140, marker="_", color="#c81919", linewidths=3, zorder=6)
        if corrected:
            axis.scatter([x_corr] * len(corrected), corrected, s=18, color="#21a366", alpha=0.55, label="calibrated (v2)" if run_idx == 0 else None)
            axis.scatter([x_corr], [sum(corrected) / len(corrected)], s=140, marker="_", color="#21a366", linewidths=3, zorder=6)
        axis.text((x_raw + x_corr) / 2.0, -0.03, label, ha="center", fontsize=9.5, transform=axis.get_xaxis_transform())

    axis.axhline(DISAGREEMENT_GATE_M, color="#17212f", linestyle="--", linewidth=1.5, label=f"frozen D2 gate = {DISAGREEMENT_GATE_M:.2f} m")
    # Robot-footprint scale anchor drawn as a short vertical bar at the axis edge.
    axis.plot([4.6, 4.6], [0.0, ROBOT_FOOTPRINT_W_M], color="#536273", linewidth=3.0, solid_capstyle="butt")
    axis.text(4.68, ROBOT_FOOTPRINT_W_M / 2.0, f"TB3 footprint\nwidth ≈ {ROBOT_FOOTPRINT_W_M:.2f} m", fontsize=7.5, color="#536273", va="center")
    axis.set_xlim(-0.6, 5.4)
    axis.set_xticks([])
    axis.set_ylabel("C↔D synchronized projected disagreement [m]")
    axis.set_title("P4 — verdict: C↔D disagreement vs the frozen 0.30 m gate", fontsize=12, weight="bold")
    axis.legend(loc="upper right", fontsize=8, frameon=True)
    axis.grid(axis="y", color="#dce3eb", linewidth=0.6)
    output = OUT / "p4_verdict_anchored.png"
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "p1_problem_happening": str(render_p1_problem_happening()),
        "p2_p3_bias_before_after": str(render_p2_p3_bias_before_after()),
        "p4_verdict_anchored": str(render_p4_verdict_anchored()),
    }
    import json

    manifest = {
        "demo": "d4_camera_method",
        "claim": (
            "Per-camera projection carried a distance-dependent near-edge bias "
            "pulling each camera toward its own wall; a frozen along-bearing "
            "calibration (intercept + slope*distance), fit against simulation "
            "truth, collapses the C↔D disagreement from 0.247 m to 0.078-0.107 m."
        ),
        "animation": "NOT YET BUILT — next step: per-frame raw-vs-corrected arrow "
        "sweep across a full route (see DEMO_LAYER_PLAN_2026-07-16.md panel A)",
        "data_sources": {
            "gt": "GT-eval-only, /ground_truth_tf via record_evaluation_truth.py",
            "projections": "recomputed from raw obs_u/obs_v via reliability.projection",
            "calibration": str(CALIBRATION.relative_to(REPO)),
        },
        "gate": {"max_cross_camera_disagreement_m": DISAGREEMENT_GATE_M},
        "outputs": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
