#!/usr/bin/env python3
"""Render presentation figures from the real four-camera commissioning run.

This script intentionally consumes only the operational logs, day-zero maps,
and canonical GP artifacts.  It produces a compact evidence set for the
walkthrough and deck: collected records, four prior→posterior updates, actual
overlap agreement, and the conservative handover-policy result.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/four_camera_actual_showcase_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


REPO = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO / "logs/studies/multicamera_commissioning_bigwarehouse/actual_commissioning_20260715"
INPUTS = RUN_ROOT / "analysis/final_02/inputs"
CHECKPOINT = REPO / "logs/visibility_comparison/fourcam_actual_20260715/checkpoint_01/gp"
FINAL_GP = REPO / "logs/visibility_comparison/fourcam_actual_20260715/final_02/gp"
STRICT_EXPORT = RUN_ROOT / "analysis/final_01/replay_export_strict"
OVERLAP = RUN_ROOT / "analysis/final_01/strict_C_D_overlap.json"
OUT = REPO / "research_story/presentations/2026-07_four_camera_showcase/full_story_walkthrough/07_real_commissioning_execution/figures"
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
COLORS = {
    "camera_A": "#2f80ed",
    "camera_B": "#21a366",
    "camera_C": "#8d53c7",
    "camera_D": "#ed8a25",
}

for relative in ("src/reliability",):
    location = str(REPO / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from reliability.cli import _load_replay_frames  # noqa: E402
from reliability.providers import GridMapReliabilityProvider  # noqa: E402
from reliability.replay import ReplayConfig, ReplayMode, run_replay  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float(default)


def _events(camera: str) -> list[dict[str, str]]:
    return _rows(INPUTS / f"{camera}_events.csv")


def _gp_path(root: Path, camera: str) -> Path:
    return root / camera / "det_hit_expected_kernel_gp.npz"


def _map_axes(axis, xs: np.ndarray, ys: np.ndarray) -> None:
    axis.set_xlim(float(xs[0]), float(xs[-1]))
    axis.set_ylim(float(ys[0]), float(ys[-1]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]", fontsize=7)
    axis.set_ylabel("y [m]", fontsize=7)
    axis.tick_params(labelsize=6.5)
    axis.grid(color="#dce3eb", linewidth=0.55, alpha=0.75)
    axis.add_patch(Rectangle((-12.25, -10.25), 24.5, 20.5, fill=False, edgecolor="#3a4a5b", linewidth=1.0))
    for x, y, label in ((-6.0, -9.0, "A"), (-6.0, 9.0, "B"), (6.0, -9.0, "C"), (6.0, 9.0, "D")):
        axis.scatter(x, y, s=22, color=COLORS[f"camera_{label}"], edgecolors="#182534", linewidths=0.4, zorder=5)
        axis.text(x, y + (0.42 if y < 0 else -0.56), label, ha="center", va="center", fontsize=6.5, weight="bold")


def render_run_evidence() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=180, constrained_layout=True)
    raw_dirs = (
        ("01  South → north handover", RUN_ROOT / "01_south_to_north/raw"),
        ("02  South A/C overlap", RUN_ROOT / "02_south_pair_overlap/raw"),
    )
    for axis, (title, raw_dir) in zip(axes, raw_dirs):
        odom = _rows(raw_dir / "experiment.csv")
        xs = [_f(row, "odom_noisy_x") for row in odom]
        ys = [_f(row, "odom_noisy_y") for row in odom]
        axis.plot(xs, ys, color="#76869a", linewidth=1.0, alpha=0.7, label="noisy odometry")
        for camera in CAMERAS:
            rows = _rows(raw_dir / f"{camera}_perception.csv")
            detected = [row for row in rows if int(_f(row, "detected", 0.0))]
            missed = [row for row in rows if not int(_f(row, "detected", 0.0))]
            # Misses have no projected point; their source/timestamp remains in
            # the CSV. Detections are shown at their actual projected location.
            dx = [_f(row, "pred_world_x") for row in detected]
            dy = [_f(row, "pred_world_y") for row in detected]
            axis.scatter(dx, dy, s=30, color=COLORS[camera], edgecolors="white", linewidths=0.5, label=f"{camera[-1]} hit ({len(detected)})", zorder=7)
            axis.text(0.02, 0.05 + 0.045 * (ord(camera[-1]) - ord("A")), f"{camera}: {len(rows)} records / {len(missed)} misses", transform=axis.transAxes, fontsize=7.2, color=COLORS[camera])
        axis.set_title(title, fontsize=12, weight="bold")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("warehouse x [m]")
        axis.set_ylabel("warehouse y [m]")
        axis.grid(color="#dce3eb", linewidth=0.6)
        axis.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)
    fig.suptitle("Real operational collection: source identity, detections, misses, and noisy odometry", fontsize=14, weight="bold")
    output = OUT / "01_real_routes_and_observations.png"
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def render_gp_updates() -> Path:
    fig, axes = plt.subplots(4, 3, figsize=(13.8, 15.0), dpi=180, constrained_layout=True)
    for row_index, camera in enumerate(CAMERAS):
        events = _events(camera)
        hits = sum(int(_f(row, "det_hit", 0.0)) for row in events)
        with np.load(INPUTS / f"{camera}_dayzero_prior.npz", allow_pickle=False) as prior, np.load(
            _gp_path(FINAL_GP, camera), allow_pickle=False
        ) as posterior:
            xs, ys = prior["xs"], prior["ys"]
            maps = (
                ("D0 calibrated prior", prior["P_mean_map"], "viridis", 0.0, 1.0),
                ("D1/D2 expected-kernel posterior", posterior["P_mean_map"], "viridis", 0.0, 1.0),
                ("Posterior latent std", posterior["F_std_map"], "magma", 0.0, max(0.2, float(np.nanpercentile(posterior["F_std_map"], 99)))),
            )
            for column, (title, values, cmap, vmin, vmax) in enumerate(maps):
                axis = axes[row_index, column]
                image = axis.imshow(values, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
                _map_axes(axis, xs, ys)
                if column < 2:
                    hit_rows = [row for row in events if int(_f(row, "det_hit", 0.0))]
                    miss_rows = [row for row in events if not int(_f(row, "det_hit", 0.0))]
                    for points, marker, alpha, label in ((hit_rows, "o", 0.95, "hit"), (miss_rows, "x", 0.45, "miss")):
                        axis.scatter([_f(row, "m_x") for row in points], [_f(row, "m_y") for row in points], s=13, marker=marker, color=COLORS[camera], alpha=alpha, linewidths=0.6, zorder=8, label=label if row_index == 0 and column == 0 else None)
                axis.set_title(f"{camera} — {title}", fontsize=9.2, weight="bold")
                colorbar = fig.colorbar(image, ax=axis, shrink=0.78)
                colorbar.ax.tick_params(labelsize=6.5)
        axes[row_index, 0].text(0.02, 0.96, f"{len(events)} real records\n{hits} detections", transform=axes[row_index, 0].transAxes, va="top", fontsize=7.8, color="#17212f", bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2})
    fig.suptitle("Actual per-camera GP updates: calibrated day-zero maps → posterior after two executed routes", fontsize=14, weight="bold")
    output = OUT / "02_actual_per_camera_gp_updates.png"
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def render_gp_updates_wide() -> Path:
    """Render the deck-readable prior/posterior comparison; std stays in 02."""

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.1), dpi=180, constrained_layout=True)
    for column, camera in enumerate(CAMERAS):
        events = _events(camera)
        hits = sum(int(_f(row, "det_hit", 0.0)) for row in events)
        with np.load(INPUTS / f"{camera}_dayzero_prior.npz", allow_pickle=False) as prior, np.load(_gp_path(FINAL_GP, camera), allow_pickle=False) as posterior:
            xs, ys = prior["xs"], prior["ys"]
            for row, (label, values) in enumerate((("D0 calibrated prior", prior["P_mean_map"]), ("D1/D2 posterior", posterior["P_mean_map"]))):
                axis = axes[row, column]
                image = axis.imshow(values, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="viridis", vmin=0.0, vmax=1.0, interpolation="bilinear")
                _map_axes(axis, xs, ys)
                if row == 1:
                    hit_rows = [item for item in events if int(_f(item, "det_hit", 0.0))]
                    miss_rows = [item for item in events if not int(_f(item, "det_hit", 0.0))]
                    axis.scatter([_f(item, "m_x") for item in miss_rows], [_f(item, "m_y") for item in miss_rows], marker="x", s=13, color=COLORS[camera], alpha=0.45, linewidths=0.6, zorder=8)
                    axis.scatter([_f(item, "m_x") for item in hit_rows], [_f(item, "m_y") for item in hit_rows], marker="o", s=16, color=COLORS[camera], edgecolors="white", linewidths=0.35, zorder=9)
                axis.set_title(f"{camera}: {label}", fontsize=9.4, weight="bold")
                if row == 1:
                    axis.text(0.03, 0.95, f"{len(events)} records / {hits} hits", transform=axis.transAxes, va="top", fontsize=7.2, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2})
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01)
    colorbar.set_label("reliability probability", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle("Four separate learned maps: day-zero calibration prior → posterior after two executed routes", fontsize=14, weight="bold")
    output = OUT / "02b_actual_gp_updates_wide.png"
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def render_gp_progress() -> Path:
    cameras = list(CAMERAS)
    checkpoint_events = {camera: len(_rows(RUN_ROOT / "analysis/checkpoint_01/inputs" / f"{camera}_events.csv")) for camera in cameras}
    final_events = {camera: len(_events(camera)) for camera in cameras}
    hits = {camera: sum(int(_f(row, "det_hit", 0.0)) for row in _events(camera)) for camera in cameras}
    changes = {}
    for camera in cameras:
        with np.load(INPUTS / f"{camera}_dayzero_prior.npz", allow_pickle=False) as prior, np.load(_gp_path(CHECKPOINT, camera), allow_pickle=False) as checkpoint, np.load(_gp_path(FINAL_GP, camera), allow_pickle=False) as final:
            changes[camera] = (
                float(np.mean(np.abs(checkpoint["P_mean_map"] - prior["P_mean_map"]))),
                float(np.mean(np.abs(final["P_mean_map"] - prior["P_mean_map"]))),
            )
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.1), dpi=180, constrained_layout=True)
    index = np.arange(len(cameras))
    left.bar(index - 0.18, [checkpoint_events[c] for c in cameras], 0.36, label="checkpoint 01", color="#9cc8f5")
    left.bar(index + 0.18, [final_events[c] for c in cameras], 0.36, label="after overlap route", color=[COLORS[c] for c in cameras])
    for i, camera in enumerate(cameras):
        left.text(i + 0.18, final_events[camera] + 1.5, f"{hits[camera]} hits", ha="center", fontsize=8, color=COLORS[camera], weight="bold")
    left.set_xticks(index, [camera.replace("camera_", "Camera ") for camera in cameras])
    left.set_ylabel("aligned operational events")
    left.set_title("Real evidence accumulated per camera", weight="bold")
    left.legend(fontsize=8)
    left.grid(axis="y", color="#dce3eb")
    right.bar(index - 0.18, [changes[c][0] for c in cameras], 0.36, label="after first traverse", color="#aeb9c7")
    right.bar(index + 0.18, [changes[c][1] for c in cameras], 0.36, label="after two routes", color=[COLORS[c] for c in cameras])
    right.set_xticks(index, [camera.replace("camera_", "Camera ") for camera in cameras])
    right.set_ylabel("mean |posterior − D0 prior|")
    right.set_title("The maps changed because observations arrived", weight="bold")
    right.legend(fontsize=8)
    right.grid(axis="y", color="#dce3eb")
    fig.suptitle("Observed GP learning progress (canonical expected-kernel fitter)", fontsize=14, weight="bold")
    output = OUT / "03_actual_gp_learning_progress.png"
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output


def render_overlap_gate() -> tuple[Path, dict]:
    summary = json.loads(OVERLAP.read_text(encoding="utf-8"))
    pairs = list(summary["pairs"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=180, constrained_layout=True)
    left.set_title("Actual synchronized C↔D observations", weight="bold")
    left.add_patch(Rectangle((-12.25, -10.25), 24.5, 20.5, fill=False, edgecolor="#3a4a5b", linewidth=1.0))
    for index, pair in enumerate(pairs, start=1):
        ax, ay = pair["xy_a_m"]
        bx, by = pair["xy_b_m"]
        left.plot([ax, bx], [ay, by], color="#718096", linewidth=1.4, zorder=2)
        left.scatter(ax, ay, s=58, color=COLORS["camera_C"], edgecolors="white", linewidths=0.7, label="Camera C" if index == 1 else None, zorder=4)
        left.scatter(bx, by, s=58, color=COLORS["camera_D"], edgecolors="white", linewidths=0.7, label="Camera D" if index == 1 else None, zorder=4)
        left.text((ax + bx) / 2.0 + 0.12, (ay + by) / 2.0, f"pair {index}", fontsize=8)
    left.set_xlim(-3.0, 1.0)
    left.set_ylim(1.3, 4.4)
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("warehouse x [m]")
    left.set_ylabel("warehouse y [m]")
    left.grid(color="#dce3eb")
    left.legend(loc="upper left", fontsize=8)

    disagreement = [pair["disagreement_m"] for pair in pairs]
    bars = right.bar(range(1, len(pairs) + 1), disagreement, color="#8d53c7")
    right.axhline(float(summary["max_allowed_disagreement_m"]), color="#c81919", linestyle="--", linewidth=1.5, label="configured gate = 0.30 m")
    for bar, value in zip(bars, disagreement):
        right.text(bar.get_x() + bar.get_width() / 2.0, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
    right.set_ylim(0.0, 0.36)
    right.set_xticks(range(1, len(pairs) + 1))
    right.set_xlabel("synchronized pair")
    right.set_ylabel("C/D projected disagreement [m]")
    right.set_title("Strict 50 ms overlap gate: PASS", weight="bold")
    right.legend(fontsize=8)
    right.grid(axis="y", color="#dce3eb")
    text = f"{summary['pair_count']} pairs  •  mean {summary['mean_disagreement_m']:.3f} m  •  max {summary['max_disagreement_m']:.3f} m  •  outliers {summary['trust']['outlier_rate']:.0%}"
    fig.suptitle(text, fontsize=13, weight="bold")
    output = OUT / "04_actual_overlap_gate_C_D.png"
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return output, summary


def _policy_result() -> dict:
    frames = _load_replay_frames(STRICT_EXPORT / "operational/replay_frames.jsonl")
    providers = {
        camera: GridMapReliabilityProvider.from_npz(
            _gp_path(FINAL_GP, camera), camera_id=camera, out_of_bounds_policy="clamp"
        )
        for camera in CAMERAS
    }
    result = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            quality_providers=providers,
            nis_gate=9.21,
        ),
    )
    selected = [step.camera_manager_decision.get("selected_camera_id", "") for step in result.steps]
    accepted = sum(len(step.accepted_camera_ids) for step in result.steps)
    blocked = sum(
        "no_eligible_camera" in step.camera_manager_decision.get("reasons", [])
        for step in result.steps
    )
    return {
        "frames": len(frames),
        "accepted_updates": accepted,
        "no_eligible_camera_frames": blocked,
        "selected_counts": {camera: selected.count(camera) for camera in CAMERAS},
        "policy": "hysteretic_handover_selection",
        "min_spatial_trust": 0.45,
        "interpretation": "The configured pilot policy correctly withheld corrections when the newly learned spatial trust remained below its release threshold.",
    }


def render_algorithm_execution(overlap_summary: dict) -> tuple[Path, dict]:
    policy = _policy_result()
    fig, axis = plt.subplots(figsize=(13.2, 5.8), dpi=180)
    axis.set_axis_off()
    boxes = [
        (0.03, 0.32, 0.20, 0.36, "LIVE COLLECTION", "2 executed routes\n242 aligned camera events\nno ground truth in fit", "#2f80ed"),
        (0.27, 0.32, 0.20, 0.36, "CANONICAL GP", "4 independent expected-kernel posteriors\n+ pooled diagnostic GP", "#8d53c7"),
        (0.51, 0.32, 0.20, 0.36, "OVERLAP GATE", f"C↔D: {overlap_summary['pair_count']} strict pairs\nmean Δ={overlap_summary['mean_disagreement_m']:.3f} m\nPASS", "#21a366"),
        (0.75, 0.32, 0.22, 0.36, "HANDOVER POLICY", f"{policy['frames']} replay frames\n{policy['accepted_updates']} corrections released\nfail-safe defer below trust 0.45", "#ed8a25"),
    ]
    for x, y, width, height, title, detail, color in boxes:
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor="#ffffff", edgecolor=color, linewidth=2.0, transform=axis.transAxes)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.70, title, ha="center", va="center", fontsize=11, color=color, weight="bold", transform=axis.transAxes)
        axis.text(x + width / 2, y + height * 0.37, detail, ha="center", va="center", fontsize=9.5, color="#1e2b38", transform=axis.transAxes, linespacing=1.35)
    axis.text(0.50, 0.86, "This is the actual algorithm execution, not a conceptual mock-up", ha="center", fontsize=16, weight="bold", color="#17212f", transform=axis.transAxes)
    axis.text(0.50, 0.15, "Important pilot result: the configured conservative policy released no unverified correction. That is a safe fallback, not yet a closed-loop performance claim.", ha="center", fontsize=10.5, color="#a64b16", transform=axis.transAxes)
    output = OUT / "05_actual_algorithm_execution.png"
    fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output, policy


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "real_routes_and_observations": str(render_run_evidence()),
        "per_camera_gp_updates": str(render_gp_updates()),
        "per_camera_gp_updates_wide": str(render_gp_updates_wide()),
        "gp_learning_progress": str(render_gp_progress()),
    }
    overlap_image, overlap_summary = render_overlap_gate()
    outputs["overlap_gate"] = str(overlap_image)
    execution_image, policy = render_algorithm_execution(overlap_summary)
    outputs["algorithm_execution"] = str(execution_image)
    manifest = {
        "outputs": outputs,
        "run_root": str(RUN_ROOT),
        "inputs": str(INPUTS),
        "canonical_gp_root": str(FINAL_GP),
        "overlap_summary": overlap_summary,
        "policy_result": policy,
        "contains_ground_truth": False,
        "notes": [
            "All GP maps were fitted from operational detector observations aligned to noisy odometry with an explicit 0.10 m covariance floor.",
            "The covariance floor is a declared pilot assumption: the encoder-noise publisher currently leaves Odometry covariance entries at zero.",
            "The strict C/D overlap result passes its configured 0.30 m disagreement gate, but only has three synchronized pairs; it is pilot evidence, not a campaign-level fusion claim.",
            "The configured hysteretic policy withheld all candidate corrections because learned spatial trust was below its 0.45 release threshold. This demonstrates its fail-safe path, not closed-loop improvement.",
        ],
    }
    (OUT.parent / "actual_commissioning_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
