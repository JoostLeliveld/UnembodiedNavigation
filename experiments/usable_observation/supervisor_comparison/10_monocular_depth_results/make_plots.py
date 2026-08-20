#!/usr/bin/env python3
"""Render the small supervisor-facing monocular-depth result bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIGURES = HERE / "figures"
RESULTS = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

INK = "#17324D"
SOFT = "#536471"
GRID = "#D8E2EA"
RAW = "#AAB4BD"
FLOOR = "#18A999"
ORACLE = "#76549A"
ORANGE = "#E07A5F"
BLUE = "#2474B5"

FRAME_ID = "aws_camA_s00303"


def _finish(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_outputs_before_after() -> None:
    """Show the same predictions without and with the floor-derived anchor."""
    adapter_dir = REPO / "experiments" / "monocular_depth_adapter"
    sys.path.insert(0, str(adapter_dir))

    from PIL import Image
    from evaluate_against_truth import (  # noqa: PLC0415
        TRUTH_DIR,
        _camera,
        analytic_floor_depth,
        anchor_target,
        apply_anchor,
        fit_scale_shift,
        free_floor_mask,
    )
    from monodepth import storage  # noqa: PLC0415

    rgb = np.asarray(Image.open(TRUTH_DIR / f"{FRAME_ID}_rgb.png").convert("RGB"))
    truth = np.load(TRUTH_DIR / f"{FRAME_ID}_depth.npy")
    floor_depth, hit_x, hit_y = analytic_floor_depth(_camera())
    floor_mask = free_floor_mask(hit_x, hit_y, floor_depth)

    run_dir = REPO / "logs/studies/monocular_depth_adapter/bs1_native_flip"
    predictions = []
    for row in RESULTS["models"]:
        sidecar = run_dir / row["id"] / f"{FRAME_ID}__{row['id']}.json"
        pred = storage.load_prediction(sidecar)
        anchor_mask = floor_mask & pred.valid
        scale, shift = fit_scale_shift(
            pred.depth,
            anchor_target(floor_depth, pred.convention),
            anchor_mask,
        )
        anchored = apply_anchor(pred.depth, pred.convention, scale, shift)
        predictions.append((row, pred, np.where(pred.valid, anchored, np.nan)))

    lo, hi = np.nanpercentile(truth, [2, 98])
    metric_norm = Normalize(vmin=lo, vmax=hi)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.02, 1.0, 1.0))
    ax_rgb = fig.add_subplot(grid[0, :2])
    ax_truth = fig.add_subplot(grid[0, 2:])
    raw_axes = [fig.add_subplot(grid[1, i]) for i in range(4)]
    anchored_axes = [fig.add_subplot(grid[2, i]) for i in range(4)]

    fig.suptitle(
        "The same monocular predictions before and after floor anchoring",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )

    ax_rgb.imshow(rgb)
    overlay = np.zeros((*floor_mask.shape, 4), dtype=float)
    overlay[floor_mask] = (24 / 255, 169 / 255, 153 / 255, 0.34)
    ax_rgb.imshow(overlay)
    ax_rgb.set_title(
        "Deployment-legal input: RGB + calibrated open-floor pixels",
        loc="left",
        color=INK,
        fontweight="bold",
        fontsize=11,
    )
    ax_rgb.legend(
        handles=[Patch(facecolor=FLOOR, alpha=0.45, label="pixels used to fit scale + shift")],
        loc="lower left",
        frameon=True,
        fontsize=9,
    )

    truth_image = ax_truth.imshow(truth, cmap="Blues", norm=metric_norm)
    ax_truth.set_title(
        "Evaluation-only reference: co-located Gazebo depth",
        loc="left",
        color=INK,
        fontweight="bold",
        fontsize=11,
    )

    metric_raw_axes = []
    for index, ((row, pred, anchored), raw_ax, anchored_ax) in enumerate(
        zip(predictions, raw_axes, anchored_axes)
    ):
        raw_depth = np.where(pred.valid, pred.depth, np.nan)
        if pred.convention.is_metric:
            raw_ax.imshow(raw_depth, cmap="Blues", norm=metric_norm)
            metric_raw_axes.append(raw_ax)
            raw_result = f"raw study MAE {row['raw_mae_m']:.3f} m"
        else:
            raw_lo, raw_hi = np.nanpercentile(raw_depth, [2, 98])
            relative_image = raw_ax.imshow(
                raw_depth,
                cmap="Blues_r",
                vmin=raw_lo,
                vmax=raw_hi,
            )
            relative_bar = fig.colorbar(relative_image, ax=raw_ax, shrink=0.68, pad=0.015)
            relative_bar.set_label("unitless inverse depth", fontsize=7.5, color=SOFT)
            relative_bar.ax.tick_params(labelsize=7, colors=SOFT)
            raw_result = "unitless raw output"

        anchored_ax.imshow(anchored, cmap="Blues", norm=metric_norm)
        raw_ax.set_title(f"{row['label']}\n{raw_result}", fontsize=9.5, color=INK)
        anchored_ax.set_title(
            f"{row['label']}\nfloor-anchored study MAE {row['floor_mae_m']:.3f} m",
            fontsize=9.5,
            color=INK,
        )

        if index == 0:
            raw_ax.set_ylabel(
                "WITHOUT FLOOR ANCHORING\nmetric outputs use the truth scale",
                color=INK,
                fontweight="bold",
                fontsize=10,
                labelpad=12,
            )
            anchored_ax.set_ylabel(
                "WITH FLOOR ANCHORING\nall outputs are metric and share one scale",
                color=INK,
                fontweight="bold",
                fontsize=10,
                labelpad=12,
            )

    metric_bar = fig.colorbar(
        truth_image,
        ax=[ax_truth, *metric_raw_axes, *anchored_axes],
        shrink=0.82,
        pad=0.01,
    )
    metric_bar.set_label("optical-axis depth [m]", fontsize=9, color=SOFT)
    metric_bar.ax.tick_params(labelsize=8, colors=SOFT)

    for ax in [ax_rgb, ax_truth, *raw_axes, *anchored_axes]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

    fig.text(
        0.5,
        0.006,
        "Anchoring changes scale, not scene shape. Gazebo depth is used only for evaluation; "
        "the fitted correction uses camera calibration and the highlighted floor pixels.",
        ha="center",
        fontsize=10,
        color=SOFT,
    )
    _finish(fig, "01_model_outputs.png")


def anchoring_accuracy() -> None:
    rows = RESULTS["models"]
    labels = [row["label"] for row in rows]
    raw = np.array([np.nan if row["raw_mae_m"] is None else row["raw_mae_m"] for row in rows])
    floor = np.array([row["floor_mae_m"] for row in rows])
    oracle = np.array([row["oracle_mae_m"] for row in rows])
    y = np.arange(len(rows))

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.2, 5.1), constrained_layout=True)
    fig.suptitle(
        "Floor anchoring recovers metric scale without a depth sensor or CAD\n"
        "MAE on 451,005 non-floor pixels · 12 frames · Camera A · one Gazebo world · lower is better",
        fontsize=15,
        fontweight="bold",
        color=INK,
        linespacing=1.35,
    )

    height = 0.22
    ax_full.barh(y - height, raw, height, color=RAW, label="Raw model output")
    ax_full.barh(y, floor, height, color=FLOOR, label="Floor anchored")
    ax_full.barh(y + height, oracle, height, color=ORACLE, label="Oracle affine")
    ax_full.set_xscale("log")
    ax_full.set_xlim(0.18, 6.4)
    ax_full.set_yticks(y, labels)
    ax_full.invert_yaxis()
    ax_full.set_xlabel("Mean absolute depth error [m], logarithmic scale")
    ax_full.set_title("Raw scale failure and recovery", loc="left", color=INK, fontweight="bold")
    ax_full.grid(axis="x", color=GRID, linewidth=0.8)
    ax_full.set_axisbelow(True)
    for values, offset in ((raw, -height), (floor, 0), (oracle, height)):
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax_full.text(value * 1.035, i + offset, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    ax_full.text(0.19, 1 - height, "unitless\n(no raw MAE)", va="center", fontsize=8, color=SOFT)
    ax_full.legend(frameon=False, loc="lower right", fontsize=9)

    ax_zoom.barh(y - 0.12, floor, 0.24, color=FLOOR, label="Floor anchored")
    ax_zoom.barh(y + 0.12, oracle, 0.24, color=ORACLE, label="Oracle affine")
    ax_zoom.set_xlim(0, 0.47)
    ax_zoom.set_yticks(y, labels)
    ax_zoom.invert_yaxis()
    ax_zoom.set_xlabel("Mean absolute depth error [m]")
    ax_zoom.set_title("Deployment-legal anchor versus oracle fit", loc="left", color=INK, fontweight="bold")
    ax_zoom.grid(axis="x", color=GRID, linewidth=0.8)
    ax_zoom.set_axisbelow(True)
    for i, (f_value, o_value) in enumerate(zip(floor, oracle)):
        ax_zoom.text(f_value + 0.008, i - 0.12, f"{f_value:.3f}", va="center", fontsize=8.5, color=INK)
        ax_zoom.text(o_value + 0.008, i + 0.12, f"{o_value:.3f}", va="center", fontsize=8.5, color=INK)
        ax_zoom.text(
            0.46,
            i,
            f"gap {(f_value - o_value) * 100:.1f} cm",
            ha="right",
            va="center",
            fontsize=8.5,
            color=SOFT,
        )
    ax_zoom.legend(frameon=False, loc="lower right", fontsize=9)

    for ax in (ax_full, ax_zoom):
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, colors=INK)
        ax.tick_params(axis="x", colors=SOFT)

    _finish(fig, "02_floor_anchoring_accuracy.png")


def compute_cost() -> None:
    rows = RESULTS["models"]
    labels = [row["label"] for row in rows]
    forward = np.array([row["forward_s"] for row in rows])
    uncertainty = np.array([row["uncertainty_pass_s"] for row in rows])
    memory = np.array([row["peak_gpu_mib"] for row in rows])
    y = np.arange(len(rows))

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    fig.suptitle(
        "Commissioning-time depth is practical; live uncertainty has a cost\n"
        "1280×720 · batch size 1 · Quadro P2000 (4032 MiB) · median time per frame",
        fontsize=15,
        fontweight="bold",
        color=INK,
        linespacing=1.35,
    )

    ax_time.barh(y, forward, color=BLUE, height=0.55, label="Depth pass")
    ax_time.barh(y, uncertainty, left=forward, color=ORANGE, height=0.55, label="Flip uncertainty pass")
    ax_time.set_yticks(y, labels)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Seconds per frame")
    ax_time.set_title("Inference time", loc="left", color=INK, fontweight="bold")
    ax_time.grid(axis="x", color=GRID, linewidth=0.8)
    ax_time.set_axisbelow(True)
    for i, total in enumerate(forward + uncertainty):
        ax_time.text(total + 0.15, i, f"{total:.2f} s", va="center", fontsize=9, color=INK)
    ax_time.set_xlim(0, 13.6)
    ax_time.legend(frameon=False, loc="lower right", fontsize=9)

    ax_mem.barh(y, memory, color=BLUE, height=0.55)
    ax_mem.axvline(4032, color=SOFT, linestyle="--", linewidth=1.8, label="GPU capacity")
    ax_mem.set_yticks(y, labels)
    ax_mem.invert_yaxis()
    ax_mem.set_xlabel("Peak allocated GPU memory [MiB]")
    ax_mem.set_title("GPU memory", loc="left", color=INK, fontweight="bold")
    ax_mem.grid(axis="x", color=GRID, linewidth=0.8)
    ax_mem.set_axisbelow(True)
    ax_mem.set_xlim(0, 4400)
    for i, value in enumerate(memory):
        ax_mem.text(value + 60, i, f"{value:.0f}", va="center", fontsize=9, color=INK)
    ax_mem.legend(frameon=False, loc="lower right", fontsize=9)

    for ax in (ax_time, ax_mem):
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, colors=INK)
        ax.tick_params(axis="x", colors=SOFT)

    _finish(fig, "03_compute_cost.png")


def main() -> None:
    model_outputs_before_after()
    anchoring_accuracy()
    compute_cost()


if __name__ == "__main__":
    main()
