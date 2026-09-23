#!/usr/bin/env python3
"""Follow one intact run and test the camera model against the outcome.

The campaign figure shows that every model was run on every task and seed.
This figure takes one intact run and walks the deployed chain along it: the
raw projection, the correction that removes its offset, the covariance the
spatial model predicted at each position it was queried, and the fused belief
that results.  The positions driven here were available to the covariance fit,
so this shows the deployed model calibrated in use, not an out-of-sample test.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from style import CAM_COLOUR, draw_warehouse, layout  # noqa: E402

PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
CAMPAIGN = ROOT / (
    "logs/track_a_draft/stage09_final_five_seed_campaign/"
    "thesis09_parallel_aisles_west"
)
PRECISION = ROOT / (
    "logs/track_a_draft/planning_precision/m2_planning_precision.npz"
)
RAW = "#9b9a94"
CORRECTED = "#1d2530"
BELIEF = "#0072B2"


def run_directory(condition: str, seed: str) -> Path:
    matches = sorted(glob.glob(str(CAMPAIGN / condition / seed / "attempts/*/experiment_*/")))
    if len(matches) != 1:
        raise ValueError(f"{condition}/{seed}: expected one run, found {len(matches)}")
    return Path(matches[0])


def load(condition: str, seed: str) -> dict:
    directory = run_directory(condition, seed)
    experiment = pd.read_csv(directory / "experiment.csv")
    step = np.hypot(experiment["state_x"].diff().fillna(0.0),
                    experiment["state_y"].diff().fillna(0.0))
    moving = step.cumsum() > 0.05
    if not moving.any():
        raise ValueError(f"{directory}: the robot never moved")
    # t = 0 is the first commanded motion, not the first logged sample.
    origin = float(experiment["stamp"][moving].iloc[0])
    experiment["t"] = experiment["stamp"] - origin
    experiment = experiment[experiment["t"] >= -1.0].copy()

    fusion = pd.read_csv(directory / "fusion_observations.csv")
    fusion = fusion[fusion["used"].astype(bool)].copy()
    fusion["t"] = fusion["stamp"] - origin
    fusion = fusion[fusion["t"] >= -1.0].copy()
    # Each admitted observation against the truth at its own capture stamp.
    fusion["raw_error"] = np.hypot(fusion["raw_obs_x"] - fusion["gt_x_at_obs"],
                                   fusion["raw_obs_y"] - fusion["gt_y_at_obs"])
    fusion["corrected_error"] = np.hypot(fusion["obs_x"] - fusion["gt_x_at_obs"],
                                         fusion["obs_y"] - fusion["gt_y_at_obs"])
    # The spatial model's prediction for that camera at that position, as one
    # equivalent circular standard deviation.
    fusion["predicted_sd"] = np.sqrt(
        0.5 * (fusion["obs_cov_xx"] + fusion["obs_cov_yy"]))

    batches = fusion.drop_duplicates("source_batch_id").copy()
    batches["fused_error"] = np.hypot(batches["fused_x"] - batches["gt_x_at_fused"],
                                      batches["fused_y"] - batches["gt_y_at_fused"])
    batches["fused_sd"] = np.sqrt(
        0.5 * (batches["fused_cov_xx"] + batches["fused_cov_yy"]))

    waypoints = pd.read_csv(directory / "global_waypoints.csv")
    return {"experiment": experiment, "fusion": fusion, "batches": batches,
            "waypoints": waypoints}


def precision_field() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(PRECISION, allow_pickle=False) as archive:
        xs = np.asarray(archive["xs"], dtype=float)
        ys = np.asarray(archive["ys"], dtype=float)
        precision = np.asarray(archive["matched_precision_m2_inv"], dtype=float)
    return xs, ys, 0.5 * np.trace(precision.sum(axis=0), axis1=-2, axis2=-1)


def draw_map(ax: plt.Axes, run: dict, field: tuple) -> None:
    xs, ys, values = field
    ax.pcolormesh(xs, ys, values, shading="nearest", cmap="viridis",
                  norm=PowerNorm(gamma=0.42, vmin=0.0,
                                 vmax=float(np.percentile(values[values > 0], 99))),
                  rasterized=True, zorder=0)
    draw_warehouse(ax, layout(), show_cameras=True, camera_labels=True, rack_alpha=0.82)
    waypoints = run["waypoints"]
    ax.plot(waypoints["x"], waypoints["y"], color="white", lw=2.3, ls="--",
            zorder=14, dash_capstyle="round")
    experiment = run["experiment"]
    ax.plot(experiment["state_x"], experiment["state_y"], color=BELIEF, lw=1.9,
            zorder=16, solid_capstyle="round")
    ax.scatter(experiment["state_x"].iloc[0], experiment["state_y"].iloc[0],
               s=26, color="#111111", zorder=20)
    ax.scatter(waypoints["x"].iloc[-1], waypoints["y"].iloc[-1], s=62,
               color="#111111", marker="*", zorder=20)
    ax.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal")
    ax.set_title("(a) Executed route over the spatial precision field",
                 fontsize=8.0, fontweight="bold")
    ax.tick_params(labelsize=6.2)
    ax.set_xlabel("east (m)", fontsize=7.0)
    ax.set_ylabel("north (m)", fontsize=7.0)


def draw_camera_map(ax: plt.Axes, run: dict) -> None:
    """Where each camera was admitted, so the time panels can be read spatially."""
    draw_warehouse(ax, layout(), show_cameras=True, camera_labels=True, rack_alpha=0.82)
    for camera, group in run["fusion"].groupby("camera"):
        ax.scatter(group["gt_x_at_obs"], group["gt_y_at_obs"], s=3.0,
                   color=CAM_COLOUR[str(camera)], linewidths=0, alpha=0.75,
                   zorder=15, label=f"camera {camera}")
    ax.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal")
    ax.set_title("(b) Admitted observations by camera", fontsize=8.0,
                 fontweight="bold")
    ax.tick_params(labelsize=6.2)
    ax.set_xlabel("east (m)", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, handletextpad=0.15,
              columnspacing=0.6, markerscale=2.4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="seed91500")
    parser.add_argument("--condition", default="spatial_intact")
    arguments = parser.parse_args()

    run = load(arguments.condition, arguments.seed)
    fusion, batches = run["fusion"], run["batches"]

    fig = plt.figure(figsize=(7.16, 8.05), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=(2.30, 1.10, 1.34, 1.10))

    draw_map(fig.add_subplot(grid[0, 0]), run, precision_field())
    draw_camera_map(fig.add_subplot(grid[0, 1]), run)

    # (c) the correction, along the drive.
    ax_correction = fig.add_subplot(grid[1, :])
    ax_correction.scatter(fusion["t"], 100.0 * fusion["raw_error"], s=3.0,
                          color=RAW, linewidths=0, label="raw projection")
    ax_correction.scatter(fusion["t"], 100.0 * fusion["corrected_error"], s=3.0,
                          color=CORRECTED, linewidths=0, label="after correction")
    ax_correction.set_yscale("log")
    ax_correction.set_ylabel("observation error (cm)")
    ax_correction.set_title("(c) Observation error along the drive, before and "
                            "after correction", fontsize=8.0)
    ax_correction.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower left",
                         markerscale=2.2)

    # (d) the predicted covariance against the realised error, per camera.
    ax_predicted = fig.add_subplot(grid[2, :], sharex=ax_correction)
    for camera, group in fusion.groupby("camera"):
        colour = CAM_COLOUR[str(camera)]
        group = group.sort_values("t")
        ax_predicted.fill_between(group["t"], 0.0, 100.0 * group["predicted_sd"],
                                  color=colour, alpha=0.18, linewidth=0, zorder=2)
        ax_predicted.plot(group["t"], 100.0 * group["predicted_sd"], color=colour,
                          lw=1.0, zorder=4)
        ax_predicted.scatter(group["t"], 100.0 * group["corrected_error"], s=3.2,
                             color=colour, linewidths=0, alpha=0.85, zorder=6)
    ax_predicted.set_ylabel("error and predicted SD (cm)")
    ax_predicted.set_ylim(0.0, 12.0)
    ax_predicted.set_title("(d) Predicted observation SD (line) against the realised "
                           "error (points), per camera", fontsize=8.0)
    handles = [Line2D([0], [0], color=CAM_COLOUR[str(c)], lw=1.4, label=f"camera {c}")
               for c in sorted(fusion["camera"].astype(str).unique())]
    ax_predicted.legend(handles=handles, frameon=False, fontsize=6.3, ncol=4,
                        loc="upper right")

    # (e) the fused belief against its own predicted spread.
    ax_belief = fig.add_subplot(grid[3, :], sharex=ax_correction)
    ax_belief.fill_between(batches["t"], 0.0, 200.0 * batches["fused_sd"],
                           color=BELIEF, alpha=0.20, linewidth=0,
                           label=r"predicted $2\sigma$ of the fused estimate")
    ax_belief.plot(batches["t"], 100.0 * batches["fused_error"], color=CORRECTED,
                   lw=0.95, label="realised fused error")
    ax_belief.set_yscale("log")
    ax_belief.set(xlabel="time after first command (s)")
    ax_belief.set_ylabel("fused error (cm)")
    ax_belief.set_title("(e) Fused estimate against its predicted spread", fontsize=8.0)
    ax_belief.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower left")

    for ax in (ax_correction, ax_predicted, ax_belief):
        ax.grid(axis="y", color="#e2e2e2", lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.6)
        ax.xaxis.label.set_fontsize(7.0)
        ax.yaxis.label.set_fontsize(7.0)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = PAPER_FIGURES / f"navigation_single_run.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
