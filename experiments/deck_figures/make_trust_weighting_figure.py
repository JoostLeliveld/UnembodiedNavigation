#!/usr/bin/env python3
"""Show the spatial model distrusting a camera before that camera is wrong.

One run, four admitted cameras.  Camera D is predicted to be the least reliable
along this stretch, and when its observations do go wrong the fusion weight has
already moved off it.  The counterfactual is equal weighting of the same
admitted observations, so the only difference is the covariance.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/deck_figures"))
from style import CAM_COLOUR, draw_warehouse, layout  # noqa: E402

PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
CAMPAIGN = ROOT / "logs/thesis_final_pipeline_v1/stage09_final_five_seed_campaign"
EQUAL = "#9b9a94"
TRUST = "#0072B2"
HIGHLIGHT = "#B2182B"


def run_directory(task: str, condition: str, seed: str) -> Path:
    matches = sorted(glob.glob(str(CAMPAIGN / task / condition / seed / "attempts/*/experiment_*/")))
    if len(matches) != 1:
        raise ValueError(f"{task}/{condition}/{seed}: found {len(matches)} runs")
    return Path(matches[0])


def load(task: str, condition: str, seed: str) -> dict:
    directory = run_directory(task, condition, seed)
    experiment = pd.read_csv(directory / "experiment.csv")
    step = np.hypot(experiment["state_x"].diff().fillna(0.0),
                    experiment["state_y"].diff().fillna(0.0))
    moving = step.cumsum() > 0.05
    if not moving.any():
        raise ValueError(f"{directory}: the robot never moved")
    origin = float(experiment["stamp"][moving].iloc[0])
    experiment["t"] = experiment["stamp"] - origin
    experiment = experiment[experiment["t"] >= -1.0].copy()

    fusion = pd.read_csv(directory / "fusion_observations.csv")
    fusion = fusion[fusion["used"].astype(bool)].copy()
    fusion["t"] = fusion["stamp"] - origin
    fusion = fusion[fusion["t"] >= -1.0].copy()
    fusion["error_cm"] = 100.0 * np.hypot(fusion["obs_x"] - fusion["gt_x_at_obs"],
                                          fusion["obs_y"] - fusion["gt_y_at_obs"])
    fusion["predicted_sd_cm"] = 100.0 * np.sqrt(
        0.5 * (fusion["obs_cov_xx"] + fusion["obs_cov_yy"]))
    # The information-form weight fusion actually applied.
    fusion["weight"] = 1.0 / (0.5 * (fusion["obs_cov_xx"] + fusion["obs_cov_yy"]))

    # Counterfactual: the same admitted observations, weighted equally.
    records = []
    for _, batch in fusion.groupby("source_batch_id"):
        if len(batch) < 2:
            continue
        truth = np.asarray([batch["gt_x_at_obs"].iloc[0], batch["gt_y_at_obs"].iloc[0]])
        equal = np.asarray([batch["obs_x"].mean(), batch["obs_y"].mean()])
        weight = batch["weight"].to_numpy()
        trust = np.asarray([(batch["obs_x"] * weight).sum() / weight.sum(),
                            (batch["obs_y"] * weight).sum() / weight.sum()])
        records.append({"t": batch["t"].mean(),
                        "equal_cm": 100.0 * float(np.linalg.norm(equal - truth)),
                        "trust_cm": 100.0 * float(np.linalg.norm(trust - truth)),
                        "cameras": "".join(sorted(batch["camera"].astype(str)))})
    batches = pd.DataFrame(records)
    members = fusion.groupby("source_batch_id")["weight"].transform("size")
    single_from = float(batches["t"].max()) if not batches.empty else float("nan")
    return {"experiment": experiment, "fusion": fusion, "batches": batches,
            "single_from": single_from,
            "single_cameras": sorted(fusion.loc[(members < 2) & (fusion["t"] > single_from),
                                                "camera"].astype(str).unique())}


def rms(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="thesis09_parallel_aisles_west")
    parser.add_argument("--condition", default="spatial_intact")
    parser.add_argument("--seed", default="seed91501")
    parser.add_argument("--camera", default="A")
    arguments = parser.parse_args()

    run = load(arguments.task, arguments.condition, arguments.seed)
    fusion, batches = run["fusion"], run["batches"]
    distrusted = arguments.camera

    fig = plt.figure(figsize=(7.16, 6.30), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(2.05, 1.25, 1.25),
                            width_ratios=(1.0, 1.0))

    # (a) where the run went, and where the distrusted camera was admitted.
    ax_map = fig.add_subplot(grid[0, 0])
    draw_warehouse(ax_map, layout(), show_cameras=True, camera_labels=True,
                   rack_alpha=0.82)
    experiment = run["experiment"]
    ax_map.plot(experiment["state_x"], experiment["state_y"], color="#333333",
                lw=1.6, zorder=14)
    for camera, group in fusion.groupby("camera"):
        ax_map.scatter(group["gt_x_at_obs"], group["gt_y_at_obs"], s=2.6,
                       color=CAM_COLOUR[str(camera)], linewidths=0, alpha=0.75,
                       zorder=15)
    worst = fusion[(fusion["camera"] == distrusted) & (fusion["error_cm"] > 20.0)]
    ax_map.scatter(worst["gt_x_at_obs"], worst["gt_y_at_obs"], s=44,
                   facecolor="none", edgecolor=HIGHLIGHT, lw=1.5, zorder=22,
                   label=f"camera {distrusted} error $>$20~cm")
    ax_map.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal")
    ax_map.set_title("(a) Run and admitted observations", fontsize=8.0,
                     fontweight="bold")
    ax_map.tick_params(labelsize=6.2)
    ax_map.set_xlabel("east (m)", fontsize=7.0)
    ax_map.set_ylabel("north (m)", fontsize=7.0)
    ax_map.legend(frameon=False, fontsize=5.9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.07), markerscale=1.0)

    # (b) the fusion weight the model assigned to each camera.
    ax_weight = fig.add_subplot(grid[0, 1])
    share = fusion.copy()
    members = share.groupby("source_batch_id")["weight"].transform("size")
    share = share[members >= 2].copy()
    total = share.groupby("source_batch_id")["weight"].transform("sum")
    share["share"] = 100.0 * share["weight"] / total
    for camera, group in share.groupby("camera"):
        group = group.sort_values("t")
        ax_weight.scatter(group["t"], group["share"], s=2.6,
                          color=CAM_COLOUR[str(camera)], linewidths=0,
                          alpha=0.85, label=f"camera {camera}")
    ax_weight.set(xlabel="time after first command (s)",
                  ylabel="share of fusion weight (%)", ylim=(0, 100))
    ax_weight.set_title("(b) Weight share in multi-camera batches", fontsize=8.0)
    if np.isfinite(run["single_from"]):
        ax_weight.text(run["single_from"], 96.0, "  only one camera reports",
                       fontsize=5.8, color="#666666", va="top")
    if np.isfinite(run["single_from"]):
        ax_weight.axvline(run["single_from"], color="#999999", lw=0.8, ls=":")
    ax_weight.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper right",
                     markerscale=2.6)

    # (c) the distrusted camera: predicted SD against its realised error.
    ax_camera = fig.add_subplot(grid[1, :])
    group = fusion[fusion["camera"] == distrusted].sort_values("t")
    ax_camera.fill_between(group["t"], 0.0, group["predicted_sd_cm"],
                           color=CAM_COLOUR[distrusted], alpha=0.22, linewidth=0,
                           label=f"camera {distrusted}: predicted SD")
    ax_camera.plot(group["t"], group["predicted_sd_cm"],
                   color=CAM_COLOUR[distrusted], lw=1.0)
    ax_camera.scatter(group["t"], group["error_cm"], s=4.0, color="#1d2530",
                      linewidths=0, label=f"camera {distrusted}: realised error")
    ax_camera.scatter(worst["t"], worst["error_cm"], s=40, facecolor="none",
                      edgecolor=HIGHLIGHT, lw=1.4, zorder=8)
    ax_camera.set_ylabel("error / predicted SD (cm)")
    visible = float(np.percentile(group["error_cm"], 97))
    ax_camera.set_ylim(0.0, max(visible * 1.25, 14.0))
    for _, point in worst.iterrows():
        ax_camera.annotate(f"{point['error_cm']:.0f}~cm",
                           xy=(point["t"], ax_camera.get_ylim()[1]),
                           xytext=(point["t"], ax_camera.get_ylim()[1] * 0.88),
                           ha="center", fontsize=6.0, color=HIGHLIGHT,
                           arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=0.9))
    ax_camera.set_title(f"(c) Camera {distrusted} is predicted to be the least "
                        "reliable here, and then is", fontsize=8.0)
    ax_camera.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left")

    # (d) the consequence: same observations, two weightings.
    ax_fused = fig.add_subplot(grid[2, :], sharex=ax_camera)
    ordered = batches.sort_values("t")
    ax_fused.plot(ordered["t"], ordered["equal_cm"], color=EQUAL, lw=1.0,
                  label=f"equal weighting (RMS {rms(ordered['equal_cm']):.1f}~cm)")
    ax_fused.plot(ordered["t"], ordered["trust_cm"], color=TRUST, lw=1.1,
                  label=f"model weighting (RMS {rms(ordered['trust_cm']):.1f}~cm)")
    ax_fused.set(xlabel="time after first command (s)")
    ax_fused.set_ylim(0.0, float(ordered[["equal_cm", "trust_cm"]].max().max()) * 1.08)
    ax_fused.set_ylabel("fused error (cm)")
    ax_fused.set_title("(d) Fused error from the same observations under both "
                       "weightings", fontsize=8.0)
    ax_fused.fill_between(ordered["t"], ordered["trust_cm"], ordered["equal_cm"],
                          where=ordered["equal_cm"] >= ordered["trust_cm"],
                          color=TRUST, alpha=0.16, linewidth=0, interpolate=True,
                          label="model weighting better")
    ax_fused.fill_between(ordered["t"], ordered["trust_cm"], ordered["equal_cm"],
                          where=ordered["equal_cm"] < ordered["trust_cm"],
                          color=HIGHLIGHT, alpha=0.16, linewidth=0, interpolate=True,
                          label="equal weighting better")
    if np.isfinite(run["single_from"]):
        ax_fused.axvline(run["single_from"], color="#999999", lw=0.8, ls=":")
    ax_fused.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper left")

    for ax in (ax_weight, ax_camera, ax_fused):
        ax.grid(axis="y", color="#e2e2e2", lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.6)
        ax.xaxis.label.set_fontsize(7.0)
        ax.yaxis.label.set_fontsize(7.0)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = PAPER_FIGURES / f"fusion_trust_weighting.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
