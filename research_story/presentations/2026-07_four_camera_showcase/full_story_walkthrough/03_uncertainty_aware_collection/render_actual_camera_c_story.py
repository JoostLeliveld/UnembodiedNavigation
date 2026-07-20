#!/usr/bin/env python3
"""Render the one-camera walkthrough from the multi-aisle Camera C campaign.

Every plotted trajectory, detector outcome, covariance ellipse, and GP map is
read from ``single_camera_c_multi_aisle_20260716``.  The script is deliberately
limited to this one-camera story; it creates no conceptual stand-ins.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUN_ROOT = REPO / "logs/studies/multicamera_commissioning_bigwarehouse/single_camera_c_multi_aisle_20260716"
EVENTS = RUN_ROOT / "analysis/inputs/camera_C_events.csv"
PRIOR = RUN_ROOT / "analysis/inputs/camera_C_dayzero_prior.npz"
POSTERIOR = (
    REPO
    / "logs/visibility_comparison/single_camera_c_multi_aisle_20260716"
    / "camera_C_expected_kernel_beta11/det_hit_expected_kernel_gp.npz"
)
FIGURES = HERE / "figures"

INK = "#17212f"
MUTED = "#637083"
GRID = "#dce5ef"
HIT = "#7951c6"
MISS = "#d24848"
ROUTE_COLORS = {"01_west": "#2f80ed", "02_middle": "#18a470", "03_east": "#e28632"}
FLOOR = "#f7f9fc"
RACK = "#b8c2cb"

# Truthful lower-east floor context from the fixed warehouse layout. Camera C
# is mounted on the south wall; E1--E3 define the three distinct driven aisles.
CAMERA_C_XY = (6.0, -10.0)
RACK_E1_SOUTH = (2.25, -6.50, 0.55, 3.90)
RACK_E2_SOUTH = (4.35, -6.50, 0.55, 3.90)
RACK_E3_SOUTH = (6.45, -6.50, 0.55, 3.90)
RACKS = (("rack E1", RACK_E1_SOUTH), ("rack E2", RACK_E2_SOUTH), ("rack E3", RACK_E3_SOUTH))
VIEW = (0.30, 8.05, -10.30, -2.20)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def campaign_data():
    events = read_csv(EVENTS)
    routes: dict[str, np.ndarray] = {}
    for run_id in ROUTE_COLORS:
        rows = read_csv(RUN_ROOT / run_id / "raw/experiment.csv")
        xy = np.asarray(
            [[f(row, "odom_noisy_x"), f(row, "odom_noisy_y")] for row in rows], dtype=float
        )
        # Retain the actual driven segment, with a mild downsample only to
        # make the three trajectories readable.
        routes[run_id] = xy[::12]
    return events, routes


def setup_axis(axis, *, title: str) -> None:
    axis.set_facecolor(FLOOR)
    for name, (x, y, width, height) in RACKS:
        axis.add_patch(Rectangle((x, y), width, height, facecolor=RACK, edgecolor="#718090", linewidth=1.0, zorder=4))
        axis.text(x + width / 2.0, y + height / 2.0, name, color="white", fontsize=7.7, rotation=90, ha="center", va="center", zorder=5)
    axis.scatter(*CAMERA_C_XY, marker="^", s=78, color=INK, zorder=8)
    axis.text(CAMERA_C_XY[0], -10.16, "Camera C", color=INK, fontsize=8.0, weight="bold", ha="center", va="top")
    axis.set(xlim=VIEW[:2], ylim=VIEW[2:], title=title, xlabel="warehouse x [m]", ylabel="warehouse y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color=GRID, linewidth=0.8, zorder=0)


def plot_routes(axis, routes: dict[str, np.ndarray], *, labels: bool = True) -> None:
    for run_id, xy in routes.items():
        label = run_id.replace("_", " ") if labels else None
        axis.plot(xy[:, 0], xy[:, 1], color=ROUTE_COLORS[run_id], linewidth=1.55, alpha=0.88, label=label, zorder=6)
        axis.scatter(xy[0, 0], xy[0, 1], s=17, color=ROUTE_COLORS[run_id], edgecolor="white", linewidth=0.5, zorder=7)


def plot_events(axis, events: list[dict[str, str]], *, small: bool = False) -> None:
    hits = [row for row in events if int(f(row, "det_hit")) == 1]
    misses = [row for row in events if int(f(row, "det_hit")) == 0]
    size = 22 if small else 32
    axis.scatter([f(r, "m_x") for r in hits], [f(r, "m_y") for r in hits], s=size, color=HIT, edgecolor="white", linewidth=0.45, label="Camera C detection", zorder=11)
    axis.scatter([f(r, "m_x") for r in misses], [f(r, "m_y") for r in misses], s=size + 8, marker="x", color=MISS, linewidth=1.15, label="Camera C miss", zorder=12)


def add_map(axis, values: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, cmap: str, vmin: float, vmax: float):
    return axis.imshow(values, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear", alpha=0.93, zorder=2)


def route_and_observations(events: list[dict[str, str]], routes: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(10.3, 7.4), dpi=180, constrained_layout=True)
    setup_axis(axis, title="Actual Camera C collection across three warehouse aisles (10.2 m total)")
    plot_routes(axis, routes)
    plot_events(axis, events)
    axis.legend(loc="upper left", fontsize=8.5, frameon=True)
    hits = sum(int(f(row, "det_hit")) for row in events)
    axis.text(0.025, 0.035, f"{len(events)} logged GP samples\n{hits} detections · {len(events) - hits} misses", transform=axis.transAxes, bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="#c5d0dc"), color=INK, fontsize=9.4, weight="bold")
    figure.savefig(FIGURES / "01_camera_c_actual_route_and_observations.png", facecolor="white")
    plt.close(figure)


def pose_uncertainty(events: list[dict[str, str]], routes: dict[str, np.ndarray]) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.2, 6.1), dpi=180, constrained_layout=True, gridspec_kw={"width_ratios": [1.02, 1.0]})
    setup_axis(left, title="Actual 1σ position uncertainty attached to Camera C samples")
    plot_routes(left, routes, labels=False)
    for index, row in enumerate(events):
        if index % 4:
            continue
        cov = np.asarray([[f(row, "S_xx"), f(row, "S_xy")], [f(row, "S_xy"), f(row, "S_yy")]], dtype=float)
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 0.0)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ellipse = Ellipse((f(row, "m_x"), f(row, "m_y")), 2.0 * np.sqrt(values[1]), 2.0 * np.sqrt(values[0]), angle=angle, fill=False, edgecolor="#2f80ed", linewidth=1.15, alpha=0.78, zorder=10)
        left.add_patch(ellipse)
    plot_events(left, events, small=True)
    left.text(0.55, -2.46, "one ellipse per fourth\nactual GP sample", fontsize=8.2, color=MUTED, va="top")

    run_order = {run: index for index, run in enumerate(ROUTE_COLORS)}
    progress = np.asarray([run_order[row["run_id"]] * 3.4 + (-f(row, "m_y") - 4.0) for row in events])
    sig_x = np.sqrt(np.asarray([f(row, "S_xx") for row in events]))
    sig_y = np.sqrt(np.asarray([f(row, "S_yy") for row in events]))
    for run_id, color in ROUTE_COLORS.items():
        select = np.asarray([row["run_id"] == run_id for row in events])
        right.plot(progress[select], sig_x[select], "o-", color=color, linewidth=1.2, markersize=3.7, label=f"{run_id.replace('_', ' ')} σx")
        right.plot(progress[select], sig_y[select], "--", color=color, linewidth=1.0, alpha=0.85, label=f"{run_id.replace('_', ' ')} σy")
    right.set(title="Recorded covariance supplied to the GP fitter", xlabel="route-bundle progress [m]", ylabel="1σ positional uncertainty [m]", ylim=(0.0, max(float(sig_x.max()), float(sig_y.max())) * 1.18))
    right.grid(color=GRID, linewidth=0.8)
    handles, labels = right.get_legend_handles_labels()
    right.legend(handles[:2], ["σx (each pass)", "σy (each pass)"], loc="upper left", fontsize=8.5)
    median_sigma = float(np.median(np.concatenate([sig_x, sig_y])))
    right.text(0.985, 0.04, f"median 1σ = {median_sigma:.3f} m\nall {len(events)} covariances propagated", transform=right.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="#c5d0dc"))
    figure.savefig(FIGURES / "02_camera_c_recorded_pose_uncertainty.png", facecolor="white")
    plt.close(figure)


def gp_before_after(events: list[dict[str, str]], routes: dict[str, np.ndarray]) -> None:
    with np.load(PRIOR, allow_pickle=False) as prior, np.load(POSTERIOR, allow_pickle=False) as posterior:
        xs = np.asarray(prior["xs"], dtype=float)
        ys = np.asarray(prior["ys"], dtype=float)
        before = np.asarray(prior["P_mean_map"], dtype=float)
        after = np.asarray(posterior["P_mean_map"], dtype=float)
        std = np.asarray(posterior["F_std_map"], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.85), dpi=180, constrained_layout=True)
    panels = [
        ("Before collection: Camera C day-zero reliability prior", before, "viridis", 0.0, 1.0, "reliability probability"),
        (f"After {len(events)} observations: smoothed GP posterior", after, "viridis", 0.0, 1.0, "reliability probability"),
        ("After collection: GP latent uncertainty", std, "magma", 0.0, max(0.12, float(np.nanpercentile(std, 99))), "latent GP σ"),
    ]
    for axis, (title, values, cmap, vmin, vmax, label) in zip(axes, panels):
        setup_axis(axis, title=title)
        image = add_map(axis, values, xs, ys, cmap=cmap, vmin=vmin, vmax=vmax)
        plot_routes(axis, routes, labels=False)
        plot_events(axis, events, small=True)
        bar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.015)
        bar.set_label(label, fontsize=8.3)
        bar.ax.tick_params(labelsize=7.8)
    axes[0].legend(loc="upper left", fontsize=7.7, frameon=True)
    figure.savefig(FIGURES / "03_camera_c_gp_before_after_update.png", facecolor="white")
    plt.close(figure)


def write_summary(events: list[dict[str, str]]) -> None:
    hits = sum(int(f(row, "det_hit")) for row in events)
    summary = {
        "source": "actual Gazebo collection",
        "campaign": str(RUN_ROOT),
        "camera_id": "camera_C",
        "runs": list(ROUTE_COLORS),
        "planned_route_length_m": 10.2,
        "events": len(events),
        "detections": hits,
        "misses": len(events) - hits,
        "prior": str(PRIOR),
        "posterior": str(POSTERIOR),
        "method": "expected-kernel GP (1.50 m length scale, 0.15 observation noise) with propagated noisy-odometry covariance and Beta(1,1) binary-target smoothing",
        "contains_ground_truth": False,
    }
    (FIGURES / "camera_c_actual_collection_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    events, routes = campaign_data()
    route_and_observations(events, routes)
    pose_uncertainty(events, routes)
    gp_before_after(events, routes)
    write_summary(events)
    for path in sorted(FIGURES.glob("0*_camera_c_*.png")):
        print(path)


if __name__ == "__main__":
    main()
