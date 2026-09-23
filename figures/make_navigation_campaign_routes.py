#!/usr/bin/env python3
"""Render the frozen six-condition Stage-09 planned-route comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from style import draw_warehouse, layout  # noqa: E402

PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
MODELS = ("global", "per_camera", "spatial")
LABEL = {"global": "global", "per_camera": "per-camera", "spatial": "spatial"}
COLOUR = {"global": "#0072B2", "per_camera": "#E69F00", "spatial": "#009E73"}
WIDTH = {"global": 4.8, "per_camera": 3.1, "spatial": 1.7}
STATES = ("intact", "removal")
SPATIAL_ARTIFACT = ROOT / (
    "logs/thesis_final_pipeline_v1/planning_precision/"
    "m2_planning_precision.npz"
)
DIAGNOSTIC = ROOT / (
    "logs/thesis_final_pipeline_v1/stage09_navigation/"
    "task_camera_pair_diagnostic.json"
)


def load_run(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f"{model}_{state}" for model in MODELS for state in STATES}
    if (report.get("schema") != "thesis_stage09_offline_global_routes.v1"
            or set(report["results"]) != expected or not (directory / ".complete").is_file()):
        raise ValueError(f"{directory}: incomplete or noncanonical route solve")
    report["_directory"] = str(directory.resolve())
    return report


def route(report: dict, model: str, state: str) -> np.ndarray:
    item = report["results"][f"{model}_{state}"]
    path = Path(report["_directory"]) / item["artifact"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["display_states"], dtype=float)[:, :2]


def decorate(ax: plt.Axes, report: dict, title: str) -> None:
    draw_warehouse(ax, layout(), show_cameras=True, camera_labels=False, rack_alpha=0.82)
    start, goal = report["task"]["start"], report["task"]["goal"]
    ax.scatter(start["x"], start["y"], s=27, color="#222222", zorder=20)
    ax.scatter(goal["x"], goal["y"], s=58, color="#222222", marker="*", zorder=20)
    ax.set_title(title, fontsize=8.5, fontweight="bold")


def removed_camera(report: dict) -> str:
    intact = set(report["results"]["spatial_intact"]["active_cameras"])
    removal = set(report["results"]["spatial_removal"]["active_cameras"])
    difference = sorted(intact - removal)
    if len(difference) != 1:
        raise ValueError(f"expected one removed camera, got {difference}")
    return difference[0].removeprefix("camera_")


def mark_removed(ax: plt.Axes, report: dict) -> None:
    camera = next(value for value in layout().cameras if value.name == removed_camera(report))
    ax.plot(camera.x, camera.y, marker="x", color="#B2182B", ms=13, mew=2.3, zorder=25)


def resample(points: np.ndarray, count: int = 180) -> tuple[np.ndarray, np.ndarray]:
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(distance) > 1e-9))
    distance, points = distance[keep], points[keep]
    query = np.linspace(0.0, distance[-1], count)
    sampled = np.column_stack((np.interp(query, distance, points[:, 0]),
                               np.interp(query, distance, points[:, 1])))
    return query, sampled


def information_profile(report: dict, field_state: str,
                        route_state: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    route_state = route_state or field_state
    xs, ys, scalar = information_grid(report, field_state)
    distance, points = resample(route(report, "spatial", route_state))
    interpolator = RegularGridInterpolator((ys, xs), scalar, bounds_error=False, fill_value=0.0)
    return distance, interpolator(points[:, [1, 0]])


def mean_information_on_route(report: dict, field_state: str,
                              route_state: str) -> float:
    xs, ys, scalar = information_grid(report, field_state)
    _, points = resample(route(report, "spatial", route_state), count=500)
    interpolator = RegularGridInterpolator((ys, xs), scalar,
                                            bounds_error=False, fill_value=0.0)
    return float(np.mean(interpolator(points[:, [1, 0]])))


def selection_margin(report: dict) -> float:
    score = removal_diagnostic(report)
    return 100.0 * (score["runner_up"]["cost"] - score["winner"]["cost"]) / score[
        "winner"]["cost"]


def removal_diagnostic(report: dict) -> dict:
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    camera = removed_camera(report)
    return diagnostic["tasks"][report["task"]["name"]]["scores"]["spatial"][
        f"remove_{camera}"]


def information_grid(report: dict, state: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact scalar spatial field consumed by the selected condition."""
    with np.load(SPATIAL_ARTIFACT, allow_pickle=False) as archive:
        xs, ys = np.asarray(archive["xs"]), np.asarray(archive["ys"])
        camera_ids = archive["camera_ids"].astype(str).tolist()
        field = np.asarray(archive["matched_precision_m2_inv"])
    active = report["results"][f"spatial_{state}"]["active_cameras"]
    network = field[[camera_ids.index(camera) for camera in active]].sum(axis=0)
    return xs, ys, 0.5 * np.trace(network, axis1=-2, axis2=-1)


def draw_information(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray,
                     values: np.ndarray, *, norm, cmap: str):
    return ax.pcolormesh(xs, ys, values, shading="nearest", cmap=cmap, norm=norm,
                         alpha=0.86, rasterized=True, zorder=0)


def save(fig: plt.Figure, stem: str) -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(PAPER_FIGURES / f"{stem}.{suffix}", dpi=240,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def detailed(report: dict) -> None:
    fig = plt.figure(figsize=(7.16, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.7, 1.0))
    ax_map = fig.add_subplot(grid[0, :])
    xs, ys, removal_field = information_grid(report, "removal")
    positive = removal_field[removal_field > 0]
    draw_information(ax_map, xs, ys, removal_field,
                     norm=PowerNorm(gamma=0.42, vmin=0.0,
                                    vmax=float(np.percentile(positive, 99))),
                     cmap="viridis")
    decorate(ax_map, report, "(a) Spatial-model planned routes")
    mark_removed(ax_map, report)
    removed = removed_camera(report)
    styles = {"intact": ("#0072B2", "-", "intact network"),
              "removal": ("#D55E00", "--", f"camera {removed} removed")}
    for state, (colour, line, label) in styles.items():
        points = route(report, "spatial", state)
        ax_map.plot(points[:, 0], points[:, 1], color=colour, lw=2.25,
                    ls=line, label=label, zorder=16)
    ax_map.legend(frameon=False, loc="lower center", ncol=2, fontsize=7.2)

    ax_info = fig.add_subplot(grid[1, 0])
    profile_styles = {
        "intact": ("#444444", "--", "intact-selected route"),
        "removal": ("#D55E00", "-", "removal-selected route"),
    }
    for route_state, (colour, line, label) in profile_styles.items():
        distance, information = information_profile(report, "removal", route_state)
        ax_info.plot(distance, information, color=colour, ls=line, lw=1.55, label=label)
    ax_info.set(title="(b) Both routes under the removal field", xlabel="route progress (m)",
                ylabel=r"mean information (m$^{-2}$)")
    ax_info.legend(frameon=False, fontsize=5.7)

    ax_cost = fig.add_subplot(grid[1, 1])
    diagnostic = removal_diagnostic(report)
    by_route = {item["route"]: item for item in diagnostic["candidates"]}
    route_names = [report["results"][f"spatial_{state}"]["selected_source"].split(":")[-1]
                   for state in STATES]
    candidates = [by_route[name] for name in route_names]
    x = np.arange(2)
    ambiguity = [item["ambiguity"] for item in candidates]
    risk = [item["risk"] for item in candidates]
    ax_cost.bar(x, ambiguity, color=[profile_styles[s][0] for s in STATES],
                width=0.62, label="ambiguity")
    ax_cost.bar(x, risk, bottom=ambiguity, color="#666666", alpha=0.65, width=0.62, label="risk")
    ax_cost.set_xticks(x, ["intact route", "removal route"])
    ax_cost.set(title="(c) Both objectives under removal", ylabel="objective contribution")
    ax_cost.text(0.5, 0.98, f"selection margin {selection_margin(report):.3f}%",
                 transform=ax_cost.transAxes, ha="center", va="top", fontsize=6.0)
    ax_cost.legend(frameon=False, fontsize=6.5)
    for ax in (ax_info, ax_cost):
        ax.grid(axis="y", color="#dddddd", lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.5)
        ax.title.set_fontsize(8.0)
        ax.xaxis.label.set_size(7.0)
        ax.yaxis.label.set_size(7.0)
    save(fig, "navigation_detailed_routes")


def campaign(reports: list[dict]) -> None:
    if len(reports) != 2:
        raise ValueError("campaign figure requires exactly two diagnostic tasks")
    all_fields = [information_grid(report, state)[2]
                  for report in reports for state in STATES]
    positive = np.concatenate([value[value > 0] for value in all_fields])
    norm = PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.percentile(positive, 99)))
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.35), constrained_layout=True)
    image = None
    for row, report in enumerate(reports):
        task_label = report["task"]["name"].replace("thesis09_", "").replace("_", " ")
        for col, state in enumerate(STATES):
            removed = removed_camera(report)
            xs, ys, field = information_grid(report, state)
            image = draw_information(axes[row, col], xs, ys, field,
                                     norm=norm, cmap="viridis")
            decorate(axes[row, col], report,
                     "intact network" if col == 0 else f"camera {removed} removed")
            if state == "removal":
                mark_removed(axes[row, col], report)
            axes[row, 0].text(-0.04, 0.5, task_label, transform=axes[row, 0].transAxes,
                              rotation=90, va="center", ha="right", fontsize=7.5)
            for model in MODELS:
                points = route(report, model, state)
                axes[row, col].plot(points[:, 0], points[:, 1], color=COLOUR[model],
                                    lw=WIDTH[model], zorder=15 + MODELS.index(model),
                                    solid_capstyle="round")
    handles = [Line2D([0], [0], color=COLOUR[m], lw=WIDTH[m], label=LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.012), fontsize=7.4)
    fig.colorbar(image, ax=axes, shrink=0.64, pad=0.015,
                label=r"spatial-model precision, "
                      r"$\frac{1}{2}\mathrm{tr}(\Lambda)$ (m$^{-2}$)")
    save(fig, "navigation_campaign_routes")


def spatial_evidence(reports: list[dict]) -> None:
    """Show the field, its removal-induced loss and the spatial-model routes."""
    if len(reports) != 2:
        raise ValueError("spatial evidence figure requires exactly two tasks")
    grids = []
    for report in reports:
        xs, ys, intact = information_grid(report, "intact")
        _, _, removal = information_grid(report, "removal")
        grids.append((xs, ys, intact, removal, np.maximum(intact - removal, 0.0)))
    positive = np.concatenate([g[2][g[2] > 0] for g in grids]
                              + [g[3][g[3] > 0] for g in grids])
    losses = np.concatenate([g[4].ravel() for g in grids])
    field_norm = PowerNorm(gamma=0.42, vmin=0.0,
                           vmax=float(np.percentile(positive, 99)))
    loss_norm = PowerNorm(gamma=0.42, vmin=0.0,
                          vmax=max(float(np.percentile(losses, 99)), 1.0))
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.15), constrained_layout=True)
    field_image = loss_image = None
    for row, (report, grid) in enumerate(zip(reports, grids, strict=True)):
        xs, ys, intact, removal, loss = grid
        task_label = ("lane 08 to lane 12" if "lane08" in report["task"]["name"]
                      else "blind corridor")
        panels = ((intact, field_norm, "viridis", "intact information"),
                  (removal, field_norm, "viridis", "information after removal"),
                  (loss, loss_norm, "magma", "information removed"))
        for col, (values, norm, cmap, title) in enumerate(panels):
            image = draw_information(axes[row, col], xs, ys, values, norm=norm, cmap=cmap)
            if col < 2:
                field_image = image
            else:
                loss_image = image
            decorate(axes[row, col], report, title)
            if col > 0:
                mark_removed(axes[row, col], report)
            intact_route = route(report, "spatial", "intact")
            removed_route = route(report, "spatial", "removal")
            if col == 0:
                axes[row, col].plot(intact_route[:, 0], intact_route[:, 1],
                                    color="#56B4E9", lw=2.2, zorder=15)
            else:
                axes[row, col].plot(intact_route[:, 0], intact_route[:, 1],
                                    color="white", lw=2.8, ls="--", zorder=14)
                axes[row, col].plot(intact_route[:, 0], intact_route[:, 1],
                                    color="#333333", lw=1.05, ls="--", zorder=15)
                axes[row, col].plot(removed_route[:, 0], removed_route[:, 1],
                                    color="#E69F00", lw=2.2, zorder=16)
        intact_mean = mean_information_on_route(report, "removal", "intact")
        removal_mean = mean_information_on_route(report, "removal", "removal")
        change = 100.0 * (removal_mean / intact_mean - 1.0)
        axes[row, 1].text(
            0.97, 0.04,
            f"mean route information: {change:+.1f}%\n"
            f"objective margin: {selection_margin(report):.3f}%",
            transform=axes[row, 1].transAxes, ha="right", va="bottom", fontsize=5.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0),
            zorder=30)
        axes[row, 0].text(-0.04, 0.5, task_label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=7.2)
    handles = [
        Line2D([0], [0], color="#56B4E9", lw=2.2, label="selected with intact network"),
        Line2D([0], [0], color="#333333", lw=1.05, ls="--",
               label="intact route under removal"),
        Line2D([0], [0], color="#E69F00", lw=2.2,
               label="selected after removal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=6.6)
    fig.colorbar(field_image, ax=axes[:, :2], shrink=0.62, pad=0.012,
                 label=r"expected information (m$^{-2}$)")
    fig.colorbar(loss_image, ax=axes[:, 2], shrink=0.62, pad=0.012,
                 label=r"removed information (m$^{-2}$)")
    save(fig, "navigation_spatial_evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [load_run(path) for path in args.runs]
    detailed(reports[0])
    if len(reports) >= 2:
        campaign(reports[:2])
        spatial_evidence(reports[:2])
    print(f"wrote navigation figures to {PAPER_FIGURES}")


if __name__ == "__main__":
    main()
