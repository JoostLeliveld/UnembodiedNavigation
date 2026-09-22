#!/usr/bin/env python3
"""Show the explicit, equal-length parallel-aisle route-choice experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/deck_figures"))
from style import draw_warehouse, layout  # noqa: E402

SCORES = ROOT / (
    "logs/thesis_final_pipeline_v1/stage09_navigation/"
    "final_bayesian_parallel_aisle_scores.json"
)
FIELD = Path(
    "/home/joostleliveld/Thesis/2026-09-21/i-ah/outputs/"
    "final_bayesian_planning_information/m2_planning_information.npz"
)
OUTPUT = ROOT.parent / "papers/Thesis/figures"
CASE = "-6.95_vs_-3.05_y-6.25_8.625"
WEST = "aisle_x_-6.95"
EAST = "aisle_x_-3.05"
WEST_X = -6.95
EAST_X = -3.05
COLOUR = {WEST: "#0072B2", EAST: "#D55E00"}


def field(active: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(FIELD, allow_pickle=False) as archive:
        xs = np.asarray(archive["xs"])
        ys = np.asarray(archive["ys"])
        cameras = archive["camera_ids"].astype(str).tolist()
        information = np.asarray(archive["expected_information_m2_inv"])
    selected = information[[cameras.index(camera) for camera in active]].sum(axis=0)
    return xs, ys, 0.5 * np.trace(selected, axis1=-2, axis2=-1)


def route_points(case: dict, route_name: str) -> np.ndarray:
    return np.asarray([case["start"][:2], *case["routes"][route_name]], dtype=float)


def main() -> int:
    global CASE, WEST, EAST, WEST_X, EAST_X, COLOUR
    variant = os.environ.get("AISLE_FIGURE_VARIANT", "primary").strip().lower()
    if variant == "secondary":
        CASE = "-3.05_vs_0.975_y-5.75_8.625"
        WEST = "aisle_x_-3.05"
        EAST = "aisle_x_0.975"
        WEST_X = -3.05
        EAST_X = 0.975
        COLOUR = {WEST: "#0072B2", EAST: "#D55E00"}
        output_stem = "navigation_explicit_aisle_choice_second"
    elif variant == "primary":
        output_stem = "navigation_explicit_aisle_choice"
    else:
        raise ValueError("AISLE_FIGURE_VARIANT must be primary or secondary")
    report = json.loads(SCORES.read_text(encoding="utf-8"))[CASE]
    intact = tuple(f"camera_{letter}" for letter in "ABCDE")
    removal = tuple(camera for camera in intact if camera != "camera_B")
    grids = [field(intact), field(removal)]
    positive = np.concatenate([values[values > 0] for _, _, values in grids])
    norm = PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.percentile(positive, 99)))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), constrained_layout=True)
    image = None
    selected = {"intact": EAST, "remove_B": WEST}
    titles = {"intact": "intact network", "remove_B": "camera B removed"}
    for ax, state, grid in zip(axes, ("intact", "remove_B"), grids, strict=True):
        xs, ys, values = grid
        image = ax.pcolormesh(xs, ys, values, shading="nearest", cmap="viridis",
                              norm=norm, alpha=0.86, rasterized=True, zorder=0)
        draw_warehouse(ax, layout(), show_cameras=True, camera_labels=False,
                       rack_alpha=0.82)
        for route_name in (WEST, EAST):
            points = route_points(report, route_name)
            is_selected = route_name == selected[state]
            ax.plot(points[:, 0], points[:, 1], color=COLOUR[route_name],
                    lw=3.0 if is_selected else 1.6,
                    ls="-" if is_selected else "--", alpha=1.0 if is_selected else 0.78,
                    zorder=17 if is_selected else 15)
        start, goal = np.asarray(report["start"][:2]), np.asarray(report["goal"])
        ax.scatter(*start, s=32, color="#222222", zorder=21)
        ax.scatter(*goal, s=67, color="#222222", marker="*", zorder=21)
        ax.text(WEST_X, 1.0, f"west aisle  $x={WEST_X:g}$ m", rotation=90,
                ha="center", va="center", fontsize=6.8, color=COLOUR[WEST],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                      "pad": 1.2}, zorder=22)
        ax.text(EAST_X, 1.0, f"east aisle  $x={EAST_X:g}$ m", rotation=90,
                ha="center", va="center", fontsize=6.8, color=COLOUR[EAST],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                      "pad": 1.2}, zorder=22)
        if state == "remove_B":
            camera = next(item for item in layout().cameras if item.name == "B")
            ax.plot(camera.x, camera.y, marker="x", color="#B2182B", ms=14,
                    mew=2.5, zorder=25)
        winner = report["states"][state]["winner"]
        candidates = report["states"][state]["candidates"]
        margin = 100.0 * (candidates[1]["cost"] - candidates[0]["cost"]) / candidates[0]["cost"]
        ax.set_title(
            f"{titles[state]}\n"
            f"selected: {'west' if winner['route'] == WEST else 'east'} aisle "
            f"({margin:.2f}% margin)",
            fontsize=8.5, fontweight="bold")

    handles = [
        Line2D([0], [0], color=COLOUR[WEST], lw=2.4, label="west-aisle seed"),
        Line2D([0], [0], color=COLOUR[EAST], lw=2.4, label="east-aisle seed"),
        Line2D([0], [0], color="#555555", lw=2.4, ls="-", label="selected"),
        Line2D([0], [0], color="#555555", lw=1.6, ls="--", label="alternative"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=7.0)
    fig.colorbar(image, ax=axes, shrink=0.74, pad=0.015,
                 label=r"expected information, "
                       r"$\frac{1}{2}\mathrm{tr}(\Lambda^{\mathrm{plan}})$ (m$^{-2}$)")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{output_stem}.{suffix}", dpi=240,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUTPUT / f"{output_stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
