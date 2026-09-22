#!/usr/bin/env python3
"""Render representative planned routes from the sealed final campaign."""
from __future__ import annotations

import argparse
import csv
import json
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

FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
FIELD = ROOT / "logs/thesis_final_pipeline_v1/planning_precision/m2_planning_precision.npz"
MODELS = ("global", "per_camera", "spatial")
COLOURS = {"global": "#0072B2", "per_camera": "#E69F00", "spatial": "#009E73"}
LABELS = {"global": "global", "per_camera": "per-camera", "spatial": "spatial"}
WIDTHS = {"global": 3.9, "per_camera": 2.5, "spatial": 1.4}


def read_plan(run: Path) -> np.ndarray:
    with (run / "global_plan.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(row["x"]), float(row["y"])] for row in rows])


def choose_runs(results: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    selected = {}
    for row in results:
        if not row.get("evidence_valid"):
            continue
        model, state = row["condition"].rsplit("_", 1)
        key = (row["task"], state)
        selected.setdefault(key, {})
        current = selected[key].get(model)
        if current is None or int(row["seed"]) < int(current["seed"]):
            selected[key][model] = row
    return selected


def field_grid(active: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(FIELD, allow_pickle=False) as archive:
        xs, ys = np.asarray(archive["xs"]), np.asarray(archive["ys"])
        ids = archive["camera_ids"].astype(str).tolist()
        precision = np.asarray(archive["matched_precision_m2_inv"])
    network = precision[[ids.index(camera) for camera in active]].sum(axis=0)
    return xs, ys, 0.5 * np.trace(network, axis1=-2, axis2=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.analysis / "run_results.json").read_text())
    selected = choose_runs(results)
    tasks = list(dict.fromkeys(row["task"] for row in results))
    if len(tasks) != 2:
        raise RuntimeError("final route figure requires exactly two tasks")

    fields = []
    for task in tasks:
        for state in ("intact", "removal"):
            row = selected[(task, state)]["spatial"]
            manifest = json.loads((ROOT / row["run"] / "run_manifest.json").read_text())
            fields.append(field_grid(manifest["camera_network_active_camera_ids"])[2])
    positive = np.concatenate([field[field > 0] for field in fields])
    norm = PowerNorm(gamma=0.42, vmin=0.0, vmax=float(np.percentile(positive, 99)))

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.1), constrained_layout=True)
    image = None
    task_labels = {
        "thesis09_parallel_aisles_west": "western parallel aisles",
        "thesis09_parallel_aisles_central": "central parallel aisles",
    }
    for row_index, task in enumerate(tasks):
        for col_index, state in enumerate(("intact", "removal")):
            ax = axes[row_index, col_index]
            representatives = selected[(task, state)]
            spatial = representatives["spatial"]
            manifest = json.loads((ROOT / spatial["run"] / "run_manifest.json").read_text())
            xs, ys, values = field_grid(manifest["camera_network_active_camera_ids"])
            image = ax.pcolormesh(xs, ys, values, shading="nearest", cmap="viridis",
                                  norm=norm, alpha=0.86, rasterized=True)
            draw_warehouse(ax, layout(), show_cameras=True, camera_labels=False, rack_alpha=0.82)
            for model in MODELS:
                points = read_plan(ROOT / representatives[model]["run"])
                ax.plot(points[:, 0], points[:, 1], color=COLOURS[model],
                        lw=WIDTHS[model], solid_capstyle="round", zorder=15)
            points = read_plan(ROOT / spatial["run"])
            ax.scatter(*points[0], s=25, color="#222222", zorder=20)
            ax.scatter(*points[-1], s=55, marker="*", color="#222222", zorder=20)
            if state == "removal":
                camera = next(camera for camera in layout().cameras if camera.name == "B")
                ax.plot(camera.x, camera.y, marker="x", color="#B2182B", ms=12,
                        mew=2.2, zorder=25)
            ax.set_title("intact network" if state == "intact" else "camera B removed",
                         fontsize=8.5, fontweight="bold")
            if col_index == 0:
                ax.text(-0.04, 0.5, task_labels.get(task, task), transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=7.5)
    handles = [Line2D([0], [0], color=COLOURS[model], lw=WIDTHS[model],
                      label=LABELS[model]) for model in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.045), fontsize=7.4)
    fig.colorbar(image, ax=axes, shrink=0.65, pad=0.015,
                 label=r"spatial-model precision, $\frac{1}{2}\mathrm{tr}(\Lambda)$ (m$^{-2}$)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"navigation_campaign_routes.{suffix}", dpi=240,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
