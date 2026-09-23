#!/usr/bin/env python3
"""Render the information removed by stopping camera B under each covariance model.

Three panels, one per covariance model, over the western task. The background is
the scalar network information that camera B contributed at each position,
which is exactly the information the planner loses when that camera is stopped.
The intact and dropout routes of the matched lowest seed are overlaid.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/deck_figures"))
from style import draw_warehouse, layout  # noqa: E402

FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
PLANNING_DIR = ROOT / "logs/thesis_final_pipeline_v1/planning_precision"
STOPPED = "camera_B"
WEST_TASK_SUFFIX = "west"

MODELS = (
    ("m0_planning_precision.npz", "global", r"(a) global $R_0$"),
    ("m1_planning_precision.npz", "per_camera", r"(b) camera-specific $R_1$"),
    ("m2_planning_precision.npz", "spatial", r"(c) spatial $R_2$"),
)


def read_plan(run_dir: Path) -> np.ndarray:
    with (run_dir / "global_plan.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(r["x"]), float(r["y"])] for r in rows])


def lost_information(artifact: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar information contributed by the stopped camera at every position."""
    with np.load(artifact, allow_pickle=False) as archive:
        xs = np.asarray(archive["xs"])
        ys = np.asarray(archive["ys"])
        ids = archive["camera_ids"].astype(str).tolist()
        precision = np.asarray(archive["matched_precision_m2_inv"])
    field = precision[ids.index(STOPPED)]
    return xs, ys, 0.5 * np.trace(field, axis1=-2, axis2=-1)


def pick_runs(results: list[dict]) -> dict[tuple[str, str], dict]:
    """Lowest matched seed per model and network state on the western task."""
    chosen: dict[tuple[str, str], dict] = {}
    for row in results:
        if not row.get("evidence_valid"):
            continue
        if not row["task"].endswith(WEST_TASK_SUFFIX):
            continue
        model, state = row["condition"].rsplit("_", 1)
        key = (model, state)
        current = chosen.get(key)
        if current is None or int(row["seed"]) < int(current["seed"]):
            chosen[key] = row
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads((args.analysis / "run_results.json").read_text())
    runs = pick_runs(results)

    fields = [lost_information(PLANNING_DIR / name) for name, _, _ in MODELS]
    finite = np.concatenate([f[np.isfinite(f) & (f > 0)].ravel() for _, _, f in fields])
    vmin, vmax = np.percentile(finite, 1), np.percentile(finite, 99)
    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)

    lay = layout()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)

    mesh = None
    for ax, (_, model, title), (xs, ys, field) in zip(axes, MODELS, fields):
        mesh = ax.pcolormesh(xs, ys, field, norm=norm, cmap="magma", shading="nearest",
                             rasterized=True, zorder=0)
        draw_warehouse(ax, lay, show_cameras=True, camera_labels=False, rack_alpha=0.30)

        for state, colour, style, width in (("intact", "#00B7EB", "-", 5.0),
                                            ("removal", "#FFFFFF", "--", 2.0)):
            row = runs.get((model, state))
            if row is None:
                continue
            plan = read_plan(ROOT / row["run"])
            ax.plot(plan[:, 0], plan[:, 1], style, color=colour, linewidth=width,
                    solid_capstyle="round", zorder=4)
            if state == "removal":
                ax.plot(plan[0, 0], plan[0, 1], "o", color="#FFFFFF", markersize=7,
                        markeredgecolor="#1d2530", zorder=5)
                ax.plot(plan[-1, 0], plan[-1, 1], "*", color="#FFFFFF", markersize=15,
                        markeredgecolor="#1d2530", zorder=5)

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    bar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.86, pad=0.015)
    bar.set_label("information removed with camera B (m$^{-2}$)", fontsize=11)

    handles = [
        Line2D([], [], color="#00B7EB", linestyle="-", linewidth=5.0, label="intact route"),
        Line2D([], [], color="#FFFFFF", linestyle="--", linewidth=2.0, label="dropout route"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.85)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"dropout_information_loss.{suffix}", dpi=240,
                    bbox_inches="tight")
    print("wrote", FIGURES / "dropout_information_loss.pdf")


if __name__ == "__main__":
    main()
