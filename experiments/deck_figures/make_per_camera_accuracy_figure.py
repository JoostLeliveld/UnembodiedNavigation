#!/usr/bin/env python3
"""Render each camera's predicted measurement accuracy in centimetres, per covariance model.

This is the same artifact as the per-camera information figure, expressed on an
interpretable scale. The planning field stores the precision Lambda = R^-1 of the matched
runtime covariance. Inverting it back and taking the larger principal standard deviation
gives the worst-axis one-sigma accuracy at that position, in centimetres: "here a sighting
from this camera is good to about 2 cm, here only to 25 cm".

Contours mark the 2, 5, 10 and 25 cm accuracy bands. Positions where the spatial model has
no local support fall back to the locked 10 m prior standard deviation; those are shown as
an explicit "no support" stratum rather than as a very large accuracy number, because they
report the absence of evidence and not a measured accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/deck_figures"))
from style import draw_warehouse, layout  # noqa: E402

FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
PLANNING_DIR = ROOT / "logs/thesis_final_pipeline_v1/planning_precision"

MODELS = (
    ("m0_planning_precision.npz", r"global $R_0$"),
    ("m1_planning_precision.npz", r"camera-specific $R_1$"),
    ("m2_planning_precision.npz", r"spatial $R_2$"),
)

# Accuracy bands in centimetres, read directly off the colour bar.
BANDS = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
PRIOR_SIGMA_CM = 1000.0   # locked 10 m prior; anything at it has no local support
NO_SUPPORT_CM = 999.0


def worst_axis_sigma_cm(artifact: Path):
    """Larger principal standard deviation of the matched covariance, in centimetres."""
    with np.load(artifact, allow_pickle=False) as archive:
        xs = np.asarray(archive["xs"])
        ys = np.asarray(archive["ys"])
        ids = archive["camera_ids"].astype(str).tolist()
        precision = np.asarray(archive["matched_precision_m2_inv"])
    covariance = np.linalg.inv(precision)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return xs, ys, ids, np.sqrt(eigenvalues[..., 1]) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning-dir", type=Path, default=PLANNING_DIR)
    parser.add_argument("--stem", default="per_camera_accuracy")
    args = parser.parse_args()

    grids = [worst_axis_sigma_cm(args.planning_dir / name) for name, _ in MODELS]
    ids = grids[-1][2]

    cmap = plt.get_cmap("viridis_r", len(BANDS) - 1)
    cmap.set_over("#2b2b2b")
    cmap.set_bad("#f2f1ee")
    norm = BoundaryNorm(BANDS, cmap.N)

    lay = layout()
    fig, axes = plt.subplots(len(MODELS), len(ids), figsize=(15.4, 8.6),
                             constrained_layout=True)

    mesh = None
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for row, ((_, model_label), (xs, ys, cam_ids, sigma)) in enumerate(zip(MODELS, grids)):
        for col, cam in enumerate(cam_ids):
            ax = axes[row, col]
            field = np.ma.masked_greater_equal(sigma[col], NO_SUPPORT_CM)
            mesh = ax.pcolormesh(xs, ys, field, cmap=cmap, norm=norm,
                                 shading="nearest", rasterized=True, zorder=0)
            draw_warehouse(ax, lay, show_cameras=True, camera_labels=False, rack_alpha=0.30)

            supported = sigma[col] < NO_SUPPORT_CM
            summary.setdefault(model_label, {})[cam] = {
                "median_sigma_cm_where_supported": (
                    float(np.median(sigma[col][supported])) if supported.any() else float("nan")),
                "area_fraction_under_5cm": float((sigma[col] < 5.0).mean()),
                "area_fraction_under_10cm": float((sigma[col] < 10.0).mean()),
                "area_fraction_no_support": float((~supported).mean()),
            }
            share = summary[model_label][cam]["area_fraction_under_10cm"] * 100.0
            ax.set_title(f"{cam.replace('camera_', 'camera ')}\n{share:.0f}% of area $<$10 cm",
                         fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(model_label, fontsize=12)

    assert mesh is not None
    bar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.7, pad=0.012,
                       ticks=BANDS, extend="max")
    bar.set_label(r"predicted one-$\sigma$ accuracy, worst axis (cm)", fontsize=11)
    bar.ax.set_yticklabels([f"{b:g}" for b in BANDS])

    fig.legend(
        handles=[Patch(facecolor="#f2f1ee", edgecolor="#8a8983",
                       label="no local support (falls back to the 10 m prior)")],
        loc="lower center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{args.stem}.{suffix}", dpi=240, bbox_inches="tight")
    print("wrote", FIGURES / f"{args.stem}.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
