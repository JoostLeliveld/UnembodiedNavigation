#!/usr/bin/env python3
"""Render the information each camera contributes at every position, per covariance model.

Five columns, one per camera; three rows, one per covariance model. The background is the
scalar information 0.5*tr(Lambda_c) that camera c contributes at each position, which is
exactly what the planner loses when that camera stops. This generalises the camera B
dropout figure to the whole installation: under R0 and R1 each camera contributes a
constant, so a removal has no location, while under R2 the contribution is concentrated in
the region that camera actually informs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from style import draw_warehouse, layout  # noqa: E402

FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
PLANNING_DIR = ROOT / "logs/thesis_final_pipeline_v1/planning_precision"

MODELS = (
    ("m0_planning_precision.npz", r"global $R_0$"),
    ("m1_planning_precision.npz", r"camera-specific $R_1$"),
    ("m2_planning_precision.npz", r"spatial $R_2$"),
)


def camera_information(artifact: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Scalar information contributed by every camera at every position."""
    with np.load(artifact, allow_pickle=False) as archive:
        xs = np.asarray(archive["xs"])
        ys = np.asarray(archive["ys"])
        ids = archive["camera_ids"].astype(str).tolist()
        precision = np.asarray(archive["matched_precision_m2_inv"])
    return xs, ys, ids, 0.5 * np.trace(precision, axis1=-2, axis2=-1)


def concentration(field: np.ndarray) -> float:
    """Share of a camera's information held by its most informative tenth of positions."""
    ordered = np.sort(field.ravel())[::-1]
    return float(ordered[: max(1, len(ordered) // 10)].sum() / ordered.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning-dir", type=Path, default=PLANNING_DIR)
    parser.add_argument("--stem", default="per_camera_information")
    args = parser.parse_args()

    grids = [camera_information(args.planning_dir / name) for name, _ in MODELS]
    ids = grids[-1][2]

    finite = np.concatenate([f[np.isfinite(f) & (f > 0)].ravel() for _, _, _, f in grids])
    norm = LogNorm(vmin=max(float(np.percentile(finite, 1)), 1e-3),
                   vmax=float(np.percentile(finite, 99)))

    lay = layout()
    fig, axes = plt.subplots(len(MODELS), len(ids), figsize=(15.4, 8.4),
                             constrained_layout=True)

    mesh: matplotlib.collections.QuadMesh | None = None
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for row, ((_, model_label), (xs, ys, cam_ids, info)) in enumerate(zip(MODELS, grids)):
        total = info.sum()
        for col, cam in enumerate(cam_ids):
            ax = axes[row, col]
            mesh = ax.pcolormesh(xs, ys, info[col], norm=norm, cmap="magma",
                                 shading="nearest", rasterized=True, zorder=0)
            draw_warehouse(ax, lay, show_cameras=True, camera_labels=False, rack_alpha=0.30)

            top10 = concentration(info[col])
            summary.setdefault(model_label, {})[cam] = {
                "share_of_network_information": float(info[col].sum() / total),
                "top_decile_share": top10,
            }
            ax.set_title(f"{cam.replace('camera_', 'camera ')}\ntop decile {top10 * 100:.0f}%",
                         fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(model_label, fontsize=12)

    assert mesh is not None
    bar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.7, pad=0.012)
    bar.set_label("information contributed by the camera (m$^{-2}$)", fontsize=11)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{args.stem}.{suffix}", dpi=240, bbox_inches="tight")
    print("wrote", FIGURES / f"{args.stem}.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
