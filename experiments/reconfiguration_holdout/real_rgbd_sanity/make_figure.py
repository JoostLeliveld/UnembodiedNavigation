#!/usr/bin/env python3
"""Render the TorWIC physical RGB-D sanity-check figure from frozen outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-real-rgbd-sanity")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_ROOT = REPO / "logs/studies/reconfiguration_holdout/real_rgbd_sanity"
EXAMPLE_FRAME = "000460"  # middle preregistered held-out index, not outcome-selected
COLORS = {"left": "#167D8D", "right": "#8E5AA6"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    results = json.loads((args.root / "results/results.json").read_text(encoding="utf-8"))
    raw_root = args.root / "torwic_subset/raw"
    rgb = np.asarray(Image.open(raw_root / f"image_left/{EXAMPLE_FRAME}.png").convert("RGB"))
    depth = np.asarray(Image.open(raw_root / f"depth_left/{EXAMPLE_FRAME}.png"), dtype=float) * 0.001
    depth[(depth < 0.4) | (depth > 20.0)] = np.nan

    fig = plt.figure(figsize=(11.4, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_depth = fig.add_subplot(grid[0, 1])
    ax_pair = fig.add_subplot(grid[1, 0])
    ax_bar = fig.add_subplot(grid[1, 1])

    ax_rgb.imshow(rgb)
    ax_rgb.set_title("A  Physical warehouse RGB (held-out)", loc="left", weight="bold")
    ax_rgb.set_axis_off()
    ax_rgb.text(
        0.01, 0.02, "TorWIC-SLAM · left camera · frame 000460",
        transform=ax_rgb.transAxes, color="white", fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2, "edgecolor": "none"},
    )

    image = ax_depth.imshow(depth, cmap="magma_r", vmin=0.4, vmax=8.0)
    ax_depth.set_title("B  Co-registered sensor depth (evaluation only)", loc="left", weight="bold")
    ax_depth.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax_depth, fraction=0.035, pad=0.02)
    colorbar.set_label("optical-axis depth [m]")

    for record in results["per_frame"]:
        color = COLORS[record["camera"]]
        values = [record["raw"]["mae_m"], record["anchored"]["mae_m"]]
        ax_pair.plot([0, 1], values, color=color, alpha=0.52, linewidth=1.1, marker="o", ms=3)
    aggregate = results["aggregate_medians"]
    medians = [aggregate["raw"]["median_mae_m"], aggregate["anchored"]["median_mae_m"]]
    ax_pair.plot([0, 1], medians, color="black", linewidth=3, marker="o", ms=6, label="14-frame median")
    for side, color in COLORS.items():
        ax_pair.plot([], [], color=color, marker="o", label=f"{side} camera")
    ax_pair.set_xticks([0, 1], ["raw UniDepth", "frozen floor-affine"])
    ax_pair.set_yscale("log")
    ax_pair.set_ylabel("structure-pixel MAE [m] (log scale)")
    ax_pair.grid(axis="y", which="both", alpha=0.25)
    ax_pair.set_title("C  Every held-out camera-frame improves", loc="left", weight="bold")
    ax_pair.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax_pair.annotate(
        f"median {medians[0]:.2f} → {medians[1]:.2f} m\n(−{100*(medians[0]-medians[1])/medians[0]:.1f}%)",
        xy=(1, medians[1]), xytext=(0.58, 0.18), textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1}, fontsize=8.5,
    )

    groups = ("left", "right", "pooled")
    arms = ("raw", "anchored", "oracle_affine")
    labels = ("raw", "frozen affine", "oracle affine")
    bar_colors = ("#A8ADB4", "#DC8B31", "#4E9A63")
    x = np.arange(len(groups), dtype=float)
    width = 0.23
    for index, (arm, label, color) in enumerate(zip(arms, labels, bar_colors)):
        values = []
        for group in groups:
            source = aggregate if group == "pooled" else results["camera_medians"][group]
            values.append(source[arm]["median_mae_m"])
        ax_bar.bar(x + (index - 1) * width, values, width, label=label, color=color)
    ax_bar.set_xticks(x, groups)
    ax_bar.set_ylabel("median structure-pixel MAE [m]")
    ax_bar.set_title("D  Improvement transfers, but a gap remains", loc="left", weight="bold")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(frameon=False, fontsize=8)
    ax_bar.text(
        0.02, 0.97,
        "held-out floor-plane diagnostic: "
        f"{aggregate['plane_floor_diagnostic']['median_mae_m']:.3f} m median MAE",
        transform=ax_bar.transAxes, va="top", fontsize=8,
    )

    fig.suptitle(
        "A once-commissioned floor affine reduces monocular-depth scale bias on real warehouse RGB-D",
        fontsize=13, weight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Component sanity check only: one mobile-camera route, two views, model-generated semantic masks; "
        "not a physical reconfiguration or navigation replication. "
        "TorWIC-SLAM: CC BY-NC 4.0 + dataset terms.",
        ha="center", va="bottom", fontsize=7.5,
    )
    png = out / "real_rgbd_sanity.png"
    pdf = out / "real_rgbd_sanity.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
