#!/usr/bin/env python3
"""Figure 1: the change the robot has to notice, shown in the camera and on the floor.

Top row is a pose-matched pair of real rendered frames from camera B -- the same
commanded robot position, the same camera, the same lighting -- before and after twelve
of twenty-seven rack rows are restocked one layer taller.  Bottom row is what that does
to floor coverage.  The reader should be able to see the intervention, not take it on
faith from a heat map.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C          # noqa: E402
import choose_layout as CL  # noqa: E402
import oracle as ora        # noqa: E402

CAP = C.CAPTURE_ROOT
PAIR = ("commissioning_grid_20260807/images/003800_xy0475_h00_external_camera_b.jpg",
        "recfg_holdout_L1/images/001900_xy0475_h00_external_camera_b.jpg")
OUT = C.OUT_ROOT / "figures"


def coverage(world_name: str, eligible: np.ndarray) -> np.ndarray:
    scene = ora.OracleScene.from_world(C.WORLDS / f"{world_name}.world.sdf", list(C.CAMERAS))
    grids = ora.visibility_grids(scene.cameras, C.floor_grid(), scene.static_prisms, (),
                                 target_height_m=C.TARGET_HEIGHT_M)
    n = np.zeros(eligible.shape, dtype=float)
    for c in C.CAMERAS:
        n += ((grids[c] == ora.VISIBLE) & eligible).astype(float)
    return n


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = C.floor_grid()
    xs, ys = grid.x_centres, grid.y_centres
    drive = CL.driveable_mask(grid, CL.lanes())
    base = ora.OracleScene.from_world(
        C.WORLDS / f"{C.ENV_BY_KEY['L0'].world_name}.world.sdf", list(C.CAMERAS))
    gx, gy = np.meshgrid(xs, ys)
    eligible = drive.copy()
    for p in base.static_prisms:
        eligible &= ~((gx >= p.xmin) & (gx <= p.xmax) & (gy >= p.ymin) & (gy <= p.ymax))

    n0 = coverage(C.ENV_BY_KEY["L0"].world_name, eligible)
    n1 = coverage(C.ENV_BY_KEY["L1"].world_name, eligible)
    dark0, dark1 = (n0 == 0) & eligible, (n1 == 0) & eligible
    went_dark = dark1 & ~dark0
    lost_one = (n1 < n0) & ~went_dark & eligible

    restocked = set(CL.SELECTED) if hasattr(CL, "SELECTED") else set()
    segs = {s["name"]: s for s in CL.rack_segments(
        C.WORLDS / f"{C.ENV_BY_KEY['L0'].world_name}.world.sdf")}

    fig = plt.figure(figsize=(7.12, 2.95))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.60)

    for j, (path, title) in enumerate(zip(
            PAIR, ("Nominal warehouse", "After restocking 12 of 27 rack rows"))):
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        ax.imshow(plt.imread(CAP / path)[70:450])
        ax.set_title(title, fontsize=6.2)
        ax.set_xticks([]); ax.set_yticks([])
    for j, (n, dark, title) in enumerate((
            (n0, dark0, f"Nominal: {int(dark0.sum())} driveable cells\nseen by no camera"),
            (n1, dark1, f"Restocked: {int(dark1.sum())} driveable cells\nseen by no camera"))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        show = np.where(eligible, n, np.nan)
        im = ax.pcolormesh(xs, ys, show, cmap="viridis", vmin=0, vmax=4, shading="nearest")
        ax.set_title(title, fontsize=5.6)
        ax.set_xlabel("x (m)", fontsize=5.6)
        if j == 0:
            ax.set_ylabel("y (m)", fontsize=5.6)
        ax.set_aspect("equal"); ax.tick_params(labelsize=4.8)
        if j == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
            cb.set_label("cameras that see this cell", fontsize=4.6, labelpad=1.5)
            cb.ax.tick_params(labelsize=4.4)

    ax = fig.add_subplot(gs[1, 4:6])
    ax.set_facecolor("#f2f2f2")
    for name, s in segs.items():
        ax.add_patch(Rectangle((s["xmin"], s["ymin"]), s["xmax"] - s["xmin"],
                               s["ymax"] - s["ymin"], facecolor="none",
                               edgecolor="#bbbbbb", lw=0.35))
    yy, xx = np.where(lost_one)
    ax.scatter(xs[xx], ys[yy], s=1.1, c="#e8a33d", label=f"lost a camera ({int(lost_one.sum())})")
    yy, xx = np.where(went_dark)
    ax.scatter(xs[xx], ys[yy], s=1.3, c="#c1272d", label=f"went dark ({int(went_dark.sum())})")
    ax.set_title("What the restock cost the network", fontsize=5.6)
    ax.set_xlabel("x (m)", fontsize=5.6)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal"); ax.tick_params(labelsize=4.8)
    ax.legend(loc="lower left", fontsize=4.4, framealpha=0.92)

    fig.suptitle("Restocking the racks removes sight-lines without touching a single aisle",
                 fontsize=7.0, y=1.005)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_world.{ext}", dpi=200, bbox_inches="tight")
    print(f"[fig1] dark {int(dark0.sum())} -> {int(dark1.sum())}, "
          f"went_dark={int(went_dark.sum())}, lost_one={int(lost_one.sum())}")
    print(f"[fig1] wrote {OUT/'fig1_world.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
