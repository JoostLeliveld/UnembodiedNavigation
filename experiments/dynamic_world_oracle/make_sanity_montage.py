#!/usr/bin/env python3
"""Put the rendered frames and the oracle map side by side, so both can be checked at once.

The oracle is a claim about what a camera can see.  The only honest way to sanity
check it is to look at the picture that camera actually produced at the same
instant and confirm the obstacle is where the map says it is.  This draws the two
together for a chosen camera at chosen instants, and marks the cells the obstacle
took away.

    python3 experiments/dynamic_world_oracle/make_sanity_montage.py \
        --run logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import oracle as ora  # noqa: E402

CAMERA_LABEL = {
    "external_camera": "Camera A (south wall, west)",
    "external_camera_b": "Camera B (north wall, west)",
    "external_camera_c": "Camera C (south wall, east)",
    "external_camera_d": "Camera D (north wall, east)",
}

# occluded, visible, out of view, solid
CODE_COLOURS = ListedColormap(["#c0392b", "#eef4ea", "#e8e8e8", "#7f8c8d"])


def _records(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines() if line]


def _pick_instants(records: list[dict], camera: str) -> list[tuple[float, str]]:
    """Choose a before / during / after triple, labelled by what is happening."""
    rows = sorted((r for r in records if r["camera_id"] == camera), key=lambda r: r["timestamp"])
    empty = [r for r in rows if not r["obstacle_state"]]
    occupied = [r for r in rows if r["obstacle_state"]]
    if not empty or not occupied:
        return [(r["timestamp"], "") for r in rows[:3]]
    first_empty = empty[0]
    first_occupied = occupied[0]
    last_occupied = occupied[-1]
    after = [r for r in empty if r["timestamp"] > last_occupied["timestamp"]]
    chosen = [
        (first_empty["timestamp"], "clear aisle"),
        (first_occupied["timestamp"], "obstacle present"),
        (last_occupied["timestamp"], "obstacle moved"),
    ]
    if after:
        chosen.append((after[0]["timestamp"], "obstacle removed"))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--camera", default="external_camera")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    run = args.run
    records = _records(run)
    manifest = json.loads((run / "manifest.json").read_text())
    grid_meta = manifest["grid"]
    instants = _pick_instants(records, args.camera)
    baseline_t = instants[0][0]
    by_t = {r["timestamp"]: r for r in records if r["camera_id"] == args.camera}
    baseline = np.load(run / by_t[baseline_t]["oracle_visibility_grid"]["path"])

    n = len(instants)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    extent = [grid_meta["xmin"], grid_meta["xmax"], grid_meta["ymin"], grid_meta["ymax"]]
    for column, (t, phase) in enumerate(instants):
        record = by_t[t]
        codes = np.load(run / record["oracle_visibility_grid"]["path"])

        ax = axes[0, column]
        if record["rgb_path"]:
            ax.imshow(Image.open(run / record["rgb_path"]))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {t:.1f} s — {phase}\nwhat the camera actually rendered",
                     fontsize=10.5, weight="bold")

        ax = axes[1, column]
        ax.imshow(codes, origin="lower", extent=extent, cmap=CODE_COLOURS, vmin=0, vmax=3,
                  interpolation="nearest")
        newly_hidden = (baseline == ora.VISIBLE) & (codes != ora.VISIBLE)
        if newly_hidden.any():
            ys, xs = np.nonzero(newly_hidden)
            res = grid_meta["resolution_m"]
            for y, x in zip(ys, xs):
                ax.add_patch(Rectangle(
                    (grid_meta["xmin"] + x * res, grid_meta["ymin"] + y * res), res, res,
                    facecolor="none", edgecolor="#1f2933", linewidth=0.35))
        for obstacle in record["obstacle_state"]:
            box = obstacle["world_aabb"]
            ax.add_patch(Rectangle(
                (box["xmin"], box["ymin"]), box["xmax"] - box["xmin"], box["ymax"] - box["ymin"],
                facecolor="#f2994a", edgecolor="#7a3e00", linewidth=1.4, zorder=5))
        visible = int((codes == ora.VISIBLE).sum())
        lost = int(newly_hidden.sum())
        ax.set_xlabel("warehouse east (m)", fontsize=9)
        if column == 0:
            ax.set_ylabel("warehouse north (m)", fontsize=9)
        ax.set_title(
            f"floor this camera can see: {visible} cells"
            + (f"\n{lost} taken away by the obstacle" if lost else "\nsame as the clear aisle"),
            fontsize=10.5, weight="bold")
        ax.tick_params(labelsize=8)

    handles = [
        Patch(facecolor="#eef4ea", edgecolor="#999", label="camera can see this floor"),
        Patch(facecolor="#c0392b", edgecolor="#999", label="in view, but the sight-line is blocked"),
        Patch(facecolor="#e8e8e8", edgecolor="#999", label="outside this camera's image"),
        Patch(facecolor="#7f8c8d", edgecolor="#999", label="solid: racks, walls, obstacle"),
        Patch(facecolor="#f2994a", edgecolor="#7a3e00", label="obstacle footprint (simulator truth)"),
        Patch(facecolor="none", edgecolor="#1f2933", label="cell lost since the clear aisle"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    scenario = manifest["scenario"]
    fig.suptitle(
        f"{CAMERA_LABEL.get(args.camera, args.camera)}: the obstacle in the picture and the hole it "
        f"leaves in the ground-truth visibility map\n"
        f"{scenario['scenario_id']} — one obstacle, {len(scenario['events'])} timed events, "
        f"four cameras, {grid_meta['resolution_m']:.2f} m floor cells at "
        f"{scenario['target_height_m']:.2f} m target height "
        f"(simulator ground truth — evaluation only)",
        fontsize=12.5, weight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))

    out = args.out or (run / "sanity_montage.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
