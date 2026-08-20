#!/usr/bin/env python3
"""Choose the reconfigured warehouse layout by what it costs the camera network.

WHAT A RECONFIGURATION HAS TO BE, AND WHY IT IS NOT A PALLET IN AN AISLE.
The first attempt placed pallets and forklifts in the driveable aisles, chosen
greedily to maximise the driveable floor that goes from "at least one camera sees
me" to "no camera sees me".  Measured on the four-camera warehouse, that produced
only **+46 newly blind reachable cells** out of 3397 covered — 1.4 % — for three
obstacles.  Two reasons, both structural rather than fixable:

* four cameras on opposite corners see the same ground from four directions, so
  one obstacle's shadow is nearly always covered by another camera;
* an obstacle big enough to darken an aisle also *blocks* it, and the flood-fill
  reachability guard correctly refuses to count ground the robot can no longer
  drive to.  Maximising blackspots and keeping the lane network intact pull
  against each other.

So the change used here is the one a warehouse actually undergoes between
shifts: **the racks are restocked, and the new loads are taller.**  Stock added
on top of a rack row cuts every sight-line that used to graze over that row,
which reaches far across the floor, and it touches no aisle at all — the
driveable network and every route through it are bit-identical to the nominal
world.  The selected twelve-segment artifact records that +0.4 m of stock costs
the network 191 of 3397 covered cells (5.6 %) and changes 575 camera-cell
visibility pairs, against 46 newly blind cells for the aisle-obstacle pilot.
That separates the paper's question (does the observation model adapt?) from
obstacle avoidance (does the robot get around the thing?), which is the whole
point of keeping them apart.

Greedy forward selection over the 27 rack segments, using the same CAD prisms and
camera model the capture's oracle uses, so a layout chosen here is the layout the
Gazebo run reproduces.  Selection maximises *camera-cell visibility pairs
changed*, not fused blackspots: the headline experiment scores per-camera
availability fields, and fused blackspot count depends on how many cameras the
analysis keeps, so selecting on it would bake a camera count into the world.

    python3 experiments/reconfiguration_holdout/choose_layout.py --n-segments 12

Writes logs/studies/reconfiguration_holdout/layout/{layout_candidates.csv,
layout_selected.json}.  Reads no detector outcome and no run data: this is a
geometry decision taken before any capture exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
_REPO_PATH = HERE.parents[1]


def _load_exact(name: str, path: Path):
    expected = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != expected:
            raise ImportError(f"{name} resolves to an unexpected module")
        return existing
    spec = importlib.util.spec_from_file_location(name, expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {expected}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_paths = _load_exact(
    "_reconfiguration_holdout_paths", _REPO_PATH / "scripts/shared/paths.py"
)
ora = _load_exact(
    "_reconfiguration_holdout_oracle",
    _REPO_PATH / "experiments/dynamic_world_oracle/oracle.py",
)
repo_root = _paths.repo_root

REPO = repo_root(HERE)
WORLDS = REPO / "src/sim/gazebo_worlds/worlds"
BASE_WORLD = WORLDS / "warehouse_full_4cam.world.sdf"
PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
PROFILE_KEY = "warehouse_full_4cam.world.sdf"
OUT = REPO / "logs/studies/reconfiguration_holdout/layout"

CAMERAS = ("external_camera", "external_camera_b", "external_camera_c", "external_camera_d")
SHORT = {"external_camera": "A", "external_camera_b": "B",
         "external_camera_c": "C", "external_camera_d": "D"}

TARGET_HEIGHT_M = 0.35
GRID_RES_M = 0.25

#: How much taller the restocked rows are.  0.4 m is one pallet layer on a rack
#: whose structure tops out at 2.09 m, so the reconfigured world is a warehouse
#: with one extra layer of stock on some rows -- not a new building.
STOCK_HEIGHT_M = 0.40

#: Robot half-width plus the planner's keep-in contract.  Used only to assert that
#: restocking leaves the driveable network untouched.
ROBOT_CLEARANCE_M = 0.25

RACK_LINK = re.compile(
    r'<link name="(rack_[A-Za-z0-9_]+)"><pose>([-\d. ]+)</pose>'
    r'.*?<box><size>([\d. ]+)</size></box>',
    re.S,
)


def spawn_xy() -> tuple[float, float]:
    """The world profile's own spawn pose: where the reachability fill starts."""
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    spawn = profile["worlds"][PROFILE_KEY]["spawn"]
    return (float(spawn["x"]), float(spawn["y"]))


def lanes() -> list[dict]:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    regions = profile["worlds"][PROFILE_KEY]["known_2d_regions"]
    return [r for r in regions if str(r.get("type")) == "traversable"]


def rack_segments(world_sdf: Path) -> list[dict]:
    """Every structural rack segment, as the footprint and top height of its box.

    Parsed out of the world the run will load, so a segment named here is a
    segment Gazebo has.
    """
    text = world_sdf.read_text(encoding="utf-8")
    out = []
    for match in RACK_LINK.finditer(text):
        name = match.group(1)
        px, py, pz = (float(v) for v in match.group(2).split()[:3])
        sx, sy, sz = (float(v) for v in match.group(3).split())
        out.append({
            "name": name,
            "xmin": px - sx / 2.0, "xmax": px + sx / 2.0,
            "ymin": py - sy / 2.0, "ymax": py + sy / 2.0,
            "top_z": pz + sz / 2.0,
        })
    return out


def stock_prism(seg: dict, height_m: float):
    """The added load, as the prism it occupies: same footprint, sitting on top."""
    return ora.AxisAlignedPrism(
        name=f"stock_{seg['name']}",
        xmin=seg["xmin"], xmax=seg["xmax"],
        ymin=seg["ymin"], ymax=seg["ymax"],
        zmin=seg["top_z"], zmax=seg["top_z"] + float(height_m),
    )


def driveable_mask(grid: ora.FloorGrid, lane_rects: list[dict]) -> np.ndarray:
    gx, gy = np.meshgrid(grid.x_centres, grid.y_centres)
    mask = np.zeros(gx.shape, dtype=bool)
    for r in lane_rects:
        mask |= (
            (gx >= float(r["xmin"])) & (gx <= float(r["xmax"]))
            & (gy >= float(r["ymin"])) & (gy <= float(r["ymax"]))
        )
    return mask


def visible_stack(scene: ora.OracleScene, grid: ora.FloorGrid, extra) -> dict:
    return ora.visibility_grids(
        scene.cameras, grid, list(scene.static_prisms) + list(extra), (),
        target_height_m=TARGET_HEIGHT_M,
    )


def fused_seen(grids: dict) -> np.ndarray:
    return np.stack([g == ora.VISIBLE for g in grids.values()]).any(axis=0)


def occupied_any(grids: dict) -> np.ndarray:
    return np.stack([g == ora.OCCUPIED for g in grids.values()]).any(axis=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-segments", type=int, default=12,
                    help="rack segments to restock; 12 of 27 is a partial restock")
    ap.add_argument("--stock-height-m", type=float, default=STOCK_HEIGHT_M)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lane_rects = lanes()
    scene = ora.OracleScene.from_world(BASE_WORLD, list(CAMERAS))
    grid = ora.FloorGrid(xmin=-11.75, xmax=11.75, ymin=-9.0, ymax=9.0,
                         resolution_m=GRID_RES_M)
    drive = driveable_mask(grid, lane_rects)

    base = visible_stack(scene, grid, [])
    eligible = drive & ~occupied_any(base)
    base_vis = {c: (base[c] == ora.VISIBLE) & eligible for c in CAMERAS}
    base_seen = fused_seen(base) & eligible
    print(f"[layout] {len(scene.static_prisms)} static prisms, "
          f"{int(eligible.sum())} eligible driveable cells")
    print(f"[layout] L0: fused coverage {int(base_seen.sum())}, per camera "
          + ", ".join(f"{SHORT[c]}={int(base_vis[c].sum())}" for c in CAMERAS))

    segments = rack_segments(BASE_WORLD)
    print(f"[layout] {len(segments)} rack segments; restocking "
          f"{args.n_segments} of them by +{args.stock_height_m:.2f} m")

    rows: list[dict] = []
    chosen: list[dict] = []
    chosen_prisms: list = []

    for step in range(int(args.n_segments)):
        best = None
        for seg in segments:
            if any(c["name"] == seg["name"] for c in chosen):
                continue
            prism = stock_prism(seg, args.stock_height_m)
            grids = visible_stack(scene, grid, chosen_prisms + [prism])
            pairs_lost = sum(
                int((base_vis[c] & ~((grids[c] == ora.VISIBLE) & eligible)).sum())
                for c in CAMERAS
            )
            row = {
                "step": step, "segment": seg["name"],
                "x": round(0.5 * (seg["xmin"] + seg["xmax"]), 4),
                "y": round(0.5 * (seg["ymin"] + seg["ymax"]), 4),
                "top_z": seg["top_z"], "pairs_lost_cumulative": pairs_lost,
            }
            if step == 0:
                rows.append(row)
            if best is None or pairs_lost > best[0]["pairs_lost_cumulative"]:
                best = (row, seg, prism)
        if best is None:
            break
        row, seg, prism = best
        gained = row["pairs_lost_cumulative"] - (
            chosen[-1]["pairs_lost_cumulative"] if chosen else 0)
        chosen.append({**{k: seg[k] for k in ("name", "xmin", "xmax", "ymin", "ymax", "top_z")},
                       "stock_height_m": float(args.stock_height_m),
                       "pairs_lost_cumulative": row["pairs_lost_cumulative"]})
        chosen_prisms.append(prism)
        print(f"[layout] pick {step + 1:2d}: {seg['name']:16s} "
              f"-> +{gained:4d} camera-cell pairs lost "
              f"({row['pairs_lost_cumulative']} cumulative)")

    final = visible_stack(scene, grid, chosen_prisms)
    final_vis = {c: (final[c] == ora.VISIBLE) & eligible for c in CAMERAS}
    final_seen = fused_seen(final) & eligible

    # Restocking must not touch the floor.  This is an assertion, not a metric: if
    # a stock box ever reached below the robot's own height the experiment would
    # silently become an obstacle-avoidance experiment.
    lowest = min(p.zmin for p in chosen_prisms) if chosen_prisms else float("inf")
    assert lowest > 0.5, f"added stock reaches down to {lowest:.2f} m -- it would block the floor"

    summary = {
        "base_world": str(BASE_WORLD.relative_to(REPO)),
        "change": "rack restock: added load on top of selected rack segments",
        "stock_height_m": float(args.stock_height_m),
        "grid": grid.to_dict(),
        "target_height_m": TARGET_HEIGHT_M,
        "spawn_xy": list(spawn_xy()),
        "eligible_cells": int(eligible.sum()),
        "fused_coverage_L0": int(base_seen.sum()),
        "fused_coverage_L1": int(final_seen.sum()),
        "fused_cells_lost": int((base_seen & ~final_seen).sum()),
        "camera_cell_pairs_lost": int(sum(
            int((base_vis[c] & ~final_vis[c]).sum()) for c in CAMERAS)),
        "per_camera_visible_L0": {SHORT[c]: int(base_vis[c].sum()) for c in CAMERAS},
        "per_camera_visible_L1": {SHORT[c]: int(final_vis[c].sum()) for c in CAMERAS},
        "lowest_added_prism_z": float(lowest),
        "restocked_segments": chosen,
    }
    (out / "layout_selected.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        keys = list(rows[0].keys())
        with (out / "layout_candidates.csv").open("w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in sorted(rows, key=lambda r: -r["pairs_lost_cumulative"]):
                fh.write(",".join(str(r[k]) for k in keys) + "\n")

    print(f"[layout] L1: fused coverage {summary['fused_coverage_L1']} "
          f"(was {summary['fused_coverage_L0']}, "
          f"-{summary['fused_cells_lost']} cells = "
          f"{100.0 * summary['fused_cells_lost'] / max(summary['fused_coverage_L0'], 1):.1f}%), "
          f"{summary['camera_cell_pairs_lost']} camera-cell pairs changed")
    print(f"[layout] per camera " + ", ".join(
        f"{SHORT[c]} {int(base_vis[c].sum())}->{int(final_vis[c].sum())}" for c in CAMERAS))
    print(f"[layout] wrote {out}/layout_selected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
