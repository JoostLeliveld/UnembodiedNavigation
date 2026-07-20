#!/usr/bin/env python3
"""Generate the floor driveable-region overlay SDF model from the PLANNER prisms.

The overlay shown in the rendered external-camera view must match the real
driveable / no-go geometry, not a hand-tuned outline. This script derives:

  * BLUE  outer driveable/workspace boundary = outline of the planner prism-union
            bbox (from `driveable_geometry_json`).
  * GREEN internal no-go/obstacle zones      = the ACTUAL obstacle footprints from
            the world's `<collision>` geometry (racks, R4 stack, crates), with
            y-contiguous boxes merged per column. This correctly leaves the OPEN
            mid-gaps of R2..R5 (no collision there) DRIVEABLE — only R1, whose mid
            gap is physically filled, is a continuous no-go column.

and emits a `<model name="known_driveable_boundary">` block of thin floor boxes
to paste into the world SDF. No coordinates are hand-tuned: the blue boundary
comes from the planner prisms and the green no-go comes from the collision geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
DEFAULT_CONFIG = REPO / "scripts/visibility_comparison/aws_f86a_camera_xy_config.yaml"
DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
DEFAULT_OUT = REPO / "logs/paper_figures/driveable_overlay_model.sdf.xml"

# Floor-decal style (matches the legacy green_boundary thin boxes).
Z = 0.035
THICK = 0.035   # line thickness (m)
HEIGHT = 0.014  # box z-size (m)
BLUE = "0.05 0.30 0.85 1"
GREEN = "0.05 0.55 0.20 1"


def _load_prisms(config_path: Path) -> list[dict]:
    cfg = yaml.safe_load(config_path.read_text())
    geo = cfg.get("driveable_geometry_json", "")
    payload = json.loads(geo) if geo else {}
    return list(payload.get("prisms") or [])


def _rect_outline_links(name: str, x0: float, x1: float, y0: float, y1: float,
                        color: str, start_idx: int) -> tuple[list[str], int]:
    """Four thin boxes outlining the axis-aligned rectangle [x0,x1]x[y0,y1]."""
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = (x1 - x0), (y1 - y0)
    edges = [
        # (pose_x, pose_y, size_x, size_y)
        (cx, y0, w + THICK, THICK),  # bottom
        (cx, y1, w + THICK, THICK),  # top
        (x0, cy, THICK, h - THICK),  # left
        (x1, cy, THICK, h - THICK),  # right
    ]
    links = []
    idx = start_idx
    for (px, py, sx, sy) in edges:
        links.append(
            f'      <link name="{name}_{idx:02d}"><pose>{px:.4f} {py:.4f} {Z:.3f} 0 0 0</pose>'
            f'<visual name="visual"><geometry><box><size>{sx:.4f} {sy:.4f} {HEIGHT:.3f}</size></box></geometry>'
            f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual></link>'
        )
        idx += 1
    return links, idx


def load_obstacle_footprints(world_path: Path) -> list[tuple[str, float, float, float, float]]:
    """Non-wall, non-ground collision footprints (name, x0, x1, y0, y1)."""
    from unav_common.occlusion_geometry import parse_occlusion_scene_from_world
    scene = parse_occlusion_scene_from_world(str(world_path), geometry_tags=("collision",))
    out = []
    for fp in scene.prisms:
        name = getattr(fp, "name", "")
        area = (fp.xmax - fp.xmin) * (fp.ymax - fp.ymin)
        if "wall" in name or name == "link" or area > 5.0:
            continue  # drop walls and the ground plane
        out.append((name, float(fp.xmin), float(fp.xmax), float(fp.ymin), float(fp.ymax)))
    return out


def _vertical_aisles(prisms):
    return sorted((p for p in prisms if (float(p["ymax"]) - float(p["ymin"])) > 3.0),
                  key=lambda p: float(p["xmin"]))


def compute_nogo_and_connectors(prisms, footprints, *, y_gap_tol: float = 0.15,
                                min_gap: float = 0.30):
    """No-go = the region between adjacent driveable aisles (full column width, so
    it TOUCHES the driveable corridors), spanning the rack band top-to-bottom, but
    SPLIT at the physical rack mid-gap so the open R2..R5 connectors stay driveable.
    Returns (nogo_blocks, connectors) as lists of (x0,x1,y0,y1)."""
    aisles = _vertical_aisles(prisms)
    horiz = [p for p in prisms if (float(p["xmax"]) - float(p["xmin"])) > 6.0]
    band_y0 = max(float(p["ymax"]) for p in horiz if float(p["ymin"]) < 0)
    band_y1 = min(float(p["ymin"]) for p in horiz if float(p["ymin"]) > 0)

    nogo, connectors = [], []
    for a, b in zip(aisles, aisles[1:]):
        cx0, cx1 = float(a["xmax"]), float(b["xmin"])  # column = gap BETWEEN aisles
        if cx1 - cx0 <= 1e-3:
            continue
        # rack collision boxes whose centre-x falls in this column
        ys = sorted((y0, y1) for (n, x0, x1, y0, y1) in footprints
                    if "rack_R" in n and cx0 <= 0.5 * (x0 + x1) <= cx1)
        # merge y-contiguous rack boxes
        merged = []
        for (y0, y1) in ys:
            if merged and y0 <= merged[-1][1] + y_gap_tol:
                merged[-1][1] = max(merged[-1][1], y1)
            else:
                merged.append([y0, y1])
        # find the (single) significant mid-gap between merged rack segments
        gap = None
        for s0, s1 in zip(merged, merged[1:]):
            if s1[0] - s0[1] >= min_gap:
                gap = (s0[1], s1[0]); break
        if gap is None:  # continuous column (e.g. R1) -> one no-go block
            nogo.append((cx0, cx1, band_y0, band_y1))
        else:            # split: lower + upper no-go, mid connector driveable
            nogo.append((cx0, cx1, band_y0, gap[0]))
            nogo.append((cx0, cx1, gap[1], band_y1))
            connectors.append((cx0, cx1, gap[0], gap[1]))
    return nogo, connectors


def build_overlay(prisms: list[dict], footprints) -> tuple[str, dict]:
    if not prisms:
        raise SystemExit("no prisms in driveable_geometry_json")

    # Outer driveable boundary = bbox of the planner prism union.
    ux0 = min(float(p["xmin"]) for p in prisms)
    ux1 = max(float(p["xmax"]) for p in prisms)
    uy0 = min(float(p["ymin"]) for p in prisms)
    uy1 = max(float(p["ymax"]) for p in prisms)

    # No-go = full inter-aisle columns (touch the driveable corridors), split at
    # the physical rack mid-gap so R2..R5 connectors stay driveable.
    nogo, connectors = compute_nogo_and_connectors(prisms, footprints)

    links: list[str] = []
    idx = 0
    blue, idx = _rect_outline_links("driveable_outer", ux0, ux1, uy0, uy1, BLUE, idx)
    links += blue
    for k, (gx0, gx1, gy0, gy1) in enumerate(nogo):
        green, idx = _rect_outline_links(f"nogo_col{k}", gx0, gx1, gy0, gy1, GREEN, idx)
        links += green

    model = (
        '    <model name="known_driveable_boundary"><static>true</static>\n'
        "      <!-- GENERATED by generate_driveable_overlay_sdf.py.\n"
        "           BLUE  = outer driveable/workspace boundary (planner prism-union bbox).\n"
        "           GREEN = no-go region = full inter-aisle columns (touch the driveable\n"
        "                   corridors), split at the physical rack mid-gap so R2..R5\n"
        "                   connectors stay driveable; R1 (filled mid) is one solid column.\n"
        "                   Do not hand-edit; regenerate from prisms + collision geometry. -->\n"
        + "\n".join(links)
        + "\n    </model>"
    )
    summary = {
        "outer_bbox": [ux0, ux1, uy0, uy1],
        "nogo_regions": [[a, b, c, d] for (a, b, c, d) in nogo],
        "connectors": [[a, b, c, d] for (a, b, c, d) in connectors],
        "n_footprints": len(footprints),
    }
    return model, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    prisms = _load_prisms(args.config)
    footprints = load_obstacle_footprints(args.world)
    model, summary = build_overlay(prisms, footprints)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(model + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"outer driveable bbox  x[{summary['outer_bbox'][0]:.3f},{summary['outer_bbox'][1]:.3f}] "
          f"y[{summary['outer_bbox'][2]:.3f},{summary['outer_bbox'][3]:.3f}]")
    print(f"obstacle footprints: {summary['n_footprints']}, "
          f"merged into {len(summary['nogo_regions'])} no-go segments:")
    for k, c in enumerate(summary["nogo_regions"]):
        print(f"  obs{k}: x[{c[0]:.3f},{c[1]:.3f}] y[{c[2]:.3f},{c[3]:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
