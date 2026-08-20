#!/usr/bin/env python3
"""Check a proposed route against EVERY collision box in the world, before capturing.

Two captures were lost on 2026-08-14 to a route placed from a stale comment rather than
from the world file, so this reads the SDF itself. It also reports, for every sampled
point, whether the camera can see the robot's ground contact and its top -- which is what
decides between "no detection", "partially occluded" and "clean", and therefore whether a
route is worth driving for an occlusion study.

    python3 experiments/filter_notebook/check_route_clearance.py            # all proposals
    python3 experiments/filter_notebook/check_route_clearance.py --study    # routes in the study file
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
STUDY = REPO / "experiments/filter_notebook/config/aws_study.yaml"

CAMERA_XYZ = (0.0, -5.5, 4.8)
ROBOT_HALF_DIAGONAL_M = 0.115      # TurtleBot3 Burger, from the URDF meshes
ROBOT_TOP_M = 0.191


def collision_boxes() -> list[dict]:
    """Every axis-aligned collision box in the world, as world-frame min/max corners."""
    root = ET.parse(WORLD).getroot()
    out = []
    for model in root.iter("model"):
        model_pose = [float(v) for v in (model.findtext("pose") or "0 0 0 0 0 0").split()]
        for link in model.iter("link"):
            link_pose = [float(v) for v in (link.findtext("pose") or "0 0 0 0 0 0").split()]
            for collision in link.iter("collision"):
                size = collision.find(".//box/size")
                if size is None:
                    continue
                sx, sy, sz = (float(v) for v in (size.text or "").split())
                col_pose = [float(v) for v in (collision.findtext("pose")
                                               or "0 0 0 0 0 0").split()]
                cx = model_pose[0] + link_pose[0] + col_pose[0]
                cy = model_pose[1] + link_pose[1] + col_pose[1]
                cz = model_pose[2] + link_pose[2] + col_pose[2]
                out.append({
                    "name": f"{model.get('name')}/{link.get('name')}",
                    "lo": np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2]),
                    "hi": np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2]),
                })
    return out


def clearance(point_xy, boxes) -> tuple[float, str]:
    """Distance from a floor point to the nearest box footprint, and which box."""
    best, who = float("inf"), ""
    px, py = float(point_xy[0]), float(point_xy[1])
    for box in boxes:
        # only boxes the robot could actually hit -- anything above its roof is irrelevant
        if box["lo"][2] > ROBOT_TOP_M:
            continue
        dx = max(box["lo"][0] - px, 0.0, px - box["hi"][0])
        dy = max(box["lo"][1] - py, 0.0, py - box["hi"][1])
        d = math.hypot(dx, dy)
        if d < best:
            best, who = d, box["name"]
    return best, who


def _segment_hits_box(a, b, box) -> bool:
    """Slab test: does the segment a->b intersect the axis-aligned box?"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if a[i] < box["lo"][i] or a[i] > box["hi"][i]:
                return False
            continue
        inv = 1.0 / d[i]
        lo = (box["lo"][i] - a[i]) * inv
        hi = (box["hi"][i] - a[i]) * inv
        if lo > hi:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


def visibility(point_xy, boxes) -> str:
    """Can the camera see the robot's ground contact, its top, both or neither."""
    ground = np.array([point_xy[0], point_xy[1], 0.02])
    top = np.array([point_xy[0], point_xy[1], ROBOT_TOP_M])
    cam = np.array(CAMERA_XYZ)
    ground_clear = not any(_segment_hits_box(cam, ground, b) for b in boxes)
    top_clear = not any(_segment_hits_box(cam, top, b) for b in boxes)
    if ground_clear and top_clear:
        return "clean"
    if top_clear:
        return "partial"      # top visible, contact point hidden -> BIASED measurement
    return "hidden"           # nothing visible -> no detection


def check(name: str, points, boxes, *, samples=200) -> dict:
    """Walk a polyline and report the worst clearance and the visibility mixture."""
    pts = np.asarray(points, dtype=float)
    legs = [np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])]
    total = float(sum(legs))
    walk = []
    for a, b in zip(pts, pts[1:]):
        n = max(2, int(samples * np.linalg.norm(b - a) / max(total, 1e-9)))
        walk.extend(a + (b - a) * t for t in np.linspace(0, 1, n))

    worst, worst_who, worst_at = float("inf"), "", None
    counts = {"clean": 0, "partial": 0, "hidden": 0}
    ranges = []
    for p in walk:
        d, who = clearance(p, boxes)
        if d < worst:
            worst, worst_who, worst_at = d, who, p
        counts[visibility(p, boxes)] += 1
        ranges.append(math.hypot(p[0] - CAMERA_XYZ[0], p[1] - CAMERA_XYZ[1]))

    margin = worst - ROBOT_HALF_DIAGONAL_M
    n = len(walk)
    return {
        "name": name, "length_m": total, "margin_m": margin,
        "nearest": worst_who, "nearest_at": worst_at,
        "range_m": (min(ranges), max(ranges)),
        "clean": 100 * counts["clean"] / n,
        "partial": 100 * counts["partial"] / n,
        "hidden": 100 * counts["hidden"] / n,
        "ok": margin > 0.10,
    }


# ---- the proposals, all at y = 1.72 or on the aisle, checked rather than asserted.
PROPOSED = {
    "cross_aisle_full_east":  [(-3.40, 1.72), (4.40, 1.72)],
    "cross_aisle_full_west":  [(4.40, 1.72), (-3.40, 1.72)],
    "cross_aisle_near_east":  [(-3.40, 1.45), (4.40, 1.45)],
    "aisle_then_cross":       [(1.075, -4.60), (1.075, 1.72), (-3.40, 1.72)],
    "cross_aisle_diagonal":   [(-3.40, 1.72), (4.40, 0.00)],
}


def report(rows) -> None:
    print(f"{'route':<24}{'len m':>7}{'margin m':>10}{'range m':>14}"
          f"{'clean%':>8}{'partial%':>9}{'hidden%':>9}  nearest obstacle")
    for r in rows:
        flag = "  " if r["ok"] else "!!"
        print(f"{flag}{r['name']:<22}{r['length_m']:>7.2f}{r['margin_m']:>10.3f}"
              f"{r['range_m'][0]:>7.1f}-{r['range_m'][1]:<6.1f}"
              f"{r['clean']:>8.0f}{r['partial']:>9.0f}{r['hidden']:>9.0f}  {r['nearest']}")
    print("\nmargin = clearance from the robot's 0.115 m half-diagonal; needs > 0.10 m.")
    print("clean = camera sees contact point and top; partial = only the top (BIASED "
          "measurement); hidden = neither (no detection).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="store_true",
                        help="check the routes already in aws_study.yaml instead")
    args = parser.parse_args()

    boxes = collision_boxes()
    print(f"{len(boxes)} collision boxes read from {WORLD.name}\n")

    if args.study:
        routes = yaml.safe_load(STUDY.read_text())["collection"]["routes"]
        rows = []
        for route in routes:
            s = route["start"]
            pts = [(s["x"], s["y"])]
            pts += ([(w["x"], w["y"]) for w in route["waypoints"]] if route.get("waypoints")
                    else [(route["goal"]["x"], route["goal"]["y"])])
            rows.append(check(route["name"], pts, boxes))
    else:
        rows = [check(name, pts, boxes) for name, pts in PROPOSED.items()]
    report(rows)
