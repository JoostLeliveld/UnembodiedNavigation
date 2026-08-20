#!/usr/bin/env python3
"""Where to put a second camera in the development warehouse, and why.

The two-world rule keeps method development out of `warehouse_full_4cam`, which evaluates
frozen methods. Fusion cannot be developed with one camera, so the answer is a two-camera
DEVELOPMENT world in the aws warehouse -- one extra `<include>` -- rather than developing
in the frozen evaluation world. That preserves the rule's purpose (never tune on the set
you evaluate on) while making fusion work possible.

Placing the camera by eye would waste captures, so this scores candidate mounts over the
drivable floor on the three things that decide whether a pair is useful:

  both-visible    can BOTH cameras see where the robot meets the floor? Ray-tested
                  against every collision box in the world file, the same test that
                  predicted the occlusion captures to within 6 points.
  crossing angle  the angle between the two sightlines at that floor point. This is the
                  variable the open question is about -- geometry says perpendicular
                  separates two cameras' biases best and opposite is degenerate, which
                  contradicts the recorded "opposite pays" fusion heuristic. A placement
                  that only produces one crossing angle cannot test it.
  angle spread    how much of the useful range a single placement covers, because a
                  placement that sweeps the angle needs fewer captures than one that does
                  not.

    python3 experiments/filter_notebook/place_second_camera.py
"""

from __future__ import annotations

import math

import numpy as np

from check_route_clearance import CAMERA_XYZ, ROBOT_TOP_M, _segment_hits_box, collision_boxes

# Camera A, from the world file: south wall, centre, 4.80 m, pitched 0.92 rad, facing north.
CAMERA_A = np.array(CAMERA_XYZ)

# Inner wall faces, read from warehouse_aws.world.sdf (wall centres +- half thickness).
WALL = {"east": 5.42, "west": -5.92, "north": 4.92, "south": -5.52}

# Candidate mounts, all at camera A's height so the pair differs in bearing and nothing
# else. Yaw points each camera at the middle of the floor.
CANDIDATES = {
    "east wall, centre":   (WALL["east"], -0.30, 4.80),
    "east wall, north end": (WALL["east"], 2.50, 4.80),
    "north wall, centre":  (-0.25, WALL["north"], 4.80),
    "west wall, centre":   (WALL["west"], -0.30, 4.80),
    "north-east corner":   (WALL["east"] - 0.6, WALL["north"] - 0.6, 4.80),
}

# The floor a robot can actually drive, sampled where it is clear of every obstacle.
EXTENT = (-5.4, 5.2, -5.2, 4.6)
ROBOT_HALF_DIAGONAL_M = 0.115


def drivable(boxes, nx=70, ny=70):
    """Floor points a robot could stand on, clear of every collision box."""
    xs = np.linspace(EXTENT[0], EXTENT[1], nx)
    ys = np.linspace(EXTENT[2], EXTENT[3], ny)
    out = []
    for x in xs:
        for y in ys:
            clear = True
            for box in boxes:
                if box["lo"][2] > ROBOT_TOP_M:
                    continue
                dx = max(box["lo"][0] - x, 0.0, x - box["hi"][0])
                dy = max(box["lo"][1] - y, 0.0, y - box["hi"][1])
                if math.hypot(dx, dy) <= ROBOT_HALF_DIAGONAL_M + 0.05:
                    clear = False
                    break
            if clear:
                out.append((x, y))
    return np.asarray(out)


def sees_contact_point(camera_xyz, x, y, boxes) -> bool:
    """Can this camera see where the robot meets the floor? (not merely its top)"""
    return not any(_segment_hits_box(np.asarray(camera_xyz),
                                     np.array([x, y, 0.02]), b) for b in boxes)


def crossing_angle_deg(a_xyz, b_xyz, x, y) -> float:
    va = np.array([x - a_xyz[0], y - a_xyz[1]])
    vb = np.array([x - b_xyz[0], y - b_xyz[1]])
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def score(name, b_xyz, floor, boxes) -> dict:
    both, angles = 0, []
    a_only = 0
    for x, y in floor:
        sa = sees_contact_point(CAMERA_A, x, y, boxes)
        sb = sees_contact_point(b_xyz, x, y, boxes)
        if sa and sb:
            both += 1
            angles.append(crossing_angle_deg(CAMERA_A, b_xyz, x, y))
        elif sa:
            a_only += 1
    angles = np.asarray(angles)
    n = len(floor)
    if not len(angles):
        return {"name": name, "both_pct": 0.0}
    # how much of the 30-150 deg band a single placement covers, in 20 deg buckets
    buckets = np.unique(np.clip(angles, 20, 170) // 20)
    return {
        "name": name, "pos": b_xyz,
        "both_pct": 100 * both / n, "a_only_pct": 100 * a_only / n,
        "angle_median": float(np.median(angles)),
        "angle_p10": float(np.percentile(angles, 10)),
        "angle_p90": float(np.percentile(angles, 90)),
        "near_perpendicular_pct": float(100 * np.mean((angles > 70) & (angles < 110))),
        "near_opposite_pct": float(100 * np.mean(angles > 150)),
        "buckets_covered": int(len(buckets)),
    }


if __name__ == "__main__":
    boxes = collision_boxes()
    floor = drivable(boxes)
    print(f"{len(floor)} drivable floor points sampled from {len(boxes)} collision boxes\n")
    print(f"{'candidate mount':<24}{'both see it':>13}{'crossing angle p10/med/p90':>30}"
          f"{'perp':>8}{'opp':>7}{'spread':>8}")
    rows = [score(name, np.array(pos), floor, boxes) for name, pos in CANDIDATES.items()]
    for r in sorted(rows, key=lambda q: -q.get("both_pct", 0)):
        if not r.get("both_pct"):
            print(f"{r['name']:<24}  sees nothing in common with camera A")
            continue
        band = f"{r['angle_p10']:.0f} / {r['angle_median']:.0f} / {r['angle_p90']:.0f}"
        print(f"{r['name']:<24}{r['both_pct']:>12.0f}%{band:>30}"
              f"{r['near_perpendicular_pct']:>7.0f}%{r['near_opposite_pct']:>6.0f}%"
              f"{r['buckets_covered']:>8}")
    print("\nboth see it = share of drivable floor where BOTH cameras see the robot's")
    print("              contact point -- the only places a pair can be fused at all.")
    print("perp / opp  = share of that floor near 90 deg / beyond 150 deg crossing.")
    print("spread      = how many 20 deg crossing-angle buckets one placement covers;")
    print("              higher means fewer captures needed to sweep the angle.")
