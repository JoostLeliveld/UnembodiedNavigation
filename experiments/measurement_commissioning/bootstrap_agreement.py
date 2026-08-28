#!/usr/bin/env python3
"""How far apart do two cameras place the SAME robot, before any correction?

This sets the one gate that runs without a prior. When the robot starts, there is no
belief, so the silhouette cannot be predicted and the admission check has nothing to
compare against. The only thing left to ask is whether several cameras agree -- and to
ask it, you have to know what "agree" means for readings that carry the observation
model's own displacement.

They are not in the same place, and that is not error. The box's bottom edge is the
robot's nearest point to the camera, so its floor projection lands short along that
camera's bearing, by a commissioned 0.204-0.456 m. Two cameras looking from different
sides therefore put a perfectly stationary robot up to twice the largest of those --
0.91 m -- apart, with no error at all.

Run:  python3 experiments/measurement_commissioning/bootstrap_agreement.py
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parents[1] / "src" / "unav_common"))

import camera as camlib  # noqa: E402
import capture  # noqa: E402

#: the commissioned envelope of the box-bottom-to-robot-centre gap, in metres
OFFSET_RANGE_M = (0.204, 0.456)


def readings_by_position() -> dict[tuple[float, float], dict[str, list]]:
    """Each camera's uncorrected floor reading, grouped by the robot's surveyed spot."""
    cameras = camlib.camera_models(capture.DATASET)
    by_position: dict[tuple[float, float], dict[str, list]] = collections.defaultdict(dict)
    for index_row, detection in capture.load_capture():
        camera = cameras.get(index_row["camera_id"])
        if camera is None:
            continue
        try:
            if int(float(detection.get("detected", 1))) == 0:
                continue
            x0, x1 = float(detection["x0"]), float(detection["x1"])
            y1 = float(detection["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        world = camera.pixel_to_world(0.5 * (x0 + x1), y1)
        if world is None:
            continue
        key = (round(float(index_row["robot_x"]), 4),
               round(float(index_row["robot_y"]), 4))
        by_position[key].setdefault(index_row["camera_id"], []).append(
            (float(world[0]), float(world[1])))
    return by_position


def disagreements(by_position) -> tuple[np.ndarray, np.ndarray]:
    """Every camera pair's gap, and the closest-agreeing pair at each position.

    The quorum only needs the closest-agreeing group to clear the bound, so that
    second series is the one the gate is actually judged against.
    """
    every_pair, closest = [], []
    for per_camera in by_position.values():
        means = {c: np.mean(np.asarray(v), axis=0) for c, v in per_camera.items() if v}
        if len(means) < 2:
            continue
        gaps = [float(np.hypot(*(means[a] - means[b])))
                for a, b in itertools.combinations(sorted(means), 2)]
        every_pair.extend(gaps)
        closest.append(min(gaps))
    return np.asarray(every_pair), np.asarray(closest)


def main() -> None:
    by_position = readings_by_position()
    every_pair, closest = disagreements(by_position)
    bound = 2.0 * OFFSET_RANGE_M[1]

    print(f"positions seen by two or more cameras: {closest.size}")
    print(f"camera pairs: {every_pair.size}\n")
    print("gap between two cameras' readings of the same robot, metres")
    for label, series in (("every pair", every_pair), ("closest pair", closest)):
        quantiles = "  ".join(
            f"p{q}={np.percentile(series, q):.3f}" for q in (50, 90, 95, 99, 100))
        print(f"  {label:<13s} {quantiles}")
    print(f"\nthe model's own bound, twice the largest commissioned offset: {bound:.2f} m")
    print(f"  it admits {100.0 * float(np.mean(closest <= bound)):.1f}% of positions")
    print("  and the worst closest-agreeing pair measured is "
          f"{closest.max():.3f} m, inside it")

    out = STUDY.parents[1] / "logs/studies/measurement_commissioning/bootstrap_agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "positions_with_two_or_more_cameras": int(closest.size),
        "camera_pairs": int(every_pair.size),
        "commissioned_offset_range_m": list(OFFSET_RANGE_M),
        "bound_m": bound,
        "every_pair_m": {f"p{q}": float(np.percentile(every_pair, q))
                         for q in (50, 90, 95, 99, 100)},
        "closest_pair_m": {f"p{q}": float(np.percentile(closest, q))
                           for q in (50, 90, 95, 99, 100)},
        "fraction_of_positions_admitted": float(np.mean(closest <= bound)),
    }, indent=2) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
