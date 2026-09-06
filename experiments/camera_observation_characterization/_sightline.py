#!/usr/bin/env python3
"""Does a rack stand between a camera and the robot, and how much does a pixel cost there?

Two geometric quantities are needed all over this study and are defined once here.

OCCLUSION. The warehouse racks are 5.226 m tall and the cameras hang at 5.00 m, so a camera
cannot see over a rack: a sightline that crosses a rack footprint in plan view is blocked, and
a 2D segment-versus-rectangle test is exact rather than an approximation. The capture ran
without the semantic pass, so `line_of_sight` in `capture_index.csv` is empty and this
geometric test is the only occlusion evidence available. It is a property of the world and the
camera pose alone -- no detector output, no ground-truth error -- so it is legitimate as an
analysis variable and, unlike the robot's true pose, a real deployment could precompute it
from the same CAD the planner already uses.

GROUND SENSITIVITY. A camera at height h looking down at ground range R sees the floor at a
depression angle theta = atan(h / R). Moving the detected box bottom by one pixel moves the
back-projected floor point by

    dR/dpixel = h / (f * sin^2(theta))

which blows up as theta gets shallow. This is the single number that explains why the same
detector is worth centimetres on one camera and decimetres on another.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for rel in ("experiments/deck_figures",):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402

RACK_HEIGHT_M = 5.226      # from warehouse_v2.world.sdf; taller than the 5.00 m camera mounts
CAMERA_HEIGHT_M = 5.00


def rack_footprints():
    """Plan-view rectangles of the storage racks."""
    return [zone for zone in D.layout().zones if zone.kind == "storage"]


def camera_positions() -> dict[str, tuple[float, float]]:
    return {camera.name: (float(camera.x), float(camera.y)) for camera in D.layout().cameras}


def _segment_crosses_rectangle(start, end, zone) -> bool:
    """Liang-Barsky: does the segment enter the axis-aligned rectangle at all?"""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    low, high = 0.0, 1.0
    for numerator, denominator in ((-dx, x0 - zone.xmin), (dx, zone.xmax - x0),
                                   (-dy, y0 - zone.ymin), (dy, zone.ymax - y0)):
        if numerator == 0.0:
            if denominator < 0.0:
                return False
            continue
        crossing = denominator / numerator
        if numerator < 0.0:
            if crossing > high:
                return False
            low = max(low, crossing)
        else:
            if crossing < low:
                return False
            high = min(high, crossing)
    return low < high


def blocked(camera_letter: str, x: float, y: float, *, racks=None, cameras=None) -> bool:
    """True when a rack stands between this camera and this floor point."""
    racks = rack_footprints() if racks is None else racks
    cameras = camera_positions() if cameras is None else cameras
    origin = cameras[camera_letter]
    return any(_segment_crosses_rectangle(origin, (x, y), zone) for zone in racks)


def blocked_mask(letters, xs, ys):
    """Vectorised convenience wrapper; caches the geometry once."""
    racks = rack_footprints()
    cameras = camera_positions()
    return [blocked(letter, x, y, racks=racks, cameras=cameras)
            for letter, x, y in zip(letters, xs, ys)]


def metres_per_pixel(ground_range_m: float, focal_px: float,
                     height_m: float = CAMERA_HEIGHT_M) -> float:
    """How far the floor point moves when the detected box bottom moves one pixel."""
    theta = math.atan2(height_m, max(ground_range_m, 1e-6))
    return height_m / (focal_px * math.sin(theta) ** 2)


def focal_pixels(capture_manifest: dict, world_profiles: dict) -> float:
    width = float(capture_manifest["cameras"][0]["image_width"])
    return (width / 2.0) / math.tan(float(world_profiles["camera_intrinsics"]["fov_h_rad"]) / 2.0)
