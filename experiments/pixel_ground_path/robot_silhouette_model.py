"""Forward silhouette model of the TurtleBot3 Burger, built from its URDF.

Every dimension here is read off `src/sim/robot_description/urdf/turtlebot3_burger.urdf.xacro`
and is therefore a CAD number, not a fitted one.  The frame is `base_footprint`: origin on
the ground under the robot, x forward, z up, which is the frame the recorded poses use.

Why a compound model and not a bounding cylinder: e1 showed a bounding cylinder
(r = half-track, h = overall height) over-predicts the real silhouette by ~4.3 px in width
and ~4.8 px in height on every camera, because the robot is wide only at the wheels
(z <= 0.066) and narrow at the top (the LiDAR, radius 0.055).  A bounding hull is the wrong
shape for predicting a silhouette even though it is the right shape for a footprint.

The single most consequential number is the body's **-0.032 m x offset** from the pose
origin.  It rotates with the robot's yaw, which is not observed at runtime, so it is the
dominant irreducible term in any yaw-blind inversion.
"""

from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np

# --- URDF, base_footprint frame -----------------------------------------------------
BASE_LINK_Z = 0.010            # base_joint origin xyz="0 0 0.010"

BODY_OFFSET_X = -0.032         # base_link visual/collision origin xyz="-0.032 0 ..."
BODY_CENTRE_Z = BASE_LINK_Z + 0.070
BODY_SIZE = (0.140, 0.140, 0.143)

WHEEL_Y = 0.080                # wheel_*_joint origin xyz="0 +-0.08 0.023"
WHEEL_CENTRE_Z = BASE_LINK_Z + 0.023
WHEEL_RADIUS = 0.033
WHEEL_HALF_WIDTH = 0.009       # cylinder length 0.018

LIDAR_OFFSET_X = -0.032        # scan_joint origin xyz="-0.032 0 0.172"
LIDAR_CENTRE_Z = BASE_LINK_Z + 0.172
LIDAR_RADIUS = 0.055
LIDAR_HALF_HEIGHT = 0.01575    # cylinder length 0.0315

# Derived, for the record: overall height and half-track.
OVERALL_HEIGHT = LIDAR_CENTRE_Z + LIDAR_HALF_HEIGHT          # 0.1978
HALF_TRACK = WHEEL_Y + WHEEL_HALF_WIDTH                       # 0.089

_N_RIM = 32
_TH = np.linspace(0.0, 2.0 * math.pi, _N_RIM, endpoint=False)
_C, _S = np.cos(_TH), np.sin(_TH)


def _body_points():
    hx, hy, hz = BODY_SIZE[0] / 2, BODY_SIZE[1] / 2, BODY_SIZE[2] / 2
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append((BODY_OFFSET_X + sx * hx, sy * hy, BODY_CENTRE_Z + sz * hz))
    return pts


def _wheel_points():
    """Rim circles of both wheels.  Axis is along y, so the rims lie in x-z planes."""
    pts = []
    for sy in (-1, 1):
        for face in (-1, 1):
            y = sy * WHEEL_Y + face * WHEEL_HALF_WIDTH
            for c, s in zip(_C, _S):
                pts.append((WHEEL_RADIUS * c, y, WHEEL_CENTRE_Z + WHEEL_RADIUS * s))
    return pts


def _lidar_points():
    """Rim circles of the LiDAR puck.  Axis is along z."""
    pts = []
    for sz in (-1, 1):
        z = LIDAR_CENTRE_Z + sz * LIDAR_HALF_HEIGHT
        for c, s in zip(_C, _S):
            pts.append((LIDAR_OFFSET_X + LIDAR_RADIUS * c, LIDAR_RADIUS * s, z))
    return pts


BODY_LOCAL = np.asarray(_body_points(), dtype=float)
WHEEL_LOCAL = np.asarray(_wheel_points(), dtype=float)
LIDAR_LOCAL = np.asarray(_lidar_points(), dtype=float)
ALL_LOCAL = np.vstack([BODY_LOCAL, WHEEL_LOCAL, LIDAR_LOCAL])

# A rotationally-symmetric stand-in: the yaw-averaged shape, for the yaw-blind estimator.
# Radius of each tier about the POSE ORIGIN, so the body's x offset is inside it.
BOUNDING_TIERS = (
    # (z_low, z_high, radius)
    (0.0, WHEEL_CENTRE_Z + WHEEL_RADIUS, HALF_TRACK),
    (BODY_CENTRE_Z - BODY_SIZE[2] / 2, BODY_CENTRE_Z + BODY_SIZE[2] / 2,
     math.hypot(abs(BODY_OFFSET_X) + BODY_SIZE[0] / 2, BODY_SIZE[1] / 2)),
    (LIDAR_CENTRE_Z - LIDAR_HALF_HEIGHT, OVERALL_HEIGHT,
     abs(LIDAR_OFFSET_X) + LIDAR_RADIUS),
)


# --- the rendered geometry -----------------------------------------------------------
# The semantic mask comes from the VISUAL meshes, not the collision primitives, and the
# two differ where it matters most: the collision cylinder for the LiDAR puts the top of
# the robot at z = 0.198 m while the lds.stl mesh tops out at 0.191 m, and the mesh is not
# centred on the scan frame.  e2 showed the collision-primitive model leaves ~50 mm of
# position error that the detector cannot explain, so the mesh is the model to use.
#
# `show_pose_markers` defaults to false in robot_description.launch.py, so the rendered
# robot is base + two tyres + LiDAR and nothing else.
_MESH_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/sim/robot_description/meshes/turtlebot3_description/meshes"
)
MESH_SCALE = 0.001  # every <mesh> carries scale="0.001 0.001 0.001"

# (relative path, translation in base_link, rotation) resolved from the URDF chain:
#   base_link visual  origin xyz="-0.032 0 0"
#   wheel_*_joint     origin xyz="0 +-0.08 0.023" rpy="-1.57 0 0", visual rpy="1.57 0 0"
#                     -> net rotation is identity, so only the translation survives
#   scan_joint        origin xyz="-0.032 0 0.172", visual origin xyz="-0.045 0 0"
MESH_PARTS = (
    ("bases/burger_base.stl", (-0.032, 0.0, 0.0)),
    ("wheels/left_tire.stl", (0.0, 0.080, 0.023)),
    ("wheels/right_tire.stl", (0.0, -0.080, 0.023)),
    ("sensors/lds.stl", (-0.077, 0.0, 0.172)),
)


def _load_stl(path):
    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:512]:
        verts = [
            [float(v) for v in line.split()[1:]]
            for line in raw.decode("utf-8", "ignore").splitlines()
            if line.split()[:1] == ["vertex"]
        ]
        return np.asarray(verts, dtype=float)
    count = struct.unpack("<I", raw[80:84])[0]
    block = np.frombuffer(raw[84:84 + 50 * count], dtype=np.uint8).reshape(count, 50)
    tris = block[:, 12:48].copy().view(np.float32).reshape(count, 3, 3)
    return tris.reshape(-1, 3).astype(float)


def _build_mesh_cloud():
    clouds = []
    for rel, (tx, ty, tz) in MESH_PARTS:
        v = _load_stl(_MESH_ROOT / rel) * MESH_SCALE
        v[:, 0] += tx
        v[:, 1] += ty
        v[:, 2] += tz + BASE_LINK_Z
        clouds.append(v)
    cloud = np.vstack(clouds)
    return np.unique(np.round(cloud, 7), axis=0)


MESH_LOCAL = _build_mesh_cloud()


def project_points(camera, pts_world):
    """Vectorised pinhole projection of an (n, 3) world array to (n, 2) pixels."""
    rel = pts_world - np.asarray(camera.cam_pos, dtype=float)
    cam = rel @ np.asarray(camera.R, dtype=float).T
    z = cam[:, 2]
    ok = z > 1.0e-9
    px = cam[ok] @ np.asarray(camera.K, dtype=float).T
    return px[:, :2] / px[:, 2:3]


def mesh_silhouette_bbox(camera, x, y, yaw, *, points=None):
    """Image bbox of the RENDERED silhouette, from the visual meshes."""
    pts = MESH_LOCAL if points is None else points
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.empty_like(pts)
    world[:, 0] = x + c * pts[:, 0] - s * pts[:, 1]
    world[:, 1] = y + s * pts[:, 0] + c * pts[:, 1]
    world[:, 2] = pts[:, 2]
    uv = project_points(camera, world)
    if uv.shape[0] == 0:
        return None
    return (float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max()))


def silhouette_bbox(camera, x, y, yaw, *, points=ALL_LOCAL):
    """Image bbox of the robot's silhouette at pose (x, y, yaw).

    Exact for the URDF collision primitives: the image of a convex solid is the convex
    hull of its projected boundary, and every bbox extreme of a box is at a vertex and of
    a cylinder is on a rim.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * points[:, 0] - s * points[:, 1]
    wy = y + s * points[:, 0] + c * points[:, 1]
    us, vs = [], []
    for px, py, pz in zip(wx, wy, points[:, 2]):
        u, v, _ = camera.world_to_pixel(px, py, pz)
        us.append(u)
        vs.append(v)
    return min(us), min(vs), max(us), max(vs)


def tier_bbox(camera, x, y):
    """Yaw-blind bbox: the union of the tier cylinders, which needs no yaw."""
    us, vs = [], []
    for z_low, z_high, radius in BOUNDING_TIERS:
        for z in (z_low, z_high):
            for c, s in zip(_C, _S):
                u, v, _ = camera.world_to_pixel(x + radius * c, y + radius * s, z)
                us.append(u)
                vs.append(v)
    return min(us), min(vs), max(us), max(vs)
