#!/usr/bin/env python3
"""Ground-truth line-of-sight oracle for the dynamic four-camera warehouse.

For every floor cell and every camera this answers one question: *could this
camera see a robot standing here, right now?*  It answers it from simulator
geometry — the CAD prisms behind the world SDF plus the exact pose of whatever
obstacle is currently spawned — not from anything a perception stack produced.

That makes every array in here EVALUATION-ONLY.  It is the yardstick a learned
visibility model is scored against; feeding it back into a model, a planner or a
filter would be scoring an answer against itself.  Nothing under ``src/`` may
import this module, and ``verify_acceptance.py`` checks that.

Cell codes in the returned grid
-------------------------------
``VISIBLE``     the camera has an unblocked sight-line to the target point
``OCCLUDED``    the point is inside the image but something blocks the sight-line
``OUT_OF_FOV``  the point does not project inside the image at all
``OCCUPIED``    the cell centre is inside solid geometry — no robot can stand there

"Visible" therefore means ``grid == VISIBLE``; the other three codes record *why*
a cell is not visible, which is what makes an occlusion event legible.
"""

from __future__ import annotations

import math
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PATHS_FILE = _HERE.parents[1] / "scripts" / "shared" / "paths.py"
_PATHS_NAME = "_dynamic_world_oracle_paths"
_paths = sys.modules.get(_PATHS_NAME)
if _paths is None:
    _spec = importlib.util.spec_from_file_location(_PATHS_NAME, _PATHS_FILE)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load repository paths helper from {_PATHS_FILE}")
    _paths = importlib.util.module_from_spec(_spec)
    sys.modules[_PATHS_NAME] = _paths
    _spec.loader.exec_module(_paths)
elif Path(getattr(_paths, "__file__", "")).resolve() != _PATHS_FILE.resolve():
    raise ImportError(f"{_PATHS_NAME} resolves to an unexpected module")
repo_root = _paths.repo_root

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "src" / "reliability"))

from unav_common.occlusion_geometry import (  # noqa: E402
    AxisAlignedPrism,
    parse_collision_scene_from_world,
    parse_occlusion_scene_from_world,
)
from reliability.projection import camera_model_from_world  # noqa: E402

OCCLUDED = 0
VISIBLE = 1
OUT_OF_FOV = 2
OCCUPIED = 3

CELL_CODE_MEANING = {
    OCCLUDED: "in the image, but geometry blocks the sight-line",
    VISIBLE: "unblocked sight-line from this camera",
    OUT_OF_FOV: "does not project inside this camera's image",
    OCCUPIED: "cell centre is inside solid geometry",
}


@dataclass(frozen=True)
class FloorGrid:
    """Cell-centre grid over the operating floor, in metres."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    resolution_m: float

    @property
    def x_centres(self) -> np.ndarray:
        n = int(round((self.xmax - self.xmin) / self.resolution_m))
        return self.xmin + (np.arange(n) + 0.5) * self.resolution_m

    @property
    def y_centres(self) -> np.ndarray:
        n = int(round((self.ymax - self.ymin) / self.resolution_m))
        return self.ymin + (np.arange(n) + 0.5) * self.resolution_m

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` — row index is y, column index is x, as for an image."""
        return (self.y_centres.size, self.x_centres.size)

    def points_at_height(self, z: float) -> np.ndarray:
        """``(ny*nx, 3)`` target points in row-major (y, then x) order."""
        gx, gy = np.meshgrid(self.x_centres, self.y_centres)
        return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, float(z))])

    def to_dict(self) -> dict:
        ny, nx = self.shape
        return {
            "xmin": self.xmin, "xmax": self.xmax,
            "ymin": self.ymin, "ymax": self.ymax,
            "resolution_m": self.resolution_m,
            "nx": nx, "ny": ny,
            "cell_order": "row-major, row index = y ascending, column index = x ascending",
        }


def _round_nm(value: float) -> float:
    """Round a recorded pose/extent to nanometres (or nanoradians).

    A gravity-free body that has been commanded to a pose still reads back with
    solver noise around 1e-13 m, and that noise differs between two runs of the
    same scenario. Recording it verbatim would make the dataset non-reproducible
    over a quantity nine orders of magnitude below the 0.25 m grid cell it feeds.
    Rounding here throws away only that noise.
    """
    return round(float(value), 9) + 0.0  # +0.0 normalises -0.0 to 0.0


@dataclass(frozen=True)
class ObstacleBox:
    """One spawned obstacle, as the world-frame prisms it actually occupies."""

    entity: str
    model: str
    prisms: tuple[AxisAlignedPrism, ...]
    pose: dict
    aabb_is_exact: bool

    @property
    def aabb(self) -> AxisAlignedPrism:
        return AxisAlignedPrism(
            name=self.entity,
            xmin=min(p.xmin for p in self.prisms), xmax=max(p.xmax for p in self.prisms),
            ymin=min(p.ymin for p in self.prisms), ymax=max(p.ymax for p in self.prisms),
            zmin=min(p.zmin for p in self.prisms), zmax=max(p.zmax for p in self.prisms),
        )

    def to_dict(self) -> dict:
        bound = self.aabb
        return {
            "entity": self.entity,
            "model": self.model,
            "pose": {k: _round_nm(v) for k, v in self.pose.items()},
            "world_aabb": {
                "xmin": _round_nm(bound.xmin), "xmax": _round_nm(bound.xmax),
                "ymin": _round_nm(bound.ymin), "ymax": _round_nm(bound.ymax),
                "zmin": _round_nm(bound.zmin), "zmax": _round_nm(bound.zmax),
            },
            "aabb_is_exact": self.aabb_is_exact,
            "parts": [
                {
                    "name": p.name,
                    "xmin": _round_nm(p.xmin), "xmax": _round_nm(p.xmax),
                    "ymin": _round_nm(p.ymin), "ymax": _round_nm(p.ymax),
                    "zmin": _round_nm(p.zmin), "zmax": _round_nm(p.zmax),
                }
                for p in self.prisms
            ],
        }


def parts_from_model_sdf(model_sdf: str | Path, model_name: str) -> tuple[AxisAlignedPrism, ...]:
    """Model-frame collision prisms of a spawnable obstacle, straight from its SDF.

    An obstacle is not a box.  The pallet's load is narrower than its base and the
    forklift is mostly air between a thin mast and two floor-level forks, so
    bounding either one with a single box makes the oracle claim occlusions the
    renderer does not agree with — measured at 29% false occlusions for the
    pallet before this existed.  Reading the parts back out of the generated model
    SDF keeps the oracle's geometry and the simulator's geometry the same object.
    """
    scene = parse_occlusion_scene_from_world(
        str(model_sdf), model_name=model_name, geometry_tags=("collision",))
    if not scene.prisms:
        raise ValueError(f"no collision geometry for model {model_name!r} in {model_sdf}")
    return scene.prisms


def parts_from_extent(extent_xyz, *, name: str = "box") -> tuple[AxisAlignedPrism, ...]:
    """A single model-frame prism, for an obstacle that really is one box."""
    ex, ey, ez = (float(v) for v in extent_xyz)
    return (AxisAlignedPrism(name=name, xmin=-ex / 2, xmax=ex / 2,
                             ymin=-ey / 2, ymax=ey / 2, zmin=0.0, zmax=ez),)


def place_obstacle(entity: str, model: str, parts, pose: dict) -> ObstacleBox:
    """Put model-frame ``parts`` into the world at ``pose``.

    Yaw is exact at multiples of 90 deg — each part's extents simply swap.  At any
    other yaw each part is replaced by the axis-aligned bound of its rotated
    footprint, which over-states it slightly; ``aabb_is_exact`` records which of
    the two you got, so a scenario cannot quietly inherit a conservative bound.
    """
    yaw = float(pose.get("yaw", 0.0))
    quarter_turns = yaw / (math.pi / 2.0)
    exact = abs(quarter_turns - round(quarter_turns)) < 1.0e-9
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x, y, z = float(pose["x"]), float(pose["y"]), float(pose.get("z", 0.0))

    placed = []
    for part in parts:
        corners = [
            (cx * cos_yaw - cy * sin_yaw, cx * sin_yaw + cy * cos_yaw)
            for cx in (part.xmin, part.xmax) for cy in (part.ymin, part.ymax)
        ]
        placed.append(AxisAlignedPrism(
            name=f"{entity}:{part.name}",
            xmin=x + min(c[0] for c in corners), xmax=x + max(c[0] for c in corners),
            ymin=y + min(c[1] for c in corners), ymax=y + max(c[1] for c in corners),
            zmin=z + part.zmin, zmax=z + part.zmax,
        ))
    return ObstacleBox(entity=entity, model=model, prisms=tuple(placed),
                       pose=dict(pose), aabb_is_exact=exact)


@dataclass
class OracleScene:
    """Static world geometry plus the cameras that look at it."""

    world_sdf: Path
    cameras: dict = field(default_factory=dict)
    static_prisms: tuple[AxisAlignedPrism, ...] = ()

    @classmethod
    def from_world(
        cls,
        world_sdf: str | Path,
        camera_ids: list[str],
        *,
        static_models: tuple[str, ...] = ("warehouse_walls", "warehouse_rack_occluders"),
    ) -> "OracleScene":
        """Cameras plus everything in the world that can block a sight-line.

        Both the collision boxes *and* the visual-only boxes of those models
        count.  A camera does not care whether a thing has physics: the blue
        rails and tan shelf boxes the generator puts on top of each rack are
        emitted with no collision, and leaving them out made the oracle call
        cells visible that Gazebo's own depth buffer showed were grazed by a rail
        — 3.7% of visible cells, three quarters of them with the renderer
        stopping between 2.09 m and 2.40 m, which is precisely rack-top height.
        """
        world_sdf = Path(world_sdf)
        solid = parse_collision_scene_from_world(str(world_sdf), model_names=static_models)
        drawn = parse_occlusion_scene_from_world(
            str(world_sdf), model_name=static_models, geometry_tags=("visual",))
        seen, prisms = set(), []
        for prism in (*solid.prisms, *drawn.prisms):
            key = (round(prism.xmin, 6), round(prism.xmax, 6), round(prism.ymin, 6),
                   round(prism.ymax, 6), round(prism.zmin, 6), round(prism.zmax, 6))
            if key in seen:
                continue          # walls carry an identical collision and visual box
            seen.add(key)
            prisms.append(prism)
        cameras = {
            camera_id: camera_model_from_world(world_sdf, include_name=camera_id)
            for camera_id in camera_ids
        }
        return cls(world_sdf=world_sdf, cameras=cameras, static_prisms=tuple(prisms))


def _prism_bounds(prisms) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([[p.xmin, p.ymin, p.zmin] for p in prisms], dtype=float)
    hi = np.array([[p.xmax, p.ymax, p.zmax] for p in prisms], dtype=float)
    return lo, hi


def segments_hit_any_prism(origin: np.ndarray, targets: np.ndarray, prisms) -> np.ndarray:
    """Vectorised slab test: does the segment origin->target[i] cross any prism?

    Same predicate as ``unav_common.occlusion_geometry.segment_occluded`` but over
    all targets and prisms at once — a 92x69 grid against 60 prisms is 380k
    segment tests per camera per frame, which is minutes of scalar Python and
    milliseconds here.  ``verify_acceptance.py`` re-checks a random sample
    against the scalar original so the two can never silently disagree.
    """
    if len(prisms) == 0:
        return np.zeros(targets.shape[0], dtype=bool)

    lo, hi = _prism_bounds(prisms)                      # (P, 3)
    o = np.asarray(origin, dtype=float).reshape(1, 1, 3)
    d = (np.asarray(targets, dtype=float)[:, None, :] - o)   # (N, 1, 3) broadcast target-origin

    eps = 1.0e-9
    parallel = np.abs(d) <= eps
    safe_d = np.where(parallel, 1.0, d)
    t0 = (lo[None, :, :] - o) / safe_d                  # (N, P, 3)
    t1 = (hi[None, :, :] - o) / safe_d
    t_lo = np.minimum(t0, t1)
    t_hi = np.maximum(t0, t1)

    # A ray parallel to an axis either sits inside that slab for all t or misses
    # the prism entirely; encode those two cases as an infinite / empty interval.
    origin_inside = (o >= lo[None, :, :] - eps) & (o <= hi[None, :, :] + eps)
    t_lo = np.where(parallel, np.where(origin_inside, -np.inf, np.inf), t_lo)
    t_hi = np.where(parallel, np.where(origin_inside, np.inf, -np.inf), t_hi)

    t_min = np.maximum(t_lo.max(axis=2), 0.0)           # (N, P)
    t_max = t_hi.min(axis=2)
    hit = (t_max >= t_min) & (t_min <= 1.0)
    return hit.any(axis=1)


def points_inside_any_prism(points: np.ndarray, prisms) -> np.ndarray:
    if len(prisms) == 0:
        return np.zeros(points.shape[0], dtype=bool)
    lo, hi = _prism_bounds(prisms)
    p = np.asarray(points, dtype=float)[:, None, :]
    inside = ((p >= lo[None, :, :]) & (p <= hi[None, :, :])).all(axis=2)
    return inside.any(axis=1)


def _projects_into_image(camera, points: np.ndarray) -> np.ndarray:
    """True where a world point lands inside the image and in front of the lens."""
    cam_pts = (np.asarray(points, dtype=float) - camera.cam_pos) @ camera.R.T
    in_front = cam_pts[:, 2] > 1.0e-9
    z = np.where(in_front, cam_pts[:, 2], 1.0)
    u = camera.K[0, 0] * cam_pts[:, 0] / z + camera.K[0, 2]
    v = camera.K[1, 1] * cam_pts[:, 1] / z + camera.K[1, 2]
    inside = (u >= 0) & (u < camera.img_width) & (v >= 0) & (v < camera.img_height)
    return in_front & inside


def visibility_grid(
    camera,
    grid: FloorGrid,
    static_prisms,
    obstacle_prisms=(),
    *,
    target_height_m: float = 0.35,
) -> np.ndarray:
    """Per-cell visibility codes for one camera at one instant, shape ``(ny, nx)``."""
    points = grid.points_at_height(target_height_m)
    all_prisms = list(static_prisms) + list(obstacle_prisms)

    codes = np.full(points.shape[0], OUT_OF_FOV, dtype=np.uint8)
    in_image = _projects_into_image(camera, points)
    codes[in_image] = OCCLUDED

    if in_image.any():
        idx = np.flatnonzero(in_image)
        clear = ~segments_hit_any_prism(camera.cam_pos, points[idx], all_prisms)
        codes[idx[clear]] = VISIBLE

    # Occupancy overrides everything: a cell inside a rack or under a pallet is
    # not a place a robot can be, so calling it "visible" or "occluded" is noise.
    occupied = points_inside_any_prism(points, all_prisms)
    codes[occupied] = OCCUPIED
    return codes.reshape(grid.shape)


def visibility_grids(
    cameras: dict,
    grid: FloorGrid,
    static_prisms,
    obstacle_prisms=(),
    *,
    target_height_m: float = 0.35,
) -> dict:
    return {
        camera_id: visibility_grid(
            camera, grid, static_prisms, obstacle_prisms, target_height_m=target_height_m
        )
        for camera_id, camera in cameras.items()
    }


def any_camera_visible(grids: dict) -> np.ndarray:
    """Cells at least one camera can see — the coverage map an operator cares about."""
    stack = np.stack([g == VISIBLE for g in grids.values()])
    return stack.any(axis=0).astype(np.uint8)


def grid_summary(codes: np.ndarray) -> dict:
    total = int(codes.size)
    return {
        "cells_total": total,
        "cells_visible": int((codes == VISIBLE).sum()),
        "cells_occluded": int((codes == OCCLUDED).sum()),
        "cells_out_of_fov": int((codes == OUT_OF_FOV).sum()),
        "cells_occupied": int((codes == OCCUPIED).sum()),
    }
