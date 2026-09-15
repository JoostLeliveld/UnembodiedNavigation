"""Hard safety gates for full-route (A -> B) global route selection.

Why this module exists
----------------------
The runtime planner's feasibility check (`base_planner._trajectory_plan_diagnostics`)
models the robot as a point with a scalar inflation radius
(`robot_collision_radius_m`). That is adequate to shape a short-horizon MPC
rollout, but it cannot answer the question a *global route selector* must answer:

    "can the complete rigid body drive this route from A to B --
     along the straights, around the corners, and through the in-place turns?"

Here the robot is modelled as an oriented rectangle (default 0.80 x 0.55 m) and
every pose of the executed trajectory -- including the poses swept during an
in-place rotation -- is checked against

  * the obstacle geometry (no body/obstacle contact), and
  * the driveable-region union (no departure from the driveable region).

Safety is a *hard constraint*: a route that fails any gate is rejected before any
objective value is compared, so no objective weighting can ever buy an unsafe
route (ground rule 8).

All checks are geometric and condition-neutral: they do not read q, R, the GP
artifact, or any planner weight, so they are identical across every condition in
a comparison (ground rule 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from unav_common.occlusion_geometry import (
    AxisAlignedPrism,
    scene_from_json,
    signed_distance_to_union_xy,
)


# Phase labels for the reference execution model (see global_route_selector).
PHASE_STRAIGHT = "straight"
PHASE_TURN_IN_PLACE = "turn_in_place"


@dataclass(frozen=True)
class RobotFootprint:
    """Rigid rectangular robot body, expressed in the body frame.

    `length_m` runs along the heading axis, `width_m` across it. The default is
    the 0.80 x 0.55 m body the global planner must be able to drive.
    """

    length_m: float = 0.80
    width_m: float = 0.55

    @property
    def half_length_m(self) -> float:
        return 0.5 * float(self.length_m)

    @property
    def half_width_m(self) -> float:
        return 0.5 * float(self.width_m)

    @property
    def circumradius_m(self) -> float:
        """Radius of the disc swept by a full in-place rotation."""
        return float(math.hypot(self.half_length_m, self.half_width_m))

    def corners(self, x: float, y: float, theta: float) -> np.ndarray:
        """Return the 4 body corners (CCW) for a pose, shape (4, 2)."""
        c, s = math.cos(float(theta)), math.sin(float(theta))
        hl, hw = self.half_length_m, self.half_width_m
        local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]], dtype=float)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return local @ rot.T + np.array([float(x), float(y)], dtype=float)

    def sample_points(self, x: float, y: float, theta: float, *, spacing_m: float = 0.05) -> np.ndarray:
        """Sample the body footprint (interior grid incl. boundary) at a pose."""
        hl, hw = self.half_length_m, self.half_width_m
        spacing = max(float(spacing_m), 1e-3)
        n_l = max(2, int(math.ceil(self.length_m / spacing)) + 1)
        n_w = max(2, int(math.ceil(self.width_m / spacing)) + 1)
        us = np.linspace(-hl, hl, n_l)
        vs = np.linspace(-hw, hw, n_w)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        local = np.column_stack([uu.reshape(-1), vv.reshape(-1)])
        c, s = math.cos(float(theta)), math.sin(float(theta))
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return local @ rot.T + np.array([float(x), float(y)], dtype=float)


def _polygon_from_prism(prism: AxisAlignedPrism) -> np.ndarray:
    return np.array(
        [
            [prism.xmin, prism.ymin],
            [prism.xmax, prism.ymin],
            [prism.xmax, prism.ymax],
            [prism.xmin, prism.ymax],
        ],
        dtype=float,
    )


def _axes_of(poly: np.ndarray) -> np.ndarray:
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-12
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])[keep]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals


def _segment_point_distance(seg_a: np.ndarray, seg_b: np.ndarray, pts: np.ndarray) -> np.ndarray:
    v = seg_b - seg_a
    denom = float(v @ v)
    if denom < 1e-12:
        return np.linalg.norm(pts - seg_a, axis=1)
    t = np.clip(((pts - seg_a) @ v) / denom, 0.0, 1.0)
    closest = seg_a[None, :] + t[:, None] * v[None, :]
    return np.linalg.norm(pts - closest, axis=1)


def convex_polygon_clearance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Signed clearance between two convex polygons.

    Positive: the gap between them (exact Euclidean distance). Negative: minus
    the SAT penetration depth. Zero: touching. Used for body-vs-obstacle checks,
    where `poly_a` is the oriented robot body and `poly_b` an obstacle footprint.
    """
    poly_a = np.asarray(poly_a, dtype=float)
    poly_b = np.asarray(poly_b, dtype=float)
    axes = np.vstack([_axes_of(poly_a), _axes_of(poly_b)])
    worst_gap = -np.inf
    for axis in axes:
        pa = poly_a @ axis
        pb = poly_b @ axis
        gap = max(float(pb.min() - pa.max()), float(pa.min() - pb.max()))
        worst_gap = max(worst_gap, gap)
    if worst_gap > 0.0:
        # Disjoint: exact polygon-polygon distance (vertex-to-edge, both ways).
        best = np.inf
        for poly_p, poly_q in ((poly_a, poly_b), (poly_b, poly_a)):
            for i in range(poly_q.shape[0]):
                d = _segment_point_distance(poly_q[i], poly_q[(i + 1) % poly_q.shape[0]], poly_p)
                best = min(best, float(d.min()))
        return float(best)
    # Overlapping: penetration depth is the smallest overlap over all axes.
    depth = np.inf
    for axis in axes:
        pa = poly_a @ axis
        pb = poly_b @ axis
        overlap = min(float(pa.max() - pb.min()), float(pb.max() - pa.min()))
        depth = min(depth, overlap)
    return -float(max(depth, 0.0))


@dataclass
class PoseSafety:
    """Per-pose geometric safety report."""

    obstacle_body_clearance_m: float
    driveable_body_clearance_m: float
    driveable_path_clearance_m: float


@dataclass
class RouteSafetyReport:
    """Aggregate hard-gate verdict for one candidate route."""

    reaches_goal: bool
    terminal_goal_distance_m: float
    obstacle_body_clearance_min_m: float
    driveable_body_clearance_min_m: float
    driveable_path_clearance_min_m: float
    obstacle_body_clearance_min_straight_m: float
    obstacle_body_clearance_min_turn_m: float
    driveable_body_clearance_min_straight_m: float
    driveable_body_clearance_min_turn_m: float
    kinematically_feasible: bool
    gates: dict = field(default_factory=dict)
    failed_gates: tuple = ()

    @property
    def safe(self) -> bool:
        return not self.failed_gates

    @property
    def min_body_clearance_m(self) -> float:
        """Smallest clearance the complete body keeps to any hard boundary."""
        return float(min(self.obstacle_body_clearance_min_m, self.driveable_body_clearance_min_m))

    @property
    def status(self) -> str:
        if self.safe:
            return "SAFE"
        return "UNSAFE:" + ",".join(self.failed_gates)


@dataclass(frozen=True)
class SafetyGateConfig:
    """Which hard gates a route must pass, and with what margins.

    Every gate is on by default, matching the stated planner requirement that the
    complete body must fit, with no obstacle contact and no departure from the
    driveable region.

    `require_body_inside_driveable` is separated out because the declared
    driveable geometry of a world may be a *conservative lane centre band* that
    is narrower than the physically free corridor. Turning it off keeps the
    body/obstacle gate and the path/driveable gate but reports the body/driveable
    clearance as a diagnostic instead of a veto. It is never turned off silently:
    the selector records the gate configuration with every result.
    """

    require_reach_goal: bool = True
    require_body_obstacle_clearance: bool = True
    require_path_inside_driveable: bool = True
    require_body_inside_driveable: bool = True
    goal_radius_m: float = 0.25
    obstacle_clearance_margin_m: float = 0.0
    driveable_clearance_margin_m: float = 0.0
    path_clearance_margin_m: float = 0.0
    footprint_sample_spacing_m: float = 0.05

    def signature(self) -> tuple:
        return (
            bool(self.require_reach_goal),
            bool(self.require_body_obstacle_clearance),
            bool(self.require_path_inside_driveable),
            bool(self.require_body_inside_driveable),
            round(float(self.goal_radius_m), 6),
            round(float(self.obstacle_clearance_margin_m), 6),
            round(float(self.driveable_clearance_margin_m), 6),
            round(float(self.path_clearance_margin_m), 6),
            round(float(self.footprint_sample_spacing_m), 6),
        )


class RouteSafetyModel:
    """Geometric hard-gate evaluator for full candidate routes."""

    def __init__(
        self,
        *,
        driveable_geometry_json: str = "",
        obstacle_geometry_json: str = "",
        footprint: RobotFootprint | None = None,
        gates: SafetyGateConfig | None = None,
    ):
        self.footprint = footprint or RobotFootprint()
        self.gates = gates or SafetyGateConfig()
        self.driveable_prisms: tuple[AxisAlignedPrism, ...] = tuple(
            scene_from_json(driveable_geometry_json).prisms
        ) if str(driveable_geometry_json or "").strip() else ()
        self.obstacle_prisms: tuple[AxisAlignedPrism, ...] = tuple(
            scene_from_json(obstacle_geometry_json).prisms
        ) if str(obstacle_geometry_json or "").strip() else ()
        self._obstacle_polys = [_polygon_from_prism(p) for p in self.obstacle_prisms]
        # Route geometry is condition-neutral, so the same candidate evaluated
        # under several (q, R) conditions or objectives reuses one verdict.
        self._route_cache: dict = {}

    # -- primitive queries -------------------------------------------------

    def obstacle_body_clearance(self, x: float, y: float, theta: float) -> float:
        """Clearance of the complete body to the nearest obstacle (m, <0 = contact)."""
        if not self._obstacle_polys:
            return float("inf")
        body = self.footprint.corners(x, y, theta)
        return float(min(convex_polygon_clearance(body, poly) for poly in self._obstacle_polys))

    def driveable_body_clearance(self, x: float, y: float, theta: float) -> float:
        """Smallest inside-distance of any body point to the driveable boundary.

        Positive: the whole body is inside the driveable union with that margin.
        Negative: some body point is outside the union, by that depth.
        """
        if not self.driveable_prisms:
            return float("inf")
        pts = self.footprint.sample_points(
            x, y, theta, spacing_m=self.gates.footprint_sample_spacing_m
        )
        signed = np.asarray(
            signed_distance_to_union_xy(self.driveable_prisms, pts, keep_in=True), dtype=float
        )
        # signed <= 0 inside the union; inside-depth is -signed.
        return float(np.min(-signed))

    def driveable_path_clearance(self, x: float, y: float) -> float:
        """Inside-distance of the path point (body centre) to the driveable boundary."""
        if not self.driveable_prisms:
            return float("inf")
        signed = float(
            signed_distance_to_union_xy(
                self.driveable_prisms, np.array([[float(x), float(y)]], dtype=float), keep_in=True
            )[0]
        )
        return -signed

    def pose_safety(self, x: float, y: float, theta: float) -> PoseSafety:
        return PoseSafety(
            obstacle_body_clearance_m=self.obstacle_body_clearance(x, y, theta),
            driveable_body_clearance_m=self.driveable_body_clearance(x, y, theta),
            driveable_path_clearance_m=self.driveable_path_clearance(x, y),
        )

    # -- full-route gate ---------------------------------------------------

    def evaluate_route(
        self,
        poses: np.ndarray,
        goal_xy: Sequence[float],
        *,
        phases: Sequence[str] | None = None,
        kinematically_feasible: bool = True,
    ) -> RouteSafetyReport:
        """Apply every hard gate to an executed pose sequence.

        `poses` is (N, 3) [x, y, theta] and must already be sampled finely enough
        that consecutive poses differ by less than the safety resolution; the
        reference execution model in `global_route_selector` guarantees this and
        includes the poses swept during in-place turns.
        """
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if poses.shape[0] == 0:
            raise ValueError("evaluate_route requires at least one pose")
        if phases is None:
            phases = [PHASE_STRAIGHT] * poses.shape[0]
        phases = list(phases)
        if len(phases) != poses.shape[0]:
            raise ValueError("phases must have one entry per pose")

        cache_key = (
            poses.tobytes(),
            tuple(np.asarray(goal_xy, dtype=float).reshape(2).tolist()),
            tuple(phases),
            bool(kinematically_feasible),
            self.gates.signature(),
            (float(self.footprint.length_m), float(self.footprint.width_m)),
        )
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            return cached

        obs_min = float("inf")
        drv_body_min = float("inf")
        drv_path_min = float("inf")
        obs_min_by_phase = {PHASE_STRAIGHT: float("inf"), PHASE_TURN_IN_PLACE: float("inf")}
        drv_min_by_phase = {PHASE_STRAIGHT: float("inf"), PHASE_TURN_IN_PLACE: float("inf")}
        for (x, y, theta), phase in zip(poses, phases):
            ps = self.pose_safety(float(x), float(y), float(theta))
            obs_min = min(obs_min, ps.obstacle_body_clearance_m)
            drv_body_min = min(drv_body_min, ps.driveable_body_clearance_m)
            drv_path_min = min(drv_path_min, ps.driveable_path_clearance_m)
            key = phase if phase in obs_min_by_phase else PHASE_STRAIGHT
            obs_min_by_phase[key] = min(obs_min_by_phase[key], ps.obstacle_body_clearance_m)
            drv_min_by_phase[key] = min(drv_min_by_phase[key], ps.driveable_body_clearance_m)

        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        terminal_distance = float(np.linalg.norm(poses[-1, :2] - goal_xy))
        reaches_goal = terminal_distance <= float(self.gates.goal_radius_m)

        gates = {
            "reaches_goal": bool(reaches_goal) or not self.gates.require_reach_goal,
            "body_obstacle_clearance": (
                obs_min >= self.gates.obstacle_clearance_margin_m
                or not self.gates.require_body_obstacle_clearance
            ),
            "path_inside_driveable": (
                drv_path_min >= self.gates.path_clearance_margin_m
                or not self.gates.require_path_inside_driveable
            ),
            "body_inside_driveable": (
                drv_body_min >= self.gates.driveable_clearance_margin_m
                or not self.gates.require_body_inside_driveable
            ),
            "kinematically_feasible": bool(kinematically_feasible),
        }
        failed = tuple(name for name, ok in gates.items() if not ok)
        report = RouteSafetyReport(
            reaches_goal=bool(reaches_goal),
            terminal_goal_distance_m=terminal_distance,
            obstacle_body_clearance_min_m=obs_min,
            driveable_body_clearance_min_m=drv_body_min,
            driveable_path_clearance_min_m=drv_path_min,
            obstacle_body_clearance_min_straight_m=obs_min_by_phase[PHASE_STRAIGHT],
            obstacle_body_clearance_min_turn_m=obs_min_by_phase[PHASE_TURN_IN_PLACE],
            driveable_body_clearance_min_straight_m=drv_min_by_phase[PHASE_STRAIGHT],
            driveable_body_clearance_min_turn_m=drv_min_by_phase[PHASE_TURN_IN_PLACE],
            kinematically_feasible=bool(kinematically_feasible),
            gates=gates,
            failed_gates=failed,
        )
        self._route_cache[cache_key] = report
        return report
