"""Map-derived multistart route-seed generator (condition-neutral, no visibility).

Rationale
---------
The hierarchical planner solves the long-horizon global EFE route ONCE (it does
not replan the global route; see efe_agent_node._plan_once). The L-BFGS-B solve
is nonconvex, so it needs initial guesses that cover the distinct route basins
of the known map. This module generates those seeds by a fixed rule from the
DRIVEABLE map alone -- not hand-drawn, not derived from the GP/detector, and
identical across conditions.

Topology of the AWS world: horizontal cross-aisles connect the vertical rack
aisles.  More than one cross-aisle can be valid for the same start and goal
(notably the south loading apron and the lower main aisle).  Every valid
cross-aisle is a distinct optimizer basin worth offering.  Collapsing all
negative-y corridors to only the highest one hid the early south crossing used
by the hard routes and forced the robot to remain in the far-west aisle.

For each valid horizontal corridor we emit lane-centre Manhattan waypoints:

    W = [ (x_start, y_corridor), (x_goal, y_corridor), (x_goal, y_goal) ]

i.e. move along the start aisle to the corridor centre-line, run along the
corridor to the goal column, then up/down the goal aisle to the goal.

Every corridor centre-line and its usable x-extent is read from the driveable
geometry, and every emitted segment is validated to lie inside the driveable
union (so a corridor that does not actually connect start and goal is rejected).
"""
from __future__ import annotations

from typing import List, Tuple, Sequence
import numpy as np

from .occlusion_geometry import scene_from_json, signed_distance_to_union_xy

XY = Tuple[float, float]


def _horizontal_corridor_centres(prisms) -> List[float]:
    """Distinct centre-y of horizontal (wide) driveable prisms, low -> high."""
    ys = []
    for p in prisms:
        w = float(p.xmax - p.xmin)
        h = float(p.ymax - p.ymin)
        if w > h:  # horizontal corridor
            ys.append(round(0.5 * (p.ymin + p.ymax), 3))
    out: List[float] = []
    for y in sorted(set(ys)):
        if not out or abs(y - out[-1]) > 0.25:
            out.append(y)
    return out


def _vertical_corridor_centres(prisms) -> List[float]:
    """Distinct centre-x of vertical (tall) driveable prisms, west -> east."""
    xs = []
    for p in prisms:
        w = float(p.xmax - p.xmin)
        h = float(p.ymax - p.ymin)
        if h > w:
            xs.append(round(0.5 * (p.xmin + p.xmax), 3))
    out: List[float] = []
    for x in sorted(set(xs)):
        if not out or abs(x - out[-1]) > 0.25:
            out.append(x)
    return out


def _segment_inside_union(prisms, a: XY, b: XY, *, step: float = 0.10, tol: float = 1e-3) -> bool:
    """True iff the straight segment a->b stays inside the driveable union."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(2, int(np.ceil(float(np.hypot(*(b - a))) / step)) + 1)
    pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
    sd = signed_distance_to_union_xy(tuple(prisms), pts, keep_in=True)
    return bool(np.all(np.asarray(sd, float) <= tol))


def _dedupe(points: Sequence[XY]) -> List[XY]:
    out: List[XY] = []
    for p in points:
        q = (float(p[0]), float(p[1]))
        if not out or float(np.hypot(q[0] - out[-1][0], q[1] - out[-1][1])) > 1e-6:
            out.append(q)
    return out


def _route_for_corridor(
    prisms,
    vertical_centres: Sequence[float],
    start: XY,
    goal: XY,
    y_corridor: float,
) -> List[XY] | None:
    """Build a lane-centre route through one horizontal cross-aisle.

    The old three-leg rule kept the exact start/goal x coordinates for both
    vertical legs. That misses a valid route whenever a rack aisle is locally
    blocked: the robot must first move along its current cross-aisle to an
    adjacent north/south aisle. Candidate vertical lanes are still read only
    from the driveable map.
    """
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])

    start_lanes = sorted(
        {sx, *(float(x) for x in vertical_centres)},
        key=lambda x: (abs(x - sx), x),
    )
    goal_lanes = sorted(
        {gx, *(float(x) for x in vertical_centres)},
        key=lambda x: (abs(x - gx), x),
    )
    for start_x in start_lanes:
        start_ok = (
            _segment_inside_union(prisms, start, (start_x, sy))
            and _segment_inside_union(prisms, (start_x, sy), (start_x, y_corridor))
        )
        if not start_ok:
            continue
        for goal_x in goal_lanes:
            points = _dedupe([
                start,
                (start_x, sy),
                (start_x, y_corridor),
                (goal_x, y_corridor),
                (goal_x, gy),
                goal,
            ])
            if all(
                _segment_inside_union(prisms, a, b)
                for a, b in zip(points, points[1:])
            ):
                return points[1:]
    return None


def generate_route_seeds(
    driveable_geometry_json: str,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> List[dict]:
    """Return condition-neutral lane-graph route seeds for ``start -> goal``.

    Output matches optimizer_initial_routes_json: a list of
    {"name": str, "waypoints": [[x,y], ...]}. Every horizontal corridor whose
    lane-centre Manhattan legs stay inside the driveable union is emitted. The
    ordering is deterministic (low y to high y), and no visibility or
    measurement data enters candidate generation.
    """
    scene = scene_from_json(driveable_geometry_json)
    prisms = tuple(scene.prisms)
    start = (float(start_xy[0]), float(start_xy[1]))
    goal = (float(goal_xy[0]), float(goal_xy[1]))

    centres = _horizontal_corridor_centres(prisms)
    vertical_centres = _vertical_corridor_centres(prisms)
    if not centres:
        return []

    below = [y for y in centres if y < 0.0]
    above = [y for y in centres if y > 0.0]
    valid_below = [
        (y, route)
        for y in below
        if (route := _route_for_corridor(
            prisms, vertical_centres, start, goal, y
        )) is not None
    ]
    valid_above = [
        (y, route)
        for y in above
        if (route := _route_for_corridor(
            prisms, vertical_centres, start, goal, y
        )) is not None
    ]

    routes: List[dict] = []
    for index, (y, waypoints) in enumerate(valid_below):
        if len(valid_below) == 1 or index == len(valid_below) - 1:
            name = "below_main_aisle"
        elif index == 0:
            name = "below_south_cross_aisle"
        else:
            name = f"below_cross_aisle_{index + 1}"
        routes.append({
            "name": name,
            "waypoints": [list(w) for w in waypoints],
        })

    for index, (y, waypoints) in enumerate(valid_above):
        if y < 3.0:
            name = "above_connector"
        elif index == len(valid_above) - 1:
            name = "above_cross_aisle"
        else:
            name = f"above_cross_aisle_{index + 1}"
        routes.append({
            "name": name,
            "waypoints": [list(w) for w in waypoints],
        })

    return routes


def generate_route_seeds_json(driveable_geometry_json: str, start_xy, goal_xy) -> str:
    import json
    return json.dumps(generate_route_seeds(driveable_geometry_json, start_xy, goal_xy))
