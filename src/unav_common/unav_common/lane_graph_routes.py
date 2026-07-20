"""Map-derived multistart route-seed generator (condition-neutral, no visibility).

Rationale
---------
The hierarchical planner solves the long-horizon global EFE route ONCE (it does
not replan the global route; see efe_agent_node._plan_once). The L-BFGS-B solve
is nonconvex, so it needs initial guesses that cover the distinct route basins
of the known map. This module generates those seeds by a fixed rule from the
DRIVEABLE map alone -- not hand-drawn, not derived from the GP/detector, and
identical across conditions.

Topology of the AWS world: a central block of rack rows separates two ways to
cross the workspace -- BELOW the racks (the lower main aisle) or ABOVE them (the
mid rack-gap connector band where it spans, otherwise the upper cross-aisle).
Those are the two homotopy classes between any start and goal. For each we emit
lane-centre Manhattan waypoints:

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


def _segment_inside_union(prisms, a: XY, b: XY, *, step: float = 0.10, tol: float = 1e-3) -> bool:
    """True iff the straight segment a->b stays inside the driveable union."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(2, int(np.ceil(float(np.hypot(*(b - a))) / step)) + 1)
    pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
    sd = signed_distance_to_union_xy(tuple(prisms), pts, keep_in=True)
    return bool(np.all(np.asarray(sd, float) <= tol))


def _manhattan(start: XY, goal: XY, y_corridor: float) -> List[XY]:
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    return [(sx, y_corridor), (gx, y_corridor), (gx, gy)]


def _route_valid(prisms, start: XY, goal: XY, y_corridor: float) -> bool:
    """Validate the three lane-centre Manhattan legs against the driveable union."""
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    legs = [((sx, sy), (sx, y_corridor)),          # start aisle -> corridor
            ((sx, y_corridor), (gx, y_corridor)),  # along the corridor
            ((gx, y_corridor), (gx, gy))]          # goal aisle -> goal
    return all(_segment_inside_union(prisms, a, b) for a, b in legs)


def generate_route_seeds(
    driveable_geometry_json: str,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> List[dict]:
    """Return the condition-neutral multistart route seeds for (start, goal).

    Output matches optimizer_initial_routes_json: a list of
    {"name": str, "waypoints": [[x,y], ...]} with at most two entries
    (below / above the rack block). Routes whose lane-centre Manhattan legs
    leave the driveable union are dropped.
    """
    scene = scene_from_json(driveable_geometry_json)
    prisms = tuple(scene.prisms)
    start = (float(start_xy[0]), float(start_xy[1]))
    goal = (float(goal_xy[0]), float(goal_xy[1]))

    centres = _horizontal_corridor_centres(prisms)
    if not centres:
        return []

    # Split corridors into those below the rack block and those above it. The
    # rack block sits between the lowest aisle-adjacent corridor and the top
    # cross-aisle; use y=0 (workspace mid) as a robust below/above split since
    # all lower corridors are well negative and all upper corridors well positive.
    below = [y for y in centres if y < 0.0]
    above = [y for y in centres if y > 0.0]

    routes: List[dict] = []

    # BELOW route: the highest below-corridor that yields a valid traverse
    # (i.e. the main aisle just under the racks, not the further loading apron).
    for y in sorted(below, reverse=True):
        if _route_valid(prisms, start, goal, y):
            routes.append({"name": "below_main_aisle", "waypoints": [list(w) for w in _manhattan(start, goal, y)]})
            break

    # ABOVE route: the LOWEST above-corridor that yields a valid traverse
    # (the mid connector band where it spans start..goal, else the upper cross-aisle).
    for y in sorted(above):
        if _route_valid(prisms, start, goal, y):
            name = "above_connector" if y < 3.0 else "above_cross_aisle"
            routes.append({"name": name, "waypoints": [list(w) for w in _manhattan(start, goal, y)]})
            break

    return routes


def generate_route_seeds_json(driveable_geometry_json: str, start_xy, goal_xy) -> str:
    import json
    return json.dumps(generate_route_seeds(driveable_geometry_json, start_xy, goal_xy))
