#!/usr/bin/env python3
"""Pre-flight: can each E4 task legally start, finish, and be driven at all?

WHY THIS EXISTS. The 2026-08-15 pilot burned three Gazebo runs discovering that its
task could not be driven. The evidence, all from artifacts:

* the local controller emitted ``driveable_clearance_violation_step_0`` 292 times,
  i.e. it rejected the very first step of the plan;
* the corrective run had to move the start 0.30 m further inside the south boundary;
* the selected global plan advanced 0.66 m of the 13.55 m needed, inside a horizon
  able to cover 18 m, so the optimizer was not short of horizon;
* ``optimizer_terminal_goal_tolerance_m`` was ``None`` in that run, so the terminal
  goal gate was inactive and every candidate counted as goal-feasible.

Together those say the failure was geometric, not an optimizer or observation-model
problem: the robot began inside the keep-in warning band, so no first step was legal.
That is cheap to check without Gazebo, and this checks it.

The keep-in contract is asymmetric on purpose: the global planner is held to a
stricter clearance than the local controller, so the global plan leaves headroom for
local tracking error. A task that only satisfies the local threshold will still fail
at step zero.

Run:
    python3 experiments/availability_paper/e0_online_efe_readiness/check_task_clearance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "src/unav_common"))
from unav_common.occlusion_geometry import signed_distance_to_union_xy  # noqa: E402

CAMPAIGN = HERE.parent / "e4_closed_loop/campaign.yaml"
SELECTED_ROUTES = C.OUT_ROOT / "e3_route_discrimination/e3_selected_routes.json"

#: The global planner's keep-in clearance, and the local controller's. The global
#: value is deliberately stricter; see the module docstring.
GLOBAL_CLEARANCE_M = 0.25
LOCAL_CLEARANCE_M = 0.20
#: Resample step when walking a route, well below one grid cell (9.8 cm).
WALK_STEP_M = 0.04


def clearance_at(points: np.ndarray, driveable_prisms) -> np.ndarray:
    """Distance inside the driveable union; positive means clear of the boundary.

    SIGN CONVENTION — the trap here. ``signed_distance_to_union_xy(..., keep_in=True)``
    returns a value that is **<= 0 inside** the union, so clearance is its negation.
    Using the raw value makes every pose inside the warehouse look like a violation.
    This matches ``solve_offline_routes.driveable_clearance``, which is the reference.
    """

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    signed = signed_distance_to_union_xy(tuple(driveable_prisms), pts, keep_in=True)
    return -np.asarray(signed, dtype=float)


def walk(route: np.ndarray, step_m: float = WALK_STEP_M) -> np.ndarray:
    """Resample a polyline so clearance is checked between waypoints, not just at them."""

    pts = np.asarray(route, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, total, max(2, int(round(total / step_m))))
    return np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(CAMPAIGN))
    args = parser.parse_args()

    campaign = yaml.safe_load(Path(args.campaign).read_text())
    tasks = list(campaign.get("tasks", {}))
    if not tasks:
        raise SystemExit(f"no tasks in {args.campaign}")

    apparatus = C.build_apparatus()
    driveable_prisms = apparatus.ctx["driveable_prisms"]

    routes: dict = {}
    if SELECTED_ROUTES.is_file():
        routes = json.loads(SELECTED_ROUTES.read_text()).get("routes", {}).get("four", {})
    else:
        print(f"note: {SELECTED_ROUTES} not found; route bodies will not be checked\n")

    failures: list[str] = []
    print(
        f"Keep-in clearance pre-flight. Global planner needs >= {GLOBAL_CLEARANCE_M} m, "
        f"local controller >= {LOCAL_CLEARANCE_M} m.\n"
    )
    header = f"{'task':<38}{'start':>9}{'goal':>9}{'route min':>11}{'verdict':>10}"
    print(header)
    print("-" * len(header))

    for name in tasks:
        spec = apparatus.tasks.get(name)
        if spec is None:
            failures.append(f"{name}: not in the world task table")
            print(f"{name:<38}{'—':>9}{'—':>9}{'—':>11}{'MISSING':>10}")
            continue

        start = np.asarray([float(spec["start"]["x"]), float(spec["start"]["y"])])
        goal = np.asarray([float(spec["goal"]["x"]), float(spec["goal"]["y"])])
        c_start = float(clearance_at(start, driveable_prisms)[0])
        c_goal = float(clearance_at(goal, driveable_prisms)[0])

        route_min = float("nan")
        task_routes = routes.get(name, {})
        if task_routes:
            mins = []
            for pts in task_routes.values():
                arr = np.asarray(pts, dtype=float)
                if len(arr) >= 2:
                    mins.append(float(np.min(clearance_at(walk(arr), driveable_prisms))))
            if mins:
                route_min = float(np.min(mins))

        bad = []
        if c_start < GLOBAL_CLEARANCE_M:
            bad.append(f"start {c_start:.3f} m")
        if c_goal < GLOBAL_CLEARANCE_M:
            bad.append(f"goal {c_goal:.3f} m")
        if np.isfinite(route_min) and route_min < GLOBAL_CLEARANCE_M:
            bad.append(f"route {route_min:.3f} m")
        verdict = "OK" if not bad else "FAIL"
        if bad:
            failures.append(f"{name}: " + ", ".join(bad) + f" < {GLOBAL_CLEARANCE_M} m")

        print(
            f"{name:<38}{c_start:>9.3f}{c_goal:>9.3f}"
            f"{route_min if np.isfinite(route_min) else float('nan'):>11.3f}{verdict:>10}"
        )

    print()
    if failures:
        print("PRE-FLIGHT FAILED. Fix these before spending campaign time:")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nA pose below the global threshold will be rejected at step zero by the\n"
            "local controller, exactly as in the pilot. Move the pose inward or choose\n"
            "a different task; do NOT lower the clearance to make the run start."
        )
        raise SystemExit(1)
    print("PRE-FLIGHT PASSED: every task start, goal and route clears the global contract.")


if __name__ == "__main__":
    main()
