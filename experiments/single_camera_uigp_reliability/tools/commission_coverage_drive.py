#!/usr/bin/env python3
"""Serpentine camera-commissioning coverage drive for ``warehouse_aws``.

Standalone rclpy node (run with ``python3 <file>``, NOT a colcon entry point).
It drives the TurtleBot along a serpentine lawnmower path over the whole
drivable region of ``warehouse_aws`` so that a fixed external camera + YOLO
detector + belief EKF observe the robot in camera-strong (near, centred) AND
camera-weak (far, image-edge, shelf-occluded) locations. The resulting
``experiment_logger`` CSVs feed the uncertain-input reliability GP
(build_belief_gp_events.py -> fit_belief_aware_gp.py) UNCHANGED.

The EFE planner is silent for this capture (launch ``enable_mission:=false``);
this external controller owns ``/cmd_vel_raw``. Sampling is therefore agnostic to
the planner that later consumes the map (breaks the map<->planner circularity
documented in plans/DATA_SOURCE_commissioning_drive.md).

Reuse: the proportional waypoint follower + steady-clock safety watchdog +
terminal-stop burst are imported wholesale from the big-warehouse commissioning
driver ``StudyRouteDriver``; the world->spawn-local rigid transform is
``world_to_spawn_local``. Only the *route* (a programmatic serpentine over
``warehouse_aws`` prisms) and the *manifest* (all-exit, this-study schema) are
new here. No ground-truth topic is ever subscribed.

    source install/setup.bash          # only needed for the real drive
    python3 commission_coverage_drive.py --dry-run          # no ROS needed
    python3 commission_coverage_drive.py \
        --command-topic /cmd_vel_raw --odom-topic /odom \
        --spawn-x 3.3 --spawn-y -1.0 --spawn-yaw 0.0 \
        --passes 1 --speed-mps 0.22 \
        --manifest-path <log_dir>/coverage_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
_DRIVE_STUDY_ROUTE = (
    REPO_ROOT
    / "experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py"
)


def _load_drive_study_route():
    """Import the big-warehouse commissioning driver as a module by file path.

    That file is a standalone tool (not a package), so it is loaded by path.
    Its ``import rclpy`` block is wrapped in try/except, so this import succeeds
    even without a sourced ROS workspace (the ``--dry-run`` path needs no ROS).
    """
    if not _DRIVE_STUDY_ROUTE.is_file():
        raise RuntimeError(
            f"Cannot reuse the follower: {_DRIVE_STUDY_ROUTE} does not exist."
        )
    name = "bigwarehouse_drive_study_route"
    spec = importlib.util.spec_from_file_location(name, _DRIVE_STUDY_ROUTE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so @dataclass can resolve cls.__module__ via sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_dsr = _load_drive_study_route()
# Reused wholesale — do NOT reimplement any of these.
StudyRouteDriver = _dsr.StudyRouteDriver              # P-heading follower + watchdog
RouteSafetyLimits = _dsr.RouteSafetyLimits            # steady-clock deadman limits
world_to_spawn_local = _dsr.world_to_spawn_local      # 2D rigid warehouse->odom
route_distance_m = _dsr.route_distance_m              # polyline length
default_runtime_limits_s = _dsr.default_runtime_limits_s


# ---------------------------------------------------------------------------
# warehouse_aws drivable geometry (centerlines).
#
# These centerlines are the mid-lines of the drivable prisms in
# scripts/visibility_comparison/warehouse_visibility_campaign.yaml
# (``driveable_geometry_json``). A robot of radius 0.125 m following a straight
# segment between two waypoints that both lie on the SAME centerline stays
# inside that prism; turns are placed only at aisle<->rail junctions that lie
# inside BOTH prisms. Deliberately reaches the far/edge corners so the log
# contains genuine detection misses, not just hits.
# ---------------------------------------------------------------------------
# Centerlines carry a deliberate WALL MARGIN (>= ~0.3 m clearance beyond the
# robot radius) so that turn overshoot (arrival radius) + command/encoder noise
# cannot push the robot into a wall. The first live drive collided on the west
# service lane at x=-5.125 (only ~0.4 m to the -5.65 wall); these are pulled in.
AISLE_X = {"A1": -3.025, "A2": -0.975, "A3": 1.075, "A4": 3.125}  # aisle prisms 0.9 m wide; centers safe
LOWER_RAIL_Y = -2.44      # lower main aisle y[-2.85,-2.03]: ~0.41 m margin each side
UPPER_RAIL_Y = 4.50       # upper cross aisle y[4.25,4.9]: pulled down from 4.575 for margin
WEST_LANE_X = -4.90       # west service lane x[-5.65,-4.6]: 0.75 m to the -5.65 wall (was -5.125)
APRON_Y = -3.00           # loading apron y[-3.45,-2.85]: pulled up from -3.15 (0.45 m to -3.45 wall)
EAST_APRON_X = 4.70       # east extent along the WIDE apron (prism x to 5.1): 0.4 m margin.
                          # Covers the east side without the narrow east N-S corridor.
MID_CONNECTOR_Y = 1.725   # mid connector row, aisles+connectors continuous at this y
MID_CONNECTOR_X_MIN = -3.025  # A1 centerline (west end of the connector sweep)
# East far-corner route (x[4.65,5.15], only 0.5 m wide) is DROPPED from the default
# pass: 0.125 m robot leaves <0.13 m clearance each side — unsafe unattended. East
# coverage still comes from the apron (x to ~4.5) and aisle A4. Add it later with a
# slow, dedicated traversal if the east column proves undersampled.
ROBOT_RADIUS_M = 0.125

_PRISM_SOURCE = {
    "world": "warehouse_aws.world.sdf",
    "origin": (
        "centerlines of driveable_geometry_json in "
        "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
    ),
    "robot_radius_m": ROBOT_RADIUS_M,
    "aisle_x_centers": dict(AISLE_X),
    "lower_rail_y": LOWER_RAIL_Y,
    "upper_rail_y": UPPER_RAIL_Y,
    "west_lane_x": WEST_LANE_X,
    "apron_y": APRON_Y,
    "mid_connector_y": MID_CONNECTOR_Y,
    "east_far_corner_route": "dropped (0.5 m corridor unsafe unattended)",
}


def build_default_waypoints() -> list[tuple[float, float]]:
    """Build one serpentine coverage pass over ``warehouse_aws`` (warehouse frame).

    Layout (turns only at junctions inside both prisms):
      * preamble: ease onto the A4 centerline from spawn, drop to the lower rail;
      * snake A4 -> A3 -> A2 -> A1 full-length, linked by the lower / upper rails;
      * west service lane full N-S (up to the upper rail, down to the apron);
      * loading apron full E-W to the east lane;
      * east far-corner route full N-S;
      * one mid-connector E-W pass at y = 1.725.
    """
    a1, a2, a3, a4 = (AISLE_X[k] for k in ("A1", "A2", "A3", "A4"))
    pts: list[tuple[float, float]] = [
        # --- preamble: onto A4 centerline, then down to the lower rail ---
        (a4, -1.0),
        (a4, LOWER_RAIL_Y),
        # --- serpentine A4 -> A3 -> A2 -> A1 (rail-linked) ---
        (a4, UPPER_RAIL_Y),        # A4 up (full aisle)
        (a3, UPPER_RAIL_Y),        # upper-rail link to A3
        (a3, LOWER_RAIL_Y),        # A3 down (full aisle)
        (a2, LOWER_RAIL_Y),        # lower-rail link to A2
        (a2, UPPER_RAIL_Y),        # A2 up (full aisle)
        (a1, UPPER_RAIL_Y),        # upper-rail link to A1
        (a1, LOWER_RAIL_Y),        # A1 down (full aisle)
        # --- west service lane (full N-S), pulled to WEST_LANE_X for wall margin ---
        (WEST_LANE_X, LOWER_RAIL_Y),   # lower-rail link out to the west lane
        (WEST_LANE_X, UPPER_RAIL_Y),   # west lane north to the upper rail
        (WEST_LANE_X, APRON_Y),        # west lane south down to the apron
        # --- loading apron full E-W (covers the east side via the WIDE apron) ---
        (EAST_APRON_X, APRON_Y),       # apron west -> east to x=EAST_APRON_X (0.4 m off the 5.1 edge)
        # --- east side up into the lower rail, then back west (wide, safe lanes) ---
        (EAST_APRON_X, LOWER_RAIL_Y),  # up from apron into the lower main aisle at the east
        (a4, LOWER_RAIL_Y),            # west along the lower rail back to A4
        # --- A4 up to the mid-connector row, then connector west to A1 ---
        (a4, MID_CONNECTOR_Y),         # up A4 (lower_main -> A4 aisle, continuous)
        (MID_CONNECTOR_X_MIN, MID_CONNECTOR_Y),  # cross west along the connector row to A1
    ]
    return pts


def apply_passes(
    base: list[tuple[float, float]], passes: int
) -> list[tuple[float, float]]:
    """Repeat the base polyline ``passes`` times, reversing on odd passes.

    Reversing on odd passes gives each cell a repeated outcome from the opposite
    travel direction. Duplicate seam points are dropped so the follower does not
    stall on a zero-length segment.
    """
    if passes < 1:
        raise ValueError("--passes must be >= 1")
    full: list[tuple[float, float]] = list(base)
    for i in range(1, passes):
        segment = list(reversed(base)) if (i % 2 == 1) else list(base)
        # segment[0] coincides with full[-1] at every seam.
        full.extend(segment[1:])
    return full


def _bounds(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }


def _load_waypoints_json(path: Path) -> list[tuple[float, float]]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of [x, y] pairs")
    points: list[tuple[float, float]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(v, (int, float)) for v in item)
        ):
            raise ValueError(f"{path}: each waypoint must be [x, y]; got {item!r}")
        points.append((float(item[0]), float(item[1])))
    return points


def build_manifest(
    *,
    node: Any | None,
    args: argparse.Namespace,
    world_waypoints: list[tuple[float, float]],
    local_waypoints: list[tuple[float, float]],
    safety_limits: Any,
    waypoint_source: str,
    completion: dict[str, Any],
) -> dict[str, Any]:
    from dataclasses import asdict

    return {
        "schema_version": "single_cam_commission_coverage_v1",
        "tool": "commission_coverage_drive.py",
        "world": _PRISM_SOURCE["world"],
        "contains_ground_truth": False,
        "command_topic": str(args.command_topic),
        "odom_topic": str(args.odom_topic),
        "use_sim_time": bool(args.use_sim_time),
        "speed_mps": float(args.speed_mps),
        "passes": int(args.passes),
        "max_angular_radps": float(args.max_angular_radps),
        "heading_gain": float(args.heading_gain),
        "arrival_radius_m": float(args.arrival_radius_m),
        "start_hold_s": float(args.start_hold_s),
        "spawn_pose_warehouse": {
            "x_m": float(args.spawn_x),
            "y_m": float(args.spawn_y),
            "yaw_rad": float(args.spawn_yaw),
            "source": "documented launch task start (route_apron_to_a3_mid); NOT ground truth",
        },
        "prism_source": _PRISM_SOURCE,
        "waypoint_source": waypoint_source,
        "waypoint_count": len(world_waypoints),
        "expected_path_length_m": route_distance_m(local_waypoints),
        "waypoints_warehouse_xy_m": [[float(x), float(y)] for x, y in world_waypoints],
        "waypoints_odom_xy_m": [[float(x), float(y)] for x, y in local_waypoints],
        "warehouse_bounds_m": _bounds(world_waypoints),
        "odom_bounds_m": _bounds(local_waypoints),
        "safety_limits": asdict(safety_limits) if safety_limits is not None else None,
        "completion": completion,
        "created_wall_time_s": time.time(),
    }


def _completion_from_node(node: Any, *, interrupted: bool) -> dict[str, Any]:
    """Assemble the all-exit completion record from the finished driver node.

    Only plain stored attributes are read (no clock access), so this is safe to
    call after the driver has shut its rclpy context down.
    """
    started = getattr(node, "started_at_s", None)
    term_sim = getattr(node, "_termination_sim_time_s", None)
    elapsed_sim = (
        None if (started is None or term_sim is None) else max(0.0, term_sim - started)
    )
    latest_pose = getattr(node, "latest_pose", None)
    final_pose = None
    if latest_pose is not None:
        final_pose = {
            "x_m": float(latest_pose[0]),
            "y_m": float(latest_pose[1]),
            "yaw_rad": float(latest_pose[2]),
        }
    succeeded = bool(getattr(node, "succeeded", False))
    reason = str(getattr(node, "termination_reason", "") or "")
    if interrupted and not reason:
        reason = "keyboard_interrupt"
    return {
        "reason": "route_complete" if succeeded else (reason or "unknown"),
        "succeeded": succeeded,
        "interrupted": bool(interrupted),
        "waypoints_reached": int(getattr(node, "waypoint_index", 0)),
        "elapsed_sim_s": elapsed_sim,
        "elapsed_wall_s": getattr(node, "_termination_wall_elapsed_s", None),
        "final_odom_pose": final_pose,
        "terminal_stops_published": int(getattr(node, "_terminal_stops_published", 0)),
    }


def _print_dry_run(
    *,
    world_waypoints: list[tuple[float, float]],
    local_waypoints: list[tuple[float, float]],
    args: argparse.Namespace,
    waypoint_source: str,
) -> None:
    wb = _bounds(world_waypoints)
    ob = _bounds(local_waypoints)
    length_m = route_distance_m(local_waypoints)
    print("=== commission coverage drive (DRY RUN) ===")
    print(f"  waypoint source : {waypoint_source}")
    print(f"  passes          : {args.passes}")
    print(f"  waypoint count  : {len(world_waypoints)}")
    print(f"  speed           : {args.speed_mps:.3f} m/s")
    print(
        f"  spawn (warehouse): x={args.spawn_x:.3f} y={args.spawn_y:.3f} "
        f"yaw={args.spawn_yaw:.3f} rad"
    )
    print(
        "  warehouse bounds: "
        f"x[{wb['x_min']:.3f}, {wb['x_max']:.3f}]  "
        f"y[{wb['y_min']:.3f}, {wb['y_max']:.3f}]"
    )
    print(
        "  odom bounds     : "
        f"x[{ob['x_min']:.3f}, {ob['x_max']:.3f}]  "
        f"y[{ob['y_min']:.3f}, {ob['y_max']:.3f}]"
    )
    print(f"  total path len  : {length_m:.2f} m")
    print(f"  ideal drive time: {length_m / max(args.speed_mps, 1e-9):.1f} s (sim)")
    print("  polyline (warehouse xy -> odom xy):")
    for idx, ((wx, wy), (ox, oy)) in enumerate(zip(world_waypoints, local_waypoints)):
        print(f"    [{idx:3d}] ({wx:+7.3f}, {wy:+7.3f})  ->  ({ox:+7.3f}, {oy:+7.3f})")
    print("=== no node created; no rclpy.init; contains_ground_truth=false ===")


def _default_manifest_path() -> Path:
    return (
        Path(tempfile.gettempdir())
        / "commission_coverage_drive"
        / "coverage_manifest.json"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-topic", default="/cmd_vel_raw",
                        help="Twist output topic (default /cmd_vel_raw so the sim's "
                             "actuation-noise node applies realistic slip).")
    parser.add_argument("--odom-topic", default="/odom",
                        help="Odometry feedback topic (spawn-local; default /odom).")
    parser.add_argument("--spawn-x", type=float, default=3.3)
    parser.add_argument("--spawn-y", type=float, default=-1.0)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--passes", type=int, default=1,
                        help="Repeat the polyline this many times, reversing on odd passes.")
    parser.add_argument("--speed-mps", type=float, default=0.22,
                        help="Forward speed (default 0.22, matching v_max).")
    parser.add_argument("--waypoints-json", type=Path, default=None,
                        help="Override the default polyline with a JSON list of "
                             "[x, y] warehouse-frame waypoints.")
    parser.add_argument("--max-angular-radps", type=float, default=0.8)
    parser.add_argument("--heading-gain", type=float, default=1.8)
    parser.add_argument("--arrival-radius-m", type=float, default=0.20)
    parser.add_argument("--start-hold-s", type=float, default=2.0)
    parser.add_argument("--initial-odom-timeout-s", type=float, default=120.0,
                        help="Wall seconds to wait for the first odometry before aborting.")
    parser.add_argument("--odom-freshness-timeout-s", type=float, default=5.0,
                        help="Max odometry age (wall & sim) before the deadman fires.")
    parser.add_argument("--max-sim-runtime-s", type=float, default=None,
                        help="Simulation-time deadman; default derives from path length / speed.")
    parser.add_argument("--max-wall-runtime-s", type=float, default=None,
                        help="Steady-wall-clock deadman; default permits RTF down to ~0.25.")
    parser.add_argument("--terminal-stop-publish-count", type=int, default=8,
                        help="Number of zero-Twist stop commands to publish on exit.")
    parser.add_argument("--terminal-stop-period-s", type=float, default=0.1)
    parser.add_argument("--manifest-path", type=Path, default=None,
                        help="Manifest JSON output path (default under a scratch dir).")
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print the polyline, then exit (no node, no rclpy).")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    positive = (args.speed_mps, args.max_angular_radps, args.heading_gain,
                args.arrival_radius_m)
    if any(not math.isfinite(v) or v <= 0.0 for v in positive):
        parser.error("--speed-mps, --max-angular-radps, --heading-gain, "
                     "--arrival-radius-m must be finite and positive")
    if args.passes < 1:
        parser.error("--passes must be >= 1")

    # Build the route (warehouse frame), then the rigid transform to odom frame.
    if args.waypoints_json is not None:
        base_waypoints = _load_waypoints_json(args.waypoints_json)
        waypoint_source = f"waypoints_json:{Path(args.waypoints_json).expanduser()}"
    else:
        base_waypoints = build_default_waypoints()
        waypoint_source = "default_serpentine"
    world_waypoints = apply_passes(base_waypoints, int(args.passes))
    local_waypoints = world_to_spawn_local(
        world_waypoints,
        spawn_x_m=float(args.spawn_x),
        spawn_y_m=float(args.spawn_y),
        spawn_yaw_rad=float(args.spawn_yaw),
    )

    if args.dry_run:
        _print_dry_run(
            world_waypoints=world_waypoints,
            local_waypoints=local_waypoints,
            args=args,
            waypoint_source=waypoint_source,
        )
        return 0

    if _dsr.rclpy is None:
        raise RuntimeError(
            "rclpy is required for the live drive; source the ROS workspace first."
        )
    rclpy = _dsr.rclpy

    default_sim_s, default_wall_s = default_runtime_limits_s(
        local_waypoints,
        speed_mps=float(args.speed_mps),
        start_hold_s=float(args.start_hold_s),
    )
    try:
        safety_limits = RouteSafetyLimits(
            initial_odom_timeout_s=float(args.initial_odom_timeout_s),
            odom_freshness_timeout_s=float(args.odom_freshness_timeout_s),
            max_sim_runtime_s=float(
                args.max_sim_runtime_s if args.max_sim_runtime_s is not None
                else default_sim_s
            ),
            max_wall_runtime_s=float(
                args.max_wall_runtime_s if args.max_wall_runtime_s is not None
                else default_wall_s
            ),
            terminal_stop_publish_count=int(args.terminal_stop_publish_count),
            terminal_stop_period_s=float(args.terminal_stop_period_s),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None else _default_manifest_path()
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    completion_context = {
        "tool": "commission_coverage_drive.py",
        "waypoint_source": waypoint_source,
        "passes": int(args.passes),
        "speed_mps": float(args.speed_mps),
        "contains_ground_truth": False,
    }

    rclpy.init()
    node = StudyRouteDriver(
        waypoints=local_waypoints,
        odom_topic=str(args.odom_topic),
        command_topic=str(args.command_topic),
        speed_mps=float(args.speed_mps),
        max_angular_radps=float(args.max_angular_radps),
        heading_gain=float(args.heading_gain),
        arrival_radius_m=float(args.arrival_radius_m),
        start_hold_s=float(args.start_hold_s),
        use_sim_time=bool(args.use_sim_time),
        safety_limits=safety_limits,
        completion_path=None,  # this tool writes its own all-exit manifest instead
        completion_context=completion_context,
    )

    interrupted = False
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        interrupted = True
        node._begin_termination("keyboard_interrupt", succeeded=False)
    finally:
        if not node.finished:
            if not node.terminating:
                node._begin_termination("spin_ended_before_route_complete", succeeded=False)
            node.publish_remaining_terminal_stops()
            node._finalize_termination()

    completion = _completion_from_node(node, interrupted=interrupted)
    manifest = build_manifest(
        node=node,
        args=args,
        world_waypoints=world_waypoints,
        local_waypoints=local_waypoints,
        safety_limits=safety_limits,
        waypoint_source=waypoint_source,
        completion=completion,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"coverage drive finished: reason={completion['reason']} "
          f"waypoints_reached={completion['waypoints_reached']}/{len(local_waypoints)}")
    print(f"manifest written: {manifest_path}")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if completion["succeeded"]:
        return 0
    return 130 if interrupted else 2


if __name__ == "__main__":
    raise SystemExit(main())
