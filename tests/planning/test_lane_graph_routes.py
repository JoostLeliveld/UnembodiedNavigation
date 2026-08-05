from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from planning.planners.base_planner import UnicyclePlannerBase
from unav_common.lane_graph_routes import generate_route_seeds
from unav_common.occlusion_geometry import scene_from_json, signed_distance_to_union_xy
from experiments.core.world_profiles import serialize_driveable_geometry_from_profile

REPO = Path(__file__).resolve().parents[2]


def _geometry(name: str, xmin: float, xmax: float, ymin: float, ymax: float) -> dict:
    return {
        "name": name,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": 0.0,
        "zmax": 1.0,
    }


def test_four_cam_drivable_map_yields_paper1_lane_seeds() -> None:
    """The 4-camera drivable map must be navigable by paper 1's seeding alone.

    Guards the regression that started this: the world shipped with a single
    floor-wide `traversable` rectangle, so `generate_route_seeds` returned
    nothing (and, worse, a floor-wide union makes every Manhattan route
    "valid", including ones straight through the rack rows). No collision
    geometry is passed here, so only the published lane-graph path can satisfy
    this test.
    """
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    world = "warehouse_full_4cam.world.sdf"
    profile = profiles["worlds"][world]
    driveable = serialize_driveable_geometry_from_profile(profile)

    tasks = yaml.safe_load(
        (REPO / "src/experiments/config/tasks.yaml").read_text("utf-8")
    )["tasks"][world]
    for name in ("rob_hardA", "rob_hardB"):
        task = next(t for t in tasks if t["name"] == name)
        seeds = generate_route_seeds(
            driveable,
            (task["start"]["x"], task["start"]["y"]),
            (task["goal"]["x"], task["goal"]["y"]),
        )
        assert seeds, f"no lane-graph seed for {name}"
        seed_names = {seed["name"] for seed in seeds}
        assert "below_south_cross_aisle" in seed_names, (
            f"{name} is missing the early south crossing used by the hard route"
        )
        assert "below_main_aisle" in seed_names
        assert "above_connector" in seed_names


def test_four_cam_lanes_are_inset_from_obstacles_like_paper1() -> None:
    """Lanes must be a conservative corridor, not the raw geometric gap.

    Paper 1 draws its lanes inset a measured median 0.25 m from obstacle faces;
    that inset is where keep_in's safety margin actually comes from. Drawing the
    4-camera lanes flush against the rack faces left nothing after corner
    rounding (0.046 m planned clearance vs the paper's 0.358 m floor).
    """
    profiles = yaml.safe_load(
        (REPO / "src/experiments/config/world_profiles.yaml").read_text("utf-8")
    )
    regions = profiles["worlds"]["warehouse_full_4cam.world.sdf"]["known_2d_regions"]
    lanes = [r for r in regions if r.get("type") == "traversable"]
    # site_boundary is the keep-in envelope, not an obstacle: lanes live inside it.
    blockers = [
        r for r in regions
        if str(r.get("type", "")).startswith("non_driveable")
    ]
    assert lanes, "no traversable lanes declared"

    for lane in lanes:
        for blocker in blockers:
            x_overlap = not (blocker["xmax"] <= lane["xmin"] or blocker["xmin"] >= lane["xmax"])
            y_overlap = not (blocker["ymax"] <= lane["ymin"] or blocker["ymin"] >= lane["ymax"])
            assert not (x_overlap and y_overlap), (
                f"lane {lane['name']} overlaps {blocker['name']}"
            )


def test_route_seed_waypoint_arrival_scales_with_global_control_step() -> None:
    planner = object.__new__(UnicyclePlannerBase)
    planner.horizon = 80
    planner.dt = 0.4
    planner.v_max = 0.6
    planner.v_min = -0.6
    planner.w_max = 1.0
    planner.w_min = -1.0
    planner.process_noise_xy = 0.01
    planner.process_noise_theta = 0.02

    controls = planner._controls_for_waypoints(
        np.array([0.0, 0.0, 0.0]),
        [(1.0, 0.0), (1.0, 1.0), (2.0, 1.0)],
    ).reshape(-1, 2)

    state = np.array([0.0, 0.0, 0.0])
    covariance = np.eye(3) * 1e-6
    for control in controls:
        state, covariance = planner.predict(state, covariance, control)
    assert np.linalg.norm(state[:2] - np.array([2.0, 1.0])) < 0.35


def test_multistart_terminal_gate_prefers_goal_reaching_candidate() -> None:
    planner = object.__new__(UnicyclePlannerBase)
    planner.optimizer_terminal_goal_tolerance_m = 0.25

    assert planner._terminal_goal_feasible(
        {"terminal_goal_distance_pred": 0.25}
    )
    assert not planner._terminal_goal_feasible(
        {"terminal_goal_distance_pred": 12.0}
    )
    assert planner._prefer_candidate(
        candidate_valid=True,
        candidate_goal_feasible=True,
        candidate_cost=1000.0,
        incumbent_valid=True,
        incumbent_goal_feasible=False,
        incumbent_cost=1.0,
    )


def test_multistart_terminal_gate_keeps_safety_ahead_of_goal_completion() -> None:
    planner = object.__new__(UnicyclePlannerBase)
    planner.optimizer_terminal_goal_tolerance_m = 0.25

    assert not planner._prefer_candidate(
        candidate_valid=False,
        candidate_goal_feasible=True,
        candidate_cost=1.0,
        incumbent_valid=True,
        incumbent_goal_feasible=False,
        incumbent_cost=1000.0,
    )
