#!/usr/bin/env python3
"""Minimal reproduction of the duration-dependent accounting defect.

Setup (deliberately as small as possible)
-----------------------------------------
A synthetic two-band corridor, two safe routes with the SAME endpoints, the
SAME constant q and the SAME constant R, one route strictly longer:

    direct : (0,0) -> (10,0)                       10.0 m
    detour : (0,0) -> (3,0) -> (3,2) -> (7,2) -> (7,0) -> (10,0)   14.0 m

Both stay inside the driveable region with the complete 0.80 x 0.55 m body, both
reach the goal, and nothing about observability distinguishes them: q = 1
everywhere and R is a single constant. A correct planner must pick `direct`.

What the legacy accounting does
-------------------------------
The legacy full-route score is the discounted sum of the per-step EFE ambiguity,
truncated at arrival. Under ET1 the per-step ambiguity is exactly

    A = log(2*pi*e) + 2*log(r)      (for R = r^2 * I_2)

-- a constant that has nothing to do with the route, but that is summed once per
step. With T_detour > T_direct the constant contributes
`A * (T_detour - T_direct)`, so the ranking is decided entirely by the SIGN of A,
and the sign of A is decided by the UNITS the measurement is written in:

    r = 2.5 px                   -> A = +4.67 nats  -> longer route penalised
    r = 2.5/1280 image widths    -> A = -9.64 nats  -> longer route REWARDED

Same robot, same camera, same physics, same q, same R; the winner flips because
the pixel coordinate was renormalised. That is the defect: a route-independent
additive constant becomes a duration term as soon as candidates are compared at
different arrival times.

The corrected objective anchors the ambiguity to a fixed reference measurement
quality and adds an explicit travel baseline, so the constant cancels exactly and
`direct` wins in both unit systems.

Run:
    python3 scripts/planning_validation/minimal_duration_bias_repro.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("src/planning", "src/unav_common"):
    _p = str(REPO_ROOT / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from planning.core.global_route_selector import (  # noqa: E402
    ExecutionModelConfig,
    GlobalRouteSelector,
    RouteCandidate,
    RouteObjectiveConfig,
)
from planning.core.route_safety import (  # noqa: E402
    RobotFootprint,
    RouteSafetyModel,
    SafetyGateConfig,
)

# Two overlapping driveable bands: a long straight corridor and a parallel bay.
CORRIDOR_JSON = json.dumps(
    {
        "model_name": "minimal_two_route_corridor",
        "prisms": [
            {"name": "main", "xmin": -1.0, "xmax": 11.0, "ymin": -1.0, "ymax": 1.0,
             "zmin": 0.0, "zmax": 0.1},
            {"name": "bay", "xmin": 2.0, "xmax": 8.0, "ymin": -1.0, "ymax": 3.0,
             "zmin": 0.0, "zmax": 0.1},
        ],
    }
)

START = np.array([0.0, 0.0, 0.0])
GOAL = np.array([10.0, 0.0])

CANDIDATES = [
    RouteCandidate("direct", ((10.0, 0.0),)),
    RouteCandidate("detour", ((3.0, 0.0), (3.0, 2.0), (7.0, 2.0), (7.0, 0.0), (10.0, 0.0))),
]

CAMERA = {
    "cam_pos": [5.0, -8.0, 6.0],
    "look_at": [5.0, 0.0, 0.0],
    "img_width": 1280,
    "img_height": 720,
    "fov_h_rad": 1.5708,
}


def build_planner(r_uv: float, *, localization_cost_mode: str, use_obs_risk: bool) -> UnicyclePlannerBase:
    """A planner with CONSTANT q (no visibility model) and CONSTANT R."""
    return UnicyclePlannerBase(
        horizon=10,
        dt=0.1,
        v_min=0.0,
        v_max=0.6,
        w_min=-1.0,
        w_max=1.0,
        control_weight=0.0,
        process_noise_xy=0.012,
        process_noise_theta=0.05,
        obs_noise_uv=2.0,
        goal_sigma_uv=2.0,
        risk_weight_obs=1.0,
        ambiguity_weight=1.0,
        optimizer_maxiter=10,
        optimizer_gtol=1e-4,
        optimizer_warm_start=False,
        seed=0,
        camera_params=CAMERA,
        use_obs_risk=use_obs_risk,
        use_ambiguity=True,
        use_visibility_model=False,
        r_visible_uv=r_uv,
        r_miss_uv=r_uv,
        goal_prior_u_std_start=50.0,
        goal_prior_v_std_start=50.0,
        goal_prior_u_std_final=12.0,
        goal_prior_v_std_final=12.0,
        goal_tightening_power=0.9,
        goal_progress_n_steps=90,
        observation_risk_scale=1.0,
        ambiguity_term_scale=1.0,
        discount_gamma=1.0,
        use_nogo_cost=False,
        nogo_mode="keep_in",
        driveable_geometry_json=CORRIDOR_JSON,
        robot_collision_radius_m=0.0,
        localization_cost_mode=localization_cost_mode,
    )


def make_selector(planner, objective: RouteObjectiveConfig) -> GlobalRouteSelector:
    safety = RouteSafetyModel(
        driveable_geometry_json=CORRIDOR_JSON,
        obstacle_geometry_json="",
        footprint=RobotFootprint(0.80, 0.55),
        gates=SafetyGateConfig(goal_radius_m=0.25),
    )
    return GlobalRouteSelector(
        planner,
        safety,
        execution=ExecutionModelConfig(dt=0.1, v_max=0.6, w_max=1.0),
        objective=objective,
        R_reference=planner.R_reference,
    )


def run_case(label: str, r_uv: float, objective: RouteObjectiveConfig, *, use_obs_risk: bool) -> dict:
    mode = "raw_ambiguity" if objective.mode == "legacy_efe" else "anchored_excess"
    planner = build_planner(r_uv, localization_cost_mode=mode, use_obs_risk=use_obs_risk)
    selector = make_selector(planner, objective)
    result = selector.select(CANDIDATES, START, np.diag([0.05 ** 2] * 3), GOAL)
    per_step_ambiguity = math.log(2.0 * math.pi * math.e) + 2.0 * math.log(r_uv)
    print(f"\n### {label}")
    print(f"    per-step ambiguity constant A = {per_step_ambiguity:+.4f} nats "
          f"(R = ({r_uv:g})^2 I)")
    print(f"    {'route':8s} {'safe':6s} {'len[m]':>8s} {'T[s]':>7s} "
          f"{'loc/amb':>12s} {'travel':>9s} {'TOTAL':>12s}  selected")
    for ev in result.evaluations:
        print(f"    {ev.name:8s} {str(ev.safety.safe):6s} {ev.route_length_m:8.2f} "
              f"{ev.travel_time_s:7.1f} {ev.localization_cost:12.4f} "
              f"{ev.travel_cost:9.2f} {ev.total_cost:12.4f}  {'<== ' if ev.selected else ''}"
              f"{ev.selected}")
    winner = result.selected.name if result.selected else "<none>"
    print(f"    -> selects: {winner}")
    return {"label": label, "winner": winner, "A": per_step_ambiguity,
            "rows": [ev.as_row() for ev in result.evaluations]}


def main() -> int:
    print(__doc__)
    print("=" * 78)
    print("Identical constant q (=1) and constant R on BOTH routes; detour is 4 m longer.")
    print("Goal attainment is a hard gate in every case, so no soft goal term is needed")
    print("to make the routes comparable.")

    # Legacy accounting with goal attainment handled as a hard gate: the score is
    # exactly the discounted sum of the raw per-step ambiguity up to arrival.
    legacy = RouteObjectiveConfig(
        mode="legacy_efe", travel_time_weight=0.0, route_length_weight=0.0,
        localization_weight=0.0, goal_risk_weight=0.0, obstacle_weight=0.0,
        discount_gamma=1.0, risk_uses_reference_R=False,
    )
    corrected = RouteObjectiveConfig(
        mode="corrected_v2", travel_time_weight=1.0, localization_weight=1.0,
        goal_risk_weight=0.0, obstacle_weight=0.0, discount_gamma=1.0,
    )

    results = []
    results.append(run_case(
        "LEGACY accounting, measurement in PIXELS (r = 2.5 px)", 2.5, legacy, use_obs_risk=False))
    results.append(run_case(
        "LEGACY accounting, SAME measurement in IMAGE WIDTHS (r = 2.5/1280)",
        2.5 / 1280.0, legacy, use_obs_risk=False))
    results.append(run_case(
        "CORRECTED objective, measurement in PIXELS (r = 2.5 px)", 2.5, corrected, use_obs_risk=False))
    results.append(run_case(
        "CORRECTED objective, SAME measurement in IMAGE WIDTHS (r = 2.5/1280)",
        2.5 / 1280.0, corrected, use_obs_risk=False))

    print("\n" + "=" * 78)
    print("VERDICT")
    legacy_px, legacy_norm, corr_px, corr_norm = results
    print(f"  legacy    : pixels -> {legacy_px['winner']:8s}   image widths -> {legacy_norm['winner']:8s}"
          f"   {'RANKING FLIPS WITH UNITS' if legacy_px['winner'] != legacy_norm['winner'] else 'stable'}")
    print(f"  corrected : pixels -> {corr_px['winner']:8s}   image widths -> {corr_norm['winner']:8s}"
          f"   {'stable' if corr_px['winner'] == corr_norm['winner'] else 'RANKING FLIPS WITH UNITS'}")
    print()
    print("  The legacy score contains a route-independent per-step constant A. Once")
    print("  candidates are compared at different arrival times it becomes A * T, i.e. a")
    print("  duration term whose sign is an artefact of the measurement units. With")
    print("  A < 0 the LONGER route wins on identical q and R.")
    print("  The corrected objective measures ambiguity relative to a fixed reference, so")
    print("  A cancels, the localization cost is exactly 0 at constant perfect")
    print("  observability, and the explicit travel baseline picks the shortest route.")

    ok = (
        legacy_px["winner"] != legacy_norm["winner"]
        and legacy_norm["winner"] == "detour"
        and corr_px["winner"] == "direct"
        and corr_norm["winner"] == "direct"
    )
    print(f"\n  reproduction {'CONFIRMED' if ok else 'NOT REPRODUCED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
