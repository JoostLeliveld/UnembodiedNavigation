"""Regression tests for the corrected global route objective.

Each test corresponds to one of the required guarantees:

  a. constant q + constant R selects the shortest safe route
  b. an unsafe short route is rejected
  c. lowering q on part of the shortest route increases its localization cost
  d. worsening R increases localization cost where measurements are available
  e. a longer observable route wins only when its localization benefit exceeds
     its extra travel
  f. constant offsets in ambiguity cannot change the route ranking
  g. the NumPy and CasADi objectives agree, and so do their candidate rankings

The world is synthetic and tiny so that every number in the assertions can be
checked by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("src/planning", "src/unav_common"):
    _p = str(REPO_ROOT / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planning.core.global_route_selector import (  # noqa: E402
    ExecutionModelConfig,
    GlobalRouteSelector,
    RouteCandidate,
    RouteObjectiveConfig,
)
from planning.core.localization_cost import (  # noqa: E402
    localization_cost_rate_np,
    reference_ambiguity,
)
from planning.core.route_safety import (  # noqa: E402
    RobotFootprint,
    RouteSafetyModel,
    SafetyGateConfig,
)
from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

# Driveable region: a long main corridor plus a parallel bay that overlaps it.
DRIVEABLE_JSON = json.dumps(
    {
        "model_name": "test_two_route_corridor",
        "prisms": [
            {"name": "main", "xmin": -1.0, "xmax": 11.0, "ymin": -1.0, "ymax": 1.0,
             "zmin": 0.0, "zmax": 0.1},
            {"name": "bay", "xmin": 2.0, "xmax": 8.0, "ymin": -1.0, "ymax": 3.0,
             "zmin": 0.0, "zmax": 0.1},
        ],
    }
)

# An obstacle that sits OUTSIDE the driveable region, used by the unsafe route.
OBSTACLE_JSON = json.dumps(
    {
        "model_name": "test_obstacles",
        "prisms": [
            {"name": "block", "xmin": 4.0, "xmax": 6.0, "ymin": -4.0, "ymax": -2.0,
             "zmin": 0.0, "zmax": 1.0},
        ],
    }
)

CAMERA = {
    "cam_pos": [5.0, -8.0, 6.0],
    "look_at": [5.0, 0.0, 0.0],
    "img_width": 1280,
    "img_height": 720,
    "fov_h_rad": 1.5708,
}

START = np.array([0.0, 0.0, 0.0])
GOAL = np.array([10.0, 0.0])
S0 = np.diag([0.05 ** 2, 0.05 ** 2, 0.05 ** 2])

DIRECT = RouteCandidate("direct", ((10.0, 0.0),))
DETOUR = RouteCandidate(
    "detour", ((3.0, 0.0), (3.0, 2.0), (7.0, 2.0), (7.0, 0.0), (10.0, 0.0))
)
# Strictly shorter than `direct` in travel time (no turns, straight line) but it
# leaves the driveable region and clips the obstacle block.
UNSAFE_SHORTCUT = RouteCandidate("unsafe_shortcut", ((5.0, -3.0), (10.0, 0.0)))


def _write_gp_artifact(path: Path, q_field) -> Path:
    """Write a synthetic availability (q) field in the GP artifact schema.

    This is a test fixture, not a fitted model: nothing here refits or replaces
    the commissioned GP.
    """
    xs = np.linspace(-2.0, 12.0, 57)
    ys = np.linspace(-5.0, 5.0, 41)
    grid = np.empty((ys.size, xs.size), dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[j, i] = float(q_field(float(x), float(y)))
    np.savez(
        path,
        xs=xs,
        ys=ys,
        P_mean_map=grid,
        P_conservative_plan_map=grid,
        camera_pos=np.asarray(CAMERA["cam_pos"], dtype=float),
        target_height=np.asarray([0.0], dtype=float),
    )
    return path


def build_planner(
    *,
    r_visible_uv: float = 2.5,
    r_miss_uv: float = 40.0,
    r_reference_uv: float | None = None,
    gp_artifact: str = "",
    localization_cost_mode: str = "anchored_excess",
    horizon: int = 10,
    dt: float = 0.1,
    use_nogo_cost: bool = False,
    discount_gamma: float = 1.0,
) -> UnicyclePlannerBase:
    return UnicyclePlannerBase(
        horizon=horizon,
        dt=dt,
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
        use_obs_risk=True,
        use_ambiguity=True,
        use_visibility_model=bool(gp_artifact),
        visibility_artifact_path=gp_artifact,
        r_visible_uv=r_visible_uv,
        r_miss_uv=r_miss_uv,
        r_reference_uv=r_reference_uv,
        goal_prior_u_std_start=50.0,
        goal_prior_v_std_start=50.0,
        goal_prior_u_std_final=12.0,
        goal_prior_v_std_final=12.0,
        goal_tightening_power=0.9,
        goal_progress_n_steps=90,
        observation_risk_scale=1.0,
        ambiguity_term_scale=1.0,
        discount_gamma=discount_gamma,
        use_nogo_cost=use_nogo_cost,
        nogo_weight=2000.0 if use_nogo_cost else 0.0,
        nogo_mode="keep_in",
        driveable_geometry_json=DRIVEABLE_JSON,
        collision_geometry_json=OBSTACLE_JSON,
        robot_collision_radius_m=0.0,
        localization_cost_mode=localization_cost_mode,
    )


def build_selector(planner, objective: RouteObjectiveConfig, *, footprint=None) -> GlobalRouteSelector:
    safety = RouteSafetyModel(
        driveable_geometry_json=DRIVEABLE_JSON,
        obstacle_geometry_json=OBSTACLE_JSON,
        footprint=footprint or RobotFootprint(0.80, 0.55),
        gates=SafetyGateConfig(goal_radius_m=0.25),
    )
    return GlobalRouteSelector(
        planner,
        safety,
        execution=ExecutionModelConfig(dt=0.1, v_max=0.6, w_max=1.0),
        objective=objective,
        R_reference=planner.R_reference,
    )


CORRECTED = RouteObjectiveConfig(
    mode="corrected_v2",
    travel_time_weight=1.0,
    localization_weight=1.0,
    goal_risk_weight=0.0,
    obstacle_weight=0.0,
    discount_gamma=1.0,
)


# ---------------------------------------------------------------------------
# (a) constant q + constant R -> shortest safe route
# ---------------------------------------------------------------------------


def test_a_constant_q_and_R_selects_shortest_safe_route():
    planner = build_planner(r_miss_uv=2.5)  # q irrelevant: R is a single constant
    result = build_selector(planner, CORRECTED).select([DETOUR, DIRECT], START, S0, GOAL)

    assert result.selected is not None
    assert result.selected.name == "direct"
    by_name = {ev.name: ev for ev in result.evaluations}
    # Perfect, constant observability contributes NOTHING to either route.
    assert by_name["direct"].localization_cost == pytest.approx(0.0, abs=1e-12)
    assert by_name["detour"].localization_cost == pytest.approx(0.0, abs=1e-12)
    # ... so the ranking is decided purely by the explicit travel baseline.
    assert by_name["direct"].travel_time_s < by_name["detour"].travel_time_s
    assert by_name["direct"].total_cost < by_name["detour"].total_cost


def test_a2_constant_but_degraded_q_still_selects_shortest_safe_route(tmp_path):
    """Uniformly poor observability must not reward a longer route either."""
    gp = _write_gp_artifact(tmp_path / "uniform_q.npz", lambda x, y: 0.30)
    planner = build_planner(gp_artifact=str(gp))
    result = build_selector(planner, CORRECTED).select([DETOUR, DIRECT], START, S0, GOAL)

    by_name = {ev.name: ev for ev in result.evaluations}
    assert result.selected.name == "direct"
    # The localization cost is a positive constant RATE, so it scales with
    # duration -- it can only ever add to the cost of taking longer.
    assert by_name["direct"].localization_cost > 0.0
    assert by_name["detour"].localization_cost > by_name["direct"].localization_cost


# ---------------------------------------------------------------------------
# (b) an unsafe short route is rejected
# ---------------------------------------------------------------------------


def test_b_unsafe_short_route_is_rejected():
    planner = build_planner(r_miss_uv=2.5)
    result = build_selector(planner, CORRECTED).select(
        [UNSAFE_SHORTCUT, DIRECT, DETOUR], START, S0, GOAL
    )
    by_name = {ev.name: ev for ev in result.evaluations}

    unsafe = by_name["unsafe_shortcut"]
    assert not unsafe.safety.safe
    assert "body_inside_driveable" in unsafe.safety.failed_gates
    assert unsafe.safety.obstacle_body_clearance_min_m < 0.0  # it hits the block
    assert result.selected is not None and result.selected.name == "direct"
    assert not unsafe.selected


def test_b2_unsafe_route_is_rejected_even_when_its_cost_is_lowest():
    """Safety is a hard gate: no weighting can buy an unsafe route."""
    planner = build_planner(r_miss_uv=2.5)
    # Make the unsafe route cheap by every term the objective contains.
    objective = RouteObjectiveConfig(
        mode="corrected_v2", travel_time_weight=1.0, localization_weight=1.0,
        goal_risk_weight=0.0, obstacle_weight=0.0, discount_gamma=1.0,
    )
    # Only the long detour is safe here, and the unsafe shortcut is cheaper.
    result = build_selector(planner, objective).select(
        [UNSAFE_SHORTCUT, DETOUR], START, S0, GOAL
    )
    by_name = {ev.name: ev for ev in result.evaluations}
    cheapest = min(result.evaluations, key=lambda ev: ev.total_cost)
    assert cheapest.name == "unsafe_shortcut", "test premise: the unsafe route scores best"
    assert result.selected.name == "detour"
    assert not by_name["unsafe_shortcut"].selected


# ---------------------------------------------------------------------------
# (c) lowering q on part of a route increases its localization cost
# ---------------------------------------------------------------------------


def test_c_lowering_q_on_part_of_the_route_increases_localization_cost(tmp_path):
    good = _write_gp_artifact(tmp_path / "q_all_good.npz", lambda x, y: 0.99)
    # Same field, except a blind patch across the middle of the main corridor.
    def patchy(x, y):
        return 0.02 if (4.0 <= x <= 6.0 and -1.0 <= y <= 1.0) else 0.99

    patched = _write_gp_artifact(tmp_path / "q_patchy.npz", patchy)

    cost_good = (
        build_selector(build_planner(gp_artifact=str(good)), CORRECTED)
        .evaluate_candidate(DIRECT, START, S0, GOAL)
    )
    cost_patched = (
        build_selector(build_planner(gp_artifact=str(patched)), CORRECTED)
        .evaluate_candidate(DIRECT, START, S0, GOAL)
    )

    # Identical geometry, identical travel time; only q changed.
    assert cost_patched.travel_time_s == pytest.approx(cost_good.travel_time_s)
    assert cost_patched.min_q < cost_good.min_q
    assert cost_patched.localization_cost > cost_good.localization_cost
    assert cost_patched.total_cost > cost_good.total_cost


# ---------------------------------------------------------------------------
# (d) worsening R increases localization cost where measurements are available
# ---------------------------------------------------------------------------


def test_d_worsening_R_increases_localization_cost(tmp_path):
    gp = _write_gp_artifact(tmp_path / "q_available.npz", lambda x, y: 0.99)
    R_ref = 2.5  # the anchor stays fixed while the achievable quality worsens

    costs = []
    for r_visible in (2.5, 5.0, 10.0):
        planner = build_planner(
            r_visible_uv=r_visible, r_miss_uv=40.0, r_reference_uv=R_ref,
            gp_artifact=str(gp),
        )
        selector = build_selector(planner, CORRECTED)
        selector.R_reference = np.diag([R_ref ** 2, R_ref ** 2])
        costs.append(selector.evaluate_candidate(DIRECT, START, S0, GOAL).localization_cost)

    assert costs[0] < costs[1] < costs[2], costs
    # At the reference quality the cost is ~0 (the small residual is the q=0.99
    # availability blend, not an R effect), and it grows by ~2*log(2) nats per
    # second of exposure for each doubling of the measurement std.
    assert costs[0] < 0.1 * costs[1], costs
    duration_s = 10.0 / 0.6  # direct route at v_max
    assert (costs[1] - costs[0]) == pytest.approx(
        duration_s * 2.0 * math.log(2.0), rel=0.05
    )


def test_d2_localization_rate_is_monotone_and_anchored():
    R_ref = np.diag([2.5 ** 2, 2.5 ** 2])
    assert localization_cost_rate_np(R_ref, R_ref) == pytest.approx(0.0, abs=1e-12)
    worse = np.diag([5.0 ** 2, 5.0 ** 2])
    assert localization_cost_rate_np(worse, R_ref) == pytest.approx(2.0 * math.log(2.0), rel=1e-12)
    # Better-than-reference can never pay a bonus.
    better = np.diag([1.0, 1.0])
    assert localization_cost_rate_np(better, R_ref) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (e) a longer observable route wins only when it earns its extra travel
# ---------------------------------------------------------------------------


def test_e_longer_observable_route_wins_only_above_the_break_even_weight(tmp_path):
    """The detour is observable, the direct route is blind in the middle."""
    def q_field(x, y):
        if 3.5 <= x <= 6.5 and -1.0 <= y <= 1.0:
            return 0.01  # blind patch on the short route
        return 0.99

    gp = _write_gp_artifact(tmp_path / "q_blind_middle.npz", q_field)

    def run(weight: float):
        planner = build_planner(gp_artifact=str(gp))
        objective = RouteObjectiveConfig(
            mode="corrected_v2", travel_time_weight=1.0, localization_weight=weight,
            goal_risk_weight=0.0, obstacle_weight=0.0, discount_gamma=1.0,
        )
        return build_selector(planner, objective).select([DIRECT, DETOUR], START, S0, GOAL)

    # Break-even weight, computed from the decomposition at unit weight.
    unit = {ev.name: ev for ev in run(1.0).evaluations}
    d_travel = unit["detour"].travel_cost - unit["direct"].travel_cost
    d_loc = unit["direct"].localization_cost - unit["detour"].localization_cost
    assert d_travel > 0.0, "the detour must really be longer"
    assert d_loc > 0.0, "the detour must really be more observable"
    break_even = d_travel / d_loc

    below = run(break_even * 0.5)
    above = run(break_even * 2.0)
    assert below.selected.name == "direct", (
        "below break-even the observability benefit does not pay for the detour"
    )
    assert above.selected.name == "detour", (
        "above break-even the measured localization benefit exceeds the extra travel"
    )
    # And the reason is visible in the decomposition, not hidden in a total.
    above_rows = {ev.name: ev for ev in above.evaluations}
    assert above_rows["detour"].localization_cost < above_rows["direct"].localization_cost
    assert above_rows["detour"].travel_cost > above_rows["direct"].travel_cost


def test_e2_no_route_flip_without_an_observability_difference(tmp_path):
    """Sanity companion to (e): equal q gives the shortest route at ANY weight."""
    gp = _write_gp_artifact(tmp_path / "q_uniform.npz", lambda x, y: 0.6)
    for weight in (0.1, 1.0, 10.0, 1000.0):
        planner = build_planner(gp_artifact=str(gp))
        objective = RouteObjectiveConfig(
            mode="corrected_v2", travel_time_weight=1.0, localization_weight=weight,
            goal_risk_weight=0.0, obstacle_weight=0.0, discount_gamma=1.0,
        )
        result = build_selector(planner, objective).select([DIRECT, DETOUR], START, S0, GOAL)
        assert result.selected.name == "direct", f"flipped at localization_weight={weight}"


# ---------------------------------------------------------------------------
# (f) constant offsets in the ambiguity cannot change the ranking
# ---------------------------------------------------------------------------


def test_f_constant_ambiguity_offset_cannot_change_the_ranking(tmp_path):
    """Rescaling the measurement units shifts every ambiguity by a constant.

    `R -> c^2 R` shifts the per-step ambiguity by `2*log(c)` for a 2-D
    measurement. The corrected objective must be blind to that shift; the legacy
    accounting is not, and that is exactly the defect being fixed.
    """
    def q_field(x, y):
        return 0.02 if (4.0 <= x <= 6.0 and -1.0 <= y <= 1.0) else 0.99

    gp = _write_gp_artifact(tmp_path / "q_patchy_offset.npz", q_field)

    def rankings(scale: float, mode: str, objective: RouteObjectiveConfig):
        planner = build_planner(
            r_visible_uv=2.5 * scale,
            r_miss_uv=40.0 * scale,
            gp_artifact=str(gp),
            localization_cost_mode=mode,
        )
        result = build_selector(planner, objective).select([DIRECT, DETOUR], START, S0, GOAL)
        order = [ev.name for ev in sorted(result.evaluations, key=lambda e: e.total_cost)]
        return result.selected.name, order, {ev.name: ev for ev in result.evaluations}

    legacy_objective = RouteObjectiveConfig(
        mode="legacy_efe", travel_time_weight=0.0, localization_weight=0.0,
        goal_risk_weight=0.0, obstacle_weight=0.0, discount_gamma=1.0,
        risk_uses_reference_R=False,
    )
    corrected_offsets = [
        rankings(scale, "anchored_excess", CORRECTED) for scale in (1.0, 1e-3, 1e3)
    ]
    assert len({r[0] for r in corrected_offsets}) == 1, "corrected ranking moved under a pure offset"
    assert len({tuple(r[1]) for r in corrected_offsets}) == 1
    # The per-step ambiguity really did shift; the cost did not.
    base_loc = corrected_offsets[0][2]["direct"].localization_cost
    for _winner, _order, rows in corrected_offsets[1:]:
        assert rows["direct"].localization_cost == pytest.approx(base_loc, rel=1e-9)
        assert rows["direct"].ambiguity_raw_nat_s != pytest.approx(
            corrected_offsets[0][2]["direct"].ambiguity_raw_nat_s
        )

    legacy_winners = {rankings(scale, "raw_ambiguity", legacy_objective)[0]
                      for scale in (1.0, 1e-3)}
    assert len(legacy_winners) == 2, (
        "expected the legacy accounting to flip under a pure unit change "
        "(this test documents the defect the correction removes)"
    )


def test_f2_reference_ambiguity_matches_the_closed_form():
    # For a 2-D measurement with R = r^2 I: A = log(2*pi*e) + 2*log(r).
    for r in (0.01, 1.0, 2.5, 40.0):
        R = np.diag([r ** 2, r ** 2])
        assert reference_ambiguity(R) == pytest.approx(
            math.log(2.0 * math.pi * math.e) + 2.0 * math.log(r), rel=1e-12
        )


# ---------------------------------------------------------------------------
# (g) NumPy and CasADi agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("localization_cost_mode", ["anchored_excess", "raw_ambiguity"])
@pytest.mark.parametrize("risk_uses_reference_R", [True, False])
def test_g_numpy_and_casadi_objectives_agree(tmp_path, localization_cost_mode, risk_uses_reference_R):
    casadi = pytest.importorskip("casadi")  # noqa: F841
    gp = _write_gp_artifact(tmp_path / "q_for_casadi.npz",
                            lambda x, y: 0.05 if 4.0 <= x <= 6.0 else 0.95)
    horizon = 8
    planner = build_planner(
        gp_artifact=str(gp), horizon=horizon, dt=0.4,
        localization_cost_mode=localization_cost_mode, discount_gamma=0.995,
    )
    planner.risk_uses_reference_R = risk_uses_reference_R
    planner.route_length_weight = 0.7
    planner.travel_time_weight = 0.3

    goal_state = planner._goal_state(GOAL, 0.0)
    goal_obs = planner._goal_obs(goal_state)
    valgrad = planner._get_casadi_valgrad(
        goal_state, goal_obs, use_observation_risk=True, use_ambiguity_term=True
    )

    rng = np.random.default_rng(7)
    numpy_values = []
    casadi_values = []
    for _ in range(6):
        u = np.column_stack(
            [rng.uniform(0.0, 0.6, horizon), rng.uniform(-1.0, 1.0, horizon)]
        ).reshape(-1)
        j_np = planner._evaluate_controls(u, START, S0, goal_state, goal_obs, None, False,
                                          progress_index=3.0)
        j_ca, _grad = valgrad(u, START, S0, goal_obs, GOAL, 3.0)
        numpy_values.append(float(j_np))
        casadi_values.append(float(j_ca))
        assert j_np == pytest.approx(j_ca, rel=1e-5, abs=1e-6)

    assert list(np.argsort(numpy_values)) == list(np.argsort(casadi_values))


def test_g2_numpy_and_casadi_candidate_rankings_agree(tmp_path):
    """Rank real route seeds with both back-ends and require the same order."""
    pytest.importorskip("casadi")
    gp = _write_gp_artifact(tmp_path / "q_for_ranking.npz",
                            lambda x, y: 0.05 if (3.5 <= x <= 6.5 and -1.0 <= y <= 1.0) else 0.95)
    horizon = 40
    dt = 0.5
    planner = build_planner(gp_artifact=str(gp), horizon=horizon, dt=dt, discount_gamma=0.995)
    planner.route_length_weight = 1.0

    goal_state = planner._goal_state(GOAL, 0.0)
    goal_obs = planner._goal_obs(goal_state)
    valgrad = planner._get_casadi_valgrad(
        goal_state, goal_obs, use_observation_risk=True, use_ambiguity_term=True
    )

    numpy_values = []
    casadi_values = []
    for candidate in (DIRECT, DETOUR, UNSAFE_SHORTCUT):
        controls = planner._controls_for_waypoints(START, list(candidate.waypoints))
        j_np = planner._evaluate_controls(controls, START, S0, goal_state, goal_obs, None, False)
        j_ca, _ = valgrad(controls, START, S0, goal_obs, GOAL, 0.0)
        numpy_values.append(float(j_np))
        casadi_values.append(float(j_ca))
        assert j_np == pytest.approx(j_ca, rel=1e-5, abs=1e-6)

    assert list(np.argsort(numpy_values)) == list(np.argsort(casadi_values))


# ---------------------------------------------------------------------------
# Structural guarantees of the safety model
# ---------------------------------------------------------------------------


def test_footprint_turn_feasibility_is_checked_not_assumed():
    """An in-place turn is only safe if the swept body fits, not just the centre."""
    narrow = json.dumps(
        {
            "model_name": "narrow_L",
            "prisms": [
                {"name": "leg_x", "xmin": -1.0, "xmax": 5.0, "ymin": -0.45, "ymax": 0.45,
                 "zmin": 0.0, "zmax": 0.1},
                {"name": "leg_y", "xmin": 4.1, "xmax": 5.0, "ymin": -0.45, "ymax": 5.0,
                 "zmin": 0.0, "zmax": 0.1},
            ],
        }
    )
    model = RouteSafetyModel(
        driveable_geometry_json=narrow,
        obstacle_geometry_json="",
        footprint=RobotFootprint(0.80, 0.55),
        gates=SafetyGateConfig(goal_radius_m=0.25),
    )
    # The path centre-line stays inside the lane the whole way ...
    assert model.driveable_path_clearance(4.55, 0.0) > 0.0
    # ... but the body cannot rotate 90 degrees there.
    assert model.driveable_body_clearance(4.55, 0.0, 0.0) > 0.0
    assert model.driveable_body_clearance(4.55, 0.0, math.pi / 4.0) < 0.0


def test_planner_anchored_term_equals_the_closed_form_rate(tmp_path):
    """One definition: the planner's anchored term IS the closed-form rate under ET1.

    The planner anchors by passing the reference through the same observation
    transform; the selector and the tests use the closed form
    0.5*log(det R_plan / det R_ref). Under ET1 these are the same number, and
    this test pins that down so the two implementations cannot drift apart.
    """
    gp = _write_gp_artifact(tmp_path / "q_mixed.npz",
                            lambda x, y: 0.05 if 4.0 <= x <= 6.0 else 0.95)
    horizon = 5
    dt = 0.4
    planner = build_planner(gp_artifact=str(gp), horizon=horizon, dt=dt, discount_gamma=1.0)
    planner.risk_weight_obs = 0.0  # isolate the localization term

    controls = np.column_stack([np.full(horizon, 0.6), np.zeros(horizon)]).reshape(-1)
    goal_state = planner._goal_state(GOAL, 0.0)
    goal_obs = planner._goal_obs(goal_state)
    _total, metrics = planner._evaluate_controls(
        controls, START, S0, goal_state, goal_obs, None, True
    )

    # Recompute the same quantity from the closed form.
    m = START.astype(float).copy()
    S = S0.copy()
    expected = 0.0
    for t in range(horizon):
        m, S = planner.predict(m, S, controls.reshape(horizon, 2)[t])
        R_plan = planner.planning_visibility_diagnostics(m, S)["R_plan"]
        expected += localization_cost_rate_np(R_plan, planner.R_reference)
    expected /= float(horizon)  # the evaluator normalises by the effective horizon

    assert metrics["ambiguity_cost"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_anchoring_shifts_the_fixed_horizon_objective_by_a_constant(tmp_path):
    """At a FIXED horizon, anchoring changes nothing the optimizer can see.

    The anchored term differs from the raw ambiguity by the reference ambiguity,
    which every candidate accumulates once per step. So at a fixed horizon the
    difference between the two objectives is a route-independent constant: the
    MPC optimum and gradients are unchanged, and the correction bites only where
    it should -- when candidates are compared at different arrival times.
    """
    gp = _write_gp_artifact(tmp_path / "q_for_offset.npz",
                            lambda x, y: 0.05 if 4.0 <= x <= 6.0 else 0.95)
    horizon = 8
    kwargs = dict(gp_artifact=str(gp), horizon=horizon, dt=0.4, discount_gamma=0.995)
    anchored = build_planner(localization_cost_mode="anchored_excess", **kwargs)
    raw = build_planner(localization_cost_mode="raw_ambiguity", **kwargs)

    goal_state = anchored._goal_state(GOAL, 0.0)
    goal_obs = anchored._goal_obs(goal_state)
    rng = np.random.default_rng(11)
    diffs = []
    for _ in range(5):
        u = np.column_stack(
            [rng.uniform(0.0, 0.6, horizon), rng.uniform(-1.0, 1.0, horizon)]
        ).reshape(-1)
        j_anchored = anchored._evaluate_controls(u, START, S0, goal_state, goal_obs, None,
                                                 False, progress_index=2.0)
        j_raw = raw._evaluate_controls(u, START, S0, goal_state, goal_obs, None,
                                       False, progress_index=2.0)
        diffs.append(float(j_raw - j_anchored))

    assert max(diffs) - min(diffs) < 1e-9 * max(1.0, abs(diffs[0])), diffs
    assert diffs[0] > 0.0  # the removed constant is the positive reference ambiguity


def test_safety_sampling_resolution_does_not_change_cost():
    """Refining the geometric check must not move a route's travel time or cost."""
    planner = build_planner(r_miss_uv=2.5)
    coarse = build_selector(planner, CORRECTED)
    fine = build_selector(planner, CORRECTED)
    fine.execution = ExecutionModelConfig(
        dt=0.1, v_max=0.6, w_max=1.0,
        safety_sample_arc_m=0.005, safety_sample_yaw_rad=0.005,
    )
    a = coarse.evaluate_candidate(DETOUR, START, S0, GOAL)
    b = fine.evaluate_candidate(DETOUR, START, S0, GOAL)
    assert a.travel_time_s == pytest.approx(b.travel_time_s)
    assert a.route_length_m == pytest.approx(b.route_length_m)
    assert a.total_cost == pytest.approx(b.total_cost)


def test_reaching_the_goal_is_a_hard_gate():
    short_of_goal = RouteCandidate("stops_short", ((6.0, 0.0),))
    planner = build_planner(r_miss_uv=2.5)
    result = build_selector(planner, CORRECTED).select(
        [short_of_goal, DIRECT], START, S0, GOAL
    )
    by_name = {ev.name: ev for ev in result.evaluations}
    assert by_name["stops_short"].total_cost < by_name["direct"].total_cost
    assert not by_name["stops_short"].safety.safe
    assert "reaches_goal" in by_name["stops_short"].safety.failed_gates
    assert result.selected.name == "direct"
