"""Full-route (A -> B) global route selection with an explicit cost decomposition.

What this is
------------
The runtime global stage solves one long-horizon EFE plan, seeded from a small
set of map-derived route candidates, and freezes it. Comparing those candidates
is therefore a *full-route* decision: the candidates differ in arrival time, not
just in the controls applied over a common fixed horizon. This module makes that
comparison explicit, auditable, and safe:

  1. every candidate is executed with one shared reference execution model, so
     the routes differ only in geometry (ground rule 7);
  2. hard safety gates (`route_safety`) reject unsafe candidates *before* any
     objective value is compared (ground rule 8);
  3. the surviving candidates are ranked by

         J = travel baseline  +  localization cost

     where the travel baseline is an explicit route length / travel time (not
     actuator effort), and the localization cost is the reference-anchored,
     non-negative excess-entropy rate of `localization_cost.py`;
  4. every term of every candidate is reported, so the reason for the choice is
     visible rather than inferred.

`RouteObjectiveConfig.legacy_efe()` reproduces the *previous* accounting -- the
raw ambiguity plus the R_plan-contaminated goal risk, discounted and truncated at
arrival -- so old and corrected selections can be compared on identical
candidates, geometry, q and R (ground rule 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Sequence

import numpy as np

from planning.core.dynamics import unicycle_step
from planning.core.efe_utils import ambiguity as ambiguity_np, risk_components, wrap_angle
from planning.core.localization_cost import localization_cost_rate_np, reference_ambiguity
from planning.core.route_safety import (
    PHASE_STRAIGHT,
    PHASE_TURN_IN_PLACE,
    RouteSafetyModel,
    RouteSafetyReport,
)


# ---------------------------------------------------------------------------
# Candidates and the shared reference execution model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteCandidate:
    """A candidate route: an ordered polyline of waypoints ending at the goal."""

    name: str
    waypoints: tuple[tuple[float, float], ...]

    @staticmethod
    def from_dict(payload: dict) -> "RouteCandidate":
        wps = tuple(
            (float(w[0]), float(w[1])) for w in payload.get("waypoints", []) if len(w) >= 2
        )
        return RouteCandidate(name=str(payload.get("name", "route")), waypoints=wps)


@dataclass(frozen=True)
class ExecutionModelConfig:
    """Reference turn-then-go execution model shared by every candidate.

    It mirrors the deployed local tracker (`efe_agent_node._simple_local_plan`):
    rotate in place onto the leg heading, then translate along it. Using one
    execution model for every candidate and every condition is what makes the
    travel baseline comparable (ground rule 7).
    """

    dt: float = 0.1
    v_max: float = 0.6
    w_max: float = 1.0
    # Resolution at which the executed trajectory is re-sampled FOR THE SAFETY
    # CHECK only. It deliberately does not touch the control tape, so refining
    # the geometric check can never change a route's travel time or cost.
    safety_sample_arc_m: float = 0.05
    safety_sample_yaw_rad: float = 0.05

    def signature(self) -> tuple:
        return (
            round(float(self.dt), 6),
            round(float(self.v_max), 6),
            round(float(self.w_max), 6),
            round(float(self.safety_sample_arc_m), 6),
            round(float(self.safety_sample_yaw_rad), 6),
        )


@dataclass
class ExecutedRoute:
    """The executed trajectory of one candidate under the reference model."""

    poses: np.ndarray  # (N+1, 3) including the start pose
    controls: np.ndarray  # (N, 2) [v, w]
    phases: tuple[str, ...]  # (N,) phase of each step
    dt: float
    route_length_m: float
    turn_angle_total_rad: float
    kinematically_feasible: bool

    @property
    def n_steps(self) -> int:
        return int(self.controls.shape[0])

    @property
    def travel_time_s(self) -> float:
        return float(self.n_steps) * float(self.dt)

    def safety_poses(self, *, max_arc_m: float = 0.05, max_yaw_rad: float = 0.05):
        """Re-sample the executed trajectory finely for the geometric safety gate.

        Purely a refinement of the *check*: the control tape, the travel time and
        every cost term are untouched, so tightening the resolution can never
        move a route's score. Translation steps are interpolated in position and
        in-place rotations in heading, so the swept poses of every turn are seen.
        """
        poses = np.asarray(self.poses, dtype=float)
        dense = [poses[0]]
        dense_phases = [PHASE_STRAIGHT]
        for i in range(1, poses.shape[0]):
            a, b = poses[i - 1], poses[i]
            phase = self.phases[i - 1] if i - 1 < len(self.phases) else PHASE_STRAIGHT
            arc = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            dyaw = abs(float(wrap_angle(float(b[2]) - float(a[2]))))
            n = max(
                1,
                int(math.ceil(arc / max(float(max_arc_m), 1e-9))),
                int(math.ceil(dyaw / max(float(max_yaw_rad), 1e-9))),
            )
            for k in range(1, n + 1):
                alpha = float(k) / float(n)
                theta = float(a[2]) + alpha * float(wrap_angle(float(b[2]) - float(a[2])))
                dense.append(
                    np.array(
                        [
                            float(a[0]) + alpha * (float(b[0]) - float(a[0])),
                            float(a[1]) + alpha * (float(b[1]) - float(a[1])),
                            wrap_angle(theta),
                        ],
                        dtype=float,
                    )
                )
                dense_phases.append(phase)
        return np.asarray(dense, dtype=float), tuple(dense_phases)


def simulate_route(
    start_xy_yaw: Sequence[float],
    candidate: RouteCandidate,
    cfg: ExecutionModelConfig,
) -> ExecutedRoute:
    """Execute a candidate polyline with the shared turn-then-go model.

    The returned pose sequence contains the poses swept during every in-place
    rotation, so the geometric safety gate sees the turns as well as the
    straights. Step sizes are additionally capped by `max_step_arc_m` /
    `max_step_yaw_rad` so the sampling is fine enough for the footprint checks.
    """
    state = np.asarray(start_xy_yaw, dtype=float).reshape(-1)[:3].astype(float)
    dt = float(cfg.dt)
    v_max = float(cfg.v_max)
    w_max = float(cfg.w_max)
    poses = [state.copy()]
    controls: list[list[float]] = []
    phases: list[str] = []
    length = 0.0
    turn_total = 0.0
    feasible = True

    def _emit(v: float, w: float, n_steps: int, phase: str):
        nonlocal state
        for _ in range(int(n_steps)):
            state = unicycle_step(state, [float(v), float(w)], dt)
            poses.append(np.asarray(state, dtype=float).copy())
            controls.append([float(v), float(w)])
            phases.append(phase)

    for wp in candidate.waypoints:
        target = np.asarray(wp, dtype=float).reshape(2)
        # --- rotate in place onto the leg heading ---
        delta = target - state[:2]
        distance = float(np.hypot(delta[0], delta[1]))
        if distance > 1e-9:
            desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
            yaw_error = wrap_angle(desired_yaw - float(state[2]))
            if abs(yaw_error) > 1e-9:
                turn_total += abs(yaw_error)
                # Enough steps that |w| <= w_max; nothing else may lengthen the turn.
                n = max(1, int(math.ceil(abs(yaw_error) / max(w_max * dt, 1e-9) - 1e-12)))
                w = yaw_error / (n * dt)
                if abs(w) > w_max + 1e-9:
                    feasible = False
                _emit(0.0, w, n, PHASE_TURN_IN_PLACE)
                state[2] = wrap_angle(desired_yaw)
                poses[-1][2] = state[2]
            # --- translate along the leg ---
            n = max(1, int(math.ceil(distance / max(v_max * dt, 1e-9) - 1e-12)))
            v = distance / (n * dt)
            if v > v_max + 1e-9:
                feasible = False
            _emit(v, 0.0, n, PHASE_STRAIGHT)
            length += distance
            state[:2] = target
            poses[-1][:2] = target

    if not controls:
        controls.append([0.0, 0.0])
        phases.append(PHASE_STRAIGHT)
        poses.append(poses[-1].copy())

    return ExecutedRoute(
        poses=np.asarray(poses, dtype=float),
        controls=np.asarray(controls, dtype=float),
        phases=tuple(phases),
        dt=dt,
        route_length_m=float(length),
        turn_angle_total_rad=float(turn_total),
        kinematically_feasible=bool(feasible),
    )


# ---------------------------------------------------------------------------
# Objective configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteObjectiveConfig:
    """Weights of the full-route objective.

    Corrected (`mode='corrected_v2'`), in seconds-equivalent:

        J = travel_time_weight * T
          + route_length_weight * L
          + localization_weight * sum_t dt * c_loc(q_t, R_t)
          ( + goal_risk_weight * ... and obstacle_weight * ..., both 0 by default:
            reaching the goal and staying safe are hard gates, not soft costs )

    `localization_weight` is the one explicit exchange rate between observability
    and travel: how many seconds of extra travel are worth avoiding one nat of
    excess measurement entropy sustained for one second. It is a declared design
    parameter -- never fitted to make a particular route win (ground rule 5) --
    and the validation reports the value at which each ranking would flip.

    Legacy (`mode='legacy_efe'`) reproduces the previous accounting: discounted
    per-step (risk with the q/R-contaminated R_plan + raw ambiguity + no-go),
    truncated at arrival, with no travel baseline.
    """

    mode: str = "corrected_v2"
    travel_time_weight: float = 1.0
    route_length_weight: float = 0.0
    localization_weight: float = 1.0
    goal_risk_weight: float = 0.0
    obstacle_weight: float = 0.0
    discount_gamma: float = 1.0
    risk_uses_reference_R: bool = True

    @staticmethod
    def legacy_efe(discount_gamma: float = 1.0) -> "RouteObjectiveConfig":
        """The pre-correction accounting, for old-vs-new comparison only."""
        return RouteObjectiveConfig(
            mode="legacy_efe",
            travel_time_weight=0.0,
            route_length_weight=0.0,
            localization_weight=0.0,
            goal_risk_weight=1.0,
            obstacle_weight=1.0,
            discount_gamma=float(discount_gamma),
            risk_uses_reference_R=False,
        )

    def describe(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------


@dataclass
class RouteEvaluation:
    """Full cost decomposition and safety verdict for one candidate route."""

    name: str
    safety: RouteSafetyReport
    route_length_m: float
    travel_time_s: float
    turn_angle_total_rad: float
    n_steps: int
    travel_cost: float
    localization_cost: float
    localization_cost_nat_s: float
    ambiguity_raw_nat_s: float
    ambiguity_reference_nat_s: float
    goal_risk_cost: float
    goal_risk_cost_reference_R: float
    goal_risk_cost_plan_R: float
    obstacle_cost: float
    total_cost: float
    total_cost_excl_obstacle: float
    mean_q: float
    min_q: float
    mean_r_plan_std: float
    max_r_plan_std: float
    terminal_belief_sigma_xy_m: float
    selected: bool = False
    executed: ExecutedRoute | None = field(default=None, repr=False)

    def as_row(self) -> dict:
        return {
            "route": self.name,
            "safety_status": self.safety.status,
            "safe": self.safety.safe,
            "reaches_goal": self.safety.reaches_goal,
            "terminal_goal_distance_m": self.safety.terminal_goal_distance_m,
            "route_length_m": self.route_length_m,
            "travel_time_s": self.travel_time_s,
            "min_body_clearance_m": self.safety.min_body_clearance_m,
            "obstacle_body_clearance_min_m": self.safety.obstacle_body_clearance_min_m,
            "driveable_body_clearance_min_m": self.safety.driveable_body_clearance_min_m,
            "driveable_body_clearance_min_turn_m": self.safety.driveable_body_clearance_min_turn_m,
            "driveable_path_clearance_min_m": self.safety.driveable_path_clearance_min_m,
            "mean_q": self.mean_q,
            "min_q": self.min_q,
            "mean_r_plan_std_px": self.mean_r_plan_std,
            "travel_cost": self.travel_cost,
            "localization_cost": self.localization_cost,
            "localization_cost_nat_s": self.localization_cost_nat_s,
            "ambiguity_raw_nat_s": self.ambiguity_raw_nat_s,
            "goal_risk_cost": self.goal_risk_cost,
            "obstacle_cost": self.obstacle_cost,
            "total_cost": self.total_cost,
            "total_cost_excl_obstacle": self.total_cost_excl_obstacle,
            "selected": self.selected,
        }


@dataclass
class RouteSelectionResult:
    """Outcome of one full-route comparison."""

    evaluations: list[RouteEvaluation]
    selected: RouteEvaluation | None
    objective: RouteObjectiveConfig
    gate_signature: tuple
    execution_signature: tuple
    notes: tuple[str, ...] = ()

    def rows(self) -> list[dict]:
        return [ev.as_row() for ev in self.evaluations]

    @property
    def safe_evaluations(self) -> list[RouteEvaluation]:
        return [ev for ev in self.evaluations if ev.safety.safe]


class GlobalRouteSelector:
    """Evaluate and rank full candidate routes for one (start, goal) task."""

    def __init__(
        self,
        planner,
        safety_model: RouteSafetyModel,
        *,
        execution: ExecutionModelConfig | None = None,
        objective: RouteObjectiveConfig | None = None,
        R_reference: np.ndarray | None = None,
        use_belief_visibility: bool = True,
        apply_expected_measurement_update: bool = False,
    ):
        # `apply_expected_measurement_update=False` mirrors the shipped per-step
        # evaluator (`base_planner._evaluate_controls`), which propagates the
        # belief by prediction only. It also removes any dependence of the belief
        # on the evaluation step size. Under ET1 the localization cost does not
        # depend on the belief covariance at all, so this choice cannot bias the
        # route ranking; it only affects the reported terminal belief sigma and
        # the belief-tube no-go diagnostic.
        self.planner = planner
        self.safety_model = safety_model
        self.execution = execution or ExecutionModelConfig(
            v_max=float(getattr(planner, "v_max", 0.6)),
            w_max=float(min(abs(getattr(planner, "w_max", 1.0)), abs(getattr(planner, "w_min", -1.0)))),
        )
        self.objective = objective or RouteObjectiveConfig()
        self.R_reference = np.asarray(
            R_reference if R_reference is not None else planner.R_visible, dtype=float
        )
        self.use_belief_visibility = bool(use_belief_visibility)
        self.apply_expected_measurement_update = bool(apply_expected_measurement_update)

    # -- stepwise accounting ----------------------------------------------

    def _visibility_diagnostics(self, m, S):
        if self.use_belief_visibility:
            return self.planner.planning_visibility_diagnostics(m, S)
        p_vis = self.planner.visibility_probability(m)
        if (not self.planner.use_visibility_model) or (self.planner.visibility_model is None):
            p_vis = 1.0
        p_eff = self.planner._visibility_effective_score(p_vis)
        R_plan = self.planner._blend_observation_covariance(p_eff)
        return {
            "p_vis": float(p_vis),
            "p_vis_eff": float(p_eff),
            "R_plan": np.asarray(R_plan, dtype=float),
            "r_plan_u_std": float(math.sqrt(max(R_plan[0, 0], 0.0))),
            "r_plan_v_std": float(math.sqrt(max(R_plan[1, 1], 0.0))),
        }

    def evaluate_candidate(
        self,
        candidate: RouteCandidate,
        start_xy_yaw: Sequence[float],
        S0: np.ndarray,
        goal_xy: Sequence[float],
    ) -> RouteEvaluation:
        executed = simulate_route(start_xy_yaw, candidate, self.execution)
        safety_poses, safety_phases = executed.safety_poses(
            max_arc_m=self.execution.safety_sample_arc_m,
            max_yaw_rad=self.execution.safety_sample_yaw_rad,
        )
        safety = self.safety_model.evaluate_route(
            safety_poses,
            goal_xy,
            phases=safety_phases,
            kinematically_feasible=executed.kinematically_feasible,
        )

        planner = self.planner
        dt = float(executed.dt)
        m = np.asarray(start_xy_yaw, dtype=float).reshape(-1)[:3].astype(float).copy()
        S = np.asarray(S0, dtype=float).copy()
        goal_state = planner._goal_state(np.asarray(goal_xy, dtype=float).reshape(2), 0.0)
        goal_obs = planner._goal_obs(goal_state)
        R_ref = self.R_reference
        amb_ref_per_step = reference_ambiguity(R_ref)

        loc_nat_s = 0.0
        amb_raw_nat_s = 0.0
        risk_ref_sum = 0.0
        risk_plan_sum = 0.0
        obstacle_sum = 0.0
        loc_weighted = 0.0
        risk_weighted = 0.0
        obstacle_weighted = 0.0
        q_values: list[float] = []
        r_std_values: list[float] = []
        gamma = float(self.objective.discount_gamma)
        n_steps = executed.n_steps

        for t in range(n_steps):
            u = executed.controls[t]
            m, S = planner.predict(m, S, u, dt=dt)
            vis = self._visibility_diagnostics(m, S)
            R_plan = np.asarray(vis["R_plan"], dtype=float)
            q_values.append(float(vis["p_vis"]))
            r_std_values.append(float(max(vis["r_plan_u_std"], vis["r_plan_v_std"])))

            mu_plan, Sigma_plan, Gamma_plan = planner.approx_observation(
                m, S, method=planner.approx_method, R_override=R_plan
            )
            mu_ref, Sigma_ref, Gamma_ref = planner.approx_observation(
                m, S, method=planner.approx_method, R_override=R_ref
            )
            amb_step = float(ambiguity_np(Sigma_plan, Gamma_plan, S))
            loc_rate = localization_cost_rate_np(R_plan, R_ref)

            progress = (float(t) / max(int(planner.goal_progress_n_steps), 1))
            goal_cov_t = planner.goal_obs_cov_for_progress(progress)
            risk_plan = float(risk_components(mu_plan, Sigma_plan, (goal_obs, goal_cov_t))["total"])
            risk_ref = float(risk_components(mu_ref, Sigma_ref, (goal_obs, goal_cov_t))["total"])
            risk_scale = float(planner.risk_weight_obs * planner.observation_risk_scale)
            amb_scale = float(planner.ambiguity_weight * planner.ambiguity_term_scale)

            S_nogo = S
            if planner.use_belief_nogo_cost:
                S_nogo = planner._expected_state_posterior_covariance(S, Sigma_plan, Gamma_plan)
            nogo_step = float(planner.obstacle_penalty(m, S_nogo))

            weight_t = gamma ** t
            loc_nat_s += dt * loc_rate
            amb_raw_nat_s += dt * amb_step
            risk_ref_sum += dt * risk_ref
            risk_plan_sum += dt * risk_plan
            obstacle_sum += dt * nogo_step

            if self.objective.mode == "legacy_efe":
                # Exactly the shipped per-step EFE accounting, truncated at arrival.
                risk_weighted += weight_t * risk_scale * risk_plan
                loc_weighted += weight_t * amb_scale * amb_step
                obstacle_weighted += weight_t * nogo_step
            else:
                risk_use = risk_ref if self.objective.risk_uses_reference_R else risk_plan
                risk_weighted += weight_t * risk_scale * risk_use
                loc_weighted += weight_t * dt * loc_rate
                obstacle_weighted += weight_t * nogo_step

            if self.apply_expected_measurement_update:
                S = planner._expected_state_posterior_covariance(S, Sigma_plan, Gamma_plan)

        if self.objective.mode == "legacy_efe":
            # Legacy: no travel baseline; the "observability" term is the RAW
            # ambiguity (constant offset included), which is exactly the defect.
            travel_cost = 0.0
            localization_cost = float(loc_weighted)
        else:
            travel_cost = float(
                self.objective.travel_time_weight * executed.travel_time_s
                + self.objective.route_length_weight * executed.route_length_m
            )
            localization_cost = float(self.objective.localization_weight * loc_weighted)
        goal_risk_cost = float(self.objective.goal_risk_weight * risk_weighted)
        obstacle_cost = float(self.objective.obstacle_weight * obstacle_weighted)
        total_excl_obstacle = travel_cost + localization_cost + goal_risk_cost
        total = total_excl_obstacle + obstacle_cost

        sigma_xy = float(math.sqrt(max(float(np.max(np.linalg.eigvalsh(S[:2, :2]))), 0.0)))
        return RouteEvaluation(
            name=candidate.name,
            safety=safety,
            route_length_m=executed.route_length_m,
            travel_time_s=executed.travel_time_s,
            turn_angle_total_rad=executed.turn_angle_total_rad,
            n_steps=n_steps,
            travel_cost=travel_cost,
            localization_cost=localization_cost,
            localization_cost_nat_s=float(loc_nat_s),
            ambiguity_raw_nat_s=float(amb_raw_nat_s),
            ambiguity_reference_nat_s=float(amb_ref_per_step * dt * n_steps),
            goal_risk_cost=goal_risk_cost,
            goal_risk_cost_reference_R=float(risk_ref_sum),
            goal_risk_cost_plan_R=float(risk_plan_sum),
            obstacle_cost=obstacle_cost,
            total_cost=float(total),
            total_cost_excl_obstacle=float(total_excl_obstacle),
            mean_q=float(np.mean(q_values)) if q_values else math.nan,
            min_q=float(np.min(q_values)) if q_values else math.nan,
            mean_r_plan_std=float(np.mean(r_std_values)) if r_std_values else math.nan,
            max_r_plan_std=float(np.max(r_std_values)) if r_std_values else math.nan,
            terminal_belief_sigma_xy_m=sigma_xy,
            executed=executed,
        )

    # -- selection ---------------------------------------------------------

    def select(
        self,
        candidates: Sequence[RouteCandidate],
        start_xy_yaw: Sequence[float],
        S0: np.ndarray,
        goal_xy: Sequence[float],
    ) -> RouteSelectionResult:
        evaluations = [
            self.evaluate_candidate(c, start_xy_yaw, S0, goal_xy) for c in candidates
        ]
        # Hard safety first: unsafe candidates never enter the cost comparison.
        safe = [ev for ev in evaluations if ev.safety.safe]
        notes: list[str] = []
        selected = None
        if safe:
            selected = min(safe, key=lambda ev: (ev.total_cost, ev.route_length_m, ev.name))
            selected.selected = True
        else:
            notes.append(
                "no candidate passed the hard safety gates; no route selected"
            )
        return RouteSelectionResult(
            evaluations=evaluations,
            selected=selected,
            objective=self.objective,
            gate_signature=self.safety_model.gates.signature(),
            execution_signature=self.execution.signature(),
            notes=tuple(notes),
        )
