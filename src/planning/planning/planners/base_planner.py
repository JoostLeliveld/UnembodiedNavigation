"""Base planner classes (pure Python, no ROS)."""

from dataclasses import dataclass
import json
import math
import time
import numpy as np

from scipy.optimize import minimize

from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import ET1, ET2, UT, ambiguity, risk_components, wrap_angle
from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel
from planning.core.visibility_gp_map import GPVisibilityMapConfig, GPVisibilityMapModel
from unav_common.camera_model import ObliqueCameraModel
from planning.core.rollout import rollout_unicycle


@dataclass
class PlanResult:
    controls: np.ndarray
    states: np.ndarray
    total_cost: float
    risk_cost: float
    ambiguity_cost: float
    control_cost: float
    risk_mean: float = 0.0
    risk_cov_trace: float = 0.0
    risk_cov_logdet: float = 0.0
    delta_risk_visibility: float = 0.0
    delta_ambiguity_visibility: float = 0.0
    obstacle_cost: float = 0.0
    backend: str = "unknown"
    optimizer_success: bool = False
    optimizer_status: int = 0
    optimizer_nit: int = 0
    optimizer_nfev: int = 0
    optimizer_message: str = ""
    solve_time_s: float = 0.0
    selected_source: str = ""
    p_vis_plan: float = 1.0
    p_vis_plan_eff: float = 1.0
    r_plan_u_std: float = np.nan
    r_plan_v_std: float = np.nan
    terminal_goal_distance_pred: float = np.nan
    terminal_goal_progress_m: float = np.nan
    fraction_horizon_low_pvis: float = np.nan
    fraction_horizon_high_ambiguity: float = np.nan
    min_predicted_obstacle_distance_m: float = np.nan
    rollout_valid: bool = True
    invalid_reason: str = ""


class UnicyclePlannerBase:
    """Shared unicycle planner logic. Subclasses define objective specifics."""

    def __init__(
        self,
        *,
        horizon,
        dt,
        v_min,
        v_max,
        w_min,
        w_max,
        control_weight,
        process_noise_xy,
        process_noise_theta,
        obs_noise_uv,
        goal_sigma_uv,
        risk_weight_obs,
        ambiguity_weight,
        optimizer_maxiter,
        optimizer_gtol,
        optimizer_warm_start,
        optimizer_warm_start_shift_steps=1,
        approx_method=None,
        use_obs_risk=None,
        use_ambiguity=None,
        seed,
        camera_params,
        use_visibility_model=False,
        visibility_target_height_m=0.0,
        visibility_geometry_json='',
        collision_geometry_json='',
        visibility_artifact_path='',
        r_visible_uv=2.5,
        r_miss_uv=120.0,
        visibility_sigma_kappa=1.0,
        goal_prior_u_std_start=80.0,
        goal_prior_v_std_start=80.0,
        goal_prior_u_std_final=18.0,
        goal_prior_v_std_final=18.0,
        goal_tightening_power=0.45,
        goal_progress_n_steps=90,
        observation_risk_scale=1.25,
        ambiguity_term_scale=1.00,
        discount_gamma=0.98,
        optimizer_maxfun=500,
        optimizer_ftol=1e-6,
        optimizer_multistart=False,
        optimizer_multistart_include_direct=True,
        optimizer_multistart_lateral_offsets='',
        optimizer_initial_routes_json='',
        use_nogo_cost=False,
        nogo_penalty_type='softplus',
        nogo_weight=0.0,
        nogo_safe_distance=0.35,
        nogo_gaussian_sigma=0.25,
        nogo_softplus_scale=0.08,
        nogo_logbarrier_scale=0.25,
        nogo_logbarrier_eps=1e-3,
        use_belief_nogo_cost=False,
        nogo_belief_kappa=1.0,
        robot_collision_radius_m=0.125,
        runtime_debug=False,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.control_weight = float(control_weight)

        self.process_noise_xy = float(process_noise_xy)
        self.process_noise_theta = float(process_noise_theta)
        self.obs_noise_uv = float(obs_noise_uv)

        self.goal_sigma_uv = float(goal_sigma_uv)
        if self.goal_sigma_uv <= 0.0:
            raise ValueError("goal_sigma_uv must be > 0.0; do not use 0.0 as a sentinel")

        self.risk_weight_obs = float(risk_weight_obs)
        self.ambiguity_weight = float(ambiguity_weight)
        self.r_visible_uv = float(r_visible_uv)
        self.r_miss_uv = float(r_miss_uv)
        self.visibility_sigma_kappa = float(max(visibility_sigma_kappa, 1e-6))
        self.goal_prior_u_std_start = float(goal_prior_u_std_start)
        self.goal_prior_v_std_start = float(goal_prior_v_std_start)
        self.goal_prior_u_std_final = float(goal_prior_u_std_final)
        self.goal_prior_v_std_final = float(goal_prior_v_std_final)
        self.goal_tightening_power = float(max(goal_tightening_power, 1e-6))
        self.goal_progress_n_steps = int(max(goal_progress_n_steps, 1))
        self.observation_risk_scale = float(observation_risk_scale)
        self.ambiguity_term_scale = float(ambiguity_term_scale)
        self.discount_gamma = float(discount_gamma)
        self.robot_collision_radius_m = float(max(robot_collision_radius_m, 0.0))

        if approx_method is None:
            self.approx_method = 'ET1'
        else:
            self.approx_method = str(approx_method).upper()
        if self.approx_method not in ('ET1', 'ET2'):
            raise ValueError("approx_method must be 'ET1' or 'ET2'")

        if use_obs_risk is None:
            self.use_obs_risk = True
        else:
            self.use_obs_risk = bool(use_obs_risk)

        if use_ambiguity is None:
            self.use_ambiguity = True
        else:
            self.use_ambiguity = bool(use_ambiguity)

        self.optimizer_maxiter = int(optimizer_maxiter)
        self.optimizer_gtol = float(optimizer_gtol)
        self.optimizer_warm_start = bool(optimizer_warm_start)
        self.optimizer_warm_start_shift_steps = int(max(optimizer_warm_start_shift_steps, 1))
        self.optimizer_maxfun = int(max(optimizer_maxfun, 1))
        self.optimizer_ftol = float(max(optimizer_ftol, 1e-12))
        self.optimizer_multistart = self._as_bool_like(optimizer_multistart)
        self.optimizer_multistart_include_direct = self._as_bool_like(
            optimizer_multistart_include_direct
        )
        self.optimizer_multistart_lateral_offsets = self._parse_float_list(
            optimizer_multistart_lateral_offsets
        )
        self.optimizer_initial_routes = self._parse_initial_routes(optimizer_initial_routes_json)
        self.rng = np.random.default_rng(int(seed))

        self.camera = ObliqueCameraModel(
            cam_pos=camera_params['cam_pos'],
            look_at=camera_params['look_at'],
            img_width=camera_params['img_width'],
            img_height=camera_params['img_height'],
            fov_h_rad=camera_params['fov_h_rad'],
        )

        self.runtime_debug = bool(runtime_debug)
        self.use_visibility_model = bool(use_visibility_model)
        self._visibility_min_prob = 1e-4
        self.visibility_model = None
        self.use_nogo_cost = bool(use_nogo_cost)
        self.nogo_penalty_type = str(nogo_penalty_type or 'softplus').strip().lower()
        self.nogo_weight = float(max(nogo_weight, 0.0))
        self.nogo_safe_distance = float(max(nogo_safe_distance, 0.0))
        self.nogo_gaussian_sigma = float(max(nogo_gaussian_sigma, 1e-6))
        self.nogo_softplus_scale = float(max(nogo_softplus_scale, 1e-6))
        self.nogo_logbarrier_scale = float(max(nogo_logbarrier_scale, 1e-6))
        self.nogo_logbarrier_eps = float(max(nogo_logbarrier_eps, 1e-6))
        self.use_belief_nogo_cost = bool(use_belief_nogo_cost)
        self.nogo_belief_kappa = float(max(nogo_belief_kappa, 1e-6))
        self.nogo_cost_model = None
        self.collision_cost_model = None

        self.g_obs = self.camera.g_uv

        self.R_visible = np.diag([
            self.r_visible_uv ** 2,
            self.r_visible_uv ** 2,
        ])
        self.R_miss = np.diag([
            self.r_miss_uv ** 2,
            self.r_miss_uv ** 2,
        ])
        self.R = self.R_visible.copy()

        if self.use_visibility_model:
            vis_cfg = GPVisibilityMapConfig(
                artifact_path=str(visibility_artifact_path or ''),
                camera_pos=tuple(np.asarray(camera_params['cam_pos'], dtype=float).tolist()),
                target_height_m=float(visibility_target_height_m),
                min_prob=self._visibility_min_prob,
            )
            self.visibility_model = GPVisibilityMapModel(vis_cfg)

        if self.use_nogo_cost:
            nogo_cfg = NogoCostConfig(
                penalty_type=self.nogo_penalty_type,
                weight=self.nogo_weight,
                safe_distance=self.nogo_safe_distance,
                gaussian_sigma=self.nogo_gaussian_sigma,
                softplus_scale=self.nogo_softplus_scale,
                logbarrier_scale=self.nogo_logbarrier_scale,
                logbarrier_eps=self.nogo_logbarrier_eps,
                geometry_json=str(visibility_geometry_json or ''),
            )
            self.nogo_cost_model = NogoZoneCostModel(nogo_cfg)

        if str(collision_geometry_json or '').strip():
            collision_cfg = NogoCostConfig(
                penalty_type='softplus',
                weight=1.0,
                safe_distance=0.0,
                gaussian_sigma=1.0,
                softplus_scale=1.0,
                logbarrier_scale=1.0,
                logbarrier_eps=1e-3,
                geometry_json=str(collision_geometry_json or ''),
            )
            self.collision_cost_model = NogoZoneCostModel(collision_cfg)

        self.prev_controls_flat = None
        self._prev_goal_xy = None
        self._casadi_valgrad_cache = {}

    @staticmethod
    def _as_bool_like(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')

    @staticmethod
    def _parse_float_list(raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            values = raw
        else:
            text = str(raw).strip()
            if not text:
                return []
            if text.startswith('[') and text.endswith(']'):
                try:
                    values = json.loads(text)
                except json.JSONDecodeError:
                    values = text[1:-1].split(',')
            else:
                values = text.split(',')
        out = []
        for item in values:
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out.append(value)
        return out

    @staticmethod
    def _parse_initial_routes(raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            payload = raw
        else:
            text = str(raw).strip()
            if not text:
                return []
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return []
        routes = []
        if not isinstance(payload, (list, tuple)):
            return routes
        for idx, route in enumerate(payload):
            if not isinstance(route, dict):
                continue
            name = str(route.get('name', f'route_{idx}')).strip() or f'route_{idx}'
            waypoints_raw = route.get('waypoints', [])
            waypoints = []
            if isinstance(waypoints_raw, (list, tuple)):
                for wp in waypoints_raw:
                    try:
                        arr = np.asarray(wp, dtype=float).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                        waypoints.append((float(arr[0]), float(arr[1])))
            if waypoints:
                routes.append({'name': name, 'waypoints': waypoints})
        return routes

    def _runtime_debug_print(self, message):
        if not self.runtime_debug:
            return
        try:
            print(message, flush=True)
        except (BrokenPipeError, OSError):
            # Launch wrappers can close stdout while planner work is still running.
            pass

    def process_noise(self, dt=None):
        step_dt = self.dt if dt is None else float(dt)
        return unicycle_process_noise(
            self.process_noise_xy, self.process_noise_theta, step_dt, base_dt=self.dt
        )

    def predict(self, m, S, u, dt=None):
        step_dt = self.dt if dt is None else float(dt)
        m_next = unicycle_step(m, u, step_dt)
        F = unicycle_jacobian(m, u, step_dt)
        Q = self.process_noise(step_dt)
        S_next = F @ S @ F.T + Q
        return m_next, S_next

    def _approx_fn_for_method(self, method):
        m = str(method).upper()
        if m == 'ET1':
            return ET1
        if m == 'ET2':
            return ET2
        if m == 'UT':
            return UT
        raise RuntimeError(f"Unknown observation approximation method: {method}")

    def approx_observation(self, m, S, method=None, R_override=None):
        approx_method = self.approx_method if method is None else method
        fn = self._approx_fn_for_method(approx_method)
        R_use = self.R if R_override is None else np.asarray(R_override, dtype=float)
        return fn(m, S, self.g_obs, addmatrix=R_use, forceHermitian=True)

    def visibility_probability(self, m):
        if (not self.use_visibility_model) or (self.visibility_model is None):
            return 1.0
        p = float(self.visibility_model.prob_state_np(m))
        return float(np.clip(p, self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    def visibility_probability_belief(self, m, S):
        if (not self.use_visibility_model) or (self.visibility_model is None):
            return 1.0
        mean_xy = np.asarray(m[:2], dtype=float).reshape(2)
        cov_xy = np.asarray(S[:2, :2], dtype=float).reshape(2, 2)
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        kappa = max(float(self.visibility_sigma_kappa), 1e-6)
        scale = np.sqrt(2.0 + kappa)
        chol = np.linalg.cholesky(cov_xy + 1e-9 * np.eye(2))
        spread = scale * chol
        sigma_points = np.vstack([
            mean_xy,
            mean_xy + spread[:, 0],
            mean_xy - spread[:, 0],
            mean_xy + spread[:, 1],
            mean_xy - spread[:, 1],
        ])
        weights = np.array(
            [kappa / (2.0 + kappa)] + [1.0 / (2.0 * (2.0 + kappa))] * 4,
            dtype=float,
        )
        samples = np.column_stack([sigma_points[:, 0], sigma_points[:, 1], np.full(5, float(m[2]), dtype=float)])
        probs = np.clip(
            np.asarray([self.visibility_model.prob_state_np(sample) for sample in samples], dtype=float),
            self._visibility_min_prob,
            1.0 - self._visibility_min_prob,
        )
        return float(np.clip(np.sum(weights * probs), self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    @staticmethod
    def _smoothstep(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _softplus(x):
        x = float(x)
        if x > 40.0:
            return x
        if x < -40.0:
            return math.exp(x)
        return math.log1p(math.exp(x))

    def _visibility_effective_score(self, p_vis):
        return float(np.clip(p_vis, self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    def _blend_observation_covariance(self, trust):
        trust = float(np.clip(trust, self._visibility_min_prob, 1.0 - self._visibility_min_prob))
        # Precision Blending: var = 1 / (trust/var_vis + (1-trust)/var_miss)
        # This makes R much more responsive to visibility improvements.
        visible_prec = 1.0 / np.maximum(np.diag(self.R_visible), 1e-6)
        miss_prec = 1.0 / np.maximum(np.diag(self.R_miss), 1e-6)
        blended_prec = trust * visible_prec + (1.0 - trust) * miss_prec
        plan_var = 1.0 / np.maximum(blended_prec, 1e-9)
        return np.diag(plan_var).astype(float)

    def goal_obs_cov_for_progress(self, progress):
        progress_fast = float(np.clip(progress, 0.0, 1.0)) ** self.goal_tightening_power
        a = self._smoothstep(progress_fast)
        sigma_u = (1.0 - a) * self.goal_prior_u_std_start + a * self.goal_prior_u_std_final
        sigma_v = (1.0 - a) * self.goal_prior_v_std_start + a * self.goal_prior_v_std_final
        return np.diag([sigma_u ** 2, sigma_v ** 2]).astype(float)

    def planning_visibility_diagnostics(self, m, S):
        # NOTE:
        # This function is planner-facing only.
        # It defines predictive observability/trust for route evaluation.
        # It must not be used as the sole source of measurement-update trust.
        p_vis = self.visibility_probability_belief(m, S)
        p_vis_eff = self._visibility_effective_score(p_vis)
        if (not self.use_visibility_model) or (self.visibility_model is None):
            p_vis = 1.0
            p_vis_eff = 1.0
        R_plan = self._blend_observation_covariance(p_vis_eff)
        return {
            'p_vis': float(p_vis),
            'p_vis_eff': float(p_vis_eff),
            'R_plan': np.asarray(R_plan, dtype=float),
            'r_plan_u_std': float(np.sqrt(max(R_plan[0, 0], 0.0))),
            'r_plan_v_std': float(np.sqrt(max(R_plan[1, 1], 0.0))),
        }

    def obstacle_penalty(self, m, S=None):
        penalty = 0.0
        if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
            if self.use_belief_nogo_cost and S is not None:
                penalty += float(self.nogo_cost_model.penalty_belief_np(
                    m,
                    S,
                    kappa=self.nogo_belief_kappa,
                ))
            else:
                penalty += float(self.nogo_cost_model.penalty_state_np(m))
        return float(penalty)

    def collision_signed_distance_state_np(self, m):
        if self.collision_cost_model is None:
            return float('inf')
        return float(self.collision_cost_model.signed_distance_state_np(m))

    def collision_clearance_state_np(self, m):
        signed_d = self.collision_signed_distance_state_np(m)
        if not math.isfinite(signed_d):
            return float('inf')
        return float(signed_d - self.robot_collision_radius_m)

    def collision_penetration_state_np(self, m):
        clearance = self.collision_clearance_state_np(m)
        if not math.isfinite(clearance):
            return 0.0
        return float(max(-clearance, 0.0))

    def _goal_distance_xy(self, state_xy, goal_xy):
        state_xy = np.asarray(state_xy, dtype=float).reshape(2)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        return float(np.linalg.norm(state_xy - goal_xy))

    def _trajectory_plan_diagnostics(self, m0, S0, controls, goal_xy):
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        controls = np.asarray(controls, dtype=float).reshape(self.horizon, 2)
        m = np.asarray(m0, dtype=float).copy()
        S = np.asarray(S0, dtype=float).copy()
        p_vis_values = []
        ambiguity_std_values = []
        min_clearance = float('inf')

        for u in controls:
            m, S = self.predict(m, S, u)
            vis_diag = self.planning_visibility_diagnostics(m, S)
            p_vis_values.append(float(vis_diag['p_vis']))
            ambiguity_std_values.append(
                float(max(vis_diag['r_plan_u_std'], vis_diag['r_plan_v_std']))
            )
            min_clearance = min(min_clearance, self.collision_clearance_state_np(m))

        current_goal_distance = self._goal_distance_xy(np.asarray(m0[:2], dtype=float), goal_xy)
        terminal_goal_distance = self._goal_distance_xy(np.asarray(m[:2], dtype=float), goal_xy)
        terminal_goal_progress = float(current_goal_distance - terminal_goal_distance)
        low_pvis_fraction = (
            float(np.mean(np.asarray(p_vis_values, dtype=float) < 0.2))
            if p_vis_values else math.nan
        )
        ambiguity_threshold = math.sqrt(max(self.r_visible_uv, 1e-6) * max(self.r_miss_uv, 1e-6))
        high_ambiguity_fraction = (
            float(np.mean(np.asarray(ambiguity_std_values, dtype=float) >= ambiguity_threshold))
            if ambiguity_std_values else math.nan
        )
        rollout_valid = bool((not math.isfinite(min_clearance)) or min_clearance >= 0.0)
        invalid_reason = ''
        if not rollout_valid:
            invalid_reason = 'predicted_collision_geometry'
        return {
            'terminal_goal_distance_pred': float(terminal_goal_distance),
            'terminal_goal_progress_m': float(terminal_goal_progress),
            'fraction_horizon_low_pvis': low_pvis_fraction,
            'fraction_horizon_high_ambiguity': high_ambiguity_fraction,
            'min_predicted_obstacle_distance_m': (
                float(min_clearance) if math.isfinite(min_clearance) else math.inf
            ),
            'rollout_valid': rollout_valid,
            'invalid_reason': invalid_reason,
        }

    def _resolve_plan_problem(self, m0, goal_xy):
        goal_theta = 0.0
        goal_state = self._goal_state(goal_xy, goal_theta)

        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity

        goal_obs = None
        goal_obs_cov = None
        if use_observation_risk or use_ambiguity_term:
            goal_obs = self._goal_obs(goal_state)
            goal_obs_cov = self._goal_obs_cov()

        return (
            goal_state,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
        )

    def evaluate_rollout_controls(self, m0, S0, goal_xy, controls, *, progress_index=0.0):
        """Evaluate one fixed control rollout using the same accounting as planner selection."""
        (
            goal_state,
            goal_obs,
            goal_obs_cov,
            _use_observation_risk,
            _use_ambiguity_term,
        ) = self._resolve_plan_problem(m0, goal_xy)
        controls = np.asarray(controls, dtype=float).reshape(self.horizon, 2)
        total_cost, metrics = self._evaluate_controls(
            controls.reshape(-1),
            np.asarray(m0, dtype=float),
            np.asarray(S0, dtype=float),
            goal_state,
            goal_obs,
            goal_obs_cov,
            True,
            progress_index=float(max(progress_index, 0.0)),
        )
        states = rollout_unicycle(np.asarray(m0, dtype=float), controls, self.dt)
        plan_diag = self._trajectory_plan_diagnostics(m0, S0, controls, goal_xy)

        p_vis_values = []
        p_vis_eff_values = []
        r_u_values = []
        r_v_values = []
        m = np.asarray(m0, dtype=float).copy()
        S = np.asarray(S0, dtype=float).copy()
        for u in controls:
            m, S = self.predict(m, S, u)
            vis_diag = self.planning_visibility_diagnostics(m, S)
            p_vis_values.append(float(vis_diag['p_vis']))
            p_vis_eff_values.append(float(vis_diag['p_vis_eff']))
            r_u_values.append(float(vis_diag['r_plan_u_std']))
            r_v_values.append(float(vis_diag['r_plan_v_std']))

        return {
            'controls': controls,
            'states': states,
            'total_cost': float(total_cost),
            'risk_cost': float(metrics.get('risk_cost', math.nan)),
            'ambiguity_cost': float(metrics.get('ambiguity_cost', math.nan)),
            'control_cost': float(metrics.get('control_cost', math.nan)),
            'obstacle_cost': float(metrics.get('obstacle_cost', 0.0)),
            'risk_mean': float(metrics.get('risk_mean', 0.0)),
            'risk_cov_trace': float(metrics.get('risk_cov_trace', 0.0)),
            'risk_cov_logdet': float(metrics.get('risk_cov_logdet', 0.0)),
            'delta_risk_visibility': float(metrics.get('delta_risk_visibility', 0.0)),
            'delta_ambiguity_visibility': float(metrics.get('delta_ambiguity_visibility', 0.0)),
            'terminal_goal_distance_pred': float(plan_diag['terminal_goal_distance_pred']),
            'terminal_goal_progress_m': float(plan_diag['terminal_goal_progress_m']),
            'fraction_horizon_low_pvis': float(plan_diag['fraction_horizon_low_pvis']),
            'fraction_horizon_high_ambiguity': float(plan_diag['fraction_horizon_high_ambiguity']),
            'min_predicted_obstacle_distance_m': float(plan_diag['min_predicted_obstacle_distance_m']),
            'rollout_valid': bool(plan_diag['rollout_valid']),
            'invalid_reason': str(plan_diag['invalid_reason']),
            'mean_p_vis_plan': float(np.mean(p_vis_values)) if p_vis_values else math.nan,
            'mean_p_vis_plan_eff': float(np.mean(p_vis_eff_values)) if p_vis_eff_values else math.nan,
            'mean_r_plan_u_std': float(np.mean(r_u_values)) if r_u_values else math.nan,
            'mean_r_plan_v_std': float(np.mean(r_v_values)) if r_v_values else math.nan,
        }

    def _nominal_controls_flat(self):
        return np.zeros(self.horizon * 2, dtype=float)

    def _shift_controls_flat(self, controls_flat, shift_steps=None):
        controls = np.asarray(controls_flat, dtype=float).reshape(self.horizon, 2)
        shift = int(self.optimizer_warm_start_shift_steps if shift_steps is None else shift_steps)
        shift = max(1, shift)
        if self.horizon <= 1:
            return controls.reshape(-1).copy()
        if shift >= self.horizon:
            shifted = np.repeat(controls[-1:, :], self.horizon, axis=0)
        else:
            shifted = np.empty_like(controls)
            shifted[:-shift, :] = controls[shift:, :]
            shifted[-shift:, :] = controls[-1, :]
        return shifted.reshape(-1)

    def _initial_controls_flat(self):
        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            return self._shift_controls_flat(self.prev_controls_flat)
        return self._nominal_controls_flat()

    def _controls_for_waypoints(self, start_xy_yaw, waypoints):
        """Crude unicycle route seed used only as optimizer initialization."""
        controls = np.zeros((self.horizon, 2), dtype=float)
        wps = [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]
        if not wps:
            return controls.reshape(-1)

        m = np.asarray(start_xy_yaw, dtype=float).reshape(-1)[:3].copy()
        S_dummy = np.eye(3, dtype=float) * 1e-6
        waypoint_idx = 0
        for k in range(self.horizon):
            if waypoint_idx >= len(wps):
                break
            target = wps[waypoint_idx]
            d = target - m[:2]
            if float(np.linalg.norm(d)) < 0.18 and waypoint_idx < len(wps) - 1:
                waypoint_idx += 1
                target = wps[waypoint_idx]
                d = target - m[:2]
            desired_yaw = math.atan2(float(d[1]), float(d[0]))
            yaw_err = wrap_angle(desired_yaw - float(m[2]))
            w = float(np.clip(yaw_err / max(self.dt, 1e-6), self.w_min, self.w_max))
            v = self.v_max if abs(yaw_err) < 0.65 else 0.0
            controls[k] = [float(np.clip(v, self.v_min, self.v_max)), w]
            m, _ = self.predict(m, S_dummy, controls[k])
        return controls.reshape(-1)

    def _build_multistart_candidates(self, m0, goal_xy):
        """Build optional optimizer seeds; these are not mission waypoints."""
        candidates: list[tuple[str, np.ndarray]] = []
        if not self.optimizer_multistart:
            return candidates

        start = np.asarray(m0, dtype=float).reshape(-1)[:3]
        goal = np.asarray(goal_xy, dtype=float).reshape(2)

        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            candidates.append(('cold', self._nominal_controls_flat()))

        if self.optimizer_multistart_include_direct:
            candidates.append(('direct_goal', self._controls_for_waypoints(start, [goal])))

        dvec = goal - start[:2]
        dist = float(np.linalg.norm(dvec))
        if dist > 1e-3:
            unit = dvec / dist
            perp = np.array([-unit[1], unit[0]], dtype=float)
            mid = start[:2] + 0.5 * dvec
            for offset in self.optimizer_multistart_lateral_offsets:
                waypoint = mid + float(offset) * perp
                candidates.append((
                    f'lateral_{float(offset):+.2f}',
                    self._controls_for_waypoints(start, [waypoint, goal]),
                ))

        for route in self.optimizer_initial_routes:
            name = str(route.get('name', 'route'))
            waypoints = list(route.get('waypoints', []))
            if waypoints:
                candidates.append((
                    f'route:{name}',
                    self._controls_for_waypoints(start, waypoints),
                ))

        return candidates

    def _objective_scales(self, controls_flat, m0, S0, goal_state, goal_obs, goal_obs_cov):
        del controls_flat, m0, S0, goal_state, goal_obs, goal_obs_cov
        return 1.0, 1.0

    @staticmethod
    def _scaled_objective_from_metrics(metrics, objective_scales):
        risk_scale, ambiguity_scale = objective_scales
        risk_term = float(metrics.get('risk_cost', 0.0)) / float(max(risk_scale, 1e-9))
        ambiguity_term = float(metrics.get('ambiguity_cost', 0.0)) / float(max(ambiguity_scale, 1e-9))
        return (
            risk_term
            + ambiguity_term
            + float(metrics.get('control_cost', 0.0))
            + float(metrics.get('obstacle_cost', 0.0))
        )

    def _evaluate_candidate_controls(
        self,
        controls_flat,
        m0,
        S0,
        goal_state,
        goal_obs,
        goal_obs_cov,
        objective_scales,
        *,
        progress_index=0.0,
    ):
        controls_flat = np.asarray(controls_flat, dtype=float).reshape(self.horizon * 2)
        controls = controls_flat.reshape(self.horizon, 2)
        total_cost, metrics = self._evaluate_controls(
            controls_flat,
            m0,
            S0,
            goal_state,
            goal_obs,
            goal_obs_cov,
            True,
            progress_index=progress_index,
        )
        scaled_total = self._scaled_objective_from_metrics(metrics, objective_scales)
        return {
            'controls_flat': controls_flat,
            'total_cost': float(total_cost),
            'metrics': {str(k): float(v) for k, v in metrics.items()},
            'scaled_total': float(scaled_total),
        }

    def _autodiff_cache_key(
        self,
        goal_state,
        goal_obs,
        use_observation_risk,
        use_ambiguity_term,
    ):
        del goal_state
        return (
            self.approx_method,
            bool(use_ambiguity_term),
            bool(use_observation_risk),
            bool(self.use_visibility_model),
            float(self.control_weight),
            float(self.risk_weight_obs),
            float(self.ambiguity_weight),
            float(self.r_visible_uv),
            float(self.r_miss_uv),
            float(self.visibility_sigma_kappa),
            float(self.goal_prior_u_std_start),
            float(self.goal_prior_v_std_start),
            float(self.goal_prior_u_std_final),
            float(self.goal_prior_v_std_final),
            float(self.goal_tightening_power),
            int(self.goal_progress_n_steps),
            float(self.observation_risk_scale),
            float(self.ambiguity_term_scale),
            float(self.discount_gamma),
            float(self.robot_collision_radius_m),
            bool(self.use_nogo_cost),
            bool(self.use_belief_nogo_cost),
            float(self.nogo_belief_kappa),
            tuple(self.nogo_cost_model.signature) if self.nogo_cost_model is not None else (),
            tuple(self.collision_cost_model.signature) if self.collision_cost_model is not None else (),
            int(self.horizon),
            float(self.dt),
            int(np.asarray(goal_obs, dtype=float).shape[0]),
            tuple(self.visibility_model.signature) if self.visibility_model is not None else (),
        )

    def _get_casadi_valgrad(
        self,
        goal_state,
        goal_obs,
        *,
        use_observation_risk,
        use_ambiguity_term,
    ):
        from planning.core import casadi_efe

        if not casadi_efe.casadi_available():
            raise RuntimeError("CasADi is not available")

        if goal_obs is None:
            goal_obs = self._goal_obs(goal_state)

        cache_key = self._autodiff_cache_key(
            goal_state,
            goal_obs,
            use_observation_risk,
            use_ambiguity_term,
        )
        valgrad = self._casadi_valgrad_cache.get(cache_key)
        self._runtime_debug_print(
            "[planner_debug] CasADi valgrad cache "
            f"{'hit' if valgrad is not None else 'miss'} "
            f"(horizon={self.horizon}, approx={self.approx_method})"
        )
        if valgrad is None:
            build_start = time.perf_counter()
            p_vis_ca = None
            if self.use_visibility_model and self.visibility_model is not None:
                p_vis_ca = self.visibility_model.make_prob_state_casadi()
            nogo_cost_ca = None
            nogo_belief_cost_ca = None
            if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
                if self.use_belief_nogo_cost:
                    nogo_belief_cost_ca = self.nogo_cost_model.make_penalty_belief_casadi(
                        kappa=self.nogo_belief_kappa,
                    )
                else:
                    nogo_cost_ca = self.nogo_cost_model.make_penalty_state_casadi()
            params_ca = casadi_efe.CasadiEfeParams(
                Q=np.array(self.process_noise(self.dt), dtype=float),
                R_visible=np.array(self.R_visible, dtype=float),
                R_miss=np.array(self.R_miss, dtype=float),
                control_weight=float(self.control_weight),
                risk_scale=float(self.risk_weight_obs * self.observation_risk_scale if use_observation_risk else 0.0),
                ambiguity_scale=float(self.ambiguity_weight * self.ambiguity_term_scale if use_ambiguity_term else 0.0),
                discount_gamma=float(self.discount_gamma),
                visibility_sigma_kappa=float(self.visibility_sigma_kappa),
                goal_prior_u_std_start=float(self.goal_prior_u_std_start),
                goal_prior_v_std_start=float(self.goal_prior_v_std_start),
                goal_prior_u_std_final=float(self.goal_prior_u_std_final),
                goal_prior_v_std_final=float(self.goal_prior_v_std_final),
                goal_tightening_power=float(self.goal_tightening_power),
                goal_progress_n_steps=int(self.goal_progress_n_steps),
                use_belief_nogo_cost=bool(self.use_belief_nogo_cost),
                time_horizon=int(self.horizon),
                dt=float(self.dt),
                Du=2,
            )
            valgrad = casadi_efe.make_efe_valgrad_fn(
                params_ca,
                self.camera.H,
                approx=self.approx_method,
                p_vis_state=p_vis_ca,
                nogo_cost=nogo_cost_ca,
                nogo_belief_cost=nogo_belief_cost_ca,
            )
            self._casadi_valgrad_cache[cache_key] = valgrad
            self._runtime_debug_print(
                "[planner_debug] CasADi valgrad function prepared in "
                f"{(time.perf_counter() - build_start) * 1000.0:.1f} ms"
            )

        return valgrad

    def observation_model_with_visibility(self, m_pred, S_pred):
        """Visibility-aware measurement shaping used by objective and correction."""
        S_pred = np.asarray(S_pred, dtype=float)
        if (not self.use_visibility_model) or (self.visibility_model is None):
            R_eff = np.asarray(self.R_visible, dtype=float)
            S_eff = S_pred.copy()
            return 1.0, R_eff, S_eff, 1.0

        diag = self.planning_visibility_diagnostics(m_pred, S_pred)
        R_eff = np.asarray(diag['R_plan'], dtype=float)
        S_eff = S_pred.copy()
        return float(diag['p_vis']), R_eff, S_eff, 1.0

    def _goal_state(self, goal_xy, theta):
        return np.array([goal_xy[0], goal_xy[1], theta], dtype=float)

    def _goal_obs(self, goal_state):
        return np.asarray(self.g_obs(goal_state), dtype=float)

    def _goal_obs_cov(self):
        return self.goal_obs_cov_for_progress(0.0)

    def _evaluate_controls(
        self,
        controls_flat,
        m0,
        S0,
        goal_state,
        goal_obs,
        goal_obs_cov,
        return_metrics=False,
        *,
        progress_index=0.0,
        R_baseline_override=None,
    ):
        del goal_obs_cov
        controls_flat = np.asarray(controls_flat, dtype=float)
        assert controls_flat.size == self.horizon * 2, f"controls_flat size {controls_flat.size} != expected {self.horizon * 2}"
        controls = controls_flat.reshape(self.horizon, 2)

        m = m0.copy()
        S = S0.copy()
        total_risk = 0.0
        total_amb = 0.0
        total_control = 0.0
        total_obstacle = 0.0
        total_risk_mean = 0.0
        total_risk_cov_trace = 0.0
        total_risk_cov_logdet = 0.0
        total_risk_const = 0.0
        total_delta_risk_visibility = 0.0
        total_delta_ambiguity_visibility = 0.0
        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity
        goal_xy = np.asarray(goal_state[:2], dtype=float).reshape(2)
        R_good = np.asarray(
            R_baseline_override
            if R_baseline_override is not None
            else np.diag([float(self.r_visible_uv) ** 2, float(self.r_visible_uv) ** 2]),
            dtype=float,
        )

        for t in range(self.horizon):
            u = controls[t]
            m, S = self.predict(m, S, u)
            vis_diag = self.planning_visibility_diagnostics(m, S)
            p_vis = vis_diag['p_vis']
            R_plan = vis_diag['R_plan']
            mu_y = Sigma_y = Gamma = None
            if use_observation_risk or use_ambiguity_term:
                mu_y, Sigma_y, Gamma = self.approx_observation(
                    m,
                    S,
                    method=self.approx_method,
                    R_override=R_plan,
                )
            weight_t = self.discount_gamma ** t
            observation_risk = 0.0
            baseline_risk = 0.0
            ambiguity_current = 0.0
            ambiguity_baseline = 0.0
            Sigma_good = None
            Gamma_good = None
            if use_observation_risk and mu_y is not None:
                goal_cov_t = self.goal_obs_cov_for_progress(
                    (float(progress_index) + float(t)) / max(self.goal_progress_n_steps, 1)
                )
                risk_parts = risk_components(mu_y, Sigma_y, (goal_obs, goal_cov_t))
                risk_scale = self.risk_weight_obs * self.observation_risk_scale
                observation_risk = risk_scale * risk_parts['total']
                total_risk_mean += weight_t * risk_scale * risk_parts['mean']
                total_risk_cov_trace += weight_t * risk_scale * risk_parts['cov_trace']
                total_risk_cov_logdet += weight_t * risk_scale * risk_parts['cov_logdet']
                total_risk_const += weight_t * risk_scale * risk_parts['const']
                mu_good, Sigma_good, Gamma_good = self.approx_observation(
                    m,
                    S,
                    method=self.approx_method,
                    R_override=R_good,
                )
                baseline_parts = risk_components(mu_good, Sigma_good, (goal_obs, goal_cov_t))
                baseline_risk = risk_scale * baseline_parts['total']
            total_risk += weight_t * observation_risk
            if use_ambiguity_term and Sigma_y is not None:
                ambiguity_scale = self.ambiguity_weight * self.ambiguity_term_scale
                ambiguity_current = ambiguity_scale * ambiguity(Sigma_y, Gamma, S)
                if Sigma_good is None or Gamma_good is None:
                    _mu_good, Sigma_good, Gamma_good = self.approx_observation(
                        m,
                        S,
                        method=self.approx_method,
                        R_override=R_good,
                    )
                ambiguity_baseline = ambiguity_scale * ambiguity(Sigma_good, Gamma_good, S)
                total_amb += weight_t * ambiguity_current
            total_delta_risk_visibility += weight_t * (observation_risk - baseline_risk)
            total_delta_ambiguity_visibility += weight_t * (ambiguity_current - ambiguity_baseline)
            total_obstacle += weight_t * self.obstacle_penalty(m, S)
            total_control += weight_t * self.control_weight * float(u[0] ** 2 + u[1] ** 2)

        total = total_risk + total_amb + total_control + total_obstacle
        if return_metrics:
            return total, {
                'risk_cost': float(total_risk),
                'ambiguity_cost': float(total_amb),
                'control_cost': float(total_control),
                'obstacle_cost': float(total_obstacle),
                'risk_mean': float(total_risk_mean),
                'risk_cov_trace': float(total_risk_cov_trace),
                'risk_cov_logdet': float(total_risk_cov_logdet),
                'risk_const': float(total_risk_const),
                'delta_risk_visibility': float(total_delta_risk_visibility),
                'delta_ambiguity_visibility': float(total_delta_ambiguity_visibility),
            }
        return total

    def plan(self, m0, S0, goal_xy, *, progress_index=0.0):
        t_plan_start = time.perf_counter()
        progress_index = float(max(progress_index, 0.0))

        # Reset warm start when goal changes by more than 0.5 m to prevent
        # stale plans from creating a stuck local minimum after goal switch.
        goal_xy_arr = np.asarray(goal_xy, dtype=float).reshape(2)
        if self._prev_goal_xy is not None:
            if float(np.linalg.norm(goal_xy_arr - self._prev_goal_xy)) > 0.5:
                self.prev_controls_flat = None
        self._prev_goal_xy = goal_xy_arr.copy()

        (
            goal_state,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
        ) = self._resolve_plan_problem(m0, goal_xy)

        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((self.w_min, self.w_max))

        x0_default = self._initial_controls_flat()
        init_candidates: list[tuple[str, np.ndarray]] = [
            ('warm_or_cold', np.asarray(x0_default, dtype=float)),
        ]
        for ms_name, ms_controls in self._build_multistart_candidates(m0, goal_xy):
            init_candidates.append((ms_name, np.asarray(ms_controls, dtype=float)))

        objective_scales = self._objective_scales(
            init_candidates[0][1], m0, S0, goal_state, goal_obs, goal_obs_cov,
        )
        best_candidate = None
        best_init_name = ''
        backend_used = 'casadi'
        optimizer_success = False
        optimizer_status = 0
        optimizer_nit = 0
        optimizer_nfev = 0
        optimizer_message = ''

        try:
            fg_calls = {'count': 0}
            goal_obs_eval = np.asarray(goal_obs if goal_obs is not None else self._goal_obs(goal_state), dtype=float)
            valgrad = self._get_casadi_valgrad(
                goal_state,
                goal_obs,
                use_observation_risk=use_observation_risk,
                use_ambiguity_term=use_ambiguity_term,
            )

            def objective(u):
                u_arr = np.asarray(u, dtype=float)
                start = time.perf_counter()
                val_out, grad_out = valgrad(
                    u_arr,
                    m0,
                    S0,
                    goal_obs_eval,
                    np.asarray(goal_xy, dtype=float).reshape(2),
                    progress_index,
                )
                if fg_calls['count'] == 0:
                    self._runtime_debug_print(
                        "[planner_debug] First CasADi objective/gradient eval returned in "
                        f"{(time.perf_counter() - start) * 1000.0:.1f} ms "
                        f"with J={val_out:.3f}, grad_norm={np.linalg.norm(grad_out):.3f}"
                    )
                fg_calls['count'] += 1
                return val_out, grad_out

            minimize_start = time.perf_counter()
            self._runtime_debug_print(
                "[planner_debug] Starting CasADi-backed scipy.optimize.minimize "
                f"(maxiter={self.optimizer_maxiter}, maxfun={self.optimizer_maxfun}, ftol={self.optimizer_ftol}, "
                f"init_candidates={len(init_candidates)})"
            )

            for init_name, x_init in init_candidates:
                # Reset the fg call counter per attempt so debug timing is per-attempt.
                attempt_start = time.perf_counter()
                try:
                    result = minimize(
                        objective,
                        np.asarray(x_init, dtype=float),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': self.optimizer_maxiter,
                            'maxfun': self.optimizer_maxfun,
                            'ftol': self.optimizer_ftol,
                            'gtol': self.optimizer_gtol,
                        },
                    )
                except Exception as exc:
                    self._runtime_debug_print(
                        f"[planner_debug] init={init_name!s} threw {type(exc).__name__}: {exc}"
                    )
                    continue
                if result.x is None or not np.all(np.isfinite(np.asarray(result.x, dtype=float))):
                    self._runtime_debug_print(
                        f"[planner_debug] init={init_name!s} returned non-finite solution; skip"
                    )
                    continue
                candidate = self._evaluate_candidate_controls(
                    np.asarray(result.x, dtype=float),
                    m0,
                    S0,
                    goal_state,
                    goal_obs,
                    goal_obs_cov,
                    objective_scales,
                    progress_index=progress_index,
                )
                ctrls_attempt = np.asarray(candidate['controls_flat'], dtype=float).reshape(self.horizon, 2)
                diag_attempt = self._trajectory_plan_diagnostics(m0, S0, ctrls_attempt, goal_xy)
                cand_valid = bool(diag_attempt['rollout_valid'])
                source_label = (
                    f'solver:shifted_warm_start' if (init_name == 'warm_or_cold' and self.prev_controls_flat is not None)
                    else f'solver:{init_name}'
                )
                candidate.update({
                    'source': source_label,
                    'rollout_valid': cand_valid,
                    'optimizer_result': result,
                })
                self._runtime_debug_print(
                    f"[planner_debug] init={init_name!s} solver finished "
                    f"J={candidate['total_cost']:.3f}, valid={cand_valid}, "
                    f"success={bool(result.success)}, status={int(result.status)}, "
                    f"nit={int(getattr(result, 'nit', 0) or 0)}, "
                    f"nfev={int(getattr(result, 'nfev', 0) or 0)}, "
                    f"dt={(time.perf_counter() - attempt_start) * 1000.0:.0f}ms"
                )

                # Candidate validity is diagnostic-only; the optimizer objective decides.
                if best_candidate is None:
                    keep = True
                else:
                    keep = float(candidate['total_cost']) < float(best_candidate['total_cost'])
                if keep:
                    best_candidate = candidate
                    best_init_name = init_name
                    optimizer_success = bool(result.success)
                    optimizer_status = int(result.status)
                    optimizer_nit = int(getattr(result, 'nit', 0) or 0)
                    optimizer_nfev = int(getattr(result, 'nfev', 0) or 0)
                    optimizer_message = str(result.message or '')

            if best_candidate is None:
                raise RuntimeError("Planner optimizer returned no finite solution from any init")
            self._runtime_debug_print(
                f"[planner_debug] Best optimizer init={best_init_name!s} "
                f"J={best_candidate['total_cost']:.3f}, valid={best_candidate.get('rollout_valid', False)}"
            )
            self._runtime_debug_print(
                "[planner_debug] CasADi-backed minimize finished in "
                f"{(time.perf_counter() - minimize_start) * 1000.0:.1f} ms "
                f"(shared_fg_evals={fg_calls['count']})"
            )
        except Exception as exc:
            raise RuntimeError(
                "Planner optimization failed "
                f"(backend={backend_used}, approx={self.approx_method}, "
                f"horizon={self.horizon}, dt={self.dt}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if best_candidate is None:
            raise RuntimeError("Planner produced no candidate solution")

        best_controls_flat = np.asarray(best_candidate['controls_flat'], dtype=float)
        self.prev_controls_flat = np.array(best_controls_flat, dtype=float)
        best_controls = self.prev_controls_flat.reshape(self.horizon, 2)
        total_cost = float(best_candidate['total_cost'])
        metrics = dict(best_candidate['metrics'])
        vis_diag = self.planning_visibility_diagnostics(m0, S0)

        states = rollout_unicycle(m0, best_controls, self.dt)
        plan_diag = self._trajectory_plan_diagnostics(m0, S0, best_controls, goal_xy)
        selected_source = str(best_candidate.get('source', ''))
        solve_time_s = float(max(time.perf_counter() - t_plan_start, 0.0))
        return PlanResult(
            controls=best_controls,
            states=states,
            total_cost=float(total_cost),
            risk_cost=float(metrics.get('risk_cost', 0.0)),
            ambiguity_cost=float(metrics.get('ambiguity_cost', 0.0)),
            control_cost=float(metrics.get('control_cost', 0.0)),
            obstacle_cost=float(metrics.get('obstacle_cost', 0.0)),
            risk_mean=float(metrics.get('risk_mean', 0.0)),
            risk_cov_trace=float(metrics.get('risk_cov_trace', 0.0)),
            risk_cov_logdet=float(metrics.get('risk_cov_logdet', 0.0)),
            delta_risk_visibility=float(metrics.get('delta_risk_visibility', 0.0)),
            delta_ambiguity_visibility=float(metrics.get('delta_ambiguity_visibility', 0.0)),
            backend=str(backend_used),
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_nit=optimizer_nit,
            optimizer_nfev=optimizer_nfev,
            optimizer_message=optimizer_message,
            solve_time_s=solve_time_s,
            selected_source=selected_source,
            p_vis_plan=float(vis_diag['p_vis']),
            p_vis_plan_eff=float(vis_diag['p_vis_eff']),
            r_plan_u_std=float(vis_diag['r_plan_u_std']),
            r_plan_v_std=float(vis_diag['r_plan_v_std']),
            terminal_goal_distance_pred=float(plan_diag['terminal_goal_distance_pred']),
            terminal_goal_progress_m=float(plan_diag['terminal_goal_progress_m']),
            fraction_horizon_low_pvis=float(plan_diag['fraction_horizon_low_pvis']),
            fraction_horizon_high_ambiguity=float(plan_diag['fraction_horizon_high_ambiguity']),
            min_predicted_obstacle_distance_m=float(plan_diag['min_predicted_obstacle_distance_m']),
            rollout_valid=bool(plan_diag['rollout_valid']),
            invalid_reason=str(plan_diag['invalid_reason']),
        )
