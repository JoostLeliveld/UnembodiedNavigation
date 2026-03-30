"""Base planner classes (pure Python, no ROS)."""

from dataclasses import dataclass
import time
import numpy as np

from scipy.optimize import fmin_l_bfgs_b, minimize

from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import ET1, ET2, UT, ambiguity, risk
from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel
from planning.core.visibility_gp_map import GPVisibilityMapConfig, GPVisibilityMapModel
from planning.core.visibility_raycast_25d import Raycast25DVisibilityConfig, Raycast25DVisibilityModel
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
    visibility_cost: float = 0.0
    obstacle_cost: float = 0.0
    backend: str = "unknown"
    optimizer_success: bool = False
    optimizer_status: int = 0
    optimizer_nit: int = 0
    optimizer_nfev: int = 0
    optimizer_message: str = ""
    solve_time_s: float = 0.0
    used_fallback: bool = False
    selected_source: str = ""
    p_vis_plan: float = 1.0
    p_vis_plan_eff: float = 1.0
    r_plan_u_std: float = np.nan
    r_plan_v_std: float = np.nan


class UnicyclePlannerBase:
    """Shared unicycle planner logic. Subclasses define objective specifics."""

    APPROX_METHOD = 'ET2'  # ET1 | ET2 | UT | None
    USE_OBS_RISK = True
    USE_AMBIGUITY = True

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
        goal_sigma_xy,
        goal_sigma_theta,
        goal_sigma_uv,
        risk_weight_state,
        risk_weight_obs,
        ambiguity_weight,
        optimizer_maxiter,
        optimizer_gtol,
        optimizer_warm_start,
        optimizer_warm_start_shift_steps=1,
        approx_method=None,
        use_obs_risk=None,
        use_ambiguity=None,
        math_mode='legacy',
        optimizer_backend=None,
        seed,
        camera_params,
        use_visibility_model=False,
        visibility_model='raycast_25d',
        visibility_weight=0.0,
        visibility_map_min_x=-5.0,
        visibility_map_max_x=5.0,
        visibility_map_min_y=-5.0,
        visibility_map_max_y=5.0,
        visibility_map_nx=140,
        visibility_map_ny=120,
        visibility_gp_length_scale=1.4,
        visibility_gp_noise_var=0.15,
        visibility_prior_occ=0.005,
        visibility_beta=1.0,
        visibility_height_tau=0.08,
        visibility_ray_samples=120,
        visibility_target_height_m=0.0,
        visibility_geometry_json='',
        visibility_gp_seed=0,
        visibility_r_bad_uv=28.0,
        visibility_cov_pos_scale=2.0,
        visibility_cov_theta_scale=0.8,
        r_visible_uv=2.5,
        r_miss_uv=420.0,
        visibility_power=3.0,
        visibility_sigma_kappa=1.0,
        goal_prior_u_std_start=80.0,
        goal_prior_v_std_start=80.0,
        goal_prior_u_std_final=18.0,
        goal_prior_v_std_final=18.0,
        goal_tightening_power=0.45,
        goal_progress_n_steps=90,
        notebook_risk_scale=1.25,
        notebook_ambiguity_scale=0.20,
        discount_gamma=0.98,
        optimizer_maxfun=500,
        optimizer_ftol=1e-6,
        use_nogo_cost=False,
        nogo_penalty_type='softplus',
        nogo_weight=0.0,
        nogo_safe_distance=0.35,
        nogo_gaussian_sigma=0.25,
        nogo_softplus_scale=0.08,
        nogo_logbarrier_scale=0.25,
        nogo_logbarrier_eps=1e-3,
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

        self.goal_sigma_xy = float(goal_sigma_xy)
        self.goal_sigma_theta = float(goal_sigma_theta)
        self.goal_sigma_uv = float(goal_sigma_uv)
        if self.goal_sigma_uv <= 0.0:
            raise ValueError("goal_sigma_uv must be > 0.0; do not use 0.0 as a sentinel")

        self.math_mode = str(math_mode or 'legacy').strip().lower()
        if self.math_mode not in ('legacy', 'notebook_simple'):
            raise ValueError("math_mode must be 'legacy' or 'notebook_simple'")
        self.risk_weight_state = float(risk_weight_state)
        self.risk_weight_obs = float(risk_weight_obs)
        self.ambiguity_weight = float(ambiguity_weight)
        self.r_visible_uv = float(r_visible_uv)
        self.r_miss_uv = float(r_miss_uv)
        self.visibility_power = float(max(visibility_power, 1e-6))
        self.visibility_sigma_kappa = float(max(visibility_sigma_kappa, 1e-6))
        self.goal_prior_u_std_start = float(goal_prior_u_std_start)
        self.goal_prior_v_std_start = float(goal_prior_v_std_start)
        self.goal_prior_u_std_final = float(goal_prior_u_std_final)
        self.goal_prior_v_std_final = float(goal_prior_v_std_final)
        self.goal_tightening_power = float(max(goal_tightening_power, 1e-6))
        self.goal_progress_n_steps = int(max(goal_progress_n_steps, 1))
        self.notebook_risk_scale = float(notebook_risk_scale)
        self.notebook_ambiguity_scale = float(notebook_ambiguity_scale)
        self.discount_gamma = float(discount_gamma)

        if approx_method is None:
            self.approx_method = self.APPROX_METHOD
        else:
            self.approx_method = str(approx_method).upper()
        if self.math_mode == 'notebook_simple' and self.approx_method != 'ET1':
            raise ValueError("math_mode=notebook_simple requires approx_method='ET1'")

        if use_obs_risk is None:
            self.use_obs_risk = bool(self.USE_OBS_RISK)
        else:
            self.use_obs_risk = bool(use_obs_risk)

        if use_ambiguity is None:
            self.use_ambiguity = bool(self.USE_AMBIGUITY)
        else:
            self.use_ambiguity = bool(use_ambiguity)

        self.optimizer_maxiter = int(optimizer_maxiter)
        self.optimizer_gtol = float(optimizer_gtol)
        self.optimizer_warm_start = bool(optimizer_warm_start)
        self.optimizer_warm_start_shift_steps = int(max(optimizer_warm_start_shift_steps, 1))
        self.optimizer_maxfun = int(max(optimizer_maxfun, 1))
        self.optimizer_ftol = float(max(optimizer_ftol, 1e-12))

        self.rng = np.random.default_rng(int(seed))

        self.camera = ObliqueCameraModel(
            cam_pos=camera_params['cam_pos'],
            look_at=camera_params['look_at'],
            img_width=camera_params['img_width'],
            img_height=camera_params['img_height'],
            fov_h_rad=camera_params['fov_h_rad'],
        )

        self.optimizer_backend = str(optimizer_backend or 'auto').lower()
        self.runtime_debug = bool(runtime_debug)
        self.use_visibility_model = bool(use_visibility_model)
        self.visibility_model_name = str(visibility_model or 'none').strip().lower()
        self.visibility_weight = float(visibility_weight)
        self.visibility_cov_pos_scale = float(max(visibility_cov_pos_scale, 0.0))
        self.visibility_cov_theta_scale = float(max(visibility_cov_theta_scale, 0.0))
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
        self.nogo_cost_model = None

        self.g_obs = self.camera.g_uv

        self.R_visible = np.diag([
            self.r_visible_uv ** 2,
            self.r_visible_uv ** 2,
        ])
        self.R_miss = np.diag([
            self.r_miss_uv ** 2,
            self.r_miss_uv ** 2,
        ])
        if self.math_mode == 'notebook_simple':
            self.R = self.R_visible.copy()
            self.R_bad = self.R_miss.copy()
        else:
            self.R = np.diag([
                self.obs_noise_uv ** 2,
                self.obs_noise_uv ** 2,
            ])
            self.R_bad = np.diag([
                float(visibility_r_bad_uv) ** 2,
                float(visibility_r_bad_uv) ** 2,
            ])

        raycast_aliases = ('raycast_25d', 'raycast25d', 'raycast')
        gp_visibility_aliases = ('gp_visibility', 'gpvis')
        valid_visibility_models = raycast_aliases + gp_visibility_aliases + ('', 'none')
        if self.visibility_model_name not in valid_visibility_models:
            raise ValueError(
                "visibility_model must be one of: raycast_25d, raycast25d, raycast, gp_visibility, gpvis"
            )
        if self.visibility_model_name in raycast_aliases:
            vis_cfg = Raycast25DVisibilityConfig(
                map_xmin=float(visibility_map_min_x),
                map_xmax=float(visibility_map_max_x),
                map_ymin=float(visibility_map_min_y),
                map_ymax=float(visibility_map_max_y),
                map_nx=int(visibility_map_nx),
                map_ny=int(visibility_map_ny),
                gp_length_scale=float(visibility_gp_length_scale),
                gp_noise_var=float(visibility_gp_noise_var),
                prior_occ=float(visibility_prior_occ),
                beta=float(visibility_beta),
                height_tau=float(visibility_height_tau),
                ray_samples=int(visibility_ray_samples),
                target_height_m=float(visibility_target_height_m),
                geometry_json=str(visibility_geometry_json or ''),
                camera_pos=tuple(np.asarray(camera_params['cam_pos'], dtype=float).tolist()),
                seed=int(visibility_gp_seed),
                min_prob=self._visibility_min_prob,
            )
            self.visibility_model = Raycast25DVisibilityModel(vis_cfg)
        elif self.visibility_model_name in gp_visibility_aliases:
            vis_cfg = GPVisibilityMapConfig(
                map_xmin=float(visibility_map_min_x),
                map_xmax=float(visibility_map_max_x),
                map_ymin=float(visibility_map_min_y),
                map_ymax=float(visibility_map_max_y),
                map_nx=int(visibility_map_nx),
                map_ny=int(visibility_map_ny),
                gp_length_scale=float(visibility_gp_length_scale),
                gp_noise_var=float(visibility_gp_noise_var),
                prior_occ=float(visibility_prior_occ),
                beta=float(visibility_beta),
                geometry_json=str(visibility_geometry_json or ''),
                camera_pos=tuple(np.asarray(camera_params['cam_pos'], dtype=float).tolist()),
                target_height_m=float(visibility_target_height_m),
                seed=int(visibility_gp_seed),
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

        self._approx_fn = None
        if self.approx_method == 'ET1':
            self._approx_fn = ET1
        elif self.approx_method == 'ET2':
            self._approx_fn = ET2
        elif self.approx_method == 'UT':
            self._approx_fn = UT

        self.prev_controls_flat = None
        self._jax_valgrad_cache = {}

    def _runtime_debug_print(self, message):
        if not self.runtime_debug:
            return
        try:
            print(message, flush=True)
        except (BrokenPipeError, OSError):
            # Launch wrappers can close stdout while a long JAX compile is still running.
            pass

    def process_noise(self, dt=None):
        step_dt = self.dt if dt is None else float(dt)
        return unicycle_process_noise(
            self.process_noise_xy, self.process_noise_theta, step_dt, base_dt=self.dt
        )

    def predict(self, m, S, u, dt=None):
        step_dt = self.dt if dt is None else float(dt)
        m_next = unicycle_step(m, u, step_dt)
        F = unicycle_jacobian(m_next, u, step_dt)
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
        if self.math_mode != 'notebook_simple':
            del S
            return self.visibility_probability(m)
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

    def _make_visibility_belief_prob_jax(self, prob_state_jax):
        import jax.numpy as jnp

        eps = float(self._visibility_min_prob)
        def p_vis_belief(m, S):
            if self.math_mode != 'notebook_simple':
                del S
                return jnp.clip(prob_state_jax(m), eps, 1.0 - eps)
            kappa = max(float(self.visibility_sigma_kappa), 1e-6)
            mean_xy = jnp.asarray(m[:2], dtype=jnp.float64).reshape((2,))
            cov_xy = jnp.asarray(S[:2, :2], dtype=jnp.float64).reshape((2, 2))
            cov_xy = 0.5 * (cov_xy + cov_xy.T)
            scale = jnp.sqrt(jnp.asarray(2.0 + kappa, dtype=jnp.float64))
            chol = jnp.linalg.cholesky(cov_xy + 1e-9 * jnp.eye(2, dtype=jnp.float64))
            spread = scale * chol
            sigma_points = jnp.vstack(
                [
                    mean_xy,
                    mean_xy + spread[:, 0],
                    mean_xy - spread[:, 0],
                    mean_xy + spread[:, 1],
                    mean_xy - spread[:, 1],
                ]
            )
            weights = jnp.asarray(
                [kappa / (2.0 + kappa)] + [1.0 / (2.0 * (2.0 + kappa))] * 4,
                dtype=jnp.float64,
            )
            vals = jnp.clip(
                jnp.asarray(
                    [
                        prob_state_jax(jnp.array([sigma_points[i, 0], sigma_points[i, 1], m[2]], dtype=m.dtype))
                        for i in range(5)
                    ]
                ),
                eps,
                1.0 - eps,
            )
            return jnp.clip(jnp.sum(weights * vals), eps, 1.0 - eps)

        return p_vis_belief

    @staticmethod
    def _smoothstep(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def goal_obs_cov_for_progress(self, progress):
        if self.math_mode != 'notebook_simple':
            return self._goal_obs_cov()
        progress_fast = float(np.clip(progress, 0.0, 1.0)) ** self.goal_tightening_power
        a = self._smoothstep(progress_fast)
        sigma_u = (1.0 - a) * self.goal_prior_u_std_start + a * self.goal_prior_u_std_final
        sigma_v = (1.0 - a) * self.goal_prior_v_std_start + a * self.goal_prior_v_std_final
        return np.diag([sigma_u ** 2, sigma_v ** 2]).astype(float)

    def notebook_visibility_diagnostics(self, m, S):
        p_vis = self.visibility_probability_belief(m, S)
        p_vis_eff = float(np.clip(p_vis ** self.visibility_power, self._visibility_min_prob, 1.0 - self._visibility_min_prob))
        if (not self.use_visibility_model) or (self.visibility_model is None):
            p_vis = 1.0
            p_vis_eff = 1.0
        R_plan = p_vis_eff * self.R_visible + (1.0 - p_vis_eff) * self.R_miss
        return {
            'p_vis': float(p_vis),
            'p_vis_eff': float(p_vis_eff),
            'R_plan': np.asarray(R_plan, dtype=float),
            'r_plan_u_std': float(np.sqrt(max(R_plan[0, 0], 0.0))),
            'r_plan_v_std': float(np.sqrt(max(R_plan[1, 1], 0.0))),
        }

    def obstacle_penalty(self, m):
        if self.nogo_cost_model is None or not self.nogo_cost_model.enabled:
            return 0.0
        return float(self.nogo_cost_model.penalty_state_np(m))

    def _resolve_plan_problem(self, m0, goal_xy):
        goal_theta = float(m0[2]) if (self.risk_weight_state > 0.0 and self.math_mode != 'notebook_simple') else 0.0
        goal_state = self._goal_state(goal_xy, goal_theta)
        goal_cov = self._goal_state_cov()

        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity
        use_state_risk = self.risk_weight_state > 0.0 and self.math_mode != 'notebook_simple'

        goal_obs = None
        goal_obs_cov = None
        if use_observation_risk or use_ambiguity_term:
            goal_obs = self._goal_obs(goal_state)
            goal_obs_cov = self._goal_obs_cov()

        return (
            goal_state,
            goal_cov,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
            use_state_risk,
        )

    def _nominal_controls_flat(self):
        if self.math_mode == 'notebook_simple':
            return np.zeros(self.horizon * 2, dtype=float)
        x0 = np.zeros(self.horizon * 2, dtype=float)
        x0[0::2] = 0.5 * (self.v_min + self.v_max)
        return x0

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

    def _objective_scales(self, controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov):
        if self.math_mode == 'notebook_simple':
            return 1.0, 1.0
        _, metrics = self._evaluate_controls(
            controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, True
        )
        risk_scale = float(max(abs(metrics[0]), 1.0))
        ambiguity_scale = float(max(abs(metrics[1]), 1.0))
        return risk_scale, ambiguity_scale

    @staticmethod
    def _scaled_objective_from_metrics(metrics, objective_scales):
        risk_scale, ambiguity_scale = objective_scales
        risk_term = float(metrics[0]) / float(max(risk_scale, 1e-9))
        ambiguity_term = float(metrics[1]) / float(max(ambiguity_scale, 1e-9))
        return (
            risk_term
            + float(metrics[2])
            + float(metrics[3])
            + float(metrics[4])
        )

    def _evaluate_candidate_controls(
        self,
        controls_flat,
        m0,
        S0,
        goal_state,
        goal_cov,
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
            goal_cov,
            goal_obs,
            goal_obs_cov,
            True,
            progress_index=progress_index,
        )
        scaled_total = self._scaled_objective_from_metrics(metrics, objective_scales)
        return {
            'controls_flat': controls_flat,
            'total_cost': float(total_cost),
            'metrics': tuple(float(v) for v in metrics),
            'scaled_total': float(scaled_total),
        }

    def _resolve_use_jax_backend(self, backend):
        allowed = ('auto', 'jax') if self.math_mode != 'notebook_simple' else ('auto', 'jax', 'scipy')
        if backend not in allowed:
            raise ValueError(
                f"Unknown optimizer_backend '{self.optimizer_backend}'. Expected {'|'.join(allowed)}."
            )

        if self.approx_method not in ('ET1', 'ET2'):
            raise RuntimeError(
                f"optimizer_backend={backend} does not support approx_method={self.approx_method} "
                "(supported: ET1, ET2)."
            )

        try:
            from planning.core import jax_efe
            if not jax_efe.jax_available():
                raise RuntimeError(f"optimizer_backend={backend} but JAX is not available")
        except Exception:
            raise
        return True

    def _get_jax_valgrad(
        self,
        goal_state,
        goal_cov,
        goal_obs,
        goal_obs_cov,
        *,
        use_observation_risk,
        use_ambiguity_term,
        use_state_risk,
    ):
        from planning.core import jax_efe
        import jax.numpy as jnp

        Q = self.process_noise(self.dt)
        if goal_obs is None:
            goal_obs = self._goal_obs(goal_state)
        if goal_obs_cov is None:
            goal_obs_cov = self._goal_obs_cov()

        cache_dir = jax_efe.configure_persistent_cache()
        if cache_dir:
            self._runtime_debug_print(
                f"[planner_debug] JAX persistent cache enabled at {cache_dir}"
            )

        cache_key = (
            self.math_mode,
            self.approx_method,
            bool(use_ambiguity_term),
            bool(use_observation_risk),
            bool(use_state_risk),
            bool(self.use_visibility_model),
            float(self.control_weight),
            float(self.risk_weight_state),
            float(self.risk_weight_obs),
            float(self.ambiguity_weight),
            float(self.r_visible_uv),
            float(self.r_miss_uv),
            float(self.visibility_power),
            float(self.visibility_sigma_kappa),
            float(self.goal_prior_u_std_start),
            float(self.goal_prior_v_std_start),
            float(self.goal_prior_u_std_final),
            float(self.goal_prior_v_std_final),
            float(self.goal_tightening_power),
            int(self.goal_progress_n_steps),
            float(self.notebook_risk_scale),
            float(self.notebook_ambiguity_scale),
            float(self.discount_gamma),
            float(self.visibility_weight),
            float(self.visibility_cov_pos_scale),
            float(self.visibility_cov_theta_scale),
            bool(self.use_nogo_cost),
            tuple(self.nogo_cost_model.signature) if self.nogo_cost_model is not None else (),
            int(self.horizon),
            float(self.dt),
            int(np.asarray(goal_state, dtype=float).shape[0]),
            int(np.asarray(goal_obs, dtype=float).shape[0]),
            tuple(np.round(np.asarray(np.diag(self.R_bad), dtype=float), 8).tolist()),
            tuple(self.visibility_model.signature) if self.visibility_model is not None else (),
        )
        valgrad = self._jax_valgrad_cache.get(cache_key)
        self._runtime_debug_print(
            "[planner_debug] JAX valgrad cache "
            f"{'hit' if valgrad is not None else 'miss'} "
            f"(horizon={self.horizon}, approx={self.approx_method})"
        )
        if valgrad is None:
            build_start = time.perf_counter()
            p_vis_jax = None
            prob_state_jax = None
            if self.use_visibility_model and self.visibility_model is not None:
                prob_state_jax = self.visibility_model.make_prob_state_jax()
                p_vis_jax = self._make_visibility_belief_prob_jax(prob_state_jax)
            nogo_cost_jax = None
            if self.nogo_cost_model is not None and self.nogo_cost_model.enabled:
                nogo_cost_jax = self.nogo_cost_model.make_penalty_state_jax()
            g_jax = jax_efe.make_g_from_homography(self.camera.H)
            if self.math_mode == 'notebook_simple':
                params_j = jax_efe.JaxNotebookSimpleParams(
                    Q=jnp.array(Q),
                    R_visible=jnp.array(self.R_visible),
                    R_miss=jnp.array(self.R_miss),
                    control_weight=float(self.control_weight),
                    risk_scale=float(self.notebook_risk_scale if use_observation_risk else 0.0),
                    ambiguity_scale=float(self.notebook_ambiguity_scale if use_ambiguity_term else 0.0),
                    discount_gamma=float(self.discount_gamma),
                    visibility_power=float(self.visibility_power),
                    visibility_sigma_kappa=float(self.visibility_sigma_kappa),
                    goal_prior_u_std_start=float(self.goal_prior_u_std_start),
                    goal_prior_v_std_start=float(self.goal_prior_v_std_start),
                    goal_prior_u_std_final=float(self.goal_prior_u_std_final),
                    goal_prior_v_std_final=float(self.goal_prior_v_std_final),
                    goal_tightening_power=float(self.goal_tightening_power),
                    goal_progress_n_steps=int(self.goal_progress_n_steps),
                    time_horizon=int(self.horizon),
                    dt=float(self.dt),
                    Du=2,
                )
                valgrad = jax_efe.make_notebook_simple_valgrad_fn(
                    params_j,
                    g_jax,
                    p_vis_state=prob_state_jax,
                    nogo_cost=nogo_cost_jax,
                    jit=True,
                )
            else:
                params_j = jax_efe.JaxUnicycleParams(
                    Q=jnp.array(Q),
                    R=jnp.array(self.R),
                    R_bad=jnp.array(self.R_bad),
                    control_weight=float(self.control_weight),
                    risk_weight_state=float(self.risk_weight_state),
                    risk_weight_obs=float(self.risk_weight_obs),
                    ambiguity_weight=float(self.ambiguity_weight),
                    visibility_weight=float(self.visibility_weight),
                    vis_cov_pos_scale=float(self.visibility_cov_pos_scale),
                    vis_cov_theta_scale=float(self.visibility_cov_theta_scale),
                    time_horizon=int(self.horizon),
                    dt=float(self.dt),
                    Du=2,
                )
                valgrad = jax_efe.make_unicycle_valgrad_fn(
                    params_j,
                    g_jax,
                    approx=self.approx_method,
                    add_ambiguity=use_ambiguity_term,
                    use_obs_risk=use_observation_risk,
                    use_state_risk=use_state_risk,
                    p_vis=p_vis_jax,
                    nogo_cost=nogo_cost_jax,
                    mode='rev',
                    jit=True,
                )
            self._jax_valgrad_cache[cache_key] = valgrad
            self._runtime_debug_print(
                "[planner_debug] JAX valgrad function prepared in "
                f"{(time.perf_counter() - build_start) * 1000.0:.1f} ms"
            )

        return valgrad, jnp

    def warmup_jax(self, m0, S0, goal_xy):
        backend = self.optimizer_backend
        use_jax = self._resolve_use_jax_backend(backend)
        if not use_jax:
            return False

        (
            goal_state,
            goal_cov,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
            use_state_risk,
        ) = self._resolve_plan_problem(m0, goal_xy)
        valgrad, jnp = self._get_jax_valgrad(
            goal_state,
            goal_cov,
            goal_obs,
            goal_obs_cov,
            use_observation_risk=use_observation_risk,
            use_ambiguity_term=use_ambiguity_term,
            use_state_risk=use_state_risk,
        )

        x0 = self._initial_controls_flat()
        warm_start = time.perf_counter()
        if self.math_mode == 'notebook_simple':
            val, grad = valgrad(
                jnp.array(x0),
                jnp.array(m0),
                jnp.array(S0),
                jnp.array(goal_obs if goal_obs is not None else self._goal_obs(goal_state)),
                jnp.asarray(0.0, dtype=jnp.float64),
            )
        else:
            objective_scales = self._objective_scales(
                x0, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov
            )
            val, grad = valgrad(
                jnp.array(x0),
                jnp.array(m0),
                jnp.array(S0),
                jnp.array(goal_state),
                jnp.array(goal_cov),
                jnp.array(goal_obs if goal_obs is not None else self._goal_obs(goal_state)),
                jnp.array(goal_obs_cov if goal_obs_cov is not None else self._goal_obs_cov()),
                jnp.array(objective_scales[0], dtype=jnp.float64),
                jnp.array(objective_scales[1], dtype=jnp.float64),
            )
        val_out = float(val)
        grad_out = np.array(grad, dtype=float)
        self._runtime_debug_print(
            "[planner_debug] JAX warm-up completed in "
            f"{(time.perf_counter() - warm_start) * 1000.0:.1f} ms "
            f"with J={val_out:.3f}, grad_norm={np.linalg.norm(grad_out):.3f}"
        )
        return True

    def observation_model_with_visibility(self, m_pred, S_pred):
        """Visibility-aware measurement shaping used by objective and correction."""
        S_pred = np.asarray(S_pred, dtype=float)
        if (not self.use_visibility_model) or (self.visibility_model is None):
            R_eff = np.asarray(self.R_visible if self.math_mode == 'notebook_simple' else self.R, dtype=float)
            S_eff = S_pred.copy()
            return 1.0, R_eff, S_eff, 1.0

        if self.math_mode == 'notebook_simple':
            diag = self.notebook_visibility_diagnostics(m_pred, S_pred)
            R_eff = np.asarray(diag['R_plan'], dtype=float)
            S_eff = S_pred.copy()
            return float(diag['p_vis']), R_eff, S_eff, 1.0

        p = self.visibility_probability_belief(m_pred, S_pred)
        q = 1.0 - p

        R_eff = p * self.R + q * self.R_bad
        S_eff = S_pred.copy()
        if S_eff.shape[0] >= 2:
            scale_xy = 1.0 + self.visibility_cov_pos_scale * q
            S_eff[0, 0] *= scale_xy
            S_eff[1, 1] *= scale_xy
        if S_eff.shape[0] >= 3:
            scale_th = 1.0 + self.visibility_cov_theta_scale * q
            S_eff[2, 2] *= scale_th
        gain_scale = p

        R_eff = 0.5 * (R_eff + R_eff.T) + 1e-9 * np.eye(R_eff.shape[0])
        S_eff = 0.5 * (S_eff + S_eff.T) + 1e-9 * np.eye(S_eff.shape[0])
        return p, R_eff, S_eff, gain_scale

    def _goal_state(self, goal_xy, theta):
        return np.array([goal_xy[0], goal_xy[1], theta], dtype=float)

    def _goal_state_cov(self):
        return np.diag([
            self.goal_sigma_xy ** 2,
            self.goal_sigma_xy ** 2,
            self.goal_sigma_theta ** 2,
        ])

    def _goal_obs(self, goal_state):
        return np.asarray(self.g_obs(goal_state), dtype=float)

    def _goal_obs_cov(self):
        if self.math_mode == 'notebook_simple':
            return self.goal_obs_cov_for_progress(0.0)
        return np.diag([
            self.goal_sigma_uv ** 2,
            self.goal_sigma_uv ** 2,
        ]).astype(float)

    def _evaluate_controls(
        self,
        controls_flat,
        m0,
        S0,
        goal_state,
        goal_cov,
        goal_obs,
        goal_obs_cov,
        return_metrics=False,
        *,
        progress_index=0.0,
    ):
        controls_flat = np.asarray(controls_flat, dtype=float)
        if controls_flat.size != self.horizon * 2:
            controls_flat = controls_flat[:self.horizon * 2]
        controls = controls_flat.reshape(self.horizon, 2)

        m = m0.copy()
        S = S0.copy()
        total_risk = 0.0
        total_amb = 0.0
        total_control = 0.0
        total_visibility = 0.0
        total_obstacle = 0.0
        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity
        use_state_risk = self.risk_weight_state > 0.0 and self.math_mode != 'notebook_simple'

        for t in range(self.horizon):
            u = controls[t]
            m, S = self.predict(m, S, u)
            if self.math_mode == 'notebook_simple':
                vis_diag = self.notebook_visibility_diagnostics(m, S)
                p_vis = vis_diag['p_vis']
                R_plan = vis_diag['R_plan']
                mu_y = Sigma_y = Gamma = None
                if use_observation_risk or use_ambiguity_term:
                    mu_y, Sigma_y, Gamma = self.approx_observation(m, S, method='ET1', R_override=R_plan)
                weight_t = self.discount_gamma ** t
                observation_risk = 0.0
                if use_observation_risk and mu_y is not None:
                    goal_cov_t = self.goal_obs_cov_for_progress(
                        (float(progress_index) + float(t)) / max(self.goal_progress_n_steps, 1)
                    )
                    observation_risk = self.notebook_risk_scale * risk(mu_y, Sigma_y, (goal_obs, goal_cov_t))
                total_risk += weight_t * observation_risk
                if use_ambiguity_term and Sigma_y is not None:
                    total_amb += weight_t * (self.notebook_ambiguity_scale * ambiguity(Sigma_y, Gamma, S))
                total_obstacle += weight_t * self.obstacle_penalty(m)
                total_control += weight_t * self.control_weight * float(u[0] ** 2 + u[1] ** 2)
            else:
                p_vis, R_eff, S_eff, _ = self.observation_model_with_visibility(m, S)

                mu_y = Sigma_y = Gamma = None
                if use_observation_risk or use_ambiguity_term:
                    mu_y, Sigma_y, Gamma = self.approx_observation(m, S_eff, R_override=R_eff)

                state_risk = 0.0
                if use_state_risk:
                    state_risk = risk(m, S, (goal_state, goal_cov))
                observation_risk = 0.0
                if use_observation_risk and mu_y is not None:
                    observation_risk = risk(mu_y, Sigma_y, (goal_obs, goal_obs_cov))

                risk_term = (
                    self.risk_weight_state * state_risk
                    + self.risk_weight_obs * observation_risk
                )
                total_risk += risk_term

                ambiguity_term = 0.0
                if use_ambiguity_term and Sigma_y is not None:
                    ambiguity_core = ambiguity(Sigma_y, Gamma, S_eff)
                    if self.use_visibility_model:
                        ambiguity_core = p_vis * ambiguity_core
                    ambiguity_term = self.ambiguity_weight * ambiguity_core
                    total_amb += ambiguity_term

                if self.use_visibility_model and self.visibility_weight > 0.0:
                    total_visibility += self.visibility_weight * (1.0 - p_vis)

                total_obstacle += self.obstacle_penalty(m)

                control_term = self.control_weight * float(u[0] ** 2 + u[1] ** 2)
                total_control += control_term

        total = total_risk + total_amb + total_control + total_visibility + total_obstacle
        if return_metrics:
            return total, (total_risk, total_amb, total_control, total_visibility, total_obstacle)
        return total

    def plan(self, m0, S0, goal_xy, *, progress_index=0.0):
        t_plan_start = time.perf_counter()
        progress_index = float(max(progress_index, 0.0))

        (
            goal_state,
            goal_cov,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
            use_state_risk,
        ) = self._resolve_plan_problem(m0, goal_xy)

        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((self.w_min, self.w_max))

        x0 = self._initial_controls_flat()
        objective_scales = self._objective_scales(
            x0, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov
        )
        best_candidate = None
        backend = self.optimizer_backend
        backend_used = 'scipy' if self.math_mode == 'notebook_simple' else 'jax'
        optimizer_success = False
        optimizer_status = 0
        optimizer_nit = 0
        optimizer_nfev = 0
        optimizer_message = ''

        try:
            import jax.numpy as jnp

            use_jax = self._resolve_use_jax_backend(backend)
            if not use_jax:
                raise RuntimeError("JAX objective is unavailable")

            valgrad, jnp = self._get_jax_valgrad(
                goal_state,
                goal_cov,
                goal_obs,
                goal_obs_cov,
                use_observation_risk=use_observation_risk,
                use_ambiguity_term=use_ambiguity_term,
                use_state_risk=use_state_risk,
            )

            fg_calls = {'count': 0}
            m0_j = jnp.array(m0)
            S0_j = jnp.array(S0)
            goal_state_j = jnp.array(goal_state)
            goal_cov_j = jnp.array(goal_cov)
            goal_obs_j = jnp.array(goal_obs if goal_obs is not None else self._goal_obs(goal_state))
            goal_obs_cov_j = jnp.array(goal_obs_cov if goal_obs_cov is not None else self._goal_obs_cov())
            progress0_j = jnp.asarray(progress_index, dtype=goal_obs_j.dtype)
            risk_scale_j = jnp.asarray(objective_scales[0], dtype=goal_cov_j.dtype)
            ambiguity_scale_j = jnp.asarray(objective_scales[1], dtype=goal_cov_j.dtype)

            if self.math_mode == 'notebook_simple':
                shared_eval = {
                    'u': None,
                    'val': None,
                    'grad': None,
                }

                def eval_valgrad_shared(u):
                    u_arr = np.asarray(u, dtype=float)
                    cached_u = shared_eval['u']
                    if cached_u is not None and cached_u.shape == u_arr.shape and np.array_equal(cached_u, u_arr):
                        return shared_eval['val'], shared_eval['grad']

                    start = time.perf_counter()
                    val, grad = valgrad(
                        jnp.array(u_arr),
                        m0_j,
                        S0_j,
                        goal_obs_j,
                        progress0_j,
                    )
                    val_out = float(val)
                    grad_out = np.asarray(grad, dtype=float)
                    if fg_calls['count'] == 0:
                        self._runtime_debug_print(
                            "[planner_debug] First notebook objective/gradient eval returned in "
                            f"{(time.perf_counter() - start) * 1000.0:.1f} ms "
                            f"with J={val_out:.3f}, grad_norm={np.linalg.norm(grad_out):.3f}"
                        )
                    fg_calls['count'] += 1
                    shared_eval['u'] = u_arr.copy()
                    shared_eval['val'] = val_out
                    shared_eval['grad'] = grad_out
                    return val_out, grad_out

                def func(u):
                    val_out, _ = eval_valgrad_shared(u)
                    return val_out

                def jac(u):
                    _, grad_out = eval_valgrad_shared(u)
                    return grad_out

                minimize_start = time.perf_counter()
                self._runtime_debug_print(
                    "[planner_debug] Starting notebook-style scipy.optimize.minimize "
                    f"(maxiter={self.optimizer_maxiter}, maxfun={self.optimizer_maxfun}, ftol={self.optimizer_ftol})"
                )
                result = minimize(
                    func,
                    np.asarray(x0, dtype=float),
                    jac=jac,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': self.optimizer_maxiter,
                        'maxfun': self.optimizer_maxfun,
                        'ftol': self.optimizer_ftol,
                    },
                )
                self._runtime_debug_print(
                    "[planner_debug] notebook minimize finished in "
                    f"{(time.perf_counter() - minimize_start) * 1000.0:.1f} ms "
                    f"(success={bool(result.success)}, status={int(result.status)}, "
                    f"nit={int(getattr(result, 'nit', 0) or 0)}, nfev={int(getattr(result, 'nfev', 0) or 0)}, "
                    f"shared_fg_evals={fg_calls['count']})"
                )
                if result.x is None or not np.all(np.isfinite(np.asarray(result.x, dtype=float))):
                    raise RuntimeError("Notebook optimizer returned no finite solution")
                if not bool(result.success):
                    raise RuntimeError(
                        f"Notebook optimizer did not converge (status={int(result.status)}, message={result.message})"
                    )
                best_candidate = self._evaluate_candidate_controls(
                    np.asarray(result.x, dtype=float),
                    m0,
                    S0,
                    goal_state,
                    goal_cov,
                    goal_obs,
                    goal_obs_cov,
                    objective_scales,
                    progress_index=progress_index,
                )
                best_candidate.update({'source': 'solver:shifted_warm_start' if self.prev_controls_flat is not None else 'solver:zero_seed'})
                optimizer_success = bool(result.success)
                optimizer_status = int(result.status)
                optimizer_nit = int(getattr(result, 'nit', 0) or 0)
                optimizer_nfev = int(getattr(result, 'nfev', 0) or 0)
                optimizer_message = str(result.message or '')
            else:
                backend_used = 'jax'

                def func_and_grad(u):
                    start = time.perf_counter()
                    val, grad = valgrad(
                        jnp.array(u),
                        m0_j,
                        S0_j,
                        goal_state_j,
                        goal_cov_j,
                        goal_obs_j,
                        goal_obs_cov_j,
                        risk_scale_j,
                        ambiguity_scale_j,
                    )
                    val_out = float(val)
                    grad_out = np.array(grad, dtype=float)
                    if fg_calls['count'] == 0:
                        self._runtime_debug_print(
                            "[planner_debug] First JAX objective/gradient eval returned in "
                            f"{(time.perf_counter() - start) * 1000.0:.1f} ms "
                            f"with J={val_out:.3f}, grad_norm={np.linalg.norm(grad_out):.3f}"
                        )
                    fg_calls['count'] += 1
                    return val_out, grad_out

                minimize_start = time.perf_counter()
                self._runtime_debug_print(
                    "[planner_debug] Starting fmin_l_bfgs_b over legacy objective "
                    f"(maxiter={self.optimizer_maxiter}, gtol={self.optimizer_gtol}, "
                    f"risk_scale={objective_scales[0]:.3f}, ambiguity_scale={objective_scales[1]:.3f})"
                )
                x_opt, _, info = fmin_l_bfgs_b(
                    func_and_grad,
                    np.asarray(x0, dtype=float),
                    bounds=bounds,
                    maxiter=self.optimizer_maxiter,
                    pgtol=self.optimizer_gtol,
                    maxls=40,
                )
                warnflag_raw = info.get('warnflag', 2)
                warnflag = 2 if warnflag_raw is None else int(warnflag_raw)
                task = str(info.get('task', '') or '')
                nit_raw = info.get('nit', 0)
                nit = 0 if nit_raw is None else int(nit_raw)
                funcalls_raw = info.get('funcalls', 0)
                funcalls = 0 if funcalls_raw is None else int(funcalls_raw)
                self._runtime_debug_print(
                    "[planner_debug] fmin_l_bfgs_b finished in "
                    f"{(time.perf_counter() - minimize_start) * 1000.0:.1f} ms "
                    f"(warnflag={warnflag}, nit={nit}, funcalls={funcalls}, task='{task}')"
                )
                if x_opt is None:
                    raise RuntimeError("Legacy optimizer produced no solution")
                x_opt = np.asarray(x_opt, dtype=float)
                if not np.all(np.isfinite(x_opt)):
                    raise RuntimeError("Legacy optimizer produced a non-finite solution")
                best_candidate = self._evaluate_candidate_controls(
                    x_opt,
                    m0,
                    S0,
                    goal_state,
                    goal_cov,
                    goal_obs,
                    goal_obs_cov,
                    objective_scales,
                    progress_index=progress_index,
                )
                best_candidate.update({'source': 'solver:shifted_warm_start' if self.prev_controls_flat is not None else 'solver:nominal_seed'})
                optimizer_success = warnflag == 0
                optimizer_status = warnflag
                optimizer_nit = nit
                optimizer_nfev = funcalls
                optimizer_message = task
        except Exception as exc:
            raise RuntimeError(
                "Planner optimization failed "
                f"(backend_request={backend}, backend_used={backend_used}, "
                f"approx={self.approx_method}, "
                f"horizon={self.horizon}, dt={self.dt}, math_mode={self.math_mode})"
            ) from exc

        if best_candidate is None:
            raise RuntimeError("Planner produced no candidate solution")

        best_controls_flat = np.asarray(best_candidate['controls_flat'], dtype=float)
        self.prev_controls_flat = np.array(best_controls_flat, dtype=float)
        best_controls = self.prev_controls_flat.reshape(self.horizon, 2)
        total_cost = float(best_candidate['total_cost'])
        metrics = tuple(best_candidate['metrics'])
        vis_diag = self.notebook_visibility_diagnostics(m0, S0) if self.math_mode == 'notebook_simple' else {
            'p_vis': self.visibility_probability_belief(m0, S0),
            'p_vis_eff': self.visibility_probability_belief(m0, S0),
            'r_plan_u_std': float(np.sqrt(max(self.R[0, 0], 0.0))),
            'r_plan_v_std': float(np.sqrt(max(self.R[1, 1], 0.0))),
        }

        states = rollout_unicycle(m0, best_controls, self.dt)
        solve_time_s = float(max(time.perf_counter() - t_plan_start, 0.0))
        return PlanResult(
            controls=best_controls,
            states=states,
            total_cost=float(total_cost),
            risk_cost=float(metrics[0]),
            ambiguity_cost=float(metrics[1]),
            control_cost=float(metrics[2]),
            visibility_cost=float(metrics[3]) if len(metrics) > 3 else 0.0,
            obstacle_cost=float(metrics[4]) if len(metrics) > 4 else 0.0,
            backend=str(backend_used),
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_nit=optimizer_nit,
            optimizer_nfev=optimizer_nfev,
            optimizer_message=optimizer_message,
            solve_time_s=solve_time_s,
            used_fallback=False,
            selected_source=str(best_candidate.get('source', '')),
            p_vis_plan=float(vis_diag['p_vis']),
            p_vis_plan_eff=float(vis_diag['p_vis_eff']),
            r_plan_u_std=float(vis_diag['r_plan_u_std']),
            r_plan_v_std=float(vis_diag['r_plan_v_std']),
        )
