"""Base planner classes (pure Python, no ROS)."""

from dataclasses import dataclass
import time
import numpy as np

from scipy.optimize import minimize

from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import ET1, ET2, UT, ambiguity, risk
from planning.core.visibility_gp import FixedGPVisibilityConfig, FixedGPVisibilityModel
from unav_common.camera_model import ObliqueCameraModel
from planning.core.rollout import rollout_unicycle
from planning.core import search_based_path_planning


@dataclass
class CostmapData:
    origin: np.ndarray
    resolution: float
    width: int
    height: int
    data: np.ndarray
    frame_id: str


@dataclass
class PlanResult:
    controls: np.ndarray
    states: np.ndarray
    total_cost: float
    risk_cost: float
    ambiguity_cost: float
    control_cost: float
    boundary_cost: float
    visibility_cost: float = 0.0
    backend: str = "unknown"
    optimizer_success: bool = False
    optimizer_status: int = 0
    optimizer_nit: int = 0
    optimizer_nfev: int = 0
    optimizer_message: str = ""
    used_fallback: bool = False  # legacy field kept for manifest/log compatibility
    solve_time_s: float = 0.0


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
        boundary_weight,
        max_cost,
        lethal_cost_threshold,
        num_samples,
        process_noise_xy,
        process_noise_theta,
        obs_noise_uv,
        obs_noise_yaw,
        goal_sigma_xy,
        goal_sigma_theta,
        goal_sigma_uv,
        goal_sigma_yaw,
        risk_weight_state,
        risk_weight_obs,
        ambiguity_weight,
        optimizer_maxiter,
        optimizer_gtol,
        optimizer_warm_start,
        approx_method=None,
        use_obs_risk=None,
        use_ambiguity=None,
        obs_mode=None,
        optimizer_backend=None,
        seed,
        camera_params,
        use_visibility_model=False,
        visibility_model='fixed_gp',
        visibility_weight=0.0,
        visibility_map_min_x=-5.0,
        visibility_map_max_x=5.0,
        visibility_map_min_y=-5.0,
        visibility_map_max_y=5.0,
        visibility_map_nx=140,
        visibility_map_ny=120,
        visibility_occ_center_x=-1.2,
        visibility_occ_center_y=-1.8,
        visibility_occ_radius=0.9,
        visibility_occ_tau=0.15,
        visibility_gp_length_scale=1.4,
        visibility_gp_noise_var=0.15,
        visibility_gp_seed=0,
        visibility_r_bad_uv=28.0,
        visibility_r_bad_yaw=1.2,
        visibility_cov_pos_scale=2.0,
        visibility_cov_theta_scale=0.8,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.control_weight = float(control_weight)
        self.boundary_weight = float(boundary_weight)
        self.max_cost = float(max_cost)
        self.lethal_cost_threshold = float(lethal_cost_threshold)
        self.num_samples = int(num_samples)

        self.process_noise_xy = float(process_noise_xy)
        self.process_noise_theta = float(process_noise_theta)
        self.obs_noise_uv = float(obs_noise_uv)
        self.obs_noise_yaw = float(obs_noise_yaw)

        self.goal_sigma_xy = float(goal_sigma_xy)
        self.goal_sigma_theta = float(goal_sigma_theta)
        self.goal_sigma_uv = float(goal_sigma_uv)
        self.goal_sigma_yaw = float(goal_sigma_yaw)

        if self.goal_sigma_uv <= 0.0:
            self.goal_sigma_uv = self.obs_noise_uv
        if self.goal_sigma_yaw <= 0.0:
            self.goal_sigma_yaw = self.obs_noise_yaw

        self.risk_weight_state = float(risk_weight_state)
        self.risk_weight_obs = float(risk_weight_obs)
        self.ambiguity_weight = float(ambiguity_weight)

        if approx_method is None:
            self.approx_method = self.APPROX_METHOD
        else:
            self.approx_method = str(approx_method).upper()

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

        self.rng = np.random.default_rng(int(seed))

        self.camera = ObliqueCameraModel(
            cam_pos=camera_params['cam_pos'],
            look_at=camera_params['look_at'],
            img_width=camera_params['img_width'],
            img_height=camera_params['img_height'],
            fov_h_rad=camera_params['fov_h_rad'],
        )

        self.obs_mode = str(obs_mode or 'uvt').lower()
        if self.obs_mode not in ('uv', 'uvt'):
            raise ValueError("obs_mode must be 'uv' or 'uvt'")
        self.optimizer_backend = str(optimizer_backend or 'scipy').lower()
        self.use_visibility_model = bool(use_visibility_model)
        self.visibility_model_name = str(visibility_model or 'none').strip().lower()
        self.visibility_weight = float(visibility_weight)
        self.visibility_cov_pos_scale = float(max(visibility_cov_pos_scale, 0.0))
        self.visibility_cov_theta_scale = float(max(visibility_cov_theta_scale, 0.0))
        self._visibility_min_prob = 1e-4
        self.visibility_model = None

        self.g_obs = self.camera.g_uv if self.obs_mode == 'uv' else self.camera.g

        # Observation noise matrix (dimension depends on obs_mode)
        if self.obs_mode == 'uv':
            self.R = np.diag([
                self.obs_noise_uv ** 2,
                self.obs_noise_uv ** 2,
            ])
        else:
            self.R = np.diag([
                self.obs_noise_uv ** 2,
                self.obs_noise_uv ** 2,
                self.obs_noise_yaw ** 2,
            ])
        if self.obs_mode == 'uv':
            self.R_bad = np.diag([
                float(visibility_r_bad_uv) ** 2,
                float(visibility_r_bad_uv) ** 2,
            ])
        else:
            self.R_bad = np.diag([
                float(visibility_r_bad_uv) ** 2,
                float(visibility_r_bad_uv) ** 2,
                float(visibility_r_bad_yaw) ** 2,
            ])

        if self.use_visibility_model and self.visibility_model_name not in ('fixed_gp', 'gp'):
            raise ValueError("visibility_model must be 'fixed_gp' (or 'gp') when enabled")
        if self.use_visibility_model and self.visibility_model_name in ('fixed_gp', 'gp'):
            vis_cfg = FixedGPVisibilityConfig(
                map_xmin=float(visibility_map_min_x),
                map_xmax=float(visibility_map_max_x),
                map_ymin=float(visibility_map_min_y),
                map_ymax=float(visibility_map_max_y),
                map_nx=int(visibility_map_nx),
                map_ny=int(visibility_map_ny),
                occ_center_x=float(visibility_occ_center_x),
                occ_center_y=float(visibility_occ_center_y),
                occ_radius=float(visibility_occ_radius),
                occ_tau=float(visibility_occ_tau),
                gp_length_scale=float(visibility_gp_length_scale),
                gp_noise_var=float(visibility_gp_noise_var),
                seed=int(visibility_gp_seed),
                min_prob=self._visibility_min_prob,
            )
            self.visibility_model = FixedGPVisibilityModel(vis_cfg)

        self._approx_fn = None
        if self.approx_method == 'ET1':
            self._approx_fn = ET1
        elif self.approx_method == 'ET2':
            self._approx_fn = ET2
        elif self.approx_method == 'UT':
            self._approx_fn = UT

        self.prev_controls_flat = None
        self._jax_valgrad_cache = {}

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

    def observation_model_with_visibility(self, m_pred, S_pred):
        """Visibility-aware measurement shaping used by objective and correction."""
        S_pred = np.asarray(S_pred, dtype=float)
        if (not self.use_visibility_model) or (self.visibility_model is None):
            R_eff = np.asarray(self.R, dtype=float)
            S_eff = S_pred.copy()
            return 1.0, R_eff, S_eff, 1.0

        p = self.visibility_probability(m_pred)
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
        if self.obs_mode == 'uv':
            return np.diag([
                self.goal_sigma_uv ** 2,
                self.goal_sigma_uv ** 2,
            ]).astype(float)
        return np.diag([
            self.goal_sigma_uv ** 2,
            self.goal_sigma_uv ** 2,
            self.goal_sigma_yaw ** 2,
        ]).astype(float)

    def _cost_at_raw(self, costmap, x, y):
        if costmap is None:
            return 0.0, True
        grid = search_based_path_planning.world_to_grid([x, y], origin=costmap.origin, resolution=costmap.resolution)[0]
        i, j = int(grid[0]), int(grid[1])
        if i < 0 or j < 0 or i >= costmap.height or j >= costmap.width:
            return self.max_cost, False
        cost = float(costmap.data[i, j])
        return cost, True

    def _evaluate_controls(self, controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, costmap, return_metrics=False):
        controls_flat = np.asarray(controls_flat, dtype=float)
        if controls_flat.size != self.horizon * 2:
            controls_flat = controls_flat[:self.horizon * 2]
        controls = controls_flat.reshape(self.horizon, 2)

        m = m0.copy()
        S = S0.copy()
        total_risk = 0.0
        total_amb = 0.0
        total_control = 0.0
        total_boundary = 0.0
        total_visibility = 0.0

        infeasible_penalty = 1e6
        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity
        use_state_risk = self.risk_weight_state > 0.0

        for t in range(self.horizon):
            u = controls[t]
            m, S = self.predict(m, S, u)
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

            control_term = self.control_weight * float(u[0] ** 2 + u[1] ** 2)
            total_control += control_term

            if self.boundary_weight > 0.0:
                cell_cost, in_bounds = self._cost_at_raw(costmap, m[0], m[1])
                if (not in_bounds) or (cell_cost < 0.0) or (cell_cost >= self.lethal_cost_threshold):
                    total_boundary += self.boundary_weight
                    total = total_risk + total_amb + total_control + total_boundary + total_visibility + infeasible_penalty
                    if return_metrics:
                        return total, (total_risk, total_amb, total_control, total_boundary, total_visibility)
                    return total
                boundary_term = self.boundary_weight * (cell_cost / max(self.max_cost, 1.0))
                total_boundary += boundary_term

        total = total_risk + total_amb + total_control + total_boundary + total_visibility
        if return_metrics:
            return total, (total_risk, total_amb, total_control, total_boundary, total_visibility)
        return total

    def plan(self, m0, S0, goal_xy, costmap):
        t_plan_start = time.perf_counter()

        # If state-risk weight is zero, keep goal theta constant to avoid
        # unnecessary optimizer/JAX recompilation churn.
        goal_theta = float(m0[2]) if self.risk_weight_state > 0.0 else 0.0
        goal_state = self._goal_state(goal_xy, goal_theta)
        goal_cov = self._goal_state_cov()

        use_observation_risk = self.use_obs_risk
        use_ambiguity_term = self.use_ambiguity
        use_state_risk = self.risk_weight_state > 0.0

        goal_obs = None
        goal_obs_cov = None
        if use_observation_risk or use_ambiguity_term:
            goal_obs = self._goal_obs(goal_state)
            goal_obs_cov = self._goal_obs_cov()

        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((self.w_min, self.w_max))

        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            x0 = np.array(self.prev_controls_flat, dtype=float)
        else:
            x0 = np.zeros(self.horizon * 2, dtype=float)
            x0[0::2] = 0.5 * (self.v_min + self.v_max)

        best_controls_flat = None
        backend_used = 'scipy'
        optimizer_success = False
        optimizer_status = 0
        optimizer_nit = 0
        optimizer_nfev = 0
        optimizer_message = ''
        used_fallback = False

        backend = self.optimizer_backend
        if backend not in ('auto', 'jax', 'scipy'):
            raise ValueError(
                f"Unknown optimizer_backend '{self.optimizer_backend}'. Expected auto|jax|scipy."
            )

        use_jax = backend in ('auto', 'jax')
        if use_jax and self.boundary_weight > 0.0:
            if backend == 'jax':
                raise RuntimeError(
                    "optimizer_backend=jax is not supported when boundary_weight > 0.0 "
                    "(costmap/boundary penalties are not implemented in the JAX objective path)."
                )
            use_jax = False

        if use_jax and self.approx_method not in ('ET1', 'ET2'):
            if backend == 'jax':
                raise RuntimeError(
                    f"optimizer_backend=jax does not support approx_method={self.approx_method} "
                    "(supported: ET1, ET2)."
                )
            use_jax = False

        if use_jax:
            try:
                from planning.core import jax_efe
                if not jax_efe.jax_available():
                    if backend == 'jax':
                        raise RuntimeError("optimizer_backend=jax but JAX is not available")
                    use_jax = False
            except Exception:
                if backend == 'jax':
                    raise
                use_jax = False

        try:
            if use_jax:
                from planning.core import jax_efe
                import jax.numpy as jnp
                backend_used = 'jax'

                Q = self.process_noise(self.dt)
                if goal_obs is None:
                    goal_obs = self._goal_obs(goal_state)
                if goal_obs_cov is None:
                    goal_obs_cov = self._goal_obs_cov()

                cache_key = (
                    self.approx_method,
                    self.obs_mode,
                    bool(use_ambiguity_term),
                    bool(use_observation_risk),
                    bool(use_state_risk),
                    bool(self.use_visibility_model),
                    float(self.control_weight),
                    float(self.risk_weight_state),
                    float(self.risk_weight_obs),
                    float(self.ambiguity_weight),
                    float(self.visibility_weight),
                    float(self.visibility_cov_pos_scale),
                    float(self.visibility_cov_theta_scale),
                    int(self.horizon),
                    float(self.dt),
                    tuple(np.round(np.asarray(goal_state, dtype=float), 8).tolist()),
                    tuple(np.round(np.asarray(np.diag(goal_cov), dtype=float), 8).tolist()),
                    tuple(np.round(np.asarray(goal_obs, dtype=float), 8).tolist()),
                    tuple(np.round(np.asarray(np.diag(goal_obs_cov), dtype=float), 8).tolist()),
                    tuple(np.round(np.asarray(np.diag(self.R_bad), dtype=float), 8).tolist()),
                    tuple(self.visibility_model.signature) if self.visibility_model is not None else (),
                )
                valgrad = self._jax_valgrad_cache.get(cache_key)
                if valgrad is None:
                    p_vis_jax = (
                        self.visibility_model.make_prob_state_jax()
                        if self.use_visibility_model and self.visibility_model is not None
                        else None
                    )
                    params_j = jax_efe.JaxUnicycleParams(
                        Q=jnp.array(Q),
                        R=jnp.array(self.R),
                        R_bad=jnp.array(self.R_bad),
                        goal_state=jnp.array(goal_state),
                        goal_state_cov=jnp.array(goal_cov),
                        goal_obs=jnp.array(goal_obs),
                        goal_obs_cov=jnp.array(goal_obs_cov),
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
                    g_jax = jax_efe.make_g_from_homography(self.camera.H, self.obs_mode)
                    valgrad = jax_efe.make_unicycle_valgrad_fn(
                        params_j,
                        g_jax,
                        approx=self.approx_method,
                        add_ambiguity=use_ambiguity_term,
                        use_obs_risk=use_observation_risk,
                        use_state_risk=use_state_risk,
                        p_vis=p_vis_jax,
                        mode='rev',
                        jit=True,
                    )
                    self._jax_valgrad_cache[cache_key] = valgrad

                def f(u):
                    val, _ = valgrad(jnp.array(u), jnp.array(m0), jnp.array(S0))
                    return float(val)

                def grad_u(u):
                    _, grad = valgrad(jnp.array(u), jnp.array(m0), jnp.array(S0))
                    return np.array(grad, dtype=float)

                result = minimize(
                    f,
                    x0,
                    jac=grad_u,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.optimizer_maxiter, 'gtol': self.optimizer_gtol},
                )
            else:
                result = minimize(
                    self._evaluate_controls,
                    x0,
                    args=(m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, costmap, False),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.optimizer_maxiter, 'gtol': self.optimizer_gtol},
                )
            optimizer_success = bool(getattr(result, 'success', False))
            optimizer_status = int(getattr(result, 'status', 0) or 0)
            optimizer_nit = int(getattr(result, 'nit', 0) or 0)
            optimizer_nfev = int(getattr(result, 'nfev', 0) or 0)
            optimizer_message = str(getattr(result, 'message', '') or '')
            if getattr(result, 'x', None) is None:
                raise RuntimeError(
                    f"Optimizer returned no solution vector (backend={backend_used}, "
                    f"status={optimizer_status}, message='{optimizer_message}')"
                )
            if not np.all(np.isfinite(result.x)):
                raise RuntimeError(
                    f"Optimizer returned non-finite controls (backend={backend_used}, "
                    f"status={optimizer_status}, message='{optimizer_message}')"
                )
            best_controls_flat = np.asarray(result.x, dtype=float)
        except Exception as exc:
            raise RuntimeError(
                "Planner optimization failed "
                f"(backend_request={backend}, backend_used={'jax' if use_jax else 'scipy'}, "
                f"approx={self.approx_method}, obs_mode={self.obs_mode}, "
                f"horizon={self.horizon}, dt={self.dt})"
            ) from exc

        self.prev_controls_flat = np.array(best_controls_flat, dtype=float)
        best_controls = self.prev_controls_flat.reshape(self.horizon, 2)
        total_cost, metrics = self._evaluate_controls(
            self.prev_controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, costmap, True
        )

        states = rollout_unicycle(m0, best_controls, self.dt)
        solve_time_s = float(max(time.perf_counter() - t_plan_start, 0.0))
        return PlanResult(
            controls=best_controls,
            states=states,
            total_cost=float(total_cost),
            risk_cost=float(metrics[0]),
            ambiguity_cost=float(metrics[1]),
            control_cost=float(metrics[2]),
            boundary_cost=float(metrics[3]),
            visibility_cost=float(metrics[4]) if len(metrics) > 4 else 0.0,
            backend=str(backend_used),
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_nit=optimizer_nit,
            optimizer_nfev=optimizer_nfev,
            optimizer_message=optimizer_message,
            used_fallback=used_fallback,
            solve_time_s=solve_time_s,
        )
