"""Base planner classes (pure Python, no ROS)."""

from dataclasses import dataclass
import time
import numpy as np

from scipy.optimize import minimize

from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import ET1, ET2, UT, ambiguity, risk
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
    backend: str = "unknown"
    optimizer_success: bool = False
    optimizer_status: int = 0
    optimizer_nit: int = 0
    optimizer_nfev: int = 0
    optimizer_message: str = ""
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
        approx_method=None,
        use_obs_risk=None,
        use_ambiguity=None,
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
        visibility_sigma_kappa=1.0,
        visibility_target_height_m=0.0,
        visibility_geometry_json='',
        visibility_gp_seed=0,
        visibility_r_bad_uv=28.0,
        visibility_cov_pos_scale=2.0,
        visibility_cov_theta_scale=0.8,
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
            self.goal_sigma_uv = self.obs_noise_uv

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

        self.optimizer_backend = str(optimizer_backend or 'auto').lower()
        self.runtime_debug = bool(runtime_debug)
        self.use_visibility_model = bool(use_visibility_model)
        self.visibility_model_name = str(visibility_model or 'none').strip().lower()
        self.visibility_weight = float(visibility_weight)
        self.visibility_cov_pos_scale = float(max(visibility_cov_pos_scale, 0.0))
        self.visibility_cov_theta_scale = float(max(visibility_cov_theta_scale, 0.0))
        self.visibility_sigma_kappa = float(max(visibility_sigma_kappa, 0.0))
        self._visibility_min_prob = 1e-4
        self.visibility_model = None

        self.g_obs = self.camera.g_uv

        self.R = np.diag([
            self.obs_noise_uv ** 2,
            self.obs_noise_uv ** 2,
        ])
        self.R_bad = np.diag([
            float(visibility_r_bad_uv) ** 2,
            float(visibility_r_bad_uv) ** 2,
        ])

        valid_visibility_models = ('raycast_25d', 'raycast25d', 'raycast')
        if self.use_visibility_model and self.visibility_model_name not in valid_visibility_models:
            raise ValueError(
                "visibility_model must be one of: raycast_25d, raycast25d, raycast"
            )
        if self.use_visibility_model and self.visibility_model_name in ('raycast_25d', 'raycast25d', 'raycast'):
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

    def _visibility_sigma_points_np(self, m, S):
        mean_xy = np.asarray(m[:2], dtype=float)
        if self.visibility_sigma_kappa <= 1e-9:
            return mean_xy.reshape(1, 2), np.array([1.0], dtype=float)

        cov_xy = np.asarray(S[:2, :2], dtype=float)
        cov_xy = 0.5 * (cov_xy + cov_xy.T) + 1e-9 * np.eye(2)
        evals, evecs = np.linalg.eigh(cov_xy)
        evals = np.clip(evals, 0.0, None)
        if float(np.max(evals)) <= 1e-12:
            return mean_xy.reshape(1, 2), np.array([1.0], dtype=float)

        sqrt_cov = evecs @ np.diag(np.sqrt(evals))
        scale = self.visibility_sigma_kappa * np.sqrt(2.0)
        deltas = [
            scale * sqrt_cov[:, 0],
            -scale * sqrt_cov[:, 0],
            scale * sqrt_cov[:, 1],
            -scale * sqrt_cov[:, 1],
        ]
        points = np.vstack([mean_xy + delta for delta in deltas])
        weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=float)
        return points, weights

    def visibility_probability_belief(self, m, S):
        if (not self.use_visibility_model) or (self.visibility_model is None):
            return 1.0
        points, weights = self._visibility_sigma_points_np(m, S)
        if points.shape[0] == 1:
            return self.visibility_probability(m)

        state = np.asarray(m, dtype=float).copy()
        probs = []
        for weight, point in zip(weights, points):
            state[0] = float(point[0])
            state[1] = float(point[1])
            probs.append(float(weight) * float(self.visibility_model.prob_state_np(state)))
        p = float(np.sum(probs))
        return float(np.clip(p, self._visibility_min_prob, 1.0 - self._visibility_min_prob))

    def _make_visibility_belief_prob_jax(self, prob_state_jax):
        import jax.numpy as jnp

        eps = float(self._visibility_min_prob)
        scale = float(self.visibility_sigma_kappa) * float(np.sqrt(2.0))
        if scale <= 1e-9:
            def p_vis_belief(m, S):
                del S
                return jnp.clip(prob_state_jax(m), eps, 1.0 - eps)
            return p_vis_belief

        def p_vis_belief(m, S):
            cov_xy = 0.5 * (S[:2, :2] + S[:2, :2].T) + 1e-9 * jnp.eye(2, dtype=S.dtype)
            evals, evecs = jnp.linalg.eigh(cov_xy)
            evals = jnp.clip(evals, 0.0, None)
            sqrt_cov = evecs @ jnp.diag(jnp.sqrt(evals))
            deltas = jnp.stack([
                scale * sqrt_cov[:, 0],
                -scale * sqrt_cov[:, 0],
                scale * sqrt_cov[:, 1],
                -scale * sqrt_cov[:, 1],
            ], axis=0)

            def _eval(delta):
                state = m.at[0].set(m[0] + delta[0])
                state = state.at[1].set(m[1] + delta[1])
                return prob_state_jax(state)

            probs = jnp.array([_eval(deltas[0]), _eval(deltas[1]), _eval(deltas[2]), _eval(deltas[3])])
            return jnp.clip(jnp.mean(probs), eps, 1.0 - eps)

        return p_vis_belief

    def _resolve_plan_problem(self, m0, goal_xy):
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

        return (
            goal_state,
            goal_cov,
            goal_obs,
            goal_obs_cov,
            use_observation_risk,
            use_ambiguity_term,
            use_state_risk,
        )

    def _initial_controls_flat(self):
        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            return np.array(self.prev_controls_flat, dtype=float)
        x0 = np.zeros(self.horizon * 2, dtype=float)
        x0[0::2] = 0.5 * (self.v_min + self.v_max)
        return x0

    def _resolve_use_jax_backend(self, backend):
        if backend not in ('auto', 'jax'):
            raise ValueError(
                f"Unknown optimizer_backend '{self.optimizer_backend}'. Expected auto|jax."
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

        cache_key = (
            self.approx_method,
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
            float(self.visibility_sigma_kappa),
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
        self._runtime_debug_print(
            "[planner_debug] JAX valgrad cache "
            f"{'hit' if valgrad is not None else 'miss'} "
            f"(horizon={self.horizon}, approx={self.approx_method})"
        )
        if valgrad is None:
            build_start = time.perf_counter()
            p_vis_jax = None
            if self.use_visibility_model and self.visibility_model is not None:
                prob_state_jax = self.visibility_model.make_prob_state_jax()
                p_vis_jax = self._make_visibility_belief_prob_jax(prob_state_jax)
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
            g_jax = jax_efe.make_g_from_homography(self.camera.H)
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
            self._runtime_debug_print(
                "[planner_debug] JAX valgrad function prepared in "
                f"{(time.perf_counter() - build_start) * 1000.0:.1f} ms"
            )

        return valgrad, jnp, goal_obs, goal_obs_cov

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
        valgrad, jnp, _, _ = self._get_jax_valgrad(
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
        val, grad = valgrad(jnp.array(x0), jnp.array(m0), jnp.array(S0))
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
            R_eff = np.asarray(self.R, dtype=float)
            S_eff = S_pred.copy()
            return 1.0, R_eff, S_eff, 1.0

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
        return np.diag([
            self.goal_sigma_uv ** 2,
            self.goal_sigma_uv ** 2,
        ]).astype(float)

    def _evaluate_controls(self, controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, return_metrics=False):
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

        total = total_risk + total_amb + total_control + total_visibility
        if return_metrics:
            return total, (total_risk, total_amb, total_control, total_visibility)
        return total

    def plan(self, m0, S0, goal_xy):
        t_plan_start = time.perf_counter()

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

        best_controls_flat = None
        backend_used = 'scipy'
        optimizer_success = False
        optimizer_status = 0
        optimizer_nit = 0
        optimizer_nfev = 0
        optimizer_message = ''
        backend = self.optimizer_backend
        use_jax = self._resolve_use_jax_backend(backend)

        try:
            if use_jax:
                import jax.numpy as jnp

                backend_used = 'jax'
                valgrad, jnp, goal_obs, goal_obs_cov = self._get_jax_valgrad(
                    goal_state,
                    goal_cov,
                    goal_obs,
                    goal_obs_cov,
                    use_observation_risk=use_observation_risk,
                    use_ambiguity_term=use_ambiguity_term,
                    use_state_risk=use_state_risk,
                )

                f_calls = {'count': 0}
                g_calls = {'count': 0}
                m0_j = jnp.array(m0)
                S0_j = jnp.array(S0)

                def f(u):
                    start = time.perf_counter()
                    val, _ = valgrad(jnp.array(u), m0_j, S0_j)
                    val_out = float(val)
                    if f_calls['count'] == 0:
                        self._runtime_debug_print(
                            "[planner_debug] First JAX objective eval returned in "
                            f"{(time.perf_counter() - start) * 1000.0:.1f} ms with J={val_out:.3f}"
                        )
                    f_calls['count'] += 1
                    return val_out

                def grad_u(u):
                    start = time.perf_counter()
                    _, grad = valgrad(jnp.array(u), m0_j, S0_j)
                    grad_out = np.array(grad, dtype=float)
                    if g_calls['count'] == 0:
                        self._runtime_debug_print(
                            "[planner_debug] First JAX gradient eval returned in "
                            f"{(time.perf_counter() - start) * 1000.0:.1f} ms with norm={np.linalg.norm(grad_out):.3f}"
                        )
                    g_calls['count'] += 1
                    return grad_out

                minimize_start = time.perf_counter()
                self._runtime_debug_print(
                    "[planner_debug] Starting scipy minimize over JAX objective "
                    f"(maxiter={self.optimizer_maxiter}, gtol={self.optimizer_gtol})"
                )
                result = minimize(
                    f,
                    x0,
                    jac=grad_u,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.optimizer_maxiter, 'gtol': self.optimizer_gtol},
                )
                self._runtime_debug_print(
                    "[planner_debug] scipy minimize finished in "
                    f"{(time.perf_counter() - minimize_start) * 1000.0:.1f} ms "
                    f"(status={getattr(result, 'status', 'n/a')}, nit={getattr(result, 'nit', 'n/a')}, nfev={getattr(result, 'nfev', 'n/a')})"
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
            else:
                raise RuntimeError(
                    "Non-autodiff optimizer path has been removed; use optimizer_backend='jax' or 'auto'."
                )
        except Exception as exc:
            raise RuntimeError(
                "Planner optimization failed "
                f"(backend_request={backend}, backend_used={backend_used}, "
                f"approx={self.approx_method}, "
                f"horizon={self.horizon}, dt={self.dt})"
            ) from exc

        self.prev_controls_flat = np.array(best_controls_flat, dtype=float)
        best_controls = self.prev_controls_flat.reshape(self.horizon, 2)
        total_cost, metrics = self._evaluate_controls(
            self.prev_controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, True
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
            visibility_cost=float(metrics[3]) if len(metrics) > 3 else 0.0,
            backend=str(backend_used),
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_nit=optimizer_nit,
            optimizer_nfev=optimizer_nfev,
            optimizer_message=optimizer_message,
            solve_time_s=solve_time_s,
        )
