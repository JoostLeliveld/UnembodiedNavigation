"""Base planner classes (pure Python, no ROS)."""

from dataclasses import dataclass
import numpy as np

from scipy.optimize import minimize

from planning.core.dynamics import unicycle_step, unicycle_jacobian, unicycle_process_noise
from planning.core.efe_utils import ET1, ET2, UT, ambiguity, risk
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
        add_ambiguity,
        optimizer_maxiter,
        optimizer_gtol,
        optimizer_warm_start,
        approx_method=None,
        use_obs_risk=None,
        use_ambiguity=None,
        seed,
        camera_params,
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
        self.add_ambiguity = bool(add_ambiguity)

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

        # Observation noise matrix
        self.R = np.diag([
            self.obs_noise_uv ** 2,
            self.obs_noise_uv ** 2,
            self.obs_noise_yaw ** 2,
        ])

        self._approx_fn = None
        if self.approx_method == 'ET1':
            self._approx_fn = ET1
        elif self.approx_method == 'ET2':
            self._approx_fn = ET2
        elif self.approx_method == 'UT':
            self._approx_fn = UT

        self.prev_controls_flat = None

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

    def approx_observation(self, m, S):
        if self._approx_fn is None:
            raise RuntimeError('Observation approximation not configured.')
        return self._approx_fn(m, S, self.camera.g, addmatrix=self.R, forceHermitian=True)

    def _goal_state(self, goal_xy, theta):
        return np.array([goal_xy[0], goal_xy[1], theta], dtype=float)

    def _goal_state_cov(self):
        return np.diag([
            self.goal_sigma_xy ** 2,
            self.goal_sigma_xy ** 2,
            self.goal_sigma_theta ** 2,
        ])

    def _goal_obs(self, goal_state):
        return np.asarray(self.camera.g(goal_state), dtype=float)

    def _goal_obs_cov(self):
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

        infeasible_penalty = 1e6
        use_obs = self.use_obs_risk
        use_amb = self.use_ambiguity and self.add_ambiguity

        for t in range(self.horizon):
            u = controls[t]
            m, S = self.predict(m, S, u)

            mu_y = Sigma_y = Gamma = None
            if use_obs or use_amb:
                mu_y, Sigma_y, Gamma = self.approx_observation(m, S)

            r_state = risk(m, S, (goal_state, goal_cov))
            r_obs = 0.0
            if use_obs and mu_y is not None:
                r_obs = risk(mu_y, Sigma_y, (goal_obs, goal_obs_cov))

            r = self.risk_weight_state * r_state + self.risk_weight_obs * r_obs
            total_risk += r

            a = 0.0
            if use_amb and Sigma_y is not None:
                a = self.ambiguity_weight * ambiguity(Sigma_y, Gamma, S)
                total_amb += a

            c = self.control_weight * float(u[0] ** 2 + u[1] ** 2)
            total_control += c

            if self.boundary_weight > 0.0:
                cell_cost, in_bounds = self._cost_at_raw(costmap, m[0], m[1])
                if (not in_bounds) or (cell_cost < 0.0) or (cell_cost >= self.lethal_cost_threshold):
                    total_boundary += self.boundary_weight
                    total = total_risk + total_amb + total_control + total_boundary + infeasible_penalty
                    if return_metrics:
                        return total, (total_risk, total_amb, total_control, total_boundary)
                    return total
                b = self.boundary_weight * (cell_cost / max(self.max_cost, 1.0))
                total_boundary += b

        total = total_risk + total_amb + total_control + total_boundary
        if return_metrics:
            return total, (total_risk, total_amb, total_control, total_boundary)
        return total

    def plan(self, m0, S0, goal_xy, costmap):
        goal_state = self._goal_state(goal_xy, m0[2])
        goal_cov = self._goal_state_cov()

        goal_obs = None
        goal_obs_cov = None
        if self.USE_OBS_RISK or self.USE_AMBIGUITY:
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
        try:
            result = minimize(
                self._evaluate_controls,
                x0,
                args=(m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, costmap, False),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.optimizer_maxiter, 'gtol': self.optimizer_gtol},
            )
            if result.success:
                best_controls_flat = result.x
        except Exception:
            best_controls_flat = None

        if best_controls_flat is None:
            if self.prev_controls_flat is not None:
                best_controls_flat = self.prev_controls_flat
            elif self.num_samples > 0:
                best_cost = float('inf')
                for _ in range(self.num_samples):
                    vs = self.rng.uniform(self.v_min, self.v_max, size=self.horizon)
                    ws = self.rng.uniform(self.w_min, self.w_max, size=self.horizon)
                    candidate = np.column_stack([vs, ws]).reshape(-1)
                    cost = self._evaluate_controls(
                        candidate, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, costmap, False
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_controls_flat = candidate
            else:
                best_controls_flat = x0

        if best_controls_flat is None:
            return None

        self.prev_controls_flat = np.array(best_controls_flat, dtype=float)
        best_controls = self.prev_controls_flat.reshape(self.horizon, 2)
        total_cost, metrics = self._evaluate_controls(
            self.prev_controls_flat, m0, S0, goal_state, goal_cov, goal_obs, goal_obs_cov, costmap, True
        )

        states = rollout_unicycle(m0, best_controls, self.dt)
        return PlanResult(
            controls=best_controls,
            states=states,
            total_cost=float(total_cost),
            risk_cost=float(metrics[0]),
            ambiguity_cost=float(metrics[1]),
            control_cost=float(metrics[2]),
            boundary_cost=float(metrics[3]),
        )
