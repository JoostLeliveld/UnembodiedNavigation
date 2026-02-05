#!/usr/bin/env python3
"""
Bayesian planner using Expected Free Energy (EFE).

Implements EFE1 (ET1) and EFE2 (ET2) approximations over BEV state.
Publishes a BEV Path for the controller to follow.
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float64MultiArray

from scipy.optimize import minimize

from planning import search_based_path_planning
from planning.efe_utils import ET1, ET2, UT, ambiguity, risk, wrap_angle
from planning.camera_model import ObliqueCameraModel


class EFEPlanner(Node):
    def __init__(self):
        super().__init__('efe_planner', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        def _declare_if_not(name, default_value):
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)

        def _as_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
            return bool(value)

        # Planner params
        _declare_if_not('planner_mode', 'efe1')  # efe1 | efe2 | efe_ut
        _declare_if_not('plan_rate', 1.0)
        _declare_if_not('horizon', 10)
        _declare_if_not('num_samples', 64)
        _declare_if_not('dt', 0.2)
        _declare_if_not('v_min', 0.0)
        _declare_if_not('v_max', 0.22)
        _declare_if_not('w_min', -1.0)
        _declare_if_not('w_max', 1.0)
        _declare_if_not('control_weight', 0.1)
        _declare_if_not('boundary_weight', 1.0)
        _declare_if_not('max_cost', 100.0)
        _declare_if_not('lethal_cost_threshold', 99.0)
        _declare_if_not('seed', 0)

        # Process/observation noise
        _declare_if_not('process_noise_xy', 0.01)
        _declare_if_not('process_noise_theta', 0.02)
        _declare_if_not('obs_noise_uv', 2.0)
        _declare_if_not('obs_noise_yaw', 0.05)

        # Goal covariance
        _declare_if_not('goal_sigma_xy', 0.25)
        _declare_if_not('goal_sigma_theta', 0.5)
        _declare_if_not('goal_sigma_uv', 0.0)
        _declare_if_not('goal_sigma_yaw', 0.0)

        # EFE weights
        _declare_if_not('risk_weight_state', 1.0)
        _declare_if_not('risk_weight_obs', 1.0)
        _declare_if_not('ambiguity_weight', 1.0)
        _declare_if_not('add_ambiguity', True)

        # Optimizer params
        _declare_if_not('optimizer_maxiter', 50)
        _declare_if_not('optimizer_gtol', 1e-4)
        _declare_if_not('optimizer_warm_start', True)

        # Pixel correction params
        _declare_if_not('use_pixel_correction', False)
        _declare_if_not('pixel_topic', '/perception/pixel_pose')
        _declare_if_not('pixel_timeout_s', 0.5)
        _declare_if_not('min_state_cov', 1e-6)

        # Camera model params (must match sim)
        _declare_if_not('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_not('look_at', [1.5, 1.5, 0.0])
        _declare_if_not('img_width', 1280)
        _declare_if_not('img_height', 720)
        _declare_if_not('fov_h_rad', 1.5708)

        self.planner_mode = self.get_parameter('planner_mode').value
        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.horizon = int(self.get_parameter('horizon').value)
        self.num_samples = int(self.get_parameter('num_samples').value)
        self.dt = float(self.get_parameter('dt').value)
        self.v_min = float(self.get_parameter('v_min').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_min = float(self.get_parameter('w_min').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.control_weight = float(self.get_parameter('control_weight').value)
        self.boundary_weight = float(self.get_parameter('boundary_weight').value)
        self.max_cost = float(self.get_parameter('max_cost').value)
        self.lethal_cost_threshold = float(self.get_parameter('lethal_cost_threshold').value)
        self.seed = int(self.get_parameter('seed').value)

        self.process_noise_xy = float(self.get_parameter('process_noise_xy').value)
        self.process_noise_theta = float(self.get_parameter('process_noise_theta').value)
        self.obs_noise_uv = float(self.get_parameter('obs_noise_uv').value)
        self.obs_noise_yaw = float(self.get_parameter('obs_noise_yaw').value)

        self.goal_sigma_xy = float(self.get_parameter('goal_sigma_xy').value)
        self.goal_sigma_theta = float(self.get_parameter('goal_sigma_theta').value)
        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)
        self.goal_sigma_yaw = float(self.get_parameter('goal_sigma_yaw').value)

        if self.goal_sigma_uv <= 0.0:
            self.goal_sigma_uv = self.obs_noise_uv
        if self.goal_sigma_yaw <= 0.0:
            self.goal_sigma_yaw = self.obs_noise_yaw

        self.risk_weight_state = float(self.get_parameter('risk_weight_state').value)
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.add_ambiguity = _as_bool(self.get_parameter('add_ambiguity').value)

        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = _as_bool(self.get_parameter('optimizer_warm_start').value)

        self.use_pixel_correction = _as_bool(self.get_parameter('use_pixel_correction').value)
        self.pixel_topic = self.get_parameter('pixel_topic').value
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.min_state_cov = float(self.get_parameter('min_state_cov').value)

        self.rng = np.random.default_rng(self.seed)

        # Camera model
        self.camera = ObliqueCameraModel(
            cam_pos=self.get_parameter('cam_pos').value,
            look_at=self.get_parameter('look_at').value,
            img_width=self.get_parameter('img_width').value,
            img_height=self.get_parameter('img_height').value,
            fov_h_rad=self.get_parameter('fov_h_rad').value,
        )

        # Subscriptions
        state_qos = QoSProfile(depth=1)
        state_qos.durability = DurabilityPolicy.VOLATILE
        self.state_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/state/bev', self._state_cb, qos_profile=state_qos
        )
        goal_qos = QoSProfile(depth=1)
        goal_qos.durability = DurabilityPolicy.VOLATILE
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos
        )
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap', self._costmap_cb, qos_profile=costmap_qos
        )
        self.pixel_sub = self.create_subscription(
            PoseStamped, self.pixel_topic, self._pixel_cb, qos_profile=state_qos
        )
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_cb, qos_profile=state_qos
        )

        # Publisher
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(Path, '/plan', qos_profile=path_qos)
        self.metrics_pub = self.create_publisher(Float64MultiArray, '/efe/metrics', 10)

        # State
        self.state_msg = None
        self.goal_msg = None
        self.costmap_msg = None
        self.prev_controls_flat = None
        self.pixel_meas = None
        self.pixel_stamp = None
        self._last_correction_log = 0.0
        self._last_stale_log = 0.0
        self.belief_m = None
        self.belief_S = None
        self.belief_stamp = None
        self.last_cmd = np.array([0.0, 0.0], dtype=float)

        self.create_timer(1.0 / max(self.plan_rate, 0.1), self._plan_once)
        self.get_logger().info(f"EFE planner started in mode {self.planner_mode}")

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_msg = msg

    def _cmd_cb(self, msg: Twist):
        self.last_cmd = np.array([msg.linear.x, msg.angular.z], dtype=float)

    def _init_belief_from_state(self):
        if self.state_msg is None:
            return False
        q = self.state_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        self.belief_m = np.array([
            self.state_msg.pose.pose.position.x,
            self.state_msg.pose.pose.position.y,
            theta
        ], dtype=float)
        cov = self.state_msg.pose.covariance
        self.belief_S = np.diag([
            cov[0] if len(cov) > 0 else 1e-6,
            cov[7] if len(cov) > 7 else 1e-6,
            cov[35] if len(cov) > 35 else 1e-6
        ]).astype(float)
        if self.min_state_cov > 0.0:
            for i in range(min(3, self.belief_S.shape[0])):
                if self.belief_S[i, i] < self.min_state_cov:
                    self.belief_S[i, i] = self.min_state_cov
        self.belief_stamp = self.state_msg.header.stamp
        return True

    def _pixel_cb(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.pixel_meas = np.array([u, v, yaw], dtype=float)
        self.pixel_stamp = msg.header.stamp

        if not self.use_pixel_correction:
            return

        try:
            now = self.get_clock().now()
            age = (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        except Exception:
            age = 0.0
        if age > self.pixel_timeout_s:
            now_wall = time.monotonic()
            if now_wall - self._last_stale_log > 2.0:
                self.get_logger().warn(f"Skipping stale pixel measurement (age {age:.2f}s)")
                self._last_stale_log = now_wall
            return

        # Initialize belief if needed (fallback to /state/bev once)
        if self.belief_m is None or self.belief_S is None:
            if not self._init_belief_from_state():
                return

        # Compute dt from last belief update (default to planner dt)
        try:
            now = Time.from_msg(msg.header.stamp)
            last = Time.from_msg(self.belief_stamp) if self.belief_stamp is not None else None
            dt_s = (now - last).nanoseconds * 1e-9 if last is not None else self.dt
            if dt_s <= 0.0:
                dt_s = self.dt
        except Exception:
            dt_s = self.dt

        # Predict belief with last commanded control
        v_cmd, w_cmd = float(self.last_cmd[0]), float(self.last_cmd[1])
        m_pred = self._unicycle_step(self.belief_m, np.array([v_cmd, w_cmd], dtype=float), dt=dt_s)
        F = self._unicycle_jacobian(m_pred, np.array([v_cmd, w_cmd], dtype=float), dt=dt_s)
        Q = np.diag([
            self.process_noise_xy ** 2,
            self.process_noise_xy ** 2,
            self.process_noise_theta ** 2
        ])
        if self.dt > 1e-9:
            Q = Q * max(dt_s / self.dt, 0.0)
        S_pred = F @ self.belief_S @ F.T + Q

        # Observation noise (pixel space)
        R = np.diag([
            self.obs_noise_uv ** 2,
            self.obs_noise_uv ** 2,
            self.obs_noise_yaw ** 2
        ])

        # Choose approximation
        mode = self.planner_mode.lower()
        if mode in ('efe1', 'et1'):
            approx = 'ET1'
        elif mode in ('efe2', 'et2'):
            approx = 'ET2'
        elif mode in ('efe_ut', 'ut'):
            approx = 'UT'
        else:
            approx = 'ET2'

        if approx == 'UT':
            mu_y, Sigma_y, Gamma = UT(m_pred, S_pred, self.camera.g, addmatrix=R, forceHermitian=True)
        elif approx == 'ET2':
            mu_y, Sigma_y, Gamma = ET2(m_pred, S_pred, self.camera.g, addmatrix=R, forceHermitian=True)
        else:
            mu_y, Sigma_y, Gamma = ET1(m_pred, S_pred, self.camera.g, addmatrix=R, forceHermitian=True)

        innov = self.pixel_meas - mu_y
        if innov.size >= 3:
            innov[2] = wrap_angle(innov[2])
        Sigma_y = (Sigma_y + Sigma_y.T) / 2.0
        Sigma_inv = np.linalg.pinv(Sigma_y)
        K = Gamma @ Sigma_inv
        self.belief_m = m_pred + K @ innov
        self.belief_m[2] = wrap_angle(self.belief_m[2])
        self.belief_S = S_pred - K @ Sigma_y @ K.T
        self.belief_S = (self.belief_S + self.belief_S.T) / 2.0
        if self.min_state_cov > 0.0:
            for i in range(min(3, self.belief_S.shape[0])):
                if self.belief_S[i, i] < self.min_state_cov:
                    self.belief_S[i, i] = self.min_state_cov
        self.belief_stamp = msg.header.stamp

        now_wall = time.monotonic()
        if now_wall - self._last_correction_log > 2.0:
            self.get_logger().info("Applied pixel correction in callback")
            self._last_correction_log = now_wall

    def _unicycle_step(self, state, control, dt=None):
        x, y, theta = state
        v, w = control
        step_dt = self.dt if dt is None else float(dt)
        x = x + v * step_dt * math.cos(theta)
        y = y + v * step_dt * math.sin(theta)
        theta = wrap_angle(theta + w * step_dt)
        return np.array([x, y, theta], dtype=float)

    def _unicycle_jacobian(self, state, control, dt=None):
        _, _, theta = state
        v, _ = control
        step_dt = self.dt if dt is None else float(dt)
        F = np.eye(3)
        F[0, 2] = -v * step_dt * math.sin(theta)
        F[1, 2] = v * step_dt * math.cos(theta)
        return F

    def _cost_at(self, x, y):
        if self.costmap_msg is None:
            return 0.0
        origin = np.asarray([
            self.costmap_msg.info.origin.position.x,
            self.costmap_msg.info.origin.position.y
        ])
        res = self.costmap_msg.info.resolution
        grid = search_based_path_planning.world_to_grid([x, y], origin=origin, resolution=res)[0]
        i, j = int(grid[0]), int(grid[1])
        if i < 0 or j < 0 or i >= self.costmap_msg.info.height or j >= self.costmap_msg.info.width:
            return self.max_cost
        costmap_matrix = np.array(self.costmap_msg.data).reshape(
            self.costmap_msg.info.height, self.costmap_msg.info.width
        )
        cost = costmap_matrix[i, j]
        if cost < 0:
            return self.max_cost
        return float(cost)

    def _cost_at_raw(self, x, y):
        if self.costmap_msg is None:
            return 0.0, True
        origin = np.asarray([
            self.costmap_msg.info.origin.position.x,
            self.costmap_msg.info.origin.position.y
        ])
        res = self.costmap_msg.info.resolution
        grid = search_based_path_planning.world_to_grid([x, y], origin=origin, resolution=res)[0]
        i, j = int(grid[0]), int(grid[1])
        if i < 0 or j < 0 or i >= self.costmap_msg.info.height or j >= self.costmap_msg.info.width:
            return self.max_cost, False
        costmap_matrix = np.array(self.costmap_msg.data).reshape(
            self.costmap_msg.info.height, self.costmap_msg.info.width
        )
        cost = float(costmap_matrix[i, j])
        return cost, True

    def _plan_once(self):
        if self.goal_msg is None or self.costmap_msg is None:
            return

        # Determine belief prior
        if self.use_pixel_correction:
            if self.belief_m is None or self.belief_S is None:
                if not self._init_belief_from_state():
                    return
            m0 = self.belief_m.copy()
            S0 = self.belief_S.copy()
            if self.belief_stamp is not None:
                try:
                    now = self.get_clock().now()
                    stamp = Time.from_msg(self.belief_stamp)
                    age = (now - stamp).nanoseconds * 1e-9
                except Exception:
                    age = 0.0
                if age > self.pixel_timeout_s:
                    now_wall = time.monotonic()
                    if now_wall - self._last_stale_log > 2.0:
                        self.get_logger().warn(f"Pixel belief stale (age {age:.2f}s)")
                        self._last_stale_log = now_wall
        else:
            if self.state_msg is None:
                return
            q = self.state_msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            theta = math.atan2(siny_cosp, cosy_cosp)
            m0 = np.array([
                self.state_msg.pose.pose.position.x,
                self.state_msg.pose.pose.position.y,
                theta
            ], dtype=float)

            cov = self.state_msg.pose.covariance
            S0 = np.diag([
                cov[0] if len(cov) > 0 else 1e-6,
                cov[7] if len(cov) > 7 else 1e-6,
                cov[35] if len(cov) > 35 else 1e-6
            ]).astype(float)
            if self.min_state_cov > 0.0:
                for i in range(min(3, S0.shape[0])):
                    if S0[i, i] < self.min_state_cov:
                        S0[i, i] = self.min_state_cov

        # Goal mean/cov in state space
        goal_theta = m0[2]
        goal_mean = np.array([
            self.goal_msg.pose.position.x,
            self.goal_msg.pose.position.y,
            goal_theta
        ], dtype=float)
        goal_cov = np.diag([
            self.goal_sigma_xy ** 2,
            self.goal_sigma_xy ** 2,
            self.goal_sigma_theta ** 2
        ])

        # Goal mean/cov in observation space (camera model)
        goal_obs_mean = np.asarray(self.camera.g(goal_mean), dtype=float)
        goal_obs_cov = np.diag([
            self.goal_sigma_uv ** 2,
            self.goal_sigma_uv ** 2,
            self.goal_sigma_yaw ** 2
        ]).astype(float)

        # Process noise
        Q = np.diag([
            self.process_noise_xy ** 2,
            self.process_noise_xy ** 2,
            self.process_noise_theta ** 2
        ])

        # Observation noise (pixel space)
        R = np.diag([
            self.obs_noise_uv ** 2,
            self.obs_noise_uv ** 2,
            self.obs_noise_yaw ** 2
        ])

        # Choose approximation
        mode = self.planner_mode.lower()
        if mode in ('efe1', 'et1'):
            approx = 'ET1'
        elif mode in ('efe2', 'et2'):
            approx = 'ET2'
        elif mode in ('efe_ut', 'ut'):
            approx = 'UT'
        else:
            self.get_logger().warn(f"Unknown planner_mode '{self.planner_mode}', defaulting to ET2")
            approx = 'ET2'

        infeasible_penalty = 1e6

        def evaluate(controls_flat, return_metrics=False):
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

            for t in range(self.horizon):
                u = controls[t]
                # Predict
                m = self._unicycle_step(m, u)
                F = self._unicycle_jacobian(m, u)
                S = F @ S @ F.T + Q

                # Observation model
                if approx == 'UT':
                    mu_y, Sigma_y, Gamma = UT(m, S, self.camera.g, addmatrix=R, forceHermitian=True)
                elif approx == 'ET2':
                    mu_y, Sigma_y, Gamma = ET2(m, S, self.camera.g, addmatrix=R, forceHermitian=True)
                else:
                    mu_y, Sigma_y, Gamma = ET1(m, S, self.camera.g, addmatrix=R, forceHermitian=True)

                # EFE terms (risk in state + risk in observation + ambiguity)
                r_state = risk(m, S, (goal_mean, goal_cov))
                r_obs = risk(mu_y, Sigma_y, (goal_obs_mean, goal_obs_cov))
                r = self.risk_weight_state * r_state + self.risk_weight_obs * r_obs
                total_risk += r

                a = 0.0
                if self.add_ambiguity:
                    a = self.ambiguity_weight * ambiguity(Sigma_y, Gamma, S)
                    total_amb += a

                c = self.control_weight * float(u[0] ** 2 + u[1] ** 2)
                total_control += c

                total = total_risk + total_amb + total_control + total_boundary

                # Boundary cost
                if self.boundary_weight > 0.0:
                    cell_cost, in_bounds = self._cost_at_raw(m[0], m[1])
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

        # Bounds for optimizer
        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((self.w_min, self.w_max))

        # Initial guess
        if self.optimizer_warm_start and self.prev_controls_flat is not None:
            x0 = np.array(self.prev_controls_flat, dtype=float)
        else:
            x0 = np.zeros(self.horizon * 2, dtype=float)
            x0[0::2] = 0.5 * (self.v_min + self.v_max)

        best_controls_flat = None

        try:
            result = minimize(
                evaluate,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.optimizer_maxiter, 'gtol': self.optimizer_gtol}
            )
            if result.success:
                best_controls_flat = result.x
            else:
                self.get_logger().warn(f"L-BFGS-B failed: {result.message}")
        except Exception as exc:
            self.get_logger().warn(f"L-BFGS-B exception: {exc}")

        # Fallbacks if optimizer fails
        if best_controls_flat is None:
            if self.prev_controls_flat is not None:
                best_controls_flat = self.prev_controls_flat
            elif self.num_samples > 0:
                best_cost = float('inf')
                for _ in range(self.num_samples):
                    vs = self.rng.uniform(self.v_min, self.v_max, size=self.horizon)
                    ws = self.rng.uniform(self.w_min, self.w_max, size=self.horizon)
                    candidate = np.column_stack([vs, ws]).reshape(-1)
                    cost = evaluate(candidate, return_metrics=False)
                    if cost < best_cost:
                        best_cost = cost
                        best_controls_flat = candidate
            else:
                best_controls_flat = x0

        if best_controls_flat is None:
            return

        self.prev_controls_flat = np.array(best_controls_flat, dtype=float)
        best_controls = self.prev_controls_flat.reshape(self.horizon, 2)
        total_cost, best_metrics = evaluate(self.prev_controls_flat, return_metrics=True)

        # Build a path from best controls
        path = Path()
        path.header.frame_id = self.costmap_msg.header.frame_id or self.state_msg.header.frame_id or 'map_bev'
        path.header.stamp = self.get_clock().now().to_msg()

        m = m0.copy()
        start_pose = PoseStamped()
        start_pose.header = path.header
        start_pose.pose.position.x = float(m[0])
        start_pose.pose.position.y = float(m[1])
        path.poses.append(start_pose)

        for t in range(self.horizon):
            m = self._unicycle_step(m, best_controls[t])
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(m[0])
            p.pose.position.y = float(m[1])
            path.poses.append(p)

        # Append goal for controller convenience
        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose.position.x = float(goal_mean[0])
        goal_pose.pose.position.y = float(goal_mean[1])
        path.poses.append(goal_pose)

        self.path_pub.publish(path)

        if best_metrics is not None:
            msg = Float64MultiArray()
            msg.data = [
                float(total_cost),
                float(best_metrics[0]),
                float(best_metrics[1]),
                float(best_metrics[2]),
                float(best_metrics[3]),
            ]
            self.metrics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EFEPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
