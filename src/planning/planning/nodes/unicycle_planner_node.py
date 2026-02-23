"""Thin ROS 2 wrapper around unicycle planners."""

import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float64MultiArray

from planning.core.efe_utils import wrap_angle
from planning.planners.base_planner import CostmapData


class UnicyclePlannerNode(Node):
    """Base class for EFE/MPC planners using unicycle dynamics."""

    NODE_NAME = 'planner'
    PLANNER_CLASS = None
    PARAM_DEFAULT_OVERRIDES = {}

    def __init__(self):
        super().__init__(self.NODE_NAME, allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        if self.PLANNER_CLASS is None:
            raise RuntimeError('PLANNER_CLASS is not set.')

        node_defaults = dict(getattr(self, 'PARAM_DEFAULT_OVERRIDES', {}) or {})

        def _declare_if_not(name, default_value):
            if name in node_defaults:
                default_value = node_defaults[name]
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)

        def _as_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
            return bool(value)

        # Planner params
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
        _declare_if_not('approx_method', 'ET2')
        _declare_if_not('use_obs_risk', True)
        _declare_if_not('use_ambiguity', True)
        _declare_if_not('obs_mode', 'uv')
        _declare_if_not('optimizer_backend', 'auto')

        # Optimizer params
        _declare_if_not('optimizer_maxiter', 50)
        _declare_if_not('optimizer_gtol', 1e-4)
        _declare_if_not('optimizer_warm_start', True)

        # Pixel correction params
        _declare_if_not('use_pixel_correction', False)
        _declare_if_not('pixel_topic', '/perception/pixel_pose')
        _declare_if_not('pixel_timeout_s', 0.5)
        _declare_if_not('pixel_correction_approx', 'ET1')
        _declare_if_not('skip_stale_pixel_correction', True)
        _declare_if_not('min_state_cov', 1e-6)
        _declare_if_not('debug_runtime', True)
        _declare_if_not('debug_log_period_s', 2.0)
        _declare_if_not('slow_plan_factor', 0.8)
        _declare_if_not('slow_correction_ms', 10.0)

        # Camera model params (must match sim)
        _declare_if_not('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_not('look_at', [1.5, 1.5, 0.0])
        _declare_if_not('img_width', 1280)
        _declare_if_not('img_height', 720)
        _declare_if_not('fov_h_rad', 1.5708)

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

        self.risk_weight_state = float(self.get_parameter('risk_weight_state').value)
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.add_ambiguity = _as_bool(self.get_parameter('add_ambiguity').value)
        self.approx_method = str(self.get_parameter('approx_method').value).upper()
        self.use_obs_risk = _as_bool(self.get_parameter('use_obs_risk').value)
        self.use_ambiguity = _as_bool(self.get_parameter('use_ambiguity').value)
        self.obs_mode = str(self.get_parameter('obs_mode').value).strip().lower()
        self.optimizer_backend = str(self.get_parameter('optimizer_backend').value).strip().lower()
        if self.obs_mode not in ('uv', 'uvt'):
            raise RuntimeError("obs_mode must be 'uv' or 'uvt'")

        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = _as_bool(self.get_parameter('optimizer_warm_start').value)

        self.use_pixel_correction = _as_bool(self.get_parameter('use_pixel_correction').value)
        self.pixel_topic = self.get_parameter('pixel_topic').value
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.pixel_correction_approx = str(
            self.get_parameter('pixel_correction_approx').value
        ).strip().upper()
        if self.pixel_correction_approx not in ('AUTO', 'ET1', 'ET2', 'UT'):
            raise RuntimeError("pixel_correction_approx must be one of: AUTO, ET1, ET2, UT")
        self.skip_stale_pixel_correction = _as_bool(
            self.get_parameter('skip_stale_pixel_correction').value
        )
        self.min_state_cov = float(self.get_parameter('min_state_cov').value)
        self.debug_runtime = _as_bool(self.get_parameter('debug_runtime').value)
        self.debug_log_period_s = max(0.2, float(self.get_parameter('debug_log_period_s').value))
        self.slow_plan_factor = max(0.1, float(self.get_parameter('slow_plan_factor').value))
        self.slow_correction_ms = max(0.1, float(self.get_parameter('slow_correction_ms').value))

        camera_params = {
            'cam_pos': self.get_parameter('cam_pos').value,
            'look_at': self.get_parameter('look_at').value,
            'img_width': int(self.get_parameter('img_width').value),
            'img_height': int(self.get_parameter('img_height').value),
            'fov_h_rad': float(self.get_parameter('fov_h_rad').value),
        }

        self.planner = self.PLANNER_CLASS(
            horizon=self.horizon,
            dt=self.dt,
            v_min=self.v_min,
            v_max=self.v_max,
            w_min=self.w_min,
            w_max=self.w_max,
            control_weight=self.control_weight,
            boundary_weight=self.boundary_weight,
            max_cost=self.max_cost,
            lethal_cost_threshold=self.lethal_cost_threshold,
            num_samples=self.num_samples,
            process_noise_xy=self.process_noise_xy,
            process_noise_theta=self.process_noise_theta,
            obs_noise_uv=self.obs_noise_uv,
            obs_noise_yaw=self.obs_noise_yaw,
            goal_sigma_xy=self.goal_sigma_xy,
            goal_sigma_theta=self.goal_sigma_theta,
            goal_sigma_uv=self.goal_sigma_uv,
            goal_sigma_yaw=self.goal_sigma_yaw,
            risk_weight_state=self.risk_weight_state,
            risk_weight_obs=self.risk_weight_obs,
            ambiguity_weight=self.ambiguity_weight,
            add_ambiguity=self.add_ambiguity,
            optimizer_maxiter=self.optimizer_maxiter,
            optimizer_gtol=self.optimizer_gtol,
            optimizer_warm_start=self.optimizer_warm_start,
            approx_method=self.approx_method,
            use_obs_risk=self.use_obs_risk,
            use_ambiguity=self.use_ambiguity,
            obs_mode=self.obs_mode,
            optimizer_backend=self.optimizer_backend,
            seed=self.seed,
            camera_params=camera_params,
        )
        # Use the planner-normalized mode as the single source of truth in this node.
        self.obs_mode = str(self.planner.obs_mode)
        self._io_group = ReentrantCallbackGroup()
        self._plan_group = MutuallyExclusiveCallbackGroup()
        self._data_lock = threading.RLock()

        # Subscriptions
        state_qos = QoSProfile(depth=1)
        state_qos.durability = DurabilityPolicy.VOLATILE
        self.state_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/state/bev', self._state_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        goal_qos = QoSProfile(depth=1)
        goal_qos.durability = DurabilityPolicy.VOLATILE
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos,
            callback_group=self._io_group
        )
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap', self._costmap_cb, qos_profile=costmap_qos,
            callback_group=self._io_group
        )
        self.pixel_sub = self.create_subscription(
            PoseStamped, self.pixel_topic, self._pixel_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )

        # Publishers
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(Path, '/plan', qos_profile=path_qos)
        self.metrics_pub = self.create_publisher(Float64MultiArray, '/efe/metrics', 10)

        # State
        self.state_msg = None
        self.goal_msg = None
        self.costmap = None
        self.pixel_meas = None
        self.pixel_stamp = None
        self._last_correction_log = 0.0
        self._last_stale_log = 0.0
        self._last_shape_mismatch_log = 0.0
        self._last_missing_costmap_log = 0.0
        self._last_frame_mismatch_log = 0.0
        self._last_runtime_log = 0.0
        self._last_slow_plan_log = 0.0
        self._last_slow_correction_log = 0.0
        self.belief_m = None
        self.belief_S = None
        self.belief_stamp = None
        self.last_cmd = np.array([0.0, 0.0], dtype=float)
        self._costmap_required = self.boundary_weight > 0.0

        self._plan_period_s = 1.0 / max(self.plan_rate, 0.1)
        self.create_timer(self._plan_period_s, self._plan_once, callback_group=self._plan_group)
        self.get_logger().info(
            f"{self.NODE_NAME} started "
            f"(approx={self.approx_method}, obs_mode={self.obs_mode}, "
            f"use_obs_risk={self.use_obs_risk}, use_ambiguity={self.use_ambiguity and self.add_ambiguity}, "
            f"boundary_weight={self.boundary_weight:.3f}, "
            f"costmap_required={self._costmap_required}, "
            f"use_pixel_correction={self.use_pixel_correction}, "
            f"pixel_correction_approx={self.pixel_correction_approx}, "
            f"debug_runtime={self.debug_runtime})"
        )

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        with self._data_lock:
            self.state_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        with self._data_lock:
            self.goal_msg = msg

    def _costmap_cb(self, msg: OccupancyGrid):
        data = np.array(msg.data, dtype=float).reshape(msg.info.height, msg.info.width)
        origin = np.asarray([
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        ], dtype=float)
        with self._data_lock:
            self.costmap = CostmapData(
                origin=origin,
                resolution=float(msg.info.resolution),
                width=int(msg.info.width),
                height=int(msg.info.height),
                data=data,
                frame_id=msg.header.frame_id,
            )

    def _cmd_cb(self, msg: Twist):
        with self._data_lock:
            self.last_cmd = np.array([msg.linear.x, msg.angular.z], dtype=float)

    def _init_belief_from_state(self):
        with self._data_lock:
            if self.state_msg is None:
                return False
            q = self.state_msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            theta = math.atan2(siny_cosp, cosy_cosp)
            self.belief_m = np.array([
                self.state_msg.pose.pose.position.x,
                self.state_msg.pose.pose.position.y,
                theta,
            ], dtype=float)
            cov = self.state_msg.pose.covariance
            self.belief_S = np.diag([
                cov[0] if len(cov) > 0 else 1e-6,
                cov[7] if len(cov) > 7 else 1e-6,
                cov[35] if len(cov) > 35 else 1e-6,
            ]).astype(float)
            if self.min_state_cov > 0.0:
                for i in range(min(3, self.belief_S.shape[0])):
                    if self.belief_S[i, i] < self.min_state_cov:
                        self.belief_S[i, i] = self.min_state_cov
            self.belief_stamp = self.state_msg.header.stamp
            return True

    def _pixel_cb(self, msg: PoseStamped):
        cb_start = time.perf_counter()
        u = msg.pose.position.x
        v = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # Keep pixel measurement dimensionality consistent with planner obs_mode.
        with self._data_lock:
            if self.obs_mode == 'uv':
                self.pixel_meas = np.array([u, v], dtype=float)
            else:
                self.pixel_meas = np.array([u, v, yaw], dtype=float)
            self.pixel_stamp = msg.header.stamp

        if not self.use_pixel_correction:
            return

        try:
            now = self.get_clock().now()
            age = (now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        except Exception:
            age = 0.0
        if self.skip_stale_pixel_correction and age > self.pixel_timeout_s:
            now_wall = time.monotonic()
            if now_wall - self._last_stale_log > 2.0:
                self.get_logger().warn(f"Skipping stale pixel measurement (age {age:.2f}s)")
                self._last_stale_log = now_wall
            return

        with self._data_lock:
            has_belief = self.belief_m is not None and self.belief_S is not None
        if not has_belief and not self._init_belief_from_state():
            return

        try:
            now = Time.from_msg(msg.header.stamp)
            with self._data_lock:
                stamp_ref = self.belief_stamp
            last = Time.from_msg(stamp_ref) if stamp_ref is not None else None
            dt_s = (now - last).nanoseconds * 1e-9 if last is not None else self.dt
            if dt_s <= 0.0:
                dt_s = self.dt
        except Exception:
            dt_s = self.dt

        with self._data_lock:
            belief_m = None if self.belief_m is None else self.belief_m.copy()
            belief_S = None if self.belief_S is None else self.belief_S.copy()
            v_cmd, w_cmd = float(self.last_cmd[0]), float(self.last_cmd[1])
            meas = None if self.pixel_meas is None else self.pixel_meas.copy()
        if belief_m is None or belief_S is None or meas is None:
            return

        m_pred, S_pred = self.planner.predict(
            belief_m, belief_S, np.array([v_cmd, w_cmd], dtype=float), dt=dt_s
        )

        corr_method = self.approx_method if self.pixel_correction_approx == 'AUTO' else self.pixel_correction_approx
        mu_y, Sigma_y, Gamma = self.planner.approx_observation(m_pred, S_pred, method=corr_method)
        mu_y = np.asarray(mu_y, dtype=float).reshape(-1)
        meas = np.asarray(meas, dtype=float).reshape(-1)
        if meas.size != mu_y.size:
            now_wall = time.monotonic()
            if now_wall - self._last_shape_mismatch_log > 2.0:
                self.get_logger().error(
                    "Pixel correction shape mismatch: "
                    f"obs_mode={self.obs_mode}, meas_dim={meas.size}, pred_dim={mu_y.size}. "
                    "Skipping correction for this message."
                )
                self._last_shape_mismatch_log = now_wall
            return
        Sigma_y = np.asarray(Sigma_y, dtype=float)
        Gamma = np.asarray(Gamma, dtype=float)
        if Sigma_y.shape != (meas.size, meas.size) or Gamma.shape[1] != meas.size:
            now_wall = time.monotonic()
            if now_wall - self._last_shape_mismatch_log > 2.0:
                self.get_logger().error(
                    "Pixel correction covariance shape mismatch: "
                    f"Sigma_y={Sigma_y.shape}, Gamma={Gamma.shape}, meas_dim={meas.size}. "
                    "Skipping correction for this message."
                )
                self._last_shape_mismatch_log = now_wall
            return
        innov = meas - mu_y
        if innov.size >= 3:
            innov[2] = wrap_angle(innov[2])
        Sigma_y = (Sigma_y + Sigma_y.T) / 2.0
        Sigma_inv = np.linalg.pinv(Sigma_y)
        K = Gamma @ Sigma_inv
        next_m = m_pred + K @ innov
        next_m[2] = wrap_angle(next_m[2])
        next_S = S_pred - K @ Sigma_y @ K.T
        next_S = (next_S + next_S.T) / 2.0
        if self.min_state_cov > 0.0:
            for i in range(min(3, next_S.shape[0])):
                if next_S[i, i] < self.min_state_cov:
                    next_S[i, i] = self.min_state_cov
        with self._data_lock:
            self.belief_m = next_m
            self.belief_S = next_S
            self.belief_stamp = msg.header.stamp

        now_wall = time.monotonic()
        if now_wall - self._last_correction_log > 2.0:
            self.get_logger().info(
                f"Applied pixel correction in callback "
                f"(method={corr_method}, age={age:.3f}s, dt={dt_s:.3f}s)"
            )
            self._last_correction_log = now_wall

        cb_ms = max((time.perf_counter() - cb_start) * 1000.0, 0.0)
        if cb_ms > self.slow_correction_ms and (now_wall - self._last_slow_correction_log) > 2.0:
            self.get_logger().warn(
                f"Slow pixel correction callback ({cb_ms:.1f} ms) "
                f"using {corr_method}; this can cause stale-belief behavior."
            )
            self._last_slow_correction_log = now_wall

    def _resolve_belief_for_planning(self):
        if self.use_pixel_correction:
            with self._data_lock:
                has_belief = self.belief_m is not None and self.belief_S is not None
            if not has_belief:
                if not self._init_belief_from_state():
                    return None, None
            with self._data_lock:
                m0 = self.belief_m.copy()
                S0 = self.belief_S.copy()
                stamp_ref = self.belief_stamp
            if stamp_ref is not None:
                try:
                    now = self.get_clock().now()
                    stamp = Time.from_msg(stamp_ref)
                    age = (now - stamp).nanoseconds * 1e-9
                except Exception:
                    age = 0.0
                if age > self.pixel_timeout_s:
                    now_wall = time.monotonic()
                    if now_wall - self._last_stale_log > 2.0:
                        self.get_logger().warn(f"Pixel belief stale (age {age:.2f}s)")
                        self._last_stale_log = now_wall
        else:
            with self._data_lock:
                state_ref = self.state_msg
            if state_ref is None:
                return None, None
            q = state_ref.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            theta = math.atan2(siny_cosp, cosy_cosp)
            m0 = np.array([
                state_ref.pose.pose.position.x,
                state_ref.pose.pose.position.y,
                theta,
            ], dtype=float)

            cov = state_ref.pose.covariance
            S0 = np.diag([
                cov[0] if len(cov) > 0 else 1e-6,
                cov[7] if len(cov) > 7 else 1e-6,
                cov[35] if len(cov) > 35 else 1e-6,
            ]).astype(float)
            if self.min_state_cov > 0.0:
                for i in range(min(3, S0.shape[0])):
                    if S0[i, i] < self.min_state_cov:
                        S0[i, i] = self.min_state_cov
        return m0, S0

    def _build_path_message(self, result, goal_xy):
        path = Path()
        with self._data_lock:
            costmap_ref = self.costmap
            state_ref = self.state_msg
        frame_id = (
            (costmap_ref.frame_id if costmap_ref is not None else '')
            or (state_ref.header.frame_id if state_ref else '')
            or 'map_bev'
        )
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        for state in result.states:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(state[0])
            p.pose.position.y = float(state[1])
            path.poses.append(p)

        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose.position.x = float(goal_xy[0])
        goal_pose.pose.position.y = float(goal_xy[1])
        path.poses.append(goal_pose)
        return path

    def _publish_plan_and_metrics(self, result, goal_xy):
        path = self._build_path_message(result, goal_xy)
        self.path_pub.publish(path)

        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            float(result.total_cost),
            float(result.risk_cost),
            float(result.ambiguity_cost),
            float(result.control_cost),
            float(result.boundary_cost),
        ]
        self.metrics_pub.publish(metrics_msg)

    def _after_plan_result(self, result):
        """Hook for subclasses (e.g. agent node) to publish extra outputs."""
        return

    def _plan_once(self):
        with self._data_lock:
            goal_ref = self.goal_msg
            costmap_ref = self.costmap
            pixel_stamp_ref = self.pixel_stamp
            state_ref = self.state_msg
        if goal_ref is None:
            return

        if self._costmap_required and costmap_ref is None:
            now_wall = time.monotonic()
            if now_wall - self._last_missing_costmap_log > 2.0:
                self.get_logger().warn(
                    "Costmap required (boundary_weight > 0) but no /costmap received yet; skipping plan step."
                )
                self._last_missing_costmap_log = now_wall
            return

        goal_frame = (goal_ref.header.frame_id or '').strip()
        state_frame = (state_ref.header.frame_id or '').strip() if state_ref is not None else ''
        costmap_frame = (costmap_ref.frame_id or '').strip() if costmap_ref is not None else ''
        now_wall = time.monotonic()
        if goal_frame and state_frame and goal_frame != state_frame:
            if now_wall - self._last_frame_mismatch_log > 2.0:
                self.get_logger().error(
                    "Frame mismatch: /goal_bev and /state/bev differ "
                    f"(goal='{goal_frame}', state='{state_frame}'). Skipping plan step."
                )
                self._last_frame_mismatch_log = now_wall
            return
        if self._costmap_required and goal_frame and costmap_frame and goal_frame != costmap_frame:
            if now_wall - self._last_frame_mismatch_log > 2.0:
                self.get_logger().error(
                    "Frame mismatch: /goal_bev and /costmap differ "
                    f"(goal='{goal_frame}', costmap='{costmap_frame}'). Skipping plan step."
                )
                self._last_frame_mismatch_log = now_wall
            return

        m0, S0 = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return

        goal_xy = (
            float(goal_ref.pose.position.x),
            float(goal_ref.pose.position.y),
        )

        plan_start = time.perf_counter()
        result = self.planner.plan(m0, S0, goal_xy, costmap_ref)
        if result is None:
            return

        self._publish_plan_and_metrics(result, goal_xy)
        self._after_plan_result(result)

        plan_elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
        solve_elapsed_ms = float(getattr(result, 'solve_time_s', 0.0)) * 1000.0
        if plan_elapsed_ms > (self.slow_plan_factor * self._plan_period_s * 1000.0):
            if now_wall - self._last_slow_plan_log > 2.0:
                self.get_logger().warn(
                    f"Slow plan cycle ({plan_elapsed_ms:.1f} ms, solver={solve_elapsed_ms:.1f} ms, "
                    f"period={self._plan_period_s * 1000.0:.1f} ms, backend={getattr(result, 'backend', 'unknown')})."
                )
                self._last_slow_plan_log = now_wall
        elif (not getattr(result, 'optimizer_success', True)) and (now_wall - self._last_slow_plan_log > 2.0):
            self.get_logger().warn(
                f"Optimizer reported non-success status={getattr(result, 'optimizer_status', 0)} "
                f"message='{getattr(result, 'optimizer_message', '')}'. "
                "Using best available controls from solver."
            )
            self._last_slow_plan_log = now_wall

        if self.debug_runtime and (now_wall - self._last_runtime_log) > self.debug_log_period_s:
            pixel_age = None
            if pixel_stamp_ref is not None:
                try:
                    pixel_age = (self.get_clock().now() - Time.from_msg(pixel_stamp_ref)).nanoseconds * 1e-9
                except Exception:
                    pixel_age = None
            self.get_logger().info(
                "Plan debug: "
                f"backend={getattr(result, 'backend', 'unknown')}, "
                f"success={getattr(result, 'optimizer_success', False)}, "
                f"status={getattr(result, 'optimizer_status', 0)}, "
                f"nit={getattr(result, 'optimizer_nit', 0)}, "
                f"nfev={getattr(result, 'optimizer_nfev', 0)}, "
                f"fallback={getattr(result, 'used_fallback', False)}, "
                f"plan_ms={plan_elapsed_ms:.1f}, solve_ms={solve_elapsed_ms:.1f}, "
                f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
                f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), "
                f"frames=({state_frame or 'n/a'}->{goal_frame or 'n/a'}), "
                f"u0=({result.controls[0, 0]:.3f},{result.controls[0, 1]:.3f}), "
                f"J={result.total_cost:.3f}, "
                f"pixel_age={pixel_age if pixel_age is not None else 'n/a'}"
            )
            self._last_runtime_log = now_wall
