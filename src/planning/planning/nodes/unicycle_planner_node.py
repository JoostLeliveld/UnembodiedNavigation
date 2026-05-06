"""Thin ROS 2 wrapper around unicycle planners."""

import math
import time
import threading
import traceback
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray, String

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from planning.core.efe_utils import wrap_angle


class UnicyclePlannerNode(Node):
    """Base class for EFE/MPC planners using unicycle dynamics."""

    NODE_NAME = 'planner'
    PLANNER_CLASS = None
    PARAM_DEFAULT_OVERRIDES = {}

    def __init__(self):
        super().__init__(self.NODE_NAME)

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

        # Standalone defaults for the visibility-aware thesis planner node.

        # Planner params
        _declare_if_not('plan_rate', 1.0)
        _declare_if_not('belief_publish_rate', 10.0)
        _declare_if_not('horizon', 10)
        _declare_if_not('dt', 0.2)
        _declare_if_not('v_min', 0.0)
        _declare_if_not('v_max', 0.22)
        _declare_if_not('w_min', -1.0)
        _declare_if_not('w_max', 1.0)
        _declare_if_not('control_weight', 0.0)
        _declare_if_not('seed', 0)

        # Process/observation noise
        _declare_if_not('process_noise_xy', 0.01)
        _declare_if_not('process_noise_theta', 0.02)
        _declare_if_not('obs_noise_uv', 2.0)

        # Goal observation covariance
        _declare_if_not('goal_sigma_uv', 2.0)

        # EFE weights
        _declare_if_not('risk_weight_obs', 1.0)
        _declare_if_not('ambiguity_weight', 1.0)
        _declare_if_not('approx_method', 'ET1')
        _declare_if_not('use_obs_risk', True)
        _declare_if_not('use_ambiguity', True)
        _declare_if_not('use_visibility_model', False)
        _declare_if_not('visibility_target_height_m', 0.0)
        _declare_if_not('visibility_geometry_json', '')
        _declare_if_not('collision_geometry_json', '')
        _declare_if_not('r_visible_uv', 2.5)
        _declare_if_not('r_miss_uv', 120.0)
        _declare_if_not('visibility_sigma_kappa', 1.0)
        _declare_if_not('goal_prior_u_std_start', 80.0)
        _declare_if_not('goal_prior_v_std_start', 80.0)
        _declare_if_not('goal_prior_u_std_final', 18.0)
        _declare_if_not('goal_prior_v_std_final', 18.0)
        _declare_if_not('goal_tightening_power', 0.45)
        _declare_if_not('goal_progress_n_steps', 90)
        _declare_if_not('observation_risk_scale', 1.25)
        _declare_if_not('ambiguity_term_scale', 1.00)
        _declare_if_not('discount_gamma', 0.98)
        _declare_if_not('use_nogo_cost', False)
        _declare_if_not('nogo_penalty_type', 'softplus')
        _declare_if_not('nogo_weight', 0.0)
        _declare_if_not('nogo_safe_distance', 0.35)
        _declare_if_not('nogo_gaussian_sigma', 0.25)
        _declare_if_not('nogo_softplus_scale', 0.08)
        _declare_if_not('nogo_logbarrier_scale', 0.25)
        _declare_if_not('nogo_logbarrier_eps', 1e-3)
        _declare_if_not('visibility_artifact_path', '')
        _declare_if_not('robot_collision_radius_m', 0.125)
        _declare_if_not('min_terminal_goal_progress_m', 0.0)
        _declare_if_not('invalid_rollout_barrier_cost', 1e6)

        # Optimizer params
        _declare_if_not('optimizer_maxiter', 50)
        _declare_if_not('optimizer_maxfun', 500)
        _declare_if_not('optimizer_ftol', 1e-6)
        _declare_if_not('optimizer_gtol', 1e-4)
        _declare_if_not('optimizer_warm_start', True)

        # Pixel correction params
        _declare_if_not('use_pixel_correction', False)
        _declare_if_not('pixel_topic', '/perception/pixel_pose')
        _declare_if_not('cmd_topic', '/cmd_vel')
        _declare_if_not('pixel_timeout_s', 0.5)
        _declare_if_not('pixel_correction_min_interval_s', 0.0)
        _declare_if_not('pixel_correction_approx', 'AUTO')
        _declare_if_not('skip_stale_pixel_correction', True)
        _declare_if_not('use_pixel_heading_correction', True)
        _declare_if_not('use_odom_heading_correction', True)
        _declare_if_not('odom_heading_correction_mode', 'kalman')
        _declare_if_not('odom_heading_timeout_s', 0.75)
        _declare_if_not('odom_heading_sigma_rad', 0.08)
        _declare_if_not('odom_yaw_offset_rad', 0.0)
        _declare_if_not('heading_pixel_noise_sigma', 0.0)
        _declare_if_not('pixel_heading_noise_floor_rad', 0.01)
        _declare_if_not('clamp_pixel_uv_theta_without_yaw', False)
        _declare_if_not('min_state_cov', 1e-6)
        _declare_if_not('debug_runtime', False)
        _declare_if_not('debug_log_period_s', 1.0)
        _declare_if_not('slow_plan_factor', 1.0)
        _declare_if_not('slow_correction_ms', 20.0)

        # Camera model params (must match sim)
        _declare_if_not('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_not('look_at', [1.5, 1.5, 0.0])
        _declare_if_not('img_width', 1280)
        _declare_if_not('img_height', 720)
        _declare_if_not('fov_h_rad', 1.5708)

        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.belief_publish_rate = float(self.get_parameter('belief_publish_rate').value)
        self.horizon = int(self.get_parameter('horizon').value)
        self.dt = float(self.get_parameter('dt').value)
        self.v_min = float(self.get_parameter('v_min').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_min = float(self.get_parameter('w_min').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.control_weight = float(self.get_parameter('control_weight').value)
        self.seed = int(self.get_parameter('seed').value)

        self.process_noise_xy = float(self.get_parameter('process_noise_xy').value)
        self.process_noise_theta = float(self.get_parameter('process_noise_theta').value)
        self.obs_noise_uv = float(self.get_parameter('obs_noise_uv').value)

        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)

        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.approx_method = str(self.get_parameter('approx_method').value).upper()
        if self.approx_method not in ('ET1', 'ET2'):
            raise RuntimeError("approx_method must be one of: ET1, ET2")
        self.planner_path_summary = (
            f'approx_method={self.approx_method}, solver=casadi_symbolic_efe'
        )
        self.use_obs_risk = _as_bool(self.get_parameter('use_obs_risk').value)
        self.use_ambiguity = _as_bool(self.get_parameter('use_ambiguity').value)
        self.use_visibility_model = _as_bool(self.get_parameter('use_visibility_model').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.visibility_geometry_json = str(self.get_parameter('visibility_geometry_json').value)
        self.collision_geometry_json = str(self.get_parameter('collision_geometry_json').value)
        self.r_visible_uv = float(self.get_parameter('r_visible_uv').value)
        self.r_miss_uv = float(self.get_parameter('r_miss_uv').value)
        self.visibility_sigma_kappa = float(self.get_parameter('visibility_sigma_kappa').value)
        self.goal_prior_u_std_start = float(self.get_parameter('goal_prior_u_std_start').value)
        self.goal_prior_v_std_start = float(self.get_parameter('goal_prior_v_std_start').value)
        self.goal_prior_u_std_final = float(self.get_parameter('goal_prior_u_std_final').value)
        self.goal_prior_v_std_final = float(self.get_parameter('goal_prior_v_std_final').value)
        self.goal_tightening_power = float(self.get_parameter('goal_tightening_power').value)
        self.goal_progress_n_steps = int(self.get_parameter('goal_progress_n_steps').value)
        self.observation_risk_scale = float(self.get_parameter('observation_risk_scale').value)
        self.ambiguity_term_scale = float(self.get_parameter('ambiguity_term_scale').value)
        self.discount_gamma = float(self.get_parameter('discount_gamma').value)
        self.use_nogo_cost = _as_bool(self.get_parameter('use_nogo_cost').value)
        self.nogo_penalty_type = str(self.get_parameter('nogo_penalty_type').value).strip().lower()
        self.nogo_weight = float(self.get_parameter('nogo_weight').value)
        self.nogo_safe_distance = float(self.get_parameter('nogo_safe_distance').value)
        self.nogo_gaussian_sigma = float(self.get_parameter('nogo_gaussian_sigma').value)
        self.nogo_softplus_scale = float(self.get_parameter('nogo_softplus_scale').value)
        self.nogo_logbarrier_scale = float(self.get_parameter('nogo_logbarrier_scale').value)
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value).strip()
        self.robot_collision_radius_m = float(self.get_parameter('robot_collision_radius_m').value)
        self.min_terminal_goal_progress_m = float(self.get_parameter('min_terminal_goal_progress_m').value)
        self.invalid_rollout_barrier_cost = float(self.get_parameter('invalid_rollout_barrier_cost').value)

        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_maxfun = int(self.get_parameter('optimizer_maxfun').value)
        self.optimizer_ftol = float(self.get_parameter('optimizer_ftol').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = _as_bool(self.get_parameter('optimizer_warm_start').value)

        self.use_pixel_correction = _as_bool(self.get_parameter('use_pixel_correction').value)
        self.pixel_topic = self.get_parameter('pixel_topic').value
        self.cmd_topic = str(self.get_parameter('cmd_topic').value).strip() or '/cmd_vel'
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.pixel_correction_min_interval_s = float(
            self.get_parameter('pixel_correction_min_interval_s').value
        )
        self.pixel_correction_approx = str(
            self.get_parameter('pixel_correction_approx').value
        ).strip().upper()
        if self.pixel_correction_approx not in ('AUTO', 'ET1', 'ET2', 'UT'):
            raise RuntimeError("pixel_correction_approx must be one of: AUTO, ET1, ET2, UT")
        self.skip_stale_pixel_correction = _as_bool(
            self.get_parameter('skip_stale_pixel_correction').value
        )
        self.use_pixel_heading_correction = _as_bool(
            self.get_parameter('use_pixel_heading_correction').value
        )
        self.use_odom_heading_correction = _as_bool(
            self.get_parameter('use_odom_heading_correction').value
        )
        self.odom_heading_correction_mode = str(
            self.get_parameter('odom_heading_correction_mode').value
        ).strip().lower()
        if self.odom_heading_correction_mode not in ('kalman', 'overwrite'):
            raise RuntimeError("odom_heading_correction_mode must be one of: kalman, overwrite")
        self.odom_heading_timeout_s = float(self.get_parameter('odom_heading_timeout_s').value)
        self.odom_heading_sigma_rad = float(self.get_parameter('odom_heading_sigma_rad').value)
        self.odom_yaw_offset_rad = float(self.get_parameter('odom_yaw_offset_rad').value)
        self.heading_pixel_noise_sigma = float(
            self.get_parameter('heading_pixel_noise_sigma').value
        )
        self.pixel_heading_noise_floor_rad = float(
            self.get_parameter('pixel_heading_noise_floor_rad').value
        )
        self.clamp_pixel_uv_theta_without_yaw = _as_bool(
            self.get_parameter('clamp_pixel_uv_theta_without_yaw').value
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
        warm_start_shift_steps = max(
            1,
            int(round((1.0 / max(self.plan_rate, 0.1)) / max(self.dt, 1e-3))),
        )

        self.planner = self.PLANNER_CLASS(
            horizon=self.horizon,
            dt=self.dt,
            v_min=self.v_min,
            v_max=self.v_max,
            w_min=self.w_min,
            w_max=self.w_max,
            control_weight=self.control_weight,
            process_noise_xy=self.process_noise_xy,
            process_noise_theta=self.process_noise_theta,
            obs_noise_uv=self.obs_noise_uv,
            goal_sigma_uv=self.goal_sigma_uv,
            risk_weight_obs=self.risk_weight_obs,
            ambiguity_weight=self.ambiguity_weight,
            optimizer_maxiter=self.optimizer_maxiter,
            optimizer_maxfun=self.optimizer_maxfun,
            optimizer_ftol=self.optimizer_ftol,
            optimizer_gtol=self.optimizer_gtol,
            optimizer_warm_start=self.optimizer_warm_start,
            optimizer_warm_start_shift_steps=warm_start_shift_steps,
            approx_method=self.approx_method,
            use_obs_risk=self.use_obs_risk,
            use_ambiguity=self.use_ambiguity,
            seed=self.seed,
            camera_params=camera_params,
            use_visibility_model=self.use_visibility_model,
            visibility_target_height_m=self.visibility_target_height_m,
            visibility_geometry_json=self.visibility_geometry_json,
            collision_geometry_json=self.collision_geometry_json,
            visibility_artifact_path=self.visibility_artifact_path,
            r_visible_uv=self.r_visible_uv,
            r_miss_uv=self.r_miss_uv,
            visibility_sigma_kappa=self.visibility_sigma_kappa,
            goal_prior_u_std_start=self.goal_prior_u_std_start,
            goal_prior_v_std_start=self.goal_prior_v_std_start,
            goal_prior_u_std_final=self.goal_prior_u_std_final,
            goal_prior_v_std_final=self.goal_prior_v_std_final,
            goal_tightening_power=self.goal_tightening_power,
            goal_progress_n_steps=self.goal_progress_n_steps,
            observation_risk_scale=self.observation_risk_scale,
            ambiguity_term_scale=self.ambiguity_term_scale,
            discount_gamma=self.discount_gamma,
            use_nogo_cost=self.use_nogo_cost,
            nogo_penalty_type=self.nogo_penalty_type,
            nogo_weight=self.nogo_weight,
            nogo_safe_distance=self.nogo_safe_distance,
            nogo_gaussian_sigma=self.nogo_gaussian_sigma,
            nogo_softplus_scale=self.nogo_softplus_scale,
            nogo_logbarrier_scale=self.nogo_logbarrier_scale,
            nogo_logbarrier_eps=self.nogo_logbarrier_eps,
            robot_collision_radius_m=self.robot_collision_radius_m,
            min_terminal_goal_progress_m=self.min_terminal_goal_progress_m,
            invalid_rollout_barrier_cost=self.invalid_rollout_barrier_cost,
            runtime_debug=self.debug_runtime,
        )
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
        goal_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos,
            callback_group=self._io_group
        )
        self.pixel_sub = self.create_subscription(
            PoseStamped, self.pixel_topic, self._pixel_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.detection_diag_sub = self.create_subscription(
            Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._detection_diag_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_topic, self._cmd_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )

        # Publishers
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(Path, '/plan', qos_profile=path_qos)
        self.plan_preview_pub = self.create_publisher(Path, '/plan_preview', qos_profile=path_qos)
        self.planner_belief_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/planner_belief', qos_profile=path_qos
        )
        self.metrics_pub = self.create_publisher(Float64MultiArray, '/efe/metrics', 10)
        self.planner_diag_pub = self.create_publisher(Float64MultiArray, '/planner/diagnostics', 10)
        self.planner_diag_text_pub = self.create_publisher(String, '/planner/diagnostics_text', 10)
        self.pixel_correction_diag_pub = self.create_publisher(
            Float64MultiArray, '/planner/pixel_correction_diagnostics', 10
        )

        # State
        self.state_msg = None
        self.goal_msg = None
        self._goal_received_logged = False
        self.pixel_meas = None
        self.pixel_stamp = None
        self.pixel_yaw_meas = None
        self.pixel_heading_sigma = math.nan
        self.odom_yaw_meas = None
        self.odom_stamp = None
        self._latest_detection_diag = None
        self._last_correction_log = 0.0
        self._last_correction_stamp = None
        self._last_stale_log = 0.0
        self._last_shape_mismatch_log = 0.0
        self._last_runtime_log = 0.0
        self._last_plan_entry_log = 0.0
        self._last_plan_return_log = 0.0
        self._last_slow_plan_log = 0.0
        self._last_slow_correction_log = 0.0
        self._fatal_stop_triggered = False
        self._goal_signature = None
        self._goal_progress_start_dist_m = None
        self.belief_m = None
        self.belief_S = None
        self.belief_stamp = None
        self.last_cmd = np.array([0.0, 0.0], dtype=float)
        self._latest_measurement_available = False
        self._latest_belief_age_s = math.nan

        self._plan_period_s = 1.0 / max(self.plan_rate, 0.1)
        self.create_timer(self._plan_period_s, self._plan_once, callback_group=self._plan_group)
        if self.belief_publish_rate > 0.0:
            self._belief_publish_period_s = 1.0 / max(self.belief_publish_rate, 0.1)
            self.create_timer(
                self._belief_publish_period_s,
                self._belief_publish_tick,
                callback_group=self._io_group,
            )
        self._pixel_correction_timer = None
        if self.use_pixel_correction and self.pixel_correction_min_interval_s > 0.0:
            correction_period = max(self.pixel_correction_min_interval_s, 0.02)
            self._pixel_correction_timer = self.create_timer(
                correction_period, self._pixel_correction_timer_cb, callback_group=self._io_group
            )
        self.get_logger().info(f'Active planner path: {self.planner_path_summary}')
        self.get_logger().info(
            f"{self.NODE_NAME} started "
            f"({self.planner_path_summary}, "
            f"use_obs_risk={self.use_obs_risk}, use_ambiguity={self.use_ambiguity}, "
            f"goal_progress_n_steps={self.goal_progress_n_steps}, "
            f"use_visibility_model={self.use_visibility_model}, "
            f"use_nogo_cost={self.use_nogo_cost}, nogo_penalty_type={self.nogo_penalty_type}, "
            f"use_pixel_correction={self.use_pixel_correction}, "
            f"cmd_topic={self.cmd_topic}, "
            f"pixel_correction_approx={self.pixel_correction_approx}, "
            f"use_pixel_heading_correction={self.use_pixel_heading_correction}, "
            f"use_odom_heading_correction={self.use_odom_heading_correction}, "
            f"odom_heading_correction_mode={self.odom_heading_correction_mode}, "
            f"clamp_pixel_uv_theta_without_yaw={self.clamp_pixel_uv_theta_without_yaw}, "
            f"debug_runtime={self.debug_runtime})"
        )

    def _publish_safe_stop_command(self):
        """Hook for agent mode; planner-only nodes can ignore."""
        return

    def _fatal_experiment_stop(self, reason: str, exc: Exception | None = None):
        if self._fatal_stop_triggered:
            return
        self._fatal_stop_triggered = True

        try:
            self._publish_safe_stop_command()
        except RuntimeError:
            pass

        detail = reason
        if exc is not None:
            detail = f"{reason}: {type(exc).__name__}: {exc}"
        self.get_logger().error(
            "Fatal experiment integrity failure. Publishing zero command and terminating node. "
            f"Reason: {detail}"
        )
        if exc is not None:
            try:
                tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self.get_logger().error(tb.rstrip())
            except (TypeError, ValueError):
                pass

        # Stop this process so runs fail fast instead of continuing with invalid behavior.
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass
        raise RuntimeError(detail) from exc

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        with self._data_lock:
            self.state_msg = msg

    def _update_goal_progress_origin(self, msg: PoseStamped):
        signature = (
            (msg.header.frame_id or '').strip() or 'map_bev',
            float(msg.pose.position.x),
            float(msg.pose.position.y),
        )
        with self._data_lock:
            previous = self._goal_signature
            changed = (
                previous is None
                or signature[0] != previous[0]
                or abs(signature[1] - previous[1]) > 1e-9
                or abs(signature[2] - previous[2]) > 1e-9
            )
            if changed:
                self._goal_signature = signature
                self._goal_progress_start_dist_m = None

    def _current_goal_progress_index(self, m0, goal_xy) -> float:
        current_dist = float(math.hypot(float(m0[0]) - float(goal_xy[0]), float(m0[1]) - float(goal_xy[1])))
        with self._data_lock:
            start_dist = self._goal_progress_start_dist_m
            if start_dist is None or (not math.isfinite(start_dist)) or start_dist <= 0.0:
                self._goal_progress_start_dist_m = current_dist
                start_dist = current_dist
        if (not math.isfinite(start_dist)) or start_dist <= 1e-9:
            return 0.0
        progress_fraction = max(min((start_dist - current_dist) / start_dist, 1.0), 0.0)
        return progress_fraction * float(max(self.goal_progress_n_steps, 1))

    def _goal_cb(self, msg: PoseStamped):
        with self._data_lock:
            self.goal_msg = msg
            first_goal = not self._goal_received_logged
            if first_goal:
                self._goal_received_logged = True
        self._update_goal_progress_origin(msg)
        if first_goal:
            self.get_logger().info(
                f"Received goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
                f"frame='{msg.header.frame_id or 'map_bev'}'"
            )

    def _cmd_cb(self, msg: Twist):
        with self._data_lock:
            self.last_cmd = np.array([msg.linear.x, msg.angular.z], dtype=float)

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _odom_cb(self, msg: Odometry):
        yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        with self._data_lock:
            self.odom_yaw_meas = wrap_angle(float(yaw + self.odom_yaw_offset_rad))
            self.odom_stamp = msg.header.stamp

    def _fresh_odom_heading_locked(self, ref_stamp) -> tuple[float | None, float]:
        if self.odom_yaw_meas is None or self.odom_stamp is None:
            return None, math.nan
        try:
            age = abs(self._stamp_to_float(ref_stamp) - self._stamp_to_float(self.odom_stamp))
        except (AttributeError, TypeError, ValueError):
            return None, math.nan
        if self.odom_heading_timeout_s > 0.0 and age > self.odom_heading_timeout_s:
            return None, age
        return float(self.odom_yaw_meas), float(age)

    def _heading_sigma_from_diag(self, diag) -> float:
        sigma_floor = float(max(self.pixel_heading_noise_floor_rad, 1e-6))
        if not diag:
            return sigma_floor
        diag_stamp = float(diag.get('stamp', math.nan))
        if not math.isfinite(diag_stamp):
            return sigma_floor
        sep = float(diag.get('separation_px', math.nan))
        if not math.isfinite(sep) or sep <= 1e-6:
            return sigma_floor
        sigma_sep = math.sqrt(2.0) * max(float(self.heading_pixel_noise_sigma), 1e-6) / max(sep, 1.0)
        return float(max(sigma_floor, sigma_sep))

    @staticmethod
    def _fuse_heading_measurement(m, S, yaw_meas: float, yaw_sigma: float):
        m = np.asarray(m, dtype=float).copy()
        S = np.asarray(S, dtype=float).copy()
        if (
            m.shape[0] < 3
            or S.shape[0] < 3
            or S.shape[1] < 3
            or not math.isfinite(float(yaw_meas))
            or not math.isfinite(float(yaw_sigma))
            or float(yaw_sigma) <= 0.0
        ):
            return m, S, False, math.nan, math.nan
        P_theta = S[:, 2].copy()
        innov_theta = wrap_angle(float(yaw_meas) - float(m[2]))
        S_theta = float(S[2, 2] + float(yaw_sigma) ** 2)
        if S_theta <= 1e-12:
            return m, S, False, innov_theta, math.nan
        K_theta = P_theta / S_theta
        m = m + K_theta * innov_theta
        m[2] = wrap_angle(m[2])
        S = S - np.outer(P_theta, P_theta) / S_theta
        S = (S + S.T) / 2.0
        return m, S, True, innov_theta, float(K_theta[2]) if K_theta.size >= 3 else math.nan

    @staticmethod
    def _overwrite_heading_measurement(m, S, yaw_meas: float, yaw_sigma: float):
        m = np.asarray(m, dtype=float).copy()
        S = np.asarray(S, dtype=float).copy()
        if (
            m.shape[0] < 3
            or S.shape[0] < 3
            or S.shape[1] < 3
            or not math.isfinite(float(yaw_meas))
        ):
            return m, S, False, math.nan, math.nan
        innov_theta = wrap_angle(float(yaw_meas) - float(m[2]))
        m[2] = wrap_angle(float(yaw_meas))
        if math.isfinite(float(yaw_sigma)) and float(yaw_sigma) > 0.0:
            S[2, :] = 0.0
            S[:, 2] = 0.0
            S[2, 2] = float(yaw_sigma) ** 2
        S = (S + S.T) / 2.0
        return m, S, True, innov_theta, 1.0

    def _apply_heading_measurement(self, m, S, yaw_meas: float, yaw_sigma: float, *, source_code: float):
        if source_code == 2.0 and self.odom_heading_correction_mode == 'overwrite':
            return self._overwrite_heading_measurement(m, S, yaw_meas, yaw_sigma)
        return self._fuse_heading_measurement(m, S, yaw_meas, yaw_sigma)

    @staticmethod
    def _stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _detection_diag_cb(self, msg: Float64MultiArray):
        try:
            diag = diagnostics_from_message(msg)
        except (KeyError, TypeError, ValueError):
            return
        with self._data_lock:
            self._latest_detection_diag = diag

    def _state_msg_to_belief(self, state_ref: PoseWithCovarianceStamped):
        """Convert the external state estimate into planner belief coordinates."""
        q = state_ref.pose.pose.orientation
        theta = self._yaw_from_quaternion(q)
        m = np.array([
            state_ref.pose.pose.position.x,
            state_ref.pose.pose.position.y,
            theta,
        ], dtype=float)

        cov = state_ref.pose.covariance
        S = np.diag([
            cov[0] if len(cov) > 0 else 1e-6,
            cov[7] if len(cov) > 7 else 1e-6,
            cov[35] if len(cov) > 35 else 1e-6,
        ]).astype(float)
        return m, self._regularize_state_covariance(S)

    def _regularize_state_covariance(self, S):
        """Keep planner belief covariance positive enough for stable updates."""
        S = np.asarray(S, dtype=float).copy()
        if self.min_state_cov > 0.0:
            for i in range(min(3, S.shape[0])):
                if S[i, i] < self.min_state_cov:
                    S[i, i] = self.min_state_cov
        return (S + S.T) / 2.0

    def _init_belief_from_state(self):
        with self._data_lock:
            if self.state_msg is None:
                return False
            self.belief_m, self.belief_S = self._state_msg_to_belief(self.state_msg)
            self.belief_stamp = self.state_msg.header.stamp
            return True

    def _matching_detection_diag_locked(self, stamp_msg):
        """Return the diagnostics message that belongs to a pixel observation."""
        if self._latest_detection_diag is None:
            return None
        diag_ref = dict(self._latest_detection_diag)
        try:
            stamp_s = self._stamp_to_float(stamp_msg)
            diag_stamp = float(diag_ref.get('stamp', math.nan))
        except (AttributeError, TypeError, ValueError):
            return None
        if (not math.isfinite(diag_stamp)) or abs(diag_stamp - stamp_s) > 1e-3:
            return None
        return diag_ref

    def _pixel_yaw_measurement_from_msg(self, msg: PoseStamped, diag_ref):
        """Extract visual yaw only when detector diagnostics explicitly support it."""
        if not (
            diag_ref is not None
            and bool(diag_ref.get('detected', False))
            and math.isfinite(float(diag_ref.get('yaw_est', math.nan)))
        ):
            return None, math.nan
        yaw_meas = self._yaw_from_quaternion(msg.pose.orientation)
        return float(yaw_meas), self._heading_sigma_from_diag(diag_ref)

    def _pixel_cb(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        with self._data_lock:
            diag_ref = self._matching_detection_diag_locked(msg.header.stamp)
            yaw_meas, yaw_sigma = self._pixel_yaw_measurement_from_msg(msg, diag_ref)
            self.pixel_meas = np.array([u, v], dtype=float)
            self.pixel_stamp = msg.header.stamp
            self.pixel_yaw_meas = yaw_meas
            self.pixel_heading_sigma = yaw_sigma

        if not self.use_pixel_correction:
            return
        if self.pixel_correction_min_interval_s > 0.0:
            return
        self._apply_pixel_correction(msg.header.stamp, source='callback')

    def _pixel_correction_timer_cb(self):
        if not self.use_pixel_correction or self.pixel_correction_min_interval_s <= 0.0:
            return
        with self._data_lock:
            stamp_ref = self.pixel_stamp
        if stamp_ref is None:
            return
        self._apply_pixel_correction(stamp_ref, source='timer')

    def _stamp_age_s(self, stamp_msg) -> float:
        try:
            return (self.get_clock().now() - Time.from_msg(stamp_msg)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _pixel_correction_age_is_invalid(self, age: float) -> bool:
        future_tolerance_s = max(float(self.pixel_timeout_s), 0.25)
        return bool(
            self.skip_stale_pixel_correction
            and (age > self.pixel_timeout_s or age < -future_tolerance_s)
        )

    def _warn_stale_pixel_once(self, message: str):
        now_wall = time.monotonic()
        if now_wall - self._last_stale_log > 2.0:
            self.get_logger().warn(message)
            self._last_stale_log = now_wall

    def _pixel_correction_is_throttled(self, stamp_msg) -> bool:
        if self.pixel_correction_min_interval_s <= 0.0:
            return False
        with self._data_lock:
            last_correction_stamp = self._last_correction_stamp
        if last_correction_stamp is None:
            return False
        try:
            dt_since_correction = (
                Time.from_msg(stamp_msg) - Time.from_msg(last_correction_stamp)
            ).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(0.0 <= dt_since_correction < self.pixel_correction_min_interval_s)

    def _pixel_correction_dt_s(self, stamp_msg) -> float | None:
        try:
            now = Time.from_msg(stamp_msg)
            with self._data_lock:
                stamp_ref = self.belief_stamp
            last = Time.from_msg(stamp_ref) if stamp_ref is not None else None
            dt_s = (now - last).nanoseconds * 1e-9 if last is not None else self.dt
            if dt_s <= 0.0:
                dt_s = self.dt
        except (AttributeError, TypeError, ValueError):
            dt_s = self.dt

        max_dt_s = max(2.0 * float(self.pixel_timeout_s), 4.0 * float(self.dt), 0.5)
        if dt_s > max_dt_s:
            self._warn_stale_pixel_once(
                f"Skipping pixel correction with implausible dt={dt_s:.2f}s "
                f"(max {max_dt_s:.2f}s)"
            )
            return None
        return float(dt_s)

    def _select_heading_measurement_locked(self, stamp_msg):
        """Prefer visual yaw when available, otherwise optionally anchor to odom yaw."""
        yaw_meas = self.pixel_yaw_meas
        yaw_sigma = float(self.pixel_heading_sigma)
        yaw_source = 1.0 if yaw_meas is not None and math.isfinite(float(yaw_meas)) else 0.0
        if yaw_source <= 0.0 and self.use_odom_heading_correction:
            odom_yaw, _odom_age = self._fresh_odom_heading_locked(stamp_msg)
            if odom_yaw is not None:
                yaw_meas = float(odom_yaw)
                yaw_sigma = float(max(
                    self.odom_heading_sigma_rad,
                    self.pixel_heading_noise_floor_rad,
                    1e-6,
                ))
                yaw_source = 2.0
        return yaw_meas, yaw_sigma, yaw_source

    def _snapshot_pixel_correction_inputs(self, stamp_msg):
        with self._data_lock:
            belief_m = None if self.belief_m is None else self.belief_m.copy()
            belief_S = None if self.belief_S is None else self.belief_S.copy()
            v_cmd, w_cmd = float(self.last_cmd[0]), float(self.last_cmd[1])
            meas = None if self.pixel_meas is None else self.pixel_meas.copy()
            yaw_meas, yaw_sigma, yaw_source = self._select_heading_measurement_locked(stamp_msg)
        if belief_m is None or belief_S is None or meas is None:
            return None
        return {
            'belief_m': belief_m,
            'belief_S': belief_S,
            'cmd': np.array([v_cmd, w_cmd], dtype=float),
            'meas': meas,
            'yaw_meas': yaw_meas,
            'yaw_sigma': yaw_sigma,
            'yaw_source': yaw_source,
        }

    def _log_pixel_shape_error_once(self, message: str):
        now_wall = time.monotonic()
        if now_wall - self._last_shape_mismatch_log > 2.0:
            self.get_logger().error(message)
            self._last_shape_mismatch_log = now_wall

    def _compute_pixel_uv_update(self, m_pred, S_eff, meas, R_eff, gain_scale, *, corr_method):
        mu_y, Sigma_y, Gamma = self.planner.approx_observation(
            m_pred, S_eff, method=corr_method, R_override=R_eff
        )
        mu_y = np.asarray(mu_y, dtype=float).reshape(-1)
        meas = np.asarray(meas, dtype=float).reshape(-1)
        if meas.size != mu_y.size:
            self._log_pixel_shape_error_once(
                "Pixel correction shape mismatch: "
                f"meas_dim={meas.size}, pred_dim={mu_y.size}. "
                "Skipping correction for this message."
            )
            return None

        Sigma_y = np.asarray(Sigma_y, dtype=float)
        Gamma = np.asarray(Gamma, dtype=float)
        if Sigma_y.shape != (meas.size, meas.size) or Gamma.shape[1] != meas.size:
            self._log_pixel_shape_error_once(
                "Pixel correction covariance shape mismatch: "
                f"Sigma_y={Sigma_y.shape}, Gamma={Gamma.shape}, meas_dim={meas.size}. "
                "Skipping correction for this message."
            )
            return None

        innov = meas - mu_y
        if innov.size >= 3:
            innov[2] = wrap_angle(innov[2])
        Sigma_y = (Sigma_y + Sigma_y.T) / 2.0
        Sigma_inv = np.linalg.pinv(Sigma_y)
        K = Gamma @ Sigma_inv
        next_m = m_pred + gain_scale * (K @ innov)
        next_m[2] = wrap_angle(next_m[2])
        next_S = S_eff - gain_scale * (Gamma @ Sigma_inv @ Gamma.T)
        next_S = (next_S + next_S.T) / 2.0
        return {
            'next_m': next_m,
            'next_S': next_S,
            'innov': innov,
            'mu_y': mu_y,
        }

    def _apply_yaw_anchor_after_pixel_update(
        self,
        next_m,
        next_S,
        m_pred,
        S_pred,
        yaw_meas,
        yaw_sigma,
        yaw_source,
    ):
        """Keep theta correction explicit: visual yaw or odom yaw, never hidden in u/v."""
        theta_update_from_uv_rad = float(wrap_angle(float(next_m[2]) - float(m_pred[2])))
        if self.clamp_pixel_uv_theta_without_yaw and yaw_source != 1.0:
            next_m[2] = float(m_pred[2])
            theta_update_from_uv_rad = 0.0
            if next_S.shape[0] >= 3:
                next_S[2, :] = S_pred[2, :]
                next_S[:, 2] = S_pred[:, 2]
                next_S = (next_S + next_S.T) / 2.0

        yaw_correction_applied = False
        innov_theta = math.nan
        k_theta_theta = math.nan
        if (
            (
                (yaw_source == 1.0 and self.use_pixel_heading_correction)
                or (yaw_source == 2.0 and self.use_odom_heading_correction)
            )
            and yaw_meas is not None
            and math.isfinite(float(yaw_meas))
            and math.isfinite(yaw_sigma)
            and yaw_sigma > 0.0
            and next_S.shape[0] >= 3
        ):
            next_m, next_S, yaw_correction_applied, innov_theta, k_theta_theta = self._apply_heading_measurement(
                next_m,
                next_S,
                float(yaw_meas),
                float(yaw_sigma),
                source_code=float(yaw_source),
            )

        return {
            'next_m': next_m,
            'next_S': next_S,
            'theta_update_from_uv_rad': theta_update_from_uv_rad,
            'yaw_correction_applied': yaw_correction_applied,
            'innov_theta': innov_theta,
            'k_theta_theta': k_theta_theta,
            'theta_update_total_rad': float(wrap_angle(float(next_m[2]) - float(m_pred[2]))),
        }

    def _publish_pixel_correction_diagnostics(
        self,
        *,
        stamp_msg,
        age,
        dt_s,
        p_vis,
        gain_scale,
        innov,
        xy_update_norm_m,
        yaw_info,
        m_pred,
        next_m,
        meas,
        mu_y,
        R_eff,
        yaw_meas,
        yaw_sigma,
        yaw_source,
    ):
        diag_msg = Float64MultiArray()
        r_eff = np.asarray(R_eff, dtype=float)
        diag_msg.data = [
            float(self._stamp_to_float(stamp_msg)),
            1.0,
            float(age),
            float(dt_s),
            float(p_vis),
            float(gain_scale),
            float(innov[0]) if innov.size > 0 else math.nan,
            float(innov[1]) if innov.size > 1 else math.nan,
            float(xy_update_norm_m),
            float(yaw_info['theta_update_from_uv_rad']),
            1.0 if yaw_info['yaw_correction_applied'] else 0.0,
            float(yaw_info['innov_theta']),
            float(yaw_info['k_theta_theta']),
            float(yaw_info['theta_update_total_rad']),
            float(m_pred[0]),
            float(m_pred[1]),
            float(m_pred[2]),
            float(next_m[0]),
            float(next_m[1]),
            float(next_m[2]),
            float(meas[0]) if meas.size > 0 else math.nan,
            float(meas[1]) if meas.size > 1 else math.nan,
            float(mu_y[0]) if mu_y.size > 0 else math.nan,
            float(mu_y[1]) if mu_y.size > 1 else math.nan,
            float(r_eff[0, 0]) if r_eff.ndim == 2 and r_eff.shape[0] > 0 and r_eff.shape[1] > 0 else math.nan,
            float(r_eff[1, 1]) if r_eff.ndim == 2 and r_eff.shape[0] > 1 and r_eff.shape[1] > 1 else math.nan,
            float(yaw_meas) if yaw_meas is not None and math.isfinite(float(yaw_meas)) else math.nan,
            float(yaw_sigma) if math.isfinite(float(yaw_sigma)) else math.nan,
            float(yaw_source),
        ]
        self.pixel_correction_diag_pub.publish(diag_msg)

    def _apply_pixel_correction(self, stamp_msg, *, source='callback'):
        cb_start = time.perf_counter()
        age = self._stamp_age_s(stamp_msg)
        if self._pixel_correction_age_is_invalid(age):
            self._warn_stale_pixel_once(
                f"Skipping time-inconsistent pixel measurement (age {age:.2f}s)"
            )
            return
        if self._pixel_correction_is_throttled(stamp_msg):
            return

        with self._data_lock:
            has_belief = self.belief_m is not None and self.belief_S is not None
        if not has_belief and not self._init_belief_from_state():
            return

        dt_s = self._pixel_correction_dt_s(stamp_msg)
        if dt_s is None:
            return

        snapshot = self._snapshot_pixel_correction_inputs(stamp_msg)
        if snapshot is None:
            return
        belief_m = snapshot['belief_m']
        belief_S = snapshot['belief_S']
        meas = snapshot['meas']
        yaw_meas = snapshot['yaw_meas']
        yaw_sigma = snapshot['yaw_sigma']
        yaw_meas_source = snapshot['yaw_source']

        m_pred, S_pred = self.planner.predict(
            belief_m, belief_S, snapshot['cmd'], dt=dt_s
        )
        p_vis, R_eff, S_eff, gain_scale = self.planner.observation_model_with_visibility(m_pred, S_pred)

        corr_method = self.approx_method if self.pixel_correction_approx == 'AUTO' else self.pixel_correction_approx
        uv_update = self._compute_pixel_uv_update(
            m_pred, S_eff, meas, R_eff, gain_scale, corr_method=corr_method
        )
        if uv_update is None:
            return

        next_m = uv_update['next_m']
        next_S = uv_update['next_S']
        innov = uv_update['innov']
        meas = np.asarray(meas, dtype=float).reshape(-1)
        mu_y = uv_update['mu_y']
        xy_update_norm_m = float(np.linalg.norm(np.asarray(next_m[:2] - m_pred[:2], dtype=float)))
        yaw_info = self._apply_yaw_anchor_after_pixel_update(
            next_m,
            next_S,
            m_pred,
            S_pred,
            yaw_meas,
            yaw_sigma,
            yaw_meas_source,
        )
        next_m = yaw_info['next_m']
        next_S = self._regularize_state_covariance(yaw_info['next_S'])
        with self._data_lock:
            self.belief_m = next_m
            self.belief_S = next_S
            self.belief_stamp = stamp_msg
            self._last_correction_stamp = stamp_msg

        self._publish_pixel_correction_diagnostics(
            stamp_msg=stamp_msg,
            age=age,
            dt_s=dt_s,
            p_vis=p_vis,
            gain_scale=gain_scale,
            innov=innov,
            xy_update_norm_m=xy_update_norm_m,
            yaw_info=yaw_info,
            m_pred=m_pred,
            next_m=next_m,
            meas=meas,
            mu_y=mu_y,
            R_eff=R_eff,
            yaw_meas=yaw_meas,
            yaw_sigma=yaw_sigma,
            yaw_source=yaw_meas_source,
        )

        now_wall = time.monotonic()
        if self.debug_runtime and (now_wall - self._last_correction_log > 2.0):
            self.get_logger().info(
                f"Applied pixel correction in {source} "
                f"(method={corr_method}, age={age:.3f}s, dt={dt_s:.3f}s, p_vis={p_vis:.3f})"
            )
            self._last_correction_log = now_wall

        cb_ms = max((time.perf_counter() - cb_start) * 1000.0, 0.0)
        if (
            self.debug_runtime
            and cb_ms > self.slow_correction_ms
            and (now_wall - self._last_slow_correction_log) > 2.0
        ):
            self.get_logger().warn(
                f"Slow pixel correction {source} ({cb_ms:.1f} ms) "
                f"using {corr_method}; this can cause stale-belief behavior."
            )
            self._last_slow_correction_log = now_wall

    def _belief_snapshot_for_planning(self):
        with self._data_lock:
            has_belief = self.belief_m is not None and self.belief_S is not None
        if not has_belief and not self._init_belief_from_state():
            return None
        with self._data_lock:
            return {
                'm': self.belief_m.copy(),
                'S': self.belief_S.copy(),
                'stamp': self.belief_stamp,
                'pixel_stamp': self.pixel_stamp,
                'last_cmd': self.last_cmd.copy(),
            }

    def _belief_age_for_planning(self, now_msg, stamp_ref) -> float | None:
        if stamp_ref is None:
            return 0.0
        try:
            raw_age_s = (Time.from_msg(now_msg) - Time.from_msg(stamp_ref)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if raw_age_s < -max(float(self.pixel_timeout_s), 0.25):
            self._warn_stale_pixel_once(
                f"Pixel belief stamp is in the future (age {raw_age_s:.2f}s); "
                "resetting belief from state."
            )
            return None
        return float(max(raw_age_s, 0.0))

    def _pixel_measurement_available_for_planning(self, now_msg, pixel_stamp_ref) -> bool:
        if pixel_stamp_ref is None:
            return False
        try:
            raw_measurement_age = (
                Time.from_msg(now_msg) - Time.from_msg(pixel_stamp_ref)
            ).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            raw_measurement_age = math.inf
        return bool(0.0 <= raw_measurement_age <= self.pixel_timeout_s)

    def _predict_belief_to_now(self, m0, S0, last_cmd, belief_age_s: float, now_msg):
        if belief_age_s <= 0.0:
            return m0, S0
        m0, S0 = self.planner.predict(
            m0, S0, np.asarray(last_cmd, dtype=float), dt=belief_age_s
        )
        with self._data_lock:
            self.belief_m = m0.copy()
            self.belief_S = S0.copy()
            self.belief_stamp = now_msg
        return m0, S0

    def _anchor_belief_yaw_to_odom_for_planning(self, m0, S0, now_msg):
        """Apply the explicit odom yaw correction used when visual yaw is unavailable."""
        if not self.use_odom_heading_correction:
            return m0, S0
        with self._data_lock:
            odom_yaw, _odom_age = self._fresh_odom_heading_locked(now_msg)
        if odom_yaw is None:
            return m0, S0
        m0, S0, applied, _innov_theta, _k_theta = self._apply_heading_measurement(
            m0,
            S0,
            float(odom_yaw),
            float(max(self.odom_heading_sigma_rad, self.pixel_heading_noise_floor_rad, 1e-6)),
            source_code=2.0,
        )
        if applied:
            with self._data_lock:
                self.belief_m = m0.copy()
                self.belief_S = S0.copy()
                self.belief_stamp = now_msg
        return m0, S0

    def _resolve_pixel_corrected_belief_for_planning(self, now_msg):
        snapshot = self._belief_snapshot_for_planning()
        if snapshot is None:
            return None, None, {}

        belief_age_s = self._belief_age_for_planning(now_msg, snapshot['stamp'])
        if belief_age_s is None:
            if not self._init_belief_from_state():
                return None, None, {}
            snapshot = self._belief_snapshot_for_planning()
            if snapshot is None:
                return None, None, {}
            belief_age_s = 0.0

        measurement_available = self._pixel_measurement_available_for_planning(
            now_msg, snapshot['pixel_stamp']
        )
        m0, S0 = self._predict_belief_to_now(
            snapshot['m'], snapshot['S'], snapshot['last_cmd'], belief_age_s, now_msg
        )
        m0, S0 = self._anchor_belief_yaw_to_odom_for_planning(m0, S0, now_msg)

        if belief_age_s > self.pixel_timeout_s:
            self._warn_stale_pixel_once(
                f"Pixel belief stale (age {belief_age_s:.2f}s); planning on prediction-only belief"
            )
        return m0, S0, {
            'measurement_available': bool(measurement_available),
            'belief_age_s': float(belief_age_s),
        }

    def _resolve_state_belief_for_planning(self):
        with self._data_lock:
            state_ref = self.state_msg
        if state_ref is None:
            return None, None, {}
        m0, S0 = self._state_msg_to_belief(state_ref)
        return m0, S0, {
            'measurement_available': False,
            'belief_age_s': 0.0,
        }

    def _resolve_belief_for_planning(self):
        now_msg = self.get_clock().now().to_msg()
        if self.use_pixel_correction:
            m0, S0, meta = self._resolve_pixel_corrected_belief_for_planning(now_msg)
        else:
            m0, S0, meta = self._resolve_state_belief_for_planning()
        if m0 is None or S0 is None:
            return None, None, {}
        measurement_available = bool(meta.get('measurement_available', False))
        belief_age_s = float(meta.get('belief_age_s', 0.0))
        self._latest_measurement_available = bool(measurement_available)
        self._latest_belief_age_s = float(belief_age_s)
        return m0, S0, {
            'measurement_available': bool(measurement_available),
            'belief_age_s': float(belief_age_s),
        }

    def _resolve_plan_frame_id(self):
        with self._data_lock:
            state_ref = self.state_msg
        return (
            (state_ref.header.frame_id if state_ref else '')
            or 'map_bev'
        )

    @staticmethod
    def _pose_covariance_from_state_covariance(S):
        pose_cov = [0.0] * 36
        if S is None:
            return pose_cov
        S = np.asarray(S, dtype=float)
        if S.shape[0] < 3 or S.shape[1] < 3:
            return pose_cov
        idx = (0, 1, 5)
        for i_src, i_dst in enumerate(idx):
            for j_src, j_dst in enumerate(idx):
                pose_cov[i_dst * 6 + j_dst] = float(S[i_src, j_src])
        return pose_cov

    def _build_path_message(self, result, goal_xy, *, append_goal=True, frame_id=None, stamp=None):
        path = Path()
        path.header.frame_id = frame_id or self._resolve_plan_frame_id()
        path.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()

        for state in result.states:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(state[0])
            p.pose.position.y = float(state[1])
            p.pose.orientation.z = math.sin(0.5 * float(state[2]))
            p.pose.orientation.w = math.cos(0.5 * float(state[2]))
            path.poses.append(p)

        if append_goal:
            goal_pose = PoseStamped()
            goal_pose.header = path.header
            goal_pose.pose.position.x = float(goal_xy[0])
            goal_pose.pose.position.y = float(goal_xy[1])
            goal_pose.pose.orientation.w = 1.0
            path.poses.append(goal_pose)
        return path

    def _belief_publish_tick(self):
        """High-rate belief publisher.

        Propagates the latest internal belief by the motion model with the
        most recent commanded velocity, then publishes the result. This
        keeps the belief mean alive (with growing covariance) between plan
        iterations and during stretches where no perception update arrives,
        which is what the post-hoc analysis needs in order to visualize
        prior-only propagation.
        """
        with self._data_lock:
            if self.belief_m is None or self.belief_S is None:
                return
            m = self.belief_m.copy()
            S = self.belief_S.copy()
            stamp_msg = self.belief_stamp
            last_cmd = np.asarray(self.last_cmd, dtype=float).copy()
        if stamp_msg is None:
            return
        try:
            age_s = max(0.0, self._stamp_age_s(stamp_msg))
        except Exception:
            age_s = 0.0
        if age_s > 1e-3:
            try:
                m, S = self.planner.predict(m, S, last_cmd, dt=age_s)
            except Exception:
                return
        belief_msg = self._build_belief_message(
            m, S, frame_id=self._resolve_plan_frame_id(),
            stamp=self.get_clock().now().to_msg(),
        )
        self.planner_belief_pub.publish(belief_msg)

    def _build_belief_message(self, m0, S0, *, frame_id=None, stamp=None):
        belief = PoseWithCovarianceStamped()
        belief.header.frame_id = frame_id or self._resolve_plan_frame_id()
        belief.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        belief.pose.pose.position.x = float(m0[0])
        belief.pose.pose.position.y = float(m0[1])
        belief.pose.pose.orientation.z = math.sin(0.5 * float(m0[2]))
        belief.pose.pose.orientation.w = math.cos(0.5 * float(m0[2]))
        belief.pose.covariance = self._pose_covariance_from_state_covariance(S0)
        return belief

    def _publish_plan_and_metrics(self, result, goal_xy, m0, S0, *, belief_meta=None):
        frame_id = self._resolve_plan_frame_id()
        stamp = self.get_clock().now().to_msg()
        path = self._build_path_message(
            result, goal_xy, append_goal=True, frame_id=frame_id, stamp=stamp
        )
        preview_path = self._build_path_message(
            result, goal_xy, append_goal=False, frame_id=frame_id, stamp=stamp
        )
        belief_msg = self._build_belief_message(m0, S0, frame_id=frame_id, stamp=stamp)
        self.path_pub.publish(path)
        self.plan_preview_pub.publish(preview_path)
        self.planner_belief_pub.publish(belief_msg)

        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            float(result.total_cost),
            float(result.risk_cost),
            float(result.ambiguity_cost),
            float(result.control_cost),
            float(result.obstacle_cost),
            float(getattr(result, 'p_vis_plan', 1.0)),
            float(getattr(result, 'p_vis_plan_eff', 1.0)),
            float(getattr(result, 'r_plan_u_std', np.nan)),
            float(getattr(result, 'r_plan_v_std', np.nan)),
            1.0 if (belief_meta or {}).get('measurement_available', False) else 0.0,
            float((belief_meta or {}).get('belief_age_s', math.nan)),
            float(getattr(result, 'terminal_goal_distance_pred', np.nan)),
            float(getattr(result, 'terminal_goal_progress_m', np.nan)),
            float(getattr(result, 'fraction_horizon_low_pvis', np.nan)),
            float(getattr(result, 'fraction_horizon_high_ambiguity', np.nan)),
            float(getattr(result, 'min_predicted_obstacle_distance_m', np.nan)),
            1.0 if getattr(result, 'rollout_valid', True) else 0.0,
            1.0 if getattr(result, 'fallback_stop_applied', False) else 0.0,
            float(getattr(result, 'risk_mean', np.nan)),
            float(getattr(result, 'risk_cov_trace', np.nan)),
            float(getattr(result, 'risk_cov_logdet', np.nan)),
            float(getattr(result, 'delta_risk_visibility', np.nan)),
            float(getattr(result, 'delta_ambiguity_visibility', np.nan)),
        ]
        self.metrics_pub.publish(metrics_msg)

    def _after_plan_result(self, result):
        """Hook for subclasses (e.g. agent node) to publish extra outputs."""
        return

    def _publish_planner_diagnostics(self, result, plan_elapsed_ms, *, belief_meta=None):
        diag = Float64MultiArray()
        diag.data = [
            1.0 if getattr(result, 'optimizer_success', False) else 0.0,
            float(getattr(result, 'optimizer_status', 0)),
            float(getattr(result, 'optimizer_nit', 0)),
            float(getattr(result, 'optimizer_nfev', 0)),
            float(plan_elapsed_ms),
            float(getattr(result, 'solve_time_s', 0.0)) * 1000.0,
            float(getattr(result, 'p_vis_plan', 1.0)),
            float(getattr(result, 'p_vis_plan_eff', 1.0)),
            float(getattr(result, 'r_plan_u_std', np.nan)),
            float(getattr(result, 'r_plan_v_std', np.nan)),
            1.0 if (belief_meta or {}).get('measurement_available', False) else 0.0,
            float((belief_meta or {}).get('belief_age_s', math.nan)),
            float(getattr(result, 'terminal_goal_distance_pred', np.nan)),
            float(getattr(result, 'terminal_goal_progress_m', np.nan)),
            float(getattr(result, 'fraction_horizon_low_pvis', np.nan)),
            float(getattr(result, 'fraction_horizon_high_ambiguity', np.nan)),
            float(getattr(result, 'min_predicted_obstacle_distance_m', np.nan)),
            1.0 if getattr(result, 'rollout_valid', True) else 0.0,
            1.0 if getattr(result, 'fallback_stop_applied', False) else 0.0,
            float(getattr(result, 'risk_mean', np.nan)),
            float(getattr(result, 'risk_cov_trace', np.nan)),
            float(getattr(result, 'risk_cov_logdet', np.nan)),
            float(getattr(result, 'delta_risk_visibility', np.nan)),
            float(getattr(result, 'delta_ambiguity_visibility', np.nan)),
        ]
        self.planner_diag_pub.publish(diag)
        diag_text = String()
        diag_parts = [str(getattr(result, 'optimizer_message', '') or '').strip()]
        invalid_reason = str(getattr(result, 'invalid_reason', '') or '').strip()
        if invalid_reason:
            diag_parts.append(f'invalid_reason={invalid_reason}')
        if bool(getattr(result, 'fallback_stop_applied', False)):
            diag_parts.append('fallback_stop_applied=1')
        diag_text.data = ' | '.join(part for part in diag_parts if part)
        self.planner_diag_text_pub.publish(diag_text)

    def _snapshot_plan_inputs(self):
        with self._data_lock:
            return {
                'goal': self.goal_msg,
                'pixel_stamp': self.pixel_stamp,
                'state': self.state_msg,
            }

    def _validate_plan_frames(self, goal_ref, state_ref) -> tuple[str, str]:
        goal_frame = (goal_ref.header.frame_id or '').strip()
        state_frame = (state_ref.header.frame_id or '').strip() if state_ref is not None else ''
        if goal_frame and state_frame and goal_frame != state_frame:
            self._fatal_experiment_stop(
                "Frame mismatch between /goal_bev and /state/bev "
                f"(goal='{goal_frame}', state='{state_frame}')"
            )
        return goal_frame, state_frame

    def _goal_xy_from_msg(self, goal_ref: PoseStamped):
        return (
            float(goal_ref.pose.position.x),
            float(goal_ref.pose.position.y),
        )

    def _call_planner(self, m0, S0, goal_xy, progress_index, *, plan_start, now_wall):
        if self.debug_runtime and (now_wall - self._last_plan_entry_log) > self.debug_log_period_s:
            self.get_logger().info(
                "Entering planner.plan: "
                f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
                f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})"
            )
            self._last_plan_entry_log = now_wall
        try:
            # Deliberately broad: any unexpected planner failure should abort the
            # run immediately instead of allowing an invalid experiment to continue.
            result = self.planner.plan(m0, S0, goal_xy, progress_index=progress_index)
        except Exception as exc:
            self._fatal_experiment_stop("Planner.solve raised an exception", exc)
            return None

        after_plan_wall = time.monotonic()
        if self.debug_runtime and (after_plan_wall - self._last_plan_return_log) > self.debug_log_period_s:
            elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
            self.get_logger().info(
                "Returned from planner.plan: "
                f"backend={getattr(result, 'backend', 'casadi') if result is not None else 'casadi'}, "
                f"elapsed_ms={elapsed_ms:.1f}, "
                f"success={getattr(result, 'optimizer_success', False) if result is not None else False}"
            )
            self._last_plan_return_log = after_plan_wall
        if result is None:
            self._fatal_experiment_stop("Planner returned no result")
            return None
        return result

    def _publish_plan_result_bundle(self, result, goal_xy, m0, S0, *, belief_meta, plan_elapsed_ms):
        self._publish_plan_and_metrics(result, goal_xy, m0, S0, belief_meta=belief_meta)
        self._after_plan_result(result)
        self._publish_planner_diagnostics(result, plan_elapsed_ms, belief_meta=belief_meta)

    def _warn_on_plan_health(self, result, plan_elapsed_ms, solve_elapsed_ms, *, now_wall):
        if self.debug_runtime and plan_elapsed_ms > (self.slow_plan_factor * self._plan_period_s * 1000.0):
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
                "Executing the selected solver-returned control sequence."
            )
            self._last_slow_plan_log = now_wall

        if bool(getattr(result, 'fallback_stop_applied', False)) and (now_wall - self._last_slow_plan_log > 0.5):
            self.get_logger().warn(
                "Planner selected a rollout that violates the collision barrier. "
                f"Publishing a zero-motion fallback for this cycle (reason={getattr(result, 'invalid_reason', '')})."
            )
            self._last_slow_plan_log = now_wall

    def _pixel_age_for_debug(self, pixel_stamp_ref):
        if pixel_stamp_ref is None:
            return None
        try:
            return (self.get_clock().now() - Time.from_msg(pixel_stamp_ref)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return None

    def _log_plan_debug_once(
        self,
        result,
        m0,
        goal_xy,
        *,
        plan_elapsed_ms,
        solve_elapsed_ms,
        goal_frame,
        state_frame,
        pixel_stamp_ref,
        now_wall,
    ):
        if not (self.debug_runtime and (now_wall - self._last_runtime_log) > self.debug_log_period_s):
            return
        pixel_age = self._pixel_age_for_debug(pixel_stamp_ref)
        self.get_logger().info(
            "Plan debug: "
            f"backend={getattr(result, 'backend', 'unknown')}, "
            f"success={getattr(result, 'optimizer_success', False)}, "
            f"status={getattr(result, 'optimizer_status', 0)}, "
            f"nit={getattr(result, 'optimizer_nit', 0)}, "
            f"nfev={getattr(result, 'optimizer_nfev', 0)}, "
            f"plan_ms={plan_elapsed_ms:.1f}, solve_ms={solve_elapsed_ms:.1f}, "
            f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
            f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), "
            f"frames=({state_frame or 'n/a'}->{goal_frame or 'n/a'}), "
            f"u0=({result.controls[0, 0]:.3f},{result.controls[0, 1]:.3f}), "
            f"J={result.total_cost:.3f}, "
            f"pixel_age={pixel_age if pixel_age is not None else 'n/a'}"
        )
        self._last_runtime_log = now_wall

    def _plan_once(self):
        inputs = self._snapshot_plan_inputs()
        goal_ref = inputs['goal']
        pixel_stamp_ref = inputs['pixel_stamp']
        state_ref = inputs['state']
        if goal_ref is None:
            return

        now_wall = time.monotonic()
        goal_frame, state_frame = self._validate_plan_frames(goal_ref, state_ref)

        m0, S0, belief_meta = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return

        goal_xy = self._goal_xy_from_msg(goal_ref)
        progress_index = self._current_goal_progress_index(m0, goal_xy)

        plan_start = time.perf_counter()
        result = self._call_planner(
            m0, S0, goal_xy, progress_index, plan_start=plan_start, now_wall=now_wall
        )
        if result is None:
            return

        plan_elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
        solve_elapsed_ms = float(getattr(result, 'solve_time_s', 0.0)) * 1000.0
        self._publish_plan_result_bundle(
            result, goal_xy, m0, S0, belief_meta=belief_meta, plan_elapsed_ms=plan_elapsed_ms
        )
        self._warn_on_plan_health(result, plan_elapsed_ms, solve_elapsed_ms, now_wall=now_wall)
        self._log_plan_debug_once(
            result,
            m0,
            goal_xy,
            plan_elapsed_ms=plan_elapsed_ms,
            solve_elapsed_ms=solve_elapsed_ms,
            goal_frame=goal_frame,
            state_frame=state_frame,
            pixel_stamp_ref=pixel_stamp_ref,
            now_wall=now_wall,
        )
