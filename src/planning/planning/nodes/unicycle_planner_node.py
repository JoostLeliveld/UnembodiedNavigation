"""Thin ROS 2 wrapper around unicycle planners."""

import math
import os
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from planning.core.efe_utils import wrap_angle


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

        # Standalone defaults for the visibility-aware thesis planner node.

        # Planner params
        _declare_if_not('plan_rate', 1.0)
        _declare_if_not('horizon', 10)
        _declare_if_not('dt', 0.2)
        _declare_if_not('v_min', 0.0)
        _declare_if_not('v_max', 0.22)
        _declare_if_not('w_min', -1.0)
        _declare_if_not('w_max', 1.0)
        _declare_if_not('control_weight', 0.1)
        _declare_if_not('seed', 0)

        # Process/observation noise
        _declare_if_not('process_noise_xy', 0.01)
        _declare_if_not('process_noise_theta', 0.02)
        _declare_if_not('obs_noise_uv', 2.0)

        # Goal covariance
        _declare_if_not('goal_sigma_xy', 0.25)
        _declare_if_not('goal_sigma_theta', 0.5)
        _declare_if_not('goal_sigma_uv', 0.0)

        # EFE weights
        _declare_if_not('risk_weight_state', 1.0)
        _declare_if_not('risk_weight_obs', 1.0)
        _declare_if_not('ambiguity_weight', 1.0)
        _declare_if_not('approx_method', 'ET2')
        _declare_if_not('use_obs_risk', True)
        _declare_if_not('use_ambiguity', True)
        _declare_if_not('optimizer_backend', 'auto')
        _declare_if_not('use_visibility_model', False)
        _declare_if_not('visibility_model', 'fixed_gp')
        _declare_if_not('visibility_weight', 0.0)
        _declare_if_not('visibility_map_min_x', -5.0)
        _declare_if_not('visibility_map_max_x', 5.0)
        _declare_if_not('visibility_map_min_y', -5.0)
        _declare_if_not('visibility_map_max_y', 5.0)
        _declare_if_not('visibility_map_nx', 140)
        _declare_if_not('visibility_map_ny', 120)
        _declare_if_not('visibility_occ_center_x', -1.2)
        _declare_if_not('visibility_occ_center_y', -1.8)
        _declare_if_not('visibility_occ_radius', 0.9)
        _declare_if_not('visibility_occ_tau', 0.15)
        _declare_if_not('visibility_gp_length_scale', 1.4)
        _declare_if_not('visibility_gp_noise_var', 0.15)
        _declare_if_not('visibility_prior_occ', 0.005)
        _declare_if_not('visibility_beta', 1.0)
        _declare_if_not('visibility_height_tau', 0.08)
        _declare_if_not('visibility_ray_samples', 120)
        _declare_if_not('visibility_sigma_kappa', 1.0)
        _declare_if_not('visibility_target_height_m', 0.0)
        _declare_if_not('visibility_geometry_json', '')
        _declare_if_not('visibility_gp_seed', 0)
        _declare_if_not('visibility_r_bad_uv', 28.0)
        _declare_if_not('visibility_cov_pos_scale', 2.0)
        _declare_if_not('visibility_cov_theta_scale', 0.8)
        _declare_if_not('publish_visibility_map', True)
        _declare_if_not('visibility_map_topic', '/visibility_map')
        _declare_if_not('publish_gp_debug_maps', True)
        _declare_if_not('visibility_gp_mean_map_topic', '/visibility_gp_mean_map')
        _declare_if_not('visibility_gp_conservative_map_topic', '/visibility_gp_conservative_map')
        _declare_if_not('publish_visibility_logic_markers', True)
        _declare_if_not('visibility_logic_marker_topic', '/visibility_logic_markers')
        _declare_if_not('experiment_run_dir_topic', '/experiment/run_dir')
        _declare_if_not('visibility_artifact_filename', 'visibility_artifacts.npz')

        # Optimizer params
        _declare_if_not('optimizer_maxiter', 50)
        _declare_if_not('optimizer_gtol', 1e-4)
        _declare_if_not('optimizer_warm_start', True)

        # Pixel correction params
        _declare_if_not('use_pixel_correction', False)
        _declare_if_not('pixel_topic', '/perception/pixel_pose')
        _declare_if_not('pixel_timeout_s', 0.5)
        _declare_if_not('pixel_correction_min_interval_s', 0.0)
        _declare_if_not('pixel_correction_approx', 'ET1')
        _declare_if_not('skip_stale_pixel_correction', True)
        _declare_if_not('min_state_cov', 1e-6)
        _declare_if_not('debug_runtime', False)
        _declare_if_not('debug_log_period_s', 1.0)
        _declare_if_not('slow_plan_factor', 1.0)
        _declare_if_not('slow_correction_ms', 20.0)
        _declare_if_not('jax_warmup_enabled', True)
        _declare_if_not('jax_warmup_poll_s', 0.5)
        _declare_if_not('jax_warmup_use_goal_hint', False)
        _declare_if_not('jax_warmup_goal_x', 0.0)
        _declare_if_not('jax_warmup_goal_y', 0.0)
        _declare_if_not('jax_warmup_goal_frame_id', 'map_bev')

        # Camera model params (must match sim)
        _declare_if_not('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_not('look_at', [1.5, 1.5, 0.0])
        _declare_if_not('img_width', 1280)
        _declare_if_not('img_height', 720)
        _declare_if_not('fov_h_rad', 1.5708)

        self.plan_rate = float(self.get_parameter('plan_rate').value)
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

        self.goal_sigma_xy = float(self.get_parameter('goal_sigma_xy').value)
        self.goal_sigma_theta = float(self.get_parameter('goal_sigma_theta').value)
        self.goal_sigma_uv = float(self.get_parameter('goal_sigma_uv').value)

        self.risk_weight_state = float(self.get_parameter('risk_weight_state').value)
        self.risk_weight_obs = float(self.get_parameter('risk_weight_obs').value)
        self.ambiguity_weight = float(self.get_parameter('ambiguity_weight').value)
        self.approx_method = str(self.get_parameter('approx_method').value).upper()
        self.use_obs_risk = _as_bool(self.get_parameter('use_obs_risk').value)
        self.use_ambiguity = _as_bool(self.get_parameter('use_ambiguity').value)
        self.optimizer_backend = str(self.get_parameter('optimizer_backend').value).strip().lower()
        self.use_visibility_model = _as_bool(self.get_parameter('use_visibility_model').value)
        self.visibility_model = str(self.get_parameter('visibility_model').value).strip().lower()
        self.visibility_weight = float(self.get_parameter('visibility_weight').value)
        self.visibility_map_min_x = float(self.get_parameter('visibility_map_min_x').value)
        self.visibility_map_max_x = float(self.get_parameter('visibility_map_max_x').value)
        self.visibility_map_min_y = float(self.get_parameter('visibility_map_min_y').value)
        self.visibility_map_max_y = float(self.get_parameter('visibility_map_max_y').value)
        self.visibility_map_nx = int(self.get_parameter('visibility_map_nx').value)
        self.visibility_map_ny = int(self.get_parameter('visibility_map_ny').value)
        self.visibility_occ_center_x = float(self.get_parameter('visibility_occ_center_x').value)
        self.visibility_occ_center_y = float(self.get_parameter('visibility_occ_center_y').value)
        self.visibility_occ_radius = float(self.get_parameter('visibility_occ_radius').value)
        self.visibility_occ_tau = float(self.get_parameter('visibility_occ_tau').value)
        self.visibility_gp_length_scale = float(self.get_parameter('visibility_gp_length_scale').value)
        self.visibility_gp_noise_var = float(self.get_parameter('visibility_gp_noise_var').value)
        self.visibility_prior_occ = float(self.get_parameter('visibility_prior_occ').value)
        self.visibility_beta = float(self.get_parameter('visibility_beta').value)
        self.visibility_height_tau = float(self.get_parameter('visibility_height_tau').value)
        self.visibility_ray_samples = int(self.get_parameter('visibility_ray_samples').value)
        self.visibility_sigma_kappa = float(self.get_parameter('visibility_sigma_kappa').value)
        self.visibility_target_height_m = float(self.get_parameter('visibility_target_height_m').value)
        self.visibility_geometry_json = str(self.get_parameter('visibility_geometry_json').value)
        self.visibility_gp_seed = int(self.get_parameter('visibility_gp_seed').value)
        self.visibility_r_bad_uv = float(self.get_parameter('visibility_r_bad_uv').value)
        self.visibility_cov_pos_scale = float(self.get_parameter('visibility_cov_pos_scale').value)
        self.visibility_cov_theta_scale = float(self.get_parameter('visibility_cov_theta_scale').value)
        self.publish_visibility_map = _as_bool(self.get_parameter('publish_visibility_map').value)
        self.visibility_map_topic = str(self.get_parameter('visibility_map_topic').value).strip() or '/visibility_map'
        self.publish_gp_debug_maps = _as_bool(self.get_parameter('publish_gp_debug_maps').value)
        self.visibility_gp_mean_map_topic = (
            str(self.get_parameter('visibility_gp_mean_map_topic').value).strip() or '/visibility_gp_mean_map'
        )
        self.visibility_gp_conservative_map_topic = (
            str(self.get_parameter('visibility_gp_conservative_map_topic').value).strip()
            or '/visibility_gp_conservative_map'
        )
        self.publish_visibility_logic_markers = _as_bool(
            self.get_parameter('publish_visibility_logic_markers').value
        )
        self.visibility_logic_marker_topic = (
            str(self.get_parameter('visibility_logic_marker_topic').value).strip()
            or '/visibility_logic_markers'
        )
        self.experiment_run_dir_topic = (
            str(self.get_parameter('experiment_run_dir_topic').value).strip() or '/experiment/run_dir'
        )
        self.visibility_artifact_filename = (
            str(self.get_parameter('visibility_artifact_filename').value).strip()
            or 'visibility_artifacts.npz'
        )

        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = _as_bool(self.get_parameter('optimizer_warm_start').value)

        self.use_pixel_correction = _as_bool(self.get_parameter('use_pixel_correction').value)
        self.pixel_topic = self.get_parameter('pixel_topic').value
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
        self.min_state_cov = float(self.get_parameter('min_state_cov').value)
        self.debug_runtime = _as_bool(self.get_parameter('debug_runtime').value)
        self.debug_log_period_s = max(0.2, float(self.get_parameter('debug_log_period_s').value))
        self.slow_plan_factor = max(0.1, float(self.get_parameter('slow_plan_factor').value))
        self.slow_correction_ms = max(0.1, float(self.get_parameter('slow_correction_ms').value))
        self.jax_warmup_enabled = _as_bool(self.get_parameter('jax_warmup_enabled').value) and self.optimizer_backend in ('jax', 'auto')
        self.jax_warmup_poll_s = max(0.1, float(self.get_parameter('jax_warmup_poll_s').value))
        self.jax_warmup_use_goal_hint = _as_bool(self.get_parameter('jax_warmup_use_goal_hint').value)
        self.jax_warmup_goal_hint = (
            float(self.get_parameter('jax_warmup_goal_x').value),
            float(self.get_parameter('jax_warmup_goal_y').value),
        )
        self.jax_warmup_goal_frame_id = str(self.get_parameter('jax_warmup_goal_frame_id').value).strip() or 'map_bev'

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
            process_noise_xy=self.process_noise_xy,
            process_noise_theta=self.process_noise_theta,
            obs_noise_uv=self.obs_noise_uv,
            goal_sigma_xy=self.goal_sigma_xy,
            goal_sigma_theta=self.goal_sigma_theta,
            goal_sigma_uv=self.goal_sigma_uv,
            risk_weight_state=self.risk_weight_state,
            risk_weight_obs=self.risk_weight_obs,
            ambiguity_weight=self.ambiguity_weight,
            optimizer_maxiter=self.optimizer_maxiter,
            optimizer_gtol=self.optimizer_gtol,
            optimizer_warm_start=self.optimizer_warm_start,
            approx_method=self.approx_method,
            use_obs_risk=self.use_obs_risk,
            use_ambiguity=self.use_ambiguity,
            optimizer_backend=self.optimizer_backend,
            seed=self.seed,
            camera_params=camera_params,
            use_visibility_model=self.use_visibility_model,
            visibility_model=self.visibility_model,
            visibility_weight=self.visibility_weight,
            visibility_map_min_x=self.visibility_map_min_x,
            visibility_map_max_x=self.visibility_map_max_x,
            visibility_map_min_y=self.visibility_map_min_y,
            visibility_map_max_y=self.visibility_map_max_y,
            visibility_map_nx=self.visibility_map_nx,
            visibility_map_ny=self.visibility_map_ny,
            visibility_occ_center_x=self.visibility_occ_center_x,
            visibility_occ_center_y=self.visibility_occ_center_y,
            visibility_occ_radius=self.visibility_occ_radius,
            visibility_occ_tau=self.visibility_occ_tau,
            visibility_gp_length_scale=self.visibility_gp_length_scale,
            visibility_gp_noise_var=self.visibility_gp_noise_var,
            visibility_prior_occ=self.visibility_prior_occ,
            visibility_beta=self.visibility_beta,
            visibility_height_tau=self.visibility_height_tau,
            visibility_ray_samples=self.visibility_ray_samples,
            visibility_sigma_kappa=self.visibility_sigma_kappa,
            visibility_target_height_m=self.visibility_target_height_m,
            visibility_geometry_json=self.visibility_geometry_json,
            visibility_gp_seed=self.visibility_gp_seed,
            visibility_r_bad_uv=self.visibility_r_bad_uv,
            visibility_cov_pos_scale=self.visibility_cov_pos_scale,
            visibility_cov_theta_scale=self.visibility_cov_theta_scale,
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
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )
        self.run_dir_sub = self.create_subscription(
            String, self.experiment_run_dir_topic, self._experiment_run_dir_cb,
            qos_profile=goal_qos, callback_group=self._io_group
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
        self.visibility_map_pub = self.create_publisher(
            OccupancyGrid, self.visibility_map_topic, qos_profile=path_qos
        )
        self.visibility_gp_mean_map_pub = self.create_publisher(
            OccupancyGrid, self.visibility_gp_mean_map_topic, qos_profile=path_qos
        )
        self.visibility_gp_conservative_map_pub = self.create_publisher(
            OccupancyGrid, self.visibility_gp_conservative_map_topic, qos_profile=path_qos
        )
        self.visibility_logic_pub = self.create_publisher(
            MarkerArray, self.visibility_logic_marker_topic, qos_profile=path_qos
        )

        # State
        self.state_msg = None
        self.goal_msg = None
        self._goal_received_logged = False
        self.pixel_meas = None
        self.pixel_stamp = None
        self._last_correction_log = 0.0
        self._last_correction_stamp = None
        self._last_stale_log = 0.0
        self._last_shape_mismatch_log = 0.0
        self._last_runtime_log = 0.0
        self._last_plan_entry_log = 0.0
        self._last_plan_return_log = 0.0
        self._last_slow_plan_log = 0.0
        self._last_slow_correction_log = 0.0
        self._last_jax_warmup_log = 0.0
        self._jax_warmup_done = not self.jax_warmup_enabled
        self._jax_warmup_started = False
        self._jax_warmup_timer = None
        self._fatal_stop_triggered = False
        self.belief_m = None
        self.belief_S = None
        self.belief_stamp = None
        self._experiment_run_dir = ''
        self._visibility_artifact_written = False
        self.last_cmd = np.array([0.0, 0.0], dtype=float)

        self._plan_period_s = 1.0 / max(self.plan_rate, 0.1)
        self.create_timer(self._plan_period_s, self._plan_once, callback_group=self._plan_group)
        if self.jax_warmup_enabled:
            self._jax_warmup_timer = self.create_timer(
                self.jax_warmup_poll_s, self._jax_warmup_timer_cb, callback_group=self._plan_group
            )
        self._pixel_correction_timer = None
        if self.use_pixel_correction and self.pixel_correction_min_interval_s > 0.0:
            correction_period = max(self.pixel_correction_min_interval_s, 0.02)
            self._pixel_correction_timer = self.create_timer(
                correction_period, self._pixel_correction_timer_cb, callback_group=self._io_group
            )
        self._publish_visibility_map_once()
        self.get_logger().info(
            f"{self.NODE_NAME} started "
            f"(approx={self.approx_method}, "
            f"use_obs_risk={self.use_obs_risk}, use_ambiguity={self.use_ambiguity}, "
            f"use_visibility_model={self.use_visibility_model}, visibility_weight={self.visibility_weight:.3f}, "
            f"use_pixel_correction={self.use_pixel_correction}, "
            f"pixel_correction_approx={self.pixel_correction_approx}, "
            f"jax_warmup_enabled={self.jax_warmup_enabled}, "
            f"jax_warmup_use_goal_hint={self.jax_warmup_use_goal_hint}, "
            f"debug_runtime={self.debug_runtime})"
        )

    def _build_grid_message_from_array(self, cfg, map_array, *, xs=None, ys=None, invert=False):
        arr = np.asarray(map_array, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            return None

        xs_arr = None if xs is None else np.asarray(xs, dtype=float).reshape(-1)
        ys_arr = None if ys is None else np.asarray(ys, dtype=float).reshape(-1)
        if (
            xs_arr is not None
            and ys_arr is not None
            and xs_arr.size == arr.shape[1]
            and ys_arr.size == arr.shape[0]
            and xs_arr.size > 1
            and ys_arr.size > 1
        ):
            res_x = float(xs_arr[1] - xs_arr[0])
            res_y = float(ys_arr[1] - ys_arr[0])
            resolution = float(max(min(abs(res_x), abs(res_y)), 1e-3))
            width = int(xs_arr.size)
            height = int(ys_arr.size)
            origin_x = float(xs_arr[0]) - 0.5 * resolution
            origin_y = float(ys_arr[0]) - 0.5 * resolution
        else:
            width = int(arr.shape[1])
            height = int(arr.shape[0])
            span_x = float(cfg.map_xmax) - float(cfg.map_xmin)
            span_y = float(cfg.map_ymax) - float(cfg.map_ymin)
            res_x = span_x / max(width - 1, 1)
            res_y = span_y / max(height - 1, 1)
            resolution = float(max(min(res_x, res_y), 1e-3))
            origin_x = float(cfg.map_xmin) - 0.5 * resolution
            origin_y = float(cfg.map_ymin) - 0.5 * resolution

        values = 1.0 - arr if invert else arr
        data = np.clip(np.rint(values * 100.0), 0, 100).astype(np.int8).ravel().tolist()

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map_bev'
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = data
        return grid

    def _sample_visibility_map_array(self, visibility_model):
        cfg = visibility_model.cfg
        xs = np.linspace(float(cfg.map_xmin), float(cfg.map_xmax), int(max(cfg.map_nx, 4)))
        ys = np.linspace(float(cfg.map_ymin), float(cfg.map_ymax), int(max(cfg.map_ny, 4)))
        arr = np.zeros((ys.size, xs.size), dtype=float)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                arr[iy, ix] = float(visibility_model.prob_state_np(np.array([x, y, 0.0], dtype=float)))
        return xs, ys, arr

    def _publish_visibility_grid(self, publisher, topic_name, cfg, map_array, *, xs=None, ys=None, invert=False):
        grid = self._build_grid_message_from_array(cfg, map_array, xs=xs, ys=ys, invert=invert)
        if grid is None:
            return False
        publisher.publish(grid)
        self.get_logger().info(
            f"Published {topic_name} ({grid.info.width}x{grid.info.height}, "
            f"res={grid.info.resolution:.3f} m/cell)"
        )
        return True

    def _write_visibility_artifacts_if_ready(self):
        if self._visibility_artifact_written or (not self._experiment_run_dir):
            return
        visibility_model = getattr(self.planner, 'visibility_model', None)
        if visibility_model is None:
            return

        p_map = getattr(visibility_model, 'P_map', None)
        xs = getattr(visibility_model, 'xs', None)
        ys = getattr(visibility_model, 'ys', None)
        if p_map is None or xs is None or ys is None:
            return

        artifact_path = os.path.join(self._experiment_run_dir, self.visibility_artifact_filename)
        cfg = getattr(visibility_model, 'cfg', None)
        geometry_json = '' if cfg is None else str(getattr(cfg, 'geometry_json', '') or '')
        rho_mean = getattr(visibility_model, 'rho_mean_map', np.zeros_like(np.asarray(p_map, dtype=float)))
        rho_cons = getattr(visibility_model, 'rho_conservative_map', np.zeros_like(np.asarray(p_map, dtype=float)))
        camera_pos = getattr(visibility_model, 'camera_pos', np.asarray([0.0, 0.0, 0.0], dtype=float))
        target_height = float(getattr(visibility_model, 'target_height', self.visibility_target_height_m))

        try:
            os.makedirs(self._experiment_run_dir, exist_ok=True)
            np.savez_compressed(
                artifact_path,
                xs=np.asarray(xs, dtype=float),
                ys=np.asarray(ys, dtype=float),
                rho_mean_map=np.asarray(rho_mean, dtype=float),
                rho_conservative_map=np.asarray(rho_cons, dtype=float),
                P_map=np.asarray(p_map, dtype=float),
                camera_pos=np.asarray(camera_pos, dtype=float).reshape(-1),
                target_height=np.asarray([target_height], dtype=float),
                geometry_json=np.asarray(str(geometry_json)),
                visibility_model=np.asarray(str(self.visibility_model)),
            )
            self._visibility_artifact_written = True
            self.get_logger().info(f"Wrote visibility artifact to {artifact_path}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to write visibility artifact to {artifact_path}: {exc}")

    def _publish_visibility_map_once(self):
        visibility_model = getattr(self.planner, 'visibility_model', None)
        if visibility_model is None:
            return

        cfg = visibility_model.cfg
        xs = getattr(visibility_model, 'xs', None)
        ys = getattr(visibility_model, 'ys', None)
        p_map = getattr(visibility_model, 'P_map', None)
        if p_map is None:
            xs, ys, p_map = self._sample_visibility_map_array(visibility_model)

        if self.publish_visibility_map:
            self._publish_visibility_grid(
                self.visibility_map_pub,
                self.visibility_map_topic,
                cfg,
                p_map,
                xs=xs,
                ys=ys,
                invert=True,
            )

        if self.publish_gp_debug_maps:
            rho_mean = getattr(visibility_model, 'rho_mean_map', None)
            rho_cons = getattr(visibility_model, 'rho_conservative_map', None)
            if rho_mean is not None:
                self._publish_visibility_grid(
                    self.visibility_gp_mean_map_pub,
                    self.visibility_gp_mean_map_topic,
                    cfg,
                    rho_mean,
                    xs=xs,
                    ys=ys,
                    invert=False,
                )
            if rho_cons is not None:
                self._publish_visibility_grid(
                    self.visibility_gp_conservative_map_pub,
                    self.visibility_gp_conservative_map_topic,
                    cfg,
                    rho_cons,
                    xs=xs,
                    ys=ys,
                    invert=False,
                )

        self._write_visibility_artifacts_if_ready()

    def _visibility_debug_support_points(self, m0, S0, target_height):
        mean = np.asarray(m0, dtype=float).reshape(-1)
        if mean.size < 2:
            return np.zeros((0, 3), dtype=float)

        points = [[float(mean[0]), float(mean[1]), float(target_height)]]
        S = None if S0 is None else np.asarray(S0, dtype=float)
        kappa = max(float(self.visibility_sigma_kappa), 0.0)
        if S is None or S.ndim != 2 or S.shape[0] < 2 or S.shape[1] < 2 or kappa <= 0.0:
            return np.asarray(points, dtype=float)

        sx = math.sqrt(max(float(S[0, 0]), 0.0)) * kappa
        sy = math.sqrt(max(float(S[1, 1]), 0.0)) * kappa
        for dx, dy in ((sx, 0.0), (-sx, 0.0), (0.0, sy), (0.0, -sy)):
            if abs(dx) + abs(dy) <= 1e-9:
                continue
            points.append([float(mean[0] + dx), float(mean[1] + dy), float(target_height)])
        return np.asarray(points, dtype=float)

    def _publish_visibility_logic_markers(self, m0, S0):
        if not self.publish_visibility_logic_markers:
            return

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        visibility_model = getattr(self.planner, 'visibility_model', None)
        if visibility_model is None:
            self.visibility_logic_pub.publish(markers)
            return

        mean = np.asarray(m0, dtype=float).reshape(-1)
        if mean.size < 2:
            self.visibility_logic_pub.publish(markers)
            return

        frame_id = 'map_bev'
        stamp = self.get_clock().now().to_msg()
        camera_pos = np.asarray(
            getattr(visibility_model, 'camera_pos', np.asarray([0.0, 0.0, 0.0], dtype=float)),
            dtype=float,
        ).reshape(-1)
        if camera_pos.size < 3:
            camera_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        target_height = float(getattr(visibility_model, 'target_height', self.visibility_target_height_m))
        support_points = self._visibility_debug_support_points(m0, S0, target_height)

        p_vis = 1.0
        try:
            if self.use_visibility_model:
                p_vis = float(self.planner.visibility_probability_belief(m0, S0))
        except Exception:
            try:
                p_vis = float(visibility_model.prob_state_np(np.array([mean[0], mean[1], 0.0], dtype=float)))
            except Exception:
                p_vis = 1.0
        p_vis = float(np.clip(p_vis, 0.0, 1.0))
        ray_r = float(1.0 - p_vis)
        ray_g = float(p_vis)

        camera_marker = Marker()
        camera_marker.header.frame_id = frame_id
        camera_marker.header.stamp = stamp
        camera_marker.ns = 'visibility_logic'
        camera_marker.id = 0
        camera_marker.type = Marker.SPHERE
        camera_marker.action = Marker.ADD
        camera_marker.pose.position.x = float(camera_pos[0])
        camera_marker.pose.position.y = float(camera_pos[1])
        camera_marker.pose.position.z = float(camera_pos[2])
        camera_marker.pose.orientation.w = 1.0
        camera_marker.scale.x = 0.24
        camera_marker.scale.y = 0.24
        camera_marker.scale.z = 0.24
        camera_marker.color.r = 0.15
        camera_marker.color.g = 0.65
        camera_marker.color.b = 1.0
        camera_marker.color.a = 0.95
        markers.markers.append(camera_marker)

        target_marker = Marker()
        target_marker.header.frame_id = frame_id
        target_marker.header.stamp = stamp
        target_marker.ns = 'visibility_logic'
        target_marker.id = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = float(mean[0])
        target_marker.pose.position.y = float(mean[1])
        target_marker.pose.position.z = target_height
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.18
        target_marker.scale.y = 0.18
        target_marker.scale.z = 0.18
        target_marker.color.r = ray_r
        target_marker.color.g = ray_g
        target_marker.color.b = 0.1
        target_marker.color.a = 0.95
        markers.markers.append(target_marker)

        ray_marker = Marker()
        ray_marker.header.frame_id = frame_id
        ray_marker.header.stamp = stamp
        ray_marker.ns = 'visibility_logic'
        ray_marker.id = 2
        ray_marker.type = Marker.LINE_STRIP
        ray_marker.action = Marker.ADD
        ray_marker.scale.x = 0.05
        ray_marker.color.r = ray_r
        ray_marker.color.g = ray_g
        ray_marker.color.b = 0.15
        ray_marker.color.a = 0.9
        ray_start = Point(x=float(camera_pos[0]), y=float(camera_pos[1]), z=float(camera_pos[2]))
        ray_end = Point(x=float(mean[0]), y=float(mean[1]), z=target_height)
        ray_marker.points = [ray_start, ray_end]
        markers.markers.append(ray_marker)

        support_marker = Marker()
        support_marker.header.frame_id = frame_id
        support_marker.header.stamp = stamp
        support_marker.ns = 'visibility_logic'
        support_marker.id = 3
        support_marker.type = Marker.SPHERE_LIST
        support_marker.action = Marker.ADD
        support_marker.scale.x = 0.10
        support_marker.scale.y = 0.10
        support_marker.scale.z = 0.10
        support_marker.color.r = 1.0
        support_marker.color.g = 0.85
        support_marker.color.b = 0.15
        support_marker.color.a = 0.9
        for pt in support_points:
            support_marker.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
        markers.markers.append(support_marker)

        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = stamp
        text_marker.ns = 'visibility_logic'
        text_marker.id = 4
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(mean[0])
        text_marker.pose.position.y = float(mean[1])
        text_marker.pose.position.z = target_height + 0.45
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.30
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.95
        text_marker.text = f"p_vis={p_vis:.2f}"
        markers.markers.append(text_marker)

        self.visibility_logic_pub.publish(markers)

    def _publish_safe_stop_command(self):
        """Hook for agent mode; planner-only nodes can ignore."""
        return

    def _fatal_experiment_stop(self, reason: str, exc: Exception | None = None):
        if self._fatal_stop_triggered:
            return
        self._fatal_stop_triggered = True

        try:
            self._publish_safe_stop_command()
        except Exception:
            pass

        detail = reason
        if exc is not None:
            detail = f"{reason}: {type(exc).__name__}: {exc}"
        self.get_logger().error(
            "Fatal experiment integrity failure. Publishing zero command and terminating node. "
            f"Reason: {detail}"
        )

        # Stop this process so runs fail fast instead of continuing with invalid behavior.
        try:
            rclpy.shutdown()
        except Exception:
            pass
        raise RuntimeError(detail) from exc

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        with self._data_lock:
            self.state_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        with self._data_lock:
            self.goal_msg = msg
            first_goal = not self._goal_received_logged
            if first_goal:
                self._goal_received_logged = True
        if first_goal:
            self.get_logger().info(
                f"Received goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
                f"frame='{msg.header.frame_id or 'map_bev'}'"
            )

    def _cmd_cb(self, msg: Twist):
        with self._data_lock:
            self.last_cmd = np.array([msg.linear.x, msg.angular.z], dtype=float)

    def _experiment_run_dir_cb(self, msg: String):
        run_dir = str(msg.data).strip()
        if not run_dir:
            return
        self._experiment_run_dir = run_dir
        self._write_visibility_artifacts_if_ready()

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
        u = msg.pose.position.x
        v = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        _ = math.atan2(siny_cosp, cosy_cosp)
        with self._data_lock:
            self.pixel_meas = np.array([u, v], dtype=float)
            self.pixel_stamp = msg.header.stamp

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

    def _apply_pixel_correction(self, stamp_msg, *, source='callback'):
        cb_start = time.perf_counter()
        try:
            now = self.get_clock().now()
            age = (now - Time.from_msg(stamp_msg)).nanoseconds * 1e-9
        except Exception:
            age = 0.0
        if self.skip_stale_pixel_correction and age > self.pixel_timeout_s:
            now_wall = time.monotonic()
            if now_wall - self._last_stale_log > 2.0:
                self.get_logger().warn(f"Skipping stale pixel measurement (age {age:.2f}s)")
                self._last_stale_log = now_wall
            return

        if self.pixel_correction_min_interval_s > 0.0:
            with self._data_lock:
                last_correction_stamp = self._last_correction_stamp
            if last_correction_stamp is not None:
                try:
                    dt_since_correction = (
                        Time.from_msg(stamp_msg) - Time.from_msg(last_correction_stamp)
                    ).nanoseconds * 1e-9
                except Exception:
                    dt_since_correction = None
                if (
                    dt_since_correction is not None
                    and 0.0 <= dt_since_correction < self.pixel_correction_min_interval_s
                ):
                    return

        with self._data_lock:
            has_belief = self.belief_m is not None and self.belief_S is not None
        if not has_belief and not self._init_belief_from_state():
            return

        try:
            now = Time.from_msg(stamp_msg)
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
        p_vis, R_eff, S_eff, gain_scale = self.planner.observation_model_with_visibility(m_pred, S_pred)

        corr_method = self.approx_method if self.pixel_correction_approx == 'AUTO' else self.pixel_correction_approx
        mu_y, Sigma_y, Gamma = self.planner.approx_observation(
            m_pred, S_eff, method=corr_method, R_override=R_eff
        )
        mu_y = np.asarray(mu_y, dtype=float).reshape(-1)
        meas = np.asarray(meas, dtype=float).reshape(-1)
        if meas.size != mu_y.size:
            now_wall = time.monotonic()
            if now_wall - self._last_shape_mismatch_log > 2.0:
                self.get_logger().error(
                    "Pixel correction shape mismatch: "
                    f"meas_dim={meas.size}, pred_dim={mu_y.size}. "
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
        next_m = m_pred + gain_scale * (K @ innov)
        next_m[2] = wrap_angle(next_m[2])
        next_S = S_eff - gain_scale * (Gamma @ Sigma_inv @ Gamma.T)
        next_S = (next_S + next_S.T) / 2.0
        if self.min_state_cov > 0.0:
            for i in range(min(3, next_S.shape[0])):
                if next_S[i, i] < self.min_state_cov:
                    next_S[i, i] = self.min_state_cov
        with self._data_lock:
            self.belief_m = next_m
            self.belief_S = next_S
            self.belief_stamp = stamp_msg
            self._last_correction_stamp = stamp_msg

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

    def _cancel_jax_warmup_timer(self):
        if self._jax_warmup_timer is None:
            return
        try:
            self._jax_warmup_timer.cancel()
        except Exception:
            pass
        self._jax_warmup_timer = None

    def _jax_warmup_timer_cb(self):
        if self._jax_warmup_done:
            self._cancel_jax_warmup_timer()
            return

        with self._data_lock:
            goal_ref = self.goal_msg
        goal_xy = None
        goal_source = 'goal topic'
        if self.jax_warmup_use_goal_hint:
            goal_xy = self.jax_warmup_goal_hint
            goal_frame = self.jax_warmup_goal_frame_id
            goal_source = 'goal hint'
        elif goal_ref is not None:
            goal_xy = (
                float(goal_ref.pose.position.x),
                float(goal_ref.pose.position.y),
            )
            goal_frame = (goal_ref.header.frame_id or '').strip() or 'map_bev'
        if goal_xy is None:
            return

        m0, S0 = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return

        self._jax_warmup_started = True
        self.get_logger().info(
            f"Starting JAX warm-up using {goal_source} for goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) frame='{goal_frame}'"
        )
        warm_start = time.perf_counter()
        try:
            warmed = bool(self.planner.warmup_jax(m0, S0, goal_xy))
        except Exception as exc:
            self._jax_warmup_done = True
            self._cancel_jax_warmup_timer()
            self.get_logger().warn(
                f"JAX warm-up failed; continuing without warm-up gate: {exc}"
            )
            return

        self._jax_warmup_done = True
        self._cancel_jax_warmup_timer()
        warm_ms = max((time.perf_counter() - warm_start) * 1000.0, 0.0)
        if warmed:
            self.get_logger().info(f"JAX warm-up completed in {warm_ms:.1f} ms")
        else:
            self.get_logger().info(
                "JAX warm-up skipped because the optimizer resolved away from the JAX path."
            )

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

    def _publish_plan_and_metrics(self, result, goal_xy, m0, S0):
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
            float(result.visibility_cost),
        ]
        self.metrics_pub.publish(metrics_msg)
        self._publish_visibility_logic_markers(m0, S0)

    def _after_plan_result(self, result):
        """Hook for subclasses (e.g. agent node) to publish extra outputs."""
        return

    def _plan_once(self):
        with self._data_lock:
            goal_ref = self.goal_msg
            pixel_stamp_ref = self.pixel_stamp
            state_ref = self.state_msg
        if goal_ref is None:
            return

        goal_frame = (goal_ref.header.frame_id or '').strip()
        state_frame = (state_ref.header.frame_id or '').strip() if state_ref is not None else ''
        now_wall = time.monotonic()
        if self.jax_warmup_enabled and not self._jax_warmup_done:
            if now_wall - self._last_jax_warmup_log > 2.0:
                self.get_logger().info(
                    "Waiting for JAX warm-up before planning."
                )
                self._last_jax_warmup_log = now_wall
            return
        if goal_frame and state_frame and goal_frame != state_frame:
            self._fatal_experiment_stop(
                "Frame mismatch between /goal_bev and /state/bev "
                f"(goal='{goal_frame}', state='{state_frame}')"
            )
            return

        m0, S0 = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return

        goal_xy = (
            float(goal_ref.pose.position.x),
            float(goal_ref.pose.position.y),
        )

        plan_start = time.perf_counter()
        if self.debug_runtime and (now_wall - self._last_plan_entry_log) > self.debug_log_period_s:
            self.get_logger().info(
                "Entering planner.plan: "
                f"backend={self.optimizer_backend}, "
                f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
                f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})"
            )
            self._last_plan_entry_log = now_wall
        try:
            result = self.planner.plan(m0, S0, goal_xy)
        except Exception as exc:
            self._fatal_experiment_stop("Planner.solve raised an exception", exc)
            return
        after_plan_wall = time.monotonic()
        if self.debug_runtime and (after_plan_wall - self._last_plan_return_log) > self.debug_log_period_s:
            elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
            self.get_logger().info(
                "Returned from planner.plan: "
                f"backend={getattr(result, 'backend', self.optimizer_backend) if result is not None else self.optimizer_backend}, "
                f"elapsed_ms={elapsed_ms:.1f}, "
                f"success={getattr(result, 'optimizer_success', False) if result is not None else False}"
            )
            self._last_plan_return_log = after_plan_wall
        if result is None:
            self._fatal_experiment_stop("Planner returned no result")
            return

        self._publish_plan_and_metrics(result, goal_xy, m0, S0)
        self._after_plan_result(result)

        plan_elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
        solve_elapsed_ms = float(getattr(result, 'solve_time_s', 0.0)) * 1000.0
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
                f"plan_ms={plan_elapsed_ms:.1f}, solve_ms={solve_elapsed_ms:.1f}, "
                f"x0=({m0[0]:.2f},{m0[1]:.2f},{m0[2]:.2f}), "
                f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}), "
                f"frames=({state_frame or 'n/a'}->{goal_frame or 'n/a'}), "
                f"u0=({result.controls[0, 0]:.3f},{result.controls[0, 1]:.3f}), "
                f"J={result.total_cost:.3f}, "
                f"pixel_age={pixel_age if pixel_age is not None else 'n/a'}"
            )
            self._last_runtime_log = now_wall
