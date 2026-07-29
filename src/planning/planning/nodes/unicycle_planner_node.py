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
from builtin_interfaces.msg import Time as TimeMsg

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)
from reliability.fusion import map_observations_from_json

from planning.core.efe_utils import wrap_angle
from planning.core import belief_correction as bc

PIXEL_DIAG_K_THETA_U_IDX = 42
PIXEL_DIAG_K_THETA_V_IDX = 43

# A metric xy correction says nothing about heading; keep its yaw variance
# non-informative so no consumer mistakes it for a heading fix. Matches
# camera_manager_node.NONINFORMATIVE_YAW_VAR.
NONINFORMATIVE_YAW_VAR = float(math.pi ** 2)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
    return bool(value)


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
        _declare_if_not('nogo_penalty_type', 'warning_band')
        _declare_if_not('nogo_weight', 0.0)
        _declare_if_not('nogo_safe_distance', 0.35)
        _declare_if_not('nogo_logbarrier_eps', 1e-3)
        _declare_if_not('nogo_warning_band', 0.05)
        _declare_if_not('nogo_near_weight', 50.0)
        _declare_if_not('use_belief_nogo_cost', False)
        _declare_if_not('nogo_belief_kappa', 1.0)
        _declare_if_not('nogo_mode', 'keep_out')
        _declare_if_not('driveable_geometry_json', '')
        _declare_if_not('visibility_artifact_path', '')
        _declare_if_not('robot_collision_radius_m', 0.125)

        # Optimizer params
        _declare_if_not('optimizer_maxiter', 50)
        _declare_if_not('optimizer_maxfun', 500)
        _declare_if_not('optimizer_ftol', 1e-6)
        _declare_if_not('optimizer_gtol', 1e-4)
        _declare_if_not('optimizer_warm_start', True)
        _declare_if_not('optimizer_multistart', False)
        _declare_if_not('optimizer_multistart_include_direct', True)
        _declare_if_not('optimizer_initial_routes_json', '')
        # Route-seed source for the multistart: 'explicit' uses
        # optimizer_initial_routes_json as-is; 'lane_graph' generates condition-
        # neutral lane-centre Manhattan seeds from the driveable map at the (one-shot)
        # global solve. See unav_common.lane_graph_routes.
        _declare_if_not('optimizer_route_seed_mode', 'explicit')
        # Two-stage (global-then-local) hierarchical planning
        _declare_if_not('use_hierarchical', False)
        _declare_if_not('global_horizon', 60)
        # Global-solve step size. 0.0 => use dt; a larger value coarsens the
        # one-shot global route solve (cost is linear in the number of steps),
        # trading discretization fidelity for a faster solve. The global plan is
        # reduced to spatial waypoints, so coarser steps mostly affect route
        # shape resolution, not the tracked path.
        _declare_if_not('global_dt', 0.0)
        _declare_if_not('local_horizon', 12)
        _declare_if_not('local_plan_rate', 4.0)
        _declare_if_not('local_optimizer_maxiter', 60)
        _declare_if_not('global_use_ambiguity', True)
        _declare_if_not('local_use_ambiguity', False)
        # Local executor observation-space goal risk. Default True keeps the locked
        # config behaviour; the belief-loop config sets this False so the local
        # layer is a STATE-space waypoint tracker (the observation-space pixel goal
        # prior is what produced the aisle-transition freeze and the ~0.42 m
        # final-approach stall). Condition-neutral; visibility stays in the global EFE.
        _declare_if_not('local_use_obs_risk', True)
        _declare_if_not('global_optimizer_multistart', True)
        _declare_if_not('local_optimizer_multistart', True)
        _declare_if_not('local_use_visibility_model', False)
        _declare_if_not('local_use_belief_nogo_cost', False)
        _declare_if_not('local_nogo_penalty_type', '')
        _declare_if_not('local_nogo_weight', -1.0)
        _declare_if_not('local_nogo_safe_distance', -1.0)
        _declare_if_not('local_goal_prior_u_std_start', -1.0)
        _declare_if_not('local_goal_prior_v_std_start', -1.0)
        _declare_if_not('local_goal_prior_u_std_final', -1.0)
        _declare_if_not('local_goal_prior_v_std_final', -1.0)
        _declare_if_not('waypoint_spacing_m', 1.0)
        _declare_if_not('waypoint_arrival_radius_m', 0.35)
        _declare_if_not('local_replan_min_remaining_s', 0.0)
        _declare_if_not('local_replan_on_waypoint_change', False)
        _declare_if_not('latency_compensate_plan_handoff', False)
        _declare_if_not('simple_tracker_yaw_gate_rad', 0.6)
        # Which waypoint-tracking law the simple local controller uses. All are
        # "execution plumbing" (track the global plan, no local EFE); they differ
        # only in HOW they track, to avoid the turn-then-go limit-cycle on sharp
        # turns. 'turn_then_go' = legacy. 'hyst_damp' = +hysteresis/damped-w/creep.
        # 'pure_pursuit' = lookahead. 'ff_fb' = path tangent/curvature feedforward
        # + cross-track feedback on belief.
        _declare_if_not('local_controller_type', 'turn_then_go')

        # Pixel correction params
        _declare_if_not('use_pixel_correction', False)
        # Multicam /state/bev belief filter. When False the legacy path hard-resets
        # the belief xy to each raw fused correction (no smoothing, no latency
        # compensation). When True the /state/bev correction is fused through the
        # same predict->EKF-update->predict cycle the single-camera pixel path
        # uses: motion-replay latency compensation to the correction stamp, a
        # covariance-weighted Kalman update, NIS + jump gating.
        _declare_if_not('state_correction_ekf', False)
        # How the multicam measurements reach the filter.
        #   'fused'      -- one pre-fused /state/bev pose per tick (camera_manager
        #                   fuses in map space, then the planner does one update).
        #   'per_camera' -- the per-camera map observations, folded in
        #                   SEQUENTIALLY, each with its own covariance and its own
        #                   gate decision. One filter instead of two in series.
        # Both run the same gate chain, so this is a clean A/B.
        _declare_if_not('state_correction_mode', 'fused')
        _declare_if_not(
            'map_observations_topic', '/reliability/camera_manager/map_observations'
        )
        _declare_if_not('pixel_topic', '/perception/pixel_pose')
        _declare_if_not('cmd_topic', '/cmd_vel')
        _declare_if_not('cmd_publish_rate', 10.0)
        _declare_if_not('pixel_timeout_s', 0.5)
        _declare_if_not('pixel_correction_min_interval_s', 0.0)
        _declare_if_not('bev_y_calibration_offset_m', 0.0)
        _declare_if_not('bev_affine_calibration', '')
        _declare_if_not('pixel_max_correction_jump_m', 0.0)
        # DIAGNOSTIC ONLY: feed ground-truth pose (TF odom->plan_frame of raw
        # /odom) as the planner belief, bypassing perception entirely. Used to
        # isolate the controller from the estimator. MUST be false for any
        # comparison/paper run.
        _declare_if_not('use_truth_localization', False)
        _declare_if_not('truth_odom_topic', '/odom')
        # Chi-squared (2-DOF) innovation gate: a pixel correction whose NIS exceeds
        # this threshold is rejected as a detector outlier. 0.0 = disabled; the
        # campaign sets 9.21 = chi2(2, 0.99).
        _declare_if_not('pixel_correction_nis_threshold', 0.0)
        _declare_if_not('pixel_correction_approx', 'AUTO')
        _declare_if_not('skip_stale_pixel_correction', True)
        _declare_if_not('odom_topic', '/odom_noisy')
        _declare_if_not('use_odom_for_predict', True)
        _declare_if_not('heading_update_mode', 'camera_xy_only')
        # Spawn yaw (map_bev - odom). The single-camera path applies this in
        # pixel_to_bev_state_node (heading = odom_yaw + offset); the multicam
        # path replaces that node, so the planner must apply it itself or the
        # belief heading stays in the raw-odom frame and the robot plans/drives
        # ~90 deg off. Default 0 keeps the single-camera path unchanged.
        _declare_if_not('odom_yaw_offset_rad', 0.0)
        # Divergence guards for the multicam /state/bev EKF. The fused correction
        # is camera-derived + manager-gated (reliable ~0.2 m), so if the predicted
        # belief lands implausibly far from a fresh correction the belief (or a
        # bad-stamp replay) has diverged -> hard re-anchor to the correction rather
        # than NIS-reject it (which locks the belief out of recovery). And cap the
        # motion-replay interval so a single far-future correction stamp cannot
        # jump the prediction tens of metres.
        _declare_if_not('state_reanchor_m', 2.0)
        _declare_if_not('state_max_predict_dt_s', 1.5)
        # Covariance added on a REJECTED /state/bev correction. A rejection must
        # never freeze the belief, so the stamp still advances and S grows -- but
        # the old +1.0 m^2 was large enough to neuter the NIS gate for the next
        # correction (S_y ~ 1.0 makes a 1.87 m innovation score NIS ~3.4 and sail
        # through at gain 0.97, which is how the unguarded jump got in). This is
        # sized so repeated rejections still recover, via the state_reanchor_m
        # guard, without blinding the gate after a single one.
        _declare_if_not('state_reject_inflate_m2', 0.05)
        # Jump limiter for the METRIC (/state/bev, per-camera) path. Default OFF,
        # and deliberately separate from pixel_max_correction_jump_m.
        #
        # NIS and a jump limit fire in OPPOSITE regimes. Confident belief + large
        # innovation -> NIS is huge (rejects, correctly) while the gain is small
        # so the update is small and a jump limit would not fire. Uncertain
        # belief + large innovation -> NIS is moderate (passes) while the gain is
        # ~1 so the update is large and a jump limit DOES fire -- precisely when a
        # large correction is warranted. On this path H = [I2 | 0] is linear, so
        # NIS is the statistically correct gate and a jump limit is redundant at
        # best.
        #
        # Worse, it deadlocks: rejecting inflates S, which RAISES the gain, which
        # RAISES the update, so the limit keeps firing. A belief genuinely 1.5 m
        # off never recovers until drift pushes the innovation past
        # state_reanchor_m -- it has to get worse first. With NIS alone the same
        # case recovers monotonically in ~6 corrections (~1.2 s at 5 Hz).
        #
        # The pixel path keeps its 0.5 m limit: nonlinear observation model, and
        # it is locked paper-1 method backing honest_campaign_v1.
        _declare_if_not('state_max_correction_jump_m', 0.0)
        # Extra measurement uncertainty added IN QUADRATURE to every metric
        # correction: R' = R + sigma^2 * I. This is the unmodelled-error term.
        #
        # conditional_cov_uv (and the fixed metric constant derived from it)
        # models one camera's DETECTOR JITTER. It is structurally silent about
        # inter-camera disagreement -- cameras A and C place the robot at
        # X + d_A and X + d_C with different calibration / contact-height /
        # bbox-bottom errors, and no single pose satisfies both. That term does
        # not exist in a single-camera system, which is why paper 1 ran happily
        # at 2.5 px (measured innovations 1.53 px, NIS median 0.29, 1 rejection
        # in 6370) while the 4-cam fused observation measures ~0.19 m vs GT.
        #
        # Additive rather than a floor: independent error sources add in
        # variance, and a floor would flatten every camera to the same value,
        # destroying the relative weighting a far camera should get vs a near one.
        #
        # 0.0 = off, which keeps the single-camera path exactly as locked. The
        # legacy fused path already had an equivalent via fusion_report_std_m
        # (0.18 m); the per-camera path bypasses that, so it needs this.
        _declare_if_not('state_measurement_inflation_std_m', 0.0)
        # Kinematic plausibility cap on the prediction, m/s. The motion replay
        # extrapolates the last command when odometry is missing (up to
        # state_max_predict_dt_s), inventing up to ~0.9 m of travel. 0 disables.
        # Left at 0 for the single-camera path: the locked campaign ran without
        # it and its evidence must stay reproducible.
        _declare_if_not('max_predict_speed_mps', 0.0)
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
        self.nogo_logbarrier_eps = float(self.get_parameter('nogo_logbarrier_eps').value)
        self.nogo_warning_band = float(self.get_parameter('nogo_warning_band').value)
        self.nogo_near_weight = float(self.get_parameter('nogo_near_weight').value)
        self.use_belief_nogo_cost = _as_bool(self.get_parameter('use_belief_nogo_cost').value)
        self.nogo_belief_kappa = float(self.get_parameter('nogo_belief_kappa').value)
        self.nogo_mode = str(self.get_parameter('nogo_mode').value or 'keep_out').strip().lower()
        self.driveable_geometry_json = str(self.get_parameter('driveable_geometry_json').value or '')
        self.visibility_artifact_path = str(self.get_parameter('visibility_artifact_path').value).strip()
        self.robot_collision_radius_m = float(self.get_parameter('robot_collision_radius_m').value)

        self.optimizer_maxiter = int(self.get_parameter('optimizer_maxiter').value)
        self.optimizer_maxfun = int(self.get_parameter('optimizer_maxfun').value)
        self.optimizer_ftol = float(self.get_parameter('optimizer_ftol').value)
        self.optimizer_gtol = float(self.get_parameter('optimizer_gtol').value)
        self.optimizer_warm_start = _as_bool(self.get_parameter('optimizer_warm_start').value)
        self.optimizer_multistart = _as_bool(self.get_parameter('optimizer_multistart').value)
        self.optimizer_multistart_include_direct = _as_bool(
            self.get_parameter('optimizer_multistart_include_direct').value
        )
        self.optimizer_initial_routes_json = str(
            self.get_parameter('optimizer_initial_routes_json').value
        )
        self.optimizer_route_seed_mode = str(
            self.get_parameter('optimizer_route_seed_mode').value or 'explicit'
        )
        self.use_hierarchical = _as_bool(self.get_parameter('use_hierarchical').value)
        self.global_horizon = int(self.get_parameter('global_horizon').value)
        _global_dt = float(self.get_parameter('global_dt').value)
        self.global_dt = _global_dt if _global_dt > 0.0 else float(self.get_parameter('dt').value)
        self.local_horizon = int(self.get_parameter('local_horizon').value)
        self.local_plan_rate = float(self.get_parameter('local_plan_rate').value)
        self.local_optimizer_maxiter = int(self.get_parameter('local_optimizer_maxiter').value)
        self.global_use_ambiguity = _as_bool(self.get_parameter('global_use_ambiguity').value)
        self.local_use_ambiguity = _as_bool(self.get_parameter('local_use_ambiguity').value)
        self.local_use_obs_risk = _as_bool(self.get_parameter('local_use_obs_risk').value)
        self.global_optimizer_multistart = _as_bool(
            self.get_parameter('global_optimizer_multistart').value
        )
        self.local_optimizer_multistart = _as_bool(
            self.get_parameter('local_optimizer_multistart').value
        )
        self.local_use_visibility_model = _as_bool(
            self.get_parameter('local_use_visibility_model').value
        )
        self.local_use_belief_nogo_cost = _as_bool(
            self.get_parameter('local_use_belief_nogo_cost').value
        )
        self.local_nogo_penalty_type = str(
            self.get_parameter('local_nogo_penalty_type').value or ''
        ).strip().lower()
        self.local_nogo_weight = float(self.get_parameter('local_nogo_weight').value)
        self.local_nogo_safe_distance = float(
            self.get_parameter('local_nogo_safe_distance').value
        )
        self.local_goal_prior_u_std_start = float(
            self.get_parameter('local_goal_prior_u_std_start').value
        )
        self.local_goal_prior_v_std_start = float(
            self.get_parameter('local_goal_prior_v_std_start').value
        )
        self.local_goal_prior_u_std_final = float(
            self.get_parameter('local_goal_prior_u_std_final').value
        )
        self.local_goal_prior_v_std_final = float(
            self.get_parameter('local_goal_prior_v_std_final').value
        )
        self.waypoint_spacing_m = float(self.get_parameter('waypoint_spacing_m').value)
        self.waypoint_arrival_radius_m = float(self.get_parameter('waypoint_arrival_radius_m').value)
        self.local_replan_min_remaining_s = max(
            0.0, float(self.get_parameter('local_replan_min_remaining_s').value)
        )
        self.local_replan_on_waypoint_change = _as_bool(
            self.get_parameter('local_replan_on_waypoint_change').value
        )
        self.latency_compensate_plan_handoff = _as_bool(
            self.get_parameter('latency_compensate_plan_handoff').value
        )
        self.simple_tracker_yaw_gate_rad = max(
            0.0, float(self.get_parameter('simple_tracker_yaw_gate_rad').value)
        )
        self.local_controller_type = str(
            self.get_parameter('local_controller_type').value
        ).strip().lower()

        self.use_pixel_correction = _as_bool(self.get_parameter('use_pixel_correction').value)
        self.state_correction_ekf = _as_bool(self.get_parameter('state_correction_ekf').value)
        self.state_correction_mode = str(
            self.get_parameter('state_correction_mode').value
        ).strip().lower()
        if self.state_correction_mode not in ('fused', 'per_camera'):
            raise RuntimeError(
                "state_correction_mode must be 'fused' or 'per_camera', "
                f"got {self.state_correction_mode!r}"
            )
        self.map_observations_topic = str(self.get_parameter('map_observations_topic').value)
        self.pixel_topic = self.get_parameter('pixel_topic').value
        self.cmd_topic = str(self.get_parameter('cmd_topic').value).strip() or '/cmd_vel'
        self.cmd_publish_rate = max(0.1, float(self.get_parameter('cmd_publish_rate').value))
        self.pixel_timeout_s = float(self.get_parameter('pixel_timeout_s').value)
        self.pixel_correction_min_interval_s = float(
            self.get_parameter('pixel_correction_min_interval_s').value
        )
        self.bev_y_calibration_offset_m = float(
            self.get_parameter('bev_y_calibration_offset_m').value
        )
        # Position-dependent affine BEV calibration (6 coeffs). When set it is the
        # single calibration used by the pixel correction, matching the state node
        # and experiment logger; the constant y-offset is only a legacy fallback.
        self._bev_affine = None
        _affine_raw = str(self.get_parameter('bev_affine_calibration').value or '').strip()
        if _affine_raw:
            try:
                _vals = [float(x) for x in _affine_raw.replace(';', ',').split(',') if x.strip() != '']
                if len(_vals) == 6:
                    self._bev_affine = _vals
            except Exception:
                self._bev_affine = None
        self.pixel_max_correction_jump_m = float(
            self.get_parameter('pixel_max_correction_jump_m').value
        )
        self.use_truth_localization = _as_bool(
            self.get_parameter('use_truth_localization').value
        )
        self.truth_odom_topic = str(self.get_parameter('truth_odom_topic').value)
        self.pixel_correction_nis_threshold = float(
            self.get_parameter('pixel_correction_nis_threshold').value
        )
        self.pixel_correction_approx = str(
            self.get_parameter('pixel_correction_approx').value
        ).strip().upper()
        if self.pixel_correction_approx not in ('AUTO', 'ET1', 'ET2', 'UT'):
            raise RuntimeError("pixel_correction_approx must be one of: AUTO, ET1, ET2, UT")
        self.skip_stale_pixel_correction = _as_bool(
            self.get_parameter('skip_stale_pixel_correction').value
        )
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.use_odom_for_predict = _as_bool(self.get_parameter('use_odom_for_predict').value)
        self.odom_yaw_offset_rad = float(self.get_parameter('odom_yaw_offset_rad').value)
        self.state_reanchor_m = float(self.get_parameter('state_reanchor_m').value)
        self.state_max_predict_dt_s = float(self.get_parameter('state_max_predict_dt_s').value)
        self.state_reject_inflate_m2 = max(
            float(self.get_parameter('state_reject_inflate_m2').value), 0.0
        )
        self.state_max_correction_jump_m = max(
            float(self.get_parameter('state_max_correction_jump_m').value), 0.0
        )
        self.state_measurement_inflation_m2 = max(
            float(self.get_parameter('state_measurement_inflation_std_m').value), 0.0
        ) ** 2
        self.max_predict_speed_mps = max(
            float(self.get_parameter('max_predict_speed_mps').value), 0.0
        )
        self._latest_odom_yaw = None
        self.heading_update_mode = str(self.get_parameter('heading_update_mode').value).strip().lower()
        if self.heading_update_mode != 'camera_xy_only':
            raise RuntimeError("heading_update_mode must be 'camera_xy_only' for current active runs")
        self.min_state_cov = float(self.get_parameter('min_state_cov').value)
        self.cov_eig_floor = 1e-9
        self._heading_anchor_applied = False
        self._state_bev_yaw_ignored = False
        self._latest_prediction_source = 0.0
        self._latest_prediction_dt = 0.0
        self._latest_u_pred_v = 0.0
        self._latest_u_pred_omega = 0.0
        self._latest_Q_theta_theta = 0.0
        self._latest_odom_delta_theta = 0.0
        self._latest_cmd_delta_theta = 0.0
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
        warm_start_shift_steps = self._warm_start_shift_steps_for_rate(self.plan_rate)

        self._camera_params = camera_params
        self._warm_start_shift_steps = warm_start_shift_steps
        self.optimizer_warm_start_shift_steps = warm_start_shift_steps
        self.planner = self._construct_planner()
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
        self.map_observations_sub = None
        if self.state_correction_ekf and self.state_correction_mode == 'per_camera':
            self.map_observations_sub = self.create_subscription(
                String, self.map_observations_topic, self._map_observations_cb,
                qos_profile=state_qos, callback_group=self._io_group
            )
            self.get_logger().info(
                "state_correction_mode=per_camera: folding per-camera map "
                f"observations from {self.map_observations_topic} into the belief "
                "sequentially (one filter, N updates)"
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
            Odometry, self.odom_topic, self._odom_cb, qos_profile=state_qos,
            callback_group=self._io_group
        )

        # DIAGNOSTIC: ground-truth localization path (TF odom->plan_frame of raw odom).
        self.truth_pose = None  # (x, y, yaw) in plan frame
        self.truth_pose_stamp = None
        self._tf_buffer = None
        self._tf_listener = None
        self.truth_odom_sub = None
        if self.use_truth_localization:
            import tf2_ros
            from tf2_geometry_msgs import do_transform_pose  # noqa: F401  (registers PoseStamped)
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            self._do_transform_pose = do_transform_pose
            self.truth_odom_sub = self.create_subscription(
                Odometry, self.truth_odom_topic, self._truth_odom_cb,
                qos_profile=state_qos, callback_group=self._io_group,
            )
            self.get_logger().warn(
                "*** use_truth_localization=TRUE — planner belief is GROUND TRUTH "
                "(perception bypassed). DIAGNOSTIC ONLY, not valid for comparison runs. ***"
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
        self.odom_vel = np.array([0.0, 0.0], dtype=float)
        # Ring buffers of timestamped motion inputs for dead reckoning.  The
        # odometry log is preferred when use_odom_for_predict=True because it
        # represents the encoder/noisy-odometry estimate of what the robot did.
        # The command log remains as a fallback and for diagnostics.
        self._cmd_log: list[tuple[float, float, float]] = []
        self._odom_log: list[tuple[float, float, float]] = []
        self._CMD_LOG_MAX_S: float = 60.0
        self._latest_measurement_available = False
        self._latest_belief_age_s = math.nan

        planner_rate = self.local_plan_rate if self.use_hierarchical else self.plan_rate
        self._plan_period_s = 1.0 / max(planner_rate, 0.1)
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
            f"use_belief_nogo_cost={self.use_belief_nogo_cost}, "
            f"use_pixel_correction={self.use_pixel_correction}, "
            f"cmd_topic={self.cmd_topic}, "
            f"pixel_correction_approx={self.pixel_correction_approx}, "
            f"heading_update_mode={self.heading_update_mode}, "
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
        # EKF mode: fold each fused correction into the belief on arrival (the
        # same architecture the single-camera pixel path uses -- corrections are
        # applied here, the planning loop only predicts the committed belief
        # forward). The legacy hard-reset path leaves the belief update to
        # _resolve_state_belief_for_planning instead.
        # In per_camera mode the belief is corrected from the per-camera
        # observations instead; applying the fused pose too would fold the same
        # measurements in twice.
        if self.state_correction_ekf and self.state_correction_mode == 'fused':
            try:
                self._apply_state_correction(msg)
            except Exception as exc:
                self.get_logger().warn(f"state correction update failed: {exc}")

    def _map_observations_cb(self, msg: String):
        if not (self.state_correction_ekf and self.state_correction_mode == 'per_camera'):
            return
        try:
            observations, _frame_id = map_observations_from_json(msg.data)
        except Exception as exc:
            self._warn_stale_pixel_once(f"rejected map-observation batch: {exc}")
            return
        try:
            self._apply_map_observations(observations)
        except Exception as exc:
            self.get_logger().warn(f"per-camera correction update failed: {exc}")

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

    def _warm_start_shift_steps_for_rate(self, plan_rate: float) -> int:
        return max(
            1,
            int(round((1.0 / max(float(plan_rate), 0.1)) / max(self.dt, 1e-3))),
        )

    def _construct_planner(self, **ov):
        """Build a planner from the node's params, with optional overrides.

        Overridable (for the two-stage global/local planners): horizon,
        visibility/no-go switches, goal tightening, optimizer settings, and
        warm-start shift. Keeping these as real overrides prevents hidden
        local/global planner behavior from diverging from the manifest.
        """
        def g(key):
            return ov[key] if key in ov else getattr(self, key)

        def g_default(key, default):
            # For optional planner kwargs that have no node-level attribute: use
            # the override if provided, otherwise the supplied default. The global
            # planner omits these so the LOCAL reference-tracking terms stay OFF
            # (0.0) for it, leaving the global EFE objective unchanged.
            return ov[key] if key in ov else default

        return self.PLANNER_CLASS(
            horizon=int(g('horizon')),
            dt=float(g_default('dt', self.dt)), v_min=self.v_min, v_max=self.v_max, w_min=self.w_min, w_max=self.w_max,
            control_weight=self.control_weight,
            process_noise_xy=self.process_noise_xy, process_noise_theta=self.process_noise_theta,
            obs_noise_uv=self.obs_noise_uv, goal_sigma_uv=self.goal_sigma_uv,
            risk_weight_obs=self.risk_weight_obs, ambiguity_weight=self.ambiguity_weight,
            optimizer_maxiter=int(g('optimizer_maxiter')), optimizer_maxfun=int(g('optimizer_maxfun')),
            optimizer_ftol=float(g('optimizer_ftol')), optimizer_gtol=float(g('optimizer_gtol')),
            optimizer_warm_start=_as_bool(g('optimizer_warm_start')),
            optimizer_warm_start_shift_steps=int(g('optimizer_warm_start_shift_steps')),
            optimizer_multistart=_as_bool(g('optimizer_multistart')),
            optimizer_multistart_include_direct=_as_bool(g('optimizer_multistart_include_direct')),
            optimizer_initial_routes_json=g('optimizer_initial_routes_json'),
            approx_method=self.approx_method, use_obs_risk=_as_bool(g('use_obs_risk')),
            use_ambiguity=_as_bool(g('use_ambiguity')), seed=self.seed, camera_params=self._camera_params,
            use_visibility_model=_as_bool(g('use_visibility_model')),
            visibility_target_height_m=self.visibility_target_height_m,
            visibility_geometry_json=self.visibility_geometry_json,
            collision_geometry_json=self.collision_geometry_json,
            visibility_artifact_path=self.visibility_artifact_path,
            r_visible_uv=self.r_visible_uv, r_miss_uv=self.r_miss_uv,
            visibility_sigma_kappa=self.visibility_sigma_kappa,
            goal_prior_u_std_start=g('goal_prior_u_std_start'),
            goal_prior_v_std_start=g('goal_prior_v_std_start'),
            goal_prior_u_std_final=g('goal_prior_u_std_final'),
            goal_prior_v_std_final=g('goal_prior_v_std_final'),
            goal_tightening_power=g('goal_tightening_power'),
            goal_progress_n_steps=int(g('goal_progress_n_steps')),
            observation_risk_scale=float(g('observation_risk_scale')),
            ambiguity_term_scale=float(g('ambiguity_term_scale')), discount_gamma=float(g('discount_gamma')),
            use_nogo_cost=_as_bool(g('use_nogo_cost')), nogo_penalty_type=str(g('nogo_penalty_type')),
            nogo_weight=float(g('nogo_weight')), nogo_safe_distance=float(g('nogo_safe_distance')),
            nogo_logbarrier_eps=float(g('nogo_logbarrier_eps')),
            nogo_warning_band=float(g('nogo_warning_band')),
            nogo_near_weight=float(g('nogo_near_weight')),
            use_belief_nogo_cost=_as_bool(g('use_belief_nogo_cost')),
            nogo_belief_kappa=float(g('nogo_belief_kappa')),
            nogo_mode=str(g('nogo_mode')), driveable_geometry_json=g('driveable_geometry_json'),
            robot_collision_radius_m=self.robot_collision_radius_m, runtime_debug=self.debug_runtime,
        )

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
        now_s = self.get_clock().now().nanoseconds * 1e-9
        with self._data_lock:
            self.last_cmd = np.array([msg.linear.x, msg.angular.z], dtype=float)
            # Log the intended pre-noise command with its sim timestamp.
            self._cmd_log.append((now_s, msg.linear.x, msg.angular.z))
            # Trim entries older than the ring-buffer horizon.
            cutoff = now_s - self._CMD_LOG_MAX_S
            while self._cmd_log and self._cmd_log[0][0] < cutoff:
                self._cmd_log.pop(0)

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _odom_cb(self, msg: Odometry):
        yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        v_odom = float(msg.twist.twist.linear.x)
        w_odom = float(msg.twist.twist.angular.z)
        self._latest_odom_yaw = float(yaw)
        try:
            stamp_s = self._stamp_to_float(msg.header.stamp)
        except (AttributeError, TypeError, ValueError):
            stamp_s = self.get_clock().now().nanoseconds * 1e-9
        with self._data_lock:
            self.odom_vel = np.array([v_odom, w_odom], dtype=float)
            self._odom_log.append((stamp_s, v_odom, w_odom))
            cutoff = stamp_s - self._CMD_LOG_MAX_S
            while self._odom_log and self._odom_log[0][0] < cutoff:
                self._odom_log.pop(0)

    def _truth_odom_cb(self, msg: Odometry):
        """DIAGNOSTIC: transform raw odom (truth) into the plan frame via TF."""
        source_frame = (msg.header.frame_id or 'odom').strip() or 'odom'
        plan_frame = self._resolve_plan_frame_id()
        pose_world = msg.pose.pose
        if source_frame != plan_frame:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    plan_frame, source_frame, Time())
                pose_world = self._do_transform_pose(msg.pose.pose, tf_msg)
            except Exception:
                return
        x = float(pose_world.position.x)
        y = float(pose_world.position.y)
        yaw = self._yaw_from_quaternion(pose_world.orientation)
        with self._data_lock:
            self.truth_pose = (x, y, yaw)
            self.truth_pose_stamp = msg.header.stamp

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
        R_xy = self._xy_covariance_from_pose(cov)
        S = np.diag([0.0, 0.0, cov[35] if len(cov) > 35 else 1e-6]).astype(float)
        S[:2, :2] = R_xy
        return m, self._regularize_state_covariance(S)

    def _xy_covariance_from_pose(self, cov, *, floor=0.0):
        """Full 2x2 xy block of a ROS pose covariance, cross terms included.

        camera_manager publishes map-frame CROSS-covariance at indices 1 and 6
        under the anisotropic covariance profiles: pixel noise projected through
        an oblique camera is a long thin ellipse, not a circle, and its axes are
        not aligned with map x/y. Reading only the diagonal silently discards
        that -- the estimate is then weighted as if the error were isotropic,
        which is the very thing the profile exists to fix.

        Falls back to the diagonal if the symmetrized block is not positive
        definite, so a malformed message degrades loudly-but-safely rather than
        poisoning the filter.
        """
        vxx = max(float(cov[0]) if len(cov) > 0 else 0.0, floor)
        vyy = max(float(cov[7]) if len(cov) > 7 else 0.0, floor)
        vxy = 0.0
        if len(cov) > 6:
            vxy = 0.5 * (float(cov[1]) + float(cov[6]))
        R = np.array([[vxx, vxy], [vxy, vyy]], dtype=float)
        if not np.all(np.isfinite(R)):
            return np.diag([vxx, vyy]).astype(float)
        if vxy != 0.0 and np.min(np.linalg.eigvalsh(R)) <= 0.0:
            self._warn_stale_pixel_once(
                "pose covariance xy block is not positive definite "
                f"(xx={vxx:.4g}, xy={vxy:.4g}, yy={vyy:.4g}); using its diagonal"
            )
            return np.diag([vxx, vyy]).astype(float)
        return R

    def _state_msg_age_s(self, state_ref: PoseWithCovarianceStamped) -> float:
        try:
            return (self.get_clock().now() - Time.from_msg(state_ref.header.stamp)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return math.inf

    def _state_msg_is_fresh(self, state_ref: PoseWithCovarianceStamped) -> bool:
        age = self._state_msg_age_s(state_ref)
        future_tolerance_s = max(float(self.pixel_timeout_s), 0.25)
        return bool(math.isfinite(age) and age <= float(self.pixel_timeout_s) and age >= -future_tolerance_s)

    def _regularize_state_covariance(self, S):
        """Keep planner belief covariance positive enough for stable updates."""
        return bc.regularize_covariance(S, self.min_state_cov)

    def _init_belief_from_state(self):
        with self._data_lock:
            if self.state_msg is None:
                return False
            if self.skip_stale_pixel_correction and not self._state_msg_is_fresh(self.state_msg):
                age = self._state_msg_age_s(self.state_msg)
                self._warn_stale_pixel_once(
                    f"Refusing to initialize planner belief from stale /state/bev (age {age:.2f}s)"
                )
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

    def _pixel_cb(self, msg: PoseStamped):
        u = msg.pose.position.x
        v = msg.pose.position.y
        # Apply the same BEV calibration the state node and logger use so all
        # nodes converge to the same world position from the same pixel: the
        # position-dependent affine when configured, else the legacy constant
        # y-offset. Re-project the calibrated world point back to a pixel.
        if self._bev_affine is not None or self.bev_y_calibration_offset_m != 0.0:
            try:
                camera = self.planner.camera
                xy = camera.pixel_to_world(u, v)
                if xy is not None:
                    if self._bev_affine is not None:
                        c = self._bev_affine
                        x_cal = c[0] * xy[0] + c[1] * xy[1] + c[2]
                        y_cal = c[3] * xy[0] + c[4] * xy[1] + c[5]
                    else:
                        x_cal, y_cal = xy[0], xy[1] + self.bev_y_calibration_offset_m
                    u_cal, v_cal, vis = camera.world_to_pixel(x_cal, y_cal, 0.0)
                    if vis:
                        u, v = u_cal, v_cal
            except Exception:
                pass
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

    def _stamp_age_s(self, stamp_msg) -> float:
        try:
            return (self.get_clock().now() - Time.from_msg(stamp_msg)).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            return 0.0

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

    def _replay_cmd_log_interval(self, m0, S0, from_stamp, to_stamp,
                                   fallback_cmd, fallback_dt):
        """Predict (m0, S0) from from_stamp to to_stamp using motion replay.

        When ``use_odom_for_predict`` is enabled, replay the configured odometry
        topic first (normally ``/odom_noisy``).  This is the paper-facing
        dead-reckoning path: camera updates correct a belief propagated by
        onboard odometry, not by the ideal command request.  If odometry samples
        are unavailable, fall back to command replay and finally to a single
        fallback prediction.
        """
        try:
            from_s = Time.from_msg(from_stamp).nanoseconds * 1e-9
            to_s   = Time.from_msg(to_stamp).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            m0, S0 = self.planner.predict(m0, S0, fallback_cmd, dt=fallback_dt)
            return m0, S0, {
                'cmd_replay_count': 0.0,
                'cmd_replay_duration_s': float(fallback_dt),
                'cmd_replay_used_fallback': 1.0,
                'motion_replay_source_code': 3.0,
            }

        if to_s <= from_s:
            return m0, S0, {
                'cmd_replay_count': 0.0,
                'cmd_replay_duration_s': 0.0,
                'cmd_replay_used_fallback': 0.0,
                'motion_replay_source_code': 0.0,
            }

        with self._data_lock:
            odom_entries = list(self._odom_log)
            cmd_entries = list(self._cmd_log)
        entries = odom_entries if self.use_odom_for_predict else cmd_entries
        source_code = 1.0 if self.use_odom_for_predict else 2.0
        previous = [(t, v, w) for t, v, w in entries if t <= from_s]
        relevant = [(t, v, w) for t, v, w in entries if from_s < t <= to_s]
        if self.use_odom_for_predict and not previous and not relevant:
            entries = cmd_entries
            source_code = 2.0
            previous = [(t, v, w) for t, v, w in entries if t <= from_s]
            relevant = [(t, v, w) for t, v, w in entries if from_s < t <= to_s]

        if previous:
            current_cmd = np.array([previous[-1][1], previous[-1][2]], dtype=float)
            used_fallback = 0.0
        elif relevant:
            # No command at the measurement stamp.  Treat the robot as
            # stationary until the first later command; this avoids applying a
            # future command to the past during delayed-measurement replay.
            current_cmd = np.array([0.0, 0.0], dtype=float)
            used_fallback = 0.0
        else:
            m0, S0 = self.planner.predict(m0, S0, fallback_cmd, dt=fallback_dt)
            return m0, S0, {
                'cmd_replay_count': 0.0,
                'cmd_replay_duration_s': float(fallback_dt),
                'cmd_replay_used_fallback': 1.0,
                'motion_replay_source_code': 3.0,
            }

        prev_t = from_s
        for t, v, w in relevant:
            dt_gap = t - prev_t
            if dt_gap > 1e-4:
                m0, S0 = self.planner.predict(
                    m0, S0, current_cmd, dt=dt_gap)
            current_cmd = np.array([v, w], dtype=float)
            prev_t = t
        dt_tail = to_s - prev_t
        if dt_tail > 1e-4:
            m0, S0 = self.planner.predict(
                m0, S0, current_cmd, dt=dt_tail)
        return m0, S0, {
            'cmd_replay_count': float(len(relevant)),
            'cmd_replay_duration_s': float(max(to_s - from_s, 0.0)),
            'cmd_replay_used_fallback': float(used_fallback),
            'motion_replay_source_code': float(source_code),
        }

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
                f"(max {max_dt_s:.2f}s); resetting belief from state."
            )
            self._init_belief_from_state()
            return None
        return float(dt_s)

    def _snapshot_pixel_correction_inputs(self, stamp_msg):
        with self._data_lock:
            belief_m = None if self.belief_m is None else self.belief_m.copy()
            belief_S = None if self.belief_S is None else self.belief_S.copy()
            belief_stamp = self.belief_stamp
            v_cmd, w_cmd = float(self.last_cmd[0]), float(self.last_cmd[1])
            meas = None if self.pixel_meas is None else self.pixel_meas.copy()
        if belief_m is None or belief_S is None or meas is None:
            return None
        return bc.CorrectionSnapshot(
            belief_m=belief_m,
            belief_S=belief_S,
            belief_stamp=belief_stamp,
            cmd=np.array([v_cmd, w_cmd], dtype=float),
            meas=meas,
            meas_stamp=stamp_msg,
            yaw_meas=None,
            yaw_sigma=math.nan,
            yaw_source=0.0,
        )

    def _correction_gates(self, *, metric_measurement: bool = False) -> bc.CorrectionGates:
        """Thresholds for the shared gate chain, from the node's parameters.

        ``metric_measurement`` enables the divergence guard, whose threshold is
        in metres and so only means anything when the measurement is a map
        position (the fused /state/bev path), not pixels.
        """
        return bc.CorrectionGates(
            pixel_timeout_s=float(self.pixel_timeout_s),
            dt_nominal_s=float(self.dt),
            skip_stale=bool(self.skip_stale_pixel_correction),
            max_jump_m=float(
                self.state_max_correction_jump_m if metric_measurement
                else self.pixel_max_correction_jump_m
            ),
            nis_threshold=float(self.pixel_correction_nis_threshold),
            cov_eig_floor=float(self.cov_eig_floor),
            min_state_cov=float(self.min_state_cov),
            reanchor_innov_m=float(self.state_reanchor_m) if metric_measurement else 0.0,
            max_predict_speed_mps=float(self.max_predict_speed_mps),
        )

    def _log_pixel_shape_error_once(self, message: str):
        now_wall = time.monotonic()
        if now_wall - self._last_shape_mismatch_log > 2.0:
            self.get_logger().error(message)
            self._last_shape_mismatch_log = now_wall

    project_to_psd = staticmethod(bc.project_to_psd)

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
        nis=float('nan'),
        accepted=True,
        reject_reason_code=0.0,
        apply_stamp_s=math.nan,
        belief_input_stamp_s=math.nan,
        cmd_replay_count=math.nan,
        cmd_replay_duration_s=math.nan,
        cmd_replay_used_fallback=math.nan,
        motion_replay_source_code=math.nan,
        nis_threshold=math.nan,
        K_theta_u=math.nan,
        K_theta_v=math.nan,
        measurement_space=bc.SPACE_PIXEL_UV,
        predict_clipped_m=0.0,
        camera_index=math.nan,
    ):
        diag_msg = Float64MultiArray()
        r_eff = np.asarray(R_eff, dtype=float)
        if not math.isfinite(float(apply_stamp_s)):
            apply_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        expected_after_u = math.nan
        expected_after_v = math.nan
        expected_after_visible = math.nan
        try:
            expected_after_u, expected_after_v, visible = self.planner.camera.world_to_pixel(
                float(next_m[0]),
                float(next_m[1]),
                0.0,
            )
            expected_after_visible = 1.0 if bool(visible) else 0.0
        except Exception:
            expected_after_u = math.nan
            expected_after_v = math.nan
            expected_after_visible = math.nan
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
            float(nis) if math.isfinite(float(nis)) else math.nan,
            1.0 if accepted else 0.0,
            float(reject_reason_code),
            float(apply_stamp_s),
            float(belief_input_stamp_s),
            float(cmd_replay_count),
            float(cmd_replay_duration_s),
            float(cmd_replay_used_fallback),
            float(nis_threshold),
            float(expected_after_u) if math.isfinite(float(expected_after_u)) else math.nan,
            float(expected_after_v) if math.isfinite(float(expected_after_v)) else math.nan,
            float(expected_after_visible) if math.isfinite(float(expected_after_visible)) else math.nan,
            float(motion_replay_source_code) if math.isfinite(float(motion_replay_source_code)) else math.nan,
            float(K_theta_u) if math.isfinite(float(K_theta_u)) else math.nan,
            float(K_theta_v) if math.isfinite(float(K_theta_v)) else math.nan,
            # Appended 2026-07-29 with the single/multicam chain consolidation.
            # The pixel_corr_* columns are now shared by both stacks, so a reader
            # needs to know which one produced the row: innov/meas are pixels
            # when 0, metres when 1.
            float(measurement_space),
            float(predict_clipped_m),
            # Which camera produced this correction in per_camera mode (index into
            # the batch, sorted by stamp); NaN for the fused/pixel paths.
            float(camera_index),
        ]
        self.pixel_correction_diag_pub.publish(diag_msg)

    _pixel_correction_reject_code = staticmethod(bc.reject_code)

    def _publish_pixel_correction_rejection(
        self,
        stamp_msg,
        *,
        reason: str,
        age=math.nan,
        dt_s=math.nan,
        m_pred=None,
        meas=None,
        mu_y=None,
        innov=None,
        xy_update_norm_m=math.nan,
        R_eff=None,
        nis=math.nan,
        belief_input_stamp_s=math.nan,
        cmd_replay_count=math.nan,
        cmd_replay_duration_s=math.nan,
        cmd_replay_used_fallback=math.nan,
        motion_replay_source_code=math.nan,
        measurement_space=bc.SPACE_PIXEL_UV,
        predict_clipped_m=0.0,
        camera_index=math.nan,
    ):
        nan_state = np.array([math.nan, math.nan, math.nan], dtype=float)
        nan_meas = np.array([math.nan, math.nan], dtype=float)
        yaw_info = {
            'theta_update_from_uv_rad': math.nan,
            'yaw_correction_applied': False,
            'innov_theta': math.nan,
            'k_theta_theta': math.nan,
            'theta_update_total_rad': math.nan,
        }
        self._publish_pixel_correction_diagnostics(
            stamp_msg=stamp_msg,
            age=age,
            dt_s=dt_s,
            p_vis=math.nan,
            gain_scale=math.nan,
            innov=np.asarray(innov if innov is not None else nan_meas, dtype=float),
            xy_update_norm_m=xy_update_norm_m,
            yaw_info=yaw_info,
            m_pred=np.asarray(m_pred if m_pred is not None else nan_state, dtype=float),
            next_m=nan_state,
            meas=np.asarray(meas if meas is not None else nan_meas, dtype=float),
            mu_y=np.asarray(mu_y if mu_y is not None else nan_meas, dtype=float),
            R_eff=np.asarray(R_eff if R_eff is not None else np.full((2, 2), math.nan), dtype=float),
            yaw_meas=math.nan,
            yaw_sigma=math.nan,
            yaw_source=0.0,
            nis=nis,
            accepted=False,
            reject_reason_code=self._pixel_correction_reject_code(reason),
            apply_stamp_s=float(self.get_clock().now().nanoseconds) * 1e-9,
            belief_input_stamp_s=belief_input_stamp_s,
            cmd_replay_count=cmd_replay_count,
            cmd_replay_duration_s=cmd_replay_duration_s,
            cmd_replay_used_fallback=cmd_replay_used_fallback,
            motion_replay_source_code=motion_replay_source_code,
            nis_threshold=float(self.pixel_correction_nis_threshold),
            measurement_space=measurement_space,
            predict_clipped_m=predict_clipped_m,
            camera_index=camera_index,
        )

    def _correction_reject_warning(self, outcome) -> str | None:
        """Operator-facing text for a rejection, or None if it warns elsewhere."""
        if outcome.reason is bc.RejectReason.JUMP_TOO_LARGE:
            return (
                f"Pixel correction jump {outcome.xy_update_norm_m:.3f} m exceeds "
                f"limit {self.pixel_max_correction_jump_m:.3f} m; rejecting"
            )
        if outcome.reason is bc.RejectReason.NIS_TOO_LARGE:
            return (
                f"Pixel correction NIS {outcome.nis:.2f} exceeds "
                f"threshold {self.pixel_correction_nis_threshold:.2f}; rejecting"
            )
        return None

    def _publish_correction_outcome(self, stamp_msg, outcome, *, camera_index=math.nan):
        """Publish diagnostics for one :class:`bc.CorrectionOutcome`."""
        snapshot = outcome.snapshot
        belief_input_stamp_s = (
            self._stamp_to_float(snapshot.belief_stamp) if snapshot is not None else math.nan
        )
        meta = outcome.replay_meta or {}
        replay_kwargs = {
            'belief_input_stamp_s': belief_input_stamp_s,
            'cmd_replay_count': float(meta.get('cmd_replay_count', math.nan)),
            'cmd_replay_duration_s': float(meta.get('cmd_replay_duration_s', math.nan)),
            'cmd_replay_used_fallback': float(meta.get('cmd_replay_used_fallback', math.nan)),
            'motion_replay_source_code': float(meta.get('motion_replay_source_code', math.nan)),
            'measurement_space': float(outcome.measurement_space),
            'predict_clipped_m': float(outcome.predict_clipped_m),
            'camera_index': float(camera_index),
        }
        if not outcome.accepted:
            self._publish_pixel_correction_rejection(
                stamp_msg,
                reason=outcome.reason,
                age=outcome.age,
                dt_s=outcome.dt_s,
                m_pred=outcome.m_pred,
                meas=outcome.meas,
                mu_y=outcome.mu_y,
                innov=outcome.innov,
                xy_update_norm_m=outcome.xy_update_norm_m,
                R_eff=outcome.R_eff,
                nis=outcome.nis,
                **replay_kwargs,
            )
            return

        K_theta_u = math.nan
        K_theta_v = math.nan
        K_mat = outcome.K
        if K_mat is not None and K_mat.shape[0] >= 3:
            if K_mat.shape[1] >= 1:
                K_theta_u = float(K_mat[2, 0])
            if K_mat.shape[1] >= 2:
                K_theta_v = float(K_mat[2, 1])

        self._publish_pixel_correction_diagnostics(
            stamp_msg=stamp_msg,
            age=outcome.age,
            dt_s=outcome.dt_s,
            p_vis=outcome.p_vis,
            gain_scale=outcome.gain_scale,
            innov=outcome.innov,
            xy_update_norm_m=outcome.xy_update_norm_m,
            yaw_info=outcome.yaw_info,
            m_pred=outcome.m_pred,
            next_m=outcome.next_m,
            meas=outcome.meas,
            mu_y=outcome.mu_y,
            R_eff=outcome.R_eff,
            yaw_meas=snapshot.yaw_meas if snapshot is not None else math.nan,
            yaw_sigma=snapshot.yaw_sigma if snapshot is not None else math.nan,
            yaw_source=snapshot.yaw_source if snapshot is not None else 0.0,
            nis=outcome.nis,
            accepted=True,
            reject_reason_code=bc.ACCEPTED_CODE,
            apply_stamp_s=float(self.get_clock().now().nanoseconds) * 1e-9,
            nis_threshold=float(self.pixel_correction_nis_threshold),
            K_theta_u=K_theta_u,
            K_theta_v=K_theta_v,
            **replay_kwargs,
        )

    def _apply_pixel_correction(self, stamp_msg, *, source='callback'):
        cb_start = time.perf_counter()
        age = self._stamp_age_s(stamp_msg)
        gates = self._correction_gates()

        # The two gates below carry node-side effects (a warn-once, a belief
        # re-init) so they stay here; the decision itself is the shared chain's.
        if gates.age_is_invalid(age):
            self._warn_stale_pixel_once(
                f"Skipping time-inconsistent pixel measurement (age {age:.2f}s)"
            )
            self._publish_correction_outcome(
                stamp_msg, bc.CorrectionOutcome(reason=bc.RejectReason.STALE_AGE, age=age)
            )
            return
        if self._pixel_correction_is_throttled(stamp_msg):
            return

        with self._data_lock:
            has_belief = self.belief_m is not None and self.belief_S is not None
        if not has_belief and not self._init_belief_from_state():
            return

        # Warns and re-initialises the belief from /state/bev on an implausible dt.
        dt_s = self._pixel_correction_dt_s(stamp_msg)
        if dt_s is None:
            self._publish_correction_outcome(
                stamp_msg, bc.CorrectionOutcome(reason=bc.RejectReason.DT_IMPLAUSIBLE, age=age)
            )
            return

        corr_method = self.approx_method if self.pixel_correction_approx == 'AUTO' else self.pixel_correction_approx
        outcome = bc.apply_correction(
            source=bc.PixelMeasurementSource(
                planner=self.planner,
                planner_for_obs=getattr(self, 'global_planner', None) or self.planner,
                snapshot_fn=lambda: self._snapshot_pixel_correction_inputs(stamp_msg),
                corr_method=corr_method,
            ),
            gates=gates,
            # Forward-predict belief from T_belief_stamp to T_pixel using the
            # configured motion replay source. Paper-facing runs prefer
            # /odom_noisy and fall back to command replay only when odometry
            # samples are unavailable.
            replay=self._replay_cmd_log_interval,
            age=age,
            dt_s=dt_s,
            on_shape_error=self._log_pixel_shape_error_once,
        )

        warning = self._correction_reject_warning(outcome)
        if warning is not None:
            self._warn_stale_pixel_once(warning)
        if outcome.accepted:
            with self._data_lock:
                self.belief_m = outcome.next_m
                self.belief_S = outcome.next_S
                self.belief_stamp = stamp_msg
                self._last_correction_stamp = stamp_msg
        self._publish_correction_outcome(stamp_msg, outcome)
        if not outcome.accepted:
            return

        p_vis = outcome.p_vis
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

    def _reset_prediction_diagnostics(self):
        self._latest_prediction_source = 0.0
        self._latest_prediction_dt = 0.0
        self._latest_u_pred_v = 0.0
        self._latest_u_pred_omega = 0.0
        self._latest_Q_theta_theta = 0.0
        self._latest_odom_delta_theta = 0.0
        self._latest_cmd_delta_theta = 0.0

    def _predict_belief_to_now(self, m0, S0, last_cmd, belief_age_s: float, now_msg, mutate=True):
        if belief_age_s <= 0.0:
            self._latest_prediction_dt = 0.0
            self._latest_prediction_source = 0.0
            return m0, S0

        # Replay timestamped motion over the interval.  For paper-facing runs
        # use the configured odometry topic (normally /odom_noisy) so dead
        # reckoning follows the encoder/noisy-odometry estimate rather than the
        # ideal requested command.  Fall back to command replay if odometry has
        # no samples for this interval.
        try:
            now_s = Time.from_msg(now_msg).nanoseconds * 1e-9
        except (AttributeError, TypeError, ValueError):
            now_s = self.get_clock().now().nanoseconds * 1e-9
        t_start = now_s - belief_age_s
        with self._data_lock:
            odom_entries = list(self._odom_log)
            cmd_entries = list(self._cmd_log)

        # Collect odom and cmd delta yaw
        relevant_odom = [(t, v, w) for t, v, w in odom_entries if t_start < t <= now_s]
        odom_delta = 0.0
        if relevant_odom:
            pt = t_start
            for t, v, w in relevant_odom:
                odom_delta += w * (t - pt)
                pt = t
            odom_delta += relevant_odom[-1][2] * (now_s - pt)
        self._latest_odom_delta_theta = float(odom_delta)

        relevant_cmd = [(t, v, w) for t, v, w in cmd_entries if t_start < t <= now_s]
        cmd_delta = 0.0
        if relevant_cmd:
            pt = t_start
            for t, v, w in relevant_cmd:
                cmd_delta += w * (t - pt)
                pt = t
            cmd_delta += relevant_cmd[-1][2] * (now_s - pt)
        self._latest_cmd_delta_theta = float(cmd_delta)

        entries = odom_entries if self.use_odom_for_predict else cmd_entries
        previous = [(t, v, w) for t, v, w in entries if t <= t_start]
        relevant = [(t, v, w) for t, v, w in entries if t_start < t <= now_s]
        source_code = 1.0 if (self.use_odom_for_predict and (previous or relevant)) else 2.0
        if self.use_odom_for_predict and not previous and not relevant:
            entries = cmd_entries
            previous = [(t, v, w) for t, v, w in entries if t <= t_start]
            relevant = [(t, v, w) for t, v, w in entries if t_start < t <= now_s]
            source_code = 2.0

        if not previous and not relevant:
            source_code = 0.0

        self._latest_prediction_source = float(source_code)
        self._latest_prediction_dt = float(belief_age_s)
        try:
            Q = self.planner.process_noise(belief_age_s)
            self._latest_Q_theta_theta = float(Q[2, 2])
        except Exception:
            self._latest_Q_theta_theta = 0.0

        if not previous and not relevant:
            self._latest_u_pred_v = 0.0
            self._latest_u_pred_omega = 0.0
            m0, S0 = self.planner.predict(
                m0, S0, np.array([0.0, 0.0], dtype=float), dt=belief_age_s
            )
            if mutate:
                with self._data_lock:
                    self.belief_m = m0.copy()
                    self.belief_S = S0.copy()
                    self.belief_stamp = now_msg
            return m0, S0

        if previous:
            current_cmd = np.array([previous[-1][1], previous[-1][2]], dtype=float)
        elif relevant:
            current_cmd = np.array([0.0, 0.0], dtype=float)

        prev_t = t_start
        for t, v, w in relevant:
            dt_gap = t - prev_t
            if dt_gap > 1e-4:
                m0, S0 = self.planner.predict(m0, S0, current_cmd, dt=dt_gap)
            current_cmd = np.array([v, w], dtype=float)
            prev_t = t

        dt_tail = now_s - prev_t
        if dt_tail > 1e-4:
            m0, S0 = self.planner.predict(m0, S0, current_cmd, dt=dt_tail)

        self._latest_u_pred_v = float(current_cmd[0])
        self._latest_u_pred_omega = float(current_cmd[1])

        if mutate:
            with self._data_lock:
                self.belief_m = m0.copy()
                self.belief_S = S0.copy()
                self.belief_stamp = now_msg
        return m0, S0

    def _map_frame_heading(self):
        """map_bev heading = raw odom yaw + spawn-yaw offset (as pixel_to_bev does).
        None until the first odometry message arrives."""
        if self._latest_odom_yaw is None:
            return None
        return float(wrap_angle(self._latest_odom_yaw + self.odom_yaw_offset_rad))

    def _anchor_belief_yaw_for_planning(self, m0, S0, now_msg, mutate=True):
        """Heading anchor.

        Single-camera path (use_pixel_correction): pixel_to_bev already bakes the
        spawn-yaw offset into /state/bev, so leave the belief yaw alone.
        Multicam path (state_correction_ekf / no pixel correction): the manager's
        /state/bev carries no heading, so anchor the belief yaw to the map-frame
        odometry heading (odom_yaw + odom_yaw_offset_rad). Without this the belief
        heading stays in the raw-odom frame and the planner steers ~90 deg off.
        """
        self._heading_anchor_applied = False
        self._state_bev_yaw_ignored = True
        if not self.use_pixel_correction:
            h = self._map_frame_heading()
            if h is not None:
                m0 = np.asarray(m0, dtype=float).copy()
                m0[2] = h
                self._heading_anchor_applied = True
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

        # When the belief stamp is much older than pixel_timeout_s, replaying
        # the full odom window produces unstable predictions: the replay
        # length grows each tick, and truncated ring-buffer entries cause the
        # predicted position to oscillate rather than drift smoothly.  Fix:
        # commit a bounded prediction (up to pixel_timeout_s) to the internal
        # belief so that subsequent planning calls start from a recent state
        # instead of re-replaying from the stale correction stamp.
        if belief_age_s > self.pixel_timeout_s:
            self._warn_stale_pixel_once(
                f"Pixel belief stale (age {belief_age_s:.2f}s); planning on prediction-only belief"
            )
            # Predict the belief forward using a capped window, then commit
            # the result so the next planning call sees a fresh stamp.
            m0, S0 = self._predict_belief_to_now(
                snapshot['m'], snapshot['S'], snapshot['last_cmd'],
                belief_age_s, now_msg, mutate=True,
            )
            m0, S0 = self._anchor_belief_yaw_for_planning(m0, S0, now_msg, mutate=True)
            # Inflate xy covariance so nogo_belief_kappa grows and the planner
            # becomes conservative. Without YOLO the heading drifts, which
            # causes growing xy error; 0.3 m²/s inflation matches the observed
            # drift rate (~0.6 m/s × sin(heading_err) at typical conditions).
            staleness_s = belief_age_s - float(self.pixel_timeout_s)
            inflate = min(staleness_s * 0.3, 1.5)
            S0 = S0.copy()
            S0[0, 0] += inflate
            S0[1, 1] += inflate
        else:
            m0, S0 = self._predict_belief_to_now(
                snapshot['m'], snapshot['S'], snapshot['last_cmd'],
                belief_age_s, now_msg, mutate=False,
            )
            m0, S0 = self._anchor_belief_yaw_for_planning(m0, S0, now_msg, mutate=False)

        return m0, S0, {
            'measurement_available': bool(measurement_available),
            'belief_age_s': float(belief_age_s),
        }

    def _resolve_state_belief_for_planning(self):
        now_msg = self.get_clock().now().to_msg()
        with self._data_lock:
            state_ref = self.state_msg
            belief_m = None if self.belief_m is None else self.belief_m.copy()
            belief_S = None if self.belief_S is None else self.belief_S.copy()
            belief_stamp = self.belief_stamp
            last_cmd = self.last_cmd.copy()
        if state_ref is None:
            return None, None, {}
        if self.skip_stale_pixel_correction and not self._state_msg_is_fresh(state_ref):
            if belief_m is None or belief_S is None or belief_stamp is None:
                return None, None, {
                    'measurement_available': False,
                    'belief_age_s': math.inf,
                }
            belief_age_s = self._belief_age_for_planning(now_msg, belief_stamp)
            if belief_age_s is None:
                return None, None, {}
            if belief_age_s > self.pixel_timeout_s:
                m0, S0 = self._predict_belief_to_now(
                    belief_m, belief_S, last_cmd, belief_age_s, now_msg, mutate=True
                )
                m0, S0 = self._anchor_belief_yaw_for_planning(
                    m0, S0, now_msg, mutate=True
                )
                inflate = min((belief_age_s - self.pixel_timeout_s) * 0.3, 1.5)
                S0 = S0.copy()
                S0[0, 0] += inflate
                S0[1, 1] += inflate
            else:
                m0, S0 = self._predict_belief_to_now(
                    belief_m, belief_S, last_cmd, belief_age_s, now_msg, mutate=False
                )
                m0, S0 = self._anchor_belief_yaw_for_planning(
                    m0, S0, now_msg, mutate=False
                )
            return m0, S0, {
                'measurement_available': False,
                'belief_age_s': float(belief_age_s),
            }
        m0, S0 = self._state_msg_to_belief(state_ref)
        self._state_bev_yaw_ignored = True
        if self.belief_m is not None:
            m_pred = self.belief_m.copy()
            S_pred = self.belief_S.copy()
            if self.belief_stamp is not None:
                dt = self._stamp_to_float(state_ref.header.stamp) - self._stamp_to_float(self.belief_stamp)
                if dt > 1e-3:
                    try:
                        m_pred_new, S_pred_new = self.planner.predict(m_pred, S_pred, last_cmd, dt=dt)
                        self.get_logger().info(f"[CAMERA_XY_ONLY] state_stamp={self._stamp_to_float(state_ref.header.stamp):.4f} prev_stamp={self._stamp_to_float(self.belief_stamp):.4f} dt={dt:.4f} S_old={S_pred[2,2]:.6f} S_new={S_pred_new[2,2]:.6f}")
                        m_pred = m_pred_new
                        S_pred = S_pred_new
                    except Exception as e:
                        self.get_logger().error(f"[CAMERA_XY_ONLY] predict failed: {e}")
            m0[2] = float(m_pred[2])
            S0[2, 2] = float(S_pred[2, 2])
            S0[2, 0] = float(S_pred[2, 0])
            S0[0, 2] = float(S_pred[0, 2])
            S0[2, 1] = float(S_pred[2, 1])
            S0[1, 2] = float(S_pred[1, 2])
        with self._data_lock:
            self.belief_m = m0.copy()
            self.belief_S = S0.copy()
            self.belief_stamp = state_ref.header.stamp
        return m0, S0, {
            'measurement_available': True,
            'belief_age_s': 0.0,
        }

    def _reanchor_belief_to_correction(self, state_msg, reason=""):
        """Re-anchor to a fused /state/bev correction (message form)."""
        self._reanchor_belief_to_xy(
            state_msg.header.stamp,
            np.array([
                float(state_msg.pose.pose.position.x),
                float(state_msg.pose.pose.position.y),
            ], dtype=float),
            self._state_measurement_cov(state_msg),
            reason=reason,
        )

    def _reanchor_belief_to_xy(self, stamp_msg, z_xy, R, reason=""):
        """Hard re-anchor the belief to a fresh metric correction (reliable) and
        reset its covariance. Recovery from a diverged belief / bad-stamp jump:
        the correction is trustworthy, so snap to it rather than dead-reckon."""
        m = np.array([float(z_xy[0]), float(z_xy[1]), 0.0], dtype=float)
        h = self._map_frame_heading()
        if h is not None:
            m[2] = h
        R = np.asarray(R, dtype=float)
        S = np.diag([0.0, 0.0, NONINFORMATIVE_YAW_VAR]).astype(float)
        S[:2, :2] = R[:2, :2]
        S = self._regularize_state_covariance(S)
        with self._data_lock:
            self.belief_m, self.belief_S = m.copy(), S.copy()
            self.belief_stamp = stamp_msg
            self._last_correction_stamp = stamp_msg
        if reason:
            self._warn_stale_pixel_once(f"belief re-anchored to metric correction ({reason})")

    def _state_measurement_cov(self, state_msg):
        """Measurement covariance of one fused /state/bev correction."""
        return self._inflate_measurement_cov(
            self._xy_covariance_from_pose(state_msg.pose.covariance, floor=self.min_state_cov)
        )

    def _inflate_measurement_cov(self, R):
        """Add the unmodelled-error term in quadrature: R' = R + sigma^2 I."""
        if self.state_measurement_inflation_m2 <= 0.0:
            return R
        R = np.asarray(R, dtype=float).copy()
        R[0, 0] += self.state_measurement_inflation_m2
        R[1, 1] += self.state_measurement_inflation_m2
        return R

    def _snapshot_metric_correction_inputs(self, stamp_msg, z_xy):
        with self._data_lock:
            belief_m = None if self.belief_m is None else self.belief_m.copy()
            belief_S = None if self.belief_S is None else self.belief_S.copy()
            belief_stamp = self.belief_stamp
            cmd = self.last_cmd.copy()
        if belief_m is None or belief_S is None or belief_stamp is None:
            return None
        return bc.CorrectionSnapshot(
            belief_m=belief_m,
            belief_S=belief_S,
            belief_stamp=belief_stamp,
            cmd=cmd,
            meas=np.asarray(z_xy, dtype=float).reshape(-1)[:2].copy(),
            meas_stamp=stamp_msg,
            yaw_meas=None,
            yaw_sigma=math.nan,
            yaw_source=0.0,
        )

    def _apply_state_correction(self, state_msg):
        """Fold one FUSED /state/bev correction into the belief (message form)."""
        self._apply_metric_correction(
            state_msg.header.stamp,
            np.array([
                float(state_msg.pose.pose.position.x),
                float(state_msg.pose.pose.position.y),
            ], dtype=float),
            self._state_measurement_cov(state_msg),
        )

    def _apply_map_observations(self, observations):
        """Fold PER-CAMERA map observations in sequentially, one filter, N updates.

        The alternative -- collapsing them into a fused pose first -- runs two
        filters in series: ``camera_manager`` seeds from a median with an
        identity prior and no motion model, and the planner then treats that
        output as an independent measurement. Sequential updating into the one
        belief that actually has a prior and a motion model is the textbook form,
        and it lets each camera carry its own measurement covariance and be
        gated (and reason-coded) on its own.

        Applied in TIMESTAMP order, because a Kalman update is only valid against
        a prior predicted to that measurement's stamp; the chain predicts between
        them. Observations not newer than the belief are dropped rather than
        buffered -- with the belief already past them they carry no information
        the filter can use without a smoother.
        """
        ordered = sorted(observations, key=lambda o: float(o.timestamp_s))
        if not ordered:
            return

        # Divergence is a claim about the BELIEF, so it needs corroboration. A
        # lone camera reporting metres away is far more likely to be a bad
        # camera than proof the belief is lost -- snapping to it on its own word
        # would hand the belief to the worst observation in the batch. So the
        # per-observation re-anchor is OFF here, and instead a QUORUM of mutually
        # agreeing cameras can re-anchor the belief before the updates run.
        # (The fused path keeps the single-observation guard: that measurement
        # has already been through the manager's NIS + disagreement gates.)
        self._reanchor_on_camera_quorum(ordered)

        accepted_any = False
        for index, obs in enumerate(ordered):
            outcome = self._apply_metric_correction(
                self._float_to_stamp(float(obs.timestamp_s)),
                np.array([float(obs.xy_m[0]), float(obs.xy_m[1])], dtype=float),
                self._observation_covariance(obs),
                camera_index=float(index),
                label=str(obs.camera_id),
                allow_reanchor=False,
                # Inflate at most once per batch, below, and only if NOTHING
                # anchored the belief: inflating per rejected camera would let a
                # single persistently bad camera balloon the covariance at
                # N x rate until it is itself waved through.
                inflate_on_reject=False,
            )
            accepted_any = accepted_any or bool(outcome is not None and outcome.accepted)

        if not accepted_any:
            self._inflate_belief_after_rejection(
                f"no camera accepted ({len(ordered)} observation(s))"
            )

    def _observation_covariance(self, obs):
        R = np.array(
            [[float(obs.covariance_m2[0][0]), float(obs.covariance_m2[0][1])],
             [float(obs.covariance_m2[1][0]), float(obs.covariance_m2[1][1])]],
            dtype=float,
        )
        R[0, 0] = max(R[0, 0], self.min_state_cov)
        R[1, 1] = max(R[1, 1], self.min_state_cov)
        return self._inflate_measurement_cov(R)

    def _reanchor_on_camera_quorum(self, ordered):
        """Re-anchor only when several mutually agreeing cameras say the belief is lost."""
        if self.state_reanchor_m <= 0.0 or len(ordered) < 2:
            return
        with self._data_lock:
            belief_m = None if self.belief_m is None else self.belief_m.copy()
        if belief_m is None:
            return
        far = [
            obs for obs in ordered
            if math.hypot(float(obs.xy_m[0]) - float(belief_m[0]),
                          float(obs.xy_m[1]) - float(belief_m[1])) > self.state_reanchor_m
        ]
        if len(far) < max(2, (len(ordered) + 1) // 2):
            return
        # They must agree with EACH OTHER far better than they disagree with the
        # belief, else this is scattered noise rather than a lost belief.
        spread = max(
            math.hypot(float(a.xy_m[0]) - float(b.xy_m[0]), float(a.xy_m[1]) - float(b.xy_m[1]))
            for a in far for b in far
        )
        if spread >= self.state_reanchor_m:
            return
        xs = sorted(float(obs.xy_m[0]) for obs in far)
        ys = sorted(float(obs.xy_m[1]) for obs in far)
        median = np.array([xs[len(xs) // 2], ys[len(ys) // 2]], dtype=float)
        newest = max(far, key=lambda o: float(o.timestamp_s))
        self._reanchor_belief_to_xy(
            self._float_to_stamp(float(newest.timestamp_s)),
            median,
            self._observation_covariance(newest),
            reason=f"{len(far)}/{len(ordered)} cameras agree, spread {spread:.2f} m",
        )

    def _inflate_belief_after_rejection(self, reason: str):
        """Grow the belief covariance so a rejection can never freeze it."""
        inflate = float(self.state_reject_inflate_m2)
        if inflate <= 0.0:
            return
        with self._data_lock:
            if self.belief_S is None:
                return
            S = np.asarray(self.belief_S, dtype=float).copy()
            S[0, 0] += inflate
            S[1, 1] += inflate
            self.belief_S = self._regularize_state_covariance(S)
        self._warn_stale_pixel_once(
            f"correction rejected ({reason}); inflating {inflate:.3f} m^2"
        )

    def _apply_metric_correction(self, stamp_msg, z_xy, R, *,
                                 camera_index=math.nan, label="fused",
                                 allow_reanchor=True, inflate_on_reject=True):
        """Fold one metric xy correction into the belief as a proper EKF update,
        running the SAME gate chain as the single-camera pixel path
        (``planning.core.belief_correction``).

        A real recursive Bayesian filter, not a hard reset: predict the belief to
        the correction (image) stamp via motion replay (latency compensation),
        then a covariance-weighted Kalman update in map-xy (H = [I2 | 0]).
        Heading stays camera_xy_only.

        Previously the multicam path was a separately written chain, and every
        parity gap in ``docs/multicam_vs_paper1_correction_parity.md`` came from
        that fork: the jump limiter was configured but never read, and the NIS
        gate inflated S by +1.0 m² on reject, which neutered it for the next
        correction. Both are now the shared implementation. A rejection still
        advances the belief stamp and inflates -- mildly -- so it can never
        *freeze* the belief, the runaway the separate chain existed to avoid.
        """
        age = self._stamp_age_s(stamp_msg)
        if not self._metric_correction_is_fresh(age):
            return None
        with self._data_lock:
            has_belief = (
                self.belief_m is not None
                and self.belief_S is not None
                and self.belief_stamp is not None
            )
            belief_stamp = self.belief_stamp

        # Bootstrap the belief from the first fresh correction.
        if not has_belief:
            self._reanchor_belief_to_xy(stamp_msg, z_xy, R)
            return None

        dt_corr = self._stamp_to_float(stamp_msg) - self._stamp_to_float(belief_stamp)
        if dt_corr <= 1e-3:
            return None  # not newer than the belief we already hold
        if dt_corr > self.state_max_predict_dt_s:
            # Implausible gap (stale belief / bad-stamp correction). Replaying it
            # would jump the prediction tens of metres -> re-anchor instead.
            self._reanchor_belief_to_xy(stamp_msg, z_xy, R, reason=f"dt={dt_corr:.1f}s")
            return None

        outcome = bc.apply_correction(
            source=bc.FusedMapMeasurementSource(
                snapshot_fn=lambda: self._snapshot_metric_correction_inputs(stamp_msg, z_xy),
                measurement_cov_fn=lambda: R,
            ),
            gates=self._correction_gates(metric_measurement=allow_reanchor),
            replay=self._replay_cmd_log_interval,
            age=age,
            dt_s=dt_corr,
        )
        self._commit_metric_correction_outcome(
            stamp_msg, R, outcome, label=label, inflate_on_reject=inflate_on_reject
        )
        self._publish_correction_outcome(stamp_msg, outcome, camera_index=camera_index)
        return outcome

    def _metric_correction_is_fresh(self, age: float) -> bool:
        future_tolerance_s = max(float(self.pixel_timeout_s), 0.25)
        return bool(
            math.isfinite(age)
            and age <= float(self.pixel_timeout_s)
            and age >= -future_tolerance_s
        )

    def _float_to_stamp(self, seconds: float):
        stamp = TimeMsg()
        stamp.sec = int(seconds)
        stamp.nanosec = int(round((float(seconds) - int(seconds)) * 1e9))
        return stamp

    def _commit_metric_correction_outcome(self, stamp_msg, R, outcome, *, label="fused",
                                          inflate_on_reject=True):
        """Apply one metric correction outcome to the belief."""
        if outcome.accepted:
            R = np.asarray(R, dtype=float)
            m_upd = np.asarray(outcome.next_m, dtype=float).copy()
            S_upd = np.asarray(outcome.next_S, dtype=float).copy()
            S_pred = np.asarray(outcome.S_pred, dtype=float)
            # camera_xy_only: heading comes from map-frame odometry
            # (odom_yaw+offset), not the xy measurement. Falls back to the
            # predicted heading pre-odom. The xy measurement says nothing about
            # heading, so its row/column keep the predicted (grown) values.
            h = self._map_frame_heading()
            m_upd[2] = float(h) if h is not None else float(outcome.m_pred[2])
            S_upd[2, :] = S_pred[2, :]
            S_upd[:, 2] = S_pred[:, 2]
            # Floor the posterior xy variance at half the measurement noise so
            # the filter never becomes so confident that the next innovation
            # trips the NIS gate -- the collapse that caused divergence at tight R.
            S_upd[0, 0] = max(float(S_upd[0, 0]), 0.5 * float(R[0, 0]))
            S_upd[1, 1] = max(float(S_upd[1, 1]), 0.5 * float(R[1, 1]))
            S_upd = self._regularize_state_covariance(S_upd)
            # Write back so the published diagnostics report what was actually
            # committed, heading override and variance floor included.
            outcome.next_m = m_upd
            outcome.next_S = S_upd
            with self._data_lock:
                self.belief_m = m_upd
                self.belief_S = S_upd
                self.belief_stamp = stamp_msg
                self._last_correction_stamp = stamp_msg
            return

        if outcome.recover == bc.RECOVER_REANCHOR:
            # The BELIEF diverged, not the measurement. Snap to the correction --
            # rejecting would lock the belief out of recovery.
            self._reanchor_belief_to_xy(
                stamp_msg, outcome.meas, R,
                reason=f"{label} |innov|={outcome.innov_norm_m:.1f}m",
            )
            return

        if outcome.m_pred is None or outcome.S_pred is None:
            return   # rejected before a prediction existed; nothing to advance

        # Hold the PREDICTION (not the rejected posterior), but ADVANCE THE STAMP
        # and inflate, so a rejection can never freeze the belief. The inflation
        # is deliberately small: the old +1.0 m^2 left S_y ~ 1.0, which scored a
        # 1.87 m innovation at NIS ~3.4 and let it through un-gated on the very
        # next correction.
        inflate = float(self.state_reject_inflate_m2) if inflate_on_reject else 0.0
        S_hold = np.asarray(outcome.S_pred, dtype=float).copy()
        S_hold[0, 0] += inflate
        S_hold[1, 1] += inflate
        with self._data_lock:
            self.belief_m = np.asarray(outcome.m_pred, dtype=float).copy()
            self.belief_S = self._regularize_state_covariance(S_hold)
            self.belief_stamp = stamp_msg
        self._warn_stale_pixel_once(
            f"{label} correction rejected ({outcome.reason.value}: "
            f"NIS {outcome.nis:.2f}, update {outcome.xy_update_norm_m:.3f} m); "
            f"holding prediction, inflating {inflate:.3f} m^2"
        )

    def _resolve_state_belief_ekf(self, now_msg):
        """Planning-time belief for the EKF /state/bev path: read-only predict of
        the committed belief to *now* (corrections are applied in
        ``_apply_state_correction`` on arrival, exactly like the pixel path's
        split between ``_apply_pixel_correction`` and its planning resolver)."""
        with self._data_lock:
            state_ref = self.state_msg
            has_belief = self.belief_m is not None and self.belief_S is not None
            belief_m = None if self.belief_m is None else self.belief_m.copy()
            belief_S = None if self.belief_S is None else self.belief_S.copy()
            belief_stamp = self.belief_stamp
            last_cmd = self.last_cmd.copy()
        self._state_bev_yaw_ignored = True
        if not has_belief:
            # No correction has bootstrapped the belief yet.
            if state_ref is not None and self._state_msg_is_fresh(state_ref):
                self._apply_state_correction(state_ref)
                with self._data_lock:
                    belief_m = None if self.belief_m is None else self.belief_m.copy()
                    belief_S = None if self.belief_S is None else self.belief_S.copy()
                    belief_stamp = self.belief_stamp
            if belief_m is None or belief_S is None:
                return None, None, {'measurement_available': False, 'belief_age_s': math.inf}

        measurement_available = bool(
            state_ref is not None and self._state_msg_is_fresh(state_ref)
        )
        belief_age_s = self._belief_age_for_planning(now_msg, belief_stamp)
        if belief_age_s is None:
            belief_age_s = 0.0
        if belief_age_s > self.pixel_timeout_s:
            m0, S0 = self._predict_belief_to_now(
                belief_m, belief_S, last_cmd, belief_age_s, now_msg, mutate=True
            )
            m0, S0 = self._anchor_belief_yaw_for_planning(m0, S0, now_msg, mutate=True)
            inflate = min((belief_age_s - float(self.pixel_timeout_s)) * 0.3, 1.5)
            S0 = S0.copy()
            S0[0, 0] += inflate
            S0[1, 1] += inflate
        else:
            m0, S0 = self._predict_belief_to_now(
                belief_m, belief_S, last_cmd, belief_age_s, now_msg, mutate=False
            )
            m0, S0 = self._anchor_belief_yaw_for_planning(m0, S0, now_msg, mutate=False)
        return m0, S0, {
            'measurement_available': measurement_available,
            'belief_age_s': float(belief_age_s),
        }

    def _resolve_truth_belief_for_planning(self):
        """DIAGNOSTIC: return ground-truth pose as the belief with tiny covariance."""
        with self._data_lock:
            truth = self.truth_pose
        if truth is None:
            return None, None, {}
        m0 = np.array([truth[0], truth[1], truth[2]], dtype=float)
        S0 = np.diag([1e-4, 1e-4, 1e-4]).astype(float)
        return m0, S0, {'measurement_available': True, 'belief_age_s': 0.0}

    def _resolve_belief_for_planning(self):
        self._heading_anchor_applied = False
        self._state_bev_yaw_ignored = False
        self._reset_prediction_diagnostics()
        now_msg = self.get_clock().now().to_msg()
        if self.use_truth_localization:
            m0, S0, meta = self._resolve_truth_belief_for_planning()
        elif self.use_pixel_correction:
            m0, S0, meta = self._resolve_pixel_corrected_belief_for_planning(now_msg)
        elif self.state_correction_ekf:
            m0, S0, meta = self._resolve_state_belief_ekf(now_msg)
        else:
            m0, S0, meta = self._resolve_state_belief_for_planning()
        if m0 is None or S0 is None:
            return None, None, {}
        measurement_available = bool(meta.get('measurement_available', False))
        belief_age_s = float(meta.get('belief_age_s', 0.0))
        self._latest_measurement_available = bool(measurement_available)
        self._latest_belief_age_s = float(belief_age_s)
        with self._data_lock:
            b_stamp = self.belief_stamp
        return m0, S0, {
            'measurement_available': bool(measurement_available),
            'belief_age_s': float(belief_age_s),
            'belief_stamp': b_stamp,
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
            predict_vel = self.odom_vel.copy() if self.use_odom_for_predict else last_cmd
        if stamp_msg is None:
            return
        try:
            age_s = max(0.0, self._stamp_age_s(stamp_msg))
        except Exception:
            age_s = 0.0
        if age_s > 1e-3:
            try:
                # Replay the timestamped odom log over [belief_stamp, now] -- the
                # SAME path the planner uses to resolve its control belief -- rather
                # than a crude single-velocity predict (odom_vel * age). The crude
                # version froze the PUBLISHED belief at speed changes: at a stop the
                # latest odom_vel is ~0, so it could not propagate the 0.5-1s-old
                # belief anchor forward, leaving the logged belief stuck at a stale
                # pose -> spurious 0.3-0.5m backward jumps in the logged trajectory
                # (the controller was unaffected; it already used this replay). This
                # only changes the monitoring/logging belief. read-only: mutate=False.
                now_msg = self.get_clock().now().to_msg()
                m, S = self._predict_belief_to_now(
                    m, S, predict_vel, age_s, now_msg, mutate=False,
                )
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
        self.path_pub.publish(path)
        self.plan_preview_pub.publish(preview_path)

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
        active_plan_age_s = math.nan
        active_plan_remaining_s = math.nan
        active_control_index = math.nan
        active_controls_len = math.nan
        active_controls_original_len = math.nan
        latency_skip_steps = float(getattr(self, '_last_latency_skip_steps', 0))
        latency_skip_s = float(getattr(self, '_last_latency_skip_s', 0.0))
        command_timer_period_s = float(getattr(self, '_cmd_timer_period_s', math.nan))
        planner_timer_period_s = float(getattr(self, '_plan_period_s', math.nan))
        pending_active_remaining_s = float(
            getattr(self, '_pending_plan_started_active_remaining_s', math.nan)
        )
        if hasattr(self, '_active_controls') and hasattr(self, '_active_plan_started_at'):
            with self._data_lock:
                active_controls = None if self._active_controls is None else self._active_controls
                active_started_at = self._active_plan_started_at
                active_controls_original_len = float(
                    getattr(self, '_active_controls_original_len', 0)
                )
            if active_controls is not None and active_started_at is not None:
                step_dt = max(float(self.dt), 1e-3)
                active_plan_age_s = max(
                    (self.get_clock().now() - active_started_at).nanoseconds * 1e-9,
                    0.0,
                )
                active_controls_len = float(active_controls.shape[0])
                active_control_index = float(
                    min(int(active_plan_age_s / step_dt), active_controls.shape[0] - 1)
                )
                active_plan_remaining_s = max(
                    float(active_controls.shape[0]) * step_dt - active_plan_age_s,
                    0.0,
                )
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
            float(getattr(result, 'risk_mean', np.nan)),
            float(getattr(result, 'risk_cov_trace', np.nan)),
            float(getattr(result, 'risk_cov_logdet', np.nan)),
            float(getattr(result, 'delta_risk_visibility', np.nan)),
            float(getattr(result, 'delta_ambiguity_visibility', np.nan)),
            active_plan_age_s,
            active_plan_remaining_s,
            active_control_index,
            active_controls_len,
            active_controls_original_len,
            latency_skip_steps,
            latency_skip_s,
            command_timer_period_s,
            planner_timer_period_s,
            pending_active_remaining_s,
            float(self._latest_prediction_source),
            float(self._latest_prediction_dt),
            float(self._latest_u_pred_v),
            float(self._latest_u_pred_omega),
            float(self._latest_Q_theta_theta),
            float(self._latest_odom_delta_theta),
            float(self._latest_cmd_delta_theta),
            1.0 if self._heading_anchor_applied else 0.0,
            1.0 if self._state_bev_yaw_ignored else 0.0,
        ]
        self.planner_diag_pub.publish(diag)
        diag_text = String()
        diag_parts = [str(getattr(result, 'optimizer_message', '') or '').strip()]
        invalid_reason = str(getattr(result, 'invalid_reason', '') or '').strip()
        if invalid_reason:
            diag_parts.append(f'invalid_reason={invalid_reason}')
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
            # Unexpected planner failures should abort the run instead of allowing
            # an invalid experiment to continue as if it were valid evidence.
            result = self.planner.plan(
                m0, S0, goal_xy, progress_index=progress_index,
            )
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
