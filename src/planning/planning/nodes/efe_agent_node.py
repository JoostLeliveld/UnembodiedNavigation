"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import math
import time

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase, extract_waypoints
from planning.core.dynamics import unicycle_step
from planning.core.efe_utils import wrap_angle


class EfeAgentNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_agent'
    PLANNER_CLASS = UnicyclePlannerBase

    def __init__(self):
        super().__init__()

        if not self.has_parameter('cmd_topic'):
            self.declare_parameter('cmd_topic', '/cmd_vel')
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.active_execution_diag_pub = self.create_publisher(
            Float64MultiArray, '/planner/active_execution_diagnostics', 10
        )
        self._active_controls = None
        self._active_plan_started_at = None
        self._active_controls_original_len = 0
        self._last_local_plan_target = None
        self._pending_plan_started_at = None
        self._pending_plan_started_active_remaining_s = 0.0
        self._last_latency_skip_steps = 0
        self._last_latency_skip_s = 0.0
        self._current_wp_idx = math.nan
        self._current_wp_count = math.nan
        self._current_wp_target = np.array([math.nan, math.nan], dtype=float)
        self._current_wp_dist = math.nan
        self._current_desired_yaw = math.nan
        self._current_yaw_error = math.nan
        self._current_tracking_yaw = math.nan
        self._current_tracking_yaw_source = 0.0  # 0=belief/state, 1=odom override
        self._cmd_timer_period_s = 1.0 / max(float(self.cmd_publish_rate), 0.1)
        self._cmd_timer = self.create_timer(
            self._cmd_timer_period_s,
            self._publish_active_plan_command,
            callback_group=self._io_group,
        )

        # Two-stage hierarchical planning: self.planner is the LOCAL (lean,
        # short-horizon) tracker; self.global_planner computes one frozen
        # long-horizon visibility-aware plan whose states become waypoints.
        self._hier_phase = 'GLOBAL'
        self._waypoints = None
        self._wp_idx = 0
        self.global_planner = None
        if self.use_hierarchical:
            local_nogo_penalty_type = self.local_nogo_penalty_type or self.nogo_penalty_type
            local_nogo_weight = (
                self.nogo_weight if self.local_nogo_weight < 0.0 else self.local_nogo_weight
            )
            local_nogo_safe_distance = (
                self.nogo_safe_distance
                if self.local_nogo_safe_distance < 0.0
                else self.local_nogo_safe_distance
            )
            local_goal_u_final = (
                self.goal_prior_u_std_final
                if self.local_goal_prior_u_std_final < 0.0
                else self.local_goal_prior_u_std_final
            )
            local_goal_v_final = (
                self.goal_prior_v_std_final
                if self.local_goal_prior_v_std_final < 0.0
                else self.local_goal_prior_v_std_final
            )
            local_goal_u_start = (
                local_goal_u_final
                if self.local_goal_prior_u_std_start < 0.0
                else self.local_goal_prior_u_std_start
            )
            local_goal_v_start = (
                local_goal_v_final
                if self.local_goal_prior_v_std_start < 0.0
                else self.local_goal_prior_v_std_start
            )
            self.planner = self._construct_planner(
                horizon=self.local_horizon,
                use_ambiguity=self.local_use_ambiguity,
                goal_progress_n_steps=self.local_horizon,
                goal_prior_u_std_start=local_goal_u_start,
                goal_prior_v_std_start=local_goal_v_start,
                goal_prior_u_std_final=local_goal_u_final,
                goal_prior_v_std_final=local_goal_v_final,
                optimizer_multistart=self.local_optimizer_multistart,
                optimizer_multistart_include_direct=True,
                optimizer_multistart_lateral_offsets='',
                optimizer_initial_routes_json='',
                optimizer_warm_start_shift_steps=self._warm_start_shift_steps_for_rate(
                    self.local_plan_rate
                ),
                optimizer_maxiter=self.local_optimizer_maxiter,
                optimizer_maxfun=self.local_optimizer_maxiter * 4,
                use_belief_nogo_cost=self.local_use_belief_nogo_cost,
                use_visibility_model=self.local_use_visibility_model,
                nogo_penalty_type=local_nogo_penalty_type,
                nogo_weight=local_nogo_weight,
                nogo_safe_distance=local_nogo_safe_distance,
            )
            self.global_planner = self._construct_planner(
                horizon=self.global_horizon,
                use_ambiguity=self.global_use_ambiguity,
                optimizer_multistart=self.global_optimizer_multistart,
                optimizer_warm_start_shift_steps=self._warm_start_shift_steps_for_rate(
                    self.plan_rate
                ),
            )
            self.get_logger().info(
                f"[hierarchical] global H={self.global_horizon} (ambiguity={self.global_use_ambiguity}, "
                f"multistart={self.global_optimizer_multistart}) -> waypoints "
                f"(spacing {self.waypoint_spacing_m} m) -> local H={self.local_horizon} "
                f"(rate={self.local_plan_rate} Hz, ambiguity={self.local_use_ambiguity}, "
                f"visibility={self.local_use_visibility_model}, "
                f"belief_nogo={self.local_use_belief_nogo_cost}, "
                f"nogo={local_nogo_penalty_type}:{local_nogo_weight}, "
                f"safe={local_nogo_safe_distance}, "
                f"local_goal_std={local_goal_u_start:.2f}->{local_goal_u_final:.2f}/"
                f"{local_goal_v_start:.2f}->{local_goal_v_final:.2f}, "
                f"replan_min_remaining={self.local_replan_min_remaining_s:.2f}s, "
                f"latency_compensate={self.latency_compensate_plan_handoff}, "
                f"cmd_rate={self.cmd_publish_rate:.1f}Hz, "
                f"multistart={self.local_optimizer_multistart}, "
                f"maxiter={self.local_optimizer_maxiter})"
            )

    def _active_plan_remaining_s(self) -> float:
        with self._data_lock:
            controls = None if self._active_controls is None else self._active_controls
            started_at = self._active_plan_started_at
        if controls is None or controls.size == 0 or started_at is None:
            return 0.0
        elapsed_s = max((self.get_clock().now() - started_at).nanoseconds * 1e-9, 0.0)
        return max(float(controls.shape[0]) * max(float(self.dt), 1e-3) - elapsed_s, 0.0)

    def _plan_once(self):
        if not self.use_hierarchical:
            return super()._plan_once()

        inputs = self._snapshot_plan_inputs()
        goal_ref = inputs['goal']
        if goal_ref is None:
            return
        now_wall = time.monotonic()
        m0, S0, belief_meta = self._resolve_belief_for_planning()
        if m0 is None or S0 is None:
            return
        final_goal = self._goal_xy_from_msg(goal_ref)

        if self._hier_phase == 'GLOBAL':
            plan_start = time.perf_counter()
            try:
                rg = self.global_planner.plan(m0, S0, final_goal)
            except Exception as exc:  # noqa: BLE001
                self._fatal_experiment_stop("Global planner raised an exception", exc)
                return
            self._waypoints = extract_waypoints(
                rg.states, spacing_m=self.waypoint_spacing_m, include_goal=True
            )
            # The long-horizon global EFE plan gives the route shape, but its
            # finite horizon may stop short of the task goal. The local tracker
            # must still end at the actual mission goal rather than treating the
            # global plan terminus as success.
            if self._waypoints:
                last_wp = np.asarray(self._waypoints[-1], dtype=float)
                if float(np.linalg.norm(last_wp - final_goal)) > 1e-3:
                    self._waypoints.append((float(final_goal[0]), float(final_goal[1])))
            else:
                self._waypoints = [(float(final_goal[0]), float(final_goal[1]))]
            self._wp_idx = 0
            self._hier_phase = 'LOCAL'
            # Bootstrap the local planner's warm start from a direct-goal seed to the
            # first waypoint.  Without this, the first local call uses a cold zero-control
            # initialisation which takes 20-30 L-BFGS-B iterations to escape; with it,
            # the first call is already near-optimal and needs only 3-8 iterations.
            if self._waypoints:
                try:
                    wp0 = np.asarray(self._waypoints[0], dtype=float)
                    seed = self.planner._controls_for_waypoints(m0[:3], [wp0])
                    self.planner.prev_controls_flat = np.asarray(seed, dtype=float).reshape(-1)
                except Exception:
                    pass
            self.get_logger().info(
                f"[hierarchical] global plan solved in {(time.perf_counter()-plan_start):.1f}s -> "
                f"{len(self._waypoints)} waypoints; switching to local tracking"
            )
            # Publish the global plan for visualization; do NOT follow it.
            self._publish_plan_and_metrics(rg, final_goal, m0, S0, belief_meta=belief_meta)
            return

        # LOCAL phase: track the current planner-derived waypoint.
        if not self._waypoints:
            return
        target = np.asarray(self._waypoints[self._wp_idx], dtype=float)
        prev_wp_idx = self._wp_idx
        while (self._wp_idx < len(self._waypoints) - 1
               and float(np.linalg.norm(m0[:2] - target)) < self.waypoint_arrival_radius_m):
            self._wp_idx += 1
            target = np.asarray(self._waypoints[self._wp_idx], dtype=float)
        if self._wp_idx != prev_wp_idx:
            try:
                seed = self.planner._controls_for_waypoints(m0[:3], [target])
                self.planner.prev_controls_flat = np.asarray(seed, dtype=float).reshape(-1)
            except Exception:
                pass
        elif self.local_replan_min_remaining_s > 0.0:
            remaining_s = self._active_plan_remaining_s()
            if remaining_s > self.local_replan_min_remaining_s:
                return
        elif (
            self.local_replan_on_waypoint_change
            and self._last_local_plan_target is not None
            and np.allclose(target, self._last_local_plan_target, atol=1e-6)
            and self._active_plan_remaining_s() > 0.0
        ):
            return

        plan_start = time.perf_counter()
        self._pending_plan_started_at = self.get_clock().now()
        self._pending_plan_started_active_remaining_s = self._active_plan_remaining_s()
        self._last_local_plan_target = target.copy()

        m_track = m0.copy()
        S_track = S0.copy()
        tracking_yaw_source = 0.0
        if self.local_tracking_use_odom_yaw:
            now_msg = self.get_clock().now().to_msg()
            with self._data_lock:
                odom_yaw, _odom_age = self._fresh_odom_heading_locked(now_msg)
            if odom_yaw is not None:
                m_track[2] = float(odom_yaw)
                S_track[2, :] = 0.0
                S_track[:, 2] = 0.0
                S_track[2, 2] = float(max(self.odom_heading_sigma_rad ** 2, 1e-6))
                tracking_yaw_source = 1.0

        dx = float(target[0] - m_track[0])
        dy = float(target[1] - m_track[1])
        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(desired_yaw - float(m_track[2]))
        self._current_wp_idx = float(self._wp_idx)
        self._current_wp_count = float(len(self._waypoints))
        self._current_wp_target = target.copy()
        self._current_wp_dist = float(math.hypot(dx, dy))
        self._current_desired_yaw = float(desired_yaw)
        self._current_yaw_error = float(yaw_error)
        self._current_tracking_yaw = float(m_track[2])
        self._current_tracking_yaw_source = float(tracking_yaw_source)

        if self.use_simple_local_controller:
            controls = self._simple_local_plan(m_track, target)
            with self._data_lock:
                self._active_controls = controls.copy()
                self._active_plan_started_at = self.get_clock().now()
                self._active_controls_original_len = int(controls.shape[0])
            self._last_latency_skip_steps = 0
            self._last_latency_skip_s = 0.0
            if controls.size > 0:
                self._publish_command(float(controls[0, 0]), float(controls[0, 1]))
            return

        result = self._call_planner(m_track, S_track, target, 0.0, plan_start=plan_start, now_wall=now_wall)
        if result is None:
            return
        plan_elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
        self._publish_plan_result_bundle(
            result, target, m_track, S_track, belief_meta=belief_meta, plan_elapsed_ms=plan_elapsed_ms
        )
        self._warn_on_plan_health(
            result, plan_elapsed_ms, float(getattr(result, 'solve_time_s', 0.0)) * 1000.0,
            now_wall=now_wall,
        )

    def _simple_local_plan(self, m0: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Proportional geometric controller — returns (local_horizon, 2) [v, w] array."""
        H = int(self.local_horizon)
        dt = float(self.dt)
        v_max = float(self.v_max)
        w_max = 1.5

        # Rotate-in-place when badly misaligned, then translate. Without this
        # gate the exp(-|yaw_err|) taper still gives ~0.13 m/s at 90 deg, so the
        # robot creeps forward through a large departure turn and can clip an
        # adjacent obstacle (e.g. C1 swinging into R5L on the initial east->north
        # turn). A standard differential-drive turn-then-go controller.
        yaw_gate = 0.6  # rad (~34 deg): no forward motion until roughly aligned

        controls = np.zeros((H, 2), dtype=float)
        state = m0[:3].copy().astype(float)
        tx, ty = float(target[0]), float(target[1])

        for i in range(H):
            dx, dy = tx - state[0], ty - state[1]
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                break
            desired_yaw = math.atan2(dy, dx)
            yaw_err = wrap_angle(desired_yaw - state[2])
            w = float(np.clip(2.0 * yaw_err, -w_max, w_max))
            if abs(yaw_err) > yaw_gate:
                v = 0.0  # rotate in place until aligned
            else:
                v = float(v_max * math.exp(-abs(yaw_err)))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)

        return controls

    def _publish_command(self, v_cmd: float, w_cmd: float):
        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)
        self.last_cmd = np.array([cmd.linear.x, cmd.angular.z], dtype=float)

    def _result_safe_to_execute(self, result) -> tuple[bool, str]:
        """Return whether a solver result may replace the active control tape.

        Optimizer non-success can be acceptable for L-BFGS-B when the returned
        rollout is feasible and makes progress. It is not acceptable to execute
        controls that leave the known driveable domain, contain non-finite values,
        or fail to move the local tracker toward its waypoint.
        """
        controls = np.asarray(getattr(result, 'controls', []), dtype=float)
        if controls.ndim != 2 or controls.shape[0] == 0 or controls.shape[1] != 2:
            return False, 'empty_or_malformed_controls'
        if not np.all(np.isfinite(controls)):
            return False, 'nonfinite_controls'
        if not bool(getattr(result, 'rollout_valid', True)):
            reason = str(getattr(result, 'invalid_reason', '') or '').strip()
            return False, reason or 'invalid_rollout'
        clearance = float(getattr(result, 'min_predicted_obstacle_distance_m', math.nan))
        if math.isfinite(clearance) and clearance < 0.0:
            return False, f'negative_predicted_clearance:{clearance:.3f}'

        if self.use_hierarchical and self._hier_phase == 'LOCAL':
            current_dist = float(self._current_wp_dist)
            terminal_dist = float(getattr(result, 'terminal_goal_distance_pred', math.nan))
            if math.isfinite(current_dist) and math.isfinite(terminal_dist):
                if terminal_dist > current_dist - 1e-3:
                    return False, (
                        f'no_waypoint_progress:{terminal_dist:.3f}>={current_dist:.3f}'
                    )

        if self._pending_plan_started_at is not None:
            plan_latency_s = max(
                (self.get_clock().now() - self._pending_plan_started_at).nanoseconds * 1e-9,
                0.0,
            )
            tape_duration_s = float(controls.shape[0]) * max(float(self.dt), 1e-3)
            if plan_latency_s > tape_duration_s:
                return False, (
                    f'stale_control_tape:{plan_latency_s:.3f}>{tape_duration_s:.3f}'
                )

        return True, ''

    def _publish_active_plan_command(self):
        with self._data_lock:
            controls = None if self._active_controls is None else self._active_controls.copy()
            started_at = self._active_plan_started_at
            original_len = int(self._active_controls_original_len)
        if controls is None or controls.size == 0 or started_at is None:
            return

        elapsed_s = max((self.get_clock().now() - started_at).nanoseconds * 1e-9, 0.0)
        step_dt = max(float(self.dt), 1e-3)
        if elapsed_s >= controls.shape[0] * step_dt:
            # Do not keep replaying the terminal control of an exhausted local plan.
            # A slow or failed replan should leave the robot stopped, not coasting
            # into a boundary on stale controls.
            with self._data_lock:
                self._active_controls = None
                self._active_plan_started_at = None
            self._publish_command(0.0, 0.0)
            return
        step_idx = min(int(elapsed_s / step_dt), controls.shape[0] - 1)
        u = controls[step_idx]
        diag = Float64MultiArray()
        diag.data = [
            elapsed_s,
            float(max(controls.shape[0] * step_dt - elapsed_s, 0.0)),
            float(step_idx),
            float(controls.shape[0]),
            float(original_len),
            float(u[0]),
            float(u[1]),
            float(self._last_latency_skip_steps),
            float(self._last_latency_skip_s),
            float(self._current_wp_idx),
            float(self._current_wp_count),
            float(self._current_wp_target[0]),
            float(self._current_wp_target[1]),
            float(self._current_wp_dist),
            float(self._current_desired_yaw),
            float(self._current_yaw_error),
            float(self._current_tracking_yaw),
            float(self._current_tracking_yaw_source),
        ]
        self.active_execution_diag_pub.publish(diag)
        self._publish_command(u[0], u[1])

    def _after_plan_result(self, result):
        # Keep following the current planned control sequence until replanning replaces it.
        safe, reason = self._result_safe_to_execute(result)
        if not safe:
            self.get_logger().warn(
                f"Rejected local control tape before execution: {reason}. "
                "Publishing safe stop instead."
            )
            self._publish_safe_stop_command()
            return
        controls = np.asarray(result.controls, dtype=float)
        started_at = self.get_clock().now()
        skip_steps = 0
        latency_s = 0.0
        if self.latency_compensate_plan_handoff and self._pending_plan_started_at is not None:
            latency_s = max((started_at - self._pending_plan_started_at).nanoseconds * 1e-9, 0.0)
            latency_s = min(latency_s, max(float(self._pending_plan_started_active_remaining_s), 0.0))
            step_dt = max(float(self.dt), 1e-3)
            skip_steps = min(int(latency_s / step_dt), int(controls.shape[0]))
            fractional_s = max(latency_s - skip_steps * step_dt, 0.0)
            if skip_steps >= controls.shape[0]:
                with self._data_lock:
                    self._active_controls = None
                    self._active_plan_started_at = None
                    self._active_controls_original_len = int(controls.shape[0])
                self._last_latency_skip_steps = int(skip_steps)
                self._last_latency_skip_s = float(latency_s)
                self._publish_command(0.0, 0.0)
                return
            if fractional_s > 0.0:
                started_at = started_at - Duration(seconds=fractional_s)
            controls = controls[skip_steps:]
        with self._data_lock:
            self._active_controls = controls.copy()
            self._active_plan_started_at = started_at
            self._active_controls_original_len = int(result.controls.shape[0])
        self._last_latency_skip_steps = int(skip_steps)
        self._last_latency_skip_s = float(latency_s)
        if controls.size == 0:
            return
        self._publish_command(controls[0, 0], controls[0, 1])

    def _publish_safe_stop_command(self):
        if not hasattr(self, 'cmd_pub'):
            return
        with self._data_lock:
            self._active_controls = None
            self._active_plan_started_at = None
            self._active_controls_original_len = 0
        self._publish_command(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = EfeAgentNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()
