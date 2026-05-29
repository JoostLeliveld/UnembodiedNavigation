"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import time

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase, extract_waypoints


class EfeAgentNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_agent'
    PLANNER_CLASS = UnicyclePlannerBase

    def __init__(self):
        super().__init__()

        if not self.has_parameter('cmd_topic'):
            self.declare_parameter('cmd_topic', '/cmd_vel')
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self._active_controls = None
        self._active_plan_started_at = None
        self._cmd_timer_period_s = max(0.02, min(self.dt, 0.1))
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
            self.planner = self._construct_planner(
                horizon=self.local_horizon,
                use_ambiguity=self.local_use_ambiguity,
                goal_progress_n_steps=self.local_horizon,
                goal_prior_u_std_start=self.goal_prior_u_std_final,
                goal_prior_v_std_start=self.goal_prior_v_std_final,
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
                f"multistart={self.local_optimizer_multistart}, "
                f"maxiter={self.local_optimizer_maxiter})"
            )

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

        plan_start = time.perf_counter()
        result = self._call_planner(m0, S0, target, 0.0, plan_start=plan_start, now_wall=now_wall)
        if result is None:
            return
        plan_elapsed_ms = max((time.perf_counter() - plan_start) * 1000.0, 0.0)
        self._publish_plan_result_bundle(
            result, target, m0, S0, belief_meta=belief_meta, plan_elapsed_ms=plan_elapsed_ms
        )
        self._warn_on_plan_health(
            result, plan_elapsed_ms, float(getattr(result, 'solve_time_s', 0.0)) * 1000.0,
            now_wall=now_wall,
        )

    def _publish_command(self, v_cmd: float, w_cmd: float):
        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)
        self.last_cmd = np.array([cmd.linear.x, cmd.angular.z], dtype=float)

    def _publish_active_plan_command(self):
        with self._data_lock:
            controls = None if self._active_controls is None else self._active_controls.copy()
            started_at = self._active_plan_started_at
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
        self._publish_command(u[0], u[1])

    def _after_plan_result(self, result):
        # Keep following the current planned control sequence until replanning replaces it.
        controls = np.asarray(result.controls, dtype=float)
        with self._data_lock:
            self._active_controls = controls.copy()
            self._active_plan_started_at = self.get_clock().now()
        if controls.size == 0:
            return
        self._publish_command(controls[0, 0], controls[0, 1])

    def _publish_safe_stop_command(self):
        if not hasattr(self, 'cmd_pub'):
            return
        with self._data_lock:
            self._active_controls = None
            self._active_plan_started_at = None
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
