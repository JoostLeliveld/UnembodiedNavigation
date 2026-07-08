"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import json
import math
import time

import numpy as np

import os

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String

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
        # persistent state for the alternative trackers (hyst_damp damping/hysteresis)
        self._ctrl_prev_w = 0.0
        self._ctrl_spin = False
        # Persist the one-shot global route artifacts (solved plan + waypoints +
        # which seed won + costs). The global route is chosen ONCE and never
        # replanned, so it is the route-choice evidence for the campaign.
        self._run_dir = None
        self._pending_global_artifact = None
        self._global_artifact_saved = False
        run_dir_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/experiment/run_dir', self._run_dir_cb, run_dir_qos)
        self._local_ref_tracking_active = False
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
            # Metric (x,y) goal-progress incentive for the local executor. -1.0
            # inherits the shared weight; >=0.0 uses the local value directly.
            # This term is condition-neutral (no GP/ambiguity/visibility) and
            # only rewards reducing distance to the planner-derived target, so it
            # neither forces a route nor breaks the C1/C2 contract. It exists to
            # remove the unreachable-target zero-velocity local minimum of the
            # observation-space risk term.
            local_goal_progress_weight = (
                self.goal_progress_weight
                if self.local_goal_progress_weight < 0.0
                else self.local_goal_progress_weight
            )
            # [DEPRECATED_LEGACY_CLEANUP] LOCAL reference-segment tracking weights are legacy (simple tracker active)
            # LOCAL reference-segment tracking weights. -1.0 inherits 0.0 (OFF, so
            # the local executor falls back to the legacy single-point goal cost);
            # >=0.0 enables the proper TRACKER objective (reference-segment tracking
            # + control smoothness). Condition-neutral (no GP/visibility/ambiguity);
            # identical for C1/C2/C3 and never queries the GP. The reference segment
            # itself is computed OUTSIDE the solve from the global planner-derived
            # waypoint polyline (known 2D route), so this is execution plumbing, not
            # route choice.
            local_ref_weight = 0.0 if self.local_ref_weight < 0.0 else self.local_ref_weight
            local_terminal_ref_weight = (
                0.0 if self.local_terminal_ref_weight < 0.0 else self.local_terminal_ref_weight
            )
            local_du_weight = 0.0 if self.local_du_weight < 0.0 else self.local_du_weight
            # When reference tracking is active it REPLACES the single-point goal
            # cost: the reference segment already pulls the mean along the route, so
            # the beyond-horizon single-point goal_progress term (which gives no
            # heading gradient and caused the spin) is disabled for the local layer.
            self._local_ref_tracking_active = (
                local_ref_weight > 0.0 or local_terminal_ref_weight > 0.0
            )
            if self._local_ref_tracking_active:
                local_goal_progress_weight = 0.0
            self.planner = self._construct_planner(
                horizon=self.local_horizon,
                use_ambiguity=self.local_use_ambiguity,
                use_obs_risk=self.local_use_obs_risk,
                goal_progress_n_steps=self.local_horizon,
                goal_progress_weight=local_goal_progress_weight,
                ref_weight=local_ref_weight,
                terminal_ref_weight=local_terminal_ref_weight,
                du_weight=local_du_weight,
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
                dt=self.global_dt,
                optimizer_jit=self.optimizer_jit,
                use_ambiguity=self.global_use_ambiguity,
                optimizer_multistart=self.global_optimizer_multistart,
                optimizer_warm_start_shift_steps=self._warm_start_shift_steps_for_rate(
                    self.plan_rate
                ),
            )
            self.get_logger().info(
                f"[hierarchical] global H={self.global_horizon} (dt={self.global_dt:.3f}s, "
                f"lookahead={self.global_horizon * self.global_dt:.1f}s, jit={self.optimizer_jit}, "
                f"ambiguity={self.global_use_ambiguity}, "
                f"multistart={self.global_optimizer_multistart}) -> waypoints "
                f"(spacing {self.waypoint_spacing_m} m) -> local H={self.local_horizon} "
                f"(rate={self.local_plan_rate} Hz, ambiguity={self.local_use_ambiguity}, "
                f"visibility={self.local_use_visibility_model}, "
                f"belief_nogo={self.local_use_belief_nogo_cost}, "
                f"nogo={local_nogo_penalty_type}:{local_nogo_weight}, "
                f"safe={local_nogo_safe_distance}, "
                f"ref_track={self._local_ref_tracking_active} "
                f"(ref={local_ref_weight},term={local_terminal_ref_weight},du={local_du_weight}), "
                f"local_goal_progress={local_goal_progress_weight}, "
                f"local_goal_std={local_goal_u_start:.2f}->{local_goal_u_final:.2f}/"
                f"{local_goal_v_start:.2f}->{local_goal_v_final:.2f}, "
                f"replan_min_remaining={self.local_replan_min_remaining_s:.2f}s, "
                f"latency_compensate={self.latency_compensate_plan_handoff}, "
                f"cmd_rate={self.cmd_publish_rate:.1f}Hz, "
                f"simple_yaw_gate={self.simple_tracker_yaw_gate_rad:.2f}rad, "
                f"multistart={self.local_optimizer_multistart}, "
                f"maxiter={self.local_optimizer_maxiter})"
            )

    def _run_dir_cb(self, msg: String):
        self._run_dir = str(msg.data or '').strip() or None
        # Flush any global-route artifact captured before the run dir was known.
        if self._run_dir and self._pending_global_artifact is not None:
            self._write_global_artifact(self._pending_global_artifact)

    def _save_global_plan_artifacts(self, rg, m0, final_goal):
        """Capture the one-shot solved global route and persist it (or defer until
        the run directory is known)."""
        if self._global_artifact_saved:
            return
        try:
            states = np.asarray(rg.states, dtype=float)
            waypoints = [(float(w[0]), float(w[1])) for w in (self._waypoints or [])]
            seeds = [
                {'name': str(r.get('name', '')), 'waypoints': [[float(a), float(b)] for a, b in r.get('waypoints', [])]}
                for r in getattr(self.global_planner, 'optimizer_initial_routes', []) or []
            ]
            meta = {
                'selected_source': str(getattr(rg, 'selected_source', '')),
                'route_seed_mode': str(getattr(self, 'optimizer_route_seed_mode', 'explicit')),
                'route_seeds': seeds,
                'total_cost': float(getattr(rg, 'total_cost', float('nan'))),
                'risk_cost': float(getattr(rg, 'risk_cost', float('nan'))),
                'ambiguity_cost': float(getattr(rg, 'ambiguity_cost', float('nan'))),
                'obstacle_cost': float(getattr(rg, 'obstacle_cost', float('nan'))),
                'rollout_valid': bool(getattr(rg, 'rollout_valid', True)),
                'terminal_goal_distance_pred': float(getattr(rg, 'terminal_goal_distance_pred', float('nan'))),
                'global_horizon': int(self.global_horizon),
                'start_xy_yaw': [float(m0[0]), float(m0[1]), float(m0[2])],
                'goal_xy': [float(final_goal[0]), float(final_goal[1])],
                'n_states': int(states.shape[0]),
                'n_waypoints': int(len(waypoints)),
            }
            artifact = {'states': states, 'waypoints': waypoints, 'meta': meta}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"failed to capture global route artifact: {exc}")
            return
        if self._run_dir:
            self._write_global_artifact(artifact)
        else:
            self._pending_global_artifact = artifact

    def _write_global_artifact(self, artifact):
        try:
            run_dir = self._run_dir
            states = artifact['states']
            waypoints = artifact['waypoints']
            meta = artifact['meta']
            with open(os.path.join(run_dir, 'global_plan.csv'), 'w', encoding='utf-8') as f:
                f.write('point_idx,x,y,theta\n')
                for i, s in enumerate(states):
                    th = float(s[2]) if states.shape[1] > 2 else float('nan')
                    f.write(f'{i},{float(s[0]):.6f},{float(s[1]):.6f},{th:.6f}\n')
            with open(os.path.join(run_dir, 'global_waypoints.csv'), 'w', encoding='utf-8') as f:
                f.write('wp_idx,x,y\n')
                for i, w in enumerate(waypoints):
                    f.write(f'{i},{float(w[0]):.6f},{float(w[1]):.6f}\n')
            with open(os.path.join(run_dir, 'global_plan_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            self._global_artifact_saved = True
            self._pending_global_artifact = None
            self.get_logger().info(
                f"[hierarchical] saved global route artifacts -> global_plan.csv "
                f"({meta['n_states']} states), global_waypoints.csv ({meta['n_waypoints']} wp), "
                f"global_plan_meta.json (chose {meta['selected_source']})"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"failed to write global route artifact: {exc}")

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
            # Generate condition-neutral lane-graph route seeds from the driveable
            # map for this (one-shot) global solve. The global route is chosen once
            # and never replanned, so these seeds provide the nonconvex optimizer's
            # route-basin coverage. Generated from geometry + actual start + goal;
            # identical across conditions; no GP/visibility input.
            if (str(getattr(self, 'optimizer_route_seed_mode', 'explicit')) == 'lane_graph'
                    and str(getattr(self, 'driveable_geometry_json', '') or '')):
                try:
                    from unav_common.lane_graph_routes import generate_route_seeds
                    seeds = generate_route_seeds(
                        self.driveable_geometry_json,
                        (float(m0[0]), float(m0[1])),
                        (float(final_goal[0]), float(final_goal[1])),
                    )
                    if seeds:
                        self.global_planner.optimizer_initial_routes = (
                            self.global_planner._parse_initial_routes(json.dumps(seeds))
                        )
                        self.get_logger().info(
                            f"[hierarchical] lane-graph route seeds: "
                            f"{[s['name'] for s in seeds]}"
                        )
                    else:
                        self.get_logger().warn(
                            "lane-graph generated 0 seeds; keeping explicit "
                            "optimizer_initial_routes (check driveable_geometry_json covers start/goal)"
                        )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"lane-graph seed generation failed ({exc}); using explicit seeds")
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
                f"[hierarchical] global plan solved in {(time.perf_counter()-plan_start):.1f}s "
                f"(backend={getattr(rg, 'backend', '?')}, "
                f"nit={getattr(rg, 'optimizer_nit', 0)}, nfev={getattr(rg, 'optimizer_nfev', 0)}, "
                f"solve={getattr(rg, 'solve_time_s', 0.0):.1f}s, jit={self.optimizer_jit}) -> "
                f"{len(self._waypoints)} waypoints; switching to local tracking"
            )
            # Publish the global plan for visualization; do NOT follow it.
            self._publish_plan_and_metrics(rg, final_goal, m0, S0, belief_meta=belief_meta)
            # Persist the solved global route (plan states + waypoints + winning
            # seed + costs) to the run directory -- this is the one-shot route choice.
            self._save_global_plan_artifacts(rg, m0, final_goal)
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
            controls = self._dispatch_local_controller(m_track, target)
            n_safe, reason = self._simple_plan_safe_to_execute(controls, m_track)
            if n_safe <= 0:
                # The immediate step itself leaves the region (not a recovery move)
                # -> genuinely unsafe, safe-stop.
                self.get_logger().warn(
                    f"[hierarchical] simple local control rejected at step 0: {reason}; safe-stopping"
                )
                with self._data_lock:
                    self._active_controls = None
                    self._active_plan_started_at = None
                    self._active_controls_original_len = 0
                self._publish_command(0.0, 0.0)
                return
            # Execute only the safe leading prefix; the tracker replans next cycle.
            controls = controls[:n_safe]
            with self._data_lock:
                self._active_controls = controls.copy()
                self._active_plan_started_at = self.get_clock().now()
                self._active_controls_original_len = int(controls.shape[0])
            self._last_latency_skip_steps = 0
            self._last_latency_skip_s = 0.0
            if controls.size > 0:
                self._publish_command(float(controls[0, 0]), float(controls[0, 1]))
            return

        # [DEPRECATED_LEGACY_CLEANUP] local EFE reference segment tracking is legacy (simple tracker active)
        ref_seq = None
        prev_u = None
        if self._local_ref_tracking_active:
            # Reference segment r_1..r_H sampled forward along the global
            # planner-derived waypoint polyline, computed OUTSIDE the solve so the
            # optimizer only sees smooth squared tracking errors. prev_u is the last
            # applied command for the control-smoothness term. Condition-neutral:
            # the polyline is the known 2D global route, not a GP/visibility cue.
            ref_seq = self._build_local_reference_segment(m_track)
            with self._data_lock:
                prev_u = np.asarray(self.last_cmd, dtype=float).reshape(2).copy()

        result = self._call_planner(
            m_track, S_track, target, 0.0, plan_start=plan_start, now_wall=now_wall,
            ref_seq=ref_seq, prev_u=prev_u,
        )
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
        yaw_gate = float(self.simple_tracker_yaw_gate_rad)

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

    def _dispatch_local_controller(self, m0: np.ndarray, target: np.ndarray) -> np.ndarray:
        ct = getattr(self, 'local_controller_type', 'turn_then_go')
        if ct == 'hyst_damp':
            return self._hyst_damp_plan(m0, target)
        if ct == 'pure_pursuit':
            return self._pure_pursuit_plan(m0)
        if ct == 'ff_fb':
            return self._ff_fb_plan(m0)
        return self._simple_local_plan(m0, target)

    def _waypoint_array(self) -> np.ndarray | None:
        wps = self._waypoints
        if not wps:
            return None
        return np.asarray([(float(w[0]), float(w[1])) for w in wps], dtype=float)

    def _hyst_damp_plan(self, m0: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Turn-then-go + hysteresis on the spin gate, rate-limited (damped) w, and a
        small forward creep instead of a full stop -- kills the sharp-turn limit-cycle."""
        H = int(self.local_horizon); dt = float(self.dt); v_max = float(self.v_max)
        gate = float(self.simple_tracker_yaw_gate_rad)
        controls = np.zeros((H, 2), dtype=float)
        state = m0[:3].copy().astype(float)
        tx, ty = float(target[0]), float(target[1])
        spin = bool(self._ctrl_spin); w_prev = float(self._ctrl_prev_w)
        for i in range(H):
            dx, dy = tx - state[0], ty - state[1]
            if math.hypot(dx, dy) < 0.05:
                break
            yaw_err = wrap_angle(math.atan2(dy, dx) - state[2])
            # hysteresis: enter spin above the gate, only leave below 0.30 rad
            spin = (abs(yaw_err) > gate) or (spin and abs(yaw_err) > 0.30)
            w_des = float(np.clip(1.0 * yaw_err, -1.0, 1.0))
            w = w_prev + float(np.clip(w_des - w_prev, -0.15, 0.15))  # damp / rate-limit
            v = 0.08 if spin else float(v_max * math.exp(-abs(yaw_err)))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)
            w_prev = w
            if i == 0:
                self._ctrl_spin = spin; self._ctrl_prev_w = w
        return controls

    def _pure_pursuit_plan(self, m0: np.ndarray) -> np.ndarray:
        """Lookahead path tracker over the global waypoint polyline (always moving)."""
        H = int(self.local_horizon); dt = float(self.dt); v_max = float(self.v_max)
        wps = self._waypoint_array()
        controls = np.zeros((H, 2), dtype=float)
        if wps is None:
            return controls
        Ld = max(3.0 * float(self.waypoint_spacing_m), 0.30)
        state = m0[:3].copy().astype(float)
        for i in range(H):
            j = int(np.argmin(np.hypot(wps[:, 0] - state[0], wps[:, 1] - state[1])))
            while j < len(wps) - 1 and np.hypot(*(wps[j] - state[:2])) < Ld:
                j += 1
            dx, dy = wps[j] - state[:2]
            if np.hypot(dx, dy) < 0.05:
                break
            alpha = wrap_angle(math.atan2(dy, dx) - state[2])
            L = max(float(np.hypot(dx, dy)), 1e-3)
            w = float(np.clip(2.0 * v_max * math.sin(alpha) / L, -1.5, 1.5))
            v = float(v_max * max(0.2, 1.0 - abs(alpha) / 1.2))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)
        return controls

    def _ff_fb_plan(self, m0: np.ndarray) -> np.ndarray:
        """Path tangent feed-forward + cross-track/heading feedback on the belief."""
        H = int(self.local_horizon); dt = float(self.dt); v_max = float(self.v_max)
        wps = self._waypoint_array()
        controls = np.zeros((H, 2), dtype=float)
        if wps is None or len(wps) < 2:
            return controls
        state = m0[:3].copy().astype(float)
        for i in range(H):
            j = int(np.argmin(np.hypot(wps[:, 0] - state[0], wps[:, 1] - state[1])))
            j2 = min(j + 1, len(wps) - 1)
            seg = wps[j2] - wps[j]; seglen = float(np.hypot(*seg))
            if seglen < 1e-4:
                break
            tang = math.atan2(seg[1], seg[0])
            nh = np.array([-math.sin(tang), math.cos(tang)])
            ct = float((state[:2] - wps[j]) @ nh)        # + = left of path
            he = wrap_angle(tang - state[2])
            v = float(np.clip(v_max * max(0.25, 1.0 - 1.2 * abs(he) - 1.5 * abs(ct)), 0.05, v_max))
            w = float(np.clip(1.5 * he - 3.0 * ct, -1.5, 1.5))
            controls[i] = [v, w]
            state = unicycle_step(state, [v, w], dt)
            if np.hypot(*(wps[-1] - state[:2])) < 0.05:
                break
        return controls

    def _simple_plan_safe_to_execute(self, controls: np.ndarray, m0: np.ndarray) -> tuple[int, str]:
        """Recovery-aware feasibility gate for the non-optimizing waypoint tracker.

        Returns the number of LEADING control steps safe to execute (>=1 -> publish
        that prefix; 0 -> safe-stop). Two reasons it is not a naive `clearance<0`
        veto: (1) in a narrow keep-in aisle a TRANSIENT belief-prediction excursion
        (e.g. odom overshoot during a hard turn) can put the predicted mean
        mm-outside the band even though truth is centred; a hard veto then freezes
        the tracker at (0,0) forever. So reject a step only if it drives an already
        negative clearance strictly WORSE than the plan start (the robot is actively
        leaving the region); holding/recovering a marginal violation is allowed.
        (2) the rollout is open-loop and we only execute the first step before
        replanning, so a violation a few steps ahead must not veto the safe
        immediate step -- we execute the safe prefix and let the next replan re-aim.
        """
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[0] == 0 or controls.shape[1] != 2:
            return 0, 'empty_or_malformed_controls'
        if not np.all(np.isfinite(controls)):
            return 0, 'nonfinite_controls'

        RECOVERY_EPS = 5e-3
        start = np.asarray(m0[:3], dtype=float)
        if self.planner.collision_cost_model is not None:
            start_coll = self.planner.collision_signed_distance_state_np(start)
        else:
            start_coll = float('inf')
        nogo = self.planner.nogo_cost_model
        if nogo is not None and nogo.enabled:
            start_nogo = nogo.clearance_state_np(start)
        else:
            start_nogo = float('inf')
        coll_floor = (min(start_coll, 0.0) - RECOVERY_EPS) if math.isfinite(start_coll) else -math.inf
        nogo_floor = (min(start_nogo, 0.0) - RECOVERY_EPS) if math.isfinite(start_nogo) else -math.inf

        state = start.copy()
        for i, u in enumerate(controls):
            state = unicycle_step(state, u, float(self.dt))
            if self.planner.collision_cost_model is not None:
                clearance = self.planner.collision_signed_distance_state_np(state)
                if math.isfinite(clearance) and clearance < 0.0 and clearance < coll_floor:
                    return i, f'collision_geometry_violation_step_{i}:{clearance:.3f}'
            if nogo is not None and nogo.enabled:
                clearance = nogo.clearance_state_np(state)
                if math.isfinite(clearance) and clearance < 0.0 and clearance < nogo_floor:
                    return i, f'driveable_clearance_violation_step_{i}:{clearance:.3f}'
        return controls.shape[0], ''

    # [DEPRECATED_LEGACY_CLEANUP] local reference segment sampling is legacy (simple tracker active)
    def _build_local_reference_segment(self, m0: np.ndarray) -> np.ndarray:
        """Sample a (local_horizon, 2) reference segment along the global waypoint
        polyline, starting from the projection of the current belief (x, y).

        Computed OUTSIDE the optimizer: the optimizer only ever sees fixed numeric
        per-step targets, so no nonsmooth projection enters the gradient. The
        polyline is the known 2D global planner-derived route (self._waypoints);
        this method never queries the GP and is identical for C1/C2/C3.

        Sampling step is ~ v_max * dt (the per-step reach), clamped at the final
        waypoint so a short horizon does not overshoot the route end.
        """
        H = int(self.local_horizon)
        step = max(float(self.v_max) * float(self.dt), 1e-3)
        pos = np.asarray(m0[:2], dtype=float).reshape(2)

        pts = [np.asarray(w, dtype=float).reshape(2) for w in (self._waypoints or [])]
        if not pts:
            # No route yet: hold position (the ref terms then add nothing useful but
            # remain well-defined). Should not happen in the LOCAL phase.
            return np.tile(pos, (H, 1))
        if len(pts) == 1:
            return np.tile(pts[0], (H, 1))

        # Project pos onto the polyline: find the closest point across all segments
        # and record (segment index, fractional position along that segment).
        best_seg = 0
        best_t = 0.0
        best_d2 = float('inf')
        best_proj = pts[0]
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            denom = float(ab @ ab)
            if denom <= 1e-12:
                t = 0.0
                proj = a
            else:
                t = float(np.clip(((pos - a) @ ab) / denom, 0.0, 1.0))
                proj = a + t * ab
            d2 = float(np.sum((pos - proj) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_seg = i
                best_t = t
                best_proj = proj

        # Walk forward along the polyline from the projection, emitting a reference
        # point every `step` metres of arc length. Clamp at the final waypoint.
        ref = np.zeros((H, 2), dtype=float)
        seg = best_seg
        cur = best_proj.copy()
        for k in range(H):
            remaining = step
            while remaining > 1e-9 and seg < len(pts) - 1:
                nxt = pts[seg + 1]
                seg_vec = nxt - cur
                seg_len = float(np.linalg.norm(seg_vec))
                if seg_len <= 1e-9:
                    seg += 1
                    cur = pts[seg].copy() if seg < len(pts) else cur
                    continue
                if seg_len >= remaining:
                    cur = cur + (remaining / seg_len) * seg_vec
                    remaining = 0.0
                else:
                    cur = nxt.copy()
                    remaining -= seg_len
                    seg += 1
            ref[k] = cur
        del best_t
        return ref

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
            # A unicycle must rotate to face the waypoint before it can reduce
            # distance, so a turning (low-translation) plan is legitimate progress,
            # not a freeze. Only require distance progress once roughly aligned;
            # while the heading error is large, allow the turn. This mirrors the
            # simple tracker's rotate-then-go gate (yaw_gate=0.6 rad) and applies
            # identically to all conditions.
            yaw_err = abs(float(getattr(self, '_current_yaw_error', 0.0)))
            require_distance_progress = yaw_err <= 0.6
            if (require_distance_progress and math.isfinite(current_dist)
                    and math.isfinite(terminal_dist)):
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
