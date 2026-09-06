"""Audit 09: deterministic production-method probes, no ROS graph or simulation.

Assertions pin observed behaviour (including defects), not repaired invariants.
Only source slices used for logger dispatch/distance are extracted; all predicates,
callbacks, tracker, safety gate and summary writer execute their production code.
Boundary substitutes: clocks, publishers, solver result, and shutdown timer.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import conftest  # noqa: E402
sys.path.insert(0, str(ROOT / 'tests/planning'))
from test_runtime_transactions import command_node  # noqa: E402
from test_planner_node_correction_wiring import _Clock, _Logger, stamp  # noqa: E402
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist  # noqa: E402
from planning.nodes.efe_agent_node import EfeAgentNode  # noqa: E402
from planning.planners.base_planner import extract_waypoints  # noqa: E402
from planning.core.dynamics import unicycle_step  # noqa: E402
from experiments.nodes.goal_mission_node import GoalMissionNode  # noqa: E402
from experiments.nodes.experiment_logger import ExperimentLogger  # noqa: E402
from sim.actuation_noise_node import ActuationNoiseNode  # noqa: E402
from unav_common.preselected_route import canonicalize_polyline_json  # noqa: E402
from unav_common.lane_graph_routes import generate_route_seeds, _dedupe  # noqa: E402

FILES = [
    'src/planning/planning/nodes/efe_agent_node.py',
    'src/planning/planning/nodes/unicycle_planner_node.py',
    'src/planning/planning/planners/base_planner.py',
    'src/planning/planning/core/tracker_guard.py',
    'src/experiments/experiments/nodes/goal_mission_node.py',
    'src/experiments/experiments/nodes/experiment_logger.py',
    'src/unav_common/unav_common/preselected_route.py',
    'src/unav_common/unav_common/lane_graph_routes.py',
    'src/sim/sim/actuation_noise_node.py',
    'src/experiments/experiments/core/visibility_launch_common.py',
    'scripts/visibility_comparison/run_visibility_campaign.py',
]


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}


def goal(x=1., y=0., frame='map_bev'):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x, msg.pose.position.y = float(x), float(y)
    msg.pose.orientation.w = 1.
    return msg


def belief(x=0., y=0., yaw=0., t=10., frame='map_bev', variance=.01, cross=0.):
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = stamp(t)
    msg.header.frame_id = frame
    msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
    msg.pose.pose.orientation.z = math.sin(yaw / 2.)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.)
    msg.pose.covariance[0] = msg.pose.covariance[7] = float(variance)
    msg.pose.covariance[1] = msg.pose.covariance[6] = float(cross)
    msg.pose.covariance[35] = .01
    return msg


def mission():
    n = object.__new__(GoalMissionNode)
    n.waypoints = [(0., 0.), (.08, 0.), (.08, .08), (1., 0.)]
    n.wp_idx = 0
    n._belief_xy = None
    n._belief_ready = False
    n._belief_sigma_m = math.inf
    n._belief_wait_logged = False
    n.initial_belief_max_sigma_m = .2
    n.arrival_radius = .6
    n.wait_for_belief = True
    n.sent_count = 0
    n.repeat_count = 0
    n.delay = 0.
    n.frame_id = 'map_bev'
    n.wall_clock = _Clock(1_788_728_000.)
    n.start_time = _Clock(1_788_727_999.).now()
    n.get_clock = lambda: _Clock(10.)
    n.log = _Logger()
    n.get_logger = lambda: n.log
    n.sent = []
    n.goal_pub = NS(publish=n.sent.append)
    return n


def tracker(points=((1., 0.),), state=(0., 0., 0.)):
    n = command_node()
    n.use_hierarchical = True
    n._hier_phase = 'LOCAL'
    n.global_planner_mode = 'efe'
    n._global_goal_xy = np.asarray(points[-1])
    n.goal_replan_move_m = 1.
    n._waypoints = list(points)
    n._wp_idx = 0
    n.waypoint_spacing_m = .2
    n.waypoint_arrival_radius_m = .1
    n.local_replan_min_remaining_s = 0.
    n.local_replan_on_waypoint_change = False
    n.local_controller_type = 'turn_then_go'
    n._last_local_plan_target = None
    n.local_horizon = 12
    n.v_max = .22
    n.w_min, n.w_max = -1., 1.
    n.simple_tracker_yaw_gate_rad = .6
    n.goal_msg = goal(*points[-1])
    n.state_msg = belief()
    n.pixel_stamp = None
    n._goal_received_logged = True
    n._update_goal_progress_origin = lambda _: None
    n.belief_m = np.asarray(state, dtype=float)
    n.belief_S = np.eye(3) * .01
    n.belief_stamp = stamp(10.)
    n._resolve_belief_for_planning = lambda: (n.belief_m.copy(), n.belief_S.copy(), {})
    n._fatal_stop_triggered = False
    n.log = _Logger()
    n.get_logger = lambda: n.log
    n.planner = NS(collision_cost_model=None, nogo_cost_model=None)
    return n


LOGGER_SOURCE = ROOT / FILES[5]
logger_tree = ast.parse(LOGGER_SOURCE.read_text())
logger_class = next(n for n in logger_tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExperimentLogger')
logger_methods = {n.name: n for n in logger_class.body if isinstance(n, ast.FunctionDef)}


def source_slice(name, signature, prefix, nodes, suffix=''):
    """Bind unmodified statements from _log_once, with explicit local inputs."""
    fn = ast.parse('def ' + name + signature + ':\n    pass\n').body[0]
    fn.body = ast.parse(prefix).body + nodes + ast.parse(suffix).body
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {'math': math}
    exec(compile(mod, str(LOGGER_SOURCE), 'exec'), ns)
    return ns[name]


log_nodes = logger_methods['_log_once'].body
op_start = next(i for i, n in enumerate(log_nodes) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'operational_goal_dist_m')
operational_distance = source_slice('operational_distance', '(self)',
    'planner_belief_ok, _, planner_belief_x, planner_belief_y, *_ = self._latest_planner_belief_pose()\n'
    'goal_x = goal_y = math.nan\n', log_nodes[op_start:op_start+2], 'return operational_goal_dist_m')
tail_start = next(i for i, n in enumerate(log_nodes) if isinstance(n, ast.If)
                  and '_contact_collision_seen' in ast.unparse(n.test))
terminal_dispatch = source_slice('terminal_dispatch',
    '(self, now_stamp, operational_goal_dist_m, cmd_v, cmd_w, contact_topic_publishers)', '', log_nodes[tail_start:])


def logger(out, name):
    n = object.__new__(ExperimentLogger)
    # All summary aggregates are empty; replace only external boundaries.
    for a in ast.walk(logger_methods['_finish_run']):
        if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == 'self':
            if not hasattr(n, a.attr):
                setattr(n, a.attr, 0.)
    n._stop_requested = n._completed = False
    n._first_cmd_stamp = 0.
    n._goal_in_radius_since = n._goal_stable_since = None
    n._goal_region_entered = False
    n._goal_region_first_stamp = math.nan
    n._contact_collision_seen = n._geom_collision_seen = False
    n._first_crash_stamp = math.nan
    n._collision_reason = ''
    n._contact_messages_seen = 0
    n._valid_run = True
    n._invalid_reason = ''
    n._motion_history = deque()
    n.auto_stop_on_goal = True
    n.goal_success_radius = .35
    n.goal_success_hold_s = 2.
    n.goal_stable_radius = .2
    n.goal_stable_hold_s = 2.
    n.goal_stable_max_displacement_m = .04
    n.stuck_window_s = 8.
    n.stuck_max_displacement_m = .08
    n.stuck_max_goal_improvement_m = .05
    n.stuck_cmd_fraction_min = .5
    n.stuck_idle_cmd_fraction_max = .1
    n.first_cmd_linear_eps = .02
    n.first_cmd_angular_eps = .1
    n.run_timeout_after_first_cmd_s = 600.
    n.goal_msg = goal(1., 0.)
    n.planner_belief_msg = belief()
    n._gt_xy = None
    n._gt_intervals, n._gt_buf = [], deque()
    n._gt_stamp_source = 'unavailable_in_synthetic_probe'
    n._frame_sanity, n._assimilation_dropped_reasons = {}, {}
    n._min_wall_distance = n._min_obstacle_distance = n._min_goal_distance = math.inf
    n.file = io.StringIO()
    n.plan_file = n.perception_file = None
    n.run_dir = str(out / name)
    Path(n.run_dir).mkdir()
    n._latest_odom_map_pose = lambda: (False, math.nan, math.nan, math.nan, math.nan)
    n._clock = _Clock(10.)
    n.get_clock = lambda: n._clock
    n.log = _Logger()
    n.get_logger = lambda: n.log
    n.publishers = 1
    n.count_publishers = lambda _: n.publishers
    n.terminal_snapshot = None
    real_finish = n._finish_run
    def finish(reason, t=None):
        if n._stop_requested:
            return
        n.terminal_snapshot = dict(reason=reason, decision_stamp=t,
            belief_stamp=n._stamp_to_float(n.planner_belief_msg.header.stamp),
            belief_frame=n.planner_belief_msg.header.frame_id,
            belief_pose=list(n._latest_planner_belief_pose()[2:5]),
            operational_goal_distance_m=operational_distance(n),
            command=list(getattr(n, 'audit_command', (math.nan, math.nan))))
        # Never initialize/shut down ROS; retain the requested deferred teardown.
        with patch('threading.Timer', lambda delay, fn: NS(start=lambda: None)):
            real_finish(reason, t)
    n._finish_run = finish
    return n


def tick(n, t, v=0., w=0., pubs=None):
    n._clock.seconds = t
    n.audit_command = (v, w)
    d = operational_distance(n)
    ok, _, x, y, *_ = n._latest_planner_belief_pose()
    n._remember_motion_sample(t, x, y, d, v, w)
    terminal_dispatch(n, t, d, v, w, n.publishers if pubs is None else pubs)


def outcome(n):
    path = Path(n.run_dir) / 'run_summary.json'
    summary = json.loads(path.read_text()) if path.exists() else None
    return dict(snapshot=n.terminal_snapshot, summary=summary,
                log=n.log.messages, history_samples=len(n._motion_history))


def commands(n):
    return [[m.linear.x, m.angular.z] for m in n.cmd_pub.messages]


def run(out):
    results = {}
    initial_hashes = hashes()
    for mode in ('negative_variance', 'indefinite_xy', 'wrong_frame', 'stale', 'nan_position'):
        n = mission()
        n._belief_cb(belief())
        m = belief(variance=-1. if mode == 'negative_variance' else .01,
                   cross=.02 if mode == 'indefinite_xy' else 0.,
                   frame='odom' if mode == 'wrong_frame' else 'map_bev',
                   t=0. if mode == 'stale' else 10.,
                   x=math.nan if mode == 'nan_position' else 0.)
        n._belief_cb(m)
        n._send_goal()
        assert n._belief_ready and n.sent
        results['mission_' + mode] = dict(ready=n._belief_ready, waypoint=n.wp_idx,
            published_goal=[n.sent[-1].pose.position.x, n.sent[-1].pose.position.y],
            goal_stamp=n.sent[-1].header.stamp.sec, sim_now=10.,
            belief_stamp=m.header.stamp.sec, belief_frame=m.header.frame_id)

    points = [(0., 0.), (.04, 0.), (.04, 0.), (.04, .04), (1., .04)]
    n = tracker(points)
    n._plan_once()
    assert n._wp_idx == 4
    results['short_duplicate_sharp_turn_handoff'] = dict(points=points, target_index=n._wp_idx,
        target=n._current_wp_target.tolist(), commands=commands(n), lane_dedupe=_dedupe(points))
    try:
        canonicalize_polyline_json(json.dumps(points))
    except ValueError as exc:
        results['preselected_duplicate_rejection'] = str(exc)
    else:
        raise AssertionError('preselected duplicate unexpectedly accepted')
    source = np.array([[0., 0., 0.], [.11, 0., 0.], [.11, .11, math.pi/2]])
    results['downsample_loses_corner'] = dict(source=source.tolist(), output=extract_waypoints(source, .2))
    assert results['downsample_loses_corner']['output'] == [(.11, .11)]
    sampled = np.column_stack((np.arange(5)*.22, np.zeros((5, 2))))
    results['spacing_is_threshold_not_maximum'] = extract_waypoints(sampled, .2)

    for delta in (.999, 1., 1.001):
        n = tracker(((1., 0.),), state=(1., 0., 0.))
        n._global_solve_done = True
        n.optimizer_route_seed_mode = 'explicit'
        n.global_planner = NS(_parse_initial_routes=lambda x: x,
                             plan=lambda *args: (_ for _ in ()).throw(ValueError('synthetic failure')))
        n.goal_msg = goal(1.+delta, 0.)
        n._plan_once()
        results['goal_change_' + str(delta)] = dict(global_goal=n._global_goal_xy.tolist(),
            current_goal=n.goal_msg.pose.position.x, target=n._current_wp_target.tolist(),
            waypoints=n._waypoints, commands=commands(n))
        if delta <= 1.:
            assert n._global_goal_xy[0] == 1. and commands(n)[-1] == [0., 0.]
        else:
            assert n._global_goal_xy[0] == 1.+delta

    n = tracker()
    n._hier_phase = 'GLOBAL'
    n._global_solve_done = False
    n.optimizer_route_seed_mode = 'explicit'
    n._publish_plan_and_metrics = lambda *a, **kw: None
    n._save_global_plan_artifacts = lambda *a: None
    def solve(m, S, target):
        n._goal_cb(goal(-2., 0.))
        n.belief_m = np.array([0., 1., math.pi])
        return NS(states=np.array([[0., 0., 0.], [1., 0., 0.]]))
    n.global_planner = NS(plan=solve)
    n._plan_once()
    assert n._global_goal_xy[0] == 1. and n.goal_msg.pose.position.x == -2.
    results['goal_and_belief_during_global_solve'] = dict(installed_route=n._waypoints,
        route_goal=n._global_goal_xy.tolist(), latest_goal=-2., current_belief=n.belief_m.tolist())

    for mode in ('goal', 'belief', 'ordinary_stop', 'fatal_stop'):
        n = tracker()
        real_dispatch = n._dispatch_local_controller
        def dispatch(m, target):
            if mode == 'goal': n._goal_cb(goal(-2., 0.))
            if mode == 'belief': n.belief_m = np.array([0., 0., math.pi])
            if mode in ('ordinary_stop', 'fatal_stop'):
                n._fatal_stop_triggered = mode == 'fatal_stop'
                n._publish_safe_stop_command()
            return real_dispatch(m, target)
        n._dispatch_local_controller = dispatch
        n._plan_once()
        results['during_local_' + mode] = dict(commands=commands(n), belief=n.belief_m.tolist(),
            goal=n.goal_msg.pose.position.x, target=n._current_wp_target.tolist())
        assert commands(n)[-1][0] == (0. if mode == 'fatal_stop' else .22)

    n = tracker()
    n._plan_once()
    n.belief_m = np.array([0., 0., math.pi])
    n._clock.seconds += .1
    n._publish_active_plan_command()
    results['belief_update_during_active_tape'] = dict(commands=commands(n), current_belief=n.belief_m.tolist())
    assert commands(n)[-1] == [.22, 0.]
    n = tracker(((.04, 0.), (1., 0.)))
    n.state_msg = belief(frame='odom')
    n.goal_msg = goal(1., 0., frame='map_bev')
    n._plan_once()
    results['hierarchical_wrong_frame_skips_validation'] = dict(index=n._wp_idx,
        state_frame=n.state_msg.header.frame_id, goal_frame=n.goal_msg.header.frame_id, commands=commands(n))
    assert n._wp_idx == 1 and commands(n)[-1][0] > 0.

    # The 5 mm allowance is re-based on every plan, allowing persistent creep.
    n = tracker()
    n.planner.collision_cost_model = object()
    n.planner.collision_signed_distance_state_np = lambda s: -float(s[0])
    pose = np.array([.001, 0., 0.])
    records = []
    tape = np.tile([.016, 0.], (12, 1))
    for _ in range(30):
        safe, why = n._simple_plan_safe_to_execute(tape, pose)
        assert safe == 1
        records.append([float(pose[0]), safe, why])
        pose = unicycle_step(pose, tape[0], .25)
    results['recovery_allowance_ratchet'] = dict(start_penetration_m=.001,
        final_penetration_m=float(pose[0]), steps=records)
    n.planner.collision_signed_distance_state_np = lambda s: math.nan
    results['nonfinite_clearance_fails_open'] = n._simple_plan_safe_to_execute(tape, pose)
    assert results['nonfinite_clearance_fails_open'][0] == 12

    for mode in ('inside', 'on_boundary', 'outside', 'stale', 'wrong_frame', 'negative_covariance', 'indefinite_covariance', 'future_stamp'):
        n = logger(out, 'goal_' + mode)
        distance = .349 if mode != 'outside' else .351
        if mode == 'on_boundary': distance = .35
        n.goal_msg = goal(distance, 0.)
        n.planner_belief_msg = belief(t=0. if mode == 'stale' else (100. if mode == 'future_stamp' else 10.),
            frame='odom' if mode == 'wrong_frame' else 'map_bev',
            variance=-1. if mode == 'negative_covariance' else .01,
            cross=.02 if mode == 'indefinite_covariance' else 0.)
        tick(n, 10.); tick(n, 12.)
        assert n._stop_requested == (mode != 'outside')
        results['logger_goal_' + mode] = outcome(n)

    n = logger(out, 'goal_changed_hold')
    n.goal_msg = goal(.30, 0.)
    tick(n, 10.)
    n._goal_cb(goal(-.30, 0.))
    tick(n, 12.)
    assert n._stop_requested
    results['logger_goal_change_inherits_hold'] = outcome(n)

    for mode in ('rotation', 'idle_replan', 'stale_belief', 'oscillation', 'mixed_activity'):
        n = logger(out, 'stuck_' + mode)
        for t in range(9):
            x = .3*math.sin(math.pi*t/4.) if mode == 'oscillation' else 0.
            yaw = .2*t if mode == 'rotation' else 0.
            n.planner_belief_msg = belief(x=x, yaw=yaw, t=0. if mode == 'stale_belief' else t)
            cmd = (0., .2) if mode == 'rotation' else ((0., 0.) if mode == 'idle_replan' else (.22, 0.))
            if mode == 'mixed_activity': cmd = (.22, 0.) if t % 4 == 0 else (0., 0.)
            tick(n, t, *cmd)
        results['stuck_' + mode] = outcome(n)
        results['stuck_' + mode]['window'] = n._motion_window_stats(8., 8.)
        results['stuck_' + mode]['heading_change_rad'] = 1.6 if mode == 'rotation' else 0.
        assert n._stop_requested == (mode != 'mixed_activity')

    n = logger(out, 'silent_contacts_goal')
    n._first_cmd_stamp = None
    n.goal_msg = goal(.3, 0.)
    tick(n, 10., .22, 0., pubs=1)
    n.publishers = 0
    tick(n, 12., 0., 0., pubs=0)
    assert n._stop_requested and n._valid_run and n._contact_messages_seen == 0
    results['silent_contact_publisher_then_disappears'] = outcome(n)
    n = logger(out, 'contact_no_publisher')
    n._first_cmd_stamp = None
    n.publishers = 0
    tick(n, 10., .22, 0.)
    assert n.terminal_snapshot['reason'] == 'infra_invalid_contact_channel'
    results['no_contact_publisher_first_command'] = outcome(n)

    # Terminal causes with the same operational pose 1 m short of the goal.
    for cause in ('planner_clearance_stop', 'controller_zero', 'tape_exhausted', 'watchdog',
                  'goal_change_wait', 'contact', 'timeout', 'fatal_stop'):
        n = tracker(state=(0., 0., 0.))
        applied = []
        if cause == 'planner_clearance_stop':
            n._simple_plan_safe_to_execute = lambda *a: (0, 'driveable_clearance_violation_step_0:-0.01')
            n._plan_once()
        elif cause == 'controller_zero':
            n._dispatch_local_controller = lambda *a: np.zeros((12, 2))
            n._plan_once()
        elif cause == 'tape_exhausted':
            n._clock.seconds = 10.2
            n._publish_active_plan_command()
        elif cause == 'fatal_stop':
            n._fatal_stop_triggered = True
            n._publish_safe_stop_command()
        elif cause == 'watchdog':
            adapter = object.__new__(ActuationNoiseNode)
            adapter.command_timeout_s = .5
            adapter._last_input_stamp_s = 9.
            adapter._watchdog_stopped = False
            adapter._pub = NS(publish=applied.append)
            adapter.get_clock = lambda: _Clock(10.)
            adapter.get_logger = lambda: _Logger()
            adapter._watchdog_tick()
        record = logger(out, 'stop_' + cause)
        if cause == 'contact':
            record._contacts_cb(NS(header=NS(stamp=stamp(8.123)), contacts=[NS(
                collision1=NS(name='turtlebot3::base_link::collision'),
                collision2=NS(name='warehouse_rack::link::collision'))]))
        elif cause == 'timeout':
            record.stuck_window_s = 0.
            tick(record, 600.)
        else:
            # Model unchanged belief and stopped output; live fatal teardown may
            # instead interrupt logger. This isolates its absence of cause input.
            for t in range(9): tick(record, t)
        results['stop_cause_' + cause] = dict(origin_commands=commands(n),
            adapter_commands=[[m.linear.x, m.angular.z] for m in applied],
            **outcome(record))

    # Success summary causes no command publication or active tape cancellation.
    n = tracker(); n._plan_once()
    record = logger(out, 'success_while_moving')
    record.goal_msg = goal(.3, 0.)
    tick(record, 10., .22, 0.); tick(record, 12., .22, 0.)
    before = len(commands(n))
    n._clock.seconds = 12.1; n._publish_active_plan_command()
    results['summary_not_physical_stop'] = dict(**outcome(record),
        additional_commands=commands(n)[before:], tape_retained=n._active_controls is not None)
    assert commands(n)[-1][0] > 0. and record._stop_requested

    assert hashes() == initial_hashes, 'Audited runtime sources changed during probe'
    payload = dict(kind='deterministic_software_audit_observed_behaviour_not_repaired_invariants',
        source_sha256=initial_hashes, cases=len(results), results=results,
        scope='No ROS init, DDS, Gazebo, camera inference, or physical outcome. Real node methods '
              'with controlled boundaries; logger distance/dispatch source statements extracted unchanged.')
    (out / 'probe_results.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(cases=len(results), results={k: (v.get('snapshot') or {x:y for x,y in v.items()
        if x not in ('summary', 'log', 'steps')}) if isinstance(v, dict) else v
        for k,v in results.items()}), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run(args.out.resolve())
