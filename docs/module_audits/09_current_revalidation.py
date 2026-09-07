"""Recheck audit 09 after packages A/B and the typed tracker guard.

Uses current production methods with controlled boundaries. Keeps the original
50-case baseline and its observed-defect assertions unchanged. No ROS graph.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np

SPEC = importlib.util.spec_from_file_location('audit09_baseline', Path(__file__).with_name('09_tracking_probe.py'))
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def tracker(*args, **kwargs):
    n = b.tracker(*args, **kwargs)
    n._command_stop_generation = 0
    n._active_plan_request = None
    return n


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    before = b.hashes()
    results = {}
    for mode in ('negative_variance', 'indefinite_xy', 'wrong_frame', 'stale', 'nan_position'):
        n = b.mission()
        n._belief_cb(b.belief())
        m = b.belief(variance=-1. if mode == 'negative_variance' else .01,
                     cross=.02 if mode == 'indefinite_xy' else 0.,
                     frame='odom' if mode == 'wrong_frame' else 'map_bev',
                     t=0. if mode == 'stale' else 10.,
                     x=math.nan if mode == 'nan_position' else 0.)
        n._belief_cb(m)
        n._send_goal()
        assert n._belief_ready and n.sent
        results['mission_' + mode] = dict(ready=n._belief_ready, waypoint=n.wp_idx,
            goal_stamp=n.sent[-1].header.stamp.sec, sim_now=10.)

    for mode in ('goal', 'belief', 'ordinary_stop', 'fatal_stop'):
        n = tracker()
        original = n._dispatch_local_controller
        def dispatch(m, target):
            if mode == 'goal': n._goal_cb(b.goal(-2., 0.))
            if mode == 'belief': n.belief_m = np.array([0., 0., math.pi])
            if mode in ('ordinary_stop', 'fatal_stop'):
                n._fatal_stop_triggered = mode == 'fatal_stop'
                n._publish_safe_stop_command()
            return original(m, target)
        n._dispatch_local_controller = dispatch
        n._plan_once()
        results['during_local_' + mode] = dict(commands=b.commands(n),
            tape_present=n._active_controls is not None, latest_goal=n.goal_msg.pose.position.x)
        assert b.commands(n)[-1][0] == (0. if mode.endswith('stop') else .22)

    for delay in (.25, .2501):
        n = tracker()
        n._active_plan_request = n._capture_plan_request()
        n._clock.seconds += delay
        disposition = n._install_control_tape([[.2, 0.]], original_len=1)
        assert disposition == ('installed' if delay == .25 else 'expired')
        results['installation_age_' + str(delay)] = dict(disposition=disposition, commands=b.commands(n))

    for delta in (.999, 1., 1.001):
        n = tracker(((1., 0.),), state=(1., 0., 0.))
        n._global_solve_done = True
        n.optimizer_route_seed_mode = 'explicit'
        n.global_planner = b.NS(_parse_initial_routes=lambda x: x,
            plan=lambda *a: (_ for _ in ()).throw(ValueError('controlled failure')))
        n.goal_msg = b.goal(1.+delta, 0.)
        n._plan_once()
        results['goal_move_' + str(delta)] = dict(route_goal=n._global_goal_xy.tolist(), commands=b.commands(n))
        assert n._global_goal_xy[0] == (1. if delta <= 1. else 1.+delta)

    n = tracker()
    n._hier_phase, n._global_solve_done, n.optimizer_route_seed_mode = 'GLOBAL', False, 'explicit'
    n._publish_plan_and_metrics = lambda *a, **kw: None
    n._save_global_plan_artifacts = lambda *a: None
    def solve(m, S, target):
        n._goal_cb(b.goal(-2., 0.))
        n.belief_m = np.array([0., 1., math.pi])
        return b.NS(states=np.array([[0., 0., 0.], [1., 0., 0.]]), rollout_valid=False)
    n.global_planner = b.NS(plan=solve)
    n._plan_once()
    assert n._global_goal_xy[0] == 1. and n._hier_phase == 'LOCAL'
    results['invalid_global_result_after_goal_and_belief_change'] = dict(route=n._waypoints,
        route_goal=n._global_goal_xy.tolist(), latest_goal=n.goal_msg.pose.position.x)

    n = tracker()
    n._plan_once()
    n.belief_m = np.array([0., 0., math.pi])
    n._clock.seconds += .1
    n._publish_active_plan_command()
    assert b.commands(n)[-1] == [.22, 0.]
    results['belief_update_active_tape'] = b.commands(n)
    n = tracker(((.04, 0.), (1., 0.)))
    n.state_msg = b.belief(frame='odom')
    n._plan_once()
    assert n._wp_idx == 1 and b.commands(n)[-1][0] > 0.
    results['wrong_frame_handoff'] = dict(index=n._wp_idx, commands=b.commands(n))
    n = tracker([(0., 0.), (.04, 0.), (.04, 0.), (.04, .04), (1., .04)])
    n._plan_once()
    assert n._wp_idx == 4
    results['short_duplicate_sharp_route'] = dict(index=n._wp_idx, commands=b.commands(n))

    n = tracker()
    n.planner.collision_cost_model = object()
    n.planner.collision_signed_distance_state_np = lambda s: -float(s[1])
    pose, target = np.array([0., .001, .08]), np.array([10., .001+10.*math.tan(.08)])
    steps = []
    for _ in range(30):
        tape = n._simple_local_plan(pose, target)
        decision = n._simple_plan_safe_to_execute(tape, pose)
        assert decision.safe_steps == 1
        steps.append(dict(penetration_m=float(pose[1]), reason=decision.reason,
                          failure=decision.failure.value, first_control=tape[0].tolist()))
        pose = b.unicycle_step(pose, tape[0], .25)
    assert pose[1] > .13
    results['renewed_recovery_allowance'] = dict(initial_penetration_m=.001,
        final_penetration_m=float(pose[1]), steps=steps)
    for label, value in [('nan', math.nan), ('negative_infinity', -math.inf), ('empty_scene_positive_infinity', math.inf)]:
        n.planner.collision_signed_distance_state_np = lambda s: value
        result = n._simple_plan_safe_to_execute(tape, pose)
        assert result.safe_steps == (12 if value == math.inf else 0)
        results['geometry_' + label] = dict(safe_steps=result.safe_steps, failure=result.failure.value, reason=result.reason)

    for mode in ('inside', 'on_boundary', 'outside', 'stale', 'wrong_frame', 'negative_covariance', 'indefinite_covariance', 'future_stamp'):
        n = b.logger(out, 'goal_' + mode)
        distance = .35 if mode == 'on_boundary' else (.351 if mode == 'outside' else .349)
        n.goal_msg = b.goal(distance, 0.)
        n.planner_belief_msg = b.belief(t=0. if mode == 'stale' else (100. if mode == 'future_stamp' else 10.),
            frame='odom' if mode == 'wrong_frame' else 'map_bev',
            variance=-1. if mode == 'negative_covariance' else .01,
            cross=.02 if mode == 'indefinite_covariance' else 0.)
        b.tick(n, 10.)
        if mode not in ('stale', 'future_stamp'): n.planner_belief_msg.header.stamp = b.stamp(12.)
        b.tick(n, 12.)
        assert n._stop_requested == (mode != 'outside')
        results['logger_' + mode] = b.outcome(n)

    n = b.logger(out, 'goal_change_hold')
    n.goal_msg = b.goal(.3, 0.)
    b.tick(n, 10.)
    n._goal_cb(b.goal(-.3, 0.))
    n.planner_belief_msg.header.stamp = b.stamp(12.)
    b.tick(n, 12.)
    assert n._stop_requested
    results['changed_goal_inherits_hold'] = b.outcome(n)
    for mode in ('rotation', 'idle_replan', 'stale_belief'):
        n = b.logger(out, 'stuck_' + mode)
        for t in range(9):
            n.planner_belief_msg = b.belief(yaw=-math.pi/2.+.2*t if mode == 'rotation' else 0.,
                t=0. if mode == 'stale_belief' else t)
            cmd = (0., .2) if mode == 'rotation' else ((0., 0.) if mode == 'idle_replan' else (.22, 0.))
            b.tick(n, t, *cmd)
        assert n.terminal_snapshot['reason'] == 'stuck'
        results['stuck_' + mode] = b.outcome(n)

    n = tracker()
    n._plan_once()
    log = b.logger(out, 'summary_without_stop')
    log.goal_msg = b.goal(.3, 0.)
    b.tick(log, 10., .22, 0.)
    log.planner_belief_msg.header.stamp = b.stamp(12.)
    b.tick(log, 12., .22, 0.)
    n._clock.seconds += .1
    n._publish_active_plan_command()
    assert log._stop_requested and b.commands(n)[-1][0] > 0.
    results['summary_without_stop'] = dict(logger=b.outcome(log), subsequent_commands=b.commands(n))

    for rel, expected in before.items():
        target = out / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(b.ROOT / rel, target)
    assert b.hashes() == before, 'Source changed during invocation; do not treat as a stable probe'
    result = dict(kind='current_revalidation_after_A_B_and_typed_guard', cases=len(results),
        source_sha256=before, results=results,
        limit='Current production methods with fake scheduling/transport. Snapshot is not a hermetic dependency bundle or physical navigation result.')
    (out / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(results), results={k:(v.get('snapshot') or v) if isinstance(v,dict) else v
          for k,v in results.items() if not k.startswith('logger_') and not k.startswith('stuck_') and k!='summary_without_stop'}),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
