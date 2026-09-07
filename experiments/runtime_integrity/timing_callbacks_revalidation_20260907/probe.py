"""Current-source timing revalidation after repair packages A/B.

These assertions pin observed behavior, including defects, not desired behavior.
No ROS initialization, executor, simulator, sleep, or experiment I/O. Existing
baseline helpers are imported without running their main or changing their files.
Source hashes bracket execution, and retained source copies anchor report lines.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import argparse
import importlib.util
import inspect
import json
import math
import sys
import threading
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    'timing_baseline', ROOT / 'experiments/runtime_integrity/timing_callbacks_20260906/probe.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

from geometry_msgs.msg import PoseStamped, Twist
from planning.nodes.efe_agent_node import EfeAgentNode
from test_command_installation import _install_node
from test_planner_node_correction_wiring import _Publisher


def command_receipt_completion_order():
    n = baseline.node_at(anchor=9., now=12.)
    # This also exercises the configured automatic command fallback when odometry
    # has no samples, not only the explicitly command-only variant.
    n.use_odom_for_predict = True
    n._odom_log = []
    entered, release = threading.Event(), threading.Event()

    class ControlledClock:
        calls = 0

        def now(self):
            self.calls += 1
            t = baseline._Clock(10. if self.calls == 1 else 11.).now()
            if self.calls == 1:
                entered.set()
                baseline.wait(release)
            return t

    n.get_clock = lambda: clock
    clock = ControlledClock()
    old, new = Twist(), Twist()
    old.linear.x, new.linear.x = .1, .2
    a = baseline.run_thread(lambda: n._cmd_cb(old))
    try:
        baseline.wait(entered)
        n._cmd_cb(new)
    finally:
        release.set()
    baseline.join(a)
    history = list(n._cmd_log)
    held = float(n.last_cmd[0])
    n.planner.predict = baseline.linear_predict
    args = (n.belief_m.copy(), n.belief_S.copy(), baseline.stamp(9.),
            baseline.stamp(12.), np.zeros(2), 3.)
    observed, _, meta = n._replay_cmd_log_interval(*args)
    n._cmd_log = sorted(history)  # Oracle only; not a proposed retrospective repair.
    ordered, _, _ = n._replay_cmd_log_interval(*args)
    assert [p[0] for p in history] == [11., 10.]
    assert held == .1 and math.isclose(observed[0], .2)
    assert math.isclose(ordered[0], .3)
    return dict(history=history, retained_last_v=held, replay_x=float(observed[0]),
                chronologically_ordered_oracle_x=float(ordered[0]),
                source_code=meta['motion_replay_source_code'], odometry_available=False)


def reset_retains_anchor_and_new_request_can_install():
    n = baseline.node_at(anchor=100., now=100.1)
    n._odom_cb(baseline.odom(100.))
    n._seen_state_source_batch_ids.add('previous_epoch')
    n._clock.seconds = 1.
    n._odom_cb(baseline.odom(1.))
    n._belief_publish_tick()
    m, P, meta = n._resolve_belief_for_planning()
    assert m is not None and meta['belief_age_s'] == 0.
    assert not n.planner_belief_pub.messages
    assert [x[0] for x in n._odom_log] == [100.]
    assert n._odom_refused_old == 1
    n._state_correction_envelope_cb(baseline.envelope('new_epoch', .95))
    row = json.loads(n.correction_assimilation_pub.published[-1])
    assert row['reason'] == 'not_newer_than_belief'

    # Package B validates this newly captured request, not the epoch of m/P used
    # by a controller to produce it. It correctly rejects pre-jump requests; this
    # is the distinct post-jump path. No claim of a complete solver/executor run.
    execution = _install_node()
    execution._clock.seconds = 1.
    execution._active_plan_request = execution._capture_plan_request()
    status = execution._install_control_tape(np.array([[.1, 0.]]), original_len=1)
    assert status == 'installed' and execution.cmd_pub.messages[-1].linear.x == .1
    return dict(ros_now=1., retained_anchor=n._stamp_to_float(n.belief_stamp),
                reported_planning_age=meta['belief_age_s'], retained_odom_stamps=[100.],
                new_epoch_odom_refused=True, public_messages=0,
                retained_old_identity='previous_epoch' in n._seen_state_source_batch_ids,
                new_epoch_correction=row['reason'], fresh_request_installation=status,
                composed_method_boundaries=True, full_solver_exercised=False)


def reset_during_public_prediction():
    n = baseline.node_at(anchor=100., now=100.1)
    entered, release = threading.Event(), threading.Event()

    def predict(m, P, u, age, target):
        entered.set()
        baseline.wait(release)
        return m, P

    n._predict_belief_to_now = predict
    worker = baseline.run_thread(n._belief_publish_tick)
    try:
        baseline.wait(entered)
        n._clock.seconds = 1.
    finally:
        release.set()
    baseline.join(worker)
    msg, = n.planner_belief_pub.messages
    assert math.isclose(n._stamp_to_float(msg.header.stamp), 100.1)
    return dict(clock_after_reset=1., published_pre_reset_target=100.1,
                in_flight_prediction_invalidated=False)


def committed_correction_without_identity_or_terminal_record():
    n = baseline.node_at(anchor=9.9)
    faults = []
    n._fatal_experiment_stop = lambda reason, exc: faults.append([reason, str(exc)])

    def fail(_message):
        raise RuntimeError('injected diagnostic publisher failure')

    n.pixel_correction_diag_pub = SimpleNamespace(publish=fail)
    n._state_correction_envelope_cb(baseline.envelope('committed_but_unrecorded'))
    assert math.isclose(n._stamp_to_float(n.belief_stamp), 9.95)
    assert n.belief_m[0] > 0.
    assert not n._seen_state_source_batch_ids
    assert not n.correction_assimilation_pub.published and len(faults) == 1
    first_x = float(n.belief_m[0])
    # Demonstrate absence of committed identity, not normal post-fatal liveness:
    # the real fatal callback shuts down; only this test stub permits redelivery.
    n.pixel_correction_diag_pub = _Publisher()
    n._state_correction_envelope_cb(baseline.envelope('committed_but_unrecorded'))
    row = json.loads(n.correction_assimilation_pub.published[-1])
    assert row['status'] == 'dropped' and row['reason'] == 'not_newer_than_belief'
    assert float(n.belief_m[0]) == first_x
    return dict(posterior_x=first_x, committed_anchor=9.95,
                identity_retained_at_failure=False, terminal_records_at_failure=0,
                fatal_called=True, redelivery_with_stubbed_shutdown=row['reason'],
                double_assimilation=False)


def goal_and_progress_signature_are_different_events():
    n = baseline.node_at()
    a, b = PoseStamped(), PoseStamped()
    a.header.frame_id = b.header.frame_id = 'map_bev'
    a.pose.position.x, b.pose.position.x = 1., 2.
    entered, release = threading.Event(), threading.Event()
    update = n._update_goal_progress_origin

    def delayed_update(msg):
        if msg is a:
            entered.set()
            baseline.wait(release)
        update(msg)

    n._update_goal_progress_origin = delayed_update
    first = baseline.run_thread(lambda: n._goal_cb(a))
    try:
        baseline.wait(entered)
        n._goal_cb(b)
    finally:
        release.set()
    baseline.join(first)
    assert n.goal_msg is b and n._goal_signature[1] == 1.
    return dict(retained_goal_x=2., retained_signature_x=1., torn_progress_metadata=True)


def bounded_unpredicted_publication_interval():
    n = baseline.node_at(anchor=10., now=10.0005)
    calls = []

    def predict(*args):
        calls.append(args)
        raise AssertionError('not expected for the current 1 ms cutoff')

    n._predict_belief_to_now = predict
    n._belief_publish_tick()
    msg, = n.planner_belief_pub.messages
    assert not calls and math.isclose(n._stamp_to_float(msg.header.stamp), 10.0005)
    return dict(anchor_state_time=10., header_time=10.0005, prediction_called=False,
                configured_cutoff_s=.001, classification='bounded numerical approximation')


def mission_system_jumps_current_default():
    n = object.__new__(baseline.GoalMissionNode)
    wall = {'seconds': 99.}
    n.wall_clock = SimpleNamespace(now=lambda: baseline.Time(
        seconds=wall['seconds'], clock_type=baseline.ClockType.SYSTEM_TIME))
    n.start_time = baseline.Time(seconds=100., clock_type=baseline.ClockType.SYSTEM_TIME)
    n.delay, n.sent_count, n.repeat_count = 3., 0, 0
    n.wait_for_belief, n._belief_xy = False, None
    n.waypoints, n.wp_idx, n.frame_id = [(1., 2.)], 0, 'map_bev'
    n.goal_pub, n.get_logger = baseline.Publisher(), lambda: baseline._Logger()
    n.repeat_unchanged_goal = False
    counts = []
    for t in [99., 103., 90., 104.]:
        wall['seconds'] = t
        n._send_goal()
        counts.append(n.sent_count)
    assert counts == [0, 1, 1, 1]
    return dict(system_now=[99., 103., 90., 104.], sent_counts=counts,
                startup_uses_system_elapsed=True, repeat_unchanged_goal_default=False,
                classification='intentional wall startup, civil jump sensitivity')


PROBES = [
    baseline.publication_overtakes,
    baseline.same_stamp_supersession_guard,
    baseline.full_covariance_snapshot,
    baseline.frame_from_different_event,
    baseline.equal_time_revision_lost_by_consumer,
    baseline.planner_metadata_from_superseding_update,
    baseline.correction_worker_starvation,
    baseline.simultaneous_duplicate_envelopes,
    baseline.pixel_timer_reads_new_measurement_with_old_stamp,
    baseline.clock_domain_error_fails_open,
    baseline.mission_clock_and_readiness,
    baseline.camera_batch_reset,
    command_receipt_completion_order,
    reset_retains_anchor_and_new_request_can_install,
    reset_during_public_prediction,
    committed_correction_without_identity_or_terminal_record,
    goal_and_progress_signature_are_different_events,
    bounded_unpredicted_publication_interval,
    mission_system_jumps_current_default,
]


def file_hashes(manifest):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in manifest['sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results.json')
    options = parser.parse_args()
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    report = dict(source_before=file_hashes(manifest), results={}, failures={},
                  meaning='Passing probes confirm observed behavior including defects.')
    for fn in PROBES:
        try:
            report['results'][fn.__name__] = fn()
        except Exception:
            report['failures'][fn.__name__] = traceback.format_exc()
    report['source_after'] = file_hashes(manifest)
    report['import_paths'] = {
        cls.__name__: inspect.getfile(cls) for cls in [baseline.UnicyclePlannerNode,
        EfeAgentNode, baseline.CameraManagerNode, baseline.GoalMissionNode]}
    report['manifest_matches_before'] = manifest['sha256'] == report['source_before']
    report['unchanged_during_execution'] = report['source_before'] == report['source_after']
    report['passed'] = len(report['results'])
    (HERE / options.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(bool(report['failures']))
