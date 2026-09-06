"""Deterministic timing audit of real node methods, without a ROS graph.

These probes assert the observed baseline, including defective behavior. A
successful run means reproduction succeeded, not that desired invariants pass.
Synchronization uses events/lock contention, never timing sleeps. Run from the
repository root after sourcing install/setup.bash. Only this directory is written.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import math
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/planning'))
sys.path.insert(0, str(ROOT / 'src/experiments'))

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.clock import ClockType
from rclpy.time import Time

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.core import motion_history
from reliability.nodes.camera_manager_node import CameraManagerNode
from perception.core.four_camera_batch import CAMERA_ORDER, FourCameraBatcher, PendingFrame
from experiments.nodes.goal_mission_node import GoalMissionNode
from test_planner_node_state_correction import make_state_node, state_msg
from test_planner_node_correction_wiring import _Clock, _Logger, stamp
from test_runtime_transactions import Publisher


def node_at(anchor=9.0, now=10.0):
    n = make_state_node(belief_stamp_s=anchor, now_s=now)
    n.heading_update_mode = 'coupled'
    n.pixel_timeout_s = .5
    n.state_reanchor_m = 0.0
    n.require_state_correction_envelope = True
    n._seen_state_source_batch_ids = set()
    n._seen_map_observation_stamps = {}
    n.use_diagnostic_odom_localization = False
    n.use_odom_for_predict = True
    n.odom_vel = np.zeros(2)
    n.planner_belief_pub = Publisher()
    n._CMD_LOG_MAX_S = 60.0
    n.stale_belief_inflate_m2_per_s = 0.0
    n.stale_belief_inflate_cap_m2 = 0.0
    n._bev_affine = None
    n.bev_y_calibration_offset_m = 0.0
    n._latest_detection_diag = None
    n._goal_received_logged = False
    n._goal_signature = None
    n._goal_progress_start_dist_m = None
    return n


def wait(event):
    assert event.wait(5), 'synchronization barrier was not reached'


def run_thread(fn):
    errors = []

    def guarded():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=guarded, daemon=True)
    t.start()
    return t, errors


def join(pair):
    t, errors = pair
    t.join(5)
    assert not t.is_alive(), 'thread did not finish'
    if errors:
        raise errors[0]


def odom(t, v=1.0, w=0.0, yaw=0.0):
    msg = Odometry()
    msg.header.stamp = stamp(t)
    msg.pose.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.pose.orientation.w = math.cos(yaw / 2)
    msg.twist.twist.linear.x = v
    msg.twist.twist.angular.z = w
    return msg


def linear_predict(m, P, u, dt):
    # Deliberately straight synthetic motion; audit interval accounting only.
    m, P = m.copy(), P.copy()
    m[0] += u[0] * dt
    m[2] += u[1] * dt
    P += np.eye(3) * dt * .001
    return m, P


def envelope(batch='frame', t=9.95):
    return SimpleNamespace(data=json.dumps(dict(
        schema_version=1, frame_id='map_bev', source_batch_id=batch,
        correction_stamp=t, xy=[.1, 0.0], covariance_m2=[[.03, 0.0], [0.0, .03]],
    )))


def publication_overtakes():
    n = node_at()
    entered, release = threading.Event(), threading.Event()

    def predict(m, P, u, age, target):
        target_s = n._stamp_to_float(target)
        if target_s == 10.0:
            entered.set()
            wait(release)
        return np.array([target_s, 0., .2]), np.eye(3) * target_s

    n._predict_belief_to_now = predict
    first = run_thread(n._belief_publish_tick)
    try:
        wait(entered)
        n._clock.seconds = 10.2
        n._belief_publish_tick()
    finally:
        release.set()
    join(first)
    stamps = [n._stamp_to_float(m.header.stamp) for m in n.planner_belief_pub.messages]
    assert stamps == [10.2, 10.0]
    for msg in n.planner_belief_pub.messages:
        assert msg.pose.pose.position.x == n._stamp_to_float(msg.header.stamp)
        assert msg.pose.covariance[0] == msg.pose.pose.position.x
        assert msg.header.frame_id == 'map_bev'
    return dict(published_stamps=stamps, each_message_coherent=True,
                latest_message_regresses=True)


def same_stamp_supersession_guard():
    n = node_at()
    before = n.belief_S.copy()

    def predict(m, P, u, age, target):
        # Real same-state-time update, supported by the per-camera interface.
        n._apply_metric_correction(stamp(9.), np.array([.1, 0.]),
                                   np.eye(2) * .03, allow_same_stamp=True)
        return m, P

    n._clock.seconds = 9.1
    n._predict_belief_to_now = predict
    n._belief_publish_tick()
    assert n._stamp_to_float(n.belief_stamp) == 9.
    assert n.belief_S[0, 0] < before[0, 0]
    assert not n.planner_belief_pub.messages
    return dict(anchor_stamp_unchanged=9.0, correction_applied=True,
                superseded_prediction_published=False)


def full_covariance_snapshot():
    n = node_at(anchor=10.)
    P = np.array([[.4, .03, .02], [.03, .3, -.01], [.02, -.01, .2]])
    n.belief_m = np.array([1., 2., .7])
    n.belief_S = P.copy()
    n._belief_publish_tick()
    msg, = n.planner_belief_pub.messages
    cov = np.asarray(msg.pose.covariance).reshape(6, 6)
    np.testing.assert_allclose(cov[np.ix_([0, 1, 5], [0, 1, 5])], P)
    assert msg.header.frame_id == 'map_bev'
    assert n._stamp_to_float(msg.header.stamp) == 10.
    assert msg.pose.pose.position.x == 1. and msg.pose.pose.position.y == 2.
    assert math.isclose(2 * math.atan2(msg.pose.pose.orientation.z,
                                    msg.pose.pose.orientation.w), .7)
    return dict(full_planar_covariance_preserved=True, frame='map_bev', stamp=10.)


def frame_from_different_event():
    n = node_at()
    n.state_msg = state_msg(0., 0., seconds=9.)
    n.state_msg.header.frame_id = 'map_bev'

    def predict(m, P, u, age, target):
        other = state_msg(50., 50., seconds=9.9)
        other.header.frame_id = 'camera_optical'
        # Compatibility callback may change state_msg, but cannot correct the
        # envelope-owned filter. Its frame is nevertheless read by publication.
        n._state_cb(other)
        return m, P

    n._predict_belief_to_now = predict
    n._belief_publish_tick()
    msg, = n.planner_belief_pub.messages
    assert msg.header.frame_id == 'camera_optical'
    assert msg.pose.pose.position.x == 0.
    return dict(belief_coordinates_frame='map_bev', published_frame=msg.header.frame_id)


def equal_time_revision_lost_by_consumer():
    n = node_at(anchor=9.9)
    n._belief_publish_tick()
    n._state_correction_envelope_cb(envelope())
    n._belief_publish_tick()  # /clock has not advanced between callbacks.
    first, second = n.planner_belief_pub.messages
    assert first.header.stamp == second.header.stamp
    assert second.pose.pose.position.x > first.pose.pose.position.x
    manager = SimpleNamespace(frame_id='map_bev', _belief_query_history=deque(),
                              get_logger=lambda: _Logger())
    CameraManagerNode._belief_query_callback(manager, first)
    CameraManagerNode._belief_query_callback(manager, second)
    assert len(manager._belief_query_history) == 1
    kept_x = manager._belief_query_history[0][1][0]
    assert kept_x == first.pose.pose.position.x
    return dict(stamp=10., published_x=[first.pose.pose.position.x, second.pose.pose.position.x],
                manager_kept_x=kept_x, manager_retained_revisions=1)


def odometry_callback_finishes_out_of_order():
    n = node_at(anchor=10., now=12.)
    entered, release = threading.Event(), threading.Event()
    convert = n._stamp_to_float

    def slow_convert(s):
        t = convert(s)
        if t == 10.:
            entered.set()
            wait(release)
        return t

    n._stamp_to_float = slow_convert
    first = run_thread(lambda: n._odom_cb(odom(10., v=1., yaw=.1)))
    try:
        wait(entered)
        n._odom_cb(odom(11., v=2., yaw=.2))
    finally:
        release.set()
    join(first)
    assert [x[0] for x in n._odom_log] == [11., 10.]
    assert n.odom_vel[0] == 1. and math.isclose(n._latest_odom_yaw, .2)
    n.planner.predict = linear_predict
    m_sorted, _, _ = replay_sorted_reference(n)
    m, _, info = n._replay_cmd_log_interval(np.zeros(3), np.eye(3),
                                           stamp(9.), stamp(12.), np.zeros(2), 3.)
    assert math.isclose(m[0], 2.) and math.isclose(m_sorted[0], 3.)
    return dict(log_stamps=[11., 10.], velocity_sample_time=10., yaw_sample_time=11.,
                replay_x=m[0], timestamp_order_reference_x=m_sorted[0],
                reports_fallback=info['cmd_replay_used_fallback'])


def replay_sorted_reference(n):
    entries = n._odom_log
    n._odom_log = sorted(entries)
    try:
        return n._replay_cmd_log_interval(np.zeros(3), np.eye(3),
                                          stamp(9.), stamp(12.), np.zeros(2), 3.)
    finally:
        n._odom_log = entries


def coverage_check_and_replay_use_different_buffers():
    n = node_at(anchor=1., now=60.9)
    n.planner.predict = linear_predict
    n._odom_log = [(float(i) / 10., 1., 0.) for i in range(10, 610)]
    checker = motion_history.covers_interval
    checks = []

    def check_then_receive(entries, start, end, gap):
        supported = checker(entries, start, end, gap)
        checks.append(supported)
        # A normal input .15 s after the camera target trims start support out of
        # the 60 s ring between the support check and the real replay snapshot.
        n._odom_cb(odom(61.05))
        return supported

    motion_history.covers_interval = check_then_receive
    try:
        n._advance_belief_over_outage(stamp(60.9), 59.9)
    finally:
        motion_history.covers_interval = checker
    assert checks == [True]
    assert math.isclose(n.belief_m[0], 59.8)
    assert not checker(n._odom_log, 1., 60.9, 1.5)
    return dict(checked_supported=True, actually_replayed_buffer_supported=False,
                committed_stamp=n._stamp_to_float(n.belief_stamp),
                expected_motion=59.9, committed_motion=float(n.belief_m[0]))


def planner_metadata_from_superseding_update():
    n = node_at(anchor=9.9)

    def predict(m, P, u, age, target):
        n._state_correction_envelope_cb(envelope())
        return m, P

    n._predict_belief_to_now = predict
    m, P, meta = n._resolve_belief_for_planning()
    assert m[0] == 0. and n.belief_m[0] > 0.
    assert math.isclose(n._stamp_to_float(meta['belief_stamp']), 9.95)
    return dict(input_anchor=9.9, returned_anchor=n._stamp_to_float(meta['belief_stamp']),
                returned_x=float(m[0]), committed_x=float(n.belief_m[0]),
                recursive_anchor_after_correction=n._stamp_to_float(n.belief_stamp))


def planning_bootstrap_bypasses_envelope():
    n = node_at(anchor=9.9)
    n.belief_m = n.belief_S = n.belief_stamp = None
    msg = state_msg(.1, 0., seconds=9.95)
    msg.header.frame_id = 'map_bev'
    n._state_cb(msg)
    assert n.belief_m is None
    n._resolve_belief_for_planning()
    assert n.belief_m is not None and not n.correction_assimilation_pub.published
    n._state_correction_envelope_cb(envelope())
    row = json.loads(n.correction_assimilation_pub.published[0])
    assert row['status'] == 'dropped' and row['reason'] == 'not_newer_than_belief'
    return dict(anonymous_bootstrap_x=float(n.belief_m[0]), identified_event_status=row['status'],
                reason=row['reason'])


def backward_reset_keeps_epoch_state():
    n = node_at(anchor=100., now=100.1)
    n._odom_cb(odom(100.))
    n._seen_state_source_batch_ids.add('previous_epoch')
    n._clock.seconds = 1.
    n._odom_cb(odom(1.))
    n._belief_publish_tick()
    m, P, meta = n._resolve_belief_for_planning()
    assert not n.planner_belief_pub.messages
    assert m is not None and meta['belief_age_s'] == 0.
    assert n._stamp_to_float(n.belief_stamp) == 100.
    n._state_correction_envelope_cb(envelope('new_epoch', .95))
    row = json.loads(n.correction_assimilation_pub.published[-1])
    assert row['reason'] == 'not_newer_than_belief'
    return dict(ros_now=1., retained_anchor=100., planning_age=meta['belief_age_s'],
                published_messages=0, odom_stamps=[x[0] for x in n._odom_log],
                retained_identity='previous_epoch' in n._seen_state_source_batch_ids,
                first_new_epoch_correction=row['reason'])


def camera_batch_reset():
    b = FourCameraBatcher(max_stamp_skew_s=.05, max_pending_wall_s=1.)
    for camera in CAMERA_ORDER:
        first = b.offer(PendingFrame(camera, 100_000_000_000, 100., 1000., None))
    reset = b.offer(PendingFrame(CAMERA_ORDER[0], 1_000_000_000, 1., 1001., None))
    assert first.batch is not None and reset.status == 'out_of_order'
    return dict(previous_epoch_status=first.status, reset_frame_status=reset.status)


def mission_clock_and_readiness():
    n = object.__new__(GoalMissionNode)
    n.wall_clock = SimpleNamespace(now=lambda: Time(seconds=1_780_000_000., clock_type=ClockType.SYSTEM_TIME))
    n.start_time = Time(seconds=1_779_999_990., clock_type=ClockType.SYSTEM_TIME)
    n._clock = _Clock(10.)
    n.get_clock = lambda: n._clock
    logger = _Logger()
    n.get_logger = lambda: logger
    n._belief_xy = None
    n._belief_ready = False
    n._belief_wait_logged = False
    n._belief_sigma_m = math.inf
    n.initial_belief_max_sigma_m = .2
    n.wait_for_belief = True
    n.sent_count = 0
    n.repeat_count = 0
    n.delay = 3.
    n.frame_id = 'map_bev'
    n.goal_pub = Publisher()
    n.waypoints = [(0., 0.), (1., 0.)]
    n.wp_idx = 0
    n.arrival_radius = .1
    old = state_msg(0., 0., seconds=1., var=.01)
    old.header.frame_id = 'unrelated_frame'
    n._belief_cb(old)
    n._clock.seconds = .1  # In-place reset / no new valid belief.
    invalid = state_msg(0., 0., seconds=.1, var=math.nan)
    n._belief_cb(invalid)
    assert n._belief_ready
    n._send_goal()
    msg, = n.goal_pub.messages
    assert n.wp_idx == 1
    # Verify real planner consumers ignore the goal stamp, checking only frame/xy.
    p = node_at(now=.1)
    p._goal_cb(msg)
    p._validate_plan_frames(p.goal_msg, p.state_msg)
    assert p._goal_xy_from_msg(p.goal_msg) == (1., 0.)
    return dict(goal_header=UnicyclePlannerNode._stamp_to_float(msg.header.stamp),
                ros_now=.1, readiness_survives_invalid_input=True,
                waypoint_advanced_using_unvalidated_belief=True,
                planner_compares_goal_stamp=False)


def mission_system_clock_jumps():
    n = object.__new__(GoalMissionNode)
    wall = {'seconds': 99.}
    n.wall_clock = SimpleNamespace(now=lambda: Time(seconds=wall['seconds'], clock_type=ClockType.SYSTEM_TIME))
    n.start_time = Time(seconds=100., clock_type=ClockType.SYSTEM_TIME)
    n.delay = 3.
    n.sent_count = 0
    n.repeat_count = 0
    n.wait_for_belief = False
    n._belief_xy = None
    n.waypoints = [(1., 2.)]
    n.wp_idx = 0
    n.frame_id = 'map_bev'
    n.goal_pub = Publisher()
    n.get_logger = lambda: _Logger()
    counts = []
    for t in [99., 103., 90., 104.]:
        wall['seconds'] = t
        n._send_goal()
        counts.append(n.sent_count)
    assert counts == [0, 1, 1, 2]
    return dict(system_now=[99., 103., 90., 104.], sent_counts=counts,
                startup_delay_uses_system_elapsed=True,
                backward_system_jump_postpones_repeats=True)


class ContentionLock:
    def __init__(self):
        self.lock = threading.RLock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


def correction_worker_starvation():
    n = node_at(anchor=9.9, now=10.3)
    n._correction_lock = ContentionLock()
    entered, release = threading.Event(), threading.Event()

    def replay(m, P, start, end, u, dt):
        if n._stamp_to_float(end) == 10.:
            entered.set()
            wait(release)
        return m, P, {}

    n._replay_cmd_log_interval = replay
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(n._apply_metric_correction, stamp(10.), np.zeros(2), np.eye(2))
        try:
            wait(entered)
            b = pool.submit(n._apply_metric_correction, stamp(10.2), np.zeros(2), np.eye(2))
            wait(n._correction_lock.contended)
            io = pool.submit(n._odom_cb, odom(10.3))
            assert not io.done()
            assert n._data_lock.acquire(blocking=False)
            n._data_lock.release()
        finally:
            release.set()
        a.result(5)
        b.result(5)
        io.result(5)
    assert n._odom_log[-1][0] == 10.3
    return dict(workers=2, one_calculating=True, one_waiting_on_correction_lock=True,
                data_lock_free=True, queued_odom_cannot_execute_until_release=True)


def simultaneous_duplicate_envelopes():
    n = node_at(anchor=9.9)
    n._correction_lock = ContentionLock()
    entered, release = threading.Event(), threading.Event()
    errors = []
    n._fatal_experiment_stop = lambda reason, exc: errors.append(str(exc))

    def replay(m, P, start, end, u, dt):
        entered.set()
        wait(release)
        return m, P, {}

    n._replay_cmd_log_interval = replay
    a = run_thread(lambda: n._state_correction_envelope_cb(envelope()))
    try:
        wait(entered)
        b = run_thread(lambda: n._state_correction_envelope_cb(envelope()))
        wait(n._correction_lock.contended)
    finally:
        release.set()
    join(a)
    join(b)
    assert len(n.correction_assimilation_pub.published) == 1
    assert len(errors) == 1 and 'duplicate source_batch_id' in errors[0]
    return dict(assimilation_count=1, duplicate_fails_closed=True)


def pixel_timer_reads_new_measurement_with_old_stamp():
    n = node_at(anchor=9.8)
    n.use_pixel_correction = True
    n.pixel_correction_min_interval_s = .05
    n.pixel_stamp = stamp(9.9)
    n.pixel_meas = np.array([.01, 0.])
    enter_update = n._apply_pixel_correction
    snapshots = []
    snapshot = n._snapshot_pixel_correction_inputs

    def capture(s):
        result = snapshot(s)
        snapshots.append(result)
        return result

    def before_update(s, **kwargs):
        newer = PoseStamped()
        newer.header.stamp = stamp(9.95)
        newer.pose.position.x = .2
        n._pixel_cb(newer)
        return enter_update(s, **kwargs)

    n._snapshot_pixel_correction_inputs = capture
    n._apply_pixel_correction = before_update
    n._pixel_correction_timer_cb()
    assert math.isclose(n._stamp_to_float(snapshots[0].meas_stamp), 9.9)
    assert snapshots[0].meas[0] == .2
    return dict(timer_target=9.9, actual_pixel_capture=9.95,
                applied_measurement_x=float(snapshots[0].meas[0]))


def clock_domain_error_fails_open():
    n = node_at()
    n.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=1000., clock_type=ClockType.SYSTEM_TIME))
    age = n._stamp_age_s(stamp(1.))
    assert age == 0. and n._metric_correction_is_fresh(age)
    return dict(clock_domain_mismatch_reported_age=age, accepted_as_fresh=True,
                active_ros_clock_path=False)


PROBES = [
    publication_overtakes, same_stamp_supersession_guard, full_covariance_snapshot,
    frame_from_different_event, equal_time_revision_lost_by_consumer,
    odometry_callback_finishes_out_of_order, coverage_check_and_replay_use_different_buffers,
    planner_metadata_from_superseding_update, planning_bootstrap_bypasses_envelope,
    backward_reset_keeps_epoch_state, camera_batch_reset, mission_clock_and_readiness,
    mission_system_clock_jumps,
    correction_worker_starvation, simultaneous_duplicate_envelopes,
    pixel_timer_reads_new_measurement_with_old_stamp, clock_domain_error_fails_open,
]


if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    expected = json.loads((directory / 'source_manifest.json').read_text())
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in expected}
    records = []
    for probe in PROBES:
        result = probe()
        records.append(dict(probe=probe.__name__, reproduction_passed=True, observed=result))
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    report = dict(meaning='Passing means the documented baseline behavior reproduced, including defects.',
                  source_sha256=hashes,
                  changed_since_audit_start=[f for f in expected if hashes[f] != expected[f]],
                  executed_source_paths={cls.__name__: str(Path(inspect.getfile(cls)).resolve())
                                         for cls in (UnicyclePlannerNode, CameraManagerNode,
                                                     GoalMissionNode, FourCameraBatcher)},
                  probes=records)
    (directory / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
