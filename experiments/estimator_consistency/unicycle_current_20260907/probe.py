"""Bounded current-source verification of package A and recursive update semantics.

No ROS graph, simulator, or experiment is started. Real callbacks, ROS messages and
planner prediction are used with a deterministic clock and captured publishers.
Assertions distinguish fixed invariants from reproductions of open defects. The
older audit probes remain unchanged. Run after sourcing install/setup.bash.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FILES = [
    'src/planning/planning/core/dynamics.py',
    'src/planning/planning/core/belief_correction.py',
    'src/planning/planning/core/motion_history.py',
    'src/planning/planning/planners/base_planner.py',
    'src/planning/planning/nodes/unicycle_planner_node.py',
    'src/planning/planning/nodes/efe_agent_node.py',
    'src/sim/sim/encoder_noise_node.py',
]


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


BEFORE = hashes()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load('audit01_original', 'experiments/estimator_consistency/module_audit_01_20260906/probe.py')
previous = base.previous
from test_planner_node_correction_wiring import stamp
from test_planner_node_state_correction import state_msg
from test_planner_node_per_camera_correction import observation
from geometry_msgs.msg import Twist
from reliability.fusion import map_observations_to_json
from planning.core import belief_correction as bc
from planning.core.motion_history import covers_interval
from planning.nodes.unicycle_planner_node import UnicyclePlannerNode


def node(**kwargs):
    n = base.node(**kwargs)
    n._seen_map_observation_stamps = {}
    n._odom_accepted_stamp_ns = None
    return n


def envelope(batch, t, xy=(0., 0.), R=None):
    return SimpleNamespace(data=json.dumps(dict(
        schema_version=1, frame_id='map_bev', source_batch_id=batch,
        correction_stamp=t, xy=list(xy),
        covariance_m2=(np.eye(2)*.03 if R is None else R).tolist())))


def anchor(n):
    return dict(m=n.belief_m.copy(), P=n.belief_S.copy(),
                t=n._stamp_to_float(n.belief_stamp))


def same_anchor(n, before):
    np.testing.assert_array_equal(n.belief_m, before['m'])
    np.testing.assert_array_equal(n.belief_S, before['P'])
    assert n._stamp_to_float(n.belief_stamp) == before['t']


def fixed_outage_snapshot():
    n = node(belief_stamp_s=0., now_s=60.)
    n._odom_log = [(k/10., 0., 1. if k < 2 else 0.) for k in range(600)]
    original = n._predict_belief_to_now
    passed = []
    def trim_after_check(*args, **kwargs):
        motion = kwargs['motion_snapshot']
        passed.append(motion)
        n._odom_cb(base.odom(60.2))
        assert n._odom_log[0][0] == .3
        assert motion.odom[0][0] == 0.
        return original(*args, **kwargs)
    n._predict_belief_to_now = trim_after_check
    n._state_correction_envelope_cb(envelope('outage-fixed', 59.9))
    np.testing.assert_allclose(n.belief_m[2], .2, atol=1e-12)
    np.testing.assert_allclose(n.belief_S[2, 2], .05+.0004*59.9, atol=1e-12)
    assert n._stamp_to_float(n.belief_stamp) == 59.9
    terminal = json.loads(n.correction_assimilation_pub.published[-1])
    assert terminal['status'] == 'dropped' and terminal['reason'] == 'replay_gap_too_large'
    return dict(verdict='fixed', checked_entries=len(passed[0].odom),
                expected_yaw=.2, observed_yaw=n.belief_m[2],
                live_history_starts_at=n._odom_log[0][0], posterior=anchor(n), terminal=terminal)


def fixed_odom_order():
    n = node(belief_stamp_s=9.4, now_s=10.1)
    for t in [9.4, 9.7, 9.5]:
        n._odom_cb(base.odom(t, .2, yaw=t))
    n._odom_cb(base.odom(9.7, -.2, w=1., yaw=-2.))
    assert [x[0] for x in n._odom_log] == [9.4, 9.700000000000001]
    assert n._odom_refused_old == n._odom_refused_duplicate == 1
    np.testing.assert_allclose(n.odom_vel, [.2, 0.])
    out = n._apply_metric_correction(stamp(10.), np.array([.12, 0.]), np.eye(2)*.03)
    np.testing.assert_allclose(out.m_pred[0], .12, atol=1e-12)
    np.testing.assert_allclose(out.S_pred[2,2]-.05, .0004*.6, atol=1e-12)
    return dict(verdict='fixed', motion=n._odom_log, expected_prior_x=.12,
                prior_x=out.m_pred[0], applied_yaw_Q=out.S_pred[2,2]-.05,
                refused_old=n._odom_refused_old, refused_duplicate=n._odom_refused_duplicate)


def fixed_bootstrap_identity():
    results = []
    for order in [('pose', 'envelope'), ('envelope', 'pose')]:
        n = node(now_s=10.)
        n.belief_m = n.belief_S = n.belief_stamp = None
        pose = state_msg(1., 2., seconds=9.95)
        pose.header.frame_id = 'map_bev'
        for event in order:
            if event == 'pose':
                n._state_cb(pose)
            else:
                n._state_correction_envelope_cb(envelope('identified-bootstrap', 9.95, (1., 2.)))
            n._resolve_state_belief_ekf(stamp(10.))
            if event == 'pose' and len(n.correction_assimilation_pub.published) == 0:
                assert n.belief_m is None
        events = [json.loads(s) for s in n.correction_assimilation_pub.published]
        assert len(events) == 1 and events[0]['status'] == 'accepted_bootstrap'
        results.append(dict(order=order, anchor=anchor(n), terminal=events[0]))
    np.testing.assert_array_equal(results[0]['anchor']['m'], results[1]['anchor']['m'])
    np.testing.assert_array_equal(results[0]['anchor']['P'], results[1]['anchor']['P'])
    return dict(verdict='fixed', orders=results)


def fixed_encoder_order_and_open_gap():
    n = base.encoder()
    for t in [0., .2, .1, .3]:
        n._odom_cb(base.odom(t, w=1.))
    np.testing.assert_allclose(n._pose_theta, .3, atol=1e-12)
    gap = base.encoder()
    for t in [0., .1, 1.1, 1.2]:
        gap._odom_cb(base.odom(t, w=1.))
    np.testing.assert_allclose(gap._pose_theta, .2, atol=1e-12)
    return dict(order_verdict='fixed', input_order=[0., .2, .1, .3],
                expected_order_yaw=.3, observed_order_yaw=n._pose_theta,
                gap_verdict='open; existing baseline rebase policy retained',
                gap_input=[0., .1, 1.1, 1.2], known_script_yaw=1.2,
                published_yaw=gap._pose_theta, published_stamp=1.2,
                published_yaw_variance=gap.messages[-1].pose.covariance[35])


class ObservedRLock:
    """Records actual competing acquisition; no sleep-based ordering assumption."""
    def __init__(self):
        self.lock = threading.RLock()
        self.contended = threading.Event()
    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.contended.set()
            self.lock.acquire()
        return self
    def __exit__(self, *_):
        self.lock.release()


def fixed_serialized_corrections():
    n = node(belief_stamp_s=9.8, now_s=10.3)
    n._odom_cb(base.odom(9.8, .2))
    n._correction_lock = ObservedRLock()
    entered, release = threading.Event(), threading.Event()
    replay = n._replay_cmd_log_interval
    read_priors, errors = [], []
    def record(m, P, t0, t1, *args):
        read_priors.append(dict(t=n._stamp_to_float(t0), m=m.copy(), P=P.copy()))
        if len(read_priors) == 1:
            entered.set()
            assert release.wait(5.)
        return replay(m, P, t0, t1, *args)
    n._replay_cmd_log_interval = record
    def apply(batch, t, x):
        try:
            n._state_correction_envelope_cb(envelope(batch, t, (x, 0.)))
        except BaseException as exc:
            errors.append(repr(exc))
    first = threading.Thread(target=apply, args=('serial-A', 10., .06))
    second = threading.Thread(target=apply, args=('serial-B', 10.2, .1))
    first.start()
    try:
        assert entered.wait(5.)
        second.start()
        assert n._correction_lock.contended.wait(5.)
        assert len(read_priors) == 1
    finally:
        release.set()
    first.join(5.); second.join(5.)
    assert not first.is_alive() and not second.is_alive() and not errors
    m, P = np.zeros(3), np.eye(3)*.05
    H = np.eye(3)[:2]
    for x in [.06, .1]:
        m, P, _, _ = base.oracle_step(m, P, [.2, 0.], .2)
        K = np.linalg.solve(H@P@H.T + np.eye(2)*.03, H@P).T
        m = m + K@(np.array([x, 0.])-H@m)
        J = np.eye(3)-K@H
        P = J@P@J.T+K@(np.eye(2)*.03)@K.T
    np.testing.assert_allclose(n.belief_m, m, atol=1e-12)
    np.testing.assert_allclose(n.belief_S, P, atol=1e-12)
    assert [p['t'] for p in read_priors] == [9.8, 10.]
    return dict(verdict='fixed', witnessed_lock_contention=True, read_priors=read_priors,
                oracle_posterior_m=m, oracle_posterior_P=P, posterior=anchor(n),
                terminal_events=[json.loads(s) for s in n.correction_assimilation_pub.published])


def fixed_same_time_camera_and_duplicates():
    n = node(belief_stamp_s=9.9, now_s=10.)
    n.state_correction_mode = 'per_camera'
    n._odom_cb(base.odom(9.9))
    observations = [observation('B', .02, 0., seconds=9.95, var=.03),
                    observation('A', .01, 0., seconds=9.95, var=.03)]
    msg = SimpleNamespace(data=map_observations_to_json(observations, frame_id='map_bev'))
    n._map_observations_cb(msg)
    expected_p = 1/(1/(.05+.0001*.05)+2/.03)
    np.testing.assert_allclose(n.belief_S[0,0], expected_p, atol=1e-12)
    assert [d[30:32] for d in n.pixel_correction_diag_pub.published] == [[1.,0.], [1.,0.]]
    before = anchor(n)
    n._map_observations_cb(msg)
    same_anchor(n, before)
    return dict(verdict='fixed', two_distinct_cameras_accepted=True,
                expected_x_variance=expected_p, posterior=before,
                duplicate_changed_anchor=False,
                diagnostics=[d[30:32] for d in n.pixel_correction_diag_pub.published],
                terminal_event_count=len(n.correction_assimilation_pub.published))


def drop_reject_and_duplicate_event_order():
    n = node(belief_stamp_s=9.8, now_s=10.2)
    n._odom_cb(base.odom(9.8, .2))
    n._state_correction_envelope_cb(envelope('accept', 10., (.04,0.)))
    accepted = anchor(n)
    n._state_correction_envelope_cb(envelope('late', 9.9))
    same_anchor(n, accepted)
    n._state_correction_envelope_cb(envelope('stale', 9.0))
    same_anchor(n, accepted)
    # A fresh NIS refusal still commits all supported motion and one stated inflation.
    prior_m, prior_P, _, _ = base.oracle_step(accepted['m'], accepted['P'], [.2,0.], .1)
    n._state_correction_envelope_cb(envelope('reject', 10.1, (100.,100.)))
    expected_P = prior_P + np.diag([.05,.05,0.])
    np.testing.assert_allclose(n.belief_m, prior_m, atol=1e-12)
    np.testing.assert_allclose(n.belief_S, expected_P, atol=1e-12)
    assert n._stamp_to_float(n.belief_stamp) == 10.1
    refused = anchor(n)
    errors = []
    n._fatal_experiment_stop = lambda reason, exc: errors.append(str(exc))
    n._state_correction_envelope_cb(envelope('reject', 10.1, (100.,100.)))
    same_anchor(n, refused)
    events = [json.loads(s) for s in n.correction_assimilation_pub.published]
    assert [e['status'] for e in events] == ['accepted', 'dropped', 'dropped', 'rejected']
    assert len(errors) == 1 and 'duplicate source_batch_id' in errors[0]
    return dict(verdict='verified', terminal_events=events, held_prediction=refused,
                expected_reject_covariance=expected_P, duplicate_fatal_error=errors,
                duplicate_mutation=False, duplicate_has_new_terminal_event=False)


def invalid_prediction_can_be_committed_after_refusal():
    n = node(belief_stamp_s=9.9, now_s=10.)
    before = anchor(n)
    msg = Twist()
    msg.linear.x = math.nan
    n._clock.seconds = 9.9
    n._cmd_cb(msg)  # Actual ingestion; no odometry is available in this fixture.
    n._clock.seconds = 10.
    with np.errstate(invalid='ignore'):
        n._state_correction_envelope_cb(envelope('invalid-motion', 10.))
    event = json.loads(n.correction_assimilation_pub.published[-1])
    assert event['status'] == 'rejected' and event['reason'] == 'update_failed'
    assert not np.isfinite(n.belief_m).all() and not np.isfinite(n.belief_S).all()
    assert n._stamp_to_float(n.belief_stamp) == 10.
    return dict(verdict='confirmed open defect; nonfinite command fault injection',
                before=before, terminal_event=event,
                finite_mean_after=bool(np.isfinite(n.belief_m).all()),
                finite_covariance_after=bool(np.isfinite(n.belief_S).all()),
                stamp_after=n._stamp_to_float(n.belief_stamp))


def current_capture_heading():
    n = node(now_s=10., odom_yaw=None)
    n.belief_m = n.belief_S = n.belief_stamp = None
    n.odom_yaw_offset_rad = .3
    for t, yaw in [(9.,0.), (9.8,.8), (9.9,.9), (10.,1.)]:
        n._odom_cb(base.odom(t, w=1., yaw=yaw))
    n._state_correction_envelope_cb(envelope('moving-bootstrap', 9.8))
    capture = anchor(n)
    m, P = n._predict_belief_to_now(capture['m'].copy(), capture['P'].copy(),
                                   np.zeros(2), .2, stamp(10.))
    np.testing.assert_allclose(capture['m'][2], 1.3, atol=1e-12)
    np.testing.assert_allclose(m[2], 1.5, atol=1e-12)
    return dict(verdict='confirmed open defect', capture_stamp=9.8, apply_stamp=10.,
                offset=.3, expected_capture_yaw=1.1, actual_capture_yaw=capture['m'][2],
                expected_now_yaw=1.3, predicted_now_yaw=m[2],
                capture_covariance=capture['P'], predicted_covariance=P)


def turning_partition_limit():
    results=[]
    for count in [1, 10, 100]:
        p=previous.planner()
        m, P=np.zeros(3), np.zeros((3,3))
        for _ in range(count):
            m, P=p.predict(m, P, np.array([.2,1.]), dt=1./count)
        results.append(dict(steps=count, m=m, P=P))
    exact=np.array([.2*math.sin(1.), .2*(1-math.cos(1.)), 1.])
    assert np.linalg.norm(results[0]['m']-exact)>np.linalg.norm(results[-1]['m']-exact)
    return dict(verdict='declared Euler/frozen-heading approximation, not a Q retuning result',
                exact_continuous_mean=exact, partitions=results)


def encoder_coherent_state_persists():
    n=base.encoder()
    for k in range(51):
        n._odom_cb(base.odom(k/10., v=.22))
    np.testing.assert_allclose(n._linear_scale_jacobian, [1.1,0.,0.], atol=1e-12)
    coherent=np.asarray(n._published_pose_covariance())-np.asarray(n._pose_cov)
    expected=(.02*1.1)**2
    np.testing.assert_allclose(coherent[0,0], expected, atol=1e-12)
    return dict(verdict='verified persistent encoder scale state', intervals=50,
                scale_jacobian=n._linear_scale_jacobian,
                expected_coherent_x_variance=expected,
                coherent_covariance=coherent)


def prediction_yaw_diagnostic_uses_wrong_endpoint():
    n=node(belief_stamp_s=9.4, now_s=10.)
    for t,v,w in [(9.35,.2,0.), (9.6,0.,.8), (9.75,.15,0.), (10.,0.,0.)]:
        n._odom_cb(base.odom(t,v,w))
    m,P=n._predict_belief_to_now(n.belief_m.copy(),n.belief_S.copy(),
                                 np.zeros(2),.6,stamp(10.))
    np.testing.assert_allclose(m[2], .12, atol=1e-12)
    np.testing.assert_allclose(n._latest_odom_delta_theta, .16, atol=1e-12)
    return dict(verdict='confirmed open diagnostic defect',
                expected_integrated_delta=.12, predicted_delta=m[2],
                reported_odom_delta=n._latest_odom_delta_theta)


def optional_heading_override_diagnostic():
    n=node(belief_stamp_s=9.9, now_s=10.)
    n.heading_update_mode='camera_xy_only'
    n.belief_m[2]=.4
    n.belief_S[0,2]=n.belief_S[2,0]=.01
    n._odom_cb(base.odom(9.9,yaw=1.))
    out=n._apply_metric_correction(stamp(10.), np.array([.03,0.]),np.eye(2)*.03)
    assert out.accepted
    np.testing.assert_allclose(out.next_m[2],1.,atol=1e-12)
    np.testing.assert_allclose(out.next_S[:2,2],0.,atol=1e-12)
    assert np.linalg.eigvalsh(out.next_S).min()>0
    actual_delta=out.next_m[2]-out.m_pred[2]
    assert not np.isclose(out.yaw_info['theta_update_total_rad'],actual_delta)
    return dict(verdict='optional covariance transformation verified; yaw diagnostic open',
                actual_heading_delta=actual_delta,
                reported_heading_delta=out.yaw_info['theta_update_total_rad'],
                committed=anchor(n))


def callback_write_inventory():
    targets={'belief_m','belief_S','belief_stamp','state_msg','_latest_odom_yaw',
             '_last_correction_stamp','_odom_origin_stamp_s','_odom_accepted_stamp_ns',
             'last_cmd','odom_vel'}
    result={}
    for filename in FILES:
        if not filename.endswith(('unicycle_planner_node.py','efe_agent_node.py')):
            continue
        source=(ROOT/filename).read_text()
        tree=ast.parse(source)
        writes=[]
        callbacks=[]
        for cls in [x for x in tree.body if isinstance(x,ast.ClassDef)]:
            for fn in [x for x in cls.body if isinstance(x,ast.FunctionDef)]:
                for expr in ast.walk(fn):
                    if isinstance(expr,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                        ts=expr.targets if isinstance(expr,ast.Assign) else [expr.target]
                        for t in ts:
                            for a in ast.walk(t):
                                if isinstance(a,ast.Attribute) and isinstance(a.value,ast.Name) and a.value.id=='self' and a.attr in targets:
                                    writes.append(dict(function=fn.name, line=expr.lineno,
                                                       field=a.attr, statement=ast.get_source_segment(source,expr)))
                    if isinstance(expr,ast.Call) and isinstance(expr.func,ast.Attribute) and expr.func.attr in ('create_subscription','create_timer'):
                        callbacks.append(dict(line=expr.lineno, owner=fn.name,
                                              registration=ast.get_source_segment(source,expr)))
        result[filename]=dict(writes=writes, callbacks=callbacks)
    return result


if __name__=='__main__':
    cases=[fixed_outage_snapshot, fixed_odom_order, fixed_bootstrap_identity,
           fixed_encoder_order_and_open_gap, fixed_serialized_corrections,
           fixed_same_time_camera_and_duplicates, drop_reject_and_duplicate_event_order,
           base.complete_trace, base.frame_and_white_noise_checks,
           current_capture_heading, previous.stale_motion,
           base.missing_prefix_and_empty_history, previous.unsupported_heading,
           invalid_prediction_can_be_committed_after_refusal,
           base.quorum_information_reuse, base.per_camera_nonrejection_inflation,
           base.almost_simultaneous_camera, previous.stale_quorum,
           previous.duplicate_pixel, base.pixel_refusal_freezes_anchor,
           previous.coherent_wrapper, previous.encoder_covariance_at_stop,
           previous.q_diagnostic, base.alias_hazard, turning_partition_limit,
           encoder_coherent_state_persists, prediction_yaw_diagnostic_uses_wrong_endpoint,
           optional_heading_override_diagnostic]
    results={}
    for case in cases:
        try:
            results[case.__name__]=case()
        except Exception as exc:
            results[case.__name__]=dict(probe_error=repr(exc))
    result=dict(source_before=BEFORE, source_after=hashes(),
                import_provenance=base.provenance(), cases=results,
                callback_write_inventory=callback_write_inventory(),
                probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['source_stable_during_probe']=result['source_before']==result['source_after']
    print(json.dumps(base.sanitize(result),indent=2,sort_keys=True,allow_nan=False))
    if not result['source_stable_during_probe'] or any('probe_error' in item for item in results.values()):
        sys.exit(1)
