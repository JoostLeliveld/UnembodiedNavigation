"""Deterministic report-01 probes; imports deployed callbacks, creates no ROS graph.

Successful execution confirms reproduction, not correctness. The cases deliberately
assert current defective behavior alongside independent desired invariants. Run after
`source install/setup.bash`; stdout is strict JSON and nothing in a drive is changed.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/planning'))
previous_path = ROOT / 'experiments/estimator_consistency/module_review_20260906/probe.py'
spec = importlib.util.spec_from_file_location('previous_estimator_probe', previous_path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
from test_planner_node_correction_wiring import stamp
from test_planner_node_per_camera_correction import observation
from planning.core.motion_history import covers_interval
from nav_msgs.msg import Odometry
from sim.encoder_noise_node import EncoderNoiseNode


def node(**kwargs):
    n = previous.node(**kwargs)
    # Match the resolved active launch, including defaults absent from the YAML.
    n.pixel_timeout_s = .5
    n.max_predict_speed_mps = .22
    n.use_odom_for_predict = True
    n._CMD_LOG_MAX_S = 60.
    n.odom_vel = np.zeros(2)
    n.require_state_correction_envelope = True
    n._seen_state_source_batch_ids = set()
    return n


def odom(t, v=0., w=0., yaw=0.):
    msg = Odometry()
    msg.header.stamp = stamp(t)
    msg.header.frame_id = 'odom'
    msg.child_frame_id = 'base_footprint'
    msg.twist.twist.linear.x = float(v)
    msg.twist.twist.angular.z = float(w)
    msg.pose.pose.orientation.z = math.sin(yaw / 2.)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.)
    return msg


def oracle_step(m, P, u, dt):
    """Continuous frozen-heading linear SDE via a block exponential, not runtime Q.

    Chosen trace controls are straight translation or stationary rotation, so the
    mean also has an exact closed-form solution equal to the declared Euler map.
    """
    theta, v, w = float(m[2]), float(u[0]), float(u[1])
    A = np.zeros((3, 3))
    A[0, 2] = -v * math.sin(theta)
    A[1, 2] = v * math.cos(theta)
    L = np.array([[math.cos(theta), 0.], [math.sin(theta), 0.], [0., 1.]])
    D = L @ np.diag([.01**2, .02**2]) @ L.T
    E = expm(np.block([[A, D], [np.zeros((3, 3)), -A.T]]) * dt)
    F = E[:3, :3]
    Q = E[:3, 3:] @ F.T
    mean = np.asarray(m).copy()
    mean[:2] += v * dt * np.array([math.cos(theta), math.sin(theta)])
    mean[2] = math.atan2(math.sin(theta + w * dt), math.cos(theta + w * dt))
    return mean, F @ P @ F.T + Q, F, Q


def complete_trace():
    n = node(belief_stamp_s=9.4, now_s=10.2)
    initial_m = np.array([1., -2., .4])
    initial_P = np.array([[.04, .005, .002], [.005, .06, -.003], [.002, -.003, .025]])
    n.belief_m, n.belief_S = initial_m.copy(), initial_P.copy()
    for t, v, w in [(9.35, .2, 0.), (9.6, 0., .8), (9.75, .15, 0.), (10., 0., 0.)]:
        n._odom_cb(odom(t, v, w))
    m, P = initial_m.copy(), initial_P.copy()
    steps = []
    for t0, t1, u in [(9.4, 9.6, [.2, 0.]), (9.6, 9.75, [0., .8]), (9.75, 10., [.15, 0.])]:
        m, P, F, Q = oracle_step(m, P, u, t1-t0)
        steps.append(dict(start=t0, end=t1, u=u, F=F, Q=Q, m=m.copy(), P=P.copy()))
    H = np.array([[1., 0., 0.], [0., 1., 0.]])
    z = m[:2] + [.02, -.01]
    R = np.array([[.015, .003], [.003, .025]])
    innovation = z - H @ m
    Sy = H @ P @ H.T + R
    K = np.linalg.solve(Sy, H @ P).T
    expected_m = m + K @ innovation
    IKH = np.eye(3) - K @ H
    expected_P = IKH @ P @ IKH.T + K @ R @ K.T
    captured = []
    commit = n._commit_metric_correction_outcome
    def capture(*args, **kwargs):
        commit(*args, **kwargs)
        captured.append(args[2])
    n._commit_metric_correction_outcome = capture
    payload = dict(schema_version=1, frame_id='map_bev', source_batch_id='audit:camera_A+camera_B@10000000000',
                   correction_stamp=10., xy=z.tolist(), covariance_m2=R.tolist())
    n._state_correction_envelope_cb(SimpleNamespace(data=json.dumps(payload)))
    out = captured[0]
    assert out.accepted
    for actual, expected in [(out.m_pred, m), (out.S_pred, P), (out.K, K),
                             (n.belief_m, expected_m), (n.belief_S, expected_P)]:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    diag = n.pixel_correction_diag_pub.published[0]
    np.testing.assert_allclose(diag[14:17], m, atol=1e-12)
    np.testing.assert_allclose(diag[17:20], expected_m, atol=1e-12)
    assert diag[30] == 1. and diag[31] == 0.
    terminal = json.loads(n.correction_assimilation_pub.published[0])
    assert terminal['status'] == 'accepted' and terminal['belief_stamp_after'] == 10.
    committed = n.belief_m.copy(), n.belief_S.copy()
    n.planner_belief_pub = SimpleNamespace(publish=lambda msg: None)
    n._belief_publish_tick()
    n._resolve_state_belief_ekf(stamp(10.2))
    np.testing.assert_array_equal(n.belief_m, committed[0])
    np.testing.assert_array_equal(n.belief_S, committed[1])
    assert n._stamp_to_float(n.belief_stamp) == 10.
    return dict(initial_stamp=9.4, initial_m=initial_m, initial_P=initial_P, odom_log=n._odom_log,
                steps=steps, z=z, R=R, H=H, innovation=innovation, innovation_covariance=Sy,
                K=K, posterior_m=expected_m, posterior_P=expected_P,
                full_covariance_max_abs_error=float(np.max(np.abs(n.belief_S-expected_P))),
                applied_nis=out.nis, diagnostics=diag, terminal_event=terminal,
                reads_did_not_commit=True)


def out_of_order_motion():
    n = node(belief_stamp_s=9.4, now_s=10.1)
    for t in [9.4, 9.7, 9.5]:
        n._odom_cb(odom(t, .2))
    durations = []
    predict = n.planner.predict
    def recording_predict(m, P, u, dt=None):
        durations.append(dt)
        return predict(m, P, u, dt=dt)
    n.planner.predict = recording_predict
    out = n._apply_metric_correction(stamp(10.), np.array([.12, 0.]), np.eye(2)*.03)
    assert not covers_interval(n._odom_log, 9.4, 10., 1.5)
    assert out.accepted
    np.testing.assert_allclose(out.m_pred[0], .16, atol=1e-12)
    np.testing.assert_allclose(sum(durations), .8, atol=1e-12)
    assert out.replay_meta['cmd_replay_used_fallback'] == 0.
    return dict(arrival_stamps=[9.4,9.7,9.5], interval_duration=.6,
                integrated_durations=durations, expected_x=.2*.6, observed_x=out.m_pred[0],
                expected_Q_theta=.02**2*.6, observed_Q_theta=out.S_pred[2,2]-.05,
                replay=out.replay_meta, resulting_stamp=n._stamp_to_float(n.belief_stamp))


def missing_prefix_and_empty_history():
    partial = node(belief_stamp_s=9.4, now_s=10.1)
    partial._odom_cb(odom(9.7, .2))
    partial._cmd_log = [(9.3, .2, 0.)]
    out = partial._apply_metric_correction(stamp(10.), np.array([.12, 0.]), np.eye(2)*.03)
    np.testing.assert_allclose(out.m_pred[0], .06, atol=1e-12)
    assert not covers_interval(partial._odom_log, 9.4, 10., 1.5)
    assert out.replay_meta['cmd_replay_used_fallback'] == 0.
    empty = node(belief_stamp_s=9.9, now_s=10.)
    empty.last_cmd = np.array([.2, 0.])
    view, _ = empty._predict_belief_to_now(empty.belief_m.copy(), empty.belief_S.copy(), empty.last_cmd, .1, stamp(10.))
    actual = empty._apply_metric_correction(stamp(10.), np.array([.02, 0.]), np.eye(2)*.03)
    assert view[0] == 0.
    np.testing.assert_allclose(actual.m_pred[0], .02, atol=1e-12)
    return dict(partial_history_prior_x=out.m_pred[0], known_script_expected_x=.12,
                partial_replay=out.replay_meta, empty_history_publication_x=view[0],
                empty_history_correction_prior_x=actual.m_pred[0])


def outage_support_snapshot_race():
    n = node(belief_stamp_s=0., now_s=60.)
    n._odom_log = [(k/10., 0., 1. if k < 2 else 0.) for k in range(600)]
    assert covers_interval(n._odom_log, 0., 59.9, 1.5)
    predict = n._predict_belief_to_now
    def input_arrives_after_coverage_check(*args):
        # This callback can run after the coverage lock is released and before
        # prediction copies the buffer again. It trims the previously verified turn.
        n._clock.seconds = 60.2
        n._odom_cb(odom(60.2))
        return predict(*args)
    n._predict_belief_to_now = input_arrives_after_coverage_check
    n._apply_metric_correction(stamp(59.9), np.zeros(2), np.eye(2)*.03,
                               source_batch_id='audit:outage-snapshot')
    np.testing.assert_allclose(n.belief_m[2], 0., atol=1e-12)
    assert n._stamp_to_float(n.belief_stamp) == 59.9
    return dict(checked_full_history=True, expected_heading=.2, observed_heading=n.belief_m[2],
                first_remaining_motion_stamp=n._odom_log[0][0],
                committed_stamp=n._stamp_to_float(n.belief_stamp),
                terminal_event=json.loads(n.correction_assimilation_pub.published[0]))


def frame_and_white_noise_checks():
    from planning.core.dynamics import unicycle_jacobian, unicycle_step
    m = np.array([1., -2., 1.2])
    u, dt, eps = np.array([.2, -.7]), .13, 1e-6
    columns = []
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        columns.append((unicycle_step(m+d,u,dt)-unicycle_step(m-d,u,dt))/(2*eps))
    finite_difference = np.column_stack(columns)
    F = unicycle_jacobian(m,u,dt)
    np.testing.assert_allclose(F,finite_difference,atol=2e-10,rtol=1e-8)
    p = previous.planner()
    one_m, one_P = p.predict(np.zeros(3),np.zeros((3,3)),np.array([.22,0.]),dt=5.)
    many_m, many_P = np.zeros(3),np.zeros((3,3))
    for _ in range(50):
        many_m,many_P=p.predict(many_m,many_P,np.array([.22,0.]),dt=.1)
    np.testing.assert_allclose(one_m,many_m,atol=1e-12)
    np.testing.assert_allclose(one_P,many_P,atol=1e-12)
    return dict(jacobian_max_abs_error=float(np.max(np.abs(F-finite_difference))),
                straight_white_covariance_partition_max_abs_error=float(np.max(np.abs(one_P-many_P))),
                straight_white_lateral_variance=one_P[1,1])


def encoder():
    n = object.__new__(EncoderNoiseNode)
    for key, value in dict(enabled=True, linear_slip_mean=.02, linear_slip_std=.05,
                           angular_slip_mean=0., angular_slip_std=.03, linear_additive_std=.004,
                           angular_additive_std=.020, correlation_alpha=.80,
                           stop_linear_deadband=1e-4, stop_angular_deadband=1e-4, max_dt_s=.5,
                           initial_position_std_m=.01, initial_yaw_std_rad=.01,
                           linear_scale_bias_std=.02, covariance_floor_m2=1e-8,
                           covariance_floor_yaw_rad2=1e-8).items():
        setattr(n, key, value)
    # Deterministic zero innovations; all configured means and covariance logic retained.
    n._rng = SimpleNamespace(gauss=lambda mean, std: 0.)
    n._last_stamp = None
    n._linear_slip_state = n._angular_slip_state = 0.
    n._pose_cov = n._initial_pose_covariance()
    n._linear_scale_jacobian = [0., 0., 0.]
    n.messages = []
    n._pub = SimpleNamespace(publish=n.messages.append)
    return n


def encoder_order_and_gap():
    n = encoder()
    for t in [0., .2, .1, .3]:
        n._odom_cb(odom(t, w=1.))
    np.testing.assert_allclose(n._pose_theta, .4, atol=1e-12)
    # A skipped long interval is followed by a fresh stamp on the unadvanced pose.
    gap = encoder()
    for t in [0., .1, 1.1, 1.2]:
        gap._odom_cb(odom(t, w=1.))
    np.testing.assert_allclose(gap._pose_theta, .2, atol=1e-12)
    return dict(out_of_order=dict(input_stamps=[0., .2, .1, .3], expected_heading=.3,
                                  observed_heading=n._pose_theta, published_stamp=.3),
                long_gap=dict(input_stamps=[0., .1, 1.1, 1.2], expected_heading=1.2,
                              observed_heading=gap._pose_theta, published_stamp=1.2,
                              omitted_turn=1., published_yaw_variance=gap.messages[-1].pose.covariance[35]))


def quorum_information_reuse():
    n = node(belief_stamp_s=9.9, now_s=10.)
    n.state_correction_mode = 'per_camera'
    n._seen_map_observation_stamps = {}
    n.state_reanchor_m = 2.
    n._apply_map_observations([observation('camera_A', 3., 0., seconds=9.95, var=.03),
                               observation('camera_B', 3., 0., seconds=9.95, var=.03)])
    # Median initializes at .03, then both source readings enter AGAIN: .03/3.
    np.testing.assert_allclose(n.belief_S[:2,:2], np.eye(2)*.01, atol=1e-12)
    return dict(cameras=2, R_each=.03, actual_xy_variance=n.belief_S[0,0],
                independent_two_reading_bootstrap_variance=.03/2,
                equivalent_information_terms=.03/n.belief_S[0,0],
                diagnostics_count=len(n.pixel_correction_diag_pub.published),
                assimilation_count=len(n.correction_assimilation_pub.published))


def per_camera_nonrejection_inflation():
    boot = node(now_s=10.)
    boot.state_correction_mode = 'per_camera'
    boot._seen_map_observation_stamps = {}
    boot.belief_m = boot.belief_S = boot.belief_stamp = None
    boot._apply_map_observations([observation('A', 0., 0., seconds=9.95, var=.03)])
    np.testing.assert_allclose(boot.belief_S[0,0], .08, atol=1e-12)
    stale = node(belief_stamp_s=10., now_s=12.)
    stale.state_correction_mode = 'per_camera'
    stale._seen_map_observation_stamps = {}
    before = stale.belief_S.copy()
    stale._apply_map_observations([observation('A', 0., 0., seconds=9., var=.03)])
    np.testing.assert_allclose(stale.belief_S[0,0]-before[0,0], .05, atol=1e-12)
    gap = node(belief_stamp_s=0., now_s=10.1)
    gap.state_correction_mode = 'per_camera'
    gap._seen_map_observation_stamps = {}
    gap._odom_log = [(k/10.,0.,0.) for k in range(101)]
    gap._apply_map_observations([observation('A',0.,0.,seconds=10.,var=.03)])
    added_inflation = gap.belief_S[0,0] - .05 - .01**2*10.
    np.testing.assert_allclose(added_inflation,.10,atol=1e-12)
    return dict(single_camera_bootstrap_R=.03, actual_bootstrap_variance=boot.belief_S[0,0],
                stale_drop_variance_added=stale.belief_S[0,0]-before[0,0],
                stale_drop_stamp_after=stale._stamp_to_float(stale.belief_stamp),
                single_long_gap_batch_inflation_added=added_inflation,
                configured_once_per_batch_inflation=.05)


def almost_simultaneous_camera():
    n = node(belief_stamp_s=9.9, now_s=10.)
    n.state_correction_mode = 'per_camera'
    n._seen_map_observation_stamps = {}
    n._odom_cb(odom(9.9))
    n._apply_map_observations([observation('A',0.,0.,seconds=9.95,var=.03),
                               observation('B',0.,0.,seconds=9.9505,var=.03)])
    decisions = [d[30:32] for d in n.pixel_correction_diag_pub.published]
    assert decisions == [[1.,0.],[0.,8.]]
    return dict(camera_separation_s=.0005, accepted_and_reason_codes=decisions,
                committed_stamp=n._stamp_to_float(n.belief_stamp),
                expected='A and B each accepted once, with 0.0005 s supported prediction')


def pixel_refusal_freezes_anchor():
    # Use the real pixel callback with the linear observation fixture.
    from test_planner_node_correction_wiring import make_node
    n = make_node(now_s=10., belief_stamp_s=9.9, meas=(100.,100.))
    n._apply_pixel_correction(stamp(9.95))
    assert n._stamp_to_float(n.belief_stamp) == 9.9
    assert n.pixel_correction_diag_pub.published[-1][30] == 0.
    return dict(rejected_stamp=9.95, anchor_stamp_after=9.9,
                reason_code=n.pixel_correction_diag_pub.published[-1][31])


def alias_hazard():
    n = node(belief_stamp_s=9.9, now_s=10.)
    n._odom_cb(odom(9.9))
    out = n._apply_metric_correction(stamp(10.), np.zeros(2), np.eye(2)*.03)
    assert np.shares_memory(n.belief_m, out.next_m)
    assert np.shares_memory(n.belief_S, out.next_S)
    out.next_m[0] = 123.
    assert n.belief_m[0] == 123.
    return dict(shared_mean=True, shared_covariance=True,
                result_mutation_changes_anchor=True,
                classification='API hazard; no current production post-commit mutator found')


def provenance():
    names = ['planning.core.dynamics','planning.core.belief_correction','planning.core.motion_history',
             'planning.planners.base_planner','planning.nodes.unicycle_planner_node','sim.encoder_noise_node',
             'experiments.core.visibility_launch_common','planning.nodes.efe_agent_node']
    result = {}
    protocol_path = ROOT / 'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json'
    protocol = json.loads(protocol_path.read_text())['sources']
    for name in names:
        module = importlib.import_module(name)
        resolved = Path(module.__file__).resolve()
        relative = str(resolved.relative_to(ROOT))
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        result[name] = dict(import_path=module.__file__, resolved_source=relative, sha256=digest,
                            frozen_protocol_match=(digest == protocol[relative]) if relative in protocol else None)
    return result


def sanitize(value):
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, dict):
        return {key: sanitize(val) for key, val in value.items()}
    if isinstance(value, (list,tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (float,np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


if __name__ == '__main__':
    cases = [complete_trace, out_of_order_motion, missing_prefix_and_empty_history,
             outage_support_snapshot_race, frame_and_white_noise_checks,
             encoder_order_and_gap, quorum_information_reuse, per_camera_nonrejection_inflation,
             almost_simultaneous_camera, pixel_refusal_freezes_anchor, alias_hazard]
    results = {}
    for case in cases:
        try:
            results[case.__name__] = case()
        except Exception as exc:
            results[case.__name__] = {'probe_error': repr(exc)}
    results['source_provenance'] = provenance()
    results['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(sanitize(results), indent=2, sort_keys=True, allow_nan=False))
    if any('probe_error' in value for value in results.values() if isinstance(value, dict)):
        sys.exit(1)
