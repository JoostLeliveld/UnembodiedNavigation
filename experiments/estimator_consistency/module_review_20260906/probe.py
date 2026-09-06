"""Read-only synthetic probes of current estimator wiring; no ROS graph or drives."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/planning'))
sys.path.insert(0, str(ROOT / 'experiments/fusion_on_fixed_routes'))
from test_planner_node_state_correction import make_state_node, state_msg
from test_planner_node_correction_wiring import stamp
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.dynamics import coherent_drift_block
from planning.core.motion_history import covers_interval
from std_msgs.msg import String
import aligned


def planner(coherent=False):
    p = object.__new__(UnicyclePlannerBase)
    p.dt = .25
    p.process_noise_xy = .01
    p.process_noise_theta = .02
    p.coherent_drift = coherent
    return p


def node(**kwargs):
    n = make_state_node(**kwargs)
    n.planner = planner()
    n.heading_update_mode = 'coupled'
    n.state_reanchor_m = 0.
    n.stale_belief_inflate_m2_per_s = 0.
    n.stale_belief_inflate_cap_m2 = 0.
    n._resolve_plan_frame_id = lambda: 'map_bev'
    return n


def stale_motion():
    n = node(belief_stamp_s=9.9, now_s=10.)
    n.use_odom_for_predict = True
    n._odom_log = [(1., .2, 0.)]
    n._cmd_log = [(9.8, 0., 0.)]
    result = n._apply_metric_correction(stamp(10.), np.array([.02, 0.]), np.eye(2)*.03)
    assert not covers_interval(n._odom_log, 9.9, 10., n.state_max_predict_dt_s)
    assert result.accepted
    assert np.isclose(result.m_pred[0], .02)
    assert result.replay_meta['motion_replay_source_code'] == 1.
    assert result.replay_meta['cmd_replay_used_fallback'] == 0.
    return dict(last_odom_stamp=1., interval=[9.9, 10.],
                coverage=False, accepted=result.accepted,
                predicted_x=float(result.m_pred[0]), replay=result.replay_meta)


def untracked_bootstrap():
    n = node(now_s=10.)
    n.belief_m = n.belief_S = n.belief_stamp = None
    n.require_state_correction_envelope = True
    n._seen_state_source_batch_ids = set()
    n._fatal_experiment_stop = lambda *args: (_ for _ in ()).throw(AssertionError(args))
    n._state_cb(state_msg(1., 2., seconds=9.95))
    assert n.belief_m is None
    n._resolve_state_belief_ekf(stamp(10.))
    assert np.allclose(n.belief_m[:2], [1., 2.])
    assert not n.correction_assimilation_pub.published
    msg = String()
    msg.data = json.dumps(dict(schema_version=1, frame_id='map_bev',
        source_batch_id='synthetic-bootstrap', correction_stamp=9.95,
        xy=[1., 2.], covariance_m2=[[.03, 0.], [0., .03]]))
    n._state_correction_envelope_cb(msg)
    event = json.loads(n.correction_assimilation_pub.published[0])
    assert event['status'] == 'dropped'
    assert event['reason'] == 'not_newer_than_belief'
    event['nis'] = None  # Missing diagnostic; preserve strict JSON in review artifacts.
    return dict(consumed_xy=n.belief_m[:2].tolist(), ledger_event=event)


def load_logger_method(method):
    path = ROOT / 'src/experiments/experiments/nodes/experiment_logger.py'
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ExperimentLogger')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method)
    module = ast.Module(body=[fn], type_ignores=[])
    ns = {'math': math, 'json': json}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns[method]


def same_clock_batches():
    rows = []
    n = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
        fusion_obs_writer=SimpleNamespace(writerow=rows.append),
        fusion_obs_file=SimpleNamespace(flush=lambda: None),
        _fusion_obs_last_stamp=None, _fusion_decision_seq=0,
        _gt_xy=None, _gt_stamp=math.nan, _gt_at=lambda _: (False, math.nan, math.nan, math.nan),
        _obs_repeat_count={}, _obs_seq_by_camera={})
    callback = load_logger_method('_fusion_decision_cb')
    for k, cap in enumerate([9.8, 9.9]):
        payload = dict(source_batch_id=f'synthetic-{k}', fused_stamp=cap,
            common_capture_stamp=cap, fused_xy=[0., 0.], fused_cov=[[.01, 0.], [0., .01]],
            accepted_camera_ids=['camera_A'], observations=[dict(camera='camera_A',
                used=True, xy=[0., 0.], cov=[[.01, 0.], [0., .01]], obs_stamp=cap)])
        callback(n, SimpleNamespace(data=json.dumps(payload)))
    assert len(rows) == 1
    return dict(delivered_distinct_batches=2, logged_rows=len(rows),
                logged_batch_ids=[r[2] for r in rows])


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def premature_event_belief():
    # A stationary reference and one correction captured at 1.0, applied at 1.5.
    with tempfile.TemporaryDirectory(prefix='synthetic_event_', dir=Path(__file__).parent) as tmp:
        run = Path(tmp)
        (run / 'run_manifest.json').write_text(json.dumps({'logging_schema_version': 6}))
        table = []
        for i in range(12):
            t = round(.9 + .1*i, 4)
            table.append(dict(stamp=t, gt_available=1, gt_stamp=t, gt_x=0., gt_y=0.,
                gt_yaw=0., planner_belief_stamp=t, planner_belief_x=1. if t < 1.5 else 0.,
                planner_belief_y=0., planner_cov_x=.01, planner_cov_xy=0., planner_cov_y=.01))
        write_csv(run/'experiment.csv', table)
        write_csv(run/'fusion_observations.csv', [dict(stamp=1.5, source_batch_id='synthetic-event',
            camera='A', used=1, obs_stamp=1., obs_x=0., obs_y=0., obs_cov_xx=.01,
            obs_cov_xy=0., obs_cov_yy=.01, fused_stamp=1., fused_x=0., fused_y=0.,
            fused_cov_xx=.01, fused_cov_xy=0., fused_cov_yy=.01, n_candidates=1, n_used=1)])
        write_csv(run/'correction_assimilations.csv', [dict(source_batch_id='synthetic-event',
            correction_stamp=1., apply_stamp=1.5, belief_stamp_after=1., status='accepted',
            reason='accepted', accepted=1, nis=1.)])
        result = aligned.belief_at_fusion_events(run)[0]
        assert np.isclose(result['error_cm'], 100.)
        chosen_stamp = 1. + result['belief_lag_after_fusion_s']
        assert chosen_stamp < 1.5
        return dict(capture_stamp=1., apply_stamp=1.5, chosen_belief_stamp=chosen_stamp,
                    scorer_label=result['assimilation_status'], synthetic_error_cm=result['error_cm'])


def q_diagnostic():
    n = node(belief_stamp_s=9.9, now_s=10.)
    n.use_odom_for_predict = True
    n._odom_log = [(9.9, 0., 0.)]
    _, P = n._predict_belief_to_now(np.zeros(3), np.zeros((3, 3)), np.zeros(2), .1, stamp(10.))
    ratio = n._latest_Q_theta_theta / P[2, 2]
    assert np.isclose(ratio, 4.)
    return dict(dt=n.planner.dt, applied_heading_process_variance=float(P[2, 2]),
                logged_Q_theta_theta=n._latest_Q_theta_theta, ratio=float(ratio))


def coherent_wrapper():
    covariances = []
    for flag in [False, True]:
        p = planner(flag)
        m, P = np.zeros(3), np.zeros((3, 3))
        for _ in range(50):
            m, P = p.predict(m, P, np.array([.22, 0.]), dt=.1)
        covariances.append(P)
    delta = covariances[1] - covariances[0]
    expected = coherent_drift_block(1.1, 0.)
    ratio = expected[1, 1] / delta[1, 1]
    assert np.isclose(ratio, 50.)
    # The ordinary white model DOES compose correctly when F P F' is retained.
    _, white_once = planner().predict(np.zeros(3), np.zeros((3, 3)), np.array([.22, 0.]), dt=5.)
    assert np.allclose(white_once, covariances[0])
    return dict(steps=50, runtime_coherent_cross_variance=float(delta[1, 1]),
                declared_total_cross_variance=float(expected[1, 1]), variance_ratio=float(ratio),
                white_Q_composition_with_F_passes=True)


def backdated_heading():
    n = node(now_s=10., odom_yaw=1.)
    n.belief_m = n.belief_S = n.belief_stamp = None
    n.use_odom_for_predict = True
    n._odom_origin_stamp_s = 0.
    n._odom_log = [(9.8, 0., 1.), (9.9, 0., 1.), (10., 0., 1.)]
    n._apply_metric_correction(stamp(9.8), np.zeros(2), np.eye(2)*.03)
    capture_theta = float(n.belief_m[2])
    m, _ = n._predict_belief_to_now(n.belief_m, n.belief_S, np.zeros(2), .2, stamp(10.))
    assert np.isclose(capture_theta, 1.)
    assert np.isclose(m[2], 1.2)
    return dict(capture_stamp=9.8, latest_odom_stamp=10., latest_odom_yaw=1.,
                capture_yaw_from_scripted_motion=.8, bootstrap_yaw=capture_theta,
                predicted_yaw_now=float(m[2]))


def unsupported_heading():
    n = node(belief_stamp_s=0., now_s=10.1)
    n.use_odom_for_predict = True
    n._odom_log = [(9., 0., 0.), (10., 0., 0.)]
    before = float(n.belief_S[2, 2])
    n._advance_belief_over_outage(stamp(10.), 10.)
    added = float(n.belief_S[2, 2]) - before
    assert np.isclose(added, .0004*1.5)
    return dict(total_gap_s=10., replay_cap_s=1.5, unknown_motion_s=8.5,
                heading_variance_added=added, xy_variance_added=float(n.belief_S[0, 0]-.05))


def stale_quorum():
    from test_planner_node_per_camera_correction import observation as make_obs
    n = node(belief_stamp_s=10., now_s=12.)
    n.state_reanchor_m = 2.
    n._seen_map_observation_stamps = {}
    ordered = [make_obs('camera_A', 3., 0., seconds=9.),
               make_obs('camera_B', 3.01, 0., seconds=9.)]
    n._apply_map_observations(ordered)
    assert np.isclose(n.belief_m[0], 3.01)
    assert np.isclose(n._stamp_to_float(n.belief_stamp), 9.)
    return dict(now=12., belief_stamp_before=10., old_camera_stamp=9.,
                belief_stamp_after=n._stamp_to_float(n.belief_stamp),
                belief_x_after=float(n.belief_m[0]))


def encoder_covariance_at_stop():
    from sim.encoder_noise_node import EncoderNoiseNode
    n = object.__new__(EncoderNoiseNode)
    n._pose_cov = [[.0001, 0., 0.], [0., .0001, 0.], [0., 0., .0001]]
    n._linear_scale_jacobian = [0., 0., 0.]
    n.correlation_alpha = .8
    n.linear_slip_std = .05
    n.angular_slip_std = .03
    n.linear_additive_std = .004
    n.angular_additive_std = .020
    n.covariance_floor_m2 = n.covariance_floor_yaw_rad2 = 1e-8
    for _ in range(100):
        n._propagate_pose_covariance(theta=0., v_true=0., w_true=0., dt=.1)
    assert n._pose_cov[0][0] > .0001
    assert n._pose_cov[2][2] > .0001
    return dict(stationary_duration_s=10., covariance_before=.0001,
                covariance_x_after=n._pose_cov[0][0], covariance_yaw_after=n._pose_cov[2][2],
                generator_stop_branch_injects_no_noise=True)


def duplicate_pixel():
    from test_planner_node_correction_wiring import make_node
    n = make_node()
    n._apply_pixel_correction(stamp(9.95))
    first = float(n.belief_S[0, 0])
    n._apply_pixel_correction(stamp(9.95))
    repeated = float(n.belief_S[0, 0])
    n._apply_pixel_correction(stamp(9.90))
    assert repeated < first
    assert np.isclose(n._stamp_to_float(n.belief_stamp), 9.9)
    return dict(first_xy_variance=first, duplicate_xy_variance=repeated,
                stamp_before_old_delivery=9.95, stamp_after_old_delivery=n._stamp_to_float(n.belief_stamp))


if __name__ == '__main__':
    cases = [stale_motion, untracked_bootstrap, same_clock_batches, premature_event_belief,
             q_diagnostic, coherent_wrapper, backdated_heading, unsupported_heading,
             stale_quorum, encoder_covariance_at_stop, duplicate_pixel]
    results = {}
    for case in cases:
        try:
            results[case.__name__] = case()
        except Exception as exc:
            results[case.__name__] = {'probe_error': repr(exc)}
    files = ['src/planning/planning/core/dynamics.py',
             'src/planning/planning/core/belief_correction.py',
             'src/planning/planning/core/motion_history.py',
             'src/planning/planning/planners/base_planner.py',
             'src/planning/planning/nodes/unicycle_planner_node.py',
             'src/sim/sim/actuation_noise_node.py',
             'src/sim/sim/encoder_noise_node.py',
             'src/experiments/experiments/nodes/experiment_logger.py',
             'experiments/fusion_on_fixed_routes/aligned.py']
    results['source_sha256'] = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    text = json.dumps(results, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if any('probe_error' in value for value in results.values()):
        sys.exit(1)
