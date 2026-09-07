"""Audit 10: actual logger/filter callbacks, isolated files, independent accounting.

No ROS graph, inference, simulator or actuator is started. ROS message types are real;
clocks, publication/delivery boundaries and geometry are controlled fixtures. Assertions
record current defects as well as positive controls; exit zero does not mean repaired.
Run from the repository root: python3 experiments/runtime_integrity/logging_accounting_20260906/probe.py
"""
from __future__ import annotations

import ast
from collections import Counter, deque
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/planning'),
               str(ROOT / 'experiments/fusion_on_fixed_routes'),
               str(ROOT / 'scripts/visibility_comparison')]
import conftest  # noqa: F401 -- source package roots, same as repository tests
from experiments.nodes import experiment_logger as EL
from experiments.core.camera_opportunity_log import CameraOpportunityLog
from experiments.core import manifest as EM
from reliability.contracts import CameraObservation
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import String, Float64MultiArray
from test_planner_node_state_correction import make_state_node
from test_planner_node_correction_wiring import _Clock, _Logger, stamp
import aligned
import run_visibility_campaign as campaign

CAMERAS = tuple('camera_' + c for c in 'ABCDE')
UPDATE = {'accepted', 'accepted_bootstrap', 'reanchored'}
VALID = UPDATE | {'rejected', 'dropped'}
LOGGER_SOURCE = Path(inspect.getfile(EL)).resolve()
TREE = ast.parse(LOGGER_SOURCE.read_text())
INIT = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == '__init__')


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


def save(path, payload):
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def header(writer):
    for n in ast.walk(INIT):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'writerow' and ast.unparse(n.func.value) == 'self.' + writer):
            return ast.literal_eval(n.args[0])
    raise AssertionError(writer)


def logger(name):
    """Initialize the real logger's parameter/default assignments, omitting ROS/file setup.

    No behavioral logger method is copied/reimplemented. Actual constructor AST supplies
    parameters, counter defaults and CSV headers, so the written schema is the real one.
    """
    node = object.__new__(EL.ExperimentLogger)
    params = {}
    for n in ast.walk(INIT):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'declare_parameter'):
            params[ast.literal_eval(n.args[0])] = ast.literal_eval(n.args[1])
    params.update(log_perception_samples=True, heading_update_mode='coupled',
                  auto_stop_on_goal=False, stuck_window_s=0.0, pixel_timeout_s=.5)
    node.get_parameter = lambda key: NS(value=params[key])
    node.get_logger = lambda: _Logger()
    node._audit_clock = _Clock(10.)
    node.get_clock = lambda: node._audit_clock
    node.count_publishers = lambda topic: 1
    node.task_start_pose = None
    # The same constructor defaults as this source revision, with no ROS actions.
    for n in INIT.body:
        if (291 <= n.lineno <= 541 or 832 <= n.lineno <= 981) and isinstance(n, (ast.Assign, ast.AnnAssign)):
            exec(compile(ast.Module(body=[n], type_ignores=[]), str(LOGGER_SOURCE), 'exec'),
                 dict(vars(EL), self=node))
    node.run_dir = str(OUT / 'runs' / name)
    Path(node.run_dir).mkdir(parents=True, exist_ok=True)
    node.run_id = 'synthetic:' + name
    node._frame_sanity_logged = True
    node._latest_odom_map_pose = lambda: (True, node._audit_clock.seconds, 0., 0., 0.)
    node._geometry_safety_at_truth = lambda x, y: dict(
        min_wall_distance_m=100., min_obstacle_distance_m=100.,
        wall_penetration_m=0., obstacle_penetration_m=0., off_map=False, inside_no_go=False)
    node.camera_model = NS(pixel_to_world=lambda u, v: (float(u), float(v)))
    node.camera_pos_xy = np.array([0., 0.])
    node.fusion_decision = node.fusion_decision_stamp = node._fusion_obs_last_stamp = None
    node._fusion_decision_seq = 0
    node._obs_repeat_count, node._obs_seq_by_camera = {}, {}
    node._audit_handles = []
    for fileattr, writerattr, filename in (
            ('file', 'writer', 'experiment.csv'),
            ('fusion_obs_file', 'fusion_obs_writer', 'fusion_observations.csv'),
            ('assimilation_file', 'assimilation_writer', 'correction_assimilations.csv'),
            ('perception_file', 'perception_writer', 'perception.csv'),
            ('plan_file', 'plan_writer', 'plan_samples.csv')):
        handle = open(Path(node.run_dir) / filename, 'w', newline='')
        setattr(node, fileattr, handle)
        setattr(node, writerattr, csv.writer(handle))
        getattr(node, writerattr).writerow(header(writerattr))
        node._audit_handles.append(handle)
    node.camera_opportunity_file = open(Path(node.run_dir) / 'camera_opportunities.jsonl', 'w')
    node.camera_opportunity_log = CameraOpportunityLog(node.camera_opportunity_file)
    node._audit_handles.append(node.camera_opportunity_file)
    save(Path(node.run_dir) / 'run_manifest.json', dict(
        logging_schema_version=7, evidence_kind='synthetic_callback_audit',
        source_sha256=hashlib.sha256(LOGGER_SOURCE.read_bytes()).hexdigest()))
    return node


def close(node):
    for handle in node._audit_handles:
        handle.close()


def finish(node, when=20.):
    node._audit_clock.seconds = when
    with patch('threading.Timer') as timer:
        node._finish_run('synthetic_end', when)
        assert timer.call_args.args[0] == .15
    return json.loads((Path(node.run_dir) / 'run_summary.json').read_text())


def decision(batch, capture=1., x=0.):
    return dict(source_batch_id=batch, common_capture_stamp=capture, fused_stamp=capture,
                accepted_camera_ids=['camera_A', 'camera_B'],
                fused_xy=[x, 0.], fused_cov=[[.03, 0.], [0., .03]],
                observations=[dict(camera=c, obs_stamp=capture, xy=[x, 0.],
                                   cov=[[.03, 0.], [0., .03]], aligned_xy=[x, 0.],
                                   aligned_cov=[[.03, 0.], [0., .03]], used=True)
                              for c in CAMERAS[:2]])


def feed_decision(node, payload, when=None):
    if when is not None:
        node._audit_clock.seconds = when
    node._fusion_decision_cb(String(data=json.dumps(payload)))


def assimilation(batch, status='accepted', capture=1., apply=1.1, reason=None):
    return dict(schema_version=1, source_batch_id=batch, correction_stamp=capture,
                apply_stamp=apply, status=status,
                reason=reason if reason is not None else status,
                accepted=status in UPDATE, nis=1., belief_stamp_after=capture)


def feed_assim(node, payload):
    node._correction_assimilation_cb(String(data=json.dumps(payload)))


def observation(camera, batch, capture, hit=True):
    return CameraObservation(camera_id=camera, source_batch_id=batch, timestamp_s=capture,
        detection_valid=hit, pixel_uv=(100., 200.) if hit else None,
        bbox_xyxy=(90., 180., 110., 200.) if hit else None,
        bbox_bottom_uv=(100., 200.) if hit else None,
        selected_pixel_source='bbox_bottom' if hit else 'none',
        detector_score=.8 if hit else 0., detector_score_raw=.8 if hit else 0.,
        image_receive_stamp_s=capture+.01, inference_start_stamp_s=capture+.02,
        inference_finish_stamp_s=capture+.05, publish_stamp_s=capture+.06,
        yolo_inference_wall_ms=30., detector_callback_wall_ms=50.,
        frame_age_at_publish_s=.06, image_frame_id=camera,
        calibration_id='synthetic').to_json()


def filter_node():
    n = make_state_node(belief_stamp_s=.8, now_s=1.1, pixel_timeout_s=.5)
    n.require_state_correction_envelope = True
    n._seen_state_source_batch_ids = set()
    n.state_reanchor_m = 0.
    n.heading_update_mode = 'coupled'
    n.use_odom_for_predict = True
    n.odom_vel = np.zeros(2)
    n.stale_belief_inflate_m2_per_s = n.stale_belief_inflate_cap_m2 = 0.
    return n


def reconstruct(run):
    """Independent file-level oracle: no logger counters, no production validator."""
    obs, ass = read_csv(run / 'fusion_observations.csv'), read_csv(run / 'correction_assimilations.csv')
    fused_ids = {r['source_batch_id'] for r in obs}
    counts = Counter(r['source_batch_id'] for r in ass)
    summary = json.loads((run / 'run_summary.json').read_text())
    statuses = Counter(r['status'] for r in ass)
    return dict(fusion_batches=sorted(fused_ids), assimilation_rows=len(ass),
        statuses=dict(statuses), update_batches=[r['source_batch_id'] for r in ass if r['status'] in UPDATE],
        missing=sorted(fused_ids - set(counts)), extra=sorted(set(counts) - fused_ids),
        duplicated=[b for b, n in counts.items() if n != 1],
        unclassifiable=[r['source_batch_id'] for r in ass if r['status'] not in VALID],
        unreasoned=[r['source_batch_id'] for r in ass if r['status'] in {'rejected', 'dropped'} and not r['reason'].strip()],
        summary_count=summary['correction_assimilation_count'], summary_valid=summary['valid_run'],
        summary_count_matches=len(ass) == summary['correction_assimilation_count'],
        dropped_fraction=(statuses['dropped']/len(ass) if ass else 0.),
        summary_dropped_fraction=summary['correction_dropped_fraction'])


def mini_run():
    n = logger('mini_run')
    run = Path(n.run_dir)
    f = filter_node()
    f.belief_m = f.belief_S = f.belief_stamp = None
    f.planner_belief_pub = NS(publish=n._planner_belief_cb)
    journal = []
    batches = {}
    def emit(kind, batch, capture, receipt, x, deliver=True):
        ident = 'strict:' + ','.join(c + '@' + str(round(capture*1e9)) for c in CAMERAS)
        batches[batch] = ident
        for camera in CAMERAS:
            n.camera_opportunity_log.append(camera, observation(camera, ident, capture), receipt)
        d = decision(ident, capture, x)
        envelope = dict(schema_version=1, frame_id='map_bev', source_batch_id=ident,
                        common_capture_stamp=capture, correction_stamp=capture,
                        xy=[x, 0.], covariance_m2=[[.03, 0.], [0., .03]], accepted_camera_ids=list(CAMERAS[:2]))
        journal.append(dict(kind=kind, source_batch_id=ident, capture=capture, receipt=receipt,
                            fused_envelope=envelope, deliver_assimilation=deliver))
        feed_decision(n, d, receipt+.001)
        f._clock.seconds = receipt
        before = len(f.correction_assimilation_pub.published)
        f._state_correction_envelope_cb(String(data=json.dumps(envelope)))
        emitted = f.correction_assimilation_pub.published[before:]
        assert len(emitted) == 1
        journal[-1]['generated_assimilation'] = json.loads(emitted[0])
        if deliver:
            n._correction_assimilation_cb(String(data=emitted[0]))
        f._belief_publish_tick()
        n._audit_clock.seconds = receipt+.01
        n._log_once()
        return envelope
    emit('bootstrap', 'b0', 1., 1.1, 0.)
    accepted = emit('accepted', 'b1', 1.2, 1.3, .05)
    emit('rejected', 'b2', 1.4, 1.5, 50.)
    emit('delayed_accepted', 'b3', 1.6, 2., .06)
    emit('stale_dropped', 'b4', 1.8, 3., .06)
    emit('missing_assimilation_delivery', 'b5', 2.2, 3.2, .06, deliver=False)
    # Misses and a partial camera batch are physical opportunities, not fictitious corrections.
    for batch, capture, cameras, hit in [('miss', 2.6, CAMERAS, False), ('missing_camera_E', 2.8, CAMERAS[:-1], True)]:
        for camera in cameras:
            n.camera_opportunity_log.append(camera, observation(camera, batch, capture, hit), 3.3)
        journal.append(dict(kind=batch, source_batch_id=batch, expected_cameras=list(CAMERAS), delivered_cameras=list(cameras)))
    feed_decision(n, dict(source_batch_id='miss', accepted_camera_ids=[], reasons=['no_eligible_synchronous_observations']), 3.4)
    # Ordinary duplicate detector delivery is visible once and marked as such.
    n.camera_opportunity_log.append('camera_A', observation('camera_A', batches['b1'], 1.2), 3.5)
    n.camera_opportunity_log.append('camera_A', '{bad JSON', 3.6)
    # Duplicate/malformed receiver exceptions do not consume a second measurement.
    fatal = []
    f._fatal_experiment_stop = lambda context, exc: fatal.append(dict(context=context, error=str(exc)))
    before = len(f.correction_assimilation_pub.published)
    prior = f.belief_m.copy()
    f._state_correction_envelope_cb(String(data=json.dumps(accepted)))
    assert len(f.correction_assimilation_pub.published) == before
    np.testing.assert_array_equal(f.belief_m, prior)
    malformed = dict(accepted, source_batch_id='malformed_fused', frame_id='wrong_frame')
    journal.append(dict(kind='malformed_fused', source_batch_id='malformed_fused', fused_envelope=malformed))
    feed_decision(n, decision('malformed_fused', 3.0), 3.7)
    # The real duplicate case terminates its process. Use a separate receiver fixture
    # for the malformed case rather than claiming live execution continues after fatal.
    malformed_receiver = filter_node()
    malformed_receiver._fatal_experiment_stop = f._fatal_experiment_stop
    malformed_receiver._state_correction_envelope_cb(String(data=json.dumps(malformed)))
    assert not malformed_receiver.correction_assimilation_pub.published
    summary = finish(n, 4.)
    close(n)
    with open(run / 'producer_oracle.jsonl', 'w') as handle:
        for entry in journal:
            handle.write(json.dumps(clean(entry), allow_nan=False) + '\n')
    result = reconstruct(run)
    opportunities = jsonl(run / 'camera_opportunities.jsonl')
    result.update(fatal_events=fatal, opportunity_rows=len(opportunities),
        detector_misses=sum(r.get('valid_contract', False) and not r['observation']['detection_valid'] for r in opportunities),
        duplicate_opportunity_deliveries=sum(r.get('duplicate', False) for r in opportunities),
        malformed_opportunity_deliveries=sum(not r.get('valid_contract', False) for r in opportunities),
        generated_assimilations=sum('generated_assimilation' in r for r in journal),
        physical_unique_fused_publications=sum('fused_envelope' in r for r in journal),
        campaign_validation=campaign._verify_correction_assimilations(run))
    assert result['statuses'] == {'accepted_bootstrap': 1, 'accepted': 2, 'rejected': 1, 'dropped': 1}
    assert len(result['missing']) == 2 and result['summary_valid']
    assert result['detector_misses'] == 5 and result['duplicate_opportunity_deliveries'] == 1
    return result


def logging_faults():
    result = {}
    for case in ('missing', 'extra', 'unreasoned_rejected', 'unknown_status', 'duplicate_delivery', 'malformed_delivery'):
        n = logger(case)
        feed_decision(n, decision('expected'))
        if case != 'missing':
            feed_assim(n, assimilation('extra' if case == 'extra' else 'expected',
                status=('rejected' if case == 'unreasoned_rejected' else 'teleported' if case == 'unknown_status' else 'accepted'),
                reason='' if case == 'unreasoned_rejected' else 'documented'))
        if case == 'duplicate_delivery': feed_assim(n, assimilation('expected'))
        if case == 'malformed_delivery': n._correction_assimilation_cb(String(data='{broken'))
        finish(n)
        close(n)
        result[case] = reconstruct(Path(n.run_dir))
        result[case]['campaign_validation'] = campaign._verify_correction_assimilations(Path(n.run_dir))
    assert all(result[c]['summary_valid'] for c in ('missing', 'extra', 'unreasoned_rejected'))
    assert result['duplicate_delivery']['assimilation_rows'] == 1
    assert result['duplicate_delivery']['campaign_validation'][0]
    assert result['malformed_delivery']['assimilation_rows'] == 1
    n = logger('receipt_collision')
    for ident, capture in [('a', 1.), ('b', 1.2)]:
        feed_decision(n, decision(ident, capture), 10.)
        feed_assim(n, assimilation(ident, capture=capture))
    finish(n); close(n)
    result['receipt_collision'] = reconstruct(Path(n.run_dir))
    assert result['receipt_collision']['fusion_batches'] == ['a']
    n = logger('post_summary_tail')
    feed_decision(n, decision('before'), 10.)
    feed_assim(n, assimilation('before'))
    finish(n, 10.1)
    feed_decision(n, decision('after', 1.2), 10.2)
    feed_assim(n, assimilation('after', capture=1.2))
    n._log_once()
    close(n)
    result['post_summary_tail'] = reconstruct(Path(n.run_dir))
    result['post_summary_tail']['last_log_stamp'] = read_csv(Path(n.run_dir)/'experiment.csv')[-1]['stamp']
    assert not result['post_summary_tail']['summary_count_matches']
    n = logger('summary_write_failure')
    feed_decision(n, decision('a')); feed_assim(n, assimilation('a'))
    try:
        with patch.object(EL.json, 'dump', side_effect=OSError('synthetic disk failure')):
            finish(n)
    except OSError:
        pass
    result['summary_write_failure'] = dict(completed_latched=n._completed,
        stop_latched=n._stop_requested, file_bytes=(Path(n.run_dir)/'run_summary.json').stat().st_size,
        parsed=campaign._read_run_summary(Path(n.run_dir)))
    # Real finish returns early forever after the failed write.
    n._finish_run('retry', 21.)
    assert result['summary_write_failure']['completed_latched']
    assert result['summary_write_failure']['parsed'] is None
    close(n)
    n = logger('opportunity_nonfinite')
    raw = json.loads(observation('camera_A', 'nan_case', 1.))
    raw['detector_score_raw'] = float('nan')
    try:
        n.camera_opportunity_log.append('camera_A', json.dumps(raw), 1.1)
    except ValueError as exc:
        result['opportunity_nonfinite'] = dict(exception=str(exc), rows_counter=n.camera_opportunity_log.rows,
                                               written_rows=0, seen=len(n.camera_opportunity_log.seen))
    else: raise AssertionError('nonfinite serialization must reproduce')
    retry = n.camera_opportunity_log.append('camera_A', observation('camera_A', 'nan_case', 1.), 1.2)
    result['opportunity_nonfinite']['first_written_is_duplicate'] = retry['duplicate']
    assert retry['duplicate']
    close(n)
    return result


def held_and_diagnostics():
    n = logger('held_commands_and_beliefs')
    n._first_cmd_stamp = 0.
    n._gt_buf = deque([(0., 0., 0., 0.), (20., 0., 0., 0.)])
    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = 'map_bev'; pose.header.stamp = stamp(1.)
    pose.pose.pose.position.x = 1.; pose.pose.pose.orientation.w = 1.
    pose.pose.covariance = [0.] * 36
    for t in (1.1, 1.2, 1.3, 1.4):
        n._audit_clock.seconds = t
        n._planner_belief_cb(pose) if t == 1.1 else None
        n._log_once()
    pose2 = PoseWithCovarianceStamped(); pose2.header.stamp = stamp(2.); pose2.pose.pose.orientation.w = 1.
    pose2.pose.pose.position.x = 0.; n._planner_belief_cb(pose2)
    # New raw receipt belongs to .1; stale noise diagnostic belongs to an old .2 request.
    raw = Twist(); raw.linear.x = .1
    actual = Twist(); actual.linear.x = .09
    n._audit_clock.seconds = 2.1
    n._cmd_raw_cb(raw); n._cmd_cb(actual)
    n._cmd_noise_diag_cb(Float64MultiArray(data=[1., 1., .2, 0., .18, 0., .9, 1., 0., 0.]))
    n._log_once()
    summary = finish(n, 2.2); close(n)
    rows = read_csv(Path(n.run_dir)/'experiment.csv')
    result = dict(held_rows=len(rows), unique_belief_stamps=len({r['planner_belief_stamp'] for r in rows}),
        summary_tick_weighted_mean=summary['mean_belief_error_gt_m'], unique_event_mean=.5,
        last_raw_value=rows[-1]['cmd_raw_v'], last_raw_receipt=rows[-1]['cmd_raw_stamp'],
        last_output_value=rows[-1]['cmd_v'], reported_noise_error=rows[-1]['cmd_noise_v_error'],
        paired_diagnostic_error=.18-.2)
    assert result['summary_tick_weighted_mean'] == .8
    assert float(result['last_raw_value']) == .2 and float(result['last_raw_receipt']) == 2.1
    return result


def additional_boundaries():
    result = {}
    n = logger('status_control')
    f = filter_node()
    f.correction_assimilation_pub = NS(publish=n._correction_assimilation_cb)
    for i, status in enumerate(('accepted_bootstrap', 'accepted', 'reanchored', 'rejected', 'dropped')):
        capture = 1. + i*.2
        feed_decision(n, decision(status, capture), capture+.05)
        f._clock.seconds = capture+.1
        f._publish_correction_assimilation(source_batch_id=status, stamp_msg=stamp(capture),
            status=status, reason='synthetic_control')
    summary = finish(n, 3.); close(n)
    result['status_control'] = reconstruct(Path(n.run_dir))
    assert len(result['status_control']['update_batches']) == 3
    assert result['status_control']['summary_count_matches']
    assert campaign._verify_correction_assimilations(Path(n.run_dir))[0]
    # Arrival gaps and accepted-update gaps are different; trailing outage is not closed.
    n = logger('gap_semantics')
    for i in range(5):
        feed_decision(n, decision(str(i), float(i+1)), i+1.1)
        feed_assim(n, assimilation(str(i), capture=float(i+1),
                                  status='accepted' if i in (0,4) else 'rejected', reason='nis_too_large'))
    s = finish(n, 10.); close(n)
    result['gap_semantics'] = dict(summary_gap_s=s['longest_correction_gap_s'],
        largest_accepted_update_interval_s=4., trailing_no_accepted_update_s=5., stop_stamp=10.)
    assert s['longest_correction_gap_s'] == 1.
    # Two source batches within one microsecond are distinct identities in exact-ns producer.
    n = logger('identity_precision')
    for batch, capture, receipt in [('a', 1., 10.), ('b', 1.0000004, 10.1)]:
        feed_decision(n, decision(batch, capture), receipt)
    n.camera_opportunity_log.append('camera_A', observation('camera_A', 'changed_payload', 1.), 11.)
    changed = n.camera_opportunity_log.append('camera_A', observation('camera_A', 'changed_payload', 1.2), 11.1)
    badschema = json.loads(observation('camera_B', 'bad_schema', 2.)); badschema['schema_version'] = 'future'
    recorded = n.camera_opportunity_log.append('camera_B', json.dumps(badschema), 12.)
    try: CameraObservation.from_dict(badschema)
    except Exception as exc: contract_error = str(exc)
    else: raise AssertionError('bad schema accepted by producer contract')
    close(n)
    result['identity_precision'] = dict(fusion_rows=[{k:r[k] for k in ('source_batch_id','camera','obs_repeat','obs_stamp')}
        for r in read_csv(Path(n.run_dir)/'fusion_observations.csv')],
        same_camera_batch_changed_stamp_duplicate=changed['duplicate'],
        logger_accepts_unknown_schema=recorded['valid_contract'], manager_contract_error=contract_error)
    assert result['identity_precision']['fusion_rows'][2]['obs_repeat'] == '1'
    n = logger('headers_at_finish')
    finish(n)
    result['headers_at_finish'] = {k:(Path(n.run_dir)/k).stat().st_size for k in
                                  ('experiment.csv','fusion_observations.csv','correction_assimilations.csv')}
    assert result['headers_at_finish']['fusion_observations.csv'] == 0
    assert result['headers_at_finish']['correction_assimilations.csv'] == 0
    close(n)
    # Heartbeat spacing cannot bound the constant transport latency of a truth stream.
    n = logger('truth_receipt_latency')
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    for actual_capture, receipt in [(1., 6.), (1.001, 6.001), (1.002, 6.002)]:
        tr = TransformStamped(); tr.child_frame_id = 'turtlebot3'
        tr.transform.translation.x = actual_capture; tr.transform.rotation.w = 1.
        n._audit_clock.seconds = receipt
        n._ground_truth_cb(TFMessage(transforms=[tr]))
    s = finish(n, 7.); close(n)
    result['truth_receipt_latency'] = dict(stamp_source=s['gt_stamp_source'],
        reported_max_sample_interval_s=s['gt_sample_interval_max_s'], injected_delay_s=5.)
    return result


def selected_runs():
    selected = ROOT / 'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json'
    payload = json.loads(selected.read_text())
    output = {'selection': str(selected.relative_to(ROOT)), 'selection_sha256': hashlib.sha256(selected.read_bytes()).hexdigest(), 'runs': []}
    for entry in payload['runs']:
        run = ROOT / entry['run']
        matched = {file: hashlib.sha256((run/file).read_bytes()).hexdigest() == digest for file,digest in entry['files'].items()}
        assert all(matched.values()), matched
        manifest = json.loads((run/'run_manifest.json').read_text())
        summary = json.loads((run/'run_summary.json').read_text())
        # Required loader supplies typed events; no accuracy statistic is computed.
        ass = aligned.assimilations(run)
        obs = aligned.observations(run)
        raw = aligned.rows(run)
        data = reconstruct(run)
        data.update(run=entry['run'], arm=entry['arm'], verified_files=matched,
                    logging_schema=aligned.schema_version(run),
                    source_batch_count=len({r['source_batch_id'] for r in obs}),
                    stop_stamp=summary['stop_stamp'], last_logger_stamp=raw[-1]['stamp'],
                    last_assimilation_apply=max(r['apply_stamp'] for r in ass),
                    contact_messages=summary['contact_messages_seen'],
                    contact_publishers=summary['contact_topic_publishers'],
                    artifact_hash_fields={k:v for k,v in manifest.items() if 'sha256' in k},
                    trace=dict(fusion=read_csv(run/'fusion_observations.csv')[0],
                               assimilation=read_csv(run/'correction_assimilations.csv')[0]))
        bid = data['trace']['assimilation']['source_batch_id']
        data['trace']['camera_opportunities'] = [r for r in jsonl(run/'camera_opportunities.jsonl')
            if r.get('observation',{}).get('source_batch_id') == bid]
        data['trace']['belief_first_after_apply'] = next((r for r in raw if float(r['planner_belief_stamp']) > ass[0]['apply_stamp']), None)
        output['runs'].append(data)
    return output


def main():
    sources = [LOGGER_SOURCE, ROOT/'src/experiments/experiments/core/camera_opportunity_log.py',
               ROOT/'src/experiments/experiments/core/manifest.py', ROOT/'src/unav_common/unav_common/manifest.py',
               ROOT/'src/planning/planning/nodes/unicycle_planner_node.py',
               ROOT/'src/reliability/reliability/nodes/camera_manager_node.py',
               ROOT/'src/planning/planning/nodes/efe_agent_node.py',
               ROOT/'src/sim/sim/actuation_noise_node.py', Path(aligned.__file__).resolve(), Path(campaign.__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    expected = json.loads((OUT/'source_manifest.json').read_text())
    if hashes != expected:
        raise RuntimeError('Audited runtime sources changed; use reproduce_snapshot.py. '
                           'Baseline evidence has not been overwritten.')
    results = dict(mini_run=mini_run(), logging_faults=logging_faults(), held_and_diagnostics=held_and_diagnostics(),
                   additional_boundaries=additional_boundaries())
    save(OUT/'results.json', results)
    save(OUT/'selected_run_accounting.json', selected_runs())
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    print(json.dumps(clean(results), indent=2, allow_nan=False))


if __name__ == '__main__': main()
