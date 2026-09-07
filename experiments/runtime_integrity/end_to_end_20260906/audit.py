"""Bounded audit 15; no ROS graph, simulator, source edits or campaign writes.

Successful execution means observations were recorded, not that acceptance passed.
Component probes intentionally demonstrate defects. Only this directory is written.
Run after source install/setup.bash, with OPENBLAS_NUM_THREADS=1.
"""
from __future__ import annotations

import ast
import collections
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import conftest  # package paths only
for relative in ('tests/planning', 'tests/perception', 'experiments/fusion_on_fixed_routes'):
    sys.path.insert(0, str(ROOT / relative))
import aligned
from geometry_msgs.msg import Twist, PoseStamped
from test_planner_node_correction_wiring import _Clock, _Logger, stamp
from test_runtime_transactions import command_node, Publisher
from sim.actuation_noise_node import ActuationNoiseNode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save(name, data):
    (HERE / name).write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')


def identity():
    files = subprocess.check_output(['rg', '--files', '--hidden', 'src', 'scripts', 'experiments',
                                     'tests'], cwd=ROOT, text=True).splitlines()
    extensions = {'.py', '.yaml', '.json', '.sdf', '.xacro', '.xml', '.cfg', '.toml'}
    sources = {f: sha(ROOT / f) for f in sorted(files)
               if Path(f).suffix in extensions and '__pycache__' not in f
               and 'source_snapshot/' not in f
               and not f.startswith('experiments/runtime_integrity/end_to_end_20260906/')
               and not any(s in f for s in ('/results.json', '/protocol.json', '/wiring.json',
                                           '/source_manifest.json', '/source_sha256.json'))}
    docs = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'docs/module_audits').glob('*.md'))}
    return dict(source_files=sources, reports=docs,
                head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                python=sys.executable, audit_sha256=sha(Path(__file__)))


def load(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def component_probes():
    estimator = load('experiments/estimator_consistency/module_audit_01_20260906/probe.py', 'audit15_estimator')
    timing = load('experiments/runtime_integrity/timing_callbacks_20260906/probe.py', 'audit15_timing')
    record = {}
    cases = [getattr(estimator, n) for n in (
        'complete_trace', 'out_of_order_motion', 'missing_prefix_and_empty_history',
        'outage_support_snapshot_race', 'frame_and_white_noise_checks', 'encoder_order_and_gap',
        'quorum_information_reuse', 'per_camera_nonrejection_inflation',
        'almost_simultaneous_camera', 'pixel_refusal_freezes_anchor', 'alias_hazard')]
    cases += timing.PROBES
    cases += [getattr(estimator.previous, n) for n in (
        'stale_motion', 'untracked_bootstrap', 'same_clock_batches', 'premature_event_belief',
        'q_diagnostic', 'coherent_wrapper', 'backdated_heading', 'unsupported_heading',
        'stale_quorum', 'encoder_covariance_at_stop', 'duplicate_pixel')]
    for case in cases:
        try:
            record[case.__name__] = dict(reproduced=True, observed=case())
        except Exception as exc:
            record[case.__name__] = dict(reproduced=False, error=repr(exc))
    record['provenance'] = estimator.provenance()
    save('component_probes.json', record)
    return estimator, record


def command_fixture():
    # Extract just imports and fixture definitions from audit 03. Do not execute
    # its old-baseline assertions or overwrite its results.json.
    path = ROOT / 'experiments/command_execution_audit_20260906/probe.py'
    body = []
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, (ast.Import, ast.ImportFrom)) or (
                isinstance(n, ast.FunctionDef) and n.name in ('local_node', 'adapter', 'pairs', 'send', 'outs')):
            body.append(n)
    env = dict(__file__=str(path), ROOT=ROOT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), env)
    return env


def command_checks():
    f = command_fixture()
    result = {}
    for mode in ('ordinary_stop', 'fatal', 'correction', 'goal', 'delay'):
        n = f['local_node']()
        n.local_controller_type = 'turn_then_go'
        n.w_min, n.w_max = -1., 1.
        def dispatch(m, target):
            if mode in ('ordinary_stop', 'fatal'):
                n._fatal_stop_triggered = mode == 'fatal'
                n._publish_safe_stop_command()
            if mode == 'correction':
                n.belief_m = np.array([0., 2., 1.])
                n.belief_stamp = stamp(10.)
            if mode == 'goal':
                goal = PoseStamped(); goal.pose.position.x = -1.; n._goal_cb(goal)
            if mode == 'delay':
                n._clock.seconds = 11.
            return np.array([[.2, .1]])
        n._dispatch_local_controller = dispatch
        n._plan_once()
        result['local_' + mode] = dict(commands=f['pairs'](n),
            belief=n.belief_m, installed=n._active_controls is not None,
            installed_time=None if n._active_plan_started_at is None else n._active_plan_started_at.nanoseconds / 1e9)
    for mode in ('silence', 'restart', 'backlog', 'reset_receipt_first', 'reset_watchdog_first', 'clip', 'stop'):
        n = f['adapter']()
        if mode != 'restart': f['send'](n)
        if mode in ('silence', 'backlog'): n._clock.seconds = 10.6; n._watchdog_tick()
        if mode == 'restart': n._clock.seconds = 100.; n._watchdog_tick()
        if mode == 'backlog': n._clock.seconds = 10.61; f['send'](n)
        if mode.startswith('reset'):
            n._clock.seconds = 9.
            if mode.endswith('watchdog_first'): n._watchdog_tick()
            f['send'](n); n._watchdog_tick()
        if mode == 'clip': n.enabled = False; f['send'](n, 2., 3.)
        if mode == 'stop': f['send'](n, 0., 0.)
        result['adapter_' + mode] = dict(commands=f['outs'](n), diagnostics=len(n.diag))
    # Couple an actual tape timer to the actual noisy-command receiver and watchdog.
    tape, adapter = command_node(), f['adapter']()
    published = []
    def deliver(msg):
        published.append([msg.linear.x, msg.angular.z])
        adapter._clock.seconds = tape._clock.seconds
        adapter._cmd_cb(msg)
    tape.cmd_pub = NS(publish=deliver)
    tape._active_plan_started_at = _Clock(10.).now()
    for t in (10., 10.1, 10.2, 10.3):
        tape._clock.seconds = t; tape._publish_active_plan_command()
    result['tape_to_adapter_expiry'] = dict(publisher=published, adapter=f['outs'](adapter),
        nominal_tape_end=10.25, zero_requested_at=10.3,
        physical_motion='not simulated; adapter outputs are not applied velocities')
    save('commands.json', result)
    return result


def event_stream(estimator):
    """Same bounded blind-turn stream, with independent complete/gapped branches."""
    result = {}
    for mode in ('complete', 'missing_prefix', 'missing_turn', 'missing_tail'):
        n = estimator.node(belief_stamp_s=0., now_s=0.)
        n.belief_m = np.zeros(3); n.belief_S = np.eye(3) * .01
        # Use the real process propagation; only odometry/observations are scripted.
        n.planner_belief_pub = Publisher()
        n._resolve_plan_frame_id = lambda: 'map_bev'
        history = [(k / 10, .2 if k < 20 or k >= 40 else 0., math.pi / 4 if 20 <= k < 40 else 0.)
                   for k in range(111)]
        if mode == 'missing_prefix': history = [e for e in history if e[0] >= 9.]
        if mode == 'missing_turn': history = [e for e in history if not 2. <= e[0] < 4.]
        if mode == 'missing_tail': history = [e for e in history if e[0] <= 7.]
        n._odom_log = history
        terminal = []
        for t, xy, bid in ((10., [.4, 1.2], 'return'), (10.2, [.4, 1.24], 'recovery'),
                           (10.1, [.4, 1.22], 'reordered'), (10.4, [.4, 1.28], 'next')):
            n._clock.seconds = max(n._clock.seconds, t + .1)
            payload = dict(schema_version=1, frame_id='map_bev', source_batch_id='audit15:' + mode + ':' + bid,
                           correction_stamp=t, xy=xy, covariance_m2=[[.03, .005], [.005, .03]])
            n._state_correction_envelope_cb(NS(data=json.dumps(payload)))
            row = json.loads(n.correction_assimilation_pub.published[-1])
            row.update(mean=n.belief_m.copy(), covariance=n.belief_S.copy())
            terminal.append(row)
        before = (n.belief_m.copy(), n.belief_S.copy(), n._stamp_to_float(n.belief_stamp))
        n._belief_publish_tick()
        unchanged = np.array_equal(before[0], n.belief_m) and np.array_equal(before[1], n.belief_S)
        result[mode] = dict(coverage=estimator.covers_interval(history, 0., 10., 1.5),
            terminal=terminal, publication_read_only=unchanged,
            independent_return_mean=[.4, 1.2, math.pi/2],
            publication_stamps=[n._stamp_to_float(x.header.stamp) for x in n.planner_belief_pub.messages])
    save('event_stream.json', result)
    return result


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)


def strict_accounting(run):
    """Audit oracle of the documented accounting rule, not a production loader fix."""
    obs = aligned.observations(run)
    try:
        terminal = aligned.assimilations(run)
    except ValueError as exc:
        return [str(exc)]
    ids, aid = {r['source_batch_id'] for r in obs}, {r['source_batch_id'] for r in terminal}
    errors = []
    if ids - aid: errors.append('missing_assimilation')
    if aid - ids: errors.append('extra_assimilation')
    for row in terminal:
        if row['status'] not in {'accepted', 'accepted_bootstrap', 'reanchored', 'rejected', 'dropped'}:
            errors.append('unclassifiable_status')
        if row['status'] in ('rejected', 'dropped') and not row['reason']:
            errors.append('refusal_without_reason')
    return errors


def alignment_fixtures():
    base = HERE / 'synthetic_runs'; base.mkdir(exist_ok=True)
    table = [dict(stamp=t+.05, gt_stamp=t, gt_available=1, gt_x=.2*t, gt_y=0., gt_yaw=0.,
                  planner_belief_stamp=t, planner_belief_x=.2*t, planner_belief_y=0.,
                  planner_cov_x=.01, planner_cov_xy=0., planner_cov_y=.01,
                  state_available=1, state_stamp=1., state_x=.2, state_y=0.)
             for t in (0., 1., 1.1, 1.5, 2.)]
    obs = [dict(stamp=1.5, source_batch_id='audit15:batch', camera='A', used=1,
                obs_stamp=1., obs_x=.2, obs_y=0., obs_cov_xx=.03, obs_cov_xy=0., obs_cov_yy=.03,
                fused_stamp=1., fused_x=.2, fused_y=0., fused_cov_xx=.03, fused_cov_xy=0., fused_cov_yy=.03,
                n_candidates=1, n_used=1)]
    row = dict(source_batch_id='audit15:batch', correction_stamp=1., apply_stamp=1.5,
               belief_stamp_after=1., status='accepted', reason='accepted', accepted=1, nis=0.)
    variants = dict(valid=[row], reasoned_drop=[dict(row, status='dropped', reason='replay_gap_too_large', accepted=0)],
                    missing=[], duplicate=[row, row], extra=[row, dict(row, source_batch_id='extra')],
                    unknown=[dict(row, status='mystery')], unreasoned=[dict(row, status='rejected', reason='', accepted=0)])
    out = {}
    for name, events in variants.items():
        run = base / name; run.mkdir(exist_ok=True)
        (run/'run_manifest.json').write_text(json.dumps(dict(logging_schema_version=6, synthetic=True)))
        (run/'run_summary.json').write_text(json.dumps(dict(completed=True, completion_reason='synthetic_end')))
        write_csv(run/'experiment.csv', table); write_csv(run/'fusion_observations.csv', obs)
        write_csv(run/'correction_assimilations.csv', events, list(row))
        output = dict(contract_errors=strict_accounting(run))
        for label, fn in [('assimilation_parser', aligned.assimilations), ('event_belief_helper', aligned.belief_at_fusion_events)]:
            try:
                values = fn(run); output[label] = dict(accepted_input=True, rows=len(values), returned=values)
            except Exception as exc:
                output[label] = dict(accepted_input=False, reason=str(exc))
        out[name] = output
    # No epoch or max interpolation gap: both queries currently produce a finite answer.
    out['reference_gap'] = dict(query=5., value=aligned.TruthSeries([0., 10.], [0., 2.], [0., 0.], [0., 0.], 'synthetic').at([5.]))
    out['epoch_overlap'] = dict(query=1., value=aligned.TruthSeries([0., 1., 0., 1.], [0., .2, 10., 10.2], [0.]*4, [0.]*4, 'synthetic').at([1.]))
    save('alignment.json', out)
    return out


def shutdown_probe():
    # Actual destructor compiled into a fake ROS base; no rclpy shutdown called.
    path = ROOT/'src/experiments/experiments/nodes/experiment_logger.py'
    cls = next(c for c in ast.parse(path.read_text()).body if isinstance(c, ast.ClassDef) and c.name == 'ExperimentLogger')
    method = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == 'destroy_node')
    wrapper = ast.ClassDef(name='LoggerUnderTest', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    class Base:
        def destroy_node(self): self.base_destroyed = True
    env = dict(Base=Base, os=os)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), 'exec'), env)
    result = {}
    class BadClose:
        def close(self): raise OSError('audit15 injected first-handle close failure')
    for fail in (False, True):
        directory = HERE / ('shutdown_failure' if fail else 'shutdown_normal'); directory.mkdir(exist_ok=True)
        n = env['LoggerUnderTest'](); n.run_dir = str(directory); n._completed = False
        handles = {}
        for attr in ('file', 'plan_file', 'perception_file', 'fusion_obs_file', 'assimilation_file', 'camera_opportunity_file'):
            h = (directory/(attr+'.txt')).open('w'); h.write('buffered terminal event\n'); handles[attr] = h; setattr(n, attr, h)
        original = n.file
        if fail: n.file = BadClose()
        error = None
        try: n.destroy_node()
        except Exception as exc: error = repr(exc)
        result[str(fail)] = dict(error=error, closed={k: h.closed for k, h in handles.items()},
            persisted={k: (directory/(k+'.txt')).read_text() for k in handles},
            summary=json.loads((directory/'run_summary.json').read_text()), base_destroyed=n.base_destroyed)
        # Test cleanup after measurement; does not turn a destructor failure into a pass.
        for h in handles.values(): h.close()
    save('shutdown.json', result)
    return result


def selected_ledgers():
    registry = json.loads((ROOT/'docs/localization_metrics_registry.json').read_text())
    selection_path = ROOT / registry['network_navigation_runtime_pilot']['selection']
    selection = json.loads(selection_path.read_text())
    result = dict(selection=str(selection_path.relative_to(ROOT)), selection_sha256=sha(selection_path), runs=[])
    for selected in selection['runs']:
        run = ROOT / selected['run']
        matches = {name: sha(run/name) == digest for name, digest in selected['files'].items()}
        if not all(matches.values()): raise ValueError(f'frozen identity mismatch: {run}')
        table = aligned.rows(run); obs = aligned.observations(run); events = aligned.assimilations(run)
        manifest = json.loads((run/'run_manifest.json').read_text()); summary = json.loads((run/'run_summary.json').read_text())
        result['runs'].append(dict(run=selected['run'], arm=selected['arm'], seed=selected['seed'],
            hashes=matches, schema=aligned.schema_version(run), table_rows=len(table),
            unique_fused=len({x['source_batch_id'] for x in obs}), terminal=len(events),
            statuses=collections.Counter(r['status'] for r in events), accounting_errors=strict_accounting(run),
            outcome=summary.get('completion_reason'), contact_messages_seen=summary.get('contact_messages_seen'),
            contact_topic_publishers=summary.get('contact_topic_publishers'),
            manifest_keys=sorted(manifest), summary_sha256=sha(run/'run_summary.json')))
    save('selected_ledgers.json', result)


def main():
    before = identity(); save('identity_start.json', before)
    failures = {}
    estimator, probes = component_probes()
    for name, fn in [('commands', command_checks), ('event_stream', lambda: event_stream(estimator)),
                     ('alignment', alignment_fixtures), ('shutdown', shutdown_probe), ('selected_ledgers', selected_ledgers)]:
        try: fn()
        except Exception as exc: failures[name] = repr(exc)
    after = identity(); save('identity_end.json', after)
    changes = [f for f, digest in before['source_files'].items() if after['source_files'].get(f) != digest]
    save('execution.json', dict(harness_errors=failures,
        failed_reproductions=[k for k, v in probes.items() if isinstance(v, dict) and v.get('reproduced') is False],
        source_changes_during_harness=changes,
        live_simulator='NOT RUN: component acceptance prerequisites fail',
        semantics='No all-green acceptance implied by a successful reproduction'))
    print(json.dumps(dict(harness_errors=failures, source_changes=changes), indent=2))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
