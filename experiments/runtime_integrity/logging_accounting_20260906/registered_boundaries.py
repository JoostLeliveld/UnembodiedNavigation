"""Additional exact registered selections discovered during final audit QA.

Read-only on selected evidence. No accuracy statistics or cross-pilot comparison.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
spec = importlib.util.spec_from_file_location('audit10_probe', OUT/'probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
expected = json.loads((OUT/'source_manifest.json').read_text())
aligned_path = 'experiments/fusion_on_fixed_routes/aligned.py'
assert hashlib.sha256((ROOT/aligned_path).read_bytes()).hexdigest() == expected[aligned_path]


def selected(name):
    path = ROOT/f'logs/studies/icra_commissioning_20260905/network_navigation_{name}_evidence/selection.json'
    content = json.loads(path.read_text())
    identity = dict(selection=str(path.relative_to(ROOT)),
                    selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return identity, content['runs']


def verify(entry):
    run = ROOT/entry['run']
    verified = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
                for name, digest in entry['files'].items()}
    assert all(verified.values()), verified
    return run, verified, json.loads((run/'run_summary.json').read_text())


recovery, entries = selected('recovery')
recovery.update(scope='Separate recovery follow-up; raw event accounting only', runs=[])
for entry in entries:
    run, verified, summary = verify(entry)
    result = probe.reconstruct(run)
    outcomes = probe.aligned.assimilations(run)
    first = summary['first_cmd_stamp']
    result.update(run=entry['run'], verified_files=verified,
                  logging_schema=probe.aligned.schema_version(run),
                  contact_messages=summary['contact_messages_seen'],
                  contact_publishers=summary['contact_topic_publishers'],
                  stop_stamp=summary['stop_stamp'], first_cmd_stamp=first,
                  applied_after_first_command=dict(Counter(r['status'] for r in outcomes
                                                           if r['apply_stamp'] >= first)))
    recovery['runs'].append(result)
probe.save(OUT/'recovery_run_accounting.json', recovery)

tracking, entries = selected('tracking')
entry = next(e for e in entries if e['run'].endswith('P1/seed210/experiment_20260906_210032'))
run, verified, summary = verify(entry)
rows = probe.aligned.rows(run)
raw = probe.read_csv(run/'experiment.csv')
tracking.update(run=entry['run'], verified_files=verified,
                scope='Registered prior tracking pilot; contact/summary accounting only',
                logging_schema=probe.aligned.schema_version(run),
                summary={k: summary[k] for k in ['completed', 'completion_reason', 'stop_stamp',
                                                 'first_crash_stamp', 'contact_messages_seen',
                                                 'collision_contact']},
                final_csv_row_number=len(raw)+1,
                final_csv_row={k: raw[-1][k] for k in ['stamp', 'contact_messages_seen',
                                                     'collision_contact', 'cmd_v', 'cmd_w']},
                rows_after_summary_stop=sum(float(r['stamp']) > summary['stop_stamp'] for r in rows))
assert summary['contact_messages_seen'] == 1
assert raw[-1]['contact_messages_seen'] == '17'
assert float(raw[-1]['stamp']) > summary['stop_stamp']
probe.save(OUT/'tracking_contact_tail.json', tracking)
print(json.dumps({'recovery_terminal_rows': [r['assimilation_rows'] for r in recovery['runs']],
                  'recovery_post_first_command_statuses': [r['applied_after_first_command'] for r in recovery['runs']],
                  'tracking_contact_tail': tracking}, indent=2))
