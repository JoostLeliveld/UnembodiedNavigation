"""Read the separately registered recovery follow-up; never select by recency."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('audit09_trace', Path(__file__).with_name('09_run_trace.py'))
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def main(out):
    registry_path = ROOT / 'docs/localization_metrics_registry.json'
    registry = json.loads(registry_path.read_text())
    group = registry['network_navigation_recovery_pilot']
    selection_path = ROOT / group['selection']
    selection = json.loads(selection_path.read_text())
    assert len(selection['runs']) == 1
    entry = selection['runs'][0]
    assert entry['arm'] == 'P0' and entry['seed'] == 210 and not entry['missing']
    run = ROOT / entry['run']
    for name, expected in entry['files'].items():
        assert trace.digest(run / name) == expected, name
    summary = json.loads((run / 'run_summary.json').read_text())
    rows = trace.aligned.rows(run)
    eligible = [row for row in rows if trace.f(row, 'stamp') <= summary['stop_stamp']]
    payload = dict(
        kind='separate_registered_recovery_followup_operational_terminal_trace',
        source_sha256={str(p.relative_to(ROOT)): trace.digest(p) for p in
                       (Path(__file__), Path(trace.__file__), registry_path, selection_path)},
        selected_files_sha256=entry['files'], run=entry['run'],
        registry_status=group['status'], summary=summary,
        last_logged_at_or_before_decision=trace.snapshot(eligible[-1]),
        rows_logged_after_stop=sum(trace.f(row, 'stamp') > summary['stop_stamp'] for row in rows),
        limit='Publication snapshot, not applied velocity or verified rest. No accuracy or causal comparison computed.',
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: payload[k] for k in ('run', 'registry_status', 'last_logged_at_or_before_decision')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
