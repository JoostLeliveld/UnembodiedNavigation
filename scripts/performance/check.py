#!/usr/bin/env python3
"""Fast development checks, with optional verification of registered recorded events."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    'performance': ['tests/planning/test_casadi_cache.py', 'tests/planning/test_status_traffic.py',
                    'tests/sim/test_clock_throttle_traffic.py', 'tests/experiments/test_goal_mission_traffic.py',
                    'tests/experiments/test_low_cpu_world.py', 'tests/visibility_comparison/test_campaign_ledger.py',
                    'tests/visibility_comparison/test_network_planner_config.py'],
    'perception': ['tests/perception/test_batched_four_camera_yolo.py', 'tests/visibility_comparison/test_yolo_selection.py'],
    'filter': ['tests/planning/test_belief_correction.py', 'tests/planning/test_runtime_transactions.py',
               'tests/planning/test_outage_motion_replay.py', 'tests/sim/test_command_watchdog.py'],
    'planner': ['tests/planning/test_efe_hit_miss_mixture.py', 'tests/planning/test_camera_network.py',
                'tests/planning/test_preselected_route.py'],
    'logging': ['tests/experiments/test_camera_opportunity_log.py', 'tests/visibility_comparison/test_campaign_ledger.py'],
}


def verify_selection(selection):
    """Check exact registered drives via the repository's alignment loader.

    This is a fast event/identity check; it is not an alternative navigation
    result, controller replay or a new selection of successful drives.
    """
    registry = json.loads((ROOT / 'docs/localization_metrics_registry.json').read_text())
    def registered_paths(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from registered_paths(child)
        elif isinstance(value, list):
            for child in value:
                yield from registered_paths(child)
        elif isinstance(value, str) and value.endswith('.json'):
            yield (ROOT / value).resolve()
    selection = selection.resolve()
    if selection not in set(registered_paths(registry)):
        raise ValueError('selection must be explicitly named in the metrics registry')
    selected = json.loads(selection.read_text())
    sys.path.insert(0, str(ROOT))
    from experiments.fusion_on_fixed_routes import aligned
    entries = selected['runs']
    if not entries or len({entry['run'] for entry in entries}) != len(entries):
        raise ValueError('empty or duplicate run selection')
    required = {'experiment.csv', 'run_manifest.json', 'run_summary.json',
                'fusion_observations.csv', 'correction_assimilations.csv'}
    reports = []
    for entry in entries:
        run = ROOT / entry['run']
        if not required.issubset(entry['files']):
            raise ValueError(f'missing frozen evidence hashes: {run}')
        for name, expected in entry['files'].items():
            if hashlib.sha256((run/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'changed evidence: {run/name}')
        table = aligned.rows(run)
        observations = aligned.observations(run)
        assimilations = aligned.assimilations(run)
        accounted = [r['source_batch_id'] for r in assimilations]
        if len(accounted) != len(set(accounted)):
            raise ValueError(f'duplicate assimilation: {run}')
        # The campaign's canonical verifier also covers explicit refusal/status
        # semantics; reuse it instead of introducing a second validity rule.
        import importlib.util
        spec = importlib.util.spec_from_file_location('performance_campaign', ROOT/'scripts/visibility_comparison/run_visibility_campaign.py')
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        valid, reason = campaign._verify_correction_assimilations(run)
        if not valid:
            raise ValueError(f'{run}: {reason}')
        reports.append(dict(run=entry['run'], logger_rows=len(table),
                            observation_rows=len(observations), assimilations=len(assimilations),
                            hash_and_event_checks='passed'))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=[*SUITES, 'all'], default='all')
    parser.add_argument('--selection', type=Path, help='Optional exact registered selection to validate after tests.')
    args = parser.parse_args()
    paths = SUITES[args.suite] if args.suite != 'all' else list(dict.fromkeys(p for group in SUITES.values() for p in group))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', UNAV_CASADI_CACHE_DIR='', UNAV_CASADI_JIT='0')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *paths], cwd=ROOT, env=env)
    if result.returncode:
        return result.returncode
    if args.selection:
        print(json.dumps(verify_selection(args.selection), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
