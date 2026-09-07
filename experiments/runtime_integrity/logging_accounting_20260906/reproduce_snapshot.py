"""Reproduce audit 10's pinned evidence without changing shared runtime sources.

The checkpoint matches every source hash recorded by the original probe. Extract
only dependencies into a temporary checkout, link frozen logs for reading, rerun
the probe/tests and compare results exactly. Baseline evidence is never replaced.
"""
from pathlib import Path
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SNAPSHOT = '3c4ddeae4a7427bf374514dad6a1d65dc728a91c'
PATHS = ['src', 'tests', 'conftest.py', 'pyproject.toml', 'config', 'schemas',
         'experiments/fusion_on_fixed_routes', 'scripts/visibility_comparison',
         'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml',
         'docs/localization_metrics_registry.json']
SUITES = ['tests/experiments/test_logger_schema.py',
          'tests/experiments/test_logger_time_alignment.py',
          'tests/experiments/test_camera_opportunity_log.py',
          'tests/experiments/test_fusion_study_alignment.py',
          'tests/planning/test_planner_node_state_correction.py',
          'tests/planning/test_runtime_transactions.py',
          'tests/perception/test_camera_acquisition_audit_04.py']
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
result = {'snapshot': SNAPSHOT, 'scope': 'Original audit source; later repairs are separate'}
archive = subprocess.check_output(['git', 'archive', SNAPSHOT, *PATHS], cwd=ROOT)
with tempfile.TemporaryDirectory(prefix='audit10-snapshot-') as temp:
    checkout = Path(temp)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(checkout, filter='data')
    (checkout/'logs').symlink_to(ROOT/'logs', target_is_directory=True)
    target = checkout/OUT.relative_to(ROOT)
    target.mkdir(parents=True)
    for name in ['probe.py', 'source_manifest.json']:
        shutil.copy2(OUT/name, target/name)
    run = subprocess.run([sys.executable, str(target/'probe.py')], cwd=checkout,
                         env=env, text=True, capture_output=True)
    result['probe_returncode'] = run.returncode
    if run.returncode:
        result['failure_output'] = run.stdout + run.stderr
    else:
        result['matching_outputs'] = {}
        for name in ['results.json', 'selected_run_accounting.json', 'source_manifest.json']:
            expected = json.loads((OUT/name).read_text())
            got = json.loads((target/name).read_text())
            assert got == expected, f'{name} differs from retained evidence'
            result['matching_outputs'][name] = True
        tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', *SUITES],
                               cwd=checkout, env=env, text=True, capture_output=True)
        result['regressions'] = {'returncode': tests.returncode,
                                 'output': tests.stdout + tests.stderr}
        run = tests
print(json.dumps(result, indent=2))
raise SystemExit(run.returncode)
