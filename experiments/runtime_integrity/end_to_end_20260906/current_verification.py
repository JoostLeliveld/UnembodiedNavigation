"""Record and test the later owner-repaired source; never overwrite audit baseline."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TESTS = [
    'tests/perception/test_camera_acquisition_audit_04.py',
    'tests/perception/test_batched_four_camera_yolo.py',
    'tests/perception/test_scheduled_camera_registry.py',
    'tests/reliability/test_source_batch_buffer.py',
    'tests/reliability/test_camera_manager_node.py',
    'tests/reliability/test_fusion_arms.py',
    'tests/reliability/test_fusion_v2.py',
    'tests/reliability/test_fusion_contracts.py',
    'tests/reliability/test_bias_floor.py',
    'tests/planning/test_casadi_cache.py',
    'tests/planning/test_status_traffic.py',
    'tests/planning/test_camera_network.py',
    'tests/planning/test_runtime_transactions.py',
    'tests/planning/test_outage_motion_replay.py',
    'tests/planning/test_tracker_guard.py',
    'tests/sim/test_clock_throttle_traffic.py',
    'tests/sim/test_command_watchdog.py',
    'tests/experiments/test_warehouse_v2_world_contract.py',
    'tests/visibility_comparison/test_network_planner_config.py',
]

def sources():
    names = subprocess.check_output(['rg', '--files', 'src', 'tests'], cwd=ROOT, text=True).splitlines()
    names += ['conftest.py', 'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml']
    return {n: (ROOT / n).read_bytes() for n in sorted(set(names))
            if Path(n).suffix in {'.py', '.yaml', '.sdf', '.xacro', '.xml', '.cfg', '.toml'}}

before = sources()
hashes = {n: hashlib.sha256(b).hexdigest() for n, b in before.items()}
with tarfile.open(HERE / 'later_sources.tar.gz', 'w:gz') as archive:
    for name, payload in before.items():
        entry = tarfile.TarInfo(name)
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
command = ['python3', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TESTS]
started = time.time()
with (HERE / 'later_component_tests.txt').open('w') as output:
    result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
after = {n: hashlib.sha256(b).hexdigest() for n, b in sources().items()}
changed = sorted(n for n in set(hashes) | set(after) if hashes.get(n) != after.get(n))
record = {'command': command, 'start_wall_unix': started, 'finish_wall_unix': time.time(),
          'returncode': result.returncode, 'sources_before': hashes, 'sources_after': after,
          'changed_during_test': changed,
          'scope': 'Owner-repaired current source component regressions, not end-to-end acceptance.'}
(HERE / 'later_verification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if not k.startswith('sources_')}, indent=2))
raise SystemExit(result.returncode)
