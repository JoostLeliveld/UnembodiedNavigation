"""Summarize timing for exact completed 1 m/s diagnostic selections."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import os
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
O = Path(os.environ.get('BAYESIAN_NAVIGATION_ROOT', REPO / 'logs/bayesian_navigation'))
version = sys.argv[1] if len(sys.argv) > 1 else 'v2'
assert version in ('v1', 'v2')
T = O / ('visibility_speed_1mps_reporting' if version == 'v1' else 'visibility_speed_1mps_v2_reporting')

def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                max=float(a.max()), sum=float(a.sum())) if len(a) else None

results = []
for cfg in json.loads((T / 'selection.json').read_text())['runs']:
    D = O / 'navigation' / cfg['id']
    summary = json.loads((D / 'summary.json').read_text())
    supervisor = json.loads((D / 'supervisor.json').read_text())
    runtime = [json.loads(line) for line in (D / 'runtime.jsonl').open()]
    start = next(r for r in runtime if r['kind'] == 'navigation_start')
    lo, hi = start['stamp_ns'] / 1e9, summary['end_stamp_ns'] / 1e9
    perf = [r for r in csv.DictReader((D / 'performance.csv').open()) if r['sim_s']]
    sw, ew = np.interp([lo, hi], [float(r['sim_s']) for r in perf],
                       [float(r['wall_elapsed_s']) for r in perf])
    active = [r for r in perf if lo <= float(r['sim_s']) <= hi]
    plans = [r for r in runtime if r['kind'] == 'plan']
    commands = [r for r in runtime if r['kind'] == 'command']
    durations = collections.Counter()
    for a, b in zip(commands, commands[1:] + [{'receive_stamp_ns': int(hi * 1e9)}]):
        dt = min(hi, b['receive_stamp_ns'] / 1e9) - max(lo, a['receive_stamp_ns'] / 1e9)
        if dt > 0:
            durations[a['reason']] += dt
    resources = [json.loads(line) for line in (D / 'resources.jsonl').open()]
    end_wall = next(r['wall_monotonic_s'] for r in reversed(runtime)
                    if r['kind'] == 'belief' and r['stamp_ns'] <= summary['end_stamp_ns'])
    resources = [r for r in resources if start['wall_monotonic_s'] <= r['monotonic_s'] <= end_wall]
    gpu = []
    memory = []
    process_cpu = collections.defaultdict(list)
    for r in resources:
        try:
            fields = r['gpu'].split(',')
            gpu.append(float(fields[0])); memory.append(float(fields[2]))
        except (KeyError, ValueError, AttributeError):
            pass
        by_kind = collections.Counter()
        for p in r['processes']:
            cmd = p['cmd']
            kind = next((label for token, label in [
                ('planner_worker.py', 'planner'), ('navigation_driver.py', 'driver'),
                ('parameter_bridge', 'bridges'), ('batched_four_camera_yolo', 'detector'),
                ('ign gazebo', 'gazebo')] if token in cmd), None)
            if kind:
                by_kind[kind] += p['cpu_pct']
        for kind, value in by_kind.items():
            process_cpu[kind].append(value)
    results.append(dict(id=cfg['id'], arm=cfg['arm'],
        timing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sim_s=summary['sim_duration_s'], driver_wall_s=summary['wall_duration_s'],
        supervisor_wall_s=supervisor['wall_s'], navigation_wall_interpolated_s=float(ew-sw),
        effective_navigation_rtf=float((hi-lo)/(ew-sw)),
        summary_detected_to_cleanup_s=supervisor['begin_monotonic']+supervisor['wall_s']-supervisor['summary_detected_monotonic'],
        host_cpu_pct=stats([float(r['host_cpu_pct']) for r in active]),
        gpu_util_pct=stats(gpu), gpu_memory_mib=stats(memory),
        owned_process_cpu_core_pct={k: stats(v) for k, v in process_cpu.items()},
        local_solve_wall_s=stats([r['solve_wall_s'] for r in plans]),
        invalid_plan_reasons=dict(collections.Counter(r['invalid_reason'] for r in plans if not r['valid'])),
        command_reason_sim_s=dict(durations),
        command_interval_sim_s=stats(np.diff([r['receive_stamp_ns']/1e9 for r in commands]))))
(T / 'timing.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps([{k:r[k] for k in ('arm','sim_s','supervisor_wall_s','effective_navigation_rtf','command_reason_sim_s')} for r in results], indent=2))
