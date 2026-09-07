"""Manifest-bound ideal tracker and final-stop probes; no simulation processes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import conftest
sys.path.insert(0, str(ROOT / 'experiments/icra_commissioning'))
sys.path.insert(0, str(ROOT / 'experiments/fusion_on_fixed_routes'))
import aligned
import tracker_preflight
from planning.nodes.efe_agent_node import EfeAgentNode
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.tracker_guard import checked_tracker_controls
from planning.core.dynamics import unicycle_step


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(group):
    registry = json.loads((ROOT / 'docs/localization_metrics_registry.json').read_text())
    path = ROOT / registry[group]['selection']
    selection = json.loads(path.read_text())
    entry = next(r for r in selection['runs'] if r['arm'] == 'P0')
    run = ROOT / entry['run']
    for name, sha in entry['files'].items(): assert digest(run / name) == sha
    return path, run


def f(row, key): return float(row[key])


def main(out):
    initial_source = ROOT / 'src/planning/planning/nodes/efe_agent_node.py'
    initial_hash = digest(initial_source)
    original_selection, original = selected('network_navigation_pilot')
    settings_path = ROOT / 'logs/studies/icra_commissioning_20260905/network_planner/full_route_v1/P0_result.json'
    settings = json.loads(settings_path.read_text())['settings']
    meta = json.loads((original / 'global_plan_meta.json').read_text())
    with (original / 'global_plan.csv').open() as stream:
        states = np.array([[float(r[k]) for k in ('x', 'y', 'theta')] for r in csv.DictReader(stream)])
    goal = np.asarray(meta['goal_xy'])
    frozen_source = ROOT / 'logs/studies/icra_commissioning_20260905/network_navigation_tracking_evidence/diagnostics/efe_agent_node_before_runtime_fix.py'
    spec = importlib.util.spec_from_file_location('audit09_frozen_tracker', frozen_source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    results = []
    for label, cls, spacing, arrival in [
        ('original_tracker_original_spacing', frozen.EfeAgentNode, 1., .35),
        ('original_tracker_dense_spacing', frozen.EfeAgentNode, .2, .1),
        ('current_tracker_original_spacing', EfeAgentNode, 1., .35),
        ('current_tracker_dense_spacing', EfeAgentNode, .2, .1),
        ('current_tracker_equal_stop_and_arrival', EfeAgentNode, .2, .05),
    ]:
        tracker_preflight.EfeAgentNode = cls
        record = tracker_preflight.simulate(states, goal, settings, spacing, arrival)
        record['case'] = label
        results.append(record)
        print({k:v for k,v in record.items() if k not in ('trajectory', 'waypoints')}, flush=True)
    tracker_preflight.EfeAgentNode = EfeAgentNode

    runtime_selection, runtime = selected('network_navigation_runtime_pilot')
    summary = json.loads((runtime / 'run_summary.json').read_text())
    rows = [r for r in aligned.rows(runtime) if summary['stop_stamp']-9 <= f(r, 'stamp') <= summary['stop_stamp']
            and f(r, 'cmd_raw_v') == f(r, 'cmd_raw_w') == 0.]
    row = rows[-1]
    pose0 = np.array([f(row, 'planner_belief_' + k) for k in ('x','y','yaw')])
    target = np.array([f(row, 'exec_wp_target_'+k) for k in ('x','y')])
    planner = UnicyclePlannerBase(**dict(settings, camera_network_artifact_path='', use_visibility_model=False,
        use_obs_risk=False, use_ambiguity=False, dt=.25, horizon=12))
    node = NS(planner=planner, dt=.25, local_horizon=12, v_max=.22, w_min=-1., w_max=1., simple_tracker_yaw_gate_rad=.6)
    with (runtime / 'global_waypoints.csv').open() as stream:
        waypoints = np.array([[float(r[k]) for k in ('x','y')] for r in csv.DictReader(stream)])
    with (runtime / 'global_plan.csv').open() as stream:
        global_states = np.array([[float(r[k]) for k in ('x','y','theta')] for r in csv.DictReader(stream)])
    continuations = []
    first = EfeAgentNode._simple_local_plan(node, pose0, target)
    first_gate = EfeAgentNode._simple_plan_safe_to_execute(node, first, pose0)
    for recovery in (False, True):
        pose = pose0.copy()
        index = int(f(row, 'exec_wp_idx'))
        trajectory = [pose.copy().tolist()]
        cmds, reasons = [], []
        status = 'timeout'
        for i in range(120):
            if np.linalg.norm(pose[:2]-waypoints[-1]) <= .35:
                status = 'goal_region'; break
            while index < len(waypoints)-1 and np.linalg.norm(pose[:2]-waypoints[index]) < .1: index += 1
            controls = EfeAgentNode._simple_local_plan(node, pose, waypoints[index])
            decision = checked_tracker_controls(controls, pose, waypoints[index], dt=.25, w_min=-1., w_max=1.,
                allow_rotation_recovery=recovery,
                safety_check=lambda u,m: EfeAgentNode._simple_plan_safe_to_execute(node, u, m))
            reasons.append(decision.reason)
            if decision.safe_steps <= 0:
                status = 'safety_stop'; break
            cmds.append(decision.controls[0].tolist())
            pose = unicycle_step(pose, decision.controls[0], .25)
            trajectory.append(pose.copy().tolist())
        continuations.append(dict(recovery=recovery, status=status, duration_s=len(cmds)*.25,
            target=target.tolist(), commands=cmds, reasons=reasons, trajectory=trajectory,
            final_goal_gap_m=float(np.linalg.norm(pose[:2]-waypoints[-1]))))
    final_stop = dict(run=str(runtime.relative_to(ROOT)), logger_stamp=f(row,'stamp'),
        belief_stamp=f(row,'planner_belief_stamp'), state=pose0.tolist(), target=target.tolist(),
        waypoint_index=int(f(row,'exec_wp_idx')), waypoint_count=len(waypoints),
        proposed_first_command=first[0].tolist(), first_gate=first_gate,
        modeled_body_clearance_m=planner.collision_clearance_state_np(pose0),
        lane_standoff_clearance_m=planner.nogo_cost_model.clearance_state_np(pose0),
        global_route_min_lane_standoff_clearance_m=min(planner.nogo_cost_model.clearance_state_np(s) for s in global_states),
        continuations=continuations,
        limit='Recomputed from final published belief; exact live controller computation snapshot absent. '
              'Goal-region entry only, no success dwell, noise, latency, contacts or physical recovery claim.')
    print({k:v for k,v in final_stop.items() if k != 'continuations'}, flush=True)
    for c in continuations: print({k:v for k,v in c.items() if k not in ('commands','reasons','trajectory')}, flush=True)
    assert digest(initial_source) == initial_hash
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise ValueError('Choose a new audit output')
    sources = [Path(__file__), initial_source, frozen_source, original_selection, runtime_selection,
               settings_path, ROOT/'experiments/icra_commissioning/tracker_preflight.py',
               ROOT/'src/planning/planning/planners/base_planner.py',
               ROOT/'src/planning/planning/core/tracker_guard.py',
               ROOT/'src/planning/planning/core/nogo_cost.py', ROOT/'src/planning/planning/core/dynamics.py']
    out.write_text(json.dumps(dict(kind='ideal_motion_software_replay_not_live_navigation',
        sources={str(p.relative_to(ROOT)):digest(p) for p in sources},
        original_run=str(original.relative_to(ROOT)), corner_results=results, final_stop=final_stop), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out.resolve())
