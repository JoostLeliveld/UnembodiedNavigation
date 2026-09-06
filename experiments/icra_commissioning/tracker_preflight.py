#!/usr/bin/env python3
"""Replay the real waypoint tracker on recorded global paths with ideal motion.

This separates path-to-controller feasibility from perception. No simulated
measurements, reference feedback, or navigation performance claim are involved.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/icra_mpl')
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/p) for p in ('src/planning', 'src/unav_common', 'src/reliability')]
from study import OUT, digest, writejson
from planning.nodes.efe_agent_node import EfeAgentNode
from planning.planners.base_planner import UnicyclePlannerBase, extract_waypoints
from planning.core.dynamics import unicycle_step


def simulate(states, goal, settings, spacing, arrival):
    waypoints = extract_waypoints(states, spacing_m=spacing, include_goal=True)
    if np.linalg.norm(np.asarray(waypoints[-1])-goal) > .001:
        waypoints.append(tuple(goal))
    local = dict(settings, camera_network_artifact_path='', use_visibility_model=False,
                 use_obs_risk=False, use_ambiguity=False, dt=.25, horizon=12)
    planner = UnicyclePlannerBase(**local)
    node = SimpleNamespace(local_horizon=12, dt=.25, v_max=.22, w_min=-1., w_max=1.,
                           simple_tracker_yaw_gate_rad=.6, planner=planner)
    state = states[0].copy(); index = 0; trajectory = [state.copy()]
    status = 'timeout'; reason = ''; controls_applied = []
    for step in range(2400):
        target = np.asarray(waypoints[index])
        while index < len(waypoints)-1 and np.linalg.norm(state[:2]-target) < arrival:
            index += 1; target = np.asarray(waypoints[index])
        if np.linalg.norm(state[:2]-goal) <= .35:
            status = 'goal_region'; break
        controls = EfeAgentNode._simple_local_plan(node, state, target)
        count, reason = EfeAgentNode._simple_plan_safe_to_execute(node, controls, state)
        if count == 0:
            status = 'safety_stop'; break
        state = unicycle_step(state, controls[0], .25)
        controls_applied.append(controls[0]); trajectory.append(state.copy())
    trajectory = np.asarray(trajectory)
    clearance = min(planner.collision_clearance_state_np(s) for s in trajectory)
    return dict(spacing_m=spacing, arrival_radius_m=arrival, status=status, stop_reason=reason,
        duration_s=.25*len(controls_applied), waypoint_index=index, waypoints=waypoints,
        goal_gap_m=float(np.linalg.norm(state[:2]-goal)),
        min_physical_clearance_m=float(clearance), trajectory=trajectory.tolist())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); run=args.run.resolve(); out=args.out.resolve()
    if out.exists(): raise ValueError('Choose a new diagnostic output')
    probe = OUT/'network_planner/full_route_v1/P0_result.json'
    settings = json.loads(probe.read_text())['settings']
    meta=json.loads((run/'global_plan_meta.json').read_text())
    with (run/'global_plan.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    print('path fields',list(rows[0]),flush=True)
    states=np.array([[float(r[k]) for k in ('x','y','theta')] for r in rows])
    results=[]
    for spacing,arrival in [(1.,.35),(.2,.1),(.2,.05)]:
        result=simulate(states,np.asarray(meta['goal_xy']),settings,spacing,arrival)
        results.append(result)
        print({k:v for k,v in result.items() if k not in ('trajectory','waypoints')},flush=True)
    writejson(out,dict(kind='ideal_motion_tracker_diagnostic_not_sensor_or_navigation_evaluation',
        run=str(run.relative_to(REPO)),results=results,
        sources={str(p.relative_to(REPO)):digest(p) for p in [Path(__file__),probe,
            run/'global_plan.csv',run/'global_plan_meta.json',
            REPO/'src/planning/planning/nodes/efe_agent_node.py']}))
