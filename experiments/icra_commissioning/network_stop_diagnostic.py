#!/usr/bin/env python3
"""Frozen local stop diagnosis; a recovery probe is not a closed-loop result."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import network_navigation_analysis as nav
from planning.nodes.efe_agent_node import EfeAgentNode
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.dynamics import unicycle_step
from planning.core.tracker_guard import checked_tracker_controls


def rotation_candidate(node, state, target):
    """Proposed recovery: rotate toward the same target without translation."""
    controls = np.zeros((node.local_horizon, 2))
    pose = state.copy()
    for i in range(node.local_horizon):
        desired = np.arctan2(target[1]-pose[1], target[0]-pose[0])
        delta = np.arctan2(np.sin(desired-pose[2]), np.cos(desired-pose[2]))
        controls[i, 1] = np.clip(2.*delta, node.w_min, node.w_max)
        pose = unicycle_step(pose, controls[i], node.dt)
    return controls


def main(selection, out):
    entry = json.loads(selection.read_text())
    run = nav.REPO/entry['run']
    for name, expected in entry['files'].items():
        if nav.digest(run/name) != expected:
            raise ValueError(f'Changed selected input: {name}')
    table = nav.aligned.rows(run)
    # Final published belief in the terminal stop window, selected without GT.
    summary = json.loads((run/'run_summary.json').read_text())
    stop = float(summary['stop_stamp'])
    rows = [r for r in table if stop-9. <= nav.f(r, 'stamp') <= stop
            and nav.f(r, 'cmd_raw_v') == nav.f(r, 'cmd_raw_w') == 0.]
    if not rows:
        raise ValueError('No recorded terminal stop to diagnose')
    row = rows[-1]
    state = np.array([nav.f(row, 'planner_belief_'+k) for k in ('x', 'y', 'yaw')])
    target = np.array([nav.f(row, 'exec_wp_target_'+k) for k in ('x', 'y')])
    settings_path = nav.PROBE/'P0_result.json'
    settings = json.loads(settings_path.read_text())['settings']
    planner = UnicyclePlannerBase(**dict(settings, camera_network_artifact_path='',
        use_visibility_model=False, use_obs_risk=False, use_ambiguity=False, dt=.25, horizon=12))
    node = SimpleNamespace(planner=planner, dt=.25, local_horizon=12, v_max=.22,
        w_min=-1., w_max=1., simple_tracker_yaw_gate_rad=.6)
    baseline = EfeAgentNode._simple_local_plan(node, state, target)
    baseline_gate = EfeAgentNode._simple_plan_safe_to_execute(node, baseline, state)
    rotate = rotation_candidate(node, state, target)
    rotate_gate = EfeAgentNode._simple_plan_safe_to_execute(node, rotate, state)
    with (run/'global_plan.csv').open() as stream:
        route = np.array([[float(r[k]) for k in ('x','y','theta')] for r in csv.DictReader(stream)])
    with (run/'global_waypoints.csv').open() as stream:
        waypoints = np.array([[float(r[k]) for k in ('x','y')] for r in csv.DictReader(stream)])
    continuations=[]
    for recovery in (False,True):
        pose=state.copy();index=int(nav.f(row,'exec_wp_idx'));trajectory=[pose.copy()]
        reason='timeout';turns=0
        for _ in range(120):
            if np.linalg.norm(pose[:2]-waypoints[-1]) <= .35:
                reason='goal_region';break
            while index<len(waypoints)-1 and np.linalg.norm(pose[:2]-waypoints[index])<.1:
                index+=1
            proposed=EfeAgentNode._simple_local_plan(node,pose,waypoints[index])
            decision=checked_tracker_controls(proposed,pose,waypoints[index],dt=node.dt,
                w_min=node.w_min,w_max=node.w_max,allow_rotation_recovery=recovery,
                safety_check=lambda u,m:EfeAgentNode._simple_plan_safe_to_execute(node,u,m))
            if decision.safe_steps<=0:
                reason=decision.reason;break
            turns+=int(decision.rotation_recovery)
            pose=unicycle_step(pose,decision.controls[0],node.dt);trajectory.append(pose.copy())
        continuations.append(dict(rotation_recovery_enabled=recovery,outcome=reason,
            duration_s=(len(trajectory)-1)*node.dt,rotation_actions=turns,
            goal_gap_m=float(np.linalg.norm(pose[:2]-waypoints[-1])),trajectory=np.array(trajectory).tolist()))
    result = dict(kind='single_stop_geometry_and_control_diagnostic', run=entry['run'],
        selection_rule='final published belief with zero command in the last nine seconds',
        log_stamp=nav.f(row, 'stamp'), belief_stamp=nav.f(row, 'planner_belief_stamp'),
        belief=state.tolist(), target=target.tolist(),
        baseline_first_control=baseline[0].tolist(), baseline_safe_steps=baseline_gate[0],
        baseline_gate_reason=baseline_gate[1],
        proposed_rotation_first_control=rotate[0].tolist(), proposed_rotation_safe_steps=rotate_gate[0],
        physical_clearance_m=planner.collision_clearance_state_np(state),
        lane_standoff_clearance_m=planner.nogo_cost_model.clearance_state_np(state),
        route_min_lane_standoff_m=min(planner.nogo_cost_model.clearance_state_np(s) for s in route),
        ideal_continuations=continuations,
        scope='Recomputes the controller and gate from a logged published belief. The exact planner snapshot is not logged. '
              'The rotation probe and ideal-motion continuation use the unchanged gate. Neither was executed live; '
              'they do not establish navigation recovery with camera errors or actuation noise.',
        sources={str(p.relative_to(nav.REPO)):nav.digest(p) for p in [selection, settings_path, Path(__file__),
            nav.REPO/'src/planning/planning/nodes/efe_agent_node.py',
            nav.REPO/'src/planning/planning/core/tracker_guard.py',
            nav.REPO/'src/planning/planning/core/nogo_cost.py']})
    out.mkdir(parents=True, exist_ok=True)
    nav.writejson(out/'stop_diagnostic.json', result)
    nav.style()
    fig, ax = nav.plt.subplots(figsize=(8.0, 4.5), layout='constrained')
    nav.map_background(ax, settings)
    xx, yy = np.meshgrid(np.linspace(9.3, 10.9, 100), np.linspace(6.25, 6.85, 70))
    clearance = np.array([planner.nogo_cost_model.clearance_state_np([x,y])
                          for x,y in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    ax.contourf(xx, yy, clearance, levels=[-2., 0., 2.], colors=['#fff0d8','#e2f1e8'], alpha=.9)
    ax.contour(xx, yy, clearance, levels=[0.], colors=['#ab6b22'], linewidths=1.)
    ax.plot(route[:,0], route[:,1], ':', color='#7e8991', label='Global planned mean')
    ax.plot(*state[:2], 'o', color='#207b70', label='Published belief at terminal stop')
    ax.plot(*target, '*', color='#31587e', ms=11, label='Current waypoint')
    next_state = unicycle_step(state, baseline[0], node.dt)
    ax.annotate('', xy=next_state[:2], xytext=state[:2],
                arrowprops=dict(arrowstyle='->',color='#b74837',lw=2.))
    ax.plot([],[],color='#b74837',label='Requested first motion: rejected')
    ax.annotate('55 cm lane-margin boundary', (9.4,6.55), (9.4,6.76), fontsize=9,
                arrowprops=dict(arrowstyle='-', color='#ab6b22'))
    ax.set(xlim=(9.3,10.9),ylim=(6.25,6.85),xlabel='World x [m]',ylabel='World y [m]')
    ax.legend(fontsize=8,loc='lower right',framealpha=.95)
    ax.set_title('A soft global preference becomes a hard local refusal\n'
                 'Same gate permits rotation in place; live recovery remains untested',fontsize=11)
    nav.savefig(fig,out/'stop_diagnostic')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.selection.resolve(),a.out.resolve())
