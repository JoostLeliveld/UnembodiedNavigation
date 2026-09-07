"""Recheck request/route acceptance after repair packages A and B.

Runs only controlled in-process fixtures. No ROS graph or experiment writes.
Extracts fixture definitions from the preserved audit probe, not its assertions.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import conftest
sys.path[:0] = [str(ROOT / 'tests/planning')]
import numpy as np
from geometry_msgs.msg import PoseStamped
from planning.nodes.efe_agent_node import EfeAgentNode
from planning.planners.base_planner import UnicyclePlannerBase
from test_runtime_transactions import command_node
from test_planner_node_correction_wiring import _Clock, _Logger, stamp

AUDIT = Path(__file__).parent
source = ast.parse((AUDIT / '08_planner_probe.py').read_text())
definitions = [item for item in source.body if isinstance(item, ast.FunctionDef)
               and item.name in ('node', 'planner')]
exec(compile(ast.Module(body=definitions, type_ignores=[]), '08_fixture_definitions', 'exec'))
baseline_node = node

def node(global_mode=False):
    value = baseline_node(global_mode)
    value._command_stop_generation = 0
    value._active_plan_request = None
    value.dt = .25
    value._active_controls = None
    value._active_plan_started_at = None
    return value

digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = list(json.loads((AUDIT / '08_source_snapshot/manifest.json').read_text())['sources'])
files += ['src/planning/planning/core/casadi_cache.py',
          'src/planning/planning/core/motion_history.py',
          'src/planning/planning/core/tracker_guard.py',
          'tests/planning/test_command_installation.py',
          'tests/planning/test_casadi_cache.py',
          'tests/planning/test_runtime_transactions.py',
          'tests/planning/test_tracker_guard.py',
          'tests/planning/test_planner_node_correction_wiring.py']
hashes = {p: digest(ROOT / p) for p in files}
OUT = dict(scope='current source, controlled route/result boundaries; no navigation metrics',
           head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           sources=hashes, cases={})
C = OUT['cases']

tiny = planner(horizon=2, dt=.25, optimizer_terminal_goal_tolerance_m=.35, optimizer_maxiter=1)
partial = tiny.plan(np.zeros(3), np.eye(3)*.01, [2.,0.])
assert partial.rollout_valid and partial.terminal_goal_distance_pred > .35
for name, result in dict(incomplete=partial,
        invalid=NS(states=np.array([[0.,0.,0.],[.1,0.,0.]]), rollout_valid=False),
        nonfinite=NS(states=np.full((2,3),math.nan), rollout_valid=False)).items():
    n = node(True)
    n.global_planner = NS(plan=lambda *a, **kw: result)
    n._plan_once()
    assert n._hier_phase == 'LOCAL'
    C['global_'+name] = dict(phase=n._hier_phase, waypoints=n._waypoints,
        returned_terminal_gap_m=getattr(result,'terminal_goal_distance_pred',None),
        request_context_cleared=n._active_plan_request is None)

for mode in (True, False):
    for change in ('none','belief','goal','config','clock','backward_clock','stop'):
        n = node(mode)
        entered, release = threading.Event(), threading.Event()
        errors, origin = [], {}
        def slow(m0, S0, goal, **kw):
            origin.update(state=m0.tolist(), goal=list(goal),
                          context_keys=sorted(n._active_plan_request))
            entered.set()
            if not release.wait(5.): raise RuntimeError('test barrier timeout')
            return NS(states=np.array([[0.,0.,0.],[.1,0.,0.]]),
                      controls=np.array([[.2,0.]]), rollout_valid=True,
                      min_predicted_obstacle_distance_m=1., terminal_goal_distance_pred=1.9)
        if mode: n.global_planner = NS(plan=slow)
        else: n.planner.plan = slow
        def run():
            try: n._plan_once()
            except BaseException as error: errors.append(repr(error))
        thread = threading.Thread(target=run)
        thread.start()
        assert entered.wait(5.), errors
        if change == 'belief':
            with n._data_lock:
                n.belief_m=np.array([1.,2.,1.]); n.belief_S=np.eye(3); n.belief_stamp=stamp(10.)
        elif change == 'goal':
            goal = PoseStamped(); goal.pose.position.x=-2.; n._goal_cb(goal)
        elif change == 'config': n.w_max=.1
        elif change == 'clock': n._clock.seconds=30.
        elif change == 'backward_clock': n._clock.seconds=5.
        elif change == 'stop': n._publish_safe_stop_command()
        release.set(); thread.join(5.)
        assert not thread.is_alive() and not errors, errors
        commands = [[v.linear.x, v.angular.z] for v in n.cmd_pub.messages]
        published_nonzero = any(v != 0 or w != 0 for v,w in commands)
        if mode: assert n._hier_phase == 'LOCAL'
        else: assert published_nonzero == (change in ('none','belief','goal','config'))
        C[('global' if mode else 'direct')+'_slow_'+change] = dict(
            origin=origin, route_installed=n._hier_phase=='LOCAL' if mode else None,
            controls_installed=n._active_controls is not None,
            commands=commands, published_nonzero=published_nonzero,
            context_cleared=n._active_plan_request is None,
            current_goal_x=n.goal_msg.pose.position.x,
            current_state=n.belief_m.tolist(), stop_generation=n._command_stop_generation)

variants = dict(missing_fields=NS(controls=np.array([[.2,0.]])),
    invalid_rollout=NS(controls=np.array([[.2,0.]]),rollout_valid=False),
    nonfinite_controls=NS(controls=np.array([[math.nan,0.]])),
    out_of_bounds=NS(controls=np.array([[2.,3.]])),
    nonfinite_cost=NS(controls=np.array([[.2,0.]]),total_cost=math.nan),
    nonfinite_states=NS(controls=np.array([[.2,0.]]),states=np.full((2,3),math.nan)),
    nonfinite_clearance=NS(controls=np.array([[.2,0.]]),min_predicted_obstacle_distance_m=-math.inf))
decisions={}
for name, result in variants.items():
    n=node(); n._active_plan_request=n._capture_plan_request()
    safe, reason = n._result_safe_to_execute(result)
    n._after_plan_result(result)
    decisions[name]=dict(gate_safe=safe, reason=reason,
        controls_installed=n._active_controls is not None,
        commands=[[v.linear.x,v.angular.z] for v in n.cmd_pub.messages])
    assert (n._active_controls is not None) == (name not in ('invalid_rollout','nonfinite_controls'))
C['direct_numerical_contract_with_valid_context']=decisions
n=node(); n._after_plan_result(variants['missing_fields'])
assert n._active_controls is None
C['missing_request_context_rejected']=dict(commands=[[v.linear.x,v.angular.z] for v in n.cmd_pub.messages])

OUT['source_changes_during_probe']={p:dict(before=h, after=digest(ROOT/p)) for p,h in hashes.items() if digest(ROOT/p)!=h}
OUT['probe_sha256']=digest(__file__)
(AUDIT/'08_current_result_checks.json').write_text(json.dumps(OUT,indent=2)+'\n')
print(json.dumps(dict(groups=len(C),head=OUT['head'],source_changes=OUT['source_changes_during_probe'],
    incomplete_gap_m=partial.terminal_goal_distance_pred,
    global_still_installs_after=['belief','goal','config','clock','backward_clock','stop'],
    direct_still_installs_after=['belief','goal','config'],
    direct_refuses_after=['clock','backward_clock','stop'],
    missing_context_refused=True),indent=2))
