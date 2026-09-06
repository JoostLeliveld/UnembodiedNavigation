"""Deterministic audit of real command methods; no ROS graph or simulation processes.
Assertions pin observed behaviour, including defects, not a desired repair contract.
Run: OPENBLAS_NUM_THREADS=1 python3 experiments/command_execution_audit_20260906/probe.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import math
import textwrap
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import conftest  # repository package paths
sys.path.insert(0, str(ROOT / 'tests/planning'))
FROZEN = '--frozen-baseline' in sys.argv
BASELINE = ROOT / 'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/source_snapshot'
def source_path(name):
    frozen = BASELINE / name
    return frozen if FROZEN and frozen.exists() else ROOT / name

if FROZEN:
    # Other audited runtime modules still match the baseline; load the EFE
    # snapshot explicitly so later coordinated repairs do not erase reproductions.
    import planning.nodes
    name = 'planning.nodes.efe_agent_node'
    spec = importlib.util.spec_from_file_location(name, source_path('src/planning/planning/nodes/efe_agent_node.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from planning.nodes.efe_agent_node import EfeAgentNode
from sim.actuation_noise_node import ActuationNoiseNode
from test_runtime_transactions import command_node
from test_planner_node_correction_wiring import _Clock, _Logger, stamp

OUT = {}
SOURCES = ['src/planning/planning/nodes/efe_agent_node.py',
           'src/planning/planning/nodes/unicycle_planner_node.py',
           'src/sim/sim/actuation_noise_node.py',
           'src/sim/robot_description/urdf/warehouse_amr.urdf.xacro',
           'src/sim/launch/bringup_sim.launch.py',
           'src/experiments/experiments/nodes/experiment_logger.py',
           'src/experiments/experiments/core/visibility_launch_common.py',
           'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml']
HASHES = {p: hashlib.sha256(source_path(p).read_bytes()).hexdigest() for p in SOURCES}

def pairs(node):
    return [[m.linear.x, m.angular.z] for m in node.cmd_pub.messages]

def local_node():
    n = command_node()
    n.use_hierarchical = True
    n._hier_phase = 'LOCAL'
    n.global_planner_mode = 'efe'
    n._global_goal_xy = np.array([1., 0.])
    n.goal_replan_move_m = .1
    n._waypoints = [(1., 0.)]
    n._wp_idx = 0
    n.waypoint_arrival_radius_m = .1
    n.local_replan_min_remaining_s = 0.
    n.local_replan_on_waypoint_change = False
    n._last_local_plan_target = None
    n.goal_msg = PoseStamped()
    n.goal_msg.pose.position.x = 1.
    n._goal_received_logged = True
    n._update_goal_progress_origin = lambda _: None
    n._snapshot_plan_inputs = lambda: {'goal': n.goal_msg}
    n.belief_m = np.zeros(3)
    n.belief_S = np.eye(3)
    n.belief_stamp = stamp(9.9)
    n._resolve_belief_for_planning = lambda: (n.belief_m.copy(), n.belief_S.copy(), {})
    n._fatal_stop_triggered = False
    n.get_logger = lambda: _Logger()
    n.planner = NS(collision_cost_model=None, nogo_cost_model=None)
    n._dispatch_local_controller = lambda m, target: np.array([[.2, .1]])
    return n

for mode in ['stop', 'fatal', 'correction', 'goal', 'delayed']:
    n = local_node()
    seen = {}
    def dispatch(m, target):
        seen['origin'] = m.tolist()
        seen['target'] = target.tolist()
        if mode in ('stop', 'fatal'):
            if mode == 'fatal':
                n._fatal_stop_triggered = True
            n._publish_safe_stop_command()
        if mode == 'correction':
            # Exact committed-state effect of an accepted correction; filter math is audit 01.
            with n._data_lock:
                n.belief_m = np.array([0., 2., 1.])
                n.belief_stamp = stamp(10.)
        if mode == 'goal':
            goal = PoseStamped(); goal.pose.position.x = -1.
            n._goal_cb(goal)
        if mode == 'delayed':
            n._clock.seconds = 11.
        return np.array([[.2, .1]])
    n._dispatch_local_controller = dispatch
    n._plan_once()
    assert pairs(n)[-1] == [.2, .1]
    OUT['local_' + mode] = dict(seen, published=pairs(n),
        current_belief=n.belief_m.tolist(), current_goal_x=n.goal_msg.pose.position.x,
        installed_at=n._active_plan_started_at.nanoseconds/1e9)

for fatal in [False, True]:
    n = command_node(); n.use_hierarchical = False
    n._fatal_stop_triggered = fatal
    n._pending_plan_started_at = None
    n.latency_compensate_plan_handoff = False
    n._publish_safe_stop_command()
    n._after_plan_result(NS(controls=np.array([[.2,.3]])))
    assert pairs(n) == [[0.,0.],[.2,.3]]
    OUT['solver_after_' + ('fatal' if fatal else 'stop')] = pairs(n)

for mode in ['correction', 'goal', 'delay']:
    n = local_node(); n.use_hierarchical = False
    n._pending_plan_started_at = None
    n.latency_compensate_plan_handoff = True
    n._snapshot_plan_inputs = lambda: {'goal': n.goal_msg, 'pixel_stamp': None, 'state': None}
    n._validate_plan_frames = lambda *_: ('map_bev', 'map_bev')
    n._current_goal_progress_index = lambda *_: 0.
    n.debug_runtime = False
    n._publish_plan_and_metrics = lambda *a, **kw: None
    n._publish_planner_diagnostics = lambda *a, **kw: None
    n._warn_on_plan_health = lambda *a, **kw: None
    n._log_plan_debug_once = lambda *a, **kw: None
    def solve(m, S, goal, **kwargs):
        if mode == 'correction': n.belief_m = np.array([0.,2.,1.])
        if mode == 'goal':
            new = PoseStamped(); new.pose.position.x = -1.; n._goal_cb(new)
        if mode == 'delay': n._clock.seconds = 20.
        return NS(controls=np.array([[.2,.1]]))
    n.planner.plan = solve
    n._plan_once()
    assert pairs(n) == [[.2,.1]]
    assert n._pending_plan_started_at is None
    OUT['full_solver_' + mode] = dict(published=pairs(n),
        current_belief=n.belief_m.tolist(), current_goal_x=n.goal_msg.pose.position.x,
        installed_at=n._active_plan_started_at.nanoseconds/1e9,
        pending_start=n._pending_plan_started_at)

n = command_node(); n._clock.seconds = 9.
n._publish_active_plan_command()
assert pairs(n) == [[.2,.1]]
OUT['planner_clock_reset'] = dict(published=pairs(n), age=n.active_execution_diag_pub.messages[-1].data[0])

n = command_node(); n._active_plan_started_at = _Clock(10.).now()
for t in [10., 10.1, 10.2, 10.3]:
    n._clock.seconds = t; n._publish_active_plan_command()
assert pairs(n) == [[.2,.1],[.2,.1],[.2,.1],[0.,0.]]
OUT['fractional_expiry'] = dict(times=[10.,10.1,10.2,10.3], published=pairs(n), intended_expiry=10.25)

n = command_node(); n.use_hierarchical = False
n._pending_plan_started_at = None; n.latency_compensate_plan_handoff = True
n._clock.seconds = 50.
# Base _plan_once never initializes pending start: old result still accepted.
n._after_plan_result(NS(controls=np.array([[.2,.3]])))
assert n._last_latency_skip_s == 0.
OUT['base_missing_latency_start'] = dict(published=pairs(n), skip=n._last_latency_skip_s)

n = command_node(); n.use_hierarchical = False
n._pending_plan_started_at = _Clock(10.).now()
n._pending_plan_started_active_remaining_s = 1.
n.latency_compensate_plan_handoff = True
n._clock.seconds = 10.6
n._publish_safe_stop_command()
n._after_plan_result(NS(controls=np.array([[.1,0.],[.2,0.],[.15,0.],[.05,0.]])))
assert n._last_latency_skip_steps == 2
OUT['latency_after_stop'] = dict(published=pairs(n), skipped_steps=n._last_latency_skip_steps,
    assumed_motion_s=n._last_latency_skip_s, installed_at=n._active_plan_started_at.nanoseconds/1e9)
n._clock.seconds = 10.75; n._publish_active_plan_command()
assert pairs(n)[-1] == [.05,0.]
OUT['latency_fractional_boundary'] = pairs(n)[-1]

n = command_node(); n.use_hierarchical = False
n._pending_plan_started_at = None; n.latency_compensate_plan_handoff = False
n._after_plan_result(NS(controls=np.array([[2.,3.]])))
assert pairs(n) == [[2.,3.]]
OUT['acceptance_missing_bounds_and_rollout_fields'] = pairs(n)

def adapter():
    n = object.__new__(ActuationNoiseNode)
    values = dict(enabled=True, linear_slip_mean=.03, linear_slip_std=.06,
        angular_slip_mean=0., angular_slip_std=.04, linear_additive_std=.008,
        angular_additive_std=.035, correlation_alpha=.85,
        stop_linear_deadband=1e-4, stop_angular_deadband=1e-4,
        linear_min=0., linear_max=.22, angular_min=-1., angular_max=1.,
        command_timeout_s=.5, _last_input_stamp_s=None, _watchdog_stopped=True,
        _linear_slip_state=0., _angular_slip_state=0.)
    for k,v in values.items(): setattr(n,k,v)
    n._rng = random.Random(210); n._clock = _Clock(10.)
    n.get_clock = lambda: n._clock; n.get_logger = lambda: _Logger()
    n.output=[]; n.diag=[]
    n._pub=NS(publish=n.output.append); n._diag_pub=NS(publish=n.diag.append)
    return n

def send(n, v=.2, w=0.):
    msg=Twist(); msg.linear.x=v; msg.angular.z=w; n._cmd_cb(msg)

def outs(n): return [[m.linear.x,m.angular.z] for m in n.output]

for timer_first in [False, True]:
    n=adapter(); send(n); n._clock.seconds=9.
    if timer_first: n._watchdog_tick()
    send(n)  # command timer re-emits a pre-reset tape; adapter cannot identify it
    n._watchdog_tick()
    assert outs(n)[-1][0] > 0. and not n._watchdog_stopped
    OUT['reset_' + ('watchdog_first' if timer_first else 'receipt_first')] = outs(n)

n=command_node()
n._publish_active_plan_command()
n._clock.seconds=10.5; n._publish_active_plan_command()
assert pairs(n)[-1] == [0.,0.]
assert len(n.active_execution_diag_pub.messages)==1
OUT['expiry_leaves_nonzero_execution_diagnostic'] = dict(published=pairs(n),
    last_diagnostic_command=list(n.active_execution_diag_pub.messages[-1].data[5:7]))

for mode in ['silence','reset','pause','restart','backlog','nonfinite','clip']:
    n=adapter()
    if mode!='restart': send(n)
    if mode=='silence': n._clock.seconds=10.6
    if mode=='reset': n._clock.seconds=9.
    if mode=='restart': n._clock.seconds=100.
    if mode=='backlog':
        n._clock.seconds=10.6; n._watchdog_tick()
        n._clock.seconds=10.61; send(n) # stale source delivery, unknowable from Twist
    if mode=='nonfinite': send(n,float('nan'))
    if mode=='clip': n.enabled=False; send(n,2.,3.)
    n._watchdog_tick(); n._watchdog_tick()
    OUT['adapter_'+mode]=dict(output=outs(n), diagnostics=len(n.diag),
        last_receipt=n._last_input_stamp_s, stopped=n._watchdog_stopped)
    if mode in ('silence','reset','nonfinite'): assert outs(n)[-1]==[0.,0.]
    if mode=='restart': assert not n.output
    if mode=='pause': assert len(n.output)==1
    if mode=='backlog': assert outs(n)[-1][0]>0.
    if mode=='clip': assert outs(n)[-1]==[.22,1.]

# Zero-order-held adapter outputs, integrated exactly on message event intervals.
# Same intended trajectory (.2 m/s, 0 rad/s for 1 s), seed 210.
for label, times in [('10hz',[i/10 for i in range(10)]),
                     ('20hz',[i/20 for i in range(20)]),
                     ('10hz_duplicates',[i/10 for i in range(10) for _ in range(2)]),
                     ('10hz_plus_4hz',sorted([i/10 for i in range(10)]+[i/4 for i in range(4)]))]:
    n=adapter(); trajectory=[]
    for t in times:
        n._clock.seconds=10.+t; send(n); trajectory.append((t,*outs(n)[-1]))
    integral=np.zeros(2)
    for i,(t,v,w) in enumerate(trajectory):
        end=trajectory[i+1][0] if i+1<len(trajectory) else 1.
        integral+=(end-t)*np.array([v,w])
    OUT['cadence_'+label]=dict(deliveries=len(times), integral_v_dt_m=integral[0],
        integral_w_dt_rad=integral[1], samples=trajectory)
assert OUT['cadence_10hz']['integral_v_dt_m'] != OUT['cadence_10hz_duplicates']['integral_v_dt_m']

# Execute the production logger's command-field calculation verbatim. A stop
# topic callback has arrived; its matching noise diagnostic has not arrived yet.
source=(ROOT/'src/experiments/experiments/nodes/experiment_logger.py').read_text()
begin=source.index('        cmd_v = self.cmd_msg.linear.x')
end=source.index('        self._maybe_log_frame_sanity',begin)
latest_diag=NS(data=[10.,1.,.2,0.,.19,.01,1.,1.,0.,.01])
logger=NS(cmd_msg=Twist(),cmd_raw_msg=Twist(),cmd_stamp_s=10.1,
          cmd_raw_stamp_s=10.1,cmd_noise_diag=latest_diag)
fields=dict(self=logger,now_stamp=10.1,math=math)
exec(textwrap.dedent(source[begin:end]),fields)
assert fields['cmd_raw_v']==.2 and fields['cmd_raw_stamp']==10.1
OUT['logger_mixed_command_events']={k:fields[k] for k in
    ['cmd_v','cmd_raw_v','cmd_raw_stamp','cmd_noise_v_error']}

assert HASHES == {p: hashlib.sha256(source_path(p).read_bytes()).hexdigest() for p in SOURCES}, 'source changed during probe'
result=dict(scope='synthetic command-method probe; no ROS graph, Gazebo, or physical-motion measurement',
            source_sha256=HASHES, cases=OUT)
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k: {a:b for a,b in v.items() if a!='samples'} if isinstance(v,dict) else v for k,v in OUT.items()},indent=2))
print(f'{len(OUT)} deterministic cases completed')
