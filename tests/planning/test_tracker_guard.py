"""A refused translation may permit a checked stationary turn toward the same target."""
from types import SimpleNamespace

import numpy as np
import pytest

from planning.core.tracker_guard import checked_tracker_controls
from planning.nodes.efe_agent_node import EfeAgentNode


def corridor_node():
    # A 1.2 m wide lane with the deployed 0.55 m centre standoff.
    lane = SimpleNamespace(enabled=True, clearance_state_np=lambda s: .05-abs(s[1]))
    planner = SimpleNamespace(collision_cost_model=None, nogo_cost_model=lane)
    return SimpleNamespace(planner=planner, local_horizon=12, dt=.25, v_max=.22,
                           w_min=-1., w_max=1., simple_tracker_yaw_gate_rad=.6)


def decision(node, state, target, enabled=True):
    controls = EfeAgentNode._simple_local_plan(node, state, target)
    return checked_tracker_controls(controls, state, target, dt=node.dt,
        w_min=node.w_min, w_max=node.w_max,
        safety_check=lambda u, m: EfeAgentNode._simple_plan_safe_to_execute(node, u, m),
        allow_rotation_recovery=enabled)


@pytest.mark.parametrize('side', [-1., 1.])
def test_shallow_outward_heading_can_recover_without_relaxing_gate(side):
    node=corridor_node();state=np.array([0., side*.049, side*.2]);target=np.array([.3, 0.])
    old=decision(node,state,target,False)
    assert old.safe_steps == 0
    d=decision(node,state,target)
    assert d.rotation_recovery and d.safe_steps > 0
    assert np.all(d.controls[:,0] == 0.)
    assert side*d.controls[0,1] < 0.
    assert np.all(abs(d.controls[:,1]) <= 1.)


def test_allowed_motion_remains_the_original_controller_output():
    node=corridor_node();state=np.zeros(3);target=np.array([.3,0.])
    d=decision(node,state,target)
    assert not d.rotation_recovery
    np.testing.assert_array_equal(d.controls,EfeAgentNode._simple_local_plan(node,state,target))


def test_aligned_motion_into_forbidden_floor_stays_refused():
    node=corridor_node();state=np.array([0.,.049,np.pi/2]);target=np.array([0.,.3])
    d=decision(node,state,target)
    assert d.safe_steps == 0 and not d.rotation_recovery


def test_rotation_must_pass_the_gate_too():
    def reject(u,m):
        return 0,'collision_geometry_violation_step_0:-0.1'
    d=checked_tracker_controls(np.tile([.2,.1],(12,1)),np.zeros(3),np.ones(2),
        dt=.25,w_min=-1.,w_max=1.,safety_check=reject,allow_rotation_recovery=True)
    assert d.safe_steps == 0 and not d.rotation_recovery


def test_malformed_controls_do_not_trigger_recovery():
    node=corridor_node()
    d=checked_tracker_controls(np.array([]),np.zeros(3),np.ones(2),
        dt=.25,w_min=-1.,w_max=1.,
        safety_check=lambda u,m:EfeAgentNode._simple_plan_safe_to_execute(node,u,m),
        allow_rotation_recovery=True)
    assert d.reason == 'empty_or_malformed_controls' and not d.rotation_recovery
