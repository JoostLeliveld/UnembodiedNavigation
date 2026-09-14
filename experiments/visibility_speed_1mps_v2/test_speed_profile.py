import copy,inspect,json
import numpy as np
import pytest
from paths import HERE
from speed_profile import PROFILE,V_MAX,LOOKAHEAD,LOCAL_GOAL_AHEAD,validate_config
from speed_planner import planner,camera_models
from path_tracker import track
from route_handoff import RouteHandoff
from planner_adapter import NetworkPlanner

def test_1mps_is_reachable_on_straight_plan_and_brakes_at_goal():
 states=np.c_[np.linspace(0,6,25),np.zeros((25,2))];controls=np.tile([1.,0.],(24,1))
 u,info=track([0,0,0],states,controls,[6,0]);np.testing.assert_allclose(u,[1.,0.],atol=1e-12)
 u,_=track([5.95,0,0],states,controls,[6,0]);assert 0<=u[0]<.04
 u,_=track([0,0,1.],states,controls,[6,0]);assert u[0]==0 and abs(u[1])<=.4
 assert LOOKAHEAD==.2 and LOCAL_GOAL_AHEAD==.65

def test_shared_profile_rejects_drift_and_missing():
 cfg=json.loads((HERE/'pilot.json').read_text())['runs'][0];validate_config(cfg)
 for bad in [{},dict(cfg,speed_profile=dict(PROFILE,max_linear_speed_m_s=.16))]:
  with pytest.raises(ValueError):validate_config(bad)
 assert all(c['speed_profile']==PROFILE for c in json.loads((HERE/'pilot.json').read_text())['runs'])

def test_global_to_local_preview_and_runtime_factory():
 h=RouteHandoff([[0,0,0],[10,0,0]],[10,0]);target,info=h.query([0,0,0]);np.testing.assert_allclose(target,[.65,0])
 cfg=json.loads((HERE/'pilot.json').read_text())['runs'][0];p=NetworkPlanner(cfg)
 assert p.p.v_max==1. and p.p.horizon==20 and p.p.dt==.25
 assert p.p.optimizer_maxiter==40 and p.p.w_max==.4
 assert 'visibility_speed_1mps_v2' in inspect.getfile(planner)


def test_curvature_limit_preserves_turn_geometry():
 states=np.array([[0,0,0],[.2,.1,.4],[.4,.2,.4]])
 u,info=track([0,0,0],states,[[1,.4],[1,.4]],[5,0])
 delta=np.array(info['target']);curvature=2*np.sin(info['heading_error_rad'])/max(np.linalg.norm(delta),.05)
 assert 0<u[0]<1 and abs(u[1])<=.4+1e-12
 np.testing.assert_allclose(u[1],u[0]*curvature,atol=1e-12)
