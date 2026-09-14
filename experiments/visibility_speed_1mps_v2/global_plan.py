"""Long-horizon IWAI multistart, retaining the joint Gaussian forecast.

A CasADi mapaccum keeps graph size bounded. The global finite-horizon cost is
undiscounted; the unchanged local objective still uses its original discount.
The original optimizer checks and compares geometry-derived seed basins.
"""
from paths import *
import json,copy,time
import numpy as np
import casadi as ca
from dataclasses import asdict
from gp_forecast import build,event_offsets
from planner_adapter import NetworkPlanner
from bayesian_field import JointFieldFilter
from speed_profile import V_MAX,PROFILE,profile_hash

def build_global(model,p,artifact):
 n=p.horizon;one=copy.copy(p);one.horizon=1;one.discount_gamma=1.
 objective,trace=build(model,one,artifact,independent=True)
 z=ca.SX.sym('state',186);u=ca.SX.sym('u',2);off=ca.SX.sym('offset');goal=ca.SX.sym('goal',2)
 m=z[:13];P=ca.reshape(z[13:182],13,13);value,_,parts=objective(m,P,u,off,goal);nm,np_,q=trace(m,P,u,off,goal)
 # Initial camera means are zero and future innovations are zero.
 nextm=ca.vertcat(nm,ca.DM.zeros(10));nz=ca.vertcat(nextm,np_,z[182:]+p.dt*parts)
 stage=ca.Function('global_stage',[z,u,off,goal],[nz,q]);scan=stage.mapaccum(n)
 m0=ca.MX.sym('m',13);P0=ca.MX.sym('P',13,13);U=ca.MX.sym('U',2,n);ofs=ca.MX.sym('ofs',n);g=ca.MX.sym('g',2)
 Z,Q=scan(ca.vertcat(m0,ca.reshape(P0,169,1),ca.DM.zeros(4)),U,ofs.T,ca.repmat(g,1,n));parts=Z[182:,-1];cost=ca.sum1(parts)
 fn=ca.Function('global_cost',[m0,P0,U,ofs,g],[cost,ca.gradient(cost,ca.vec(U)),parts]);tr=ca.Function('global_trace',[m0,P0,U,ofs,g],[Z[:3,:],Z[13:182,:],Q])
 return fn,tr

def solve(cfg,out):
 p=NetworkPlanner(cfg);longest=max(x['length_m'] for x in cfg['global_routes'])
 p.p.horizon=int(np.ceil((longest/V_MAX+PROFILE['global_turn_allowance_s'])/.25));p.p.optimizer_maxiter=8;p.p.optimizer_maxfun=16;p.p.optimizer_warm_start=False;p.p.optimizer_multistart_include_direct=False;p.p.optimizer_terminal_goal_tolerance_m=.20;p.p.discount_gamma=1.;p.p.optimizer_initial_routes=[dict(name=x['name'],waypoints=x['waypoints']) for x in cfg['global_routes']]
 p.fn,p.trace=build_global(p.model,p.p,p.artifact)
 f=JointFieldFilter(p.model,[cfg['x'],cfg['y'],cfg['yaw']],np.diag([.01,.01,np.deg2rad(5)**2]));assert np.all(f.mean[3:]==0)
 p.mean=f.mean;p.cov=f.cov;p.offsets=event_offsets(0,p.artifact['opportunity_period_s'],p.p.horizon,.25)
 # Install best geometry seed as the common initial candidate, avoiding a
 # fruitless thousand-control stationary solve; every other basin still enters.
 first=p.p._controls_for_waypoints(f.mean[:3],p.p.optimizer_initial_routes[0]['waypoints']);p.p._initial_controls_flat=lambda:first.copy()
 seeds=[]
 for route in p.p.optimizer_initial_routes:
  u=p.p._controls_for_waypoints(f.mean[:3],route['waypoints']).reshape(-1,2)
  val,grad,parts=p.fn(f.mean,f.cov,u.T,p.offsets,cfg['goal']);assert np.isfinite(np.array(grad)).all()
  seeds.append(dict(name=route['name'],cost=float(val),parts=np.array(parts).ravel().tolist()))
 (out/'global_seed_costs.json').write_text(json.dumps(seeds,indent=2));print('GLOBAL SEEDS',cfg['line'],p.p.horizon,seeds,flush=True)
 result=p.plan(f.mean,f.cov,cfg['goal']);r=asdict(result);r={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()};r.update(speed_profile_sha256=profile_hash(),speed_profile=PROFILE,forecast=p.last_trace,horizon=p.p.horizon,dt=.25,global_discount=1.,role='Full-route global CasADi IWAI multistart before Gazebo',reference_inputs=False)
 (out/'global_plan.json').write_text(json.dumps(r,indent=2));assert result.rollout_valid,result.invalid_reason;assert result.terminal_goal_distance_pred<=.2,result.terminal_goal_distance_pred
 print('GLOBAL RESULT',cfg['line'],result.selected_source,result.solve_time_s,result.terminal_goal_distance_pred,flush=True)
 return r
if __name__=='__main__':
 cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);solve(cfg,out)
