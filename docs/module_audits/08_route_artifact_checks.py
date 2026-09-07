"""Read-only resolved configuration, frozen planned-route, artifact and forecast audit.

Source install/setup.bash first; set ROS_LOG_DIR=/tmp/08_ros_log. Launch actions
are constructed but never executed. All writes belong to this audit directory.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import conftest
sys.path.insert(0,str(ROOT/'experiments/icra_commissioning'))
import numpy as np
import casadi as ca
from scipy.spatial import cKDTree
from network_route_probe import resolve
from planning.core.camera_network import CameraNetworkModel, projection_jacobian
from planning.planners.base_planner import UnicyclePlannerBase
from planning.core.dynamics import unicycle_step,unicycle_jacobian,unicycle_process_noise
from planning.core.nogo_cost import NogoZoneCostModel,NogoCostConfig
from unav_common.occlusion_geometry import AxisAlignedPrism,OcclusionScene

digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT={'scope':'model/software only; explicit saved planned routes, no run accuracy', 'cases':{}}
C=OUT['cases']
cfg=ROOT/'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml'
resolved={a:resolve(cfg,'fusion_network_traverse',a,210) for a in ['P0','P1','P2']}
settings=resolved['P0']['settings']
OUT['resolved_configuration']=resolved
OUT['configuration_hash']=digest(cfg)
C['active_arm_setting_differences']={k:[v['settings'][k] for v in resolved.values()]
    for k in settings if len({repr(v['settings'][k]) for v in resolved.values()})>1}
OUT['imports']={k:sys.modules[k].__file__ for k in [
    'planning.planners.base_planner','planning.core.camera_network','planning.nodes.unicycle_planner_node',
    'experiments.core.visibility_launch_common']}
root=ROOT/'logs/studies/icra_commissioning_20260905/network_planner'
manifest=json.loads((root/'manifest.json').read_text())
networks={k:CameraNetworkModel(ROOT/v['path']) for k,v in manifest['artifacts'].items()}
OUT['network_manifest_hash']=digest(root/'manifest.json')
OUT['artifact_hashes']={k:n.sha256 for k,n in networks.items()}
assert all(n.sha256==manifest['artifacts'][k]['sha256'] for k,n in networks.items())
base=networks['uniform']
C['arm_arrays']={k:dict(camera_ids=n.camera_ids,R_equal=bool(np.array_equal(n.R,base.R)),
    Rmiss_equal=bool(np.array_equal(n.R_miss,base.R_miss)),axes_equal=bool(np.array_equal(n.xs,base.xs) and np.array_equal(n.ys,base.ys)),
    score_max_abs_difference=float(np.max(np.abs(n.fields['score']-base.fields['score']))),
    availability_max_abs_difference=float(np.max(np.abs(n.fields['availability']-base.fields['availability'])))) for k,n in networks.items()}
# Read-only equivalent of verifier hash checks, collecting every mismatch instead of overwriting verification.json.
checks=[]
def check(path,expected,stage):
    actual=digest(path) if path.exists() else None
    checks.append(dict(path=str(path.relative_to(ROOT)),stage=stage,expected=expected,actual=actual,match=actual==expected))
for p,h in manifest['sources'].items():check(ROOT/p,h,'export source')
protocol=json.loads((root/'probe_reviewed/protocol.json').read_text())
results=json.loads((root/'probe_reviewed/results.json').read_text())
check(root/'manifest.json',protocol['network_manifest_sha256'],'probe manifest')
check(root/'probe_reviewed/protocol.json',results['protocol_sha256'],'probe protocol')
for p,h in protocol['sources'].items():check(ROOT/p,h,'probe source')
for p,h in results['files'].items():check(root/'probe_reviewed'/p,h,'probe output')
OUT['verification_checks']=checks
C['verification_summary']=dict(checked=len(checks),mismatches=[c for c in checks if not c['match']])
# Follow the nested fit manifests too: runtime loader only enforces schema/arrays.
for source in [ROOT/'logs/studies/icra_commissioning_20260905/manifest.json',
               ROOT/'logs/studies/icra_commissioning_20260905/field_study/manifest.json']:
    for p,h in json.loads(source.read_text()).get('files',{}).items():check(ROOT/p,h,'nested commissioning source')

route_root=root/'full_route_v1'
route_summary=json.loads((route_root/'summary.json').read_text())
OUT['saved_route_hashes']={a:digest(route_root/f'{a}_result.json') for a in ['P0','P1','P2']}
OUT['saved_route_source_comparison']={p:dict(expected=h,current=digest(ROOT/p),match=h==digest(ROOT/p))
    for p,h in route_summary['sources'].items()}
route_checks={}
for arm in ['P0','P1','P2']:
    print('checking saved route',arm,flush=True)
    r=json.loads((route_root/f'{arm}_result.json').read_text())
    p=UnicyclePlannerBase(**r['settings'])
    m=np.array(r['state']);P=np.array(r['initial_P']);u=np.array(r['result']['controls'])
    actual=[m.copy()];cov=[P.copy()]
    for control in u:
        m,P=p.predict(m,P,control);actual.append(m.copy());cov.append(P.copy())
    actual=np.array(actual);cov=np.array(cov)
    np.testing.assert_allclose(actual,np.array(r['result']['states']),atol=1e-12)
    dense=np.vstack([a+np.linspace(0,1,max(2,math.ceil(np.linalg.norm(b[:2]-a[:2])/.02)+1))[:,None]*(b-a)
                     for a,b in zip(actual[:-1],actual[1:])])
    # Independent primitive rectangle-to-disc distance, avoiding planner clearance wrapper.
    boxes=p.collision_cost_model.prisms
    def body_clearance(point):
        values=[]
        x,y=point[:2]
        for b in boxes:
            dx=max(b.xmin-x,0.,x-b.xmax);dy=max(b.ymin-y,0.,y-b.ymax)
            d=math.hypot(dx,dy)
            if dx==dy==0.:d=-min(x-b.xmin,b.xmax-x,y-b.ymin,b.ymax-y)
            values.append(d-p.robot_collision_radius_m)
        return min(values)
    dense_clear=min(body_clearance(x) for x in dense)
    model_errors=[];denominators=[]
    for x in dense:
        den=(p.camera.H@np.r_[x[:2],1.])[2];denominators.append(float(den))
    # Chart linearization support: denominator mean +/- 3 standard deviations over XY P.
    lows=[];ratios=[]
    for x,pp in zip(actual,cov):
        den=float((p.camera.H@np.r_[x[:2],1.])[2]);sd=math.sqrt(max(float(p.camera.H[2,:2]@pp[:2,:2]@p.camera.H[2,:2]),0.))
        lows.append(abs(den)-3*sd);ratios.append(sd/abs(den))
    evald=p.evaluate_rollout_controls(np.array(r['state']),np.array(r['initial_P']),r['goal'],u)
    route_checks[arm]=dict(goal_gap_recomputed_m=float(np.linalg.norm(actual[-1,:2]-r['goal'])),
        dense_body_clearance_m=dense_clear,waypoint_polyline_length_m=float(np.linalg.norm(np.diff(actual[:,:2],axis=0),axis=1).sum()),
        recorded_status=r['result']['optimizer_status'],recorded_success=r['result']['optimizer_success'],
        recorded_message=r['result']['optimizer_message'],current_rollout_valid=evald['rollout_valid'],
        diagnostic_cost_recomputed=evald['total_cost'],diagnostic_cost_recorded=r['result']['total_cost'],
        max_cov_asymmetry=float(np.max(np.abs(cov-cov.swapaxes(-1,-2)))),min_cov_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
        terminal_P=cov[-1].tolist(),chart_min_abs_denominator=float(min(abs(v) for v in denominators)),
        chart_min_three_sigma_denominator_bound=float(min(lows)),chart_max_relative_denominator_sd=float(max(ratios)),
        chart_max_condition=float(max(np.linalg.cond(projection_jacobian(p.camera.H,x)) for x in actual)),
        max_abs_omega=float(np.max(np.abs(u[:,1]))),
        max_one_step_euler_vs_twist_gap_m=float(max(abs(v)*math.hypot(1-math.sin(w)/w,(1-math.cos(w))/w)
            if abs(w)>1e-8 else 0. for v,w in u)))
C['saved_full_routes']=route_checks
C['short_probe_to_active_effective_differences']={k:dict(short=v,active=settings.get(k))
    for k,v in protocol['settings'].items() if settings.get(k)!=v}

# Future.py's exact integration loop on a prescribed control stream containing an internal turn.
source=ROOT/'experiments/icra_commissioning/future.py'
tree=ast.parse(source.read_text())
loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and n.lineno==91)
code=compile(ast.Module(body=[loop],type_ignores=[]),str(source),'exec')
class NoCamera:
    def forecast(self,state,P,approx):return P.copy(),0.,0.
forecasts={}
for cadence in (.2,1.):
    env=dict(np=np,unicycle_jacobian=unicycle_jacobian,unicycle_process_noise=unicycle_process_noise,
        unicycle_step=unicycle_step,start=0.,previous=0.,horizon=1.,cadence=cadence,state=np.zeros(3),
        P=np.eye(3)*.01,tt=np.array([0.,.2]),uu=np.array([[.22,0.],[0.,1.]]),joint=NoCamera(),approx='branch',qs=[],support=[])
    exec(code,env)
    forecasts[str(cadence)]=dict(state=env['state'].tolist(),P=env['P'].tolist())
C['future_cadence_changes_prescribed_motion']=forecasts
assert np.allclose(forecasts['0.2']['state'],[.044,0.,.8])
assert np.allclose(forecasts['1.0']['state'],[.22,0.,0.])
# Execute the exact class with only its numerical dependencies, not the archival main().
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointCommissioning')
features=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pose_features')
from model import update
ns=dict(np=np,cKDTree=cKDTree,update=update)
exec(compile(ast.Module(body=[features,klass],type_ignores=[]),str(source),'exec'),ns)
j=ns['JointCommissioning'].__new__(ns['JointCommissioning'])
poses=np.zeros((12,3));poses[1:,0]=np.arange(10.,21.)
j.tree=cKDTree(ns['pose_features'](poses));j.outcomes=[[]]+[[('A',np.eye(2)*.01)] for _ in range(11)]
post,q,distance=j.forecast(np.zeros(3),np.eye(3),'branch')
C['future_neighbor_support']=dict(nearest_distance=distance,farthest_distance=20.,q_any=q,posterior_xy_trace=float(np.trace(post[:2,:2])))
# One-step branch averaging is exact; repeated compression does not keep the full event tree.
def hit(v):return v*.04/(v+.04)
q=.4;v=.2;Q=.01
first=q*hit(v)+(1-q)*v
compressed=q*hit(first+Q)+(1-q)*(first+Q)
full=sum((q if a else 1-q)*(q if b else 1-q)*(hit((hit(v) if a else v)+Q) if b else (hit(v) if a else v)+Q)
         for a,b in [(0,0),(0,1),(1,0),(1,1)])
C['two_step_branch_compression']=dict(compressed_variance=compressed,full_tree_expected_variance=full)

def geometry(x0,x1,y0,y1):return OcclusionScene((AxisAlignedPrism('box',x0,x1,y0,y1,0.,1.),)).to_json()
base_settings=dict(settings);base_settings.update(horizon=1,dt=1.,use_visibility_model=False,camera_network_artifact_path='',
    use_nogo_cost=False,collision_geometry_json=geometry(.02,.0201,0.,1.))
p=UnicyclePlannerBase(**base_settings)
radius=p.robot_collision_radius_m
start=np.array([0.,-radius+.00005,0.]);end=unicycle_step(start,[.04,0.],1.)
diag=p._trajectory_plan_diagnostics(start,np.eye(3)*.01,[[.04,0.]],[1.,0.])
mid=start.copy();mid[0]=.02005
C['sampled_disc_tangency']=dict(start_clearance=p.collision_clearance_state_np(start),end_clearance=p.collision_clearance_state_np(end),
    intermediate_clearance=p.collision_clearance_state_np(mid),rollout_valid=diag['rollout_valid'])
assert diag['rollout_valid'] and p.collision_clearance_state_np(mid)<0.
# Starting point is omitted: a first endpoint outside an initially penetrating footprint is admitted.
base_settings['collision_geometry_json']=geometry(-1.,0.,-1.,1.)
p=UnicyclePlannerBase(**base_settings);start=np.array([radius-.001,0.,0.])
diag=p._trajectory_plan_diagnostics(start,np.eye(3)*.01,[[.04,0.]],[2.,0.])
C['starting_footprint_omitted']=dict(start_clearance=p.collision_clearance_state_np(start),rollout_valid=diag['rollout_valid'])
# Invalid geometry returns NaN; the planner treats that as no finite constraint.
base_settings['collision_geometry_json']=geometry(math.nan,1.,-1.,1.)
p=UnicyclePlannerBase(**base_settings)
diag=p._trajectory_plan_diagnostics(np.zeros(3),np.eye(3)*.01,[[.04,0.]],[2.,0.])
C['nonfinite_geometry_fail_open']=dict(raw_distance=p.collision_signed_distance_state_np(np.zeros(3)),
    wrapper_clearance=p.collision_clearance_state_np(np.zeros(3)),rollout_valid=diag['rollout_valid'])
ng=NogoZoneCostModel(NogoCostConfig(weight=40.,safe_distance=.55,geometry_json=geometry(-2.,2.,-2.,2.),mode='keep_in'))
x=ca.MX.sym('ng08',3);cost=ng.make_penalty_state_casadi()(x);fn=ca.Function('ng08f',[x],[cost,ca.gradient(cost,x)])
pos=np.array([1.43,.13,0.]);value,gradient=fn(pos);eps=1e-6
fd=np.array([(ng.penalty_state_np(pos+np.eye(3)[i]*eps)-ng.penalty_state_np(pos-np.eye(3)[i]*eps))/(2*eps) for i in range(3)])
C['nogo_smooth_derivative']=dict(numpy_cost=ng.penalty_state_np(pos),casadi_cost=float(value),
    max_gradient_error=float(np.max(np.abs(np.array(gradient).ravel()-fd))))
np.testing.assert_allclose(np.array(gradient).ravel(),fd,atol=1e-5,rtol=1e-5)
OUT['sources']={p:digest(ROOT/p) for p in json.loads((Path(__file__).with_name('08_planner_probe_results.json')).read_text())['sources']}
OUT['probe_sha256']=digest(__file__)
Path(__file__).with_name('08_route_artifact_results.json').write_text(json.dumps(OUT,indent=2)+'\n')
print('COMPLETED',len(C),'case groups',flush=True)
