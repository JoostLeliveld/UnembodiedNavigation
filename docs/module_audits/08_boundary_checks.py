"""Additional artifact/mean-model limiting cases. No running node or campaign is changed."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import conftest
sys.path.insert(0,str(ROOT/'experiments/icra_commissioning'))
import numpy as np
import yaml
from network_route_probe import resolve
from planning.core.camera_network import CameraNetworkModel
from planning.planners.base_planner import UnicyclePlannerBase

digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT={}
config=ROOT/'experiments/icra_commissioning/network_navigation_runtime_pilot.yaml'
resolved=resolve(config,'fusion_network_traverse','P0',210)
settings=resolved['settings']
net=CameraNetworkModel(settings['camera_network_artifact_path'])
state=np.array([-6.7,-6.3,.17]);P=np.diag([.012,.025,.013])
base=net.planning_diagnostics(state,P,UnicyclePlannerBase(**settings).camera.H)
max_error=0.
for order in itertools.permutations(net.camera_ids):
    n=CameraNetworkModel(net.path,cameras=order)
    R=n.proxy_ground_covariance(n.query_belief(state,P)['score'])
    max_error=max(max_error,float(np.max(np.abs(R-base['network_R_proxy_m2']))))
OUT['all_camera_orders']=dict(permutations=120,maximum_metric_covariance_difference=max_error)
with tempfile.TemporaryDirectory(prefix='planner08_boundary_') as td:
    td=Path(td)
    # Roster change is accepted by actual launch resolution, retaining the full planner model.
    cfg=yaml.safe_load(config.read_text());cfg['manager_camera_ids']='camera_A,camera_B'
    path=td/'two.yaml';path.write_text(yaml.safe_dump(cfg))
    two=resolve(path,'fusion_network_traverse','P0',210)
    p=UnicyclePlannerBase(**two['settings'])
    OUT['camera_roster_not_bound']=dict(manager_ids=two['resolved']['manager_camera_ids'],
        planner_ids=p.camera_network.camera_ids,artifact=p.camera_network.sha256)
    # Change only q, keeping score/R/geometry fixed: the active objective is unchanged.
    with np.load(net.path) as d:payload={k:d[k].copy() for k in d.files}
    payload['availability'][:]=0.;q0=td/'q0.npz';np.savez(q0,**payload)
    payload['availability'][:]=1.;q1=td/'q1.npz';np.savez(q1,**payload)
    values=[]
    for path in [q0,q1]:
        s=dict(settings);s.update(horizon=4,use_nogo_cost=False,camera_network_artifact_path=str(path))
        p=UnicyclePlannerBase(**s);goal=np.array([-5.,-6.,0.]);obs=p._goal_obs(goal)
        fn=p._get_casadi_valgrad(goal,obs,use_observation_risk=True,use_ambiguity_term=True)
        values.append(fn(np.tile([.1,.02],4),state,P,obs,goal[:2],0.)[0])
    np.testing.assert_equal(*values)
    OUT['availability_not_objective_input']=dict(q_zero_objective=values[0],q_one_objective=values[1])
    # Coherent drift opt-in changes numerical propagation but is absent from CasADi.
    p.coherent_drift=True
    raw=p._evaluate_controls(np.tile([.1,.02],4),state,P,goal,obs,None)
    symbolic=fn(np.tile([.1,.02],4),state,P,obs,goal[:2],0.)[0]
    heff=sum(p.discount_gamma**t for t in range(4))
    OUT['optional_coherent_drift_mismatch']=dict(numpy_normalized=raw/heff,casadi=symbolic)
    assert not np.isclose(raw/heff,symbolic,rtol=1e-8,atol=1e-10)

OUT['sources']={p:digest(ROOT/p) for p in json.loads((Path(__file__).with_name('08_planner_probe_results.json')).read_text())['sources']}
OUT['probe_sha256']=digest(__file__)
Path(__file__).with_name('08_boundary_results.json').write_text(json.dumps(OUT,indent=2)+'\n')
print(json.dumps({k:v for k,v in OUT.items() if k!='sources'},indent=2))
