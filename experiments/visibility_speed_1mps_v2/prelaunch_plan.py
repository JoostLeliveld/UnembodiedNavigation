from paths import *
import json
import numpy as np
from dataclasses import asdict
from planner_adapter import NetworkPlanner
from bayesian_field import JointFieldFilter
from route_handoff import RouteHandoff
import shutil
from speed_profile import profile_hash,validate_config
D=Path(sys.argv[1]);cfg=json.loads((D/'capture.json').read_text());p=NetworkPlanner(cfg)
f=JointFieldFilter(p.model,[cfg['x'],cfg['y'],cfg['yaw']],np.diag([.1**2,.1**2,np.deg2rad(5)**2]),cfg['arm']=='IndependentIndependent')
validate_config(cfg)
source=T/('global_'+cfg['arm'])/'global_plan.json'
shutil.copy2(source,D/'global_plan.json')
global_plan=json.loads(source.read_text());assert global_plan['rollout_valid'];assert global_plan['speed_profile_sha256']==profile_hash()
handoff=RouteHandoff(global_plan['states'],cfg['goal']);target,routing=handoff.query(f.mean)
result=p.plan(f.mean,f.cov,target);assert result.rollout_valid,result.invalid_reason
r=asdict(result);r={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}
r.update(forecast=p.last_trace,role='Nominal local 5-second horizon solve before Gazebo; live solve required before execution',reference_inputs=False)
(D/'prelaunch_plan.json').write_text(json.dumps(r,indent=2))
print('PRELAUNCH',cfg['id'],result.solve_time_s,result.backend,flush=True)
