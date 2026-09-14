from paths import *
import json
import numpy as np
from dataclasses import asdict
from planner_adapter import NetworkPlanner
from route_handoff import RouteHandoff
from speed_profile import profile_hash
D=Path(sys.argv[1]);cfg=json.loads((D/'capture.json').read_text());p=NetworkPlanner(cfg)
global_plan=json.loads((D/'global_plan.json').read_text());assert global_plan['speed_profile_sha256']==profile_hash();handoff=RouteHandoff(global_plan['states'],cfg['goal'])
boot=json.loads((D/'prelaunch_plan.json').read_text());p.p.prev_controls_flat=np.array(boot['controls']).reshape(-1)
for line in sys.stdin:
    a=json.loads(line)
    target,routing=handoff.query(a['mean'])
    result=p.plan(np.array(a['mean']),np.array(a['cov']),target,
                  progress_index=a['progress'],source_age_s=a['source_age_s'])
    r=asdict(result);r={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}
    r['forecast']=dict(p.last_trace,**routing,global_final_goal=cfg['goal'])
    print(json.dumps(r),flush=True)
