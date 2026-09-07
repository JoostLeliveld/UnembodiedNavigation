"""Verify the concurrent cache-key repair separately from the preserved baseline."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import conftest
import numpy as np
from planning.planners.base_planner import UnicyclePlannerBase

p=UnicyclePlannerBase(horizon=5,dt=.25,v_min=0.,v_max=.22,w_min=-1.,w_max=1.,
    control_weight=.02,process_noise_xy=.01,process_noise_theta=.02,obs_noise_uv=2.5,
    goal_sigma_uv=30.,risk_weight_obs=1.,ambiguity_weight=1.,optimizer_maxiter=3,
    optimizer_gtol=1e-5,optimizer_warm_start=False,seed=210,
    camera_params=dict(cam_pos=(-5.,-5.,5.),look_at=(0.,0.,0.),img_width=1280,img_height=720,fov_h_rad=1.2))
goal=np.array([1.2,1.,0.]);obs=p._goal_obs(goal);m=np.array([.2,.4,.1]);P=np.diag([.2,.3,.05]);u=np.tile([.17,.06],5)
def function():return p._get_casadi_valgrad(goal,obs,use_observation_risk=True,use_ambiguity_term=True)
def value(fn):return fn(u,m,P,obs,goal[:2],0.)[0]
before=function();v0=value(before)
p.process_noise_theta=.7
after=function();v1=value(after)
p._casadi_valgrad_cache.clear();v2=value(function())
assert after is not before and v1==v2 and not np.isclose(v0,v1)
old=json.loads(Path(__file__).with_name('08_source_snapshot').joinpath('manifest.json').read_text())['sources']
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={p:digest(ROOT/p) for p in old}
sources['src/planning/planning/core/casadi_cache.py']=digest(ROOT/'src/planning/planning/core/casadi_cache.py')
OUT=dict(scope='new cache-key Q invalidation independently verified; broader cache/JIT covered only by focused regressions',
    before=v0,after_without_manual_clear=v1,after_manual_clear=v2,new_function=after is not before,
    sources=sources,baseline_source_differences={p:dict(baseline=old.get(p),followup=h) for p,h in sources.items() if p not in old or old[p]!=h},
    probe_sha256=digest(__file__))
Path(__file__).with_name('08_cache_followup_results.json').write_text(json.dumps(OUT,indent=2)+'\n')
print(json.dumps({k:v for k,v in OUT.items() if k not in ['sources','baseline_source_differences']},indent=2))
