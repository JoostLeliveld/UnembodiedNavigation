"""Numerical equivalence and minimal serialization check for concurrent algebra edits."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import conftest
import casadi as ca
import numpy as np
from planning.core.casadi_efe import risk_ca
from planning.core.efe_utils import risk_components
OUT={'scope':'synthetic algebra/serialization; not navigation or calibration',
     'casadi_version':ca.__version__,'cases':{}}
C=OUT['cases']
diagonal=ca.MX.sym('diagonal',2);rhs=ca.MX.sym('rhs',2,2)
raw=ca.Function('audit08_diagonal_solve',[diagonal,rhs],[ca.solve(ca.diag(diagonal),rhs)])
try:
    loaded=ca.Function.deserialize(raw.serialize())
    C['raw_diagonal_solve_serialization']={'reload_succeeded':True}
except RuntimeError as error:
    C['raw_diagonal_solve_serialization']={'reload_succeeded':False,'error':str(error)}
mu=ca.MX.sym('mean',2);sigma=ca.MX.sym('sigma',2,2);target=ca.MX.sym('target',2)
value=risk_ca(mu,sigma,target,ca.diag(diagonal))
fn=ca.Function('audit08_repaired_risk',[mu,sigma,target,diagonal],[value,ca.gradient(value,mu)])
loaded=ca.Function.deserialize(fn.serialize())
rng=np.random.default_rng(8017)
differences=[];parts_diff=[];gradient_errors=[];roundtrip=[]
for _ in range(20):
    a=rng.normal(size=(2,2));s=a@a.T+np.eye(2)*.1
    mean=rng.normal(size=2);goal=rng.normal(size=2);d=rng.uniform(.1,3.,2)
    # Independent dense inverse reference includes the existing 1e-9 jitter.
    t=np.diag(d);ss=s+np.eye(2)*1e-9;tt=t+np.eye(2)*1e-9;delta=goal-mean
    expected=.5*(np.trace(np.linalg.inv(tt)@ss)+delta@np.linalg.inv(tt)@delta-2
                   +np.linalg.slogdet(tt)[1]-np.linalg.slogdet(ss)[1])
    actual,gradient=fn(mean,s,goal,d)
    differences.append(abs(float(actual)-expected))
    numerical=[]
    for i in range(2):
        step=np.eye(2)[i]*1e-5
        numerical.append((float(fn(mean+step,s,goal,d)[0])-float(fn(mean-step,s,goal,d)[0]))/2e-5)
    gradient_errors.append(float(np.max(np.abs(np.array(gradient).reshape(-1)-numerical))))
    reload_value,reload_gradient=loaded(mean,s,goal,d)
    roundtrip.append(max(abs(float(reload_value)-float(actual)),
        float(np.max(np.abs(np.array(reload_gradient)-np.array(gradient))))))
    # Current NumPy helper is algebraically the dense KL decomposition.
    parts=risk_components(mean,s,(goal,t))
    independent=.5*(np.trace(np.linalg.inv(t)@s)+delta@np.linalg.inv(t)@delta-2
                    +np.linalg.slogdet(t)[1]-np.linalg.slogdet(s)[1])
    parts_diff.append(abs(parts['total']-independent))
assert max(differences)<1e-11
assert max(gradient_errors)<1e-7
assert max(roundtrip)==0
assert max(parts_diff)<1e-11
C['repaired_risk']=dict(examples=20,max_value_error=max(differences),
    max_finite_difference_gradient_error=max(gradient_errors),max_serialized_reload_error=max(roundtrip))
C['numpy_risk_solve_equivalence']=dict(examples=20,max_difference=max(parts_diff))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT['sources']={p:digest(ROOT/p) for p in ['src/planning/planning/core/efe_utils.py','src/planning/planning/core/casadi_efe.py','src/planning/planning/core/casadi_cache.py']}
OUT['probe_sha256']=digest(__file__)
Path(__file__).with_suffix('.json').write_text(json.dumps(OUT,indent=2)+'\n')
print(json.dumps(OUT,indent=2))
