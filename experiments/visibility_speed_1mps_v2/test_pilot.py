from paths import *
import json,copy
import numpy as np
import casadi as ca
import pytest
from planner_adapter import NetworkPlanner
from gp_forecast import build,gp_function,expected_update,event_offsets
from global_plan import build_global
from bayesian_field import JointFieldFilter

@pytest.mark.parametrize('arm',['Uniform','IWAI','Commissioned'])
def test_global_scan_gradient_and_forecast(arm):
 p=NetworkPlanner(dict(seed=1200,arm=arm));p.p.discount_gamma=1
 f=JointFieldFilter(p.model,[-5.13,-6.63,.03],np.diag([.01,.01,.0076]))
 fn,tr=build_global(p.model,p.p,p.artifact);ref,rt=build(p.model,p.p,p.artifact,independent=True)
 U=np.tile([.08,.02],(20,1)).T;off=event_offsets(0,p.artifact['opportunity_period_s']);args=(f.mean,f.cov,U,off,[4.4,-6.6]);a=fn(*args);b=ref(*args)
 for aa,bb in zip(a,b):np.testing.assert_allclose(np.array(aa),.25*np.array(bb),rtol=1e-9,atol=1e-8)
 for aa,bb in zip(tr(*args),rt(*args)):np.testing.assert_allclose(np.array(aa),np.array(bb),atol=1e-10)
 cov=np.array(tr(*args)[1]).T.reshape(-1,13,13);assert np.linalg.eigvalsh(cov).min()>0
 grad=np.array(a[1]).ravel();v=U.T.reshape(-1);h=1e-6
 for i in [0,1,18,39]:
  plus=v.copy();minus=v.copy();plus[i]+=h;minus[i]-=h
  finite=(float(fn(f.mean,f.cov,plus.reshape(-1,2).T,off,args[-1])[0])-float(fn(f.mean,f.cov,minus.reshape(-1,2).T,off,args[-1])[0]))/(2*h)
  np.testing.assert_allclose(grad[i],finite,rtol=5e-4,atol=1e-4)

def test_field_interpolation_and_uniform_invariance():
 a=json.loads((HERE/'gp.json').read_text());xy=[a['xs'][13],a['ys'][22]]
 for variant,key in [('IWAI','score'),('Commissioned','availability')]:
  a['variant']=variant;g=gp_function(a);np.testing.assert_allclose(np.array(g(xy)).ravel(),[a['fields'][c][key][22][13] for c in ['camera_'+x for x in 'ABCDE']],atol=1e-12)
 a['variant']='Uniform';g=gp_function(a);np.testing.assert_array_equal(g([-8,-5]),g([9,4]))

def test_observation_interface_limits():
 p=NetworkPlanner(dict(seed=1200,arm='Uniform'));P=ca.DM(np.diag(np.linspace(.01,.2,13)));z=ca.DM.zeros(5);one=ca.DM.ones(5);a=p.artifact.copy();a['variant']='Commissioned'
 np.testing.assert_allclose(expected_update(P,p.model,z,True,a),P,atol=1e-12)
 hit=np.array(expected_update(P,p.model,one,True,a));a['variant']='IWAI';np.testing.assert_allclose(expected_update(P,p.model,one,True,a),hit,atol=1e-12)
 weak=np.array(expected_update(P,p.model,z,True,a));assert np.trace(weak[:2,:2])>np.trace(hit[:2,:2]);assert np.linalg.eigvalsh(weak).min()>0
