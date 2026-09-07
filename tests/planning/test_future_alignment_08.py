"""Camera cadence must not change prescribed motion or its no-camera covariance."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

path=Path(__file__).resolve().parents[2]/'experiments/icra_commissioning/future_rollout.py'
spec=importlib.util.spec_from_file_location('future_rollout_08_test',path)
future=importlib.util.module_from_spec(spec);spec.loader.exec_module(future)


def run(controls,times,cadence,*,start=0.,end=1.,forecast=None):
    grid=future.common_motion_grid(start,end,times,(.2,1.))
    return future.forecast_window(np.zeros(3),np.eye(3),start=start,end=end,
        control_times=times,controls=controls,motion_grid=grid,
        opportunities=future.observation_times(start,end,cadence),method='branch',
        forecast=forecast or (lambda m,P,method:(P,0.,0.)))


def test_changed_control_inside_slow_camera_interval_is_integrated_in_both_arms():
    times=[0.,.2,1.];controls=[[.22,0.],[0.,1.],[0.,1.]]
    fast,slow=run(controls,times,.2),run(controls,times,1.)
    np.testing.assert_allclose(fast['state'],[.044,0.,.8],atol=1e-14)
    np.testing.assert_array_equal(fast['state'],slow['state'])
    np.testing.assert_array_equal(fast['covariance'],slow['covariance'])


def test_constant_turning_motion_uses_identical_euler_and_Q_grid():
    times=[0.,1.];controls=[[.22,.6],[.22,.6]]
    fast,slow=run(controls,times,.2),run(controls,times,1.)
    np.testing.assert_array_equal(fast['state'],slow['state'])
    np.testing.assert_array_equal(fast['covariance'],slow['covariance'])
    assert fast['motion_steps']==slow['motion_steps']==5


def test_camera_updates_only_change_covariance_on_declared_opportunities():
    seen=[]
    def hits(m,P,method):seen.append(m.copy());return P*.9,1.,dict(nearest=0.,farthest=.2)
    fast=run([[.22,.6],[.22,.6]],[0.,1.],.2,forecast=hits)
    slow=run([[.22,.6],[.22,.6]],[0.,1.],1.,forecast=hits)
    np.testing.assert_array_equal(fast['state'],slow['state'])
    assert len(fast['opportunities'])==5 and len(slow['opportunities'])==1
    assert np.trace(fast['covariance'])<np.trace(slow['covariance'])


def test_anchor_and_nonmultiple_endpoint_are_preserved_exactly():
    start,end=.03,1.06
    result=run([[.22,0.],[0.,0.],[0.,0.]],[0.,.5,1.1],1.,start=start,end=end)
    assert result['start_s']==pytest.approx(start) and result['end_s']==pytest.approx(end)
    assert result['opportunities'][0]['time_s']==pytest.approx(1.03)
    assert len(result['opportunities'])==1
    assert result['state'][0]==pytest.approx(.22*(.5-start))


@pytest.mark.parametrize('times',[[.1,1.],[0.,.9],[0.,.5,.5,1.],[0.,np.nan,1.]])
def test_missing_or_invalid_control_support_is_refused(times):
    with pytest.raises(ValueError):future.common_motion_grid(0.,1.,times,[.2,1.])


def test_omitting_control_change_or_using_a_cadence_specific_grid_is_refused():
    kwargs=dict(state=np.zeros(3),covariance=np.eye(3),start=0.,end=1.,
        control_times=[0.,.2,1.],controls=[[.22,0.],[0.,1.],[0.,1.]],
        motion_grid=[0.,1.],opportunities=[1.],method='branch',forecast=lambda m,P,a:(P,0.,0.))
    with pytest.raises(ValueError,match='control change'):future.forecast_window(**kwargs)
    kwargs.update(motion_grid=[0.,.2,1.],opportunities=[.5,1.])
    with pytest.raises(ValueError,match='common motion grid'):future.forecast_window(**kwargs)


def test_diagnostic_entry_point_timestamps_support_and_output_contract(tmp_path):
    # Isolate the historical scripts' generic sibling module names from other
    # experiment suites; execute the actual entry point/helpers without fitting.
    script=r'''
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'experiments/icra_commissioning')
import future
out=Path(sys.argv[1])
time=np.array([0.,.11,.31,1.21,1.41])
assert future.window_indices(time,.05,1.)==(1,3)
assert future.window_indices(time,1.,1.) is None
cache=out/'trace.npz'
np.savez(cache,time=time,state=np.zeros((5,3)),covariance=np.tile(np.eye(3),(5,1,1)))
identities={};trace=future.load_trace(cache,identities)
assert len(identities)==1 and len(trace['time'])==5
np.savez(cache,time=time,state=np.zeros((5,3)),covariance=np.tile(np.diag([1.,1.,-.1]),(5,1,1)))
try:future.load_trace(cache,{})
except ValueError:pass
else:raise AssertionError('indefinite cached P accepted')
summary=future.summarize([])
assert all(r['windows']==0 and r['predicted_rms_cm'] is None for r in summary)
json.dumps(summary,allow_nan=False)
model=future.JointCommissioning.__new__(future.JointCommissioning)
poses=np.zeros((12,3));poses[1:,0]=3.
model.tree=cKDTree(future.pose_features(poses));model.outcomes=[[] for _ in poses]
P=np.eye(3)
post,q,support=model.forecast(np.zeros(3),P,'branch',return_support=True)
np.testing.assert_allclose(post,P,atol=1e-15)
assert q==0. and support['nearest_supported'] and support['contributors_within_threshold']==1
assert support['farthest_distance']==3.
model.outcomes=[[('camera_A',np.eye(2))] for _ in poses]
branch=model.forecast(np.zeros(3),P,'branch')[0]
information=model.forecast(np.zeros(3),P,'information')[0]
np.testing.assert_allclose(branch,information,atol=1e-14)
before=set(out.iterdir());sys.argv=['future.py','--output-dir',str(out)]
try:future.main()
except ValueError as error:assert 'already exists' in str(error)
else:raise AssertionError('existing output accepted')
assert set(out.iterdir())==before
print('entry point and diagnostic contracts passed')
'''
    env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/'mpl'))
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],cwd=path.parents[2],
                          env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
