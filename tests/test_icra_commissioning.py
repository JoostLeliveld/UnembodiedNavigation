import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/icra_commissioning'))
from model import covariance, expected_posterior, update, ray_basis


def test_bias_is_not_covariance():
    e=np.array([[0.,1.],[1.,0.],[-1.,0.],[0.,-1.]])
    np.testing.assert_allclose(covariance(e),covariance(e+[100.,-50.]))


def test_miss_keeps_prior_and_certain_hit_matches_update():
    P=np.diag([.2,.5,.1]); R=np.eye(2)*.1
    np.testing.assert_allclose(expected_posterior(P,[(0,R)]),P)
    hit=update(np.zeros(3),P,np.zeros(2),R)[1]
    np.testing.assert_allclose(expected_posterior(P,[(1,R)]),hit)
    np.testing.assert_allclose(expected_posterior(P,[(.3,R)]),.3*hit+.7*P)
    optimistic=np.linalg.inv(np.linalg.inv(P[:2,:2])+.3*np.linalg.inv(R))
    assert np.linalg.eigvalsh(expected_posterior(P,[(.3,R)])[:2,:2]-optimistic).min()>0


def test_complementary_views_and_order_invariance():
    P=np.eye(3);a=np.diag([.001,.5]);b=np.diag([.5,.001])
    both=expected_posterior(P,[(1,a),(1,b)])
    np.testing.assert_allclose(both,expected_posterior(P,[(1,b),(1,a)]))
    assert np.trace(both[:2,:2])<np.trace(expected_posterior(P,[(1,a)])[:2,:2])


def test_observable_ray_basis_is_orthonormal():
    B=ray_basis(np.array([[1.,2.],[-3.,4.]]),np.array([0.,0.]))
    np.testing.assert_allclose(B.transpose(0,2,1)@B,np.tile(np.eye(2),(2,1,1)),atol=1e-15)
    with pytest.raises(ValueError):ray_basis(np.zeros(2),np.zeros(2))


def _minimal_replay(duplicate=False):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'experiments/icra_commissioning/replay.py'
    spec=importlib.util.spec_from_file_location('commissioning_replay_test',path)
    replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
    import aligned
    truth=aligned.TruthSeries([0.,1.],[0.,0.],[0.,0.],[0.,0.],'fixture')
    row=dict(camera='camera_A',t=.5,original_z=np.zeros(2),original_R=np.eye(2)*.01,batch='frame-1')
    return replay.run_filter({'task_start_pose':{'x':0.,'y':0.,'yaw':0.}},truth,
        {0.:np.zeros(2),1.:np.zeros(2)},[row,row] if duplicate else [row],{},'recorded',['camera_A'])


def test_each_camera_frame_updates_once():
    with pytest.raises(ValueError,match='duplicate physical camera update'):_minimal_replay(True)


def test_camera_event_does_not_change_evaluation_sample_population():
    score,records,events=_minimal_replay()
    assert len(events)==1 and score['updates']==1
    assert [r['t'] for r in records]==[0.,1.]
    assert np.isfinite(events[0]['innovation_covariance']).all()
