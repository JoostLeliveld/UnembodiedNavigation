"""Structural checks use algebraic fixtures, not simulated experimental evidence."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/icra_commissioning'))
from commissioned_field import pose_embedding,integrated_sigmoid,CommissionedField
from reliability.observation_gp import ObservabilityGP,_fbag


def test_gp_latent_cache_preserves_legacy_predictions():
    X=np.array([[x,y] for x in range(3) for y in range(3)],float)
    labels=np.array([0,0,1,0,1,1,1,1,0],float)
    m=ObservabilityGP().fit(X,labels)
    q=np.array([[.25,.25],[1.2,1.4]])
    mu,sd=m.predict_latent(q)
    oldmu,oldsd=_fbag()._fit_predict_gp(m._Xb,m._yb,m._alpha,q,length_scale=m.length_scale)
    np.testing.assert_allclose(mu,oldmu);np.testing.assert_allclose(sd,oldsd)
    np.testing.assert_allclose(m.predict_proba(q),1/(1+np.exp(-oldmu)))
    assert np.all(sd>=0)


def test_link_integration_is_probability_not_measurement_variance():
    np.testing.assert_allclose(integrated_sigmoid(np.zeros(3),np.array([0.,1.,3.])),.5)
    assert integrated_sigmoid(np.array([2.]),np.array([2.]))[0] < 1/(1+np.exp(-2))


def fixture_field():
    X=np.zeros((4,3));X[:,0]=[0.,.1,.2,.3]
    hits=np.zeros((4,5),bool);hits[:2,:2]=True
    # Each camera needs a supported quality example, even if normally absent.
    hits[0,2:]=True
    R=np.tile(np.eye(2)*.01,(4,5,1,1))
    return CommissionedField(X,hits,R,{},R[0])


def test_joint_outcomes_and_information_approximation_are_distinct():
    f=fixture_field();P=np.diag([.1,.1,.03]);x=np.array([.1,0.,0.])
    branch,_=f.forecast(x,P,'local_joint','branch',True)
    information,_=f.forecast(x,P,'local_joint','information',True)
    assert np.linalg.eigvalsh(branch).min()>0
    assert np.linalg.eigvalsh(branch-information).min()>-1e-12
    independent,_=f.forecast(x,P,'local_independent','branch',True)
    assert not np.allclose(branch,independent)


def test_unsupported_queries_produce_no_camera_update():
    f=fixture_field();P=np.eye(3)
    post,meta=f.forecast([10.,10.,0.],P)
    np.testing.assert_array_equal(post,P);assert not meta['supported']
    with pytest.raises(ValueError):pose_embedding([[0.,0.,float('nan')]])
    with pytest.raises(ValueError):pose_embedding([[0.,0.,0.,.9]])


def test_forecast_uses_all_odometry_changes_including_a_turn():
    from field_driving import propagate
    from replay import unicycle_step,unicycle_jacobian,unicycle_process_noise
    x=np.array([0.,0.,0.]);P=np.diag([.1,.2,.03])
    tt=np.array([0.,.4,.9]);uu=np.array([[1.,0.],[0.,1.],[1.,0.]])
    actual,C=propagate(x,P,0.,1.4,tt,uu)
    expected=x.copy();R=P.copy()
    for u,dt in zip(uu,[.4,.5,.5]):
        F=unicycle_jacobian(expected,u,dt)
        Q=unicycle_process_noise(.01,.02,dt,theta=expected[2],v=u[0])
        R=F@R@F.T+Q;expected=unicycle_step(expected,u,dt)
    np.testing.assert_allclose(actual,expected);np.testing.assert_allclose(C,R)
    # Holding just the first velocity throughout this window loses the turn.
    assert not np.allclose(actual,unicycle_step(x,uu[0],1.4))
