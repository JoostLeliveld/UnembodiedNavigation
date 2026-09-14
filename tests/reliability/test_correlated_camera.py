import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/reliability'))
from reliability.correlated_camera import (CorrelatedCameraFilter, information_increment,
    AdaptiveMeasurementCovariance, AdaptiveGaussianCameraFilter)


def test_local_tracker_exports_only_new_information():
    rng = np.random.default_rng(7)
    m, P = np.zeros(4), np.eye(4)
    H, R = np.eye(4)[:2], np.array([[.1, .02], [.02, .2]])
    for _ in range(30):
        prior, C = m.copy(), P+.01*np.eye(4)
        z = rng.normal(size=2)
        K = np.linalg.solve(H@C@H.T+R, H@C).T
        m = prior+K@(z-H@prior)
        A = np.eye(4)-K@H
        P = A@C@A.T+K@R@K.T
        J, eta = information_increment(prior, C, m, P)
        np.testing.assert_allclose(J, H.T@np.linalg.solve(R, H), atol=1e-10)
        np.testing.assert_allclose(eta, H.T@np.linalg.solve(R, z), atol=1e-10)


def test_repeated_frames_cannot_remove_static_bias():
    B, R = np.eye(2)*.04, np.eye(2)*.001
    f = CorrelatedCameraFilter([0., 0.], np.eye(2), ['A'], {'A': B}, 2.)
    for _ in range(200):
        f.observe('A', [.2, -.1], R)
    # Closed form for N independent fast noises plus ONE shared bias draw.
    expected = np.linalg.inv(np.eye(2)+np.linalg.inv(B+R/200))
    np.testing.assert_allclose(f.covariance[:2, :2], expected, atol=1e-12)


def test_forecast_matches_execution_and_retains_shared_camera_bias():
    f = CorrelatedCameraFilter([0., 0.], np.eye(2), ['A','B'],
        {'A': np.eye(2)*.02, 'B': np.eye(2)*.03}, 2., np.eye(2)*.04)
    R = np.eye(2)*.01
    predicted = f.forecast([(1., [('A', R), ('B', R)])])
    f.observe('A', [.1, -.1], R)
    f.observe('B', [.2, -.2], R)
    np.testing.assert_allclose(predicted, f.covariance, atol=1e-12)
    assert np.linalg.eigvalsh(f.covariance[:2, :2]).min() > .04


def test_miss_forecast_and_bias_decay():
    f = CorrelatedCameraFilter([0., 0.], np.eye(2), ['A'], {'A': np.eye(2)*.04}, 2.)
    f.observe('A', [1., 0.], np.eye(2)*.01)
    C = f.covariance.copy()
    np.testing.assert_equal(f.forecast([(1., [])]), C)
    f.predict(f.mean[:2], np.eye(2), np.zeros((2,2)), 2.)
    np.testing.assert_allclose(f.covariance[:2, 2:], C[:2, 2:]*np.exp(-1))
    np.testing.assert_allclose(f.covariance[2:, 2:], C[2:, 2:]*np.exp(-2)+.04*np.eye(2)*(1-np.exp(-2)))


def test_joint_update_order_independent():
    def make():
        return CorrelatedCameraFilter([0.,0.], np.eye(2), ['A','B'],
            {'A': np.eye(2)*.02, 'B': np.eye(2)*.03}, 2., np.eye(2)*.04)
    a, b = make(), make()
    for camera in ['A','B']:
        a.observe(camera, [.1, .2], np.eye(2)*.01)
    for camera in ['B','A']:
        b.observe(camera, [.1, .2], np.eye(2)*.01)
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-12)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-12)


def test_prefit_postfit_pairs_recover_measurement_covariance():
    rng=np.random.default_rng(9)
    R=np.array([[.04,.01],[.01,.02]])
    prior=np.array([[.08,.005],[.005,.06]])
    S=prior+R; A=R@np.linalg.inv(S)
    learner=AdaptiveMeasurementCovariance(np.eye(2),window=5000,min_samples=5000)
    for nu in rng.multivariate_normal(np.zeros(2),S,5000):
        learner.observe(nu,A@nu)
    np.testing.assert_allclose(learner.value,R,atol=.0025)


def test_camera_filter_learns_R_after_update_and_exports_only_increment():
    R=np.array([[.04,.01],[.01,.02]])
    f=AdaptiveGaussianCameraFilter(np.zeros(2),np.eye(2),R,window=4,min_samples=4)
    packet=f.update(np.array([.2,-.1]),np.eye(2))
    np.testing.assert_allclose(packet['R_used'],R)
    np.testing.assert_allclose(packet['information'],np.linalg.inv(R),atol=1e-12)
    np.testing.assert_allclose(packet['information_vector'],np.linalg.solve(R,[.2,-.1]),atol=1e-12)

