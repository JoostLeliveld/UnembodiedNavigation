"""Research camera/robot cascade with explicit persistent observation error.

Each image enters once as z = H_robot x + b_camera + b_shared + epsilon.
The augmented covariance retains robot/bias and inter-camera cross-covariances.
This module is independent of ROS; it is not enabled by existing launch files.
"""
from __future__ import annotations

import copy
from collections import deque
import numpy as np


def information_increment(prior_mean, prior_cov, posterior_mean, posterior_cov):
    """New Gaussian likelihood from one LOCAL linear-Gaussian tracker update.

    Both distributions must describe the same state at the same time, immediately
    before/after ONE observation. Prediction is not an observation. Keep the full
    information factor: its precision may be singular (e.g. position-only sensing).
    This is not a general decorrelator for arbitrary or nonlinear track histories.
    """
    a, b = np.asarray(prior_mean), np.asarray(posterior_mean)
    Ja, Jb = np.linalg.inv(prior_cov), np.linalg.inv(posterior_cov)
    return (Jb - Ja + (Jb - Ja).T) / 2, Jb @ b - Ja @ a


class AdaptiveMeasurementCovariance:
    """Learn camera R from a window of KF pre/post-fit residual pairs.

    For a linear KF, E[(z-Hx_plus)(z-Hx_minus)^T] = R. The estimate is
    symmetrized and projected onto a declared eigenvalue interval. It affects the
    next update only. The NN never predicts R.
    """
    def __init__(self, initial, window=50, min_samples=12,
                 min_variance=1e-5, max_variance=1.):
        self.value=np.asarray(initial,float)
        np.linalg.cholesky(self.value)
        self.samples=deque(maxlen=int(window)); self.min_samples=int(min_samples)
        self.bounds=(float(min_variance),float(max_variance))
        if window<min_samples or min_samples<2 or not 0<self.bounds[0]<self.bounds[1]:
            raise ValueError('invalid adaptive covariance settings')

    def observe(self,prefit,postfit):
        nu,mu=np.asarray(prefit,float),np.asarray(postfit,float)
        if nu.shape!=(2,) or mu.shape!=(2,) or not np.isfinite([*nu,*mu]).all():
            raise ValueError('finite 2D residuals required')
        self.samples.append(.5*(np.outer(mu,nu)+np.outer(nu,mu)))
        if len(self.samples)>=self.min_samples:
            C=np.mean(self.samples,axis=0)
            values,vectors=np.linalg.eigh((C+C.T)/2)
            values=np.clip(values,*self.bounds)
            self.value=vectors@np.diag(values)@vectors.T
        return self.value.copy()


class AdaptiveGaussianCameraFilter:
    """First stage of the cascade: position filtering plus residual-learned R.

    ``update`` returns the local posterior and the incremental information factor
    relative to its immediately preceding prior. A downstream Gaussian filter must
    consume that factor (or an explicitly correlation-aware joint state), rather
    than treating the local posterior as an independent measurement.
    """
    def __init__(self,mean,covariance,initial_R,**adaptation):
        self.mean=np.asarray(mean,float); self.covariance=np.asarray(covariance,float)
        self.R=AdaptiveMeasurementCovariance(initial_R,**adaptation)

    def predict(self,F,Q):
        F,Q=np.asarray(F,float),np.asarray(Q,float)
        self.mean=F@self.mean; self.covariance=F@self.covariance@F.T+Q

    def update(self,z,H):
        z,H=np.asarray(z,float),np.asarray(H,float)
        prior_mean,prior_cov=self.mean.copy(),self.covariance.copy()
        S=H@prior_cov@H.T+self.R.value; nu=z-H@prior_mean
        K=np.linalg.solve(S,H@prior_cov).T; A=np.eye(len(self.mean))-K@H
        self.mean=prior_mean+K@nu
        self.covariance=A@prior_cov@A.T+K@self.R.value@K.T
        mu=z-H@self.mean
        used_R=self.R.value.copy(); self.R.observe(nu,mu)
        J,eta=information_increment(prior_mean,prior_cov,self.mean,self.covariance)
        return dict(mean=self.mean.copy(),covariance=self.covariance.copy(),
                    R_used=used_R,R_next=self.R.value.copy(),prefit=nu,postfit=mu,
                    information=J,information_vector=eta)


class CorrelatedCameraFilter:
    """Joint robot state and stationary Gauss-Markov camera-bias states.

    Bias covariance is in world m², with an optional common world-frame component.
    R passed to observe is the FAST component only. Supplying total marginal R
    there and again as bias covariance would count measurement noise twice.
    Robot dynamics/Jacobians/Q are supplied by the caller, unchanged.
    """
    def __init__(self, mean, covariance, camera_ids, bias_covariances,
                 correlation_time_s, shared_covariance=None):
        self.camera_ids = tuple(camera_ids)
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError('camera IDs must be nonempty and unique')
        self.robot_dim = len(mean)
        self.tau = float(correlation_time_s)
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('positive finite correlation time required')
        self.blocks = [np.asarray(bias_covariances[c], float) for c in self.camera_ids]
        self.shared = shared_covariance is not None
        if self.shared:
            self.blocks.append(np.asarray(shared_covariance, float))
        n = self.robot_dim + 2 * len(self.blocks)
        self.mean = np.zeros(n)
        self.mean[:self.robot_dim] = mean
        self.covariance = np.zeros((n, n))
        self.covariance[:self.robot_dim, :self.robot_dim] = covariance
        for j, B in enumerate(self.blocks):
            if B.shape != (2, 2) or not np.isfinite(B).all() or not np.allclose(B, B.T) or np.linalg.eigvalsh(B).min() < 0:
                raise ValueError('bias covariance must be symmetric positive semidefinite')
            s = self._slice(j)
            self.covariance[s, s] = B

    def _slice(self, j):
        start = self.robot_dim + 2 * j
        return slice(start, start + 2)

    def predict(self, robot_mean, robot_F, robot_Q, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('nonnegative finite dt required')
        phi = np.exp(-dt / self.tau)
        F = np.eye(len(self.mean)) * phi
        Q = np.zeros_like(self.covariance)
        d = self.robot_dim
        F[:d, :d], Q[:d, :d] = robot_F, robot_Q
        self.mean[:d] = robot_mean
        self.mean[d:] *= phi
        for j, B in enumerate(self.blocks):
            s = self._slice(j)
            Q[s, s] = (1 - phi**2) * B
        self.covariance = F @ self.covariance @ F.T + Q

    def observation_matrix(self, camera_id, robot_H=None):
        H = np.zeros((2, len(self.mean)))
        H[:, :self.robot_dim] = (np.eye(self.robot_dim)[:2] if robot_H is None else robot_H)
        H[:, self._slice(self.camera_ids.index(camera_id))] = np.eye(2)
        if self.shared:
            H[:, self._slice(len(self.camera_ids))] = np.eye(2)
        return H

    def observe(self, camera_id, measurement, fast_covariance, robot_H=None):
        H = self.observation_matrix(camera_id, robot_H)
        R = np.asarray(fast_covariance, float)
        if R.shape != (2, 2) or not np.isfinite(R).all() or not np.allclose(R, R.T):
            raise ValueError('fast covariance must be finite symmetric 2x2')
        np.linalg.cholesky(R)
        innovation = np.asarray(measurement) - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = np.linalg.solve(S, H @ self.covariance).T
        self.mean += K @ innovation
        A = np.eye(len(self.mean)) - K @ H
        self.covariance = A @ self.covariance @ A.T + K @ R @ K.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        return innovation, S

    def forecast(self, outcomes):
        """Expected conditional covariance for explicitly weighted joint hit sets.

        outcomes = [(probability, [(camera_id, R_fast), ...]), ...]. A miss is [].
        Uses exactly observe's covariance update including persistent/shared bias.
        This is E[Cov(state | future data)], not covariance of the predictive
        mixture (which also includes variation of future posterior means).
        Forecast callers must marginalize unknown future image features upstream.
        """
        if not outcomes or any(not np.isfinite(p) or p < 0 for p, _ in outcomes) or not np.isclose(sum(p for p, _ in outcomes), 1):
            raise ValueError('outcome probabilities must sum to one')
        expected = np.zeros_like(self.covariance)
        for probability, cameras in outcomes:
            if len({c for c, _ in cameras}) != len(cameras):
                raise ValueError('duplicate camera in one future opportunity')
            branch = copy.deepcopy(self)
            for camera, R in cameras:
                H = branch.observation_matrix(camera)
                branch.observe(camera, H @ branch.mean, R)
            expected += probability * branch.covariance
        return expected
