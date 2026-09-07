"""ROS-free commissioned ground-reference observations and exact one-step forecasts.

All model inputs are observed geometry/box features. Reference poses appear only in fit.
Units: z,b metres; R square metres. Covariance and second moment are distinct artifacts.
"""
from itertools import product
import numpy as np
from sklearn.cluster import KMeans

FLOOR = 1e-6  # numerical variance floor, m^2, fixed for every arm


def covariance(e):
    e = np.asarray(e, float)
    if len(e) < 3:
        raise ValueError('at least three residuals required')
    centered = e - e.mean(axis=0)
    return centered.T @ centered / len(e) + FLOOR * np.eye(2)


def ray_basis(raw, camera_xy):
    delta = np.asarray(raw) - camera_xy
    d = np.linalg.norm(delta, axis=-1)
    if np.any(d < 1e-8):
        raise ValueError('undefined camera ray')
    u = delta / d[..., None]
    return np.stack((u, np.stack((-u[..., 1], u[..., 0]), axis=-1)), axis=-1)


class CameraModel:
    """Shared per-camera bias; covariance-only arms with identical acceptance.

    geometry: full covariance in the observable ray frame, in three range regimes.
    spatial: full world covariance in four clusters of corrected XY.
    confidence: three detector-score regimes (score is tested, not assumed accuracy).
    Sparse cells shrink toward the camera-level covariance, 20 pseudo-observations.
    Conditional residual means are saved, never silently subtracted in scoring.
    """
    def __init__(self, kind='constant', shrink=20):
        self.kind, self.shrink = kind, shrink
        self.scale, self.isotropic_shrink = 1., 0.

    def _features(self, rows):
        if self.kind == 'spatial':
            return np.asarray([r['z'] for r in rows])
        return np.asarray([[r['distance'] if self.kind == 'geometry' else r['confidence']]
                           for r in rows])

    def _cells(self, rows, fit=False):
        if self.kind in ('constant', 'diagonal', 'isotropic'):
            return np.zeros(len(rows), dtype=int)
        x = self._features(rows)
        if self.kind == 'spatial':
            if fit:
                self.cluster = KMeans(n_clusters=4, random_state=509, n_init=10).fit(x)
            return self.cluster.predict(x)
        if fit:
            self.edges = np.quantile(x[:, 0], [1/3, 2/3])
        return np.searchsorted(self.edges, x[:, 0])

    def fit(self, rows, bias=None):
        e = np.asarray([r['z'] - r['truth'] for r in rows])
        self.bias = e.mean(axis=0) if bias is None else np.asarray(bias)
        r = e - self.bias
        self.n = len(rows)
        self.support = np.quantile([q['distance'] for q in rows], [0, 1])
        cells = self._cells(rows, fit=True)
        if self.kind == 'geometry':
            B = np.asarray([q['basis'] for q in rows])
            r = np.einsum('nji,nj->ni', B, r)
        base = covariance(r)
        self.covs, self.means, self.counts = {}, {}, {}
        for cell in np.unique(cells):
            sample = r[cells == cell]
            n = len(sample)
            C = covariance(sample) if n >= 3 else base
            C = (n*C + self.shrink*base)/(n+self.shrink)
            if self.kind == 'diagonal': C = np.diag(np.diag(C))
            if self.kind == 'isotropic': C = np.eye(2)*np.trace(C)/2
            self.covs[int(cell)] = C
            self.means[int(cell)] = sample.mean(axis=0)
            self.counts[int(cell)] = n
        self.second_moment = e.T @ e / len(e)
        self.raw_covariance = covariance(e)
        return self

    def predict(self, rows):
        cells = self._cells(rows)
        R = np.asarray([self.covs[int(c)] for c in cells])
        if self.kind == 'geometry':
            B = np.asarray([r['basis'] for r in rows])
            R = B @ R @ B.transpose(0, 2, 1)
        isotropic = np.trace(R, axis1=1, axis2=2)[:, None, None]*np.eye(2)/2
        R = self.scale*((1-self.isotropic_shrink)*R+self.isotropic_shrink*isotropic)
        return np.asarray([r['z'] for r in rows])-self.bias, R


def update(m, P, z, R):
    """Linear position update, stable solve and Joseph form; returns pre-gate NIS."""
    H = np.eye(len(m))[:2]
    S = H @ P @ H.T + R
    nu = z - H @ m
    K = np.linalg.solve(S, H @ P).T
    A = np.eye(len(m)) - K @ H
    post = A @ P @ A.T + K @ R @ K.T
    return m + K @ nu, (post+post.T)/2, float(nu @ np.linalg.solve(S, nu))


def expected_posterior(P, qualities):
    """Enumerate independent camera hit/miss outcomes for ONE observation opportunity.

    qualities is [(q,R),...]. Miss means no update; absence carries no information.
    This is not a claim that real camera outcomes or successive errors are independent.
    """
    if len(qualities)>8: raise ValueError('enumeration limited to eight cameras')
    result = np.zeros_like(P)
    for mask in product((0, 1), repeat=len(qualities)):
        probability, post = 1., P.copy()
        for hit, (q, R) in zip(mask, qualities):
            if not 0 <= q <= 1: raise ValueError('invalid usability probability')
            probability *= q if hit else 1-q
            if hit: _, post, _ = update(np.zeros(len(P)), post, np.zeros(2), R)
        result += probability*post
    return result
