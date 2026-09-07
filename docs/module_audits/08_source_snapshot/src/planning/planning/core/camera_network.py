"""A camera network for the IWAI objective and a separate hit/miss reference.

The planner proxy blends precisions using an expected detector SCORE. That proxy
is not measurement noise, detection probability, or an exact expected posterior.
Actual camera measurements remain metric reference-XY observations. A fixed camera
chart maps the network's metric proxy to the existing IWAI cost coordinates; the
goal preference and objective weights therefore do not change with camera count.
"""
from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _spd(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape[-2:] != (2, 2) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite 2x2 matrices')
    if not np.allclose(value, value.swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f'{name} must be symmetric')
    if np.linalg.eigvalsh(value).min() <= 0:
        raise ValueError(f'{name} must be positive definite')
    return value


def projection_jacobian(H, state):
    """Original-image pixels per metre for the fixed cost chart, not a detector."""
    H = np.asarray(H, dtype=float).reshape(3, 3)
    point = np.r_[np.asarray(state, dtype=float)[:2], 1.]
    projected = H @ point
    if abs(projected[2]) < 1e-8:
        raise ValueError('network cost chart is singular at this query')
    return (H[:2, :2]*projected[2] - projected[:2, None]*H[2, :2])/projected[2]**2


class CameraNetworkModel:
    """Frozen per-camera score and availability grids, with full metric quality.

    NPZ array layout: fields [camera,y,x], covariances [camera,2,2]. Bilinear
    interpolation matches the CasADi path. Outside the commissioned grid the
    score and usable-detection probability are zero. No image or GT is queried.
    """
    def __init__(self, artifact_path, cameras=None):
        self.path = Path(artifact_path).expanduser().resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata_json'].item()))
            if self.metadata.get('schema') != 'camera_network.iwai.v1':
                raise ValueError('unsupported camera-network artifact schema')
            if self.metadata.get('reference') != 'robot_ground_reference_xy':
                raise ValueError('network must measure the declared ground reference')
            if self.metadata.get('frame') != 'map_bev' or self.metadata.get('covariance_units') != 'm2':
                raise ValueError('network covariance must be map_bev square metres')
            if self.metadata.get('score_target') != 'detector_score_with_miss_zero':
                raise ValueError('the IWAI proxy requires an explicitly labelled detector-score field')
            if self.metadata.get('availability_target') != 'valid_detection_finite_ground_projection':
                raise ValueError('availability must describe the declared pre-gate detection event')
            ids = tuple(str(c) for c in data['camera_ids'])
            if not ids or len(set(ids)) != len(ids):
                raise ValueError('unique camera IDs required')
            self.camera_ids = ids if cameras is None else tuple(cameras)
            if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
                raise ValueError('camera mask must be nonempty and unique')
            if not set(self.camera_ids) <= set(ids):
                raise ValueError('camera mask contains an unknown camera')
            indices = [ids.index(c) for c in self.camera_ids]
            self.xs, self.ys = np.asarray(data['xs'], float), np.asarray(data['ys'], float)
            for axis in (self.xs, self.ys):
                if axis.ndim != 1 or len(axis)<2 or not np.isfinite(axis).all() or not (np.diff(axis)>0).all():
                    raise ValueError('network axes must be finite and strictly increasing')
            self.fields = {}
            for key in ('score', 'availability'):
                grid = np.asarray(data[key], float)
                if grid.shape != (len(ids), len(self.ys), len(self.xs)):
                    raise ValueError(f'{key} camera/y/x dimensions differ')
                if not np.isfinite(grid).all() or np.any((grid<0)|(grid>1)):
                    raise ValueError(f'{key} must be in [0,1]')
                self.fields[key] = grid[indices]
            self.R = _spd(data['R_cond_m2'], 'conditional R')[indices]
            self.R_miss = _spd(data['R_miss_proxy_m2'], 'miss proxy R')[indices]
            if self.R.shape != (len(indices), 2, 2) or self.R_miss.shape != self.R.shape:
                raise ValueError('one covariance per selected camera is required')
            if np.linalg.eigvalsh(self.R_miss-self.R).min() < -1e-12:
                raise ValueError('miss proxy cannot be more precise than conditional R')
        self.precision = np.linalg.solve(self.R, np.broadcast_to(np.eye(2), self.R.shape))
        self.miss_precision = np.linalg.solve(self.R_miss, np.broadcast_to(np.eye(2), self.R.shape))
        self.interpolators = {key: [RegularGridInterpolator((self.ys,self.xs), grid,
            bounds_error=False, fill_value=0.) for grid in maps] for key,maps in self.fields.items()}

    @property
    def signature(self):
        return ('camera_network.iwai.v1', self.sha256, self.camera_ids)

    def query(self, state):
        state = np.asarray(state, float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError('network query requires finite predicted [x,y,yaw]')
        return {key:np.array([float(f([state[1],state[0]]).item()) for f in fs])
                for key,fs in self.interpolators.items()}

    def query_belief(self, state, P, kappa=1.):
        state, P = np.asarray(state,float), np.asarray(P,float)
        self.query(state)  # validate the public input before deriving sigma points
        if P.shape != (3,3) or not np.isfinite(P).all():
            raise ValueError('network query covariance must be finite 3x3')
        kappa = max(float(kappa), 1e-6)
        spread = np.sqrt(2+kappa)*np.linalg.cholesky((P[:2,:2]+P[:2,:2].T)/2+1e-9*np.eye(2))
        offsets = [np.zeros(2),spread[:,0],-spread[:,0],spread[:,1],-spread[:,1]]
        weights = [kappa/(2+kappa)]+[1/(2*(2+kappa))]*4
        out = {key:np.zeros(len(self.camera_ids)) for key in self.fields}
        for offset, weight in zip(offsets,weights):
            query = state.copy(); query[:2] += offset
            for key,value in self.query(query).items(): out[key] += weight*value
        return out

    def proxy_ground_covariance(self, score):
        """Designed IWAI precision blend; finite at a miss, never a runtime R."""
        score = np.asarray(score,float)
        if score.shape != (len(self.camera_ids),) or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
            raise ValueError('one score in [0,1] per camera is required')
        info = (score[:,None,None]*self.precision + (1-score[:,None,None])*self.miss_precision).sum(axis=0)
        return np.linalg.solve(info,np.eye(2))

    def planning_diagnostics(self, state, P, H, kappa=1.):
        query = self.query_belief(state,P,kappa)
        ground = self.proxy_ground_covariance(query['score'])
        J = projection_jacobian(H,state)
        uv = J @ ground @ J.T
        return dict(p_vis=float(query['score'].mean()), p_vis_eff=float(query['score'].mean()),
            R_plan=uv, r_plan_u_std=float(np.sqrt(uv[0,0])), r_plan_v_std=float(np.sqrt(uv[1,1])),
            network_score=query['score'], network_availability=query['availability'],
            network_R_proxy_m2=ground, network_artifact_sha256=self.sha256,
            p_vis_semantics='mean_expected_detector_score_not_detection_probability')

    def forecast_posterior(self, state, P, mode='branch'):
        """One-step availability reference, conditional independent camera errors.

        Uses q and R_cond, never the score or R_miss_proxy. Branch averaging is
        exact for this one-step linear model at the supplied predicted pose.
        Missingness is not treated as evidence about state. Temporal independence
        and the approximation to the actual robust runtime fusion are unvalidated.
        """
        q = self.query(state)['availability']
        P = np.asarray(P,float)
        if P.shape != (3,3) or not np.isfinite(P).all() or not np.allclose(P,P.T) or np.linalg.eigvalsh(P).min()<=0:
            raise ValueError('forecast prior must be SPD 3x3')
        if mode == 'information':
            info = np.linalg.solve(P,np.eye(3))
            info[:2,:2] += (q[:,None,None]*self.precision).sum(axis=0)
            return np.linalg.solve(info,np.eye(3))
        if mode != 'branch': raise ValueError('forecast mode must be branch or information')
        if len(q)>5: raise ValueError('exact reference is bounded to at most five cameras')
        out = np.zeros_like(P); H = np.eye(3)[:2]
        for mask in product((0,1),repeat=len(q)):
            weight = float(np.prod([q[i] if hit else 1-q[i] for i,hit in enumerate(mask)]))
            if weight == 0: continue
            post = P.copy()
            for hit,R in zip(mask,self.R):
                if not hit: continue
                K = np.linalg.solve(H @ post @ H.T+R,H @ post).T
                A = np.eye(3)-K @ H
                post = A @ post @ A.T+K @ R @ K.T
            out += weight*post
        return (out+out.T)/2

    def make_proxy_covariance_casadi(self, H, kappa=1.):
        import casadi as ca
        from planning.core.casadi_efe import _xy_visibility_sigma_points_ca
        interpolators=[]
        for i,grid in enumerate(self.fields['score']):
            interpolators.append(ca.interpolant(f'network_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(),self.ys.tolist()],grid.T.ravel(order='F').tolist()))
        H = ca.DM(np.asarray(H,float))
        def evaluate(m,P):
            points,weights = _xy_visibility_sigma_points_ca(m[:2],P[:2,:2],kappa)
            scores=[]
            for interp in interpolators:
                total=0
                for xy,weight in zip(points,weights):
                    inside=ca.logic_and(ca.logic_and(xy[0]>=self.xs[0],xy[0]<=self.xs[-1]),
                                        ca.logic_and(xy[1]>=self.ys[0],xy[1]<=self.ys[-1]))
                    bounded=ca.vertcat(ca.fmin(ca.fmax(xy[0],self.xs[0]),self.xs[-1]),
                                       ca.fmin(ca.fmax(xy[1],self.ys[0]),self.ys[-1]))
                    total += weight*ca.if_else(inside,interp(bounded),0.)
                scores.append(total)
            info=ca.DM.zeros(2,2)
            for score,visible,miss in zip(scores,self.precision,self.miss_precision):
                info += score*ca.DM(visible)+(1-score)*ca.DM(miss)
            ground=ca.solve(info,ca.DM.eye(2))
            projected=H @ ca.vertcat(m[0],m[1],1.)
            J=(H[:2,:2]*projected[2]-projected[:2] @ H[2,:2])/projected[2]**2
            return J @ ground @ J.T
        return evaluate
