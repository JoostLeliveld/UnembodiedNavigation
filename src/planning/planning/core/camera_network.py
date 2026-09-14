"""Camera-network models for legacy and metric planning objectives.

The planner proxy blends precisions using an expected detector SCORE. That proxy
is not measurement noise, detection probability, or an exact expected posterior.
Actual camera measurements remain metric reference-XY observations. The legacy
objective maps a metric score proxy through one fixed camera chart. The metric
objective instead enumerates usable-detection events and updates the predicted
ground-plane belief directly.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import MappingProxyType
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from planning.core.plan_validation import immutable_array, validate_covariance


@lru_cache(maxsize=5)
def _binary_report_masks(camera_count: int) -> np.ndarray:
    """All hit/miss report branches as a reusable numeric matrix."""
    masks = np.asarray(list(product((0, 1), repeat=camera_count)), dtype=float)
    masks.setflags(write=False)
    return masks


def _source_hashes(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('network source_hashes must be a nonempty mapping')
    for path, digest in value.items():
        if (not isinstance(path, str) or not path.strip() or Path(path).is_absolute()
                or '..' in Path(path).parts or not isinstance(digest, str)
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('network source_hashes requires relative names and SHA-256 digests')
    return dict(value)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


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
    H = np.asarray(H, dtype=float)
    state = np.asarray(state, dtype=float)
    if H.shape != (3,3) or not np.isfinite(H).all() or state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError('cost chart requires finite 3x3 homography and [x,y,yaw]')
    if np.linalg.matrix_rank(H) != 3:
        raise ValueError('cost chart homography must be nonsingular')
    point = np.r_[state[:2], 1.]
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
    def __setattr__(self, name, value):
        if getattr(self, '_loaded', False) and not name.startswith('_'):
            raise AttributeError('loaded camera network is immutable; construct a new model')
        object.__setattr__(self, name, value)

    def __init__(self, artifact_path, cameras=None, *, expected_sha256=None,
                 expected_source_hashes=None, expected_camera_ids=None):
        self.path = Path(artifact_path).expanduser().resolve()
        artifact_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('camera-network artifact SHA-256 mismatch')
        # Hash and parse one immutable byte snapshot; a replaced file cannot race
        # a separate np.load(path) into having the old identity and new arrays.
        with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata_json'].item()))
            if not isinstance(self.metadata, dict):
                raise ValueError('network metadata must be an object')
            sources = _source_hashes(self.metadata.get('source_hashes'))
            if expected_source_hashes is not None and sources != _source_hashes(expected_source_hashes):
                raise ValueError('camera-network source provenance differs from expected manifest')
            schema = self.metadata.get('schema')
            if schema not in ('camera_network.iwai.v1', 'camera_network.thesis_stage09.v1',
                              'camera_network.thesis_stage09.v2'):
                raise ValueError('unsupported camera-network artifact schema')
            if self.metadata.get('reference') != 'robot_ground_reference_xy':
                raise ValueError('network must measure the declared ground reference')
            if self.metadata.get('frame') != 'map_bev' or self.metadata.get('covariance_units') != 'm2':
                raise ValueError('network covariance must be map_bev square metres')
            if schema == 'camera_network.iwai.v1':
                if self.metadata.get('score_target') != 'detector_score_with_miss_zero':
                    raise ValueError('the IWAI proxy requires an explicitly labelled detector-score field')
                if self.metadata.get('availability_target') != 'valid_detection_finite_ground_projection':
                    raise ValueError('availability must describe the declared pre-gate detection event')
            else:
                if self.metadata.get('score_target') != 'unused_in_metric_expected_belief':
                    raise ValueError('Stage-09 score must be explicitly marked unused')
                if self.metadata.get('availability_target') != 'stage06_admitted_localization_measurement':
                    raise ValueError('Stage-09 availability must target complete-chain admission')
            raw_ids = np.asarray(data['camera_ids'])
            if raw_ids.ndim != 1 or raw_ids.dtype.kind not in ('U','S'):
                raise ValueError('camera IDs must be a one-dimensional string axis')
            ids = tuple(raw_ids.astype(str))
            if not ids or len(set(ids)) != len(ids) or any(not c or c != c.strip() for c in ids):
                raise ValueError('unique camera IDs required')
            if expected_camera_ids is not None:
                expected = tuple(expected_camera_ids)
                if not expected or len(set(expected)) != len(expected) or set(expected) != set(ids):
                    raise ValueError('camera-network roster differs from declared future cameras')
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
                self.fields[key] = immutable_array(grid[indices])
            full_R = _spd(data['R_cond_m2'], 'conditional R')
            full_miss = _spd(data['R_miss_proxy_m2'], 'miss proxy R')
            if full_R.shape != (len(ids), 2, 2) or full_miss.shape != full_R.shape:
                raise ValueError('one covariance per artifact camera is required before masking')
            self.R = immutable_array(full_R[indices])
            self.R_miss = immutable_array(full_miss[indices])
            if np.linalg.eigvalsh(self.R_miss-self.R).min() < -1e-12:
                raise ValueError('miss proxy cannot be more precise than conditional R')
            self.dynamic_R = schema == 'camera_network.thesis_stage09.v2'
            if self.dynamic_R:
                self.headings = np.asarray(data['headings'], dtype=float)
                if (self.headings.ndim != 1 or len(self.headings) < 3
                        or not np.isfinite(self.headings).all()
                        or not (np.diff(self.headings) > 0).all()
                        or abs(self.headings[0]) > 1e-12
                        or abs(self.headings[-1] - 2.0 * np.pi) > 1e-10):
                    raise ValueError('dynamic covariance headings must span [0,2pi]')
                full_field = _spd(data['R_cond_field_m2'], 'conditional R field')
                expected = (len(ids), len(self.headings), len(self.ys), len(self.xs), 2, 2)
                if full_field.shape != expected:
                    raise ValueError(f'conditional R field shape differs: {full_field.shape} != {expected}')
                self.R_field = immutable_array(full_field[indices])
        self.xs, self.ys = immutable_array(self.xs), immutable_array(self.ys)
        self.fields = MappingProxyType(self.fields)
        self.metadata = _freeze(self.metadata)
        self.precision = immutable_array(np.linalg.solve(self.R, np.broadcast_to(np.eye(2), self.R.shape)))
        self.miss_precision = immutable_array(np.linalg.solve(self.R_miss, np.broadcast_to(np.eye(2), self.R.shape)))
        self.interpolators = {key: [RegularGridInterpolator((self.ys,self.xs), grid,
            bounds_error=False, fill_value=0.) for grid in maps] for key,maps in self.fields.items()}
        self.interpolators = MappingProxyType({key: tuple(value) for key,value in self.interpolators.items()})
        if self.dynamic_R:
            self.R_interpolators = tuple(
                tuple(tuple(
                    RegularGridInterpolator(
                        (self.headings, self.ys, self.xs), field[..., row, column],
                        bounds_error=False, fill_value=None,
                    )
                    for column in range(2)) for row in range(2))
                for field in self.R_field
            )
        else:
            self.R_interpolators = ()
        self._loaded = True

    @property
    def signature(self):
        return (self.metadata['schema'], self.sha256, self.camera_ids)

    def query(self, state):
        state = np.asarray(state, float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError('network query requires finite predicted [x,y,yaw]')
        out = {key:np.array([float(f([state[1],state[0]]).item()) for f in fs])
               for key,fs in self.interpolators.items()}
        if self.dynamic_R:
            yaw = float(state[2] % (2.0 * np.pi))
            matrices = []
            for camera in self.R_interpolators:
                matrix = np.asarray([
                    [float(camera[row][column]([yaw, state[1], state[0]]).item())
                     for column in range(2)] for row in range(2)
                ])
                matrices.append((matrix + matrix.T) / 2.0)
            out['conditional_covariance'] = _spd(
                np.stack(matrices), 'interpolated conditional R'
            )
        else:
            out['conditional_covariance'] = self.R
        return out

    def query_belief(self, state, P, kappa=1.):
        state, P = np.asarray(state,float), np.asarray(P,float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError('network query requires finite predicted [x,y,yaw]')
        P = validate_covariance(P)
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError('sigma-point kappa must be finite and positive')
        kappa = max(float(kappa), 1e-6)
        spread = np.sqrt(2+kappa)*np.linalg.cholesky((P[:2,:2]+P[:2,:2].T)/2+1e-9*np.eye(2))
        offsets = [np.zeros(2),spread[:,0],-spread[:,0],spread[:,1],-spread[:,1]]
        weights = np.asarray([kappa/(2+kappa)]+[1/(2*(2+kappa))]*4, dtype=float)
        # RegularGridInterpolator accepts a point batch.  Query only the score
        # and availability fields needed by the sigma-point average; calling
        # ``query`` here also interpolated every dynamic covariance element at
        # every sigma point even though those values were discarded.
        points_xy = state[None, :2] + np.asarray(offsets, dtype=float)
        points_yx = points_xy[:, ::-1]
        out = {
            key: np.asarray([
                float(np.dot(weights, interpolator(points_yx)))
                for interpolator in interpolators
            ], dtype=float)
            for key, interpolators in self.interpolators.items()
        }
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
        query = self.query(state)
        q = query['availability']
        conditional_R = query['conditional_covariance']
        P = validate_covariance(P, positive_definite=(mode == 'information'), name='forecast prior')
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
            for hit,R in zip(mask,conditional_R):
                if not hit: continue
                K = np.linalg.solve(H @ post @ H.T+R,H @ post).T
                A = np.eye(3)-K @ H
                post = A @ post @ A.T+K @ R @ K.T
            out += weight*post
        return (out+out.T)/2

    def expected_belief(self, state, P, kappa=1., opportunities=1):
        """Expected posterior covariance and ambiguity after camera opportunities.

        Availability is averaged over the predicted XY belief. Camera hit events
        and conditional residuals are independent in this planning approximation.
        A miss performs no update. The robust runtime fusion remains a separate
        estimator and must not be inferred from this forecast.

        The returned ambiguity is the expected posterior entropy minus the prior
        entropy, i.e. the negative expected information gain. Reporting the
        posterior entropy alone adds ``0.5 d log(2 pi e)`` at every step, a
        constant that does not depend on what the cameras see, so summing it over
        a route charges for the number of steps. Differencing against the prior
        removes it: a step with no expected report contributes exactly zero and a
        step in a well-observed lane contributes a negative amount. Routes of
        different length are then comparable, which is what route selection needs.
        """
        if (isinstance(opportunities, bool) or int(opportunities) != opportunities
                or int(opportunities) < 1):
            raise ValueError('opportunities must be a positive integer')
        P = validate_covariance(P, positive_definite=True, name='network forecast prior')
        conditional_R = self.query(state)['conditional_covariance']
        if len(conditional_R) > 5:
            raise ValueError('exact expected-belief forecast is bounded to at most five cameras')
        # For each fixed hit subset, sequential linear-Gaussian updates equal
        # ``inv(inv(P) + sum(H.T @ inv(R_i) @ H))``.  Evaluate all 2^N subsets
        # as one NumPy batch.  This preserves exact branch enumeration while
        # avoiding millions of tiny Python-level solves during route selection.
        masks = _binary_report_masks(len(conditional_R))
        measurement_information = np.zeros((len(conditional_R), 3, 3), dtype=float)
        measurement_information[:, :2, :2] = np.linalg.inv(conditional_R)
        expected_entropy = np.nan
        for _ in range(int(opportunities)):
            # Belief-averaged availability changes as the covariance contracts,
            # so each camera opportunity must query it from the current prior.
            q = np.clip(self.query_belief(state, P, kappa)['availability'], 0., 1.)
            weights = np.prod(
                np.where(masks != 0.0, q[None, :], 1.0 - q[None, :]), axis=1)
            branch_information = (
                np.linalg.inv(P)[None, :, :]
                + np.einsum('bc,cij->bij', masks, measurement_information)
            )
            posts = np.linalg.inv(branch_information)
            posts = 0.5 * (posts + posts.swapaxes(-1, -2))
            signs, logdets = np.linalg.slogdet(posts[:, :2, :2])
            if np.any(signs <= 0):
                raise ValueError('network posterior position covariance must be positive definite')
            P = np.einsum('b,bij->ij', weights, posts)
            P = (P + P.T) / 2.
            expected_entropy = float(np.sum(
                weights * 0.5 * (2.0 * np.log(2.0 * np.pi * np.e) + logdets)
            ))
        return P, expected_entropy

    def effective_observation_covariance(self, state, P, kappa=1., *, no_report_var=0.):
        """Availability-weighted observation covariance at the predicted pose.

        Kouw (IWAI 2024, Thm 1) shows that under a first-order extended
        transform the ambiguity term reduces to ``0.5 log|R|``: the state
        covariance cancels, so with a single fixed sensor the term is constant
        over states and induces no preference. Here ``R`` is not fixed. Each
        camera contributes its commissioned ``R_i(p, psi)`` weighted by the
        probability ``q_i(p)`` that it returns a usable measurement, so the
        effective precision is ``sum_i q_i R_i^-1``. A camera that is not
        expected to report adds no precision. The result is a spatial field, so
        the first-order ambiguity term does vary over the workspace.
        """
        conditional_R = self.query(state)['conditional_covariance']
        availability = np.clip(
            self.query_belief(state, P, kappa)['availability'], 0., 1.)
        precision = np.zeros((2, 2), dtype=float)
        for weight, R in zip(availability, conditional_R):
            precision += float(weight) * np.linalg.inv(R)
        # Floor the precision so a fully unobserved position stays finite.
        # A step with no reading is not infinitely uncertain: the belief grows
        # by exactly one step of process noise. Treat it as a reading of that
        # quality, so the floor is the motion model rather than a chosen
        # constant. ``process_noise_step_var`` is set by the planner.
        floor_var = float(no_report_var)
        if floor_var > 0.:
            precision += (1.0 / floor_var) * np.eye(2)
        return np.linalg.inv(precision)

    def make_expected_belief_casadi(self, kappa=1., opportunities=1):
        """Return the differentiable counterpart of :meth:`expected_belief`."""
        import casadi as ca
        from planning.core.casadi_efe import (
            _differential_entropy_ca, _xy_visibility_sigma_points_ca,
        )
        if (isinstance(opportunities, bool) or int(opportunities) != opportunities
                or int(opportunities) < 1):
            raise ValueError('opportunities must be a positive integer')
        if len(self.camera_ids) > 5:
            raise ValueError('exact expected-belief forecast is bounded to at most five cameras')
        interpolators = [
            ca.interpolant(
                f'network_availability_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(), self.ys.tolist()], grid.T.ravel(order='F').tolist(),
            )
            for i, grid in enumerate(self.fields['availability'])
        ]
        covariance_interpolators = None
        if self.dynamic_R:
            covariance_interpolators = [
                [[
                    ca.interpolant(
                        f'network_R_{self.sha256[:10]}_{camera}_{row}_{column}', 'linear',
                        [self.xs.tolist(), self.ys.tolist(), self.headings.tolist()],
                        np.transpose(self.R_field[camera, ..., row, column], (2, 1, 0))
                        .ravel(order='F').tolist(),
                    )
                    for column in range(2)] for row in range(2)]
                for camera in range(len(self.camera_ids))
            ]
        H = ca.DM(np.eye(3)[:2])

        def evaluate(m, P):
            conditional_R = []
            if covariance_interpolators is None:
                conditional_R = [ca.DM(value) for value in self.R]
            else:
                yaw = ca.atan2(ca.sin(m[2]), ca.cos(m[2]))
                yaw = ca.if_else(yaw < 0.0, yaw + 2.0 * np.pi, yaw)
                bounded = ca.vertcat(
                    ca.fmin(ca.fmax(m[0], self.xs[0]), self.xs[-1]),
                    ca.fmin(ca.fmax(m[1], self.ys[0]), self.ys[-1]),
                    yaw,
                )
                for camera in covariance_interpolators:
                    matrix = ca.vertcat(
                        ca.horzcat(camera[0][0](bounded), camera[0][1](bounded)),
                        ca.horzcat(camera[1][0](bounded), camera[1][1](bounded)),
                    )
                    conditional_R.append(.5 * (matrix + matrix.T))

            prior = .5 * (P + P.T)
            expected_entropy = 0.
            for _ in range(int(opportunities)):
                # Match the NumPy reference: re-average q over the contracted
                # belief before every opportunity within the global step.
                points, weights = _xy_visibility_sigma_points_ca(
                    m[:2], prior[:2, :2], kappa)
                availability = []
                for interp in interpolators:
                    total = 0.
                    for xy, weight in zip(points, weights):
                        inside = ca.logic_and(
                            ca.logic_and(xy[0] >= self.xs[0], xy[0] <= self.xs[-1]),
                            ca.logic_and(xy[1] >= self.ys[0], xy[1] <= self.ys[-1]),
                        )
                        bounded_xy = ca.vertcat(
                            ca.fmin(ca.fmax(xy[0], self.xs[0]), self.xs[-1]),
                            ca.fmin(ca.fmax(xy[1], self.ys[0]), self.ys[-1]),
                        )
                        total += float(weight) * ca.if_else(
                            inside, interp(bounded_xy), 0.)
                    availability.append(ca.fmin(ca.fmax(total, 0.), 1.))
                expected = ca.MX.zeros(3, 3)
                expected_entropy = 0.
                for mask in product((0, 1), repeat=len(availability)):
                    branch_weight = 1.
                    post = prior
                    for i, (hit, R_ca) in enumerate(zip(mask, conditional_R)):
                        q = availability[i]
                        branch_weight *= q if hit else 1. - q
                        if not hit:
                            continue
                        innovation = H @ post @ H.T + R_ca + 1e-9 * ca.DM.eye(2)
                        K = ca.solve(innovation, H @ post).T
                        A = ca.DM.eye(3) - K @ H
                        post = A @ post @ A.T + K @ R_ca @ K.T
                        post = .5 * (post + post.T)
                    expected += branch_weight * post
                    expected_entropy += branch_weight * _differential_entropy_ca(post[:2, :2])
                prior = .5 * (expected + expected.T)
            return .5 * (expected + expected.T), expected_entropy

        return evaluate

    def make_effective_covariance_casadi(self, kappa=1., *, no_report_var=0.):
        """Differentiable counterpart of :meth:`effective_observation_covariance`."""
        import casadi as ca
        from planning.core.casadi_efe import _xy_visibility_sigma_points_ca
        availability_interpolators = [
            ca.interpolant(
                f'network_reff_q_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(), self.ys.tolist()], grid.T.ravel(order='F').tolist(),
            )
            for i, grid in enumerate(self.fields['availability'])
        ]
        covariance_interpolators = None
        if self.dynamic_R:
            covariance_interpolators = [
                [[
                    ca.interpolant(
                        f'network_reff_R_{self.sha256[:10]}_{camera}_{row}_{column}',
                        'linear',
                        [self.xs.tolist(), self.ys.tolist(), self.headings.tolist()],
                        np.transpose(self.R_field[camera, ..., row, column], (2, 1, 0))
                        .ravel(order='F').tolist(),
                    )
                    for column in range(2)] for row in range(2)]
                for camera in range(len(self.camera_ids))
            ]
        floor_var = float(no_report_var)

        def evaluate(m, P):
            if covariance_interpolators is None:
                conditional_R = [ca.DM(value) for value in self.R]
            else:
                yaw = ca.atan2(ca.sin(m[2]), ca.cos(m[2]))
                yaw = ca.if_else(yaw < 0.0, yaw + 2.0 * np.pi, yaw)
                bounded = ca.vertcat(
                    ca.fmin(ca.fmax(m[0], self.xs[0]), self.xs[-1]),
                    ca.fmin(ca.fmax(m[1], self.ys[0]), self.ys[-1]),
                    yaw,
                )
                conditional_R = []
                for camera in covariance_interpolators:
                    matrix = ca.vertcat(
                        ca.horzcat(camera[0][0](bounded), camera[0][1](bounded)),
                        ca.horzcat(camera[1][0](bounded), camera[1][1](bounded)),
                    )
                    conditional_R.append(.5 * (matrix + matrix.T))
            points, weights = _xy_visibility_sigma_points_ca(m[:2], P[:2, :2], kappa)
            precision = ((1.0 / floor_var) if floor_var > 0. else 0.) * ca.DM.eye(2)
            for interp, R_ca in zip(availability_interpolators, conditional_R):
                total = 0.
                for xy, weight in zip(points, weights):
                    inside = ca.logic_and(
                        ca.logic_and(xy[0] >= self.xs[0], xy[0] <= self.xs[-1]),
                        ca.logic_and(xy[1] >= self.ys[0], xy[1] <= self.ys[-1]),
                    )
                    bounded_xy = ca.vertcat(
                        ca.fmin(ca.fmax(xy[0], self.xs[0]), self.xs[-1]),
                        ca.fmin(ca.fmax(xy[1], self.ys[0]), self.ys[-1]),
                    )
                    total += float(weight) * ca.if_else(inside, interp(bounded_xy), 0.)
                q = ca.fmin(ca.fmax(total, 0.), 1.)
                precision = precision + q * ca.inv(R_ca + 1e-12 * ca.DM.eye(2))
            return ca.inv(precision)

        return evaluate

    def make_proxy_covariance_casadi(self, H, kappa=1.):
        import casadi as ca
        from planning.core.casadi_efe import _xy_visibility_sigma_points_ca
        interpolators=[]
        for i,grid in enumerate(self.fields['score']):
            interpolators.append(ca.interpolant(f'network_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(),self.ys.tolist()],grid.T.ravel(order='F').tolist()))
        H_numpy = np.asarray(H,float)
        if H_numpy.shape != (3,3) or not np.isfinite(H_numpy).all() or np.linalg.matrix_rank(H_numpy) != 3:
            raise ValueError('cost chart homography must be finite and nonsingular')
        H = ca.DM(H_numpy)
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
            # Same numerical domain as the NumPy chart. Invalid queries must
            # reach the finite-objective guard, not a fictitious finite chart.
            return ca.if_else(ca.fabs(projected[2]) >= 1e-8,
                              J @ ground @ J.T, ca.DM(np.full((2,2), np.nan)))
        return evaluate
