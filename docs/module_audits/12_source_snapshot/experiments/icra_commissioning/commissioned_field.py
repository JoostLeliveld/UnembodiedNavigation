"""Planner-facing camera field: availability and outcome-conditioned quality.

Only predicted pose enters query methods. Training observations supply labels and
quality regimes. GP function variance is never used as measurement covariance.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from reliability.observation_gp import ObservabilityGP
from model import expected_posterior, update


CAMERAS = tuple(f"camera_{c}" for c in "ABCDE")


def pose_embedding(poses, heading=True):
    poses = np.atleast_2d(np.asarray(poses, float))
    if poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError("field query is finite predicted pose [x,y,heading], with no image inputs")
    if heading:
        return np.column_stack((poses[:, :2], np.cos(poses[:, 2]), np.sin(poses[:, 2])))
    return poses[:, :2]


def integrated_sigmoid(mu, sigma):
    """Gauss-Hermite integration of the logistic link under the latent Gaussian."""
    nodes, weights = np.polynomial.hermite.hermgauss(20)
    f = np.asarray(mu)[..., None] + np.sqrt(2)*np.asarray(sigma)[..., None]*nodes
    values = 1/(1+np.exp(-np.clip(f, -60, 60)))
    return np.sum(values*weights, axis=-1)/np.sqrt(np.pi)


class AvailabilityModel:
    def __init__(self, kind, parameter, geometry):
        self.kind, self.parameter, self.geometry = kind, parameter, geometry

    def _geometry_features(self, poses, camera):
        model = self.geometry[camera]
        result=[]
        for x,y,yaw in np.atleast_2d(poses):
            u,v,visible=model.world_to_pixel(x,y)
            d=float(np.linalg.norm(np.array([x,y])-np.asarray(model.cam_pos)[:2]))
            row=[float(visible),d/10,1/max(d,1),float(np.clip(u/1280,-4,4)),float(np.clip(v/720,-4,4))]
            row=np.nan_to_num(row,nan=0.,posinf=4.,neginf=-4.).tolist()
            if self.kind == 'geometry_heading': row += [np.cos(yaw),np.sin(yaw)]
            result.append(row)
        return np.asarray(result)

    def fit(self, poses, hits):
        self.poses=np.asarray(poses);self.hits=np.asarray(hits,float)
        self.rate=self.hits.mean(axis=0)
        self.support_tree=cKDTree(self.poses[:,:2])
        self.models=[]
        if self.kind.startswith('local'):
            self.tree=cKDTree(pose_embedding(poses,self.kind=='local_heading'))
        elif self.kind.startswith('geometry'):
            for j,c in enumerate(CAMERAS):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=float(self.parameter),max_iter=1000))
                model.fit(self._geometry_features(poses,c),hits[:,j]);self.models.append(model)
        elif self.kind.startswith('gp'):
            length,noise=self.parameter
            for j in range(5):
                self.models.append(ObservabilityGP(length_scale=length,noise_var=noise).fit(poses[:,:2],hits[:,j]))
        elif self.kind != 'constant':raise ValueError(self.kind)
        return self

    def predict(self, poses):
        poses=np.atleast_2d(np.asarray(poses,float));pose_embedding(poses)
        if self.kind=='constant':q=np.tile(self.rate,(len(poses),1))
        elif self.kind.startswith('geometry'):
            q=np.column_stack([m.predict_proba(self._geometry_features(poses,c))[:,1]
                               for c,m in zip(CAMERAS,self.models)])
        elif self.kind.startswith('local'):
            k=min(int(self.parameter),len(self.poses))
            _,idx=self.tree.query(pose_embedding(poses,self.kind=='local_heading'),k=k)
            if k==1:idx=idx[:,None]
            q=(self.hits[idx].sum(axis=1)+1)/(k+2)
        else:
            values=[]
            for model in self.models:
                mu,sigma=model.predict_latent(poses[:,:2])
                values.append(integrated_sigmoid(mu,sigma) if self.kind=='gp_integrated'
                              else 1/(1+np.exp(-np.clip(mu,-60,60))))
            q=np.column_stack(values)
        return np.clip(q,1e-4,1-1e-4)

    def support(self, poses):
        return self.support_tree.query(np.atleast_2d(poses)[:,:2])[0] <= 2.


class CommissionedField:
    """Empirical joint outcomes plus alternative marginal availability models.

    R samples are predictions of the frozen current-image model at commissioned
    opportunities, used as a distribution over future unknown quality features.
    They are not covariance estimates from the mean NN's training residuals.
    """
    def __init__(self, poses, hits, covariances, availability, constant_covariances):
        self.poses=np.asarray(poses);self.hits=np.asarray(hits,bool)
        self.covariances=np.asarray(covariances)
        self.availability=availability
        self.constant_covariances=np.asarray(constant_covariances)
        self.tree=cKDTree(pose_embedding(self.poses))
        self.xy_tree=cKDTree(self.poses[:,:2])
        self.hit_trees={}
        for j in range(5):
            ids=np.flatnonzero(self.hits[:,j])
            self.hit_trees[j]=(ids,cKDTree(pose_embedding(self.poses[ids])))

    def joint_outcomes(self, state, k=24, constant_quality=False):
        query=pose_embedding([state])[0]
        if self.xy_tree.query(np.asarray(state)[:2])[0]>2.:
            return [(1.,[])],False
        _,ids=self.tree.query(query,k=min(k,len(self.poses)))
        ids=np.atleast_1d(ids);out=[]
        for i in ids:
            cameras=[(CAMERAS[j],self.constant_covariances[j] if constant_quality else self.covariances[i,j])
                     for j in range(5) if self.hits[i,j]]
            out.append((1/len(ids),cameras))
        return out,True

    def quality(self, state, camera_index, k=24):
        ids,tree=self.hit_trees[camera_index]
        _,nearest=tree.query(pose_embedding([state])[0],k=min(k,len(ids)))
        # This averaged R is an explicit approximation, compared with joint
        # quality-outcome averaging; it is not asserted to yield E[P+].
        return self.covariances[ids[np.atleast_1d(nearest)],camera_index].mean(axis=0)

    def forecast(self, state, P, availability_kind='local_joint', mode='branch',
                 constant_quality=False):
        """One-step reference/approximation; multistep caller owns cadence and memory."""
        pose_embedding([state])
        if self.xy_tree.query(np.asarray(state)[:2])[0]>2.:
            return P.copy(),dict(supported=False,q_any=0.)
        if availability_kind=='local_joint':
            outcomes,_=self.joint_outcomes(state,constant_quality=constant_quality)
        else:
            if availability_kind=='local_independent':
                local,_=self.joint_outcomes(state,constant_quality=constant_quality)
                q=np.array([sum(p for p,views in local if any(c==camera for c,R in views))
                            for camera in CAMERAS])
            else:
                q=self.availability[availability_kind].predict([state])[0]
            Rs=[self.constant_covariances[j] if constant_quality else self.quality(state,j) for j in range(5)]
            outcomes=[]
            for mask in product((0,1),repeat=5):
                p=float(np.prod([q[j] if hit else 1-q[j] for j,hit in enumerate(mask)]))
                outcomes.append((p,[(CAMERAS[j],Rs[j]) for j,hit in enumerate(mask) if hit]))
        if mode=='branch':
            answer=np.zeros_like(P)
            for probability,views in outcomes:
                post=P.copy()
                for camera,R in views:_,post,_=update(np.zeros(len(P)),post,np.zeros(2),R)
                answer+=probability*post
        elif mode=='information':
            information=np.linalg.solve(P,np.eye(len(P)))
            for probability,views in outcomes:
                for camera,R in views:information[:2,:2]+=probability*np.linalg.solve(R,np.eye(2))
            answer=np.linalg.solve(information,np.eye(len(P)))
        else:raise ValueError(mode)
        return (answer+answer.T)/2,dict(supported=True,q_any=sum(p for p,v in outcomes if v))
