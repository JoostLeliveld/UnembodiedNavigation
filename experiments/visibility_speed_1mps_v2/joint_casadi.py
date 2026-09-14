"""CasADi forecast of the frozen joint process, with explicit binary masks.

Masks are parameters, never differentiable surrogates for hit probabilities.
Scenario generation and route-dependent availability live outside this graph.
"""
from pathlib import Path
import sys
import casadi as ca
import numpy as np

O = Path(__file__).resolve().parent.parent
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(O), str(O/'final_model'), str(R/'src/planning')]
from bayesian_field import CAMERAS
from planning.core.casadi_efe import unicycle_process_noise_ca


def build_step(model, independent=False):
    """Match JointFieldFilter.predict_increment and camera update exactly."""
    m = ca.SX.sym('mean',13)
    P = ca.SX.sym('covariance',13,13)
    body = ca.SX.sym('body',2)
    yaw = ca.SX.sym('dyaw')
    Q = ca.SX.sym('Q',3,3)
    visible = ca.SX.sym('visible',5)
    c,s = ca.cos(m[2]),ca.sin(m[2])
    delta = ca.vertcat(c*body[0]-s*body[1],s*body[0]+c*body[1])
    F = ca.SX.eye(13)
    F[0,2],F[1,2] = -delta[1],delta[0]
    distance = ca.if_else(ca.sumsqr(body)>0,ca.sqrt(ca.sumsqr(body)),0.)
    a = ca.vertcat(ca.DM.ones(3),*[ca.repmat(ca.exp(-distance/model.blocks[c]['length']),2,1) for c in CAMERAS])
    if independent:
        a = ca.DM.ones(13)
    A = ca.diag(a)
    propagated = A@P@A.T
    if not independent:
        for j,camera in enumerate(CAMERAS):
            k=3+2*j
            propagated[k:k+2,k:k+2] += (1-a[k]**2)*ca.DM(model.blocks[camera]['B'])
    propagated = F@propagated@F.T
    propagated[:3,:3] += Q
    out_m = A@m
    out_m[:2] += delta
    out_m[2] += yaw
    for j,camera in enumerate(CAMERAS):
        H = np.zeros((2,13));H[:,:2]=np.eye(2)
        block=model.blocks[camera]
        noise=block['R'].copy()
        if independent:
            noise += block['B']
        else:
            H[:,3+2*j:5+2*j]=np.eye(2)
        H=ca.DM(H);noise=ca.DM(noise)
        S=H@propagated@H.T+noise
        K=ca.solve(S,H@propagated).T
        cross=K@H@propagated
        updated=propagated-cross-cross.T+K@S@K.T
        propagated=ca.if_else(visible[j]>.5,updated,propagated)
        propagated=(propagated+propagated.T)/2
    return ca.Function('joint_step_independent' if independent else 'joint_step_correlated',
        [m,P,body,yaw,Q,visible],[out_m,propagated])


def build_rollout(model,horizon,dt,independent=False):
    m0=ca.SX.sym('m0',13);P0=ca.SX.sym('P0',13,13)
    controls=ca.SX.sym('controls',2,horizon)
    masks=ca.SX.sym('masks',5,horizon)
    fn=build_step(model,independent)
    m,P=m0,P0
    means=[];covariances=[]
    for t in range(horizon):
        v,w=controls[0,t],controls[1,t]
        Q=unicycle_process_noise_ca(.01,.02,dt,m[2],v)
        m,P=fn(m,P,ca.vertcat(v*dt,0),w*dt,Q,masks[:,t])
        means.append(m);covariances.append(ca.reshape(P,169,1))
    return ca.Function('joint_rollout',[m0,P0,controls,masks],
        [ca.horzcat(*means),ca.horzcat(*covariances)])
