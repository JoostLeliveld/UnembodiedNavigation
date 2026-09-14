"""Three planning observation interfaces with fixed independent future readings.

P - q K S K' is the expected conditional covariance for one Bernoulli
measurement at a fixed prior. Sequentially reusing that moment is an
approximation to branching, NOT exact filtering over unknown future outcomes.
It retains the joint pose/camera covariance and never treats q as a hit.
"""
from paths import *
import json
import numpy as np
import casadi as ca
from joint_casadi import build_step
from bayesian_field import CAMERAS
from planning.core.casadi_efe import make_g_from_homography,unicycle_process_noise_ca,risk_ca,_differential_entropy_ca

def gp_function(artifact):
    xy=ca.SX.sym('xy',2);out=[];variant=artifact['variant']
    if variant not in ['Uniform','IWAI','Commissioned']:raise ValueError(variant)
    for c in CAMERAS:
        f=artifact['fields'][c]
        if variant=='Uniform':q=ca.DM(f['uniform_score'])
        else:
            values=np.array(f['score' if variant=='IWAI' else 'availability'])
            lookup=ca.interpolant('field_'+c+'_'+variant,'linear',[artifact['xs'],artifact['ys']],values.T.ravel(order='F'))
            pos=ca.vertcat(ca.fmin(artifact['xs'][-1],ca.fmax(artifact['xs'][0],xy[0])),ca.fmin(artifact['ys'][-1],ca.fmax(artifact['ys'][0],xy[1])))
            q=lookup(pos)
        out.append(ca.fmin(1-1e-4,ca.fmax(1e-4,q)))
    return ca.Function('camera_field_'+variant,[xy],[ca.vertcat(*out)])

def belief_probability(gp,m,P):
    # Positive-weight five-point XY quadrature, covariance exactly reproduced.
    a=ca.sqrt(ca.fmax(P[0,0],1e-12));b=P[1,0]/a
    d=ca.sqrt(ca.fmax(P[1,1]-b*b,1e-12))
    L=ca.vertcat(ca.horzcat(a,0),ca.horzcat(b,d))
    q=gp(m[:2])/3
    for j in range(2):
        delta=np.sqrt(3)*L[:,j]
        q+=(gp(m[:2]+delta)+gp(m[:2]-delta))/6
    return q

def expected_update(P,model,field,independent=False,artifact=None):
    # Future readings are independent in all pilot arms. Runtime fusion is fixed.
    assert independent and artifact is not None
    for j,c in enumerate(CAMERAS):
        H=np.zeros((2,13));H[:,:2]=np.eye(2);H=ca.DM(H)
        M=model.blocks[c]['R']+model.blocks[c]['B']
        if artifact['variant']=='Commissioned':
            noise=ca.DM(M);chance=field[j]
        else:
            miss=M+artifact['miss_extra_std_m']**2*np.eye(2)
            precision=field[j]*ca.DM(np.linalg.inv(M))+(1-field[j])*ca.DM(np.linalg.inv(miss))
            noise=ca.solve(precision,ca.DM.eye(2));chance=1.
        S=H@P@H.T+noise;K=ca.solve(S,H@P).T
        P=P-chance*(K@S@K.T);P=(P+P.T)/2
    return P

def event_offsets(age,period,horizon=20,dt=.25):
    if not np.isfinite(age) or age<0 or period<=dt:raise ValueError('invalid event clock')
    # Phase of a periodic surrogate. A long outage does not insert past events.
    next_time=period-(age%period)
    offsets=np.full(horizon,-1.)
    while next_time<=horizon*dt+1e-10:
        k=min(int(np.ceil(next_time/dt-1e-10))-1,horizon-1)
        if k>=0:offsets[k]=np.clip(next_time-k*dt,0,dt)
        next_time+=period
    return offsets

def build(model,planner,artifact,independent=False):
    n,dt=planner.horizon,planner.dt
    m0=ca.SX.sym('m0',13);P0=ca.SX.sym('P0',13,13)
    U=ca.SX.sym('U',2,n);offsets=ca.SX.sym('offsets',n);goal=ca.SX.sym('goal',2)
    gp=gp_function(artifact);prop=build_step(model,independent)
    x=ca.SX.sym('x',3);g=ca.Function('metric_goal_chart',[x],[x[:2]])
    dg=ca.Function('chart_jac',[x],[ca.jacobian(g(x),x)])
    ggoal=g(ca.vertcat(goal,0));goal_cov=ca.DM.eye(2)*.15**2
    assert np.allclose(planner.goal_obs_cov_for_progress(0),planner.goal_obs_cov_for_progress(1))
    nogo=planner.nogo_cost_model.make_penalty_state_casadi()
    if planner.use_belief_nogo_cost:raise ValueError('This integration retains the study mean-state no-go objective')
    m,P=m0,P0;parts=ca.SX.zeros(4,1);mm=[];pp=[];qq=[]
    def motion(m,P,v,w,d,step_heading):
        # Production noise function regularizes dt=0 internally; mask the entire
        # zero-duration transition to preserve identity at event boundaries.
        safe=ca.fmax(d,1e-9)
        Q=unicycle_process_noise_ca(planner.process_noise_xy,planner.process_noise_theta,safe,m[2],v)
        # Interpolate the existing Euler control interval without changing its
        # grid-boundary mean when an observation falls inside the interval.
        angle=step_heading-m[2]
        body=v*d*ca.vertcat(ca.cos(angle),ca.sin(angle))
        nm,np_=prop(m,P,body,w*d,Q,ca.DM.zeros(5))
        return ca.if_else(d>0,nm,m),ca.if_else(d>0,np_,P)
    for t in range(n):
        v,w=U[0,t],U[1,t];has=offsets[t]>=0
        heading=m[2]
        predicted,prior=motion(m,P,v,w,dt,heading)
        first=ca.if_else(has,offsets[t],dt)
        m,P=motion(m,P,v,w,first,heading)
        q=belief_probability(gp,m,P)
        updated=expected_update(P,model,q,independent,artifact)
        P=ca.if_else(has,updated,P)
        m,P=motion(m,P,v,w,dt-first,heading)
        J=dg(predicted[:3]);S=J@prior[:3,:3]@J.T
        stage=ca.vertcat(
            planner.risk_weight_obs*planner.observation_risk_scale*risk_ca(g(predicted[:3]),S,ggoal,goal_cov),
            planner.ambiguity_weight*planner.ambiguity_term_scale*_differential_entropy_ca(P[:2,:2]),
            planner.control_weight*ca.sumsqr(U[:,t]),nogo(m[:3]))
        parts+=planner.discount_gamma**t*stage
        mm.append(m[:3]);pp.append(ca.reshape(P,169,1));qq.append(q)
    total=ca.sum1(parts)
    args=[m0,P0,U,offsets,goal]
    objective=ca.Function('gp_joint_efe',args,[total,ca.gradient(total,ca.vec(U)),parts])
    trace=ca.Function('gp_joint_trace',args,[ca.horzcat(*mm),ca.horzcat(*pp),ca.horzcat(*qq)])
    return objective,trace
