"""Belief feedback on a CasADi-generated path, independent of solve duration.

Uses the existing pure-pursuit principle and unchanged actuator bounds. The
optimizer still supplies all path points; no oracle or scripted route enters.
"""
import numpy as np
from speed_profile import V_MAX,LOOKAHEAD,PROFILE

def track(pose,states,controls,goal,lookahead=LOOKAHEAD):
    xy=np.asarray(states)[:,:2];p=np.asarray(pose);segments=np.diff(xy,axis=0)
    lengths=np.linalg.norm(segments,axis=1);den=np.maximum(lengths**2,1e-12)
    alpha=np.clip(np.sum((p[:2]-xy[:-1])*segments,axis=1)/den,0,1)
    projections=xy[:-1]+alpha[:,None]*segments
    k=int(np.argmin(np.linalg.norm(projections-p[:2],axis=1)))
    deviation=float(np.linalg.norm(projections[k]-p[:2]))
    arclength=np.r_[0,np.cumsum(lengths)];s=arclength[k]+alpha[k]*lengths[k]
    target_s=min(s+lookahead,arclength[-1])
    j=min(int(np.searchsorted(arclength,target_s,side='right')-1),len(lengths)-1)
    fraction=np.clip((target_s-arclength[j])/max(lengths[j],1e-12),0,1)
    target=xy[j]+fraction*segments[j]
    delta=target-p[:2];distance=float(np.linalg.norm(delta));goal_distance=float(np.linalg.norm(np.asarray(goal)-p[:2]))
    heading=np.arctan2(delta[1],delta[0]);error=float(np.arctan2(np.sin(heading-p[2]),np.cos(heading-p[2])))
    if arclength[-1]<.01:
        return np.asarray(controls)[0].copy(),dict(deviation_m=deviation,target=target.tolist(),mode='in_place_plan')
    v=min(V_MAX,.65*goal_distance,PROFILE['tracker_target_gain_per_s']*distance)
    if abs(error)>.6:v=0.;w=np.clip(1.2*error,-.4,.4)
    else:w=np.clip(2*v*np.sin(error)/max(distance,.05),-.4,.4)
    if arclength[-1]-s<.02:v=0.
    return np.array([v,w]),dict(deviation_m=deviation,target=target.tolist(),mode='belief_path_feedback',heading_error_rad=error)
