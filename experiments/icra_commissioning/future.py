#!/usr/bin/env python3
"""Empirical joint-outcome forecasts from commissioning, validated on a recorded route.

No future image, detection, or reference pose enters the forecast. Queries use the
replay's predicted state. Commissioning reference poses supply the installation map.
Future measured controls are prescribed as a fixed-route input for this diagnostic;
this is not a demonstration of closed-loop route selection.
"""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import REPO,OUT,CAPTURE,readcsv,load,writejson,digest
from replay import unicycle_step,unicycle_jacobian,unicycle_process_noise
from model import update


def pose_features(x):
    x=np.asarray(x)
    return np.column_stack((x[:,0],x[:,1],np.cos(x[:,2]),np.sin(x[:,2])))


class JointCommissioning:
    """Nearest 12 joint camera outcomes, preserving within-pose hit/miss dependence.

    Probability represents variation over nearby commissioned configurations. Errors
    across cameras remain block diagonal; consecutive opportunities are approximated
    as independent. These are explicit hypotheses tested downstream, not guarantees.
    """
    def __init__(self,out,kind):
        data,_=load(out);models=joblib.load(out/'models.joblib')
        raw=readcsv(REPO/CAPTURE/'bias_update_interpretations.csv')
        roles=json.loads((out/'manifest.json').read_text())['roles']
        from study import tile
        poses={}; outcomes=defaultdict(list)
        for r in raw:
            role='mean_train' if r['split']=='train' else roles[tile(r)]
            if role not in ['mean_train','covariance_fit']:continue
            poses[f"{r['pose_id']}:{r['repetition_id']}"]=[float(r[k]) for k in ['robot_x','robot_y','robot_yaw']]
        for r in data:
            if r['frame'] not in poses:continue
            _,R=models[r['camera'],kind].predict([r])
            outcomes[r['frame']].append((r['camera'],R[0]))
        ids=sorted(poses);self.outcomes=[outcomes[k] for k in ids]
        self.tree=cKDTree(pose_features([poses[k] for k in ids]));self.cameras=models

    def forecast(self,state,P,approx):
        distances,indices=self.tree.query(pose_features([state])[0],k=12)
        # Outside 2 m/radian embedding support, refuse to forecast a camera update.
        if distances[0]>2.:return P.copy(),0.,float(distances[0])
        outcomes=[self.outcomes[i] for i in indices]
        if approx=='branch':
            answer=np.zeros_like(P)
            for outcome in outcomes:
                post=P.copy()
                for camera,R in outcome: _,post,_=update(np.zeros(3),post,np.zeros(2),R)
                answer+=post/len(outcomes)
        else:
            J=np.linalg.inv(P)
            for outcome in outcomes:
                for camera,R in outcome:J[:2,:2]+=np.linalg.solve(R,np.eye(2))/len(outcomes)
            answer=np.linalg.solve(J,np.eye(3))
        q=sum(bool(o) for o in outcomes)/len(outcomes)
        return answer,q,float(distances[0])


def main():
    selection=json.loads((OUT/'driving_manifest.json').read_text());allrows=[]
    joint=JointCommissioning(OUT,'confidence')
    for entry in selection['runs']:
        run=REPO/entry['run'];m=json.loads((run/'run_manifest.json').read_text())
        trace=np.load(OUT/f"replay_{m['run_id']}_confidence.npz")
        rows=readcsv(run/'experiment.csv'); odom={}
        for r in rows:
            t=float(r['odom_noisy_stamp']);u=[float(r['odom_noisy_v']),float(r['odom_noisy_w'])]
            if np.isfinite([t,*u]).all():odom.setdefault(t,u)
        tt=np.array(sorted(odom));uu=np.array([odom[t] for t in tt])
        for horizon in [1.,3.,5.]:
            for start in np.arange(trace['time'][0],trace['time'][-1]-horizon,2.):
                i=np.searchsorted(trace['time'],start);j=np.searchsorted(trace['time'],start+horizon)
                for cadence in [.2,1.]:
                    for approx in ['branch','information']:
                        state=trace['state'][i].copy();P=trace['covariance'][i].copy();qs=[];support=[]
                        previous=start
                        for t in np.arange(start+cadence,start+horizon+1e-7,cadence):
                            idx=max(0,np.searchsorted(tt,previous,side='right')-1);u=uu[idx];dt=t-previous
                            F=unicycle_jacobian(state,u,dt);Q=unicycle_process_noise(.01,.02,dt,theta=state[2],v=u[0])
                            P=F@P@F.T+Q;state=unicycle_step(state,u,dt)
                            P,q,dist=joint.forecast(state,P,approx);qs.append(q);support.append(dist);previous=t
                        allrows.append(dict(run=entry['run'],start_s=float(start),horizon_s=horizon,cadence_s=cadence,
                            method=approx,predicted_trace_m2=float(np.trace(P[:2,:2])),
                            realized_squared_error_m2=float(np.sum(trace['error'][j]**2)),
                            realized_filter_trace_m2=float(np.trace(trace['covariance'][j,:2,:2])),
                            mean_usable_probability=float(np.mean(qs)),max_support_distance=float(max(support))))
    summary=[]
    for h in [1.,3.,5.]:
        for cadence in [.2,1.]:
            for method in ['branch','information']:
                r=[r for r in allrows if r['horizon_s']==h and r['cadence_s']==cadence and r['method']==method]
                p=np.array([v['predicted_trace_m2'] for v in r]);e=np.array([v['realized_squared_error_m2'] for v in r])
                summary.append(dict(horizon_s=h,cadence_s=cadence,method=method,windows=len(r),
                    predicted_rms_cm=float(100*np.sqrt(p.mean())),realized_rms_cm=float(100*np.sqrt(e.mean())),
                    spearman_with_squared_error=float(spearmanr(p,e).statistic)))
    writejson(OUT/'future_results.json',dict(status='one_route_development_forecast',rows=allrows,summary=summary,
      method='12 empirical joint outcomes; pose metric (x,y,cos heading,sin heading); support distance <=2',
      limitations=['Overlapping windows are not independent trials.',
        'Future measured controls supplied as fixed-route input, not estimated from future images.',
        'No route-ranking claim from one route; no closed-loop sensor-model planning comparison.',
        'Reference camera errors and initial belief error may be temporally dependent.',
        'Changing forecast cadence does not change replay estimator cadence; this is a sensitivity diagnostic.',
        'Joint camera availability represented; residual noise still assumed block diagonal.']))
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for method in ['branch','information']:
        r=[r for r in allrows if r['horizon_s']==3 and r['cadence_s']==1 and r['method']==method]
        axes[0].plot([v['start_s'] for v in r],[100*np.sqrt(v['predicted_trace_m2']) for v in r],label=method)
    axes[0].plot([v['start_s'] for v in r],[100*np.sqrt(v['realized_squared_error_m2']) for v in r],label='realized norm',alpha=.55)
    axes[0].set(xlabel='Forecast start (s)',ylabel='Position error / predicted RMS (cm)',title='3 s forecasts; 1 Hz opportunities');axes[0].legend(fontsize=7)
    axes[1].scatter([100*np.sqrt(v['predicted_trace_m2']) for v in r],[100*np.sqrt(v['realized_squared_error_m2']) for v in r],s=10,alpha=.5)
    axes[1].set(xlabel='Expected-information predicted RMS (cm)',ylabel='Realized position error (cm)',title='Dependent windows; one recorded route')
    fig.savefig(OUT/'future.pdf');fig.savefig(OUT/'future.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
