#!/usr/bin/env python3
"""Empirical joint-outcome forecasts from commissioning, validated on a recorded route.

No future image, detection, or reference pose enters the forecast. Queries use the
replay's predicted state. Commissioning reference poses supply the installation map.
Future measured controls are prescribed as a fixed-route input for this diagnostic;
this is not a demonstration of closed-loop route selection.
"""
import argparse,csv,hashlib,io,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/p) for p in ['src/planning','src/unav_common','experiments/fusion_on_fixed_routes']]
from study import OUT,load,writejson,digest
import aligned
from future_rollout import common_motion_grid,forecast_window,observation_times
from planning.core.plan_validation import validate_covariance
from model import update

REQUIRED=['run_manifest.json','run_summary.json','experiment.csv','fusion_observations.csv','correction_assimilations.csv']
CADENCES=(.2,1.)
METHODS=('branch','information')
HORIZONS=(1.,3.,5.)


def read_snapshot(path,identities):
    """Hash exactly the bytes passed to a parser; retain their input identity."""
    path=Path(path).resolve();data=path.read_bytes()
    identities[str(path)]=hashlib.sha256(data).hexdigest()
    return data


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
        out=Path(out);self.input_hashes={}
        manifest=json.loads(read_snapshot(out/'manifest.json',self.input_hashes))
        data,_=load(out,write_records=False)
        models=joblib.load(io.BytesIO(read_snapshot(out/'models.joblib',self.input_hashes)))
        raw=list(csv.DictReader(io.StringIO(read_snapshot(
            REPO/manifest['capture']/'bias_update_interpretations.csv',self.input_hashes).decode())))
        roles=manifest['roles']
        for name,expected in manifest['files'].items():
            parsed_hash=self.input_hashes.get(str((REPO/name).resolve()),expected)
            if parsed_hash!=expected:raise ValueError(f'parsed commissioning input differs from manifest: {name}')
            if digest(REPO/name)!=expected:raise ValueError(f'commissioning input changed: {name}')
            self.input_hashes[str((REPO/name).resolve())]=expected
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
        ids=sorted(poses)
        if len(ids)<12:raise ValueError('joint forecast requires at least 12 commissioned poses')
        features=pose_features([poses[k] for k in ids])
        if not np.isfinite(features).all():raise ValueError('nonfinite commissioned pose')
        self.outcomes=[outcomes[k] for k in ids]
        for outcome in self.outcomes:
            cameras=[camera for camera,R in outcome]
            if len(cameras)!=len(set(cameras)):raise ValueError('duplicate camera in commissioned outcome')
            for camera,R in outcome:validate_covariance(R,dimension=2,positive_definite=True,name='conditional camera R')
        self.tree=cKDTree(features);self.cameras=models

    def forecast(self,state,P,approx,*,return_support=False):
        if approx not in METHODS:raise ValueError('unsupported forecast approximation')
        state=np.asarray(state,float)
        if state.shape!=(3,) or not np.isfinite(state).all():raise ValueError('invalid forecast state')
        P=validate_covariance(P,positive_definite=approx=='information')
        distances,indices=self.tree.query(pose_features([state])[0],k=12)
        support=dict(nearest_distance=float(distances[0]),farthest_distance=float(distances[-1]),
                     contributors=12,contributors_within_threshold=int(np.count_nonzero(distances<=2.)),
                     nearest_supported=bool(distances[0]<=2.))
        report=support if return_support else float(distances[0])
        # The historical gate uses the nearest pose only. Do not imply that all
        # twelve contributors are within the threshold or change their weighting.
        if distances[0]>2.:return P.copy(),0.,report
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
        return validate_covariance(answer),q,report


def load_trace(path,identities):
    with np.load(io.BytesIO(read_snapshot(path,identities)),allow_pickle=False) as artifact:
        trace={key:np.array(artifact[key],copy=True) for key in ('time','state','covariance')}
    t=trace['time'];m=trace['state'];P=trace['covariance']
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('cached replay timestamps must be finite and strictly increasing')
    if m.shape!=(len(t),3) or not np.isfinite(m).all() or P.shape!=(len(t),3,3):
        raise ValueError('invalid cached replay belief arrays')
    for covariance in P:validate_covariance(covariance)
    return trace


def window_indices(times,nominal_start,horizon):
    """Use identifiable cached belief instants, recording both timing offsets."""
    i=int(np.searchsorted(times,nominal_start))
    if i>=len(times):return None
    j=int(np.searchsorted(times,times[i]+horizon))
    return None if j>=len(times) else (i,j)


def summarize(rows):
    summary=[]
    for horizon in HORIZONS:
        for cadence in CADENCES:
            for method in METHODS:
                r=[v for v in rows if v['horizon_s']==horizon and v['cadence_s']==cadence and v['method']==method]
                p=np.array([v['predicted_trace_m2'] for v in r]);e=np.array([v['realized_squared_error_m2'] for v in r])
                correlation=None
                if len(r)>1 and np.ptp(p)>0 and np.ptp(e)>0:
                    correlation=float(spearmanr(p,e).statistic)
                summary.append(dict(horizon_s=horizon,cadence_s=cadence,method=method,windows=len(r),
                    predicted_rms_cm=float(100*np.sqrt(p.mean())) if len(r) else None,
                    realized_rms_cm=float(100*np.sqrt(e.mean())) if len(r) else None,
                    spearman_with_squared_error=correlation))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=OUT)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True,help='New directory; existing outputs are refused.')
    parser.add_argument('--max-reference-gap-s',type=float,default=None,
                        help='Explicit maximum interpolation bracket; default requires exact logged truth.')
    args=parser.parse_args()
    if args.output_dir.exists():raise ValueError('forecast output directory already exists')
    identities={};selection_path=args.selection or args.input_dir/'driving_manifest.json'
    sources=[Path(__file__),Path(__file__).with_name('future_rollout.py'),Path(__file__).with_name('model.py'),
             Path(__file__).with_name('study.py'),REPO/'experiments/fusion_on_fixed_routes/aligned.py',
             REPO/'src/planning/planning/core/dynamics.py',REPO/'src/planning/planning/core/plan_validation.py']
    source_bytes={p:read_snapshot(p,identities) for p in sources}
    selection=json.loads(read_snapshot(selection_path,identities));allrows=[];skipped=[]
    if not selection.get('runs'):raise ValueError('nonempty frozen driving selection required')
    names=[entry['run'] for entry in selection['runs']]
    if len(set(names))!=len(names):raise ValueError('duplicate run in frozen driving selection')
    joint=JointCommissioning(args.input_dir,'confidence');identities.update(joint.input_hashes)
    for entry in selection['runs']:
        run,m,_=aligned.verify_frozen_entry(entry,REQUIRED,repo=REPO)
        aligned.validate_run_ledger(run)
        if not m['use_odom_for_predict'] or m['odom_topic']!='/odom_noisy':raise ValueError('wrong prescribed motion input')
        if (m['process_noise_xy'],m['process_noise_theta'])!=(.01,.02):raise ValueError('forecast Q differs from selected run')
        identities.update({str((run/name).resolve()):sha for name,sha in entry['files'].items()})
        trace=load_trace(args.input_dir/f"replay_{m['run_id']}_confidence.npz",identities)
        table=aligned.rows(run);truth=aligned.truth_series(run,table,max_reference_gap_s=args.max_reference_gap_s)
        odom=aligned.measured_odometry(table);tt=np.array(list(odom));uu=np.array(list(odom.values()))
        mission_start,mission_end=aligned.mission_interval(run)
        for horizon in HORIZONS:
            for start in np.arange(trace['time'][0],trace['time'][-1]-horizon,2.):
                pair=window_indices(trace['time'],start,horizon)
                reason=None
                if pair is None:reason='cached endpoint unavailable'
                else:
                    i,j=pair;anchor=float(trace['time'][i]);endpoint=float(trace['time'][j])
                    if not mission_start<=anchor<endpoint<=mission_end:reason='outside declared mission interval'
                    elif anchor<tt[0] or endpoint>tt[-1]:reason='prescribed-control endpoint unsupported'
                    elif not bool(truth.support(endpoint)):reason='endpoint reference unsupported'
                if reason:
                    skipped.append(dict(run=entry['run'],nominal_start_s=float(start),horizon_s=horizon,reason=reason));continue
                grid=common_motion_grid(anchor,endpoint,tt,CADENCES)
                gx,gy=truth.at(endpoint)
                error=trace['state'][j,:2]-[float(gx),float(gy)]
                for cadence in CADENCES:
                    for approx in METHODS:
                        result=forecast_window(trace['state'][i],trace['covariance'][i],start=anchor,end=endpoint,
                            control_times=tt,controls=uu,motion_grid=grid,opportunities=observation_times(anchor,endpoint,cadence),
                            forecast=lambda state,P,method:joint.forecast(state,P,method,return_support=True),method=approx)
                        P=result['covariance'];events=result['opportunities'];support=[v['support'] for v in events]
                        allrows.append(dict(run=entry['run'],nominal_start_s=float(start),start_s=anchor,end_s=endpoint,
                            horizon_s=horizon,effective_horizon_s=endpoint-anchor,start_offset_s=anchor-float(start),
                            endpoint_offset_s=endpoint-(anchor+horizon),cadence_s=cadence,motion_steps=result['motion_steps'],
                            predicted_state=result['state'].tolist(),opportunity_count=len(events),
                            method=approx,predicted_trace_m2=float(np.trace(P[:2,:2])),
                            realized_squared_error_m2=float(error@error),
                            realized_filter_trace_m2=float(np.trace(trace['covariance'][j,:2,:2])),
                            mean_usable_probability=float(np.mean([v['usable_probability'] for v in events])) if events else None,
                            max_support_distance=max((v['nearest_distance'] for v in support),default=None),
                            farthest_contributor_distance=max((v['farthest_distance'] for v in support),default=None),
                            opportunities_with_all_contributors_supported=sum(v['contributors_within_threshold']==12 for v in support)))
    summary=summarize(allrows)
    # A diagnostic cannot quietly absorb inputs changed while it was running.
    for name,expected in identities.items():
        if digest(name)!=expected:raise ValueError(f'forecast input changed during analysis: {name}')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for source in sources:
        destination=args.output_dir/'source_snapshot'/source.relative_to(REPO)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source_bytes[source])
    writejson(args.output_dir/'protocol.json',dict(schema=2,input_sha256=identities,
        source_sha256={str(p.relative_to(REPO)):identities[str(p.resolve())] for p in sources},selection=str(selection_path.resolve()),
        initial_belief='byte-identified historical replay cache; historical derivation not independently reproduced',
        reference_method='exact_logged_gt' if args.max_reference_gap_s is None else 'bounded_interpolation_of_logged_gt',
        max_reference_gap_s=args.max_reference_gap_s,process_noise_xy=.01,process_noise_theta=.02,
        motion_grid='same union of all compared opportunity times and every measured-control change; nanosecond clock',
        timing='actual cached belief start and endpoint; requested and effective horizons recorded separately',
        cadences_s=CADENCES,horizons_s=HORIZONS,methods=METHODS))
    writejson(args.output_dir/'future_results.json',dict(status='fixed_route_development_forecast' if allrows else 'no_supported_windows',
      rows=allrows,summary=summary,skipped_windows=skipped,
      method='12 equally weighted empirical joint outcomes; pose metric (x,y,cos heading,sin heading); nearest-only gate <=2',
      limitations=['Overlapping windows are not independent trials.',
        'Future measured controls supplied as fixed-route input, not estimated from future images.',
        'No route-ranking claim from one route; no closed-loop sensor-model planning comparison.',
        'Reference camera errors and initial belief error may be temporally dependent.',
        'Changing forecast cadence does not change replay estimator cadence; this is a sensitivity diagnostic.',
        'Joint camera availability represented; residual noise still assumed block diagonal.',
        'The nearest support gate does not ensure that all 12 contributors are supported.',
        'Cached initial and realized beliefs are historical diagnostic inputs; their derivation is not certified by current code.',
        'No calibration or improved-navigation claim follows from this numerical diagnostic.']))
    if not allrows:
        print(json.dumps(summary,indent=2));return
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for method in METHODS:
        r=[r for r in allrows if r['horizon_s']==3 and r['cadence_s']==1 and r['method']==method]
        axes[0].plot([v['start_s'] for v in r],[100*np.sqrt(v['predicted_trace_m2']) for v in r],label=method)
    axes[0].plot([v['start_s'] for v in r],[100*np.sqrt(v['realized_squared_error_m2']) for v in r],label='realized norm',alpha=.55)
    axes[0].set(xlabel='Forecast start (s)',ylabel='Position error / predicted RMS (cm)',title='3 s forecasts; 1 Hz opportunities');axes[0].legend(fontsize=7)
    axes[1].scatter([100*np.sqrt(v['predicted_trace_m2']) for v in r],[100*np.sqrt(v['realized_squared_error_m2']) for v in r],s=10,alpha=.5)
    axes[1].set(xlabel='Expected-information predicted RMS (cm)',ylabel='Realized position error (cm)',title='Dependent windows; selected recorded routes')
    fig.savefig(args.output_dir/'future.pdf');fig.savefig(args.output_dir/'future.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
