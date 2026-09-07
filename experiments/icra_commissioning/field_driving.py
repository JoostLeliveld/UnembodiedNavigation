#!/usr/bin/env python3
"""Frozen commissioned-field application to schema-7 driving opportunities.

Run selection is the campaign's explicit task/condition/seed ledger, never a directory
glob. Each completed run is hashed before loading through aligned.py. Fixed-route
forecasts use prescribed recorded controls; they are not closed-loop planning results.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/icra_mpl')
import argparse,json,sys,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import joblib
import yaml

from field_study import REPO,OUT,FIELD_OUT,REQ,CAPTURE,ARTIFACT,CAMERAS,M,writejson,digest
from replay import aligned,run_filter,unicycle_step,unicycle_jacobian,unicycle_process_noise
from derive_interpretations import camera_models
from reliability.learned_box_correction import LearnedBoxCorrection
from model import ray_basis
from study import score

PILOT=OUT/'field_pilot'
REQUIRED=['run_manifest.json','run_summary.json','experiment.csv','fusion_observations.csv',
          'correction_assimilations.csv','camera_opportunities.jsonl']


def frozen_runs(out):
    cfg=yaml.safe_load((REPO/'experiments/icra_commissioning/field_pilot_cpu.yaml').read_text())
    ledger=json.loads((PILOT/'campaign_log.json').read_text())
    expected={f'{t}__N1__seed{s}' for t,g in cfg['tasks'].items() for s in g['seeds']}
    if set(ledger)-expected:raise ValueError('extra task/condition/seed entries in field campaign ledger')
    entries=[];pending=[]
    for task,config in cfg['tasks'].items():
        for seed in config['seeds']:
            key=f'{task}__N1__seed{seed}';event=ledger.get(key)
            if not event or not event.get('finished_at'):
                pending.append(key);continue
            if not event.get('run_dir'):
                entries.append(dict(key=key,task=task,seed=seed,status='infrastructure_invalid',event=event));continue
            run=Path(event['run_dir']);m=json.loads((run/'run_manifest.json').read_text())
            if (m['task'],int(m['seed']))!=(task,seed):raise ValueError('ledger identity mismatch')
            if int(m['logging_schema_version'])<7:raise ValueError('complete opportunity log required')
            entry=dict(key=key,run=str(run.relative_to(REPO)),task=task,seed=seed,
                files={name:digest(run/name) for name in aligned.required_artifacts(run,REQUIRED)},status='completed',
                field_sha256=digest(FIELD_OUT/'field.joblib'),requirements_sha256=digest(REQ))
            directory=out/key;directory.mkdir(parents=True,exist_ok=True)
            selection=directory/'manifest.json'
            if selection.exists():
                if json.loads(selection.read_text())!=entry:raise ValueError('frozen run changed')
            else:writejson(selection,entry)
            entries.append(entry)
    selection=dict(status='complete' if not pending else 'partial_no_aggregate_claim',
        runs=entries,pending=pending,selection_source=str((PILOT/'campaign_log.json').relative_to(REPO)),
        config_sha256=digest(REPO/'experiments/icra_commissioning/field_pilot_cpu.yaml'))
    selected_path=out/'campaign_selection.json'
    if selected_path.exists():
        prior=json.loads(selected_path.read_text())
        if prior.get('status')=='complete' and prior!=selection:raise ValueError('completed frozen campaign selection changed')
    writejson(selected_path,selection)
    return entries,pending


def load_run(entry,*,max_reference_gap_s=None):
    run,m,summary=aligned.verify_frozen_entry(entry,REQUIRED,minimum_schema=7,repo=REPO)
    if entry.get('field_sha256')!=digest(FIELD_OUT/'field.joblib'):
        raise ValueError('frozen field artifact differs')
    if entry.get('requirements_sha256')!=digest(REQ):
        raise ValueError('frozen field requirements differ')
    if (m['process_noise_xy'],m['process_noise_theta'])!=(.01,.02):raise ValueError('Q differs')
    if not m['use_odom_for_predict'] or m['odom_topic']!='/odom_noisy':raise ValueError('odometry contract differs')
    ledger=aligned.validate_run_ledger(run)
    table=aligned.rows(run);truth=aligned.truth_series(run,table,max_reference_gap_s=max_reference_gap_s)
    ass=aligned.assimilations(run)
    start,stop=aligned.mission_interval(run)
    odom=aligned.measured_odometry(table,start=start,stop=stop)
    m=dict(m,replay_start_stamp=start)
    field_inputs=json.loads((FIELD_OUT/'manifest.json').read_text())['files']
    required_inputs=[ARTIFACT,f'{CAPTURE}/capture_manifest.json',str((OUT/'models.joblib').relative_to(REPO))]
    for name in required_inputs:
        if name not in field_inputs or digest(REPO/name)!=field_inputs[name]:
            raise ValueError(f'changed or unfrozen replay model/geometry input: {name}')
    mean=LearnedBoxCorrection(REPO/ARTIFACT)
    geometry=camera_models(json.loads((REPO/CAPTURE/'capture_manifest.json').read_text()))
    delivered,duplicates=aligned.camera_opportunities(run)
    bybatch=defaultdict(dict)
    for o in delivered:
        t=o['timestamp_s']
        if not start<=t<=stop:continue
        c=o['camera_id'];b=o['source_batch_id']
        bybatch[b][c]=o
    batches=[];readings=[];unscored=0
    for batch,obs in bybatch.items():
        if set(obs)!=set(CAMERAS):raise ValueError(f'incomplete in-drive opportunity batch {batch}')
        times=[o['timestamp_s'] for o in obs.values()]
        if max(times)-min(times)>.05+1e-8:raise ValueError('camera batch skew')
        # Current campaign is truly simultaneous; do not silently collapse skew.
        if max(times)-min(times)>1e-7:raise ValueError('forecast study requires simultaneous frames')
        t=times[0];gx,gy=truth.at([t]);yaw=truth.yaw_at([t])[0]
        supported=bool(np.isfinite([gx[0],gy[0],yaw]).all())
        if not supported:unscored+=1
        hits=np.zeros(5,bool)
        for j,c in enumerate(CAMERAS):
            o=obs[c]
            if not o['detection_valid']:continue
            box=o['bbox_xyxy'];raw=geometry[c].pixel_to_world(*o['bbox_bottom_uv'])
            if raw is None or not np.isfinite(raw).all():continue
            z=mean.correct(c,raw,box,o['detector_score'])
            if z is None:raise ValueError('frozen mean rejected projected hit')
            hits[j]=True
            readings.append(dict(t=t,camera=c,batch=batch,z=np.asarray(z),raw=np.asarray(raw),
                truth=np.array([gx[0],gy[0]]),confidence=o['detector_score'],
                basis=ray_basis(np.asarray(raw),np.asarray(mean._geometry[c]['xy'])),
                distance=float(np.linalg.norm(np.asarray(raw)-mean._geometry[c]['xy']))))
        batches.append(dict(t=t,batch=batch,hits=hits,reference=np.array([gx[0],gy[0],yaw]),reference_supported=supported))
    batches.sort(key=lambda r:r['t']);readings.sort(key=lambda r:(r['t'],r['camera']))
    valid_batches={r['batch'] for r in batches}
    in_drive_fused={a['source_batch_id'] for a in ass if start<=a['correction_stamp']<=stop}
    # Fused envelopes may describe batches whose captures predate first command;
    # accounting is checked above, opportunity completeness is checked independently.
    live=aligned.aligned_error_cm(run,'belief',table,max_reference_gap_s=max_reference_gap_s)
    keep=aligned.landed_mask(live['stamp'])&np.isfinite(live['aligned_cm'])&(live['stamp']>=start)&(live['stamp']<=stop)
    error=live['aligned_cm'][keep]
    mission_accounting=aligned.correction_accounting(run)
    population=aligned.landed_mask(live['stamp'])&live['have']&(live['stamp']>=start)&(live['stamp']<=stop)
    accounting=dict(batches=len(batches),unscored_reference_batches=unscored,duplicate_deliveries=duplicates,
        per_camera_opportunities=len(batches),fresh_readings=len(readings),fused_batches=len(ledger.by_batch),
        dropped_fraction=mission_accounting['correction_dropped_fraction'],
        longest_gap_s=mission_accounting['longest_correction_gap_s'],
        correction_accounting=mission_accounting,
        reference_max_gap_s=max_reference_gap_s,reference_method=live['reference_method'],
        gt_stamp_source=summary.get('gt_stamp_source'),outcome=summary['completion_reason'],
        live_belief_n=len(error),live_belief_reference_unscoreable=int(population.sum())-len(error),
        live_belief_median_cm=float(np.median(error)) if len(error) else None,
        live_belief_p95_cm=float(np.quantile(error,.95)) if len(error) else None,
        live_path_length_m=summary.get('path_length_m'),elapsed_s=stop-start,
        processed_batch_median_interval_s=float(np.median(np.diff([b['t'] for b in batches]))) if len(batches)>1 else None)
    return m,summary,truth,odom,readings,batches,accounting


def propagate(state,P,start,end,tt,uu):
    """ZOH measured motion with every odometry change included, never one u per window."""
    state=np.asarray(state).copy();P=np.asarray(P).copy()
    if end<start:raise ValueError('backwards propagation')
    previous=float(start)
    changes=tt[(tt>start)&(tt<end)]
    for t in [*changes,end]:
        idx=np.searchsorted(tt,previous,side='right')-1
        if idx<0:raise ValueError('missing causal control at forecast start')
        u=uu[idx];dt=float(t-previous)
        if dt:
            F=unicycle_jacobian(state,u,dt)
            Q=unicycle_process_noise(.01,.02,dt,theta=state[2],v=u[0])
            P=F@P@F.T+Q;state=unicycle_step(state,u,dt)
        previous=float(t)
    return state,(P+P.T)/2


def coarse_batches(batches,interval=1.):
    selected=[];previous=-np.inf
    for b in batches:
        if b['t']-previous>=interval-1e-8:selected.append(b);previous=b['t']
    return selected


def analyze(entry,out,*,max_reference_gap_s=None):
    directory=out/entry['key'];result_path=directory/'results.json'
    # Validate the selected run before any cache access. Cache ownership includes
    # the full selection, models, implementation and explicit reference policy.
    m,summary,truth,odom,readings,batches,accounting=load_run(entry,max_reference_gap_s=max_reference_gap_s)
    inputs=dict(selection=entry,max_reference_gap_s=max_reference_gap_s,
        files={str(p.relative_to(REPO)):digest(p) for p in [Path(__file__),Path(aligned.__file__),
            REPO/'experiments/icra_commissioning/replay.py',REPO/'experiments/icra_commissioning/study.py',
            REPO/'experiments/icra_commissioning/model.py',REPO/'src/unav_common/unav_common/correction_ledger.py',
            REPO/'scripts/shared/metrics.py',FIELD_OUT/'manifest.json',FIELD_OUT/'field.joblib',OUT/'models.joblib',REPO/ARTIFACT]})
    if result_path.exists():
        cached=json.loads(result_path.read_text())
        if cached.get('analysis_inputs')!=inputs:
            raise ValueError('existing analysis was produced from different/unknown inputs; use a new output directory')
        return cached
    directory.mkdir(parents=True,exist_ok=True)
    field=joblib.load(FIELD_OUT/'field.joblib');models=joblib.load(OUT/'models.joblib')
    coarse=coarse_batches(batches);chosen={b['batch'] for b in coarse}
    sub=[r for r in readings if r['batch'] in chosen]
    comparisons={};reference=None
    for rate,items in [('full',readings),('1Hz_policy',sub)]:
        for kind in ['constant','geometry','confidence','confidence_bias']:
            s,trace,innov=run_filter(m,truth,odom,items,models,kind,list(CAMERAS),0,prediction_times=[b['t'] for b in batches])
            comparisons[rate+'/'+kind]=s
            if rate=='1Hz_policy' and kind=='confidence':reference=trace
    for c in CAMERAS:
        if not any(r['camera']==c for r in readings):continue
        s,_,_=run_filter(m,truth,odom,readings,models,'constant',[c],0,prediction_times=[b['t'] for b in batches])
        comparisons['single/'+c]=s
    if not reference:raise ValueError('no reference-supported replay samples; choose an explicit reference policy')
    tt=np.array(sorted(odom));uu=np.asarray([odom[t] for t in tt])
    times=np.array([r['t'] for r in reference]);states=np.array([r['state'] for r in reference])
    covs=np.array([r['P'] for r in reference]);errs=np.array([r['error'] for r in reference])
    np.savez_compressed(directory/'reference_replay.npz',time=times,state=states,covariance=covs,error=errs)
    # Query state strictly precedes the current image; propagate using past odometry.
    query=[];observed=[];gtquery=[];batch_times=[]
    for b in batches:
        idx=np.searchsorted(times,b['t'],side='left')-1
        if idx<0:continue
        state,_=propagate(states[idx],covs[idx],times[idx],b['t'],tt,uu)
        query.append(state);gtquery.append(b['reference']);observed.append(b['hits']);batch_times.append(b['t'])
    query=np.asarray(query);gtquery=np.asarray(gtquery);observed=np.asarray(observed)
    availability={};qsave={}
    for kind,model in field.availability.items():
        supported=np.isfinite(gtquery).all(axis=1)
        predicted=model.predict(query);diagnostic=model.predict(gtquery[supported]) if supported.any() else None
        availability[kind]=dict(brier=M.brier(observed,predicted),logloss=M.logloss(observed,predicted),
            reference_pose_query_brier=M.brier(observed[supported],diagnostic) if diagnostic is not None else None,
            reference_pose_query_n=int(supported.sum()),supported_fraction=float(model.support(query).mean()),
            per_camera={c:dict(brier=M.brier(observed[:,j],predicted[:,j]),
                observed_fraction=float(observed[:,j].mean()),predicted_fraction=float(predicted[:,j].mean())) for j,c in enumerate(CAMERAS)})
        qsave[kind]=predicted
    np.savez_compressed(directory/'availability.npz',time=batch_times,query=query,reference_pose=gtquery,hits=observed,**qsave)
    # Normalized reference residual dependence, within each run and camera.
    temporal=[]
    for c in CAMERAS:
        rows=[r for r in readings if r['camera']==c and np.isfinite(r['truth']).all()]
        if len(rows)<10:continue
        z,R=models[c,'confidence'].predict(rows);e=z-np.array([r['truth'] for r in rows])
        normalized=np.linalg.solve(np.linalg.cholesky(R),e[...,None])[...,0]
        ts=np.array([r['t'] for r in rows]);pos=np.array([r['truth'] for r in rows])
        for lag in [1,2,5,10]:
            if len(rows)<lag+10:continue
            dt=ts[lag:]-ts[:-lag];valid=dt<max(2.,lag*.5)
            if valid.sum()<10:continue
            temporal.append(dict(camera=c,lag=lag,pairs=int(valid.sum()),lag_s=float(np.median(dt[valid])),
                distance_m=float(np.median(np.linalg.norm(pos[lag:]-pos[:-lag],axis=1)[valid])),
                correlation=[float(np.corrcoef(normalized[:-lag,j][valid],normalized[lag:,j][valid])[0,1])
                    if min(np.std(normalized[:-lag,j][valid]),np.std(normalized[lag:,j][valid]))>0 else None for j in range(2)]))
    # Frozen forecast settings. Future recorded controls are prescribed route inputs;
    # future camera images, detections and GT never enter a forecast query.
    forecasts=[]
    variants=[('constant','branch'),('geometry_xy','branch'),('gp_xy','branch'),
              ('gp_integrated','branch'),('local_joint','branch'),('local_joint','information')]
    for start in np.arange(times[0]+1,times[-1]-5,5.):
        idx=np.searchsorted(times,start,side='right')-1
        s0,P0=propagate(states[idx],covs[idx],times[idx],start,tt,uu)
        for method,approx in variants:
            state=s0.copy();P=P0.copy();previous=start;qs=[];support=[]
            for step in range(1,6):
                end=start+step
                state,P=propagate(state,P,previous,end,tt,uu)
                P,meta=field.forecast(state,P,method,approx)
                qs.append(meta['q_any']);support.append(meta['supported']);previous=end
                if step not in [1,3,5]:continue
                j=np.searchsorted(times,end,side='right')-1
                # Score the reference estimator at the exact forecast endpoint by
                # propagating its last available posterior through the remaining dt.
                actual_state,actual_P=propagate(states[j],covs[j],times[j],end,tt,uu)
                gx,gy=truth.at([end]);error=actual_state[:2]-[gx[0],gy[0]]
                if not np.isfinite(error).all():continue
                forecasts.append(dict(start_s=float(start),horizon_s=step,method=method,approximation=approx,
                    predicted_trace_m2=float(np.trace(P[:2,:2])),actual_filter_trace_m2=float(np.trace(actual_P[:2,:2])),
                    actual_squared_error_m2=float(error@error),mean_q_any=float(np.mean(qs)),
                    supported_fraction=float(np.mean(support))))
    # Independent Q diagnostic: truth initializes evaluation windows only. Main
    # estimator and field inputs remain truth-free and robot Q is never refitted.
    qcheck=[]
    for horizon in [1.,3.,5.]:
        errors=[];predicted=[]
        for start in np.arange(times[0],times[-1]-horizon,horizon):
            gx,gy=truth.at([start]);yaw=truth.yaw_at([start])[0]
            if not np.isfinite([gx[0],gy[0],yaw]).all():continue
            state,P=propagate([gx[0],gy[0],yaw],np.zeros((3,3)),start,start+horizon,tt,uu)
            ex,ey=truth.at([start+horizon]);e=state[:2]-[ex[0],ey[0]]
            if not np.isfinite(e).all():continue
            errors.append(e);predicted.append(P[:2,:2])
        if errors:qcheck.append(dict(horizon_s=horizon,score=score(np.array(errors),np.array(predicted),['one_run']*len(errors))))
    result=dict(run=entry['run'],task=entry['task'],seed=entry['seed'],accounting=accounting,analysis_inputs=inputs,
        replay=comparisons,availability=availability,temporal=temporal,forecasts=forecasts,Q_diagnostic=qcheck,
        coarse_batch_median_interval_s=float(np.median(np.diff([b['t'] for b in coarse]))) if len(coarse)>1 else None,
        selected_coarse_batches=len(coarse),field_sha256=digest(FIELD_OUT/'field.joblib'),
        limitations=['Capture-time idealized replay; arrival latency and live refusal policy are not reproduced.',
            'Recorded controls prescribe the future route; no closed-loop field-based planning claim.',
            'Forecast uses 1Hz opportunities; actual first-fresh >=1s policy may be slower and is reported.',
            'One trajectory is one experimental unit; forecast windows and residual lag pairs are dependent.',
            'Reference transform may use receipt sim clock; timestamp provenance is reported.',
            'Reference-pose queries and Q-window initialization are evaluation-only diagnostics.',
            'The optional bias process uses previously frozen diagnostic parameters, not identified parameters.'])
    writejson(result_path,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=FIELD_OUT/'driving');p.add_argument('--max-reference-gap-s',type=float);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    entries,pending=frozen_runs(args.output)
    for entry in entries:
        if entry['status']!='completed':continue
        print('analyze',entry['key'],flush=True);analyze(entry,args.output,max_reference_gap_s=args.max_reference_gap_s)
    print('pending',pending,flush=True)


if __name__=='__main__':main()
