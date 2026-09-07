#!/usr/bin/env python3
"""Capture-time, fixed-Q replay on an explicit frozen list, through aligned.py.

The log samples odometry at 10 Hz; this replay uses zero-order-held measured noisy
velocities. It is a causal capture-time comparison, not reproduction of online delay
handling. All methods see the same logged preselected observation population, with
no NIS gate. Camera misses absent from legacy fusion logs are not invented.
"""
import argparse,csv,json,sys,math,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/p) for p in ['src/planning','src/reliability','src/experiments','src/unav_common',
  'experiments/fusion_on_fixed_routes','experiments/camera_observation_characterization']]
import aligned
from derive_interpretations import camera_models
from reliability.learned_box_correction import LearnedBoxCorrection
from planning.core.dynamics import unicycle_step,unicycle_jacobian,unicycle_process_noise
from model import update,ray_basis,expected_posterior
from study import OUT,ARTIFACT,CAPTURE,KINDS,digest,readcsv,writejson,score

REQUIRED=['run_manifest.json','run_summary.json','experiment.csv','fusion_observations.csv','correction_assimilations.csv']

def freeze(path,runs):
    if path.exists():raise RuntimeError('selection exists')
    entries=[]
    for name in runs:
        p=Path(name).resolve(); m=json.loads((p/'run_manifest.json').read_text())
        if m['logging_schema_version']<4:raise ValueError('schema <4')
        entries.append(dict(run=str(p.relative_to(REPO)),files={f:digest(p/f) for f in REQUIRED},
            role='development_replay',seed=m['seed'],task=m['task']))
    writejson(path,dict(status='diagnostic_replay_not_paper_fusion_selection',runs=entries,
        Q=dict(xy=.01,theta=.02),init='declared task_start_pose; diag(0.05 m,0.05 m,5 deg)^2',
        gate='same logged observation population; no replay NIS gate',
        rate='full unique capture rate and common minimum interval 1 s per camera'))


def load(entry,mean,geometry):
    p=REPO/entry['run']
    for f,h in entry['files'].items():
        if digest(p/f)!=h:raise ValueError(f'changed run artifact: {p/f}')
    m=json.loads((p/'run_manifest.json').read_text())
    if not m['use_odom_for_predict'] or m['odom_topic']!='/odom_noisy':raise ValueError('wrong odometry input')
    if (m['process_noise_xy'],m['process_noise_theta'])!=(.01,.02):raise ValueError('Q changed')
    # Accounting checked independently of numerical accuracy.
    ass=aligned.assimilations(p);obs=aligned.observations(p)
    bids={o['source_batch_id'] for o in obs}
    if bids!={a['source_batch_id'] for a in ass}:raise ValueError('correction accounting mismatch')
    if any(a['status'] not in ['accepted','accepted_bootstrap','reanchored','rejected','dropped'] or
           (a['status'] in ['rejected','dropped'] and not a['reason']) for a in ass):raise ValueError('unexplained outcome')
    table=aligned.rows(p); truth=aligned.truth_series(p,table)
    odom={}
    for r in table:
        try:t=float(r['odom_noisy_stamp']);u=np.array([float(r['odom_noisy_v']),float(r['odom_noisy_w'])])
        except (KeyError,ValueError):continue
        if math.isfinite(t) and np.isfinite(u).all():odom.setdefault(t,u)
    if not odom:raise ValueError('no measured odometry')
    rawrows={}
    for r in readcsv(p/'fusion_observations.csv'):
        rawrows.setdefault((r['camera'],round(float(r['obs_stamp']),6)),r)
    readings=[]
    for o in aligned.readings(p,admitted_only=False):
        r=rawrows[(o['camera'],round(o['obs_stamp'],6))];cam='camera_'+o['camera']
        raw=np.array([float(r['raw_obs_x']),float(r['raw_obs_y'])])
        if not np.isfinite(raw).all():raise ValueError('raw reading absent')
        u,v,visible=geometry[cam].world_to_pixel(*raw)
        box=[u-o['bbox_w_px']/2,v-o['bbox_h_px'],u+o['bbox_w_px']/2,v]
        z=mean.correct(cam,raw,box,o['conf'])
        if z is None:raise ValueError('mean correction cannot be reproduced')
        readings.append(dict(t=o['obs_stamp'],camera=cam,z=np.asarray(z),raw=raw,truth=o['truth'],
          batch=o['source_batch_id'],basis=ray_basis(raw,np.array(mean._geometry[cam]['xy'])),
          distance=float(np.linalg.norm(raw-mean._geometry[cam]['xy'])),confidence=o['conf'],
          original_z=np.array([o['obs_x'],o['obs_y']]),original_R=o['cov']))
    return m,truth,odom,readings,ass


def run_filter(m,truth,odom,readings,models,kind,cameras,interval=0,*,prediction_times=None):
    """Replay a camera mask on an optional common motion-propagation event grid.

    ``prediction_times`` adds prediction-only events, never observations. Supplying
    every original capture time holds the numerical process-noise integration grid
    fixed when comparing camera subsets. Omitting it preserves historical calls.
    """
    chosen=[];last={}
    for r in readings:
        if r['camera'] not in cameras:continue
        if r['t']-last.get(r['camera'],-np.inf)<interval:continue
        chosen.append(r);last[r['camera']]=r['t']
    bytime=defaultdict(list)
    for r in chosen:bytime[r['t']].append(r)
    grid=set() if prediction_times is None else set(prediction_times)
    if not all(math.isfinite(t) for t in grid):raise ValueError('nonfinite prediction event')
    times=sorted(set(odom)|set(bytime)|grid)
    start=m['task_start_pose']
    state=np.array([start[k] for k in ['x','y','yaw']],float); P=np.diag([.05,.05,np.deg2rad(5)])**2
    persistent=kind=='confidence_bias'
    if persistent:
        # Focused diagnostic: half the marginal camera error is a 2 s stationary
        # latent process in normalized residual coordinates. Robot Q is unchanged.
        state=np.r_[state,np.zeros(2*len(cameras))]
        big=np.eye(len(state))*.5;big[:3,:3]=P;P=big
    control=np.zeros(2);previous=times[0];records=[];innovations=[]
    seen=set()
    for t in times:
        if t<previous:raise ValueError('noncausal time')
        dt=t-previous
        if dt:
            F=unicycle_jacobian(state[:3],control,dt)
            Q=unicycle_process_noise(.01,.02,dt,theta=state[2],v=control[0])
            if persistent:
                phi=np.exp(-dt/2.)
                FF=np.eye(len(state))*phi;FF[:3,:3]=F
                QQ=np.eye(len(state))*.5*(1-phi**2);QQ[:3,:3]=Q
                P=FF@P@FF.T+QQ
                state[:3]=unicycle_step(state[:3],control,dt);state[3:]*=phi
            else:
                P=F@P@F.T+Q;state=unicycle_step(state,control,dt)
        # Measurement before motion samples stamped at t affect the next interval.
        for r in bytime.get(t,[]):
            key=(r['camera'],t)
            if key in seen:raise ValueError('duplicate physical camera update')
            seen.add(key)
            if kind=='recorded':z,R=r['original_z'],r['original_R']
            else:
                zz,RR=models[r['camera'],'confidence' if persistent else kind].predict([r]);z,R=zz[0],RR[0]
            pre=state.copy();Pm=P.copy()
            if persistent:
                H=np.zeros((2,len(state)));H[:,:2]=np.eye(2)
                b=3+2*cameras.index(r['camera']);H[:,b:b+2]=np.linalg.cholesky(R)
                fast=R*.5;S=H@P@H.T+fast;nu=z-H@state
                K=np.linalg.solve(S,H@P).T;J=np.eye(len(state))-K@H
                state=state+K@nu;P=J@P@J.T+K@fast@K.T
                nis=float(nu@np.linalg.solve(S,nu))
            else:state,P,nis=update(state,P,z,R)
            state[2]=(state[2]+np.pi)%(2*np.pi)-np.pi
            innovations.append(dict(t=t,camera=r['camera'],nis=nis,
                innovation=(nu if persistent else z-pre[:2]).tolist(),
                innovation_covariance=(S if persistent else Pm[:2,:2]+R).tolist(),
                observation_H=(H if persistent else np.eye(len(pre))[:2]).tolist(),
                pre_state=pre.tolist(),pre_covariance=Pm.tolist(),R=R.tolist(),
                independent_R=(fast if persistent else R).tolist(),batch=r['batch']))
        if t in odom:control=odom[t]
        gx,gy=truth.at([t]);yaw=truth.yaw_at([t])[0]
        # Every arm is scored on the same odometry-stamp grid, including subsampling
        # and single-camera arms. Camera event density must not change metric weights.
        if t in odom and np.isfinite([gx[0],gy[0],yaw]).all():
            err=state[:2]-[gx[0],gy[0]]
            records.append(dict(t=t,state=state.copy(),P=P.copy(),error=err,
                yaw_error=(state[2]-yaw+np.pi)%(2*np.pi)-np.pi))
        previous=t
    e=np.array([r['error'] for r in records]);C=np.array([r['P'][:2,:2] for r in records])
    s=score(e,C,['one_drive']*len(e));s['updates']=len(seen)
    s['yaw_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean([r['yaw_error']**2 for r in records]))))
    s['pre_gate_mean_nis']=float(np.mean([r['nis'] for r in innovations])) if innovations else None
    return s,records,innovations


def main(selection,out):
    manifest=json.loads(selection.read_text()); models=joblib.load(out/'models.joblib')
    mean=LearnedBoxCorrection(REPO/ARTIFACT)
    geometry=camera_models(json.loads((REPO/CAPTURE/'capture_manifest.json').read_text()))
    allscores=[];temporal=[];cross=[]
    for entry in manifest['runs']:
        m,truth,odom,readings,ass=load(entry,mean,geometry)
        cams=sorted({r['camera'] for r in readings})
        # Residual dependence is computed at capture time, within this run only.
        residual_by_batch=defaultdict(dict)
        for c in cams:
            rows=[r for r in readings if r['camera']==c]
            z,R=models[c,'constant'].predict(rows)
            e=z-np.array([r['truth'] for r in rows])
            white=np.linalg.solve(np.linalg.cholesky(R),e[...,None])[...,0]
            for r,w in zip(rows,white):residual_by_batch[r['batch']][c]=w
            for lag in [1,2,5,10,20]:
                if len(rows)<=lag+3:continue
                dt=np.array([rows[i+lag]['t']-rows[i]['t'] for i in range(len(rows)-lag)])
                # exclude long visibility gaps when interpreting nominal lag correlation
                keep=dt<max(2.,lag*.5)
                if keep.sum()<10:continue
                rho=[float(np.corrcoef(white[:-lag,j][keep],white[lag:,j][keep])[0,1]) for j in range(2)]
                temporal.append(dict(run=entry['run'],camera=c,lag=lag,pairs=int(keep.sum()),
                    median_lag_s=float(np.median(dt[keep])),correlation=rho))
        for i,a in enumerate(cams):
            for b in cams[i+1:]:
                paired=[(v[a],v[b]) for v in residual_by_batch.values() if a in v and b in v]
                if len(paired)<10:continue
                aa,bb=np.asarray(paired).transpose(1,0,2)
                cross.append(dict(run=entry['run'],cameras=[a,b],n=len(paired),
                    correlation=np.corrcoef(aa.T,bb.T)[:2,2:].tolist()))
        fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
        for kind in ['recorded',*KINDS,'confidence_bias']:
            for interval in [0,1.]:
                s,records,innov=run_filter(m,truth,odom,readings,models,kind,cams,interval)
                allscores.append(dict(run=entry['run'],kind=kind,cameras='all',interval_s=interval,score=s))
                if interval==0:
                    axes[0].plot([r['t'] for r in records],[100*np.linalg.norm(r['error']) for r in records],label=kind,alpha=.8)
                    axes[1].plot([r['t'] for r in records],[100*np.sqrt(np.trace(r['P'][:2,:2])) for r in records],label=kind,alpha=.8)
                    if kind in ['constant','confidence']:
                        np.savez_compressed(out/f"replay_{m['run_id']}_{kind}.npz",time=[r['t'] for r in records],
                            state=[r['state'] for r in records],covariance=[r['P'] for r in records],error=[r['error'] for r in records])
                        writejson(out/f"innovations_{m['run_id']}_{kind}.json",innov)
        for c in cams:
            for kind in ['constant','confidence']:
                s,_,_=run_filter(m,truth,odom,readings,models,kind,[c])
                allscores.append(dict(run=entry['run'],kind=kind,cameras=c,interval_s=0,score=s))
        axes[0].set(ylabel='Belief position error (cm)',title='Capture-time development replay; measured noisy odometry; Q frozen')
        axes[1].set(ylabel='Position covariance sqrt(trace) (cm)',xlabel='Simulation capture time (s)')
        axes[0].legend(ncol=4,fontsize=7)
        fig.savefig(out/f"replay_{m['run_id']}.pdf");fig.savefig(out/f"replay_{m['run_id']}.png",dpi=180);plt.close(fig)
    writejson(out/'replay_results.json',dict(status='diagnostic_fixed_Q_replay',selection_sha256=digest(selection),
      model_sha256=digest(out/'models.joblib'),scores=allscores,temporal=temporal,cross_camera=cross,
      limitations=['Per-camera raw boxes reconstructed from raw ground point and box dimensions.',
        'Legacy fusion log excludes some refused camera opportunities; availability not identifiable here.',
        'No live latency or NIS gating; logged 10 Hz measured odometry is coarser than online history.',
        'Recorded arm uses original logged observation mean and R; it is not a covariance-only arm.',
        'Frames are not independent replicates; per-run scores only; no significance with one drive.']))
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for c in sorted({r['camera'] for r in temporal}):
        rr=[r for r in temporal if r['camera']==c]
        ax.plot([r['median_lag_s'] for r in rr],[r['correlation'][0] for r in rr],'-o',label=c)
    ax.axhline(0,color='k',lw=.7);ax.set(xlabel='Within-run lag (s)',ylabel='Whitened x residual correlation',title='Camera residual persistence; no cross-run pairs');ax.legend(fontsize=7)
    fig.savefig(out/'temporal.pdf');fig.savefig(out/'temporal.png',dpi=180);plt.close(fig)
    print('saved',out/'replay_results.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--selection',type=Path,default=OUT/'driving_manifest.json');p.add_argument('--run',action='append');p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();freeze(a.selection,a.run) if a.action=='freeze' else main(a.selection,a.output)
