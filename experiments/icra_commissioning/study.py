#!/usr/bin/env python3
"""Manifest-bound development comparison for the frozen existing NN mean.

Run freeze before fit. No directory globs select evidence. The old TEST tiles have
already informed development: results are development validation, never a final test.
"""
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import joblib
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src/reliability')]
from reliability.learned_box_correction import LearnedBoxCorrection
from model import CameraModel, ray_basis

CAPTURE = 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
ARTIFACT = 'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'
OUT = REPO/'logs/studies/icra_commissioning_20260905'
KINDS = ('constant', 'diagonal', 'isotropic', 'geometry', 'spatial', 'confidence')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open() as f: return list(csv.DictReader(f))
def writejson(p, d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def tile(r): return f"{math.floor(float(r['robot_x'])/2)}:{math.floor(float(r['robot_y'])/2)}"

def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    path=out/'manifest.json'
    if path.exists(): raise RuntimeError('manifest already frozen')
    files=[ARTIFACT]+[f'{CAPTURE}/{x}' for x in ['capture_manifest.json','capture_index.csv',
         'observation_interpretations.csv','observation_interpretations_manifest.json',
         'bias_update_interpretations.csv','bias_update_interpretations_manifest.json']]
    rows=readcsv(REPO/CAPTURE/'bias_update_interpretations.csv')
    # Assignment is made on all opportunities, before reading errors or ranking models.
    tiles=sorted({tile(r) for r in rows if r['split']=='test'})
    rng=np.random.default_rng(509); rng.shuffle(tiles)
    roles={t:['covariance_fit','selection','evaluation'][i%3] for i,t in enumerate(tiles)}
    writejson(path,dict(schema=1,status='development_only',files={p:digest(REPO/p) for p in files},
       capture=CAPTURE,mean_artifact=ARTIFACT,roles=roles,seed=509,
       grouping='2 m tile; all headings/cameras/repeats together; cross-role hit image hashes excluded',
       inference='same installation, held-out spatial tiles, previously examined development data',
       Q=dict(process_noise_xy=0.01,process_noise_theta=0.02,frozen=True),
       covariance_models=list(KINDS),no_final_test_claim=True))
    print(path)

def load(out):
    manifest=json.loads((out/'manifest.json').read_text())
    for p,h in manifest['files'].items():
        if digest(REPO/p)!=h: raise RuntimeError(f'input changed: {p}')
    rows=readcsv(REPO/manifest['capture']/'bias_update_interpretations.csv')
    capture={ (r['camera_id'],r['pose_id'],r['repetition_id']):r for r in
              readcsv(REPO/manifest['capture']/'capture_index.csv') }
    mean=LearnedBoxCorrection(REPO/manifest['mean_artifact'])
    hash_roles=defaultdict(set)
    for r in rows:
        role='mean_train' if r['split']=='train' else manifest['roles'][tile(r)]
        if r['raw_valid']=='1': hash_roles[r['image_sha1']].add(role)
    data=[]; counts=Counter(); records=[]
    for r in rows:
        camera=r['camera_id']; role='mean_train' if r['split']=='train' else manifest['roles'][tile(r)]
        cap=capture[(camera,r['pose_id'],r['repetition_id'])]
        reason='accepted' if r['raw_valid']=='1' else 'no_valid_detection'
        if reason=='accepted' and len(hash_roles[r['image_sha1']])>1: reason='cross_role_duplicate_image'
        counts[(role,reason)]+=1
        record=dict(run=manifest['capture'],camera=camera,frame=f"{r['pose_id']}:{r['repetition_id']}",
          capture_stamp=cap['image_stamp_s'],image=r['image'],image_sha1=r['image_sha1'],
          group=tile(r),role=role,accepted=reason=='accepted',reason=reason,
          reference='commanded simulator model origin, static capture; actual pose verification absent',
          geometry_id=manifest['files'][f'{CAPTURE}/capture_manifest.json'])
        if reason!='accepted': records.append(record); continue
        raw=np.array([float(r['raw_x']),float(r['raw_y'])]); box=[float(r[k]) for k in ['x0','y0','x1','y1']]
        z=np.asarray(mean.correct(camera,raw,box,float(r['confidence'])))
        if z.shape!=(2,) or not np.isfinite(z).all(): raise ValueError('mean failed')
        # Reproduce existing mean before changing uncertainty.
        np.testing.assert_allclose(z,[float(r['nn_x']),float(r['nn_y'])],atol=1e-9,rtol=0)
        truth=np.array([float(r['robot_x']),float(r['robot_y'])]); geom=mean._geometry[camera]
        data.append(dict(z=z,truth=truth,raw=raw,camera=camera,role=role,group=tile(r),
          basis=ray_basis(raw,np.array(geom['xy'])),distance=float(np.linalg.norm(raw-geom['xy'])),
          confidence=float(r['confidence']),heading=float(r['robot_yaw']),frame=record['frame']))
        record.update(raw_xy=raw.tolist(),corrected_xy=z.tolist(),bbox=box,confidence=float(r['confidence']))
        records.append(record)
    with (out/'records.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    return data,{f'{a}/{b}':n for (a,b),n in counts.items()}

def score(e,R,groups):
    v=np.linalg.solve(R,e[...,None])[...,0]; nees=np.einsum('ni,ni->n',e,v)
    nll=.5*(np.linalg.slogdet(R)[1]+nees+2*np.log(2*np.pi))
    err=100*np.linalg.norm(e,axis=1)
    group_scores=[np.mean(nll[np.asarray(groups)==g]) for g in sorted(set(groups))]
    rng=np.random.default_rng(509)
    boot=np.mean(rng.choice(group_scores,(1000,len(group_scores))),axis=1)
    return dict(n=len(e),groups=len(set(groups)),median_cm=float(np.median(err)),p95_cm=float(np.quantile(err,.95)),
      rmse_cm=float(np.sqrt(np.mean(err**2))),bias_cm=(100*e.mean(axis=0)).tolist(),
      gaussian_nll=float(nll.mean()),group_mean_nll=float(np.mean(group_scores)),
      group_bootstrap_nll_ci95=np.quantile(boot,[.025,.975]).tolist(),
      coverage={str(p):float(np.mean(nees<=chi2.ppf(p,2))) for p in [.5,.8,.9,.95,.99]},
      mean_mahalanobis2=float(nees.mean()),rms_sigma_cm=float(100*np.sqrt(np.mean(np.trace(R,axis1=1,axis2=2))/2)))

def fit(out):
    start=time.perf_counter(); data,counts=load(out)
    models={}; scores={}; covrows=[]
    cameras=sorted({r['camera'] for r in data})
    for c in cameras:
        fitrows=[r for r in data if r['camera']==c and r['role']=='covariance_fit']
        bias=np.mean([r['z']-r['truth'] for r in fitrows],axis=0)
        for kind in KINDS:
            models[c,kind]=CameraModel(kind).fit(fitrows,bias)
            m=models[c,kind]
            covrows.append(dict(camera=c,model=kind,n=m.n,bias_m=m.bias.tolist(),
                covariance_m2=m.raw_covariance.tolist(),second_moment_m2=m.second_moment.tolist(),
                cell_counts=m.counts,cell_remaining_mean_m={str(k):v.tolist() for k,v in m.means.items()}))
    tuning={}
    selection=[r for r in data if r['role']=='selection']
    # Tune the same two global regularizers for every arm, only on selection tiles.
    # This is working-Gaussian calibration; the centered fit covariance is kept above.
    for kind in KINDS:
        candidates=[]
        for tau in [0., .25, .5, .75, 1.]:
            e=[];Rs=[]
            for r in selection:
                m=models[r['camera'],kind];m.scale=1.;m.isotropic_shrink=tau
                z,R=m.predict([r]);e.append(z[0]-r['truth']);Rs.append(R[0])
            e=np.asarray(e);Rs=np.asarray(Rs)
            v=np.einsum('ni,ni->n',e,np.linalg.solve(Rs,e[...,None])[...,0])
            groups=np.asarray([r['group'] for r in selection])
            scale=max(.05,float(np.mean([v[groups==g].mean()/2 for g in set(groups)])))
            s=score(e,Rs*scale,groups)
            candidates.append((s['group_mean_nll'],tau,scale))
        loss,tau,scale=min(candidates)
        for c in cameras:models[c,kind].scale=scale;models[c,kind].isotropic_shrink=tau
        tuning[kind]=dict(scale=scale,isotropic_shrink=tau,selection_group_nll=loss)
    for role in ['selection','evaluation']:
        rows=[r for r in data if r['role']==role]
        for kind in KINDS:
            e=[]; R=[]
            for r in rows:
                z,cov=models[r['camera'],kind].predict([r]);e.append(z[0]-r['truth']);R.append(cov[0])
            scores[f'{role}/{kind}']=score(np.array(e),np.array(R),[r['group'] for r in rows])
    selected=min(KINDS,key=lambda k:scores[f'selection/{k}']['group_mean_nll'])
    joblib.dump(models,out/'models.joblib')
    # Repeat commissioning-budget subsets by whole tiles; same evaluation data and mean.
    budget=[]; fitrows=[r for r in data if r['role']=='covariance_fit']
    groups=sorted({r['group'] for r in fitrows}); evalrows=[r for r in data if r['role']=='evaluation']
    for frac in [.25,.5,1.]:
        for seed in range(5):
            chosen=set(np.random.default_rng(seed).choice(groups,max(5,int(frac*len(groups))),replace=False))
            sub=[r for r in fitrows if r['group'] in chosen]
            if any(sum(r['camera']==c for r in sub)<15 for c in cameras):continue
            mm={c:CameraModel('constant').fit([r for r in sub if r['camera']==c]) for c in cameras}
            e=[];R=[]
            for r in evalrows:
                z,C=mm[r['camera']].predict([r]);e.append(z[0]-r['truth']);R.append(C[0])
            budget.append(dict(fraction=frac,seed=seed,tiles=len(chosen),poses=len({tuple(r['truth']) for r in sub}),
                score=score(np.array(e),np.array(R),[r['group'] for r in evalrows])))
    result=dict(status='development_static_comparison',counts=counts,scores=scores,
        selected_on_selection=selected,tuning=tuning,fit_seconds=time.perf_counter()-start,models=covrows,budget=budget,
        limitations=['Static references are commanded poses; no independent physical reference.',
        'Old field captures lack source_batch_id and verified settle timestamps.',
        'Residual distribution is over held-out configurations, not independent repeated noise.',
        'No new-installation or final-evaluation claim; old held-out tiles already examined.',
        'Conditional residual means remain: centered covariance alone does not remove them.',
        'Budget excludes fixed NN training cost (3249 boxes); report that cost separately.'])
    writejson(out/'results.json',result)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for kind in KINDS:
        s=scores[f'evaluation/{kind}'];axes[0].plot([float(k) for k in s['coverage']],list(s['coverage'].values()),'-o',label=kind)
    axes[0].plot([.5,1],[.5,1],'k--');axes[0].set(xlabel='Nominal ellipse probability',ylabel='Empirical containment',title='Held-out development tiles')
    axes[0].legend(fontsize=7)
    for frac in [.25,.5,1.]:
        b=[v for v in budget if v['fraction']==frac]
        if b:axes[1].scatter([v['poses'] for v in b],[100*v['score']['coverage']['0.95'] for v in b])
    axes[1].axhline(95,color='k',ls='--');axes[1].set(xlabel='Covariance-fit positions (NN fixed)',ylabel='95% containment (%)',title='Constant covariance budget; 5 subsets')
    fig.savefig(out/'calibration_budget.pdf');fig.savefig(out/'calibration_budget.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    rows=[r for r in data if r['role']=='evaluation']
    for ax,which,title in zip(axes,['raw','z'],['Raw bbox → floor','Frozen NN reference estimate']):
        xy=np.array([r['truth'] for r in rows]);e=np.array([r[which]-r['truth'] for r in rows]);
        ax.quiver(xy[:,0],xy[:,1],e[:,0],e[:,1],angles='xy',scale_units='xy',scale=1,width=.002)
        ax.set(xlabel='Map x (m)',ylabel='Map y (m)',title=title,aspect='equal')
    fig.suptitle(f'Observed residual vectors; {len(rows)} readings (not local bias estimates)')
    fig.savefig(out/'residual_fields.pdf');fig.savefig(out/'residual_fields.png',dpi=180);plt.close(fig)
    print(json.dumps({'selected':selected,'evaluation':{k:scores[f'evaluation/{k}'] for k in KINDS}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','fit']);p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();(freeze if a.action=='freeze' else fit)(a.output)
