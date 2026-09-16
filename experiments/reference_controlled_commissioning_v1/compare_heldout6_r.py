#!/usr/bin/env python3
"""Fit provisional R on train-12 residuals and score the six held-out drives."""
from __future__ import annotations
import argparse, json, math, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compare_provisional_train12_r import sha, spd, scatter, scores

MODEL="box_mlp_visibility_residual"; KEY=MODEL+"_prediction_ray_m"

def local_query(train_xy,train_r,train_d,train_c,query_xy,query_c,k=120,bw=.75):
    spatial=np.empty((len(query_xy),2,2)); hier=np.empty_like(spatial); mean=np.zeros((len(query_xy),2))
    for cam in sorted(set(query_c.tolist())):
        tr=np.flatnonzero(train_c==cam); qu=np.flatnonzero(query_c==cam)
        points=train_xy[tr]; values=train_r[tr]; drives=train_d[tr]; tree=cKDTree(points); glob=scatter(values)
        distance,neighbor=tree.query(query_xy[qu],k=min(k,len(points)))
        if neighbor.ndim==1: neighbor=neighbor[:,None]; distance=distance[:,None]
        for qi,dist,local in zip(qu,distance,neighbor):
            w=np.exp(-.5*(dist/bw)**2); v=values[local]; d=drives[local]; capped=w.copy()
            for drive in np.unique(d):
                use=d==drive
                if capped[use].sum()>0: capped[use]/=capped[use].sum()
            sc=sum(wi*np.outer(e,e) for wi,e in zip(capped,v))
            spatial[qi]=spd((sc+4*glob)/(capped.sum()+4))
            means=[]; covs=[]; wd=[]
            for drive in np.unique(d):
                use=d==drive; ww=w[use]
                if ww.sum()<=1e-12: continue
                vv=v[use]; m=np.average(vv,axis=0,weights=ww)
                cov=sum(wi*np.outer(e-m,e-m) for wi,e in zip(ww,vv))/ww.sum()
                means.append(m); covs.append(spd(cov)); wd.append(min(1.,ww.sum()/5.))
            wd=np.asarray(wd); n=wd.sum(); means=np.asarray(means); m=np.average(means,axis=0,weights=wd)
            within=np.average(np.asarray(covs),axis=0,weights=wd)
            between=sum(wi*np.outer(e-m,e-m) for wi,e in zip(wd,means))/n
            kappa=1+n; nu=5+n; psi=glob*2+n*(within+between)+(n/kappa)*np.outer(m,m)
            deg=max(nu-1,3.01); scale=spd(((kappa+1)/(kappa*deg))*psi)
            mean[qi]=n*m/kappa; hier[qi]=spd(scale*deg/(deg-2))
    return spatial,mean,hier

def main():
    p=argparse.ArgumentParser(); p.add_argument('--comparison-root',type=Path,required=True)
    p.add_argument('--dataset-contract',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); stage=out.with_name(out.name+'.incomplete')
    if out.exists() or stage.exists(): raise FileExistsError(out)
    stage.mkdir(parents=True)
    pred=a.comparison_root/'candidate_predictions.npz'; report=a.comparison_root/'correction_comparison.json'
    meta=json.loads(report.read_text()); assert meta['selected_correction']==MODEL
    with np.load(pred,allow_pickle=False) as pz:
        data={x:np.asarray(pz[x]) for x in pz.files}
    part=data['historical_partition']; train=np.flatnonzero(np.isin(part,['fit','development'])); test=np.flatnonzero(part=='audit')
    target=np.asarray(data['target_ray_m'],float); prediction=np.asarray(data[KEY],float); residual=target-prediction
    with np.load(a.dataset_contract,allow_pickle=False) as d:
        lookup={(str(x),str(c),str(f),int(t)):i for i,(x,c,f,t) in enumerate(zip(d['drive'],d['camera'],d['source_frame_id'],d['stamp_ns']))}
        idx=np.asarray([lookup[(str(x),str(c),str(f),int(t))] for x,c,f,t in zip(data['drive'],data['camera'],data['source_frame_id'],data['stamp_ns'])])
        xy=np.asarray(d['raw'][idx],float)
    trr=residual[train]; ter=residual[test]; trc=data['camera'][train]; tec=data['camera'][test]; trd=data['drive'][train]; ted=data['drive'][test]
    global_cov=scatter(trr); gp=np.repeat(global_cov[None],len(test),axis=0); cp=np.empty_like(gp)
    for cam in sorted(set(tec.tolist())): cp[tec==cam]=scatter(trr[trc==cam])
    spatial,hm,hc=local_query(xy[train],trr,trd,trc,xy[test],tec)
    methods={'R0_global_full':scores(ter,gp,ted),'R1_per_camera_full':scores(ter,cp,ted),
             'R2_spatial_residual':scores(ter,spatial,ted),'R3_hierarchical_predictive':scores(ter-hm,hc,ted)}
    result={'schema':'provisional_heldout6_matched_R_comparison.v1','status':'complete_secondary_heldout_evaluation',
      'created_utc':datetime.now(timezone.utc).isoformat(),'model_id':MODEL,
      'fit_population':{'rows':len(train),'drives':12},'evaluation_population':{'rows':len(test),'drives':6},
      'warning':'The six drives are held out from fitting, but they previously selected the correction family; this is secondary, not fresh final evidence.',
      'methods':methods,'inputs':{'predictions':str(pred.resolve()),'predictions_sha256':sha(pred),
      'dataset_contract':str(a.dataset_contract.resolve()),'dataset_contract_sha256':sha(a.dataset_contract)},
      'implementation_sha256':sha(Path(__file__))}
    (stage/'comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); os.replace(stage,out)
    print(json.dumps({k:{x:v[x] for x in ('equal_drive_mean_nll','equal_drive_coverage','equal_drive_median_95_ellipse_area_cm2','mean_mahalanobis2')} for k,v in methods.items()},indent=2))
if __name__=='__main__': main()
