#!/usr/bin/env python3
"""Compare provisional R methods on frozen train-12 in-sample residuals."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

CHI2 = {"50": 1.38629436112, "90": 4.60517018599,
        "95": 5.99146454711, "99": 9.21034037198}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def spd(x, floor=1e-6):
    x=.5*(x+x.T); val,vec=np.linalg.eigh(x)
    return (vec*np.maximum(val,floor))@vec.T
def scatter(values): return spd(values.T@values/max(len(values),1))

def scores(residual, covariance, drive):
    inv=np.linalg.inv(covariance)
    d2=np.einsum("ni,nij,nj->n",residual,inv,residual)
    det=np.maximum(np.linalg.det(covariance),1e-18)
    nll=.5*(2*math.log(2*math.pi)+np.log(det)+d2)
    area=math.pi*CHI2["95"]*np.sqrt(det)
    per={d:{"n":int(np.sum(drive==d)),"mean_nll":float(nll[drive==d].mean()),
            "coverage":{k:float(np.mean(d2[drive==d]<=v)) for k,v in CHI2.items()},
            "median_95_ellipse_area_cm2":float(1e4*np.median(area[drive==d]))}
         for d in sorted(set(drive.tolist()))}
    return {"n":len(residual),"drive_count":len(per),
        "equal_drive_mean_nll":float(np.mean([x["mean_nll"] for x in per.values()])),
        "equal_drive_coverage":{k:float(np.mean([x["coverage"][k] for x in per.values()])) for k in CHI2},
        "equal_drive_median_95_ellipse_area_cm2":float(np.mean([x["median_95_ellipse_area_cm2"] for x in per.values()])),
        "mean_mahalanobis2":float(d2.mean()),"per_drive":per}

def local_predictions(xy,residual,drive,camera,k=120,bandwidth=.75):
    spatial=np.empty((len(residual),2,2)); hier=np.empty_like(spatial); mean=np.zeros_like(residual)
    for cam in sorted(set(camera.tolist())):
        members=np.flatnonzero(camera==cam); points=xy[members]; values=residual[members]; ds=drive[members]
        global_cov=scatter(values); tree=cKDTree(points)
        _,neighbors=tree.query(points,k=min(k+1,len(points)))
        if neighbors.ndim==1: neighbors=neighbors[:,None]
        for row,(global_index,candidates) in enumerate(zip(members,neighbors)):
            local=candidates[candidates!=row][:k]
            distance=np.linalg.norm(points[local]-points[row],axis=1)
            weight=np.exp(-.5*(distance/bandwidth)**2)
            local_values=values[local]; local_drives=ds[local]
            capped=weight.copy()
            for d in np.unique(local_drives):
                use=local_drives==d
                if capped[use].sum()>0: capped[use]/=capped[use].sum()
            sc=sum(w*np.outer(e,e) for w,e in zip(capped,local_values))
            spatial[global_index]=spd((sc+4*global_cov)/(capped.sum()+4))
            means=[]; covs=[]; wd=[]
            for d in np.unique(local_drives):
                use=local_drives==d; w=weight[use]
                if w.sum()<=1e-12: continue
                v=local_values[use]; m=np.average(v,axis=0,weights=w)
                c=sum(wi*np.outer(e-m,e-m) for wi,e in zip(w,v))/w.sum()
                means.append(m); covs.append(spd(c)); wd.append(min(1.,w.sum()/5.))
            wd=np.asarray(wd); n=wd.sum(); means=np.asarray(means)
            m=np.average(means,axis=0,weights=wd); within=np.average(np.asarray(covs),axis=0,weights=wd)
            between=sum(w*np.outer(v-m,v-m) for w,v in zip(wd,means))/n
            kappa=1+n; nu=5+n; psi=global_cov*2+n*(within+between)+(n/kappa)*np.outer(m,m)
            degrees=max(nu-1,3.01); scale=spd(((kappa+1)/(kappa*degrees))*psi)
            mean[global_index]=n*m/kappa; hier[global_index]=spd(scale*degrees/(degrees-2))
    return spatial,mean,hier

def main():
    p=argparse.ArgumentParser(); p.add_argument('--residual-root',type=Path,required=True)
    p.add_argument('--dataset-contract',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); stage=out.with_name(out.name+'.incomplete')
    if out.exists() or stage.exists(): raise FileExistsError(out)
    stage.mkdir(parents=True)
    rp=a.residual_root/'train12_in_sample_residuals.npz'
    with np.load(rp,allow_pickle=False) as r:
        residual=np.asarray(r['residual_ray_m'],float); drive=r['drive']; camera=r['camera']; frames=r['source_frame_id']; stamps=r['stamp_ns']
    with np.load(a.dataset_contract,allow_pickle=False) as d:
        lookup={(str(x),str(c),str(f),int(t)):i for i,(x,c,f,t) in enumerate(zip(d['drive'],d['camera'],d['source_frame_id'],d['stamp_ns']))}
        index=np.asarray([lookup[(str(x),str(c),str(f),int(t))] for x,c,f,t in zip(drive,camera,frames,stamps)])
        raw=np.asarray(d['raw'][index],float)
    global_cov=scatter(residual); global_pred=np.repeat(global_cov[None],len(residual),axis=0)
    camera_pred=np.empty_like(global_pred)
    for cam in sorted(set(camera.tolist())): camera_pred[camera==cam]=scatter(residual[camera==cam])
    spatial,hier_mean,hier_cov=local_predictions(raw,residual,drive,camera)
    methods={"R0_global_full":scores(residual,global_pred,drive),
             "R1_per_camera_full":scores(residual,camera_pred,drive),
             "R2_spatial_residual":scores(residual,spatial,drive),
             "R3_hierarchical_predictive":scores(residual-hier_mean,hier_cov,drive)}
    result={"schema":"provisional_train12_matched_R_comparison.v1","status":"complete_in_sample_diagnostic",
        "created_utc":datetime.now(timezone.utc).isoformat(),"model_id":"box_mlp_visibility_residual",
        "population":{"rows":len(residual),"complete_drives":len(set(drive.tolist())),"audit_rows":0},
        "evaluation":"same 12 drives used to fit correction and R; local query excludes itself",
        "warning":"In-sample diagnostic only. Coverage and NLL are optimistic and cannot select final R.",
        "local_contract":{"neighbors":120,"bandwidth_m":.75,"drive_weight_cap":"normalize each drive before spatial scatter"},
        "methods":methods,"inputs":{"residual_npz":str(rp.resolve()),"residual_sha256":sha(rp),
        "dataset_contract":str(a.dataset_contract.resolve()),"dataset_contract_sha256":sha(a.dataset_contract)},
        "implementation_sha256":sha(Path(__file__))}
    path=stage/'comparison.json'; path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    os.replace(stage,out)
    print(json.dumps({k:{x:v[x] for x in ('equal_drive_mean_nll','equal_drive_coverage','equal_drive_median_95_ellipse_area_cm2','mean_mahalanobis2')} for k,v in methods.items()},indent=2))

if __name__=='__main__': main()
