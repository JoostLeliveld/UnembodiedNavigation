#!/usr/bin/env python3
"""Apply frozen study models to the existing second static configuration capture.

Same camera installation; previously examined dense-line development configurations.
No fitting, no transfer-to-new-camera claim, and no temporal-noise claim.
"""
import json
import numpy as np
import joblib
from study import REPO,OUT,ARTIFACT,readcsv,writejson,digest,score,KINDS
from model import ray_basis
from reliability.learned_box_correction import LearnedBoxCorrection

NAME='logs/perception_datasets/warehouse_v2_generalization_20260902'

def main():
    p=REPO/NAME;manifest_path=OUT/'generalization_manifest.json'
    files=['capture_manifest.json','capture_index.csv','observation_interpretations.csv','observation_interpretations_manifest.json']
    if not manifest_path.exists():
        writejson(manifest_path,dict(role='previously_examined_same_installation_development',capture=NAME,
            files={f:digest(p/f) for f in files},models_sha256=digest(OUT/'models.joblib')))
    frozen=json.loads(manifest_path.read_text())
    for f,h in frozen['files'].items():
        if digest(p/f)!=h:raise ValueError('input changed')
    if digest(OUT/'models.joblib')!=frozen['models_sha256']:raise ValueError('model changed')
    rows=readcsv(p/'observation_interpretations.csv');mean=LearnedBoxCorrection(REPO/ARTIFACT);data=[]
    for r in rows:
        if r['raw_valid']!='1':continue
        c=r['camera_id'];raw=np.array([float(r['raw_x']),float(r['raw_y'])]);box=[float(r[k]) for k in ['x0','y0','x1','y1']]
        z=np.array(mean.correct(c,raw,box,float(r['confidence'])))
        # Adjacent line samples are grouped by capture-condition label, not treated as replicas.
        data.append(dict(camera=c,z=z,truth=np.array([float(r['robot_x']),float(r['robot_y'])]),
            raw=raw,basis=ray_basis(raw,np.array(mean._geometry[c]['xy'])),
            distance=float(np.linalg.norm(raw-mean._geometry[c]['xy'])),confidence=float(r['confidence']),
            group=r['dataset_split']))
    models=joblib.load(OUT/'models.joblib');results={}
    for kind in KINDS:
        e=[];R=[]
        for r in data:
            z,C=models[r['camera'],kind].predict([r]);e.append(z[0]-r['truth']);R.append(C[0])
        results[kind]=score(np.array(e),np.array(R),[r['group'] for r in data])
    writejson(OUT/'generalization_results.json',dict(status='same_installation_development_configurations',
        attempts=len(rows),readings=len(data),groups=len({r['group'] for r in data}),scores=results,
        limitation='Dense nearby static poses, previously inspected; not a new installation or independent driving trial.'))
    print({k:(v['rmse_cm'],v['coverage']['0.95']) for k,v in results.items()})
if __name__=='__main__':main()
