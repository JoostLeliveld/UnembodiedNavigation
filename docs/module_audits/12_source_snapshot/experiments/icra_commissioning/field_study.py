#!/usr/bin/env python3
"""Commissioned-field comparison with frozen grouped roles and explicit outcomes.

This evaluates previously inspected static development data. New driving runs are
separately frozen and scored by field_driving.py; no retrospective final-test claim.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/icra_mpl')
import argparse,json,sys,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import joblib
import yaml
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/p) for p in ['src/reliability','src/planning','src/unav_common',
 'src/experiments','experiments/camera_observation_characterization','scripts/shared']]
import metrics as M
from study import OUT,CAPTURE,ARTIFACT,readcsv,writejson,digest,tile,score
from derive_interpretations import camera_models
from reliability.learned_box_correction import LearnedBoxCorrection
from model import ray_basis,update
from commissioned_field import CAMERAS,AvailabilityModel,CommissionedField,pose_embedding

FIELD_OUT=OUT/'field_study'
REQ=REPO/'experiments/icra_commissioning/field_requirements.yaml'


def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('field manifest already frozen')
    old=json.loads((OUT/'manifest.json').read_text())
    writejson(out/'manifest.json',dict(schema='commissioned_field_study.v1',
        evidence='grouped_static_development_with_future_frozen_model_driving_application',
        files={**old['files'],str((OUT/'manifest.json').relative_to(REPO)):digest(OUT/'manifest.json'),
               str((OUT/'models.joblib').relative_to(REPO)):digest(OUT/'models.joblib'),
               str(REQ.relative_to(REPO)):digest(REQ)},
        roles=old['roles'],mean_artifact=ARTIFACT,
        split='mean_train + covariance_fit for availability; selection for hyperparameters; evaluation untouched by field selection',
        quality_source='frozen residual-calibrated models; field stores predicted R at training covariates, not mean-training residual estimates',
        requirements=yaml.safe_load(REQ.read_text()),
        sources={str(p.relative_to(REPO)):digest(p) for p in [Path(__file__).resolve(),
            REPO/'experiments/icra_commissioning/commissioned_field.py',
            REPO/'src/reliability/reliability/observation_gp.py',
            REPO/'scripts/visibility_comparison/fit_belief_aware_gp.py']}))


def load_data(out):
    manifest=json.loads((out/'manifest.json').read_text())
    for p,h in manifest['files'].items():
        if digest(REPO/p)!=h:raise ValueError(f'changed field input {p}')
    rows=readcsv(REPO/CAPTURE/'bias_update_interpretations.csv')
    mean=LearnedBoxCorrection(REPO/ARTIFACT);models=joblib.load(OUT/'models.joblib')
    frames={};byhash=defaultdict(set)
    for r in rows:
        role='fit' if r['split']=='train' else {'covariance_fit':'fit','selection':'selection','evaluation':'evaluation'}[manifest['roles'][tile(r)]]
        frame=f"{r['pose_id']}:{r['repetition_id']}"
        if frame not in frames:
            frames[frame]=dict(id=frame,pose=[float(r[k]) for k in ['robot_x','robot_y','robot_yaw']],
                role=role,group=tile(r),hits=np.zeros(5,bool),R=np.full((5,2,2),np.nan),
                error=np.full((5,2),np.nan),raw_error=np.full((5,2),np.nan),cameras=set())
        f=frames[frame];j=CAMERAS.index(r['camera_id'])
        assert f['role']==role and j not in f['cameras'];f['cameras'].add(j)
        if r['raw_valid']!='1':continue
        byhash[r['image_sha1']].add(role)
        raw=np.array([float(r['raw_x']),float(r['raw_y'])]);box=[float(r[k]) for k in ['x0','y0','x1','y1']]
        z=mean.correct(r['camera_id'],raw,box,float(r['confidence']))
        np.testing.assert_allclose(z,[float(r['nn_x']),float(r['nn_y'])],atol=1e-9,rtol=0)
        sample=dict(z=np.asarray(z),raw=raw,confidence=float(r['confidence']),
                    distance=np.linalg.norm(raw-mean._geometry[r['camera_id']]['xy']),
                    basis=ray_basis(raw,np.asarray(mean._geometry[r['camera_id']]['xy'])))
        corrected,R=models[r['camera_id'],'confidence'].predict([sample])
        f['hits'][j]=True;f['R'][j]=R[0];f['error'][j]=corrected[0]-f['pose'][:2]
        f['raw_error'][j]=raw-f['pose'][:2]
    if any(len(v)>1 for v in byhash.values()):raise ValueError('duplicate hit images across field roles')
    assert all(len(f['cameras'])==5 for f in frames.values())
    records=[frames[k] for k in sorted(frames)]
    for a in ['fit','selection','evaluation']:
        for b in ['fit','selection','evaluation']:
            if a!=b:assert not ({f['group'] for f in records if f['role']==a}&{f['group'] for f in records if f['role']==b})
    data={key:np.asarray([f[key] for f in records]) for key in ['id','pose','role','group','hits','R','error','raw_error']}
    geometry=camera_models(json.loads((REPO/CAPTURE/'capture_manifest.json').read_text()))
    # Full constant covariance is held fixed when testing availability effects.
    constant=[]
    for c in CAMERAS:
        m=models[c,'constant'];C=m.covs[0]
        constant.append(m.scale*((1-m.isotropic_shrink)*C+m.isotropic_shrink*np.eye(2)*np.trace(C)/2))
    return data,geometry,np.asarray(constant)


def grouped_probability_score(hits,pred,groups):
    per=[np.mean([M.brier(hits[groups==g,j],pred[groups==g,j]) for j in range(5)]) for g in sorted(set(groups))]
    return dict(equal_tile_brier=float(np.mean(per)),brier=M.brier(hits,pred),
        logloss=M.logloss(hits,pred),ece=M.ece(hits.ravel(),pred.ravel()),
        cameras={c:dict(brier=M.brier(hits[:,j],pred[:,j]),logloss=M.logloss(hits[:,j],pred[:,j]),
            observed_fraction=float(hits[:,j].mean()),predicted_fraction=float(pred[:,j].mean())) for j,c in enumerate(CAMERAS)},
        tile_scores=dict(zip(sorted(set(groups)),map(float,per))))


def parameter_grid(kind,req):
    if kind=='constant':return [None]
    if kind.startswith('geometry'):return req['geometry_logistic_C']
    if kind.startswith('local'):return req['local_neighbors']
    return [(l,n) for l in req['gp_length_scale_m'] for n in req['gp_noise_variance']]


def fit(out):
    start=time.perf_counter();data,geometry,constant=load_data(out)
    req=yaml.safe_load(REQ.read_text())['predeclared_comparison'];methods=req['availability_methods']
    fitmask=data['role']=='fit';sel=data['role']=='selection';test=data['role']=='evaluation'
    selected={};selection={};scores={};predictions={};candidates=[]
    for kind in methods:
        best=None
        for parameter in parameter_grid(kind,req):
            model=AvailabilityModel(kind,parameter,geometry).fit(data['pose'][fitmask],data['hits'][fitmask])
            p=model.predict(data['pose'][sel]);s=grouped_probability_score(data['hits'][sel],p,data['group'][sel])
            candidates.append(dict(kind=kind,parameter=parameter,score=s['equal_tile_brier']))
            if best is None or s['equal_tile_brier']<best[0]:best=(s['equal_tile_brier'],model,parameter)
        selected[kind]=best[1];selection[kind]=dict(parameter=best[2],equal_tile_brier=best[0])
        print('selected',kind,selection[kind],flush=True)
    # Freeze fitted models before touching field evaluation predictions.
    field=CommissionedField(data['pose'][fitmask],data['hits'][fitmask],data['R'][fitmask],selected,constant)
    joblib.dump(field,out/'field.joblib',compress=3)
    writejson(out/'selection.json',dict(selection=selection,candidates=candidates,
        artifact_sha256=digest(out/'field.joblib'),selection_role='static selection tiles only'))
    for role,mask in [('selection',sel),('evaluation',test)]:
        for kind,model in selected.items():
            p=model.predict(data['pose'][mask]);predictions[role+'_'+kind]=p
            s=grouped_probability_score(data['hits'][mask],p,data['group'][mask]);s['supported_fraction']=float(model.support(data['pose'][mask]).mean())
            scores[role+'/'+kind]=s
    np.savez_compressed(out/'static_predictions.npz',poses=data['pose'][test],hits=data['hits'][test],
        groups=data['group'][test],error=data['error'][test],raw_error=data['raw_error'][test],**predictions)
    paired={};rng=np.random.default_rng(509)
    for kind in methods[1:]:
        a=scores['evaluation/constant']['tile_scores'];b=scores['evaluation/'+kind]['tile_scores']
        delta=np.array([a[k]-b[k] for k in sorted(a)])
        bs=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
        paired[kind]=dict(constant_minus_model=float(delta.mean()),descriptive_tile_bootstrap95=np.quantile(bs,[.025,.975]).tolist())
    # Sensor-model diagnosis: exact descriptive variance decomposition by position,
    # with heading variation inside each position; this is not repeated sensor noise.
    decomposition=[]
    for j,c in enumerate(CAMERAS):
        mask=test&data['hits'][:,j];e=data['error'][mask,j];xy=data['pose'][mask,:2]
        positions,inv=np.unique(xy,axis=0,return_inverse=True);mu=e.mean(axis=0)
        within=np.zeros((2,2));between=np.zeros((2,2))
        for k in range(len(positions)):
            values=e[inv==k];m=values.mean(axis=0);d=values-m
            within+=d.T@d/len(e);between+=len(values)*np.outer(m-mu,m-mu)/len(e)
        total=(e-mu).T@(e-mu)/len(e)
        np.testing.assert_allclose(total,within+between,atol=1e-12)
        decomposition.append(dict(camera=c,readings=len(e),positions=len(positions),
            between_position_fraction=float(np.trace(between)/np.trace(total)),
            global_mean_cm=(100*mu).tolist(),total_covariance_m2=total.tolist(),
            within_position_heading_scatter_m2=within.tolist(),between_position_mean_scatter_m2=between.tolist()))
    writejson(out/'static_results.json',dict(status='grouped_static_development',
        attempted_frames={r:int((data['role']==r).sum()) for r in ['fit','selection','evaluation']},
        heldout_opportunities=int(test.sum()*5),heldout_groups=len(set(data['group'][test])),
        scores=scores,paired_tile_differences=paired,variance_decomposition=decomposition,
        seconds=time.perf_counter()-start,requirements_sha256=digest(REQ),
        limitations=['Previously examined static development configurations, not a fresh final test.',
          'Camera availability here means returned detection with finite floor projection, before manager gates.',
          'Probabilities summarize heading and nearby-configuration variation; deterministic repeats are not independent trials.',
          'Bootstrap tiles are descriptive spatial blocks; this does not establish independent-run significance.']))
    return data,field


def evaluate_quality_and_outcomes(out,data,field):
    mask=data['role']=='evaluation';quality={};testposes=data['pose'][mask]
    for method in ['constant','local_mean_R','local_mixture']:
        errors=[];covariances=[];groups=[];mixture_nll=[]
        for i in np.flatnonzero(mask):
            state=data['pose'][i]
            for j in range(5):
                if not data['hits'][i,j]:continue
                e=data['error'][i,j];errors.append(e);groups.append(data['group'][i])
                if method=='constant':Rs=field.constant_covariances[j][None]
                else:
                    ids,tree=field.hit_trees[j]
                    _,near=tree.query(pose_embedding([state])[0],k=min(24,len(ids)))
                    Rs=field.covariances[ids[np.atleast_1d(near)],j]
                covariances.append(Rs.mean(axis=0))
                if method=='local_mixture':
                    v=np.linalg.solve(Rs,np.tile(e,(len(Rs),1))[...,None])[...,0]
                    ll=-.5*(np.linalg.slogdet(Rs)[1]+np.einsum('ni,i->n',v,e)+2*np.log(2*np.pi))
                    mixture_nll.append(float(-logsumexp(ll)+np.log(len(Rs))))
        quality[method]=score(np.asarray(errors),np.asarray(covariances),groups)
        if mixture_nll:quality[method]['mixture_gaussian_nll']=float(np.mean(mixture_nll))
    # Controlled one-opportunity calculation using observed hit/miss patterns.
    # This validates availability's effect on an algebraic covariance update only.
    P=np.diag([.1,.1,np.deg2rad(5)])**2
    rows=[]
    for n,i in enumerate(np.flatnonzero(mask)):
        actual=P.copy()
        for j in range(5):
            if data['hits'][i,j]:_,actual,_=update(np.zeros(3),actual,np.zeros(2),field.constant_covariances[j])
        for method in ['constant','geometry_xy','gp_xy','gp_integrated','local_joint','local_independent']:
            branch,info=field.forecast(data['pose'][i],P,method,'branch',constant_quality=True)
            information,_=field.forecast(data['pose'][i],P,method,'information',constant_quality=True)
            rows.append(dict(frame=str(data['id'][i]),group=str(data['group'][i]),method=method,
                observed_outcome_trace_m2=float(np.trace(actual[:2,:2])),
                branch_trace_m2=float(np.trace(branch[:2,:2])),information_trace_m2=float(np.trace(information[:2,:2])),
                supported=info['supported']))
    writejson(out/'quality_and_outcomes.json',dict(quality=quality,posterior_rows=rows,
        posterior_scope='Same fixed prior and full constant R; actual detected camera set supplies outcome reference. Not realized error calibration.',
        mixture_scope='Current-image confidence is unknown to field query; commissioned quality regimes are marginalized. No future-image input.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--output',type=Path,default=FIELD_OUT)
    args=p.parse_args()
    if args.action=='freeze':freeze(args.output)
    else:
        data,field=fit(args.output);evaluate_quality_and_outcomes(args.output,data,field)


if __name__=='__main__':main()
