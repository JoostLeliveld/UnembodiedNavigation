#!/usr/bin/env python3
"""Export three opt-in IWAI network fields from frozen development commissioning.

The score GP is fit to heading-averaged detector scores (miss=0), separately
from the existing probability GP. R is the same frozen full constant covariance
in all three arms. No new performance claim or final-test selection is made here.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/icra_mpl')
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/p) for p in ('src/planning', 'src/reliability', 'src/unav_common',
    'src/experiments', 'experiments/camera_observation_characterization')]
from study import OUT, readcsv, tile, digest, writejson
from commissioned_field import CAMERAS
from planning.core.camera_network import CameraNetworkModel
from reliability.observation_gp import _fbag

DEST = OUT/'network_planner'
KINDS = ('uniform', 'geometry', 'gp')


def grouped_scores(rows, roles):
    """One equally weighted XY target per camera; preserve detection misses.

    Pose repetitions/headings are an empirical nuisance distribution, not
    independent draws of Gaussian sensor noise. No evaluation tile is fitted.
    """
    groups = defaultdict(list)
    hash_roles = defaultdict(set)
    counts = Counter()
    for r in rows:
        role = 'mean_train' if r['split']=='train' else roles[tile(r)]
        if r['raw_valid']=='1': hash_roles[r['image_sha1']].add(role)
        counts[role] += 1
        if role not in ('mean_train', 'covariance_fit'): continue
        score = float(r['confidence']) if r['detected']=='1' else 0.
        if not np.isfinite(score) or not 0 <= score <= 1: raise ValueError('invalid detector score')
        groups[(r['camera_id'], r['position_id'])].append((r, score))
    if any(len(v)>1 for v in hash_roles.values()):
        raise ValueError('duplicate accepted image across source roles')
    out = {}
    for camera in CAMERAS:
        ids = sorted(k[1] for k in groups if k[0]==camera)
        X=[]; y=[]; opportunity_ids=[]
        for ident in ids:
            samples=groups[(camera,ident)]
            poses=np.array([[float(r['robot_x']),float(r['robot_y'])] for r,_ in samples])
            np.testing.assert_allclose(poses, np.tile(poses[0],(len(poses),1)))
            X.append(poses[0]); y.append(np.mean([s for _,s in samples]))
            opportunity_ids.extend(f"{r['pose_id']}:{r['repetition_id']}:{camera}" for r,_ in samples)
        if len(ids)<3: raise ValueError('insufficient independent commissioned positions')
        out[camera] = dict(X=np.asarray(X), y=np.asarray(y), position_ids=ids,
                          opportunity_ids=opportunity_ids)
    return out, dict(counts)


def export(out, step=.5, length=1., noise=.05, miss_extra_std=5.):
    if (out/'manifest.json').exists():
        raise RuntimeError('output already frozen; use a new --out directory')
    if min(step,length,noise,miss_extra_std)<=0: raise ValueError('positive export parameters required')
    start=time.perf_counter()
    source_manifest=OUT/'manifest.json'; frozen=json.loads(source_manifest.read_text())
    for p,h in frozen['files'].items():
        if digest(REPO/p)!=h: raise RuntimeError(f'changed frozen input: {p}')
    field_path=OUT/'field_study/field.joblib'
    selection=OUT/'field_study/selection.json'
    if digest(field_path)!=json.loads(selection.read_text())['artifact_sha256']:
        raise RuntimeError('availability artifact differs from its frozen selection')
    field_manifest=json.loads((OUT/'field_study/manifest.json').read_text())
    for p,h in field_manifest['files'].items():
        if digest(REPO/p)!=h: raise RuntimeError(f'changed availability input: {p}')
    field=joblib.load(field_path)
    models=joblib.load(OUT/'models.joblib')
    rows=readcsv(REPO/frozen['capture']/'bias_update_interpretations.csv')
    groups,counts=grouped_scores(rows,frozen['roles'])
    xy=np.vstack([g['X'] for g in groups.values()])
    # Map extent is installation/commissioning support, never an online truth input.
    xs=np.arange(np.floor(xy[:,0].min()/step)*step,np.ceil(xy[:,0].max()/step)*step+step/2,step)
    ys=np.arange(np.floor(xy[:,1].min()/step)*step,np.ceil(xy[:,1].max()/step)*step+step/2,step)
    X,Y=np.meshgrid(xs,ys); query=np.column_stack([X.ravel(),Y.ravel(),np.zeros(X.size)])
    support=cKDTree(xy).query(query[:,:2])[0]<=2.
    score_maps={kind:[] for kind in KINDS}
    geom=field.availability['geometry_xy']
    fit_details={}
    for camera in CAMERAS:
        group=groups[camera]; targets=np.clip(group['y'],1e-4,1-1e-4)
        poses=np.column_stack([group['X'],np.zeros(len(targets))])
        regression=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(
            geom._geometry_features(poses,camera),logit(targets))
        gp=_fbag()._fit_latent_gp_model(group['X'],logit(targets),
            np.full(len(targets),noise),length_scale=length)
        predictions=dict(uniform=np.full(X.size,group['y'].mean()),
            geometry=expit(regression.predict(geom._geometry_features(query,camera))),
            gp=expit(gp.predict(query[:,:2])))
        for kind,values in predictions.items():
            score_maps[kind].append(np.where(support,values,0.).reshape(X.shape))
        fit_details[camera]={k:v for k,v in group.items() if k not in ('X','y')}
        fit_details[camera].update(n_positions=len(targets),n_opportunities=len(group['opportunity_ids']))
    R=[];bias=[]
    for camera in CAMERAS:
        _,cov=models[camera,'constant'].predict([dict(z=np.zeros(2))])
        R.append(cov[0]);bias.append(models[camera,'constant'].bias.tolist())
    R=np.asarray(R)
    np.testing.assert_allclose(R,field.constant_covariances)
    source_paths=[source_manifest,field_path,selection,OUT/'field_study/manifest.json',
        OUT/'models.joblib',Path(__file__).resolve(),REPO/'src/reliability/reliability/observation_gp.py',
        REPO/'scripts/visibility_comparison/fit_belief_aware_gp.py']
    sources={str(p.relative_to(REPO)):digest(p) for p in source_paths}
    common=dict(schema='camera_network.iwai.v1',reference='robot_ground_reference_xy',frame='map_bev',
        covariance_units='m2',score_target='detector_score_with_miss_zero',
        availability_target='valid_detection_finite_ground_projection',
        evidence='previously inspected grouped static development; planner setup only',
        conditioning='predicted XY; heading marginalized over commissioned headings',
        mean_definition='existing bbox-feature NN then subtract frozen per-camera residual bias',
        required_runtime_mean_offset_m=dict(zip(CAMERAS,bias)),
        runtime_equivalence='not established: live mean offset/R and robust fusion differ',
        score_fit='mean_train + covariance_fit only; equal weight per commissioned position',
        score_hyperparameters=dict(gp_length_m=length,gp_logit_noise_variance=noise,ridge_alpha=1.,
            status='fixed implementation-pilot settings, not selected on final outcomes'),
        availability_fit='reused frozen, separately targeted availability model; selection tiles used',
        support=dict(grid_step_m=step,nearest_training_xy_max_m=2.,outside_value=0.),
        miss_proxy_extra_std_m=miss_extra_std,
        miss_proxy_definition='R_miss_proxy = R_cond + std^2 I; designed IWAI endpoint, not measured miss noise',
        independence='matrix information addition is provisional for the current robust runtime fusion',
        Q=frozen['Q'],source_hashes=sources)
    out.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for kind,qkind in zip(KINDS,('constant','geometry_xy','gp_xy')):
        availability=field.availability[qkind].predict(query).T.reshape((5,*X.shape))
        availability[:,~support.reshape(X.shape)]=0.
        meta={**common,'score_model':kind,'availability_model':qkind}
        path=out/f'{kind}.npz'
        np.savez_compressed(path,xs=xs,ys=ys,camera_ids=np.asarray(CAMERAS),
            score=np.asarray(score_maps[kind]),availability=availability,R_cond_m2=R,
            R_miss_proxy_m2=R+miss_extra_std**2*np.eye(2),metadata_json=json.dumps(meta,sort_keys=True))
        CameraNetworkModel(path)  # validate exactly the runtime-readable representation
        artifacts[kind]=dict(path=str(path.relative_to(REPO)),sha256=digest(path))
    elapsed=time.perf_counter()-start
    manifest=dict(schema='network_planner_export.v1',sources=sources,artifacts=artifacts,
        source_role_counts=counts,score_fit=fit_details,grid_shape=list(X.shape),
        elapsed_export_seconds=elapsed,common_metadata=common,
        cost_scope='export time includes score fitting and grid prediction; excludes prior NN, R, and availability fitting')
    writejson(out/'manifest.json',manifest)
    print(json.dumps(dict(out=str(out),seconds=elapsed,positions_per_camera={c:len(g['X']) for c,g in groups.items()})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEST)
    parser.add_argument('--grid-step-m',type=float,default=.5)
    parser.add_argument('--gp-length-m',type=float,default=1.)
    parser.add_argument('--gp-noise',type=float,default=.05)
    parser.add_argument('--miss-extra-std-m',type=float,default=5.)
    args=parser.parse_args()
    export(args.out.resolve(),args.grid_step_m,args.gp_length_m,args.gp_noise,args.miss_extra_std_m)
