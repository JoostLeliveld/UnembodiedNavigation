#!/usr/bin/env python3
"""Temporary-only exporter/import and malformed dataset boundary probes; no model fitting."""
from __future__ import annotations
import copy, csv, hashlib, importlib.util, json, os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('src/reliability','src/planning','scripts/perception','experiments/icra_commissioning')]
import numpy as np
from reliability.reference_calibration import ReferenceCalibration
from planning.core.camera_network import CameraNetworkModel
from reliability.observation_gp import ObservabilityGP
import export_reference_calibration as EXPORT
import build_residual_bias_dataset as BUILD

STUDY=ROOT/'logs/studies/icra_commissioning_20260905'
MODEL=ROOT/'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'
CAMERAS=tuple('camera_'+c for c in 'ABCDE')
results={}

def outcome(fn):
    try:
        v=fn()
        return dict(accepted=True,value=v)
    except Exception as e:return dict(accepted=False,error=type(e).__name__+': '+str(e))

def writecsv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

with tempfile.TemporaryDirectory(prefix='module12_boundary_') as name:
    tmp=Path(name)
    reference=json.loads((STUDY/'network_planner/reference_calibration.json').read_text())
    # Exercise the real export function; it reads frozen models/fields and never fits.
    dest=tmp/'roundtrip.json';EXPORT.export(dest)
    exported=json.loads(dest.read_text())
    results['real_reference_export_roundtrip']={'identical_json':exported==reference,
        'identical_bytes':dest.read_bytes()==(STUDY/'network_planner/reference_calibration.json').read_bytes()}
    reference_checks={}
    mutations={
        'version':lambda d:d.update(schema='v999'),
        'units':lambda d:d.pop('covariance_units'),
        'NN_hash':lambda d:d.update(mean_checkpoint_sha256='0'*64),
        'missing_camera':lambda d:d['cameras'].pop('camera_B'),
        'nonfinite_R':lambda d:d['cameras']['camera_B'].update(R_m2=[[float('nan'),0],[0,1]]),
        'missing_provenance':lambda d:d.pop('source_hashes'),
        'forged_provenance':lambda d:d.update(source_hashes={'models.joblib':'0'*64}),
    }
    for key,mutate in mutations.items():
        d=copy.deepcopy(reference);mutate(d);p=tmp/f'reference_{key}.json';p.write_text(json.dumps(d))
        reference_checks[key]=outcome(lambda p=p:ReferenceCalibration(p,MODEL,CAMERAS).apply('camera_B',[0.,0.]))
    reference_checks['unknown_camera']=outcome(lambda:ReferenceCalibration(dest,MODEL,('camera_Z',)))
    reference_checks['nonfinite_input']=outcome(lambda:ReferenceCalibration(dest,MODEL,CAMERAS).apply('camera_B',[float('nan'),0.]))
    results['reference_loader']=reference_checks
    with np.load(STUDY/'network_planner/gp.npz',allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    network_checks={}
    variants={
        'version':lambda m:m.update(schema='v999'),
        'units':lambda m:m.pop('covariance_units'),
        'missing_provenance':lambda m:m.pop('source_hashes'),
        'forged_provenance':lambda m:m.update(source_hashes={'models.joblib':'0'*64}),
    }
    for key,mutate in variants.items():
        a=arrays.copy();m=json.loads(str(a['metadata_json'].item()));mutate(m);a['metadata_json']=json.dumps(m)
        p=tmp/f'network_{key}.npz';np.savez_compressed(p,**a)
        network_checks[key]=outcome(lambda p=p:CameraNetworkModel(p).query([0.,0.,0.]))
    network_checks['unknown_camera']=outcome(lambda:CameraNetworkModel(STUDY/'network_planner/gp.npz',cameras=['camera_Z']))
    net=CameraNetworkModel(STUDY/'network_planner/gp.npz')
    network_checks['nonfinite_input']=outcome(lambda:net.query([float('inf'),0,0]))
    # Reorder both the keyed metadata and matching camera axes: numbers must agree by ID.
    indices=[4,2,0,3,1];a=arrays.copy()
    for key in ('camera_ids','score','availability','R_cond_m2','R_miss_proxy_m2'):a[key]=a[key][indices]
    p=tmp/'permuted.npz';np.savez_compressed(p,**a)
    perm=CameraNetworkModel(p,cameras=CAMERAS)
    queries=[[-8,-8,0],[0,0,1],[3,4,2]]
    network_checks['camera_permutation_max_difference']=max(float(np.max(np.abs(net.query(q)[k]-perm.query(q)[k]))) for q in queries for k in ('score','availability'))
    results['network_loader']=network_checks
    # Degenerate GP fast path does not call a fitting routine; input guards are bypassed.
    degenerate=ObservabilityGP().fit(np.array([[0.,0.]]),np.array([1.]))
    results['degenerate_gp_nonfinite_query']=outcome(lambda:degenerate.predict_proba([[float('nan'),0]]))
    # Run real residual dataset builder with a deliberately short fake backend result.
    centre=tmp/'centre';centre.mkdir();source=tmp/'source';(source/'camera_A').mkdir(parents=True)
    (centre/'dataset_manifest.json').write_text(json.dumps({'source_root':str(source)}))
    rows=[dict(positive='1',split='train',robot_x=str(i),robot_y='0',camera='camera_A',sample_index=str(i),image=f'{i}.png') for i in range(2)]
    rows.append(dict(rows[0],positive='0',sample_index='negative'))
    writecsv(centre/'records.csv',rows)
    writecsv(source/'camera_A/label_diagnostics.csv',[dict(accepted='1',sample_kind='positive',sample_index=str(i)) for i in range(2)])
    weights=tmp/'fake_weights.pt';weights.write_bytes(b'frozen fake inference identity')
    class ShortBackend:
        def __init__(self,*a):pass
        def predict(self,*a,**kw):return [SimpleNamespace(boxes=None)]
    import ultralytics
    with patch.object(ultralytics,'YOLO',ShortBackend):
        payload=BUILD.build(centre,weights,tmp/'residual_dataset',imgsz=960,confidence=.25,batch=2,device='cpu')
    with (tmp/'residual_dataset/records.csv').open() as f:output=list(csv.DictReader(f))
    results['provisional_dataset_short_backend']={'source_positive_rows':2,'output_rows':len(output),
        'status':payload['status'],'completion_marker_written':(tmp/'residual_dataset/.complete').exists()}

def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return str(v)
dest=Path(__file__).resolve().parent/'12_boundary_results.json'
dest.write_text(json.dumps(results,indent=2,default=serial)+'\n')
print(json.dumps(results,indent=2,default=serial))
