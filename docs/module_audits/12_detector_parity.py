#!/usr/bin/env python3
"""Frozen native YOLO inference only: saved pixels, file/array inputs, and chunking."""
from __future__ import annotations
import ast,csv,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/perception'))
import cv2,numpy as np,torch,ultralytics
from ultralytics import YOLO
from perception.core.four_camera_batch import CAMERA_ORDER,BatchContractError,validate_batch_results
from perception.core.yolo_selection import select_best_detection,target_class_ids
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
CAP=ROOT/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
WEIGHTS=ROOT/'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt'
SOURCE=ROOT/'src/perception/perception/nodes/batched_four_camera_yolo_node.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(WEIGHTS);source_bytes=SOURCE.read_bytes();source_before=hashlib.sha256(source_bytes).hexdigest()
with (CAP/'bbox_observations.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['pose_id']=='0']
rows.sort(key=lambda r:r['camera_id']);assert len(rows)==5
paths=[CAP/r['image'] for r in rows];images=[cv2.imread(str(p)) for p in paths]
model=YOLO(str(WEIGHTS))
offline=model.predict(source=[str(p) for p in paths],imgsz=960,conf=.001,iou=.45,batch=5,stream=False,verbose=False,device='cpu')
tree=ast.parse(source_bytes)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_predict_batch' for m in n.body))
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_batch')
env=dict(np=np,Any=object,time=time,CAMERA_ORDER=CAMERA_ORDER,BatchContractError=BatchContractError,validate_batch_results=validate_batch_results)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(SOURCE),'exec'),env)
node=SimpleNamespace(model=model,inference_chunk=2,image_size=960,predict_conf_floor=.05,iou_threshold=.45,device='cpu')
runtime=env['_predict_batch'](node,images)
ids=target_class_ids(model.names,'robot',-1)
def selected(r):return select_best_detection(r,target_ids=ids,confidence_threshold=.25,use_masks=False,mask_min_area=0,mask_bottom_band_px=3)
result=[]
for row,a,b in zip(rows,offline,runtime,strict=True):
    oa,rb=selected(a),selected(b)
    saved=bool(int(row['detected']))
    rec=dict(camera=row['camera_id'],pose_id=row['pose_id'],image=row['image'],decoded_sha1=row['image_sha1'],saved_detected=saved,
        offline_detected=bool(oa['detected']),runtime_detected=bool(rb['detected']),
        offline_candidate_count=oa['n_candidates'],runtime_candidate_count=rb['n_candidates'])
    assert saved==rec['offline_detected']==rec['runtime_detected']
    if saved:
        historical=np.array([float(row[k]) for k in ('x0','y0','x1','y1')])
        rec.update(offline_runtime_max_box_delta_px=float(np.max(np.abs(np.array(oa['bbox_xyxy'])-rb['bbox_xyxy']))),
            saved_offline_max_box_delta_px=float(np.max(np.abs(historical-oa['bbox_xyxy']))),
            offline_runtime_score_delta=abs(float(oa['confidence'])-float(rb['confidence'])),
            saved_offline_score_delta=abs(float(row['confidence'])-float(oa['confidence'])))
        assert rec['offline_runtime_max_box_delta_px']<.01
    result.append(rec)
payload=dict(scope='one preselected complete static pose, five saved views; native CPU, no fit; not a detector accuracy evaluation',
    detector_sha256=before,source_sha256=source_before,source_path=str(SOURCE),runtime_method_source=ast.unparse(method),torch_version=torch.__version__,ultralytics_version=ultralytics.__version__,
    offline=dict(source='saved lossless files',batch_size=5,predict_confidence_floor=.001),
    runtime=dict(source='identical decoded BGR arrays',inference_chunk=2,predict_confidence_floor=.05),
    common=dict(image_size=960,score_threshold=.25,iou=.45,device='cpu',masks=False),
    examples=result,unchanged_weights=before==sha(WEIGHTS),unchanged_source=source_before==sha(SOURCE))
dest=Path(__file__).resolve().parent/'12_detector_parity_results.json';dest.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
