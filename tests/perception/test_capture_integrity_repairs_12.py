"""Desired capture/export invariants; temporary fixtures and no ROS runtime."""
import copy
import csv
import importlib.util
import io
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('experiments/camera_observation_characterization', 'src/unav_common')]
import capture_bbox_grid as C
import audit_capture_resume as A
import run_bbox_detector as D
import derive_interpretations as I
import reencode_capture_lossless as E
from unav_common.capture_integrity import (
    CaptureIndexWriter, atomic_csv, capture_lock, checked_image, digest, historical_bytes, pixel_hash,
)

CAMERAS = tuple('camera_' + c for c in 'ABCDE')

def read_rows(path):
    return list(csv.DictReader(path.open()))


@pytest.fixture
def stopped_writer(tmp_path, monkeypatch):
    world = tmp_path / 'world.sdf'; world.write_text('<sdf/>')
    profiles = tmp_path / 'profiles.yaml'; profiles.write_text('camera_intrinsics: {}')
    out = tmp_path / 'capture'
    specs = tuple(C.CameraSpec(c, c, '/'+c, '/label/'+c, None, (0,0,5,0,.5,0), 2, 2) for c in CAMERAS)
    poses = [dict(x=i,y=0,yaw=0,position_id=i,yaw_idx=0,x_idx=i,y_idx=0) for i in range(3)]
    monkeypatch.setattr(C, '_capture_transport_environment', lambda **kw: {'partition':'fixture'})
    monkeypatch.setattr(C, '_camera_specs', lambda *a: (specs, {'world_name':'fixture'}, str(world)))
    monkeypatch.setattr(C, '_pose_plan', lambda *a: (poses, {}))
    monkeypatch.setattr(C, '_expected_geometry', lambda *a: {})
    monkeypatch.setattr(C.rclpy, 'init', lambda: None)
    monkeypatch.setattr(C.rclpy, 'ok', lambda: False)
    class Node:
        def __init__(self, **kw): self.n=0
        def wait_ready(self, ids): pass
        def destroy_node(self): pass
        def capture(self, *args):
            self.n += 1
            if self.n == 2: raise KeyboardInterrupt('controlled interruption')
            self.capture_context = dict(command_issue_ns=1_000_000_000,
                command_ack_ns=1_100_000_000, settle_barrier_ns=1_900_000_000)
            return {c:C.Pair(np.full((2,2,3), j, np.uint8),None,2_000_000_000,None,np.nan)
                    for j,c in enumerate(CAMERAS)}
    monkeypatch.setattr(C, 'FiveCameraCapture', Node)
    args = ['capture', '--out', str(out), '--world-profiles', str(profiles)]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(KeyboardInterrupt): C.main()
    assert len(read_rows(out/'capture_index.csv')) == 5
    return out,args


@pytest.mark.parametrize('defect', ['duplicate','missing','gap','timing','pixels','source','settings','mixed_pose','partial_status','identity'])
def test_real_writer_refuses_invalid_resume_before_ros(stopped_writer, monkeypatch, defect):
    out,args=stopped_writer
    index=out/'capture_index.csv'; rows=read_rows(index)
    manifest=out/'capture_manifest.json'; meta=json.loads(manifest.read_text())
    if defect=='duplicate': rows[-1]['camera_id']=rows[0]['camera_id']
    if defect=='missing': rows.pop()
    if defect=='gap':
        for row in rows: row['pose_id']='1';row['source_batch_id']='pose_000001_r00'
    if defect=='timing': rows[-1]['image_stamp_s']='2.2'
    if defect=='pixels': (out/rows[-1]['image']).write_bytes(b'corrupt')
    if defect=='source': meta['capture_helper_sha256']='0'*64
    if defect=='settings': args += ['--settle-s','1.25']
    if defect=='mixed_pose': rows[-1]['robot_x']='2'
    if defect=='partial_status': rows[-1]['capture_status']='failed'
    if defect=='identity': rows[-1]['image_id']=''
    atomic_csv(index,rows,rows[0].keys());manifest.write_text(json.dumps(meta))
    before=(index.read_bytes(),manifest.read_bytes())
    monkeypatch.setattr(C.rclpy,'init',lambda:pytest.fail('unsafe prefix reached ROS initialization'))
    monkeypatch.setattr(sys,'argv',args+['--resume'])
    with pytest.raises(RuntimeError,match='resume|prefix'): C.main()
    assert before==(index.read_bytes(),manifest.read_bytes())


def test_complete_failed_batch_retained_and_repeats_supported(stopped_writer):
    out,_=stopped_writer;mp=out/'capture_manifest.json';m=json.loads(mp.read_text())
    rows=read_rows(out/'capture_index.csv')
    m['plan'].update(repeats=2,planned_rows=30)
    second=[]
    for row in rows:
        r=dict(row,repetition_id='1',source_batch_id='pose_000000_r01',capture_status='failed',image='',image_sha1='',capture_error='transport outage')
        second.append(r)
    atomic_csv(out/'capture_index.csv',rows+second,C.FIELDS);mp.write_text(json.dumps(m))
    report=A.require_resume(out,m)
    assert report['next_pose_id']==1 and report['next_repetition_id']==0
    assert len(read_rows(out/'capture_index.csv'))==10


def test_writer_commits_only_complete_batches(tmp_path,monkeypatch):
    path=tmp_path/'index.csv';w=CaptureIndexWriter(path,['camera'],2)
    w.writerow({'camera':'A'})
    assert read_rows(path)==[]
    w.writerow({'camera':'B'})
    assert len(read_rows(path))==2
    w.writerow({'camera':'A'})
    with pytest.raises(RuntimeError): w.flush()
    assert len(read_rows(path))==2
    with capture_lock(tmp_path):
        with pytest.raises(RuntimeError,match='locked'):
            with capture_lock(tmp_path): pass


def fake_node(monkeypatch,mode):
    node=object.__new__(C.FiveCameraCapture)
    node.settle_s=.1;node.timeout_s=1;node.min_new_rgb=3;node.min_new_labels=1
    node.with_semantic=False;node.batch_sync_slop_s=.05
    node.rgb={c:deque(maxlen=90) for c in CAMERAS};node.rgb_count={c:0 for c in CAMERAS}
    node.label_count={c:0 for c in CAMERAS};node._last_consumed={c:0 for c in CAMERAS}
    clock=[100_000_000_000];wall=[0.]
    def monotonic():wall[0]+=.01;return wall[0]
    node._sim_time_ns=lambda:clock[0]
    node._set_pose=lambda *args:None
    def spin():
        clock[0]+=50_000_000
        stamp=99_000_000_000 if mode=='delayed' else 100_150_000_000 if mode=='duplicate' else clock[0]
        for c in CAMERAS:
            node.rgb_count[c]+=1
            node.rgb[c].append((node.rgb_count[c],stamp,np.zeros((2,2,3),np.uint8)))
    node._spin=spin
    monkeypatch.setattr(C.rclpy,'ok',lambda:True)
    monkeypatch.setattr(C.time,'monotonic',monotonic)
    return node


@pytest.mark.parametrize('mode',['delayed','duplicate'])
def test_real_capture_rejects_old_and_repeated_stamps(monkeypatch,mode):
    node=fake_node(monkeypatch,mode)
    with pytest.raises(RuntimeError,match='frame timeout'): node.capture(0,0,0,CAMERAS)


def test_real_capture_accepts_distinct_fresh_frames_and_advances(monkeypatch):
    node=fake_node(monkeypatch,'fresh')
    a=node.capture(0,0,0,tuple(reversed(CAMERAS)))
    b=node.capture(1,0,0,CAMERAS)
    assert set(a)==set(b)==set(CAMERAS)
    assert all(b[c].image_stamp_ns>a[c].image_stamp_ns for c in CAMERAS)
    assert all(p.image_stamp_ns>node.capture_context['settle_barrier_ns'] for p in b.values())


def detector_fixture(tmp_path,monkeypatch):
    rows=[]
    for i,(status,value) in enumerate([('ok',255),('ok',0),('failed',None)]):
        image=np.full((10,10,3),value or 0,np.uint8)
        if status=='ok':cv2.imwrite(str(tmp_path/f'{i}.png'),image)
        rows.append(dict(pose_id=str(i),position_id=str(i),repetition_id='0',heading_id='0',camera_id='camera_A',
            capture_status=status,capture_error='outage' if status=='failed' else '',
            image=f'{i}.png' if status=='ok' else '',image_sha1=pixel_hash(image) if status=='ok' else '',
            robot_x='0',robot_y='0',robot_yaw='0'))
    atomic_csv(tmp_path/'capture_index.csv',rows,rows[0].keys())
    meta=dict(status='complete_with_failed_batches',cameras=[dict(camera_id='camera_A',image_width=10,image_height=10)],
              capture_index_sha256=digest((tmp_path/'capture_index.csv').read_bytes()))
    (tmp_path/'capture_manifest.json').write_text(json.dumps(meta))
    weights=tmp_path/'weights.pt';weights.write_bytes(b'frozen fake detector')
    class Model:
        names={0:'robot'}
        def __init__(self,path):assert Path(path).read_bytes()==weights.read_bytes()
        def predict(self,**kw):return [SimpleNamespace(hit=bool(im[0,0,0])) for im in kw['source']]
    monkeypatch.setattr(D,'YOLO',Model)
    monkeypatch.setattr(D,'select_best_detection',lambda r,**kw:dict(detected=r.hit,bbox_xyxy=[2,2,8,8] if r.hit else None,
        n_candidates=int(r.hit),confidence=.9 if r.hit else 0,bbox_bottom_u=5,bbox_bottom_v=8))
    args=SimpleNamespace(weights=weights,image_size=960,confidence_threshold=.25,predict_confidence_floor=.001,
        iou_threshold=.45,batch_size=2,device='cpu',fixed_offset_m=.309)
    return rows,args


def test_detector_rejects_changed_pixels_before_inference(tmp_path,monkeypatch):
    rows,args=detector_fixture(tmp_path,monkeypatch)
    cv2.imwrite(str(tmp_path/'0.png'),np.zeros((10,10,3),np.uint8))
    monkeypatch.setattr(D,'YOLO',lambda *a:pytest.fail('changed images reached detector'))
    with pytest.raises(ValueError,match='hash mismatch'):D.run(args,tmp_path)
    assert not (tmp_path/'bbox_detector_manifest.json').exists()


def test_miss_false_positive_and_acquisition_failure_keep_distinct_labels(tmp_path,monkeypatch):
    rows,args=detector_fixture(tmp_path,monkeypatch);D.run(args,tmp_path)
    output=read_rows(tmp_path/'bbox_observations.csv')
    assert [r['detected'] for r in output]==['1','0','']
    assert [r['inference_status'] for r in output]==['detected','miss','not_attempted_acquisition_failed']
    meta=json.loads((tmp_path/'bbox_detector_manifest.json').read_text())
    assert (meta['opportunity_rows'],meta['inference_rows'],meta['acquisition_failed_rows'])==(3,2,1)
    # A confident box on a known empty synthetic image is a detector return, not a TP label.
    assert output[0]['capture_status']=='ok'
    with pytest.raises(RuntimeError,match='already exist'):D.run(args,tmp_path)
    path=tmp_path/'bbox_observations.csv';path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError,match='changed input'):I.derive(args,tmp_path)


def test_converter_recovers_after_second_encode_failure(tmp_path,monkeypatch):
    rows=[]
    for i in range(2):
        image=np.full((10,10,3),i,np.uint8);cv2.imwrite(str(tmp_path/f'{i}.png'),image)
        rows.append(dict(capture_status='ok',image=f'{i}.png',image_sha1=pixel_hash(image)))
    atomic_csv(tmp_path/'capture_index.csv',rows,rows[0].keys())
    (tmp_path/'capture_manifest.json').write_text(json.dumps(dict(status='running')))
    encode=E.cv2.imencode;calls=[0]
    def fail_second(*args,**kw):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('interrupted codec')
        return encode(*args,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(E.cv2,'imencode',fail_second)
        with pytest.raises(RuntimeError,match='interrupted'):E.convert(tmp_path)
    for row in read_rows(tmp_path/'capture_index.csv'): checked_image(tmp_path,row)
    summary=E.convert(tmp_path)
    assert summary['files_reencoded']==2
    assert all(r['image'].endswith('.webp') for r in read_rows(tmp_path/'capture_index.csv'))
    assert E.convert(tmp_path)==summary


def test_historical_profile_loads_exact_frozen_bytes():
    m=json.loads((ROOT/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/capture_manifest.json').read_text())
    data=historical_bytes(Path(m['world_profiles_path']),m['world_profiles_sha256'],repo=ROOT)
    assert digest(data)==m['world_profiles_sha256']
    assert set(I.camera_models(m))==set(CAMERAS)
    with pytest.raises(ValueError,match='no exact historical'):
        historical_bytes(Path(m['world_profiles_path']),'0'*64,repo=ROOT)
