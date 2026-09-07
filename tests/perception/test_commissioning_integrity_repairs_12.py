"""Covariance/dataset provenance guards, with no model training."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('scripts/perception','experiments/camera_observation_characterization','src/unav_common')]
import build_residual_bias_dataset as B
import residual_bias_model as R
import train_residual_bias as T
import dataset_split_utils as S
import fit_bias_updates as F
import learn_measurement_covariance as V
from unav_common.capture_integrity import atomic_csv, digest, pixel_hash


def test_sparse_covariance_cannot_claim_out_of_fold_residuals(monkeypatch):
    monkeypatch.setattr(V,'fit_mean_model',lambda *a:pytest.fail('unsupported split attempted a fit'))
    data=dict(split=np.array(['train']*4),e=np.ones((4,2)),rows=[{'position_id':str(i)} for i in range(4)])
    with pytest.raises(ValueError,match='insufficient'):
        V.out_of_fold_residuals(data,'nn',0,2)


def test_successful_image_cannot_leak_between_adjacent_tiles():
    rows=[dict(position_id=str(i),robot_x=str(x),robot_y='.1',image_sha1='identical')
          for i,x in enumerate((1.999,2.001))]
    split=F.tile_split(rows,2)
    for row in rows:row['split']=split[row['position_id']]
    with pytest.raises(ValueError,match='both mean train and test'):F.validate_split_images(rows)


def test_split_endpoints_and_nonfinite_fraction():
    rows=S.build_pose_records([0,1],[0],[0,1])
    for mode in ('cyclic','spatial_cell'):
        assert S.assign_splits(rows,val_fraction=0,split_mode=mode)==['train']*4
        assert S.assign_splits(rows,val_fraction=1,split_mode=mode)==['val']*4
        with pytest.raises(ValueError):S.assign_splits(rows,val_fraction=np.nan,split_mode=mode)


def feature_row():
    return dict(image_width=1280,image_height=720,camera='camera_B',box_bottom_u=500,
        box_bottom_v=400,box_x1=450,box_x2=550,box_y1=300,box_y2=400,
        box_confidence=.9,baseline_range_m=5,baseline_bearing_rad=.1)


def test_online_features_do_not_read_commanded_heading():
    row=feature_row();row['robot_yaw']='forbidden command reference'
    assert np.isfinite(R.online_feature_vector(row,heading_rad=.2)).all()
    assert np.isfinite(R.feature_vector(row,disable_heading=True)).all()
    with pytest.raises(ValueError,match='operational heading'):R.online_feature_vector(row)
    row['box_confidence']=np.nan
    with pytest.raises(ValueError):R.online_feature_vector(row,heading_rad=.2)


def centre_fixture(tmp_path):
    source=tmp_path/'source';camera=source/'camera_A';camera.mkdir(parents=True)
    rows=[];diagnostics=[]
    for i in range(2):
        pixels=np.full((10,10,3),i,np.uint8);path=camera/f'{i}.png';cv2.imwrite(str(path),pixels)
        rows.append(dict(positive='1',split='train',robot_x=str(i),robot_y='0',camera='camera_A',
            sample_index=str(i),image=str(path),image_width='10',image_height='10'))
        diagnostics.append(dict(accepted='1',sample_kind='positive',sample_index=str(i),image_sha1=pixel_hash(pixels)))
    diagnostic=camera/'label_diagnostics.csv';atomic_csv(diagnostic,diagnostics,diagnostics[0].keys())
    original=camera/'dataset_manifest.json';original.write_text('{}')
    centre=tmp_path/'centre';centre.mkdir()
    records=centre/'records.csv';atomic_csv(records,rows,rows[0].keys())
    manifest=centre/'dataset_manifest.json'
    manifest.write_text(json.dumps(dict(source_root=str(source),records_sha256=digest(records.read_bytes()),
        sources={'camera_A':dict(manifest=str(original),manifest_sha256=digest(original.read_bytes()),
                                diagnostics=str(diagnostic),diagnostics_sha256=digest(diagnostic.read_bytes()))})))
    (centre/'.complete').write_text(json.dumps(dict(manifest_sha256=digest(manifest.read_bytes()))))
    weights=tmp_path/'weights.pt';weights.write_bytes(b'frozen fake weights')
    return centre,weights


def test_short_backend_never_seals_a_partial_dataset(tmp_path,monkeypatch):
    import ultralytics
    centre,weights=centre_fixture(tmp_path)
    class Backend:
        def __init__(self,*a):pass
        def predict(self,*a,**kw):return [SimpleNamespace(boxes=None)]
    monkeypatch.setattr(ultralytics,'YOLO',Backend)
    with pytest.raises(RuntimeError,match='cardinality'):
        B.build(centre,weights,tmp_path/'output',imgsz=960,confidence=.25,batch=2,device='cpu')
    assert not (tmp_path/'output').exists()
    assert not (tmp_path/'output.incomplete/.complete').exists()


def test_changed_source_diagnostics_refused_before_detector(tmp_path,monkeypatch):
    centre,weights=centre_fixture(tmp_path)
    (tmp_path/'source/camera_A/label_diagnostics.csv').write_text('changed')
    with pytest.raises(ValueError,match='changed source'):
        B.build(centre,weights,tmp_path/'output',imgsz=960,confidence=.25,batch=2,device='cpu')


def test_trainer_checks_completion_hash_before_output_or_fit(tmp_path):
    dataset=tmp_path/'data';dataset.mkdir()
    (dataset/'.complete').write_text(json.dumps(dict(manifest_sha256='0'*64)))
    (dataset/'dataset_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='completion digest'):
        T.train(dataset,tmp_path/'trained',disable_heading=True,epochs=1,patience=1,batch=1,learning_rate=.001,seed=0,device='cpu')
    assert not (tmp_path/'trained').exists()
