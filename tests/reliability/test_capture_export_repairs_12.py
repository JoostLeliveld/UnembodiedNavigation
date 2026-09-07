"""Frozen loader equality and operational residual invariants; no model fitting."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import joblib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/icra_commissioning'))
from reliability.learned_box_correction import LearnedBoxCorrection
from reliability.reference_calibration import ReferenceCalibration
from reliability.operational_residual import build_operational_residuals, summarize_residuals
from reliability.observation_gp import ObservabilityGP
from reliability.observation_exporter import export_observation_dataset, ExporterConfig
from reliability.observation_gates import UsableObservationGateConfig

MEAN=ROOT/'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'
CAL=ROOT/'logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json'


@pytest.mark.parametrize('mutation', ['target','missing_camera','extra_camera','nan_geometry','duplicate_camera'])
def test_nn_semantic_metadata_required(tmp_path,mutation):
    p=joblib.load(MEAN)
    if mutation=='target':p.pop('target')
    if mutation=='missing_camera':p['camera_geometry'].pop('camera_B')
    if mutation=='extra_camera':p['camera_geometry']['camera_Z']=p['camera_geometry']['camera_B']
    if mutation=='nan_geometry':p['camera_geometry']['camera_B']['yaw']=np.nan
    if mutation=='duplicate_camera':p['camera_ids'].append('camera_B')
    dest=tmp_path/'model.joblib';joblib.dump(p,dest)
    with pytest.raises(ValueError):LearnedBoxCorrection(dest)


def test_nn_loaded_identity_comes_from_same_deserialized_bytes(tmp_path,monkeypatch):
    data=MEAN.read_bytes();p=tmp_path/'model.joblib';p.write_bytes(data)
    read=Path.read_bytes
    def swapped(path):
        result=read(path)
        if path==p:p.write_bytes(b'changed after read')
        return result
    monkeypatch.setattr(Path,'read_bytes',swapped)
    loaded=LearnedBoxCorrection(p)
    assert loaded.sha256==hashlib.sha256(data).hexdigest()
    calibration=ReferenceCalibration(CAL,loaded,loaded.camera_ids)
    z=loaded.correct('camera_B',[0,-5],[600,400,660,440],.93)
    assert z is not None and np.isfinite(calibration.apply('camera_B',z)[0]).all()


def test_reference_hash_and_values_use_one_read(tmp_path,monkeypatch):
    encoded=CAL.read_bytes();p=tmp_path/'calibration.json';p.write_bytes(encoded)
    changed=json.loads(encoded);changed['cameras']['camera_B']['R_m2']=[[2,0],[0,2]]
    read=Path.read_bytes
    def swapped(path):
        result=read(path)
        if path==p:p.write_text(json.dumps(changed))
        return result
    monkeypatch.setattr(Path,'read_bytes',swapped)
    loaded=ReferenceCalibration(p,MEAN,['camera_B'])
    assert loaded.sha256==hashlib.sha256(encoded).hexdigest()
    np.testing.assert_array_equal(loaded.covariance['camera_B'],json.loads(encoded)['cameras']['camera_B']['R_m2'])


@pytest.mark.parametrize('mutation',['missing','forged','expected','duplicate_request'])
def test_reference_provenance_and_requested_cameras(tmp_path,mutation):
    p=tmp_path/'calibration.json';data=json.loads(CAL.read_bytes());kwargs={};cameras=['camera_B']
    if mutation=='missing':data.pop('source_hashes')
    if mutation=='forged':data['source_hashes']={'nonexistent.joblib':'0'*64}
    if mutation=='expected':kwargs['expected_source_hashes']={'wrong.py':'0'*64}
    if mutation=='duplicate_request':cameras*=2
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):ReferenceCalibration(p,MEAN,cameras,**kwargs)


def measurement(i,z):return SimpleNamespace(index=i,z=z,source='camera_A',image_id=f'physical/{i}')


def test_pixel_residual_uses_function_value_and_jacobian_separately():
    mu=np.array([[1.,2.],[2.,3.]])
    H=np.diag([2.,3.]);offset=np.array([640.,360.])
    kwargs=dict(smoothed_mean=mu,smoothed_cov=np.tile(np.eye(2),(2,1,1)),
        measurements=[measurement(0,H@mu[0]+offset)],camera_id='camera_A',frame='uv')
    with pytest.raises(ValueError,match='function'):build_operational_residuals(**kwargs,observation_jacobian=H)
    r=build_operational_residuals(**kwargs,observation_jacobian=lambda state:H,observation_function=lambda state:H@state+offset)[0]
    np.testing.assert_allclose(r.residual,[0,0]);np.testing.assert_allclose(r.state_projection,H@H.T)
    assert not r.held_out  # unspecified source provenance must not claim independence


def test_repeated_physical_residuals_cannot_change_sample_count():
    mu=np.zeros((2,2));P=np.zeros((2,2,2));m=[measurement(0,[1,0]),measurement(1,[-1,0])]
    kwargs=dict(smoothed_mean=mu,smoothed_cov=P,camera_id='camera_A',anchored_by=['camera_B'])
    with pytest.raises(ValueError,match='duplicate'):build_operational_residuals(**kwargs,measurements=m*2)
    records=build_operational_residuals(**kwargs,measurements=m)
    assert summarize_residuals(records).sample_count==2
    with pytest.raises(ValueError,match='duplicate'):summarize_residuals(records*10)


def test_constant_gp_does_not_bypass_finite_input_guard():
    model=ObservabilityGP().fit(np.array([[0.,0.]]),np.array([1.]))
    with pytest.raises(ValueError):model.predict_proba([[np.nan,0]])
    assert np.isfinite(model.predict_proba([[0.,0.]])).all()


def test_unidentifiable_legacy_export_is_refused_before_writes(tmp_path):
    with pytest.raises(RuntimeError,match='physical image/opportunity'):
        export_observation_dataset(str(tmp_path),UsableObservationGateConfig(image_width_px=1280,image_height_px=720),ExporterConfig(),str(tmp_path/'out'))
    assert not (tmp_path/'out').exists()
