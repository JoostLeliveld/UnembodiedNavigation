"""Read-only artifact audit; writes JSON to stdout. Run from repository root."""
from pathlib import Path
import sys, json, csv, math, collections, hashlib, itertools
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for p in ['src/reliability','src/unav_common','src/state','experiments/icra_commissioning','experiments/camera_observation_characterization']:
    sys.path.insert(0,str(ROOT/p))
import joblib
from reliability.projection import camera_model_from_world, project_observation_to_world_with_covariance, _floor_spd_2x2
from reliability.learned_box_correction import LearnedBoxCorrection
from reliability.reference_calibration import ReferenceCalibration
from reliability.contracts import CameraObservation
from reliability.silhouette_observation import observation_jacobian, equivalent_position_measurement, _inverse_2x2
from unav_common.camera_model import ObliqueCameraModel
from state.core.pixel_to_bev import PixelToBevTransformer
import reliability.nodes.camera_manager_node as M
from fit_bias_updates import features, apply_correction
mean=ROOT/'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'
calpath=ROOT/'logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json'
manifest=ROOT/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/capture_manifest.json'
table=manifest.parent/'bias_update_interpretations.csv'
world=ROOT/'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
items=json.loads(manifest.read_text())['cameras']
cameras={i['camera_id']:camera_model_from_world(world,include_name=i['camera_model']) for i in items}
model=LearnedBoxCorrection(mean); payload=joblib.load(mean); cal=ReferenceCalibration(calpath,mean,cameras)
offline_models=joblib.load(ROOT/'logs/studies/icra_commissioning_20260905/models.joblib')
M._with_provider_quality=lambda o,*a:o

def mapped(contract, cams=cameras, admission=False):
    fake=SimpleNamespace(_latest={contract.camera_id:contract},commissioned_pixel_cov_by_camera={},commissioned_pixel_cov=((1.,0.),(0.,1.)),camera_models=cams,covariance_profile=M.COMMISSIONED_REFERENCE_COVARIANCE,_belief_query_history=[],reliability_query_max_time_delta_s=.2,admission_gate=admission,silhouette_correction=False,observation_model=M.OBSERVATION_MODEL_LEARNED_NN,learned_correction=model,reference_calibration=cal,_gate_rejections=collections.Counter(),_silhouette_status_by_camera={},_detection_extras_by_camera={},_reliability_query_source_by_camera={},replay_config=None)
    return M.CameraManagerNode._map_observations(fake,1.)

def contract(row, **changes):
    args=dict(camera_id=row['camera_id'],timestamp_s=1.,detection_valid=True,pixel_uv=(float(row['u_bbox_bottom']),float(row['v_bbox_bottom'])),detector_score=float(row['confidence']),bbox_xyxy=tuple(float(row[k]) for k in ('x0','y0','x1','y1')),selected_pixel_source='bbox_bottom')
    args.update(changes)
    return CameraObservation(**args)

rows=[r for r in csv.DictReader(table.open()) if r['raw_valid']=='1']
errors=dict(raw=0.,features=0.,nn=0.,manager_z=0.,manager_R=0.,ray_oracle=0.,jacobian=0.,covariance=0.)
counts=collections.Counter(); example=None
for r in rows:
    c=r['camera_id']; cam=cameras[c]; obs=contract(r); raw=np.array(cam.pixel_to_world(*obs.pixel_uv))
    errors['raw']=max(errors['raw'],float(np.max(abs(raw-[float(r['raw_x']),float(r['raw_y'])]))))
    geom=payload['camera_geometry'][c]
    f=list(features(r,geom))+[float(c==k) for k in payload['camera_ids']]
    errors['features']=max(errors['features'],float(np.max(abs(np.array(model._features(c,raw,obs.bbox_xyxy,obs.detector_score))-f))))
    offline=apply_correction(r,geom,payload['neural_model'].predict([f])[0])
    nn=np.array(model.correct(c,raw,obs.bbox_xyxy,obs.detector_score))
    errors['nn']=max(errors['nn'],float(np.max(abs(nn-offline))))
    got=mapped(obs)[0]
    z,R=offline_models[c,'constant'].predict([dict(z=offline)])
    errors['manager_z']=max(errors['manager_z'],float(np.max(abs(np.array(got.xy_m)-z[0]))))
    errors['manager_R']=max(errors['manager_R'],float(np.max(abs(np.array(got.covariance_m2)-R[0]))))
    counts[c]+=1
    if example is None:
        example=dict(camera=c,box=obs.bbox_xyxy,pixel=obs.pixel_uv,stamp=1.,stamp_note='synthetic contract stamp; static table has no capture event stamp used here',raw=raw.tolist(),features=f,ray_correction=payload['neural_model'].predict([f])[0].tolist(),nn=nn.tolist(),bias=cal.bias[c].tolist(),z=list(got.xy_m),R=np.array(got.covariance_m2).tolist())
# Independent SDF RPY oracle: camera local +X forward, -Y optical right, -Z optical down.
for item in items:
    cam=cameras[item['camera_id']]; x,y,z,roll,pitch,yaw=item['pose_xyz_rpy']
    cp,sp,cy,sy=math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    rotation=np.array([[cy*cp,-sy,cy*sp],[sy*cp,cy,sy*sp],[-sp,0,cp]])
    for u,v in [(0.,0.),(1280.,720.),(640.,360.),(100.,700.)]:
        f=640/math.tan(1.5708/2); ray=rotation@np.array([1.,-(u-640)/f,-(v-360)/f])
        expected=np.array([x,y,z])-z/ray[2]*ray
        errors['ray_oracle']=max(errors['ray_oracle'],float(np.max(abs(np.array(cam.pixel_to_world(u,v))-expected[:2]))))
        A=cam.H_inv; w=A@np.array([u,v,1.]); J=(A[:2,:2]*w[2]-w[:2,None]*A[2,:2])/w[2]**2
        step=1e-3
        fd=np.column_stack([(np.array(cam.pixel_to_world(u+step*(a==0),v+step*(a==1)))-np.array(cam.pixel_to_world(u-step*(a==0),v-step*(a==1))))/(2*step) for a in (0,1)])
        errors['jacobian']=max(errors['jacobian'],float(np.max(abs(J-fd))))
        R=np.array([[4.,1.2],[1.2,2.]])
        o=CameraObservation(camera_id=item['camera_id'],timestamp_s=1.,detection_valid=True,pixel_uv=(u,v),conditional_cov_uv=R)
        mapped_R=np.array(project_observation_to_world_with_covariance(o,cam,jacobian_step_px=step)[1])
        errors['covariance']=max(errors['covariance'],float(np.max(abs(mapped_R-J@R@J.T))))
cam=cameras['camera_A']; A=cam.H_inv; horizon=-(A[2,0]*640+A[2,2])/A[2,1]
invalid={}
for name,pixel in [('above_horizon',(640.,horizon-10)),('near_horizon',(640.,horizon+1e-6)),('nan',(float('nan'),400.))]:
    invalid[name]=dict(pixel=pixel,homography=cam.pixel_to_world(*pixel),ray=cam.pixel_to_world_at_z(*pixel,0.))
r=rows[0]; original=contract(r); original_z=np.array(mapped(original)[0].xy_m)
wrong=dict(cameras); wrong[original.camera_id]=cameras['camera_A']
invalid['wrong_camera_map_output']=list(mapped(original,wrong)[0].xy_m)
invalid['wrong_camera_map_shift_m']=float(np.linalg.norm(np.array(invalid['wrong_camera_map_output'])-original_z))
for name,changes in [('wrong_calibration',dict(calibration_id='foreign_world_camera_B')),('pixel_box_disagree',dict(pixel_uv=(original.pixel_uv[0]+20,original.pixel_uv[1]))),('wrong_frame',dict(image_frame_id='foreign_optical')),('zero_box',dict(bbox_xyxy=(600.,400.,600.,400.))),('reversed_box',dict(bbox_xyxy=(660.,440.,600.,400.)))]:
    out=mapped(contract(r,**changes),admission=True)
    invalid[name]=dict(mapped=len(out),z=list(out[0].xy_m) if out else None)
# Camera processing-order permutations must not change geometry or NN identity.
permutation_max=0.
for order in itertools.permutations(cameras):
    changed=mapped(original,{c:cameras[c] for c in order})[0]
    permutation_max=max(permutation_max,float(np.max(abs(np.array(changed.xy_m)-original_z))))
R=np.array([[.04,.012],[.012,.09]])
H=np.array(observation_jacobian(cam,0.,0.,.3)); hsmall=np.array(observation_jacobian(cam,0.,0.,.3,step_m=1e-4))
eq=equivalent_position_measurement((.2,.1),R,cam,(0.,0.),.3)
optional=dict(hull_jacobian_step_difference=float(np.max(abs(H-hsmall))),hull_covariance_error=float(np.max(abs(np.array(eq[1])-np.linalg.inv(H)@R@np.linalg.inv(H).T))),condition_30_accepted=_inverse_2x2(((30.,0.),(0.,1.)),max_condition=20.) is not None)
transform=PixelToBevTransformer(cam.cam_pos,cam.look_at,1280,720,1.5708)
u,v=640.,400.; step=1e-4
J=np.column_stack([(np.array(cam.pixel_to_world(u+step*(a==0),v+step*(a==1)))-np.array(cam.pixel_to_world(u-step*(a==0),v-step*(a==1))))/(2*step) for a in (0,1)])
optional['pixel_noise_helper']=transform.pixel_noise_to_metric(u,v,.001)
optional['world_axis_std_oracle']=np.sqrt(np.diag(J@J.T))*.001
optional['floor_rank_one_eigenvalues']=np.linalg.eigvalsh(_floor_spd_2x2(((1e8,1e8),(1e8,1e8)),1e-12))
result=dict(source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/reliability/reliability/nodes/camera_manager_node.py',ROOT/'src/unav_common/unav_common/camera_model.py',ROOT/'src/reliability/reliability/projection.py',ROOT/'src/reliability/reliability/learned_box_correction.py',ROOT/'src/reliability/reliability/reference_calibration.py']},hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [mean,calpath,manifest,table,world,Path(__file__)]},counts=dict(counts),max_abs_errors=errors,example=example,invalid=invalid,permutations=120,permutation_max=permutation_max,optional=optional,horizon_v=horizon)
print(json.dumps(result,indent=2,default=lambda v:v.tolist()))
