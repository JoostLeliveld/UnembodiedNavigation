#!/usr/bin/env python3
"""Read-only frozen-artifact audit plus temporary synthetic defect reproductions.

No detector/NN fit, simulation, ROS initialization, or source/artifact mutation.
Assertions pin observed behavior, including defects, rather than certify repairs.
"""
from __future__ import annotations
import argparse, ast, copy, csv, hashlib, importlib.util, itertools, json, math, os
from pathlib import Path
from collections import Counter, defaultdict, deque
from types import SimpleNamespace
from unittest.mock import patch
import sys, tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/module12_mpl')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/p) for p in ('src/reliability','src/planning','src/state',
    'src/unav_common','src/experiments','scripts/perception','experiments/icra_commissioning',
    'experiments/camera_observation_characterization')]
import numpy as np
import cv2, joblib

OUT = Path(__file__).resolve().parent
CAP = ROOT/'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
MODEL = ROOT/'logs/perception_models/box_feature_bias_correction_20260831/models.joblib'
STUDY = ROOT/'logs/studies/icra_commissioning_20260905'
CAMERAS = tuple('camera_'+c for c in 'ABCDE')
RESULT = {}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def pixels_sha(a):
    h=hashlib.sha1();h.update(str(a.shape).encode('ascii'));h.update(str(a.dtype).encode('ascii'))
    h.update(a.tobytes());return h.hexdigest()

def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))

def writecsv(p,rows,fields=None):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def module(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def emit(name,value):
    RESULT[name]=value;print(name, json.dumps(value,default=str)[:900],flush=True)

def raised(fn):
    try:return {'accepted':True,'value':fn()}
    except Exception as e:return {'accepted':False,'error':type(e).__name__+': '+str(e)}

def ast_functions(path,names,env):
    tree=ast.parse(Path(path).read_text())
    code=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),'exec'),env)
    return env

def capture_cases(tmp):
    C=module('module12_capture','experiments/camera_observation_characterization/capture_bbox_grid.py')
    A=module('module12_resume','experiments/camera_observation_characterization/audit_capture_resume.py')
    # Run the actual resume branch in isolation before ROS initialization.
    tree=ast.parse(Path(C.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    branch=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='bool(args.resume)')
    branch_code=compile(ast.fix_missing_locations(ast.Module(body=branch.body,type_ignores=[])),C.__file__,'exec')
    cases={}
    for defect in ('none','duplicate_camera','missing_camera','pose_gap','timing','corrupt_image','changed_code',
                   'changed_geometry','changed_noise','mixed_pose','partial_success','complete_repeats','empty_prefix'):
        p=tmp/defect;p.mkdir(); repo=p/'repo'
        sources=[repo/x for x in ('poses.json','world.sdf','profiles.yaml',
            'experiments/camera_observation_characterization/capture_bbox_grid.py','scripts/perception/capture_yolo_dataset.py')]
        for f in sources:f.parent.mkdir(parents=True,exist_ok=True);f.write_text('frozen')
        repeats=2 if defect=='complete_repeats' else 1
        plan=dict(pose_count=3,planned_rows=15*repeats,repeats=repeats,batch_sync_slop_ms=50,
                  pose_file=str(sources[0]),pose_file_sha256=sha(sources[0]))
        manifest=dict(status='running',plan=plan,cameras=[{'camera_id':c} for c in CAMERAS],sensor_perturbation={'stddev_dn':0})
        for key,f in zip(('world','world_profiles','capture_script','capture_helper'),sources[1:]):
            manifest[key+'_sha256']=sha(f);manifest[key+'_path']=str(f)
        a=np.arange(432,dtype=np.uint8).reshape(12,12,3);cv2.imwrite(str(p/'image.png'),a)
        rows=[]
        for rep in range(repeats):
            for c in CAMERAS:
                r={k:'' for k in C.FIELDS};r.update(pose_id='0',repetition_id=str(rep),camera_id=c,
                    source_batch_id=f'pose_000000_r{rep:02d}',capture_status='ok',image='image.png',
                    image_sha1=pixels_sha(a),image_stamp_s=str(1+rep),batch_image_span_s='0',robot_x='0',robot_y='0',robot_yaw='0')
                rows.append(r)
        expected=copy.deepcopy(manifest)
        if defect=='duplicate_camera':rows[-1]['camera_id']=rows[0]['camera_id']
        if defect=='missing_camera':rows.pop()
        if defect=='pose_gap':
            for r in rows:r['pose_id']='1';r['source_batch_id']='pose_000001_r00'
        if defect=='timing':rows[-1]['image_stamp_s']='1.2'
        if defect=='corrupt_image':(p/'image.png').write_bytes(b'corrupt')
        if defect=='changed_code':sources[-1].write_text('changed')
        if defect=='changed_geometry':expected['cameras'][0]['pose_xyz_rpy']=[100,0,5,0,0,0]
        if defect=='changed_noise':expected['sensor_perturbation']['stddev_dn']=20
        if defect=='mixed_pose':rows[-1]['robot_x']='10'
        if defect=='partial_success':rows[-1]['capture_status']='failed';rows[-1]['image']=''
        if defect=='empty_prefix':rows=[]
        (p/'capture_manifest.json').write_text(json.dumps(manifest));writecsv(p/'capture_index.csv',rows,C.FIELDS)
        with patch.object(A.shutil,'disk_usage',return_value=SimpleNamespace(free=10**12)):
            preflight=A.audit(p,repo=repo)
        env=vars(C).copy();env.update(args=SimpleNamespace(resume=True),out=p,manifest_path=p/'capture_manifest.json',
            index_path=p/'capture_index.csv',plan=plan,specs=[None]*5,expected_manifest=expected)
        outcome=raised(lambda:exec(branch_code,env))
        cases[defect]={'writer_accepted':outcome['accepted'],'writer_error':outcome.get('error'),
                       'preflight_ready':preflight['ready'],'failed_checks':[k for k,v in preflight['checks'].items() if not v],
                       'start_pose_id':env.get('start_pose_id'),'retained_rows':env.get('rows_written')}
    assert cases['duplicate_camera']['writer_accepted'] and not cases['duplicate_camera']['preflight_ready']
    assert cases['corrupt_image']['writer_accepted'] and cases['changed_code']['writer_accepted']
    assert not cases['missing_camera']['writer_accepted']
    assert cases['mixed_pose']['preflight_ready']
    emit('capture_resume',cases)

    # Actual capture method, with only ROS boundary and clocks replaced.
    node=object.__new__(C.FiveCameraCapture)
    node.settle_s=.2;node.timeout_s=5.;node.min_new_rgb=3;node.with_semantic=False;node.batch_sync_slop_s=.05
    node.rgb_count={c:0 for c in CAMERAS};node.label_count={c:0 for c in CAMERAS}
    node.rgb={c:deque(maxlen=30) for c in CAMERAS}
    wall=[0.];commands=[]
    def monotonic():wall[0]+=.1;return wall[0]
    def spin():
        for j,c in enumerate(CAMERAS):
            node.rgb_count[c]+=1
            node.rgb[c].append((node.rgb_count[c],99_000_000_000,np.full((2,2,3),j,dtype=np.uint8)))
    node._spin=spin;node._set_pose=lambda *args:commands.append(dict(pose=args,sim_stamp_s=100.))
    with patch.object(C.rclpy,'ok',return_value=True),patch.object(C.time,'monotonic',side_effect=monotonic):
        first=node.capture(0,0,0,CAMERAS);second=node.capture(10,0,0,CAMERAS)
    assert all(p.image_stamp_ns==99_000_000_000 for p in first.values())
    assert [p.image_stamp_ns for p in first.values()]==[p.image_stamp_ns for p in second.values()]
    pair=lambda stamp,tag:C.Pair(np.full((1,1,3),tag,np.uint8),None,stamp,None,math.nan)
    mixed={c:[pair(100_000_000_000 if c==CAMERAS[0] else 100_040_000_000,j)] for j,c in enumerate(CAMERAS)}
    selected=C._closest_timestamp_batch(mixed,max_span_s=.05)
    permuted=[]
    for order in itertools.permutations(CAMERAS):
        s=C._closest_timestamp_batch({c:mixed[c] for c in order},max_span_s=.05)
        permuted.append({c:int(v.image[0,0,0]) for c,v in s.items()}=={c:j for j,c in enumerate(CAMERAS)})
    emit('capture_freshness',{'commands':commands,'both_captures_stamp_s':99,'duplicate_callbacks_satisfy_min_new_rgb':True,
        'different_pose_timestamp_window_accepted':selected is not None,'permutations_preserve_camera_pixels':sum(permuted)})

    # Converter interruption: first rename succeeds, second codec operation fails.
    E=module('module12_reencode','experiments/camera_observation_characterization/reencode_capture_lossless.py')
    p=tmp/'converter';p.mkdir();rr=[]
    for j in range(2):
        im=np.full((5,5,3),j,np.uint8);cv2.imwrite(str(p/f'{j}.png'),im)
        rr.append(dict(capture_status='ok',image=f'{j}.png',image_sha1=pixels_sha(im)))
    writecsv(p/'capture_index.csv',rr);(p/'capture_manifest.json').write_text(json.dumps({'status':'running'}))
    original=E.cv2.imencode;calls=[0]
    def fail_second(*args,**kwargs):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('injected codec interruption')
        return original(*args,**kwargs)
    with patch.object(sys,'argv',['reencode','--capture',str(p)]),patch.object(E.cv2,'imencode',side_effect=fail_second):
        stopped=raised(E.main)
    missing=[r['image'] for r in readcsv(p/'capture_index.csv') if not (p/r['image']).is_file()]
    assert missing==['0.png']
    emit('lossless_converter_interruption',{'failure':stopped,'indexed_files_missing':missing,'verified_pixels_survive_in_webp':pixels_sha(cv2.imread(str(p/'0.webp')))==rr[0]['image_sha1']})

def detector_cases(tmp):
    D=module('module12_detector','experiments/camera_observation_characterization/run_bbox_detector.py')
    p=tmp/'detector';p.mkdir();a=np.zeros((10,10,3),np.uint8);b=np.full_like(a,255)
    # Claimed original zero-image hash; actual on-disk image is changed.
    cv2.imwrite(str(p/'changed.png'),b);cv2.imwrite(str(p/'miss.png'),a)
    m=dict(status='complete_with_failed_batches',cameras=[dict(camera_id='camera_A',image_width=10,image_height=10)])
    (p/'capture_manifest.json').write_text(json.dumps(m));weights=p/'frozen.pt';weights.write_bytes(b'stub')
    rr=[]
    for j,(status,image,hash_) in enumerate([('ok','changed.png',pixels_sha(a)),('ok','miss.png',pixels_sha(b)),('failed','','')]):
        rr.append(dict(pose_id=j,position_id=j,heading_id=0,repetition_id=0,source_batch_id=f'pose_{j:06d}_r00',
            camera_id='camera_A',image=image,image_sha1=hash_,capture_status=status))
    writecsv(p/'capture_index.csv',rr)
    inputs=[]
    class FrozenFakeDetector:
        names={0:'robot'}
        def __init__(self,*a):pass
        def predict(self,**kwargs):
            images=[cv2.imread(f) for f in kwargs['source']];inputs.extend(pixels_sha(i) for i in images)
            return [SimpleNamespace(tag=int(i[0,0,0])) for i in images]
    def select(result,**kw):
        hit=result.tag==255
        return dict(bbox_xyxy=[2,2,8,8] if hit else None,detected=hit,n_candidates=int(hit),confidence=.9 if hit else 0,
                    bbox_bottom_u=5,bbox_bottom_v=8)
    with patch.object(D,'YOLO',FrozenFakeDetector),patch.object(D,'select_best_detection',side_effect=select),patch.object(sys,'argv',['det','--capture',str(p),'--weights',str(weights)]):D.main()
    result=readcsv(p/'bbox_observations.csv');meta=json.loads((p/'bbox_detector_manifest.json').read_text())
    assert result[0]['image_sha1']!=inputs[0] and len(result)==2 and result[1]['detected']=='0'
    emit('detector_input_identity_and_denominator',{'capture_rows':3,'detector_rows':len(result),'reported_attempt_rows':meta['attempt_rows'],
        'capture_failure_omitted':True,'real_miss_retained':True,'changed_pixels_accepted_with_old_hash':True,
        'positive_on_known_synthetic_empty_image_retained_as_detection':result[0]['detected']=='1'})

def dataset_cases(tmp):
    import dataset_split_utils as S
    import fit_bias_updates as F
    rows=[{'position_id':str(j),'robot_x':str(x),'robot_y':'.1','image_sha1':'same_detection'} for j,x in enumerate((1.999,2.001))]
    split=F.tile_split(rows,2.)
    records=S.build_pose_records([0,1],[0],[0,1])
    emit('splits',{'adjacent_positions_distance_m':.002,'checkerboard_splits':split,
        'identical_image_allowed_across_sides':len(set(split.values()))==2,
        'fraction_zero_grouped':S.assign_splits(records,val_fraction=0,split_mode='spatial_cell',spatial_block_size=1),
        'fraction_one_grouped':S.assign_splits(records,val_fraction=1,split_mode='spatial_cell',spatial_block_size=1)})
    # This small case never enters fitting: it proves the fallback reuses in-sample values.
    env=ast_functions(ROOT/'experiments/camera_observation_characterization/learn_measurement_covariance.py',
       ['out_of_fold_residuals'],{'np':np})
    data=dict(split=np.array(['train']*4),e=np.array([[.01,.02]]*4),
              rows=[{'position_id':str(i)} for i in range(4)],geometry={})
    oof=env['out_of_fold_residuals'](data,'nn',0,5)
    assert np.array_equal(oof,data['e'])
    emit('small_oof_fallback',{'rows':4,'fit_calls':0,'returns_in_sample_residuals_as_oof':True})
    # Actual iterator with controlled operational belief samples; no evaluator/truth input.
    import reliability.observation_exporter as E
    p=tmp/'honest_campaign_v1';folder=p/'route'/'C1'/'seed0'/'exp';folder.mkdir(parents=True)
    row=dict(log_stamp='1.0',detected='1',yolo_score_raw='.9',pixel_pose_available='1',pred_world_x='1',
             bbox_xmin='2',bbox_ymin='2',bbox_xmax='8',bbox_ymax='8',obs_u='5',obs_v='8')
    writecsv(folder/'perception.csv',[row,row,{**row,'log_stamp':'1.1','detected':'0','yolo_score_raw':'0','pixel_pose_available':'0'}])
    belief=dict(stamp=np.array([.9,1.05]),bx=np.array([0,9.]),by=np.array([0,0]),yaw=[0,0])
    stats={}
    with patch.object(E,'_load_belief',return_value=belief):out=list(E.iter_honest_campaign_raw_records(str(p),E.ExporterConfig(),stats))
    assert out[0][2]['state_x']==9 and len(out)==3
    emit('legacy_exporter',{'rows':len(out),'duplicate_rows_kept':out[0]==out[1],'joined_belief_stamp_s':1.05,
        'perception_receipt_stamp_s':1.,'miss_retained':not out[2][2]['detection_received'],'stats':stats,
        'frame_expected_and_received_assumed':out[0][2]['frame_expected'] and out[0][2]['frame_received']})
    import residual_bias_model as R
    rr=dict(image_width=1280,image_height=720,robot_yaw=.2,baseline_bearing_rad=.1,camera='camera_A',
        box_bottom_u=640,box_bottom_v=400,box_x1=600,box_x2=680,box_y1=300,box_y2=400,
        box_confidence=.8,baseline_range_m=5)
    first=R.feature_vector(rr);second=R.feature_vector({**rr,'robot_yaw':1.2})
    missing=rr.copy();del missing['robot_yaw']
    emit('provisional_heading_features',{'implicit_commanded_yaw_changes_features':not np.array_equal(first,second),
        'explicit_online_heading_works_without_truth':np.isfinite(R.feature_vector(missing,heading_rad=.2)).all().item(),
        'disable_heading_still_requires_robot_yaw':not raised(lambda:R.feature_vector(missing,disable_heading=True))['accepted'],
        'nonfinite_score_returns_nonfinite_features':not np.isfinite(R.feature_vector({**rr,'box_confidence':np.nan})).all().item()})

def operational_cases():
    from reliability.operational_residual import build_operational_residuals,summarize_residuals,shrink_summary
    mu=np.array([[1.,2.],[2.,3.]])
    P=np.zeros((2,2,2))
    meas=[SimpleNamespace(index=i,z=mu[i]+[.1*(-1)**i,0],source='camera_A') for i in range(2)]
    rec=build_operational_residuals(smoothed_mean=mu,smoothed_cov=P,measurements=meas,camera_id='camera_A')
    a=summarize_residuals(rec);b=summarize_residuals(rec*10)
    s1=shrink_summary(a,np.eye(2));s2=shrink_summary(b,np.eye(2))
    # Affine projection is the simplest counterexample to interpreting its Jacobian as h.
    H=np.diag([2.,3.]);offset=np.array([640.,360.])
    uv=build_operational_residuals(smoothed_mean=mu,smoothed_cov=P,
       measurements=[SimpleNamespace(index=0,z=H@mu[0]+offset,source='camera_A')],
       camera_id='camera_A',frame='uv',observation_jacobian=H)
    emit('operational_residuals',{'unique_records':a.sample_count,'duplicated_records':b.sample_count,
        'shrinkage_before':s1.shrinkage_lambda,'shrinkage_after_duplicates':s2.shrinkage_lambda,
        'default_empty_anchor_list_claims_held_out':a.held_out,'affine_uv_expected_residual':[0,0],
        'affine_uv_observed_residual':uv[0].residual})

def frozen_checks(tmp,check_pixels):
    import fit_bias_updates as F
    from reliability.learned_box_correction import LearnedBoxCorrection
    from reliability.reference_calibration import ReferenceCalibration
    from planning.core.camera_network import CameraNetworkModel
    from scipy.spatial import cKDTree
    import model  # required by the frozen commissioning pickle
    import commissioned_field  # required by the frozen field pickle
    # Record bytes before any model loads, including exact source import identities.
    source_files=[ROOT/x for x in (
        'experiments/camera_observation_characterization/capture_bbox_grid.py',
        'experiments/camera_observation_characterization/audit_capture_resume.py',
        'experiments/camera_observation_characterization/run_bbox_detector.py',
        'experiments/camera_observation_characterization/derive_interpretations.py',
        'experiments/camera_observation_characterization/reencode_capture_lossless.py',
        'experiments/camera_observation_characterization/fit_bias_updates.py',
        'experiments/camera_observation_characterization/package_bias_model.py',
        'experiments/camera_observation_characterization/learn_measurement_covariance.py',
        'scripts/perception/dataset_split_utils.py','scripts/perception/build_residual_bias_dataset.py',
        'scripts/perception/residual_bias_model.py','scripts/perception/train_residual_bias.py',
        'experiments/icra_commissioning/model.py','experiments/icra_commissioning/commissioned_field.py',
        'experiments/icra_commissioning/study.py','experiments/icra_commissioning/field_study.py',
        'experiments/icra_commissioning/export_reference_calibration.py','experiments/icra_commissioning/export_network_planner.py',
        'src/reliability/reliability/observation_exporter.py','src/reliability/reliability/observation_gp.py',
        'src/reliability/reliability/operational_residual.py','src/reliability/reliability/learned_box_correction.py',
        'src/reliability/reliability/reference_calibration.py','src/planning/planning/core/camera_network.py',
        'scripts/visibility_comparison/fit_belief_aware_gp.py','src/reliability/reliability/nodes/camera_manager_node.py')]
    frozen_files=[MODEL,ROOT/'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt',
        STUDY/'models.joblib',STUDY/'manifest.json',STUDY/'field_study/field.joblib',
        STUDY/'field_study/manifest.json',STUDY/'field_study/selection.json',
        STUDY/'network_planner/manifest.json',STUDY/'network_planner/reference_calibration.json',
        ROOT/'logs/studies/measurement_commissioning/calibration.json']
    frozen_files += [STUDY/'network_planner'/f'{k}.npz' for k in ('uniform','geometry','gp')]
    frozen_files += [CAP/x for x in ('capture_index.csv','capture_manifest.json','bbox_observations.csv',
        'bbox_detector_manifest.json','observation_interpretations.csv','observation_interpretations_manifest.json',
        'bias_update_interpretations.csv','bias_update_interpretations_manifest.json')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files+frozen_files}
    references=[]
    manifests=[CAP/'capture_manifest.json',CAP/'bbox_detector_manifest.json',CAP/'observation_interpretations_manifest.json',
        CAP/'bias_update_interpretations_manifest.json',MODEL.parent/'summary.json',STUDY/'manifest.json',
        STUDY/'field_study/manifest.json',STUDY/'network_planner/manifest.json',STUDY/'network_planner/reference_calibration.json']
    for p in manifests:
        d=json.loads(p.read_text())
        for section in ('files','sources','source_hashes'):
            for rel,h in d.get(section,{}).items():
                aliases={'bias_update_interpretations':CAP/'bias_update_interpretations.csv',
                         'bias_update_manifest':CAP/'bias_update_interpretations_manifest.json',
                         'capture_manifest':CAP/'capture_manifest.json'}
                f=aliases.get(rel,Path(rel));f=f if f.is_absolute() else ROOT/f
                references.append(dict(manifest=str(p.relative_to(ROOT)),field=section,path=str(f),expected=h,
                    actual=sha(f) if f.is_file() else None,matches=f.is_file() and sha(f)==h))
    capmeta=json.loads((CAP/'capture_manifest.json').read_text());detmeta=json.loads((CAP/'bbox_detector_manifest.json').read_text())
    for label,p,h in [('world',Path(capmeta['world_path']),capmeta['world_sha256']),
        ('profiles',Path(capmeta['world_profiles_path']),capmeta['world_profiles_sha256']),
        ('detector_weights',Path(detmeta['weights']),detmeta['weights_sha256']),
        ('capture_index',CAP/'capture_index.csv',capmeta['capture_index_sha256']),
        ('detector_output',CAP/'bbox_observations.csv',detmeta['bbox_observations_sha256'])]:
        references.append(dict(manifest='capture/detector',field=label,path=str(p),expected=h,actual=sha(p),matches=sha(p)==h))
    records=readcsv(CAP/'capture_index.csv');rows=readcsv(CAP/'bias_update_interpretations.csv')
    batches=defaultdict(list)
    for r in records:batches[r['pose_id'],r['repetition_id']].append(r)
    actualspans=[max(float(r['image_stamp_s']) for r in rs)-min(float(r['image_stamp_s']) for r in rs) for rs in batches.values()]
    raw=[r for r in rows if r['raw_valid']=='1'];hit_hashes=Counter(r['image_sha1'] for r in raw)
    hash_splits=defaultdict(set)
    for r in rows:hash_splits[r['image_sha1']].add(r['split'])
    cross={h for h,s in hash_splits.items() if len(s)>1}
    positions={r['position_id']:(float(r['robot_x']),float(r['robot_y']),r['split']) for r in rows}
    trainxy=np.array([v[:2] for v in positions.values() if v[2]=='train']);testxy=np.array([v[:2] for v in positions.values() if v[2]=='test'])
    distances=cKDTree(trainxy).query(testxy)[0]
    manifest=json.loads((STUDY/'manifest.json').read_text())
    def role(r):return 'mean_train' if r['split']=='train' else manifest['roles'][f"{math.floor(float(r['robot_x'])/2)}:{math.floor(float(r['robot_y'])/2)}"]
    image_roles=defaultdict(set)
    for r in raw:image_roles[r['image_sha1']].add(role(r))
    emit('frozen_capture',{'schema':capmeta['schema'],'rows':len(records),'poses':len(batches),'positions':len(positions),
       'hits':len(raw),'misses':sum(r['detected']=='0' for r in rows),'unique_decoded_hashes':len(hash_splits),
       'source_batch_ids_present':all(bool(r.get('source_batch_id')) for r in records),
       'exact_camera_batches':all(len(rs)==5 and {r['camera_id'] for r in rs}==set(CAMERAS) for rs in batches.values()),
       'poses_with_span_above_005s':sum(s>.05+1e-8 for s in actualspans),'max_span_s':max(actualspans),
       'reported_span_matches_recomputed':all(abs(float(r['batch_image_span_s'])-s)<1e-8 for rs,s in zip(batches.values(),actualspans) for r in rs),
       'duplicate_hit_hash_groups':sum(v>1 for v in hit_hashes.values()),'cross_split_hash_groups':len(cross),
       'cross_split_hit_hash_groups':len(cross & set(hit_hashes)),
       'cross_commissioning_role_hit_hash_groups':sum(len(s)>1 for s in image_roles.values()),
       'test_to_train_position_distance_m':dict(min=float(distances.min()),median=float(np.median(distances)),max=float(distances.max())),
       'role_opportunities':dict(Counter(role(r) for r in rows)),'role_hits':dict(Counter(role(r) for r in raw))})
    pixel_results=[]
    unique={r['image']:r for r in records}
    chosen=unique.values() if check_pixels else list(unique.values())[:10]
    for j,r in enumerate(chosen):
        p=CAP/r['image'];a=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if a is None or pixels_sha(a)!=r['image_sha1']:
            pixel_results.append({'image':r['image'],'reason':'missing/corrupt/changed_pixels'})
        if j and j%1000==0:print('verified images',j,flush=True)
    emit('frozen_pixels',{'scope':'all indexed unique paths' if check_pixels else 'first 10 paths',
        'checked_paths':len(unique) if check_pixels else min(10,len(unique)),'problems':pixel_results})
    # Frozen inference parity, no refit.
    payload=joblib.load(MODEL);mean=LearnedBoxCorrection(MODEL)
    cal=ReferenceCalibration(STUDY/'network_planner/reference_calibration.json',MODEL,CAMERAS)
    models=joblib.load(STUDY/'models.joblib')
    delta=[];rdelta=[];featured=[];example=None
    for r in raw:
        c=r['camera_id'];g=payload['camera_geometry'][c]
        onehot=np.array([c==x for x in payload['camera_ids']],float)
        x=np.concatenate([F.features(r,g),onehot]);box=[float(r[k]) for k in ('x0','y0','x1','y1')]
        rawxy=[float(r['raw_x']),float(r['raw_y'])]
        offline=F.apply_correction(r,g,payload['neural_model'].predict(x[None])[0]);runtime=mean.correct(c,rawxy,box,float(r['confidence']))
        delta.append(float(np.max(np.abs(offline-runtime))));featured.append(float(np.max(np.abs(x-mean._features(c,rawxy,box,float(r['confidence']))))))
        z,R=cal.apply(c,runtime);zm,Rm=models[c,'constant'].predict([{'z':offline}])
        rdelta.append(float(max(np.max(np.abs(zm[0]-z)),np.max(np.abs(Rm[0]-R)))))
        if example is None and c=='camera_B' and role(r)=='covariance_fit':
            caprow=next(q for q in records if q['pose_id']==r['pose_id'] and q['camera_id']==c and q['repetition_id']==r['repetition_id'])
            example=dict(capture=caprow,interpretation=r,role=role(r),features=x.tolist(),raw_xy=rawxy,
                         nn_xy=list(runtime),reference_xy=list(z),R_m2=R,mean_hash=sha(MODEL))
    assert max(delta)<1e-12 and max(rdelta)<1e-12
    emit('frozen_inference',{'rows':len(raw),'max_feature_difference':max(featured),'max_nn_xy_difference_m':max(delta),
       'max_reference_z_R_difference':max(rdelta),'no_fitting':True})
    RESULT['sample_trace']=example
    query_args=('camera_B',example['raw_xy'],[float(example['interpretation'][k]) for k in ('x0','y0','x1','y1')],float(example['interpretation']['confidence']))
    known=mean.correct(*query_args)
    checks={'unknown_camera_returns_none':mean.correct('unknown',*query_args[1:]) is None,
        'nonfinite_raw_returns_none':mean.correct('camera_B',[np.nan,0],*query_args[2:]) is None,
        'nonfinite_score_returns_none':mean.correct(*query_args[:3],np.nan) is None,
        'missing_bbox_returns_none':mean.correct('camera_B',query_args[1],None,.5) is None}
    variants={'unknown_version':lambda p:p.update(schema='v999'),
        'missing_feature_metadata':lambda p:p.pop('feature_names'),
        'missing_target':lambda p:p.pop('target'),
        'missing_B_geometry':lambda p:p['camera_geometry'].pop('camera_B'),
        'foreign_camera_geometry':lambda p:p['camera_geometry'].update(camera_Z=copy.deepcopy(p['camera_geometry']['camera_B']))}
    for name,mutate in variants.items():
        v=copy.deepcopy(payload);mutate(v);p=tmp/(name+'.joblib');joblib.dump(v,p)
        def operation():
            loaded=LearnedBoxCorrection(p)
            out=loaded.correct('camera_Z',*query_args[1:]) if name=='foreign_camera_geometry' else loaded.correct(*query_args)
            return None if out is None else list(out)
        checks[name]=raised(operation)
    permutations=[]
    for order in itertools.permutations(CAMERAS):
        c=ReferenceCalibration(STUDY/'network_planner/reference_calibration.json',MODEL,order)
        permutations.append(c.apply('camera_B',known)==cal.apply('camera_B',known))
    checks['reference_camera_permutations_equal']=sum(permutations)
    field=joblib.load(STUDY/'field_study/field.joblib');p=tmp/'field_roundtrip.joblib';joblib.dump(field,p);roundtrip=joblib.load(p)
    queries=field.poses[:4];gpdelta=[]
    for name,m in field.availability.items():
        gpdelta.append(float(np.max(np.abs(m.predict(queries)-roundtrip.availability[name].predict(queries)))))
    checks['frozen_field_pickle_roundtrip_max_q_difference']=max(gpdelta)
    checks['field_nan_pose_rejected']=not raised(lambda:field.forecast([np.nan,0,0],np.eye(3)))['accepted']
    # Exact GP grid export should reproduce the original fitted q at grid nodes.
    qdiff={}
    for kind,qkind in zip(('uniform','geometry','gp'),('constant','geometry_xy','gp_xy')):
        with np.load(STUDY/'network_planner'/f'{kind}.npz',allow_pickle=False) as d:
            X,Y=np.meshgrid(d['xs'],d['ys']);Q=np.column_stack([X.ravel(),Y.ravel(),np.zeros(X.size)])
            q=field.availability[qkind].predict(Q).T.reshape(d['availability'].shape)
            supported=np.any(d['availability']>0,axis=0)
            qdiff[kind]=float(np.max(np.abs(q[:,supported]-d['availability'][:,supported])))
    checks['frozen_field_to_npz_max_q_difference_at_supported_grid_nodes']=qdiff
    emit('artifact_input_and_roundtrips',checks)
    changed=[str(p.relative_to(ROOT)) for p in frozen_files if sha(p)!=hashes[str(p.relative_to(ROOT))]]
    emit('artifact_preservation',{'changed_frozen_files':changed})
    RESULT['provenance']={'sha256':hashes,'references':references,'imported_sources':dict(model=model.__file__,commissioned_field=commissioned_field.__file__,F=F.__file__)}
    emit('provenance_check_summary',{'checked':len(references),'mismatches':[r for r in references if not r['matches']]})

def main():
    p=argparse.ArgumentParser();p.add_argument('--all-pixels',action='store_true');p.add_argument('--synthetic-only',action='store_true');args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='module12_') as td:
        tmp=Path(td)
        capture_cases(tmp);detector_cases(tmp);dataset_cases(tmp);operational_cases()
        if not args.synthetic_only:frozen_checks(tmp,args.all_pixels)
    (OUT/'12_probe_results.json').write_text(json.dumps(RESULT,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
    print('wrote',OUT/'12_probe_results.json')

if __name__=='__main__':main()
