"""One manifest-selected diagnostic run, observation-function equality only."""
import runpy, contextlib, io, json, csv, hashlib, sys
from pathlib import Path
import numpy as np
# Reuse the complete static geometry audit setup without repeating its JSON output.
with contextlib.redirect_stdout(io.StringIO()):
    p=runpy.run_path(str(Path(__file__).with_name('05_geometry_probe.py')))
ROOT=p['ROOT']; sys.path.insert(0,str(ROOT/'experiments/fusion_on_fixed_routes'))
import aligned
selection=ROOT/'logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json'
entry=next(e for e in json.loads(selection.read_text())['runs'] if e['key']=='fusion_network_traverse__P0__seed210')
run=ROOT/entry['run']
for name in ['run_manifest.json','experiment.csv','fusion_observations.csv','camera_opportunities.jsonl']:
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==entry['files'][name]
loaded=aligned.rows(run) # required loader; no truth or performance scoring needed.
delivered={}
for line in (run/'camera_opportunities.jsonl').open():
    envelope=json.loads(line); o=envelope['observation']
    if o['detection_valid'] and envelope['valid_contract']:
        delivered.setdefault((o['source_batch_id'],o['camera_id']),envelope)
count=0; worst=np.zeros(3); trace=None; seen=set(); malformed=0
for row in csv.DictReader((run/'fusion_observations.csv').open()):
    camera='camera_'+row['camera'] if len(row['camera'])==1 else row['camera']
    key=(row['source_batch_id'],camera)
    if key in seen or key not in delivered: continue
    seen.add(key); e=delivered[key]; o=p['CameraObservation'].from_dict(e['observation'])
    box=o.bbox_xyxy
    malformed+=int(not np.allclose(o.pixel_uv,[(box[0]+box[2])/2,box[3]],rtol=0,atol=1e-9) or o.calibration_id!='warehouse_v2_'+camera or o.image_frame_id!=camera)
    raw=np.array(p['cameras'][camera].pixel_to_world(*o.pixel_uv)); nn=p['model'].correct(camera,raw,box,o.detector_score)
    z,R=p['cal'].apply(camera,nn)
    logged_raw=np.array([float(row['raw_obs_x']),float(row['raw_obs_y'])]); logged_z=np.array([float(row['obs_x']),float(row['obs_y'])]); logged_R=np.array([[float(row['obs_cov_xx']),float(row['obs_cov_xy'])],[float(row['obs_cov_xy']),float(row['obs_cov_yy'])]])
    worst=np.maximum(worst,[np.max(abs(raw-logged_raw)),np.max(abs(np.array(z)-logged_z)),np.max(abs(np.array(R)-logged_R))]); count+=1
    if trace is None:
        trace=dict(source_batch_id=o.source_batch_id,camera=camera,calibration_id=o.calibration_id,image_frame_id=o.image_frame_id,box=box,pixel=o.pixel_uv,capture_stamp_s=o.timestamp_s,image_receive_stamp_s=o.image_receive_stamp_s,inference_start_stamp_s=o.inference_start_stamp_s,inference_finish_stamp_s=o.inference_finish_stamp_s,publish_stamp_s=o.publish_stamp_s,logger_receive_stamp_s=e['receive_stamp_s'],raw=raw.tolist(),nn=nn,bias=p['cal'].bias[camera].tolist(),z=z,R=R)
assert count>0 and np.all(worst<[1e-8,1e-8,1e-10])
print(json.dumps(dict(run=entry['run'],selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),checked=count,max_abs_difference_raw_z_R=worst.tolist(),malformed_logged_inputs=malformed,trace=trace),indent=2))
