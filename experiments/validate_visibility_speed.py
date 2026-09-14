from pathlib import Path
import json,sys,math,hashlib,pickle,collections,os
import numpy as np
R=Path(__file__).resolve().parents[1];O=Path(os.environ.get('BAYESIAN_NAVIGATION_ROOT',R/'logs/bayesian_navigation'));version=sys.argv[1] if len(sys.argv)>1 else 'v1';H=R/('experiments/visibility_speed_1mps_'+version);T=O/('visibility_speed_1mps_reporting' if version=='v1' else 'visibility_speed_1mps_'+version+'_reporting')
sys.path[:0]=[str(H),str(O),str(O/'final_model'),str(R/'experiments/fusion_on_fixed_routes'),str(R/'src/unav_common'),str(O/'visibility_pilot_reporting')]
from runtime_engine import RuntimeEngine
from aligned import TruthSeries
from route_handoff import RouteHandoff
from audit_sources import audit
from unav_common.occlusion_geometry import scene_from_json
from fast_geometry import PreparedFootprint
from data import metrics
rows=[]
selection=json.loads((T/'selection.json').read_text())
for cfg in selection['runs']:
 D=O/'navigation'/cfg['id']
 assert (D/'supervisor.json').exists(),('Missing selected completed run',cfg['id'])
 s=json.loads((D/'summary.json').read_text());rt=[json.loads(l) for l in (D/'runtime.jsonl').open()];ev=[json.loads(l) for l in (D/'events.jsonl').open()];od={};source_pass=True
 for path,h in json.loads((D/'source_freeze.json').read_text()).items():source_pass &= hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 for r in ev:
  if r['topic']!='/odom_noisy':continue
  m=r['message'];p=m['pose']['pose'];st=m['header']['stamp'];q=p['orientation'];od[st['sec']*10**9+st['nanosec']]=[p['position']['x'],p['position']['y'],math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))]
 model=pickle.loads((D/'source_snapshot/workspace/final_model/final_process.pkl').read_bytes());e=RuntimeEngine(model,[cfg['x'],cfg['y'],cfg['yaw']],np.diag([.01,.01,np.deg2rad(5)**2]),False);dm=[];dc=[];statuses=True
 for r in rt:
  if r['kind']=='belief':
   e.add_odometry(r['stamp_ns'],np.array(od[r['stamp_ns']]));dm.append(float(np.max(abs(e.current.mean[:3]-r['mean']))));dc.append(float(np.max(abs(e.current.cov[:3,:3]-r['covariance']))))
  elif r['kind']=='batch' and r['result'].get('reason')!='run_ended_pending':
   obs=[dict(observation=dict(camera_id=z['camera'],source_frame_id=z['source_frame_id'],capture_stamp_ns=z['capture_stamp_ns']),features=np.array(z['features'],dtype=np.float32),world_xy=np.array(z['world_xy'])) for z in r.get('readings',[])];z=e.assimilate(r['batch_id'],obs);statuses &= z['status']==r['result']['status'] and z['reason']==r['result']['reason']
 nt=[];poses=[]
 for l in (D/'native_pose.jsonl').open():
  r=json.loads(l);p=next((p for p in r.get('pose',[]) if p.get('name')=='turtlebot3'),None)
  if p:
   sh=r['header']['stamp'];q=p['orientation'];nt.append(int(sh.get('sec',0))+int(sh.get('nsec',0))*1e-9);poses.append([p['position']['x'],p['position']['y'],math.atan2(2*(q.get('w',0)*q.get('z',0)+q.get('x',0)*q.get('y',0)),1-2*(q.get('y',0)**2+q.get('z',0)**2))])
 poses=np.array(poses);truth=TruthSeries(nt,*poses.T,'native dynamic_pose',max_reference_gap_s=.15);start=next(r['stamp_ns'] for r in rt if r['kind']=='navigation_start');end=s['end_stamp_ns'];br=[r for r in rt if r['kind']=='belief' and start<=r['stamp_ns']<=end];st=np.array([r['stamp_ns']/1e9 for r in br]);support=truth.support(st);mask=(np.array(nt)>=start/1e9)&(np.array(nt)<=end/1e9);xy=poses[mask];ground=np.array(truth.at(st[support])).T;means=np.array([r['mean'] for r in br]);cov=np.array([r['covariance'] for r in br]);met=metrics(means[support,:2]-ground,cov[support,:2,:2]);end_xy=np.array(truth.at([end/1e9]))[:,0];goal_error=float(np.linalg.norm(end_xy-cfg['goal']))
 body=PreparedFootprint(scene_from_json((D/'source_snapshot/collision_geometry.json').read_text()).prisms);minclear=min(body.sweep_clearance(a,b) for a,b in zip(xy[:-1],xy[1:]))
 observations=[json.loads(r['message']['data']) for r in ev if 'camera_observation/' in r['topic']];observed=[o['source_frame_id'] for o in observations];members=[k for r in rt if r['kind']=='batch' for k in r['all_member_ids']];ledger=len(members)==len(set(members)) and set(members)==set(observed)
 h=RouteHandoff(json.loads((D/'global_plan.json').read_text())['states'],cfg['goal']);hd=[]
 for r in rt:
  if r['kind']!='plan':continue
  f=r['forecast'];target,z=h.query(f['full_input_mean']);hd.extend([float(np.max(abs(target-f['local_goal'])))]+[abs(z[k]-f[k]) for k in ['global_progress_m','global_path_deviation_m','global_length_m']])
 try:identity=audit(D)
 except Exception as ex:identity={'passed':False,'error':repr(ex)}
 commands=[r for r in rt if r['kind']=='command' and start<=r['receive_stamp_ns']<=end];applied=[r['message']['linear']['x'] for r in ev if r['topic']=='/cmd_vel' and start<=r['receive_stamp_ns']<=end];cmdv=np.array([r['v'] for r in commands]);move=cmdv[cmdv>0];bids={o['source_batch_id']:o for o in observations if start<=o['capture_stamp_ns']<=end};cts=np.sort([o['capture_stamp_ns']/1e9 for o in bids.values()]);sup=json.loads((D/'supervisor.json').read_text())
 result=dict(evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),id=cfg['id'],arm=cfg['arm'],status=s['status'],sim_s=s['sim_duration_s'],driver_wall_s=s['wall_duration_s'],total_wall_s=sup['wall_s'],max_command_m_s=float(cmdv.max()),p95_moving_command_m_s=float(np.quantile(move,.95)) if len(move) else 0,max_applied_command_m_s=max(applied,default=0),fraction_commands_at_least_0_9=float(np.mean(cmdv>=.9)),command_reasons=dict(collections.Counter(r['reason'] for r in commands)),path_m=float(np.linalg.norm(np.diff(xy[:,:2],axis=0),axis=1).sum()),physical_goal_error_cm=goal_error*100,min_interpolated_body_clearance_m=minclear,navigation_pass=bool(s['status']=='belief_goal_reached' and goal_error<=.1 and minclear>=0),source_pass=bool(source_pass),replay_max_mean_diff=max(dm),replay_max_cov_diff=max(dc),replay_pass=bool(max(dm)<1e-7 and max(dc)<1e-9 and statuses),unsupported_belief_stamps=int((~support).sum()),ledger_pass=bool(ledger),handoff_max_difference=max(hd),handoff_pass=bool(max(hd)<1e-10),detector_identity=identity,received_cycles=len(bids),median_capture_interval_s=float(np.median(np.diff(cts))) if len(cts)>1 else None,**met)
 rows.append(result);print(json.dumps(result),flush=True)
 (T/'validation.json').write_text(json.dumps(rows,indent=2))
