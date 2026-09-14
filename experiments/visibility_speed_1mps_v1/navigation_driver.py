from paths import *
"""Dedicated live joint-localization / CasADi EFE integration harness.

Reference poses are recorded by a subprocess only. Controller and estimator
receive camera outputs, encoder odometry, commanded initialization and map.
This isolated campaign uses the GP network forecast and logs each full joint
planning input and covariance trajectory. Reference input remains offline only.
"""
import json,time,math,os,subprocess,signal,pickle,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rosidl_runtime_py.convert import message_to_ordereddict
from rgb_runtime import RGBEstimator
from runtime_engine import RuntimeEngine
from speed_profile import PROFILE,validate_config,profile_hash
from speed_planner import planner
from planning.planners.base_planner import PlanResult
D=Path(os.environ['CAPTURE_SWEEP_DIR']);cfg=json.loads((D/'capture.json').read_text());validate_config(cfg);model=pickle.loads((O/cfg['model_file']).read_bytes());rgb=RGBEstimator();engine=RuntimeEngine(model,[cfg['x'],cfg['y'],cfg['yaw']],np.diag([.1**2,.1**2,np.deg2rad(5)**2]),independent=cfg['arm']=='IndependentIndependent');plan=planner(rgb.models[cfg.get('planner_camera','camera_A')],cfg['seed']);worker=subprocess.Popen(['python3',str(HERE/'planner_worker.py'),str(D)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(D/'planner_worker.log').open('w'),text=True,bufsize=1);goal=np.array(cfg['goal']);pool=ThreadPoolExecutor(max_workers=1)
rclpy.init();node=rclpy.create_node('bayesian_navigation');node.set_parameters([rclpy.parameter.Parameter('use_sim_time',value=True)])
native_file=(D/'native_pose.jsonl').open('w');native=subprocess.Popen(['ign','topic','-e','--json-output','-t','/world/warehouse_v2/dynamic_pose/info'],stdout=native_file,stderr=subprocess.DEVNULL)
f=(D/'events.jsonl').open('w');jf=(D/'runtime.jsonl').open('w');batches=defaultdict(dict);seen=set();terminal=set();counts=Counter();odom=None
wall=time.monotonic();start=None;future=None;lastplan=-1e9;lastpub=-1e9;active=None;status='wall_timeout';failure='';plan_count=0;start_belief=None;end_stamp=None;last_source_stamp=None

def solve(mean,cov,goal,progress_index,source_age_s):
 worker.stdin.write(json.dumps(dict(mean=mean.tolist(),cov=cov.tolist(),goal=goal.tolist(),progress=progress_index,source_age_s=source_age_s))+'\n');worker.stdin.flush();line=worker.stdout.readline();assert line, 'planner process stopped';r=json.loads(line);forecast=r.pop('forecast');r['states']=np.array(r['states']);r['controls']=np.array(r['controls']);result=PlanResult(**r);result.forecast=forecast;return result

def write(kind,**kw):jf.write(json.dumps(dict(kind=kind,wall_monotonic_s=time.monotonic(),receive_stamp_ns=node.get_clock().now().nanoseconds,**kw),separators=(',',':'))+'\n');jf.flush()
def log(topic,msg):f.write(json.dumps(dict(topic=topic,receive_stamp_ns=node.get_clock().now().nanoseconds,message=message_to_ordereddict(msg)),separators=(',',':'))+'\n');f.flush()
def yaw(m):
 q=m.pose.pose.orientation;return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
def odcb(msg):
 global odom
 odom=msg;log('/odom_noisy',msg);p=msg.pose.pose.position;s=msg.header.stamp;t=s.sec*10**9+s.nanosec;engine.add_odometry(t,[p.x,p.y,yaw(msg)]);write('belief',stamp_ns=t,mean=engine.current.mean[:3].tolist(),covariance=engine.current.cov[:3,:3].tolist())
def camcb(cam,msg):
 log('/perception/camera_observation/'+cam,msg);o=json.loads(msg.data);key=o['source_frame_id'];assert key not in seen;seen.add(key);batches[o['source_batch_id']][cam]=o
node.create_subscription(Odometry,'/odom_noisy',odcb,QoSProfile(depth=1000,reliability=ReliabilityPolicy.BEST_EFFORT))
node.create_subscription(Odometry,'/odom',lambda m:log('/odom',m),qos_profile_sensor_data)
node.create_subscription(Twist,'/cmd_vel',lambda m:log('/cmd_vel',m),100)
node.create_subscription(String,'/sim/contact_channel_status',lambda m:log('/sim/contact_channel_status',m),100)
try:
 from ros_gz_interfaces.msg import Contacts
 node.create_subscription(Contacts,'/world_contacts',lambda m:log('/world_contacts',m),100)
except ImportError:write('contact_subscription_unavailable')
for c in ['camera_'+x for x in 'ABCDE']:node.create_subscription(String,'/perception/camera_observation/'+c,lambda m,c=c:camcb(c,m),100)
pub=node.create_publisher(Twist,'/cmd_vel_raw',10)
try:
 while time.monotonic()-wall<720:
  # Drain a bounded set of queued messages before each inference/control decision.
  for _ in range(20):rclpy.spin_once(node,timeout_sec=.001)
  now=node.get_clock().now().nanoseconds/1e9
  for bid in list(batches):
   group=batches[bid]
   if len(group)!=5:continue
   hi=max(int(o['capture_stamp_ns']) for o in group.values())
   if not engine.times or hi>engine.times[-1]:continue
   last_source_stamp=hi;rgb_t=time.perf_counter();readings,refusals=rgb.predict(list(group.values()),D/'crops');rgb_dt=time.perf_counter()-rgb_t;res=engine.assimilate(bid,readings);counts[res['status']]+=1
   write('batch',rgb_wall_s=rgb_dt,batch_id=bid,result=res,all_member_ids=[o['source_frame_id'] for o in group.values()],refusals=[dict(source_frame_id=o['source_frame_id'],reason=r) for o,r in refusals],readings=[dict(source_frame_id=r['observation']['source_frame_id'],camera=r['observation']['camera_id'],capture_stamp_ns=r['observation']['capture_stamp_ns'],features=r['features'].tolist(),world_xy=r['world_xy'].tolist()) for r in readings]);terminal.add(bid);del batches[bid]
  if engine.times and counts['accepted']>=2 and start is None:start=now;start_belief=engine.current.mean[:3].tolist();write('navigation_start',stamp_ns=engine.times[-1],goal=goal.tolist(),mean=start_belief)
  if future is not None and future.done():
   result=future.result();activation=submitted;rebased=False
   if np.linalg.norm(engine.current.mean[:2]-result.states[0,:2])<.02 and abs(np.arctan2(np.sin(engine.current.mean[2]-result.states[0,2]),np.cos(engine.current.mean[2]-result.states[0,2])))<.05:
    activation=now;rebased=True
   active=(result,activation);future=None;plan_count+=1;write('plan',activation_sim_s=activation,rebased_stationary=rebased,index=plan_count,belief_stamp_ns=plan_stamp,submitted_sim_s=submitted,backend=result.backend,solve_wall_s=result.solve_time_s,optimizer_success=result.optimizer_success,valid=result.rollout_valid,invalid_reason=result.invalid_reason,cost=result.total_cost,risk=result.risk_cost,ambiguity=result.ambiguity_cost,obstacle=result.obstacle_cost,states=result.states.tolist(),controls=result.controls.tolist(),forecast=result.forecast)
  if start is not None:
   distance=float(np.linalg.norm(engine.current.mean[:2]-goal))
   if distance<=cfg.get('goal_tolerance',.08):status='belief_goal_reached';end_stamp=engine.times[-1];break
   if now-start>cfg.get('sim_timeout',70):status='sim_timeout';end_stamp=engine.times[-1];break
   if future is None and now-lastplan>=.5:
    m=engine.current.mean.copy();S=engine.current.cov.copy();submitted=now;plan_stamp=engine.times[-1];future=pool.submit(solve,m,S,goal,progress_index=plan_count,source_age_s=max(0.,(engine.times[-1]-(last_source_stamp or engine.times[-1]))/1e9));lastplan=now
  if now-lastpub>=.1:
   cmd=Twist();reason='waiting'
   if active is not None and engine.times and now-engine.times[-1]/1e9<.5:
    result,t0=active
    if result.rollout_valid:
     from path_tracker import track
     pose=engine.current.mean[:3];u,tracking=track(pose,result.states,result.controls,goal)
     if tracking['deviation_m']<.15:
      from unav_common.rectangular_footprint import constant_twist_pose
      end=constant_twist_pose(pose,u,.1);clearance=plan._footprint_collision_model.sweep_clearance(pose,end,control=u,dt=.1);P=engine.current.cov[:3,:3];margin=2.45*(np.sqrt(np.linalg.eigvalsh(P[:2,:2]).max())+plan._footprint_collision_model.radius*np.sqrt(P[2,2]))
      if clearance>margin:cmd.linear.x=float(u[0]);cmd.angular.z=float(u[1]);reason='plan'
      else:reason='belief_clearance_guard'
     else:reason='plan_pose_deviation'
    else:reason='plan_horizon_expired'
   pub.publish(cmd);log('/cmd_vel_raw',cmd);write('command',reason=reason,v=cmd.linear.x,w=cmd.angular.z,belief_stamp_ns=engine.times[-1] if engine.times else None);lastpub=now
except KeyboardInterrupt:
 status='interrupted';failure='Run interrupted during infrastructure audit'
except Exception:
 status='exception';failure=traceback.format_exc();print(failure,flush=True)
finally:
 terminal_mean=engine.current.mean[:3].tolist();terminal_cov=engine.current.cov[:3,:3].tolist();terminal_stamp=engine.times[-1] if engine.times else None
 for _ in range(20):
  if not rclpy.ok():break
  pub.publish(Twist());rclpy.spin_once(node,timeout_sec=.02)
 for bid,group in batches.items():write('batch',batch_id=bid,result=dict(status='dropped',reason='run_ended_pending'),all_member_ids=[o['source_frame_id'] for o in group.values()]);counts['pending_at_stop']+=1
 native.send_signal(signal.SIGINT)
 try:native.wait(timeout=3)
 except subprocess.TimeoutExpired:native.terminate();native.wait(timeout=3)
 native_file.close();pool.shutdown(wait=True);worker.stdin.close();worker.wait(timeout=10);summary=dict(speed_profile=PROFILE,speed_profile_sha256=profile_hash(),status=status,failure=failure,sim_duration_s=None if start is None else now-start,wall_duration_s=time.monotonic()-wall,goal=goal.tolist(),end_belief=terminal_mean,end_covariance=terminal_cov,end_stamp_ns=end_stamp or terminal_stamp,start_belief=start_belief,plans=plan_count,batches=dict(counts),camera_frames=len(seen),controller_inputs='RGB camera readings, encoder odometry, commissioned model, commanded initialization and static map only',planner_backend='casadi',forecast='Fixed independent future-reading forecast; '+cfg['arm']+' observation interface; runtime fusion retains joint camera-error state');(D/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True);f.close();jf.close();node.destroy_node();
 if rclpy.ok():rclpy.shutdown()
