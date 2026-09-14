import time,csv,signal,os
from pathlib import Path
import psutil,rclpy
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data
rclpy.init();node=rclpy.create_node('capture_performance_monitor');sim=None;stop=False
def callback(msg):
 global sim
 sim=msg.clock.sec+msg.clock.nanosec*1e-9
node.create_subscription(Clock,'/clock',callback,qos_profile_sensor_data)
def end(*args):
 global stop
 stop=True
signal.signal(signal.SIGTERM,end);signal.signal(signal.SIGINT,end)
out=Path(os.environ['CAPTURE_SWEEP_DIR'])/'performance.csv';lastwall=time.monotonic();lastsim=None;cache={};start=lastwall
with out.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['wall_elapsed_s','sim_s','rtf','host_cpu_pct','gazebo_cpu_core_pct','gazebo_rss_mb','detector_cpu_core_pct','gazebo_pids']);w.writeheader()
 while not stop and time.monotonic()-start<1500:
  rclpy.spin_once(node,timeout_sec=.1);now=time.monotonic()
  if now-lastwall<2:continue
  gz=det=mem=0.;pids=[]
  for proc in psutil.process_iter(['pid','name','cmdline']):
   try:
    name=proc.info['name'];cmd=' '.join(proc.info['cmdline'] or [])
    kind='gz' if name in ['gzserver','gzclient'] or ('ign gazebo' in cmd and name not in ['bash','timeout']) else ('det' if 'batched_four_camera_yolo' in cmd and name not in ['bash','timeout'] else None)
    if kind:
     if os.environ.get('IGN_PARTITION') and proc.environ().get('IGN_PARTITION')!=os.environ['IGN_PARTITION']:continue
     p=cache.setdefault(proc.pid,proc);cpu=p.cpu_percent()
     if kind=='gz':gz+=cpu;mem+=p.memory_info().rss/2**20;pids.append(proc.pid)
     else:det+=cpu
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  rtf=(sim-lastsim)/(now-lastwall) if sim is not None and lastsim is not None and sim>=lastsim else None
  w.writerow(dict(wall_elapsed_s=now-start,sim_s=sim,rtf=rtf,host_cpu_pct=psutil.cpu_percent(),gazebo_cpu_core_pct=gz,gazebo_rss_mb=mem,detector_cpu_core_pct=det,gazebo_pids=' '.join(map(str,pids))));f.flush();lastwall=now;lastsim=sim
node.destroy_node();rclpy.shutdown()
