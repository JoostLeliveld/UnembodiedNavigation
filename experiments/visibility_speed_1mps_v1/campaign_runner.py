import os,sys,time,json,subprocess,signal,threading,csv
from pathlib import Path
import psutil
from paths import *
H=HERE
from cleanup_owned import cleanup,owned_pids
from speed_profile import validate_config,profile_hash
import hashlib,shutil
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=json.loads((H/'pilot.json').read_text())
pre=json.loads((T/'preflight.json').read_text());assert len(pre)==3 and all(x['passed'] for x in pre)
base=json.loads((O/'visibility_pilot_v1/pilot_sources.json').read_text())
assert all(digest(p)==h for p,h in base.items()),'Original source drift; review before acquisition'
files=list(base)+[str(p) for p in H.iterdir() if p.suffix in ['.py','.json']]+[str(T/('global_'+c['arm'])/'global_plan.json') for c in selection['runs']]
frozen={p:digest(p) for p in files}
(T/'source_freeze.json').write_text(json.dumps(frozen,indent=2))
for cfg in selection['runs']:
 name=cfg['arm'];D=O/'navigation'/cfg['id'];validate_config(cfg)
 if sys.argv[1:] and name not in sys.argv[1:]:continue
 assert all(digest(p)==h for p,h in frozen.items()),'Source drift'
 assert not D.exists(),'Preserve previous attempt'
 D.mkdir();(D/'capture.json').write_text(json.dumps(cfg,indent=2));(D/'source_freeze.json').write_text(json.dumps(frozen,indent=2))
 snap=D/'source_snapshot';snap.mkdir()
 for path in files:
  p=Path(path)
  if p.suffix=='.pt':continue
  target=snap/('workspace/'+str(p.relative_to(O)) if p.is_relative_to(O) else 'repository/'+str(p.relative_to(R)))
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 shutil.copy2(O/'collision_geometry.json',snap/'collision_geometry.json')
 for n in ['global_plan.json','prelaunch_plan.json']:shutil.copy2(T/('local_'+name)/n,D/n)
 (D/'robot_deployment.json').write_text(json.dumps({'sha256':digest(R/'install/sim/share/sim/robot_description/urdf/warehouse_amr.urdf.xacro')}))
 env=os.environ.copy();env.update(CAPTURE_SWEEP_DIR=str(D),ROS_DOMAIN_ID='218',IGN_PARTITION='speed1_01a0805b_'+name,GZ_PARTITION='speed1_01a0805b_'+name,ROS_LOG_DIR=str(D/'roslogs'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',UNAV_CASADI_CACHE_DIR=str(H/'casadi_cache'))
 stop=threading.Event()
 def sample():
  cache={}
  with (D/'resources.jsonl').open('w') as f:
   while not stop.is_set():
    processes=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
     try:
      if p.environ().get('CAPTURE_SWEEP_DIR')!=str(D):continue
      q=cache.setdefault(p.pid,p);processes.append(dict(pid=p.pid,name=p.name(),cmd=' '.join(p.cmdline()),cpu_pct=q.cpu_percent(),rss_mb=p.memory_info().rss/2**20))
     except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    f.write(json.dumps(dict(monotonic_s=time.monotonic(),host_cpu_pct=psutil.cpu_percent(),gpu=gpu.stdout.strip(),processes=processes))+'\n');f.flush();stop.wait(2)
 t=threading.Thread(target=sample);t.start();begin=time.monotonic();reason='supervisor_wall_timeout';procs=[]
 try:
  with (D/'launch.log').open('w') as log,(D/'performance.log').open('w') as perf:
   procs=[subprocess.Popen(['python3',str(H/'performance.py')],env=env,stdout=perf,stderr=subprocess.STDOUT,start_new_session=True),subprocess.Popen(['ros2','launch',str(H/'sweep.launch.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)]
   print('START',name,flush=True)
   while time.monotonic()-begin<780:
    if (D/'summary.json').exists():reason='driver_finished';break
    if procs[1].poll() is not None:reason='launch_exited';break
    time.sleep(.5)
   detected=time.monotonic()
   for p in procs:
    try:os.killpg(p.pid,signal.SIGINT)
    except ProcessLookupError:pass
   if True:
    deadline=time.monotonic()+5
    while any(p.poll() is None for p in procs) and time.monotonic()<deadline:time.sleep(.02)
   else:time.sleep(5)
   for p in procs:
    if p.poll() is None:
     try:os.killpg(p.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:p.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   leftovers=cleanup(D,exclude=(os.getpid(),os.getppid())) if owned_pids(D) else []
 finally:stop.set();t.join()
 (D/'supervisor.json').write_text(json.dumps(dict(reason=reason,wall_s=time.monotonic()-begin,begin_monotonic=begin,summary_detected_monotonic=detected,cleanup_pids=leftovers,exits=[p.returncode for p in procs]),indent=2))
 print('FINISH',name,(D/'summary.json').read_text()[:150] if (D/'summary.json').exists() else reason,flush=True)
 (T/'campaign_state.json').write_text(json.dumps({'last_arm':name,'last_reason':reason,'last_directory':str(D)},indent=2))
 if reason!='driver_finished':break
