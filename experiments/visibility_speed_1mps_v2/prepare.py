from paths import *
import json,subprocess,os
import numpy as np
from global_plan import solve
from speed_profile import validate_config,PROFILE
from speed_planner import planner,camera_models
from path_tracker import track
from unav_common.rectangular_footprint import constant_twist_pose

def main():
 rows=[]
 for cfg in json.loads((HERE/'pilot.json').read_text())['runs']:
  validate_config(cfg);D=T/('global_'+cfg['arm']);D.mkdir(exist_ok=True)
  assert not (D/'global_plan.json').exists(),'Preserve preparation attempts'
  g=solve(cfg,D)
  L=T/('local_'+cfg['arm']);L.mkdir(exist_ok=True);(L/'capture.json').write_text(json.dumps(cfg,indent=2))
  subprocess.run([sys.executable,str(HERE/'prelaunch_plan.py'),str(L)],check=True,timeout=120)
  b=json.loads((L/'prelaunch_plan.json').read_text());p=planner(camera_models(json.loads((R/'logs/perception_datasets/warehouse_v2_icra_p1_geometry_20260904/capture_manifest.json').read_text()))['camera_A']);body=p._footprint_collision_model
  pose=np.array([cfg['x'],cfg['y'],cfg['yaw']]);u,_=track(pose,b['states'],b['controls'],cfg['goal']);clear=body.sweep_clearance(pose,constant_twist_pose(pose,u,.1),control=u,dt=.1);margin=2.45*(.1+body.radius*np.deg2rad(5))
  P=np.array(g['forecast']['joint_covariance']);ok=bool(g['rollout_valid'] and b['rollout_valid'] and clear>margin and np.isfinite(P).all() and np.linalg.eigvalsh(P).min()>0 and np.max(np.array(g['controls'])[:,0])<=1+1e-12)
  row=dict(id=cfg['id'],passed=ok,global_horizon=g['horizon'],max_global_speed=float(np.max(np.array(g['controls'])[:,0])),initial_command=u.tolist(),initial_clearance_m=clear,guard_margin_m=margin,global_source=g['selected_source'],global_terminal_distance=g['terminal_goal_distance_pred']);rows.append(row);(T/'preflight.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
  assert ok,'Preflight gate failed'
if __name__=='__main__':main()
