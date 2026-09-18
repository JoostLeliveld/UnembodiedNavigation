from pathlib import Path
import os,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from speed_profile import V_MAX,PROFILE,validate_config
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription,ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
O=Path(__file__).resolve().parent;R=O.parents[1]
D=Path(os.environ.get('CAPTURE_SWEEP_DIR',str(O/'sweep')))
def generate_launch_description():
 cfg=json.loads((D/"capture.json").read_text());validate_config(cfg)
 sim=IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(get_package_share_directory('sim'))/'launch/bringup_sim.launch.py')),launch_arguments={'world':'warehouse_v2.world.sdf','world_name':'warehouse_v2','headless':'true','reset_world':'false','use_lidar':'false','bridge_scan':'false','bridge_camera_a':'true','bridge_camera_b':'true','bridge_camera_c':'true','bridge_camera_d':'true','bridge_camera_e':'true','bridge_contacts':'true','spawn_x':str(cfg['x']),'spawn_y':str(cfg['y']),'spawn_z':'0.01','spawn_yaw':str(cfg['yaw'])}.items())
 detector=Node(package='perception',executable='batched_four_camera_yolo_node',parameters=[{'use_sim_time':True,'calibration_world':'warehouse_v2','model_path':str(R/'logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt'),'device':'0','image_size':960,'confidence_threshold':.25,'iou_threshold':.45,'class_name':'robot','class_id':0,'use_masks':False,'min_bbox_area_px':0.,'max_batch_stamp_skew_s':.05,'max_pending_wall_s':.5,'synchronization_mode':'strict','debug_crop_dir':str(D/'crops'),'outcome_journal_path':str(D/'detector_outcomes.jsonl')}],output='log')
 encoder=Node(package='sim',executable='encoder_noise_node',parameters=[{'use_sim_time':True,'enabled':True,'seed':cfg['seed']}],output='log')
 actuator=Node(package='sim',executable='actuation_noise_node',parameters=[{'use_sim_time':True,'enabled':True,'seed':cfg['seed'],'linear_min':PROFILE['actuator_linear_min_m_s'],'linear_max':V_MAX}],output='log')
 driver=ExecuteProcess(cmd=['python3',str(O/('rotation_driver.py' if cfg.get('rotation_probe') else ('navigation_driver.py' if cfg.get('navigation') else 'sweep_driver.py')))],output='screen')
 return LaunchDescription([sim,encoder,actuator,detector,driver])
