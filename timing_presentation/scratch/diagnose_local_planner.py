import os
import sys
import numpy as np
import yaml
import json

# Add planning paths
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning')
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/src/unav_common')
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments')
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison')

from planning.planners.base_planner import extract_waypoints

# Load world profile and config
manifest_path = '/home/joostleliveld/Thesis/timing_presentation/runs/gazebo/hier_v1/experiment_20260527_163319/run_manifest.json'
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# Let's inspect the global plan and waypoints for both runs
runs = ['experiment_20260527_163022', 'experiment_20260527_163319']
for run in runs:
    print(f"\n=================== RUN: {run} ===================")
    plan_path = f'/home/joostleliveld/Thesis/timing_presentation/runs/gazebo/hier_v1/{run}/plan_samples.csv'
    if not os.path.exists(plan_path):
        print(f"Path does not exist: {plan_path}")
        continue
    
    import pandas as pd
    df = pd.read_csv(plan_path)
    stamps = df['plan_stamp'].unique()
    global_plan_stamp = stamps[0]
    
    global_df = df[df['plan_stamp'] == global_plan_stamp]
    global_pts = global_df[['x', 'y']].values
    
    # Extract waypoints
    spacing_m = manifest.get('waypoint_spacing_m', 1.0)
    wps = extract_waypoints(global_pts, spacing_m=spacing_m, include_goal=True)
    print(f"Global plan stamp: {global_plan_stamp}")
    print("Extracted waypoints:")
    for idx, wp in enumerate(wps):
        print(f"  wp_{idx}: {wp}")
        
    # Let's see the first local step inputs
    first_local_stamp = stamps[1]
    local_df = df[df['plan_stamp'] == first_local_stamp]
    local_pts = local_df[['x', 'y']].values
    print(f"First local plan stamp: {first_local_stamp}")
    print(f"First local planned states (length {len(local_pts)}):")
    for idx, pt in enumerate(local_pts[:5]):
        print(f"  pt_{idx}: {pt}")
    if len(local_pts) > 5:
        print(f"  ...")
        print(f"  pt_{len(local_pts)-1}: {local_pts[-1]}")
