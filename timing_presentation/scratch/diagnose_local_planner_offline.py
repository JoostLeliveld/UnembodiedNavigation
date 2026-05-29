import os
import sys
import numpy as np
import json
import time

# Add planning paths
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning')
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/src/unav_common')
sys.path.insert(0, '/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments')

from planning.planners.base_planner import UnicyclePlannerBase
from experiments.core.world_profiles import load_world_profiles, serialize_driveable_geometry_from_profile

# Load profiles
profiles = load_world_profiles('/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments/config/world_profiles.yaml')
profile = profiles['worlds']['warehouse_aws.world.sdf']
driveable_geom = serialize_driveable_geometry_from_profile(profile)

# We want to create the local planner parameters manually from run_manifest.json
manifest_path = '/home/joostleliveld/Thesis/timing_presentation/runs/gazebo/hier_v1/experiment_20260527_163319/run_manifest.json'
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# Load GP to get camera parameters
gp = np.load(manifest['visibility_artifact_path'])
camera_params = {
    "cam_pos": tuple(gp["camera_pos"][:3].tolist()),
    "look_at": tuple(gp["look_at"][:3].tolist()),
    "img_width": int(gp["img_width"]),
    "img_height": int(gp["img_height"]),
    "fov_h_rad": float(gp["fov_h_rad"]),
}

# Initial state from second run at stamp 9.851
m0 = np.array([3.366847, -1.058317, -4.8917e-12])
S0 = np.eye(3) * 0.01  # small covariance

# Target waypoint
target = np.array([3.52004, 0.16132])

def test_planner(penalty_type, weight, safe_dist=0.25, use_vis=True):
    print(f"\n--- Testing Planner with use_visibility_model={use_vis} ---")
    planner = UnicyclePlannerBase(
        horizon=12,
        dt=0.25,
        v_min=0.0,
        v_max=1.5,
        w_min=-1.0,
        w_max=1.0,
        control_weight=0.0,
        process_noise_xy=0.01,
        process_noise_theta=0.02,
        obs_noise_uv=2.0,
        goal_sigma_uv=2.0,
        risk_weight_obs=1.0,
        ambiguity_weight=6.0,
        optimizer_maxiter=60,
        optimizer_maxfun=900,
        optimizer_ftol=1e-6,
        optimizer_gtol=1e-4,
        optimizer_warm_start=True,
        optimizer_multistart=True,
        optimizer_multistart_lateral_offsets='',
        approx_method='ET1',
        use_obs_risk=True,
        use_ambiguity=False,
        seed=0,
        camera_params=camera_params,
        use_visibility_model=use_vis,
        visibility_artifact_path=manifest['visibility_artifact_path'],
        collision_geometry_json=manifest['collision_geometry_json'],
        r_visible_uv=2.5,
        r_miss_uv=15.0,
        goal_prior_u_std_start=20.0,
        goal_prior_v_std_start=20.0,
        goal_prior_u_std_final=20.0,
        goal_prior_v_std_final=20.0,
        goal_tightening_power=0.45,
        goal_progress_n_steps=12,
        use_nogo_cost=True,
        nogo_penalty_type=penalty_type,
        nogo_weight=weight,
        nogo_safe_distance=safe_dist,
        nogo_logbarrier_scale=0.25,
        nogo_logbarrier_eps=0.01,
        use_belief_nogo_cost=False,
        nogo_mode='keep_in',
        driveable_geometry_json=driveable_geom,
        robot_collision_radius_m=0.125,
        runtime_debug=True,
    )
    
    # Warmup / compile
    planner.plan(m0, S0, target)
    
    # Plan and time
    t0 = time.perf_counter()
    res = planner.plan(m0, S0, target)
    t_elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"Success: {res.optimizer_success}, Message: {res.optimizer_message}")
    print(f"Total cost: {res.total_cost:.3f}, Valid: {res.rollout_valid}, Solve time: {t_elapsed:.1f} ms")

test_planner('softplus', 600.0, use_vis=False)
test_planner('softplus', 600.0, use_vis=True)
