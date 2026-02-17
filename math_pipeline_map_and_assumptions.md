# Math Pipeline Map and Assumption Audit
Updated: 2026-02-12

## 1) Purpose
This document maps:
1. the mathematics currently implemented in the ROS stack,
2. how those math blocks connect end-to-end,
3. which assumptions are currently hard-coded,
4. which assumptions should be retired for stronger science and future modularity.

This is an implementation-truth document, not an intent document.

## 2) State, Observation, and Coordinate Spaces

### 2.1 State spaces
1. Planner state (SE2):
   - `s = [x, y, theta]`
   - Code: `src/planning/planning/core/dynamics.py` (`unicycle_step`)
2. Belief in planner/state node:
   - Gaussian belief over state:
   - `q(s) = N(m, S)`
3. A* state:
   - grid index + world path conversion
   - Code: `src/planning/planning/core/search_based_path_planning.py`

### 2.2 Observation spaces
1. Pixel-only observation:
   - `o = [u, v]` (`obs_mode=uv`)
2. Pixel + heading observation:
   - `o = [u, v, theta]` (`obs_mode=uvt`)
3. Observation function:
   - `o = g(s)` from homography/pinhole mapping
   - Code: `src/unav_common/unav_common/camera_model.py` (`g_uv`, `g`)

### 2.3 Frames and mappings
1. World/BEV frame: `map_bev` (planner and state contract)
2. Odometry frame: `odom` (sim truth stream)
3. Image frame: pixel plane (`/perception/pixel_pose` stores `u,v` in position x,y)

## 3) Camera and Homography Math

### 3.1 Intrinsics
Implemented:
- `f = (W/2)/tan(fov_h/2)`
- `K = [[f,0,cx],[0,f,cy],[0,0,1]]`
- `cx=W/2, cy=H/2`
Code:
- `src/unav_common/unav_common/camera_model.py` (`_compute_intrinsics`)

### 3.2 Extrinsics and homography
Implemented:
1. camera position `C = cam_pos`
2. look-at rotation `R` from `(cam_pos, look_at, up_hint)`
3. translation `t = -R C`
4. planar homography:
   - `H = K [r1 r2 t]`
   - `H_inv = H^{-1}`
Code:
- `src/unav_common/unav_common/camera_model.py`

### 3.3 Forward and inverse mapping
1. World -> pixel (ground-plane):
   - `[u~,v~,w~]^T = H [x,y,1]^T`
   - `u=u~/w~, v=v~/w~`
2. Pixel -> world:
   - `[x~,y~,w~]^T = H_inv [u,v,1]^T`
   - `x=x~/w~, y=y~/w~`
Code:
- `src/unav_common/unav_common/camera_model.py` (`world_to_pixel`, `pixel_to_world`)

### 3.4 Visibility logic
A point is marked visible if:
1. pixel bounds check passes (`0<=u<W`, `0<=v<H`), and
2. camera depth check `cam_pt[2] > 0`.
Code:
- `src/unav_common/unav_common/camera_model.py`

## 4) Perception Math

### 4.1 Homography sim node
Input:
- `/odom` truth pose
Operation:
1. read true `(x,y,theta)` from odometry
2. project `(x,y)` to `(u,v)` with homography model
3. publish `/perception/pixel_pose` with:
   - `position.x=u`, `position.y=v`
   - orientation copied directly from odom orientation
Code:
- `src/perception/perception/nodes/homography_sim_node.py`
- `src/perception/perception/core/homography.py`

Mathematical implication:
- in simulation, pixel measurement and yaw are tightly coupled to truth; yaw is not inferred from image.

### 4.2 ArUco detector node (new backend)
Input:
- `/external_camera/image_raw`

Operation:
1. detect marker corners with OpenCV ArUco dictionary detector
2. select marker by `target_marker_id` (or largest marker if id < 0)
3. compute pixel center `(u,v)` from 4 marker corners
4. optional yaw estimate from top-edge corners via inverse homography
5. publish `/perception/pixel_pose` (`PoseStamped`) with:
   - `position.x=u`, `position.y=v`
   - orientation from marker yaw if enabled

Fallback (currently implemented):
1. if ArUco dictionary detection fails, optional template fallback runs
2. fallback uses contour proposals + perspective warp + binary template matching
3. best candidate above threshold is emitted as pixel observation

Code:
- `src/perception/perception/nodes/aruco_detector_node.py`

Mathematical implication:
- primary detector is discrete marker-ID matching; not a continuous observation model
- template fallback is deterministic pattern matching, not a calibrated probabilistic sensor model
- this backend currently provides pixel observations but not a principled observation covariance output

## 5) State Estimation Math

### 5.1 ORACLE mode
`/state/bev` is copied from `/odom`, covariance set to tiny diagonal.
Code:
- `src/state/state/nodes/pixel_to_bev_state_node.py` (`_odom_callback`)
- `src/state/state/core/noise.py`

### 5.2 PIXEL mode
Input:
- `/perception/pixel_pose`
Operation:
1. optional pixel noise on `(u,v)`
2. optional transform noise on camera pose/look-at
3. inverse homography `pixel_to_world(u,v) -> (x,y)`
4. yaw copied from incoming message orientation
5. covariance built as diagonal from local finite-difference metric sensitivities.
Code:
- `src/state/state/nodes/pixel_to_bev_state_node.py`
- `src/state/state/core/pixel_to_bev.py`
- `src/state/state/core/noise.py`

Metric-noise mapping used:
1. evaluate world point at `(u,v)`
2. evaluate at `(u+sigma_pix, v)` and `(u, v+sigma_pix)`
3. convert displacement magnitudes to `sigma_x`, `sigma_y`.

## 6) Planning Math (Unicycle EFE/MPC-like)

### 6.1 Dynamics
Unicycle model:
1. `x_{k+1} = x_k + v_k dt cos(theta_k)`
2. `y_{k+1} = y_k + v_k dt sin(theta_k)`
3. `theta_{k+1} = wrap(theta_k + w_k dt)`
Code:
- `src/planning/planning/core/dynamics.py`

Jacobian:
- `F = d f / d s`
Code:
- `src/planning/planning/core/dynamics.py` (`unicycle_jacobian`)

Process noise:
- diagonal `Q` scaled by `dt/base_dt`.
Code:
- `src/planning/planning/core/dynamics.py` (`unicycle_process_noise`)

### 6.2 Observation transforms
Given belief `(m,S)` and observation map `g`:
1. ET1:
   - `mu_y = g(m)`
   - `Sigma_y = J S J^T + R`
   - `Gamma = S J^T`
2. ET2:
   - ET1 terms + Hessian trace corrections
3. UT:
   - sigma-point transform
Code:
- `src/planning/planning/core/efe_utils.py` (`ET1`, `ET2`, `UT`)

JAX equivalents:
- `src/planning/planning/core/jax_efe.py`

### 6.3 Risk term
Implemented risk:
- Gaussian KL:
  - `risk = KL(N(mu,Sigma) || N(mu_goal,S_goal))`
Code:
- `src/planning/planning/core/efe_utils.py` (`risk`)

Risk is used in:
1. state space (`m,S` vs goal state Gaussian)
2. observation space (`mu_y,Sigma_y` vs goal observation Gaussian)
Combined with weights:
- `w_state * risk_state + w_obs * risk_obs`
Code:
- `src/planning/planning/planners/base_planner.py` (`_evaluate_controls`)

### 6.4 Ambiguity term
Implemented ambiguity:
1. `Sigma_cond = Sigma_y - Gamma^T S^{-1} Gamma`
2. `ambiguity = 0.5 * (d log(2*pi*e) + logdet(Sigma_cond))`
Code:
- `src/planning/planning/core/efe_utils.py` (`ambiguity`)

Interpretation:
- this is a Gaussian conditional-entropy-like quantity under chosen approximation.
- it is not explicit mutual information implementation.

### 6.5 Objective assembled in planner
Per rollout:
1. propagate belief through dynamics
2. compute transformed observation stats (`ET1/ET2/UT`)
3. accumulate:
   - weighted risk (state + obs)
   - weighted ambiguity (if enabled)
   - control regularization `w_u*(v^2+w^2)`
   - boundary/cost penalty from costmap

Boundary behavior:
1. if out-of-bounds or lethal, add infeasible penalty `1e6`
2. otherwise add normalized cell cost contribution

Code:
- `src/planning/planning/planners/base_planner.py` (`_evaluate_controls`)

### 6.6 Optimization backend
1. SciPy L-BFGS-B (cost-only)
2. JAX value+grad + SciPy L-BFGS-B (when enabled)
3. fallback random-sampling candidates if optimize fails
Code:
- `src/planning/planning/planners/base_planner.py` (`plan`)

JAX bypass conditions:
1. boundary weight > 0
2. unsupported approx mode
3. JAX unavailable

## 7) Planner-side Pixel Correction Math

When enabled (`use_pixel_correction=true`), unicycle planner node performs a measurement update:
1. prediction:
   - `(m_pred,S_pred) = f(m,S,u_last)`
2. observation prediction:
   - `(mu_y,Sigma_y,Gamma) = approx_observation(m_pred,S_pred)`
3. innovation:
   - `innov = y_meas - mu_y` (yaw wrapped only when present)
4. Kalman-like gain:
   - `K = Gamma pinv(Sigma_y)`
5. update:
   - `m = m_pred + K innov`
   - `S = S_pred - K Sigma_y K^T`
Code:
- `src/planning/planning/nodes/unicycle_planner_node.py` (`_pixel_cb`)

Important:
- measurement dimension follows `obs_mode` after recent fix:
  - `uv` -> `[u,v]`
  - `uvt` -> `[u,v,yaw]`

## 8) A* Planning Math

1. occupancy/cost grid from `/costmap`
2. world-to-grid transform via floor((x-origin)/resolution)
3. graph shortest path (Dijkstra in NetworkX) on traversable cells
4. optional diagonal connectivity
5. path converted back to world cell centers
Code:
- `src/planning/planning/core/search_based_path_planning.py`
- `src/planning/planning/planners/astar_planner.py`

## 9) Control Math (Pure Pursuit)

Given path and robot pose:
1. choose lookahead point on path within sphere/corridor
2. compute heading error:
   - `yaw_error = atan2(dy,dx) - yaw`
3. angular velocity:
   - `w = clamp(kp * yaw_error, +/- max_angular)`
4. linear velocity:
   - zero for large heading error
   - otherwise scaled by `cos(yaw_error)` and goal slowdown
Code:
- `src/control/control/core/path_follow.py`
- `src/control/control/core/pure_pursuit.py`

## 10) Experiment-layer Math and Mappings

### 10.1 Costmap construction
1. rectangular boundaries converted to occupancy grid
2. wall margin marked lethal
3. optional circular obstacle inserted
Code:
- `src/experiments/experiments/nodes/boundary_cost_node.py`

### 10.2 Goal mission
1. fixed goal `(goal_x,goal_y)` published with repeats/delay
2. uses system-time clock
Code:
- `src/experiments/experiments/nodes/goal_mission_node.py`

### 10.3 Metrics logging
Logs:
1. state, covariance, cmd, goal distance, plan length
2. EFE metrics vector `[total,risk,ambiguity,control,boundary]`
Code:
- `src/experiments/experiments/nodes/experiment_logger.py`

## 11) End-to-end Mathematical Dataflow

### 11.1 Planner + controller pipeline
1. `/odom` -> `homography_sim_node` -> `/perception/pixel_pose` (if pixel mode)
2. `/perception/pixel_pose` or `/odom` -> `pixel_to_bev_state_node` -> `/state/bev`
3. `/state/bev` + `/goal_bev` + `/costmap` -> planner -> `/plan` + `/efe/metrics`
4. `/state/bev` + `/plan` -> pure pursuit -> `/cmd_vel`

### 11.2 Agent pipeline
1. same state path until planner input
2. `efe_agent` outputs `/cmd_vel` directly (also `/plan`, `/efe/metrics`)

## 12) Assumptions Currently Made

### 12.1 Assumptions that are acceptable for this stage (keep explicitly)
1. Planar ground (`Z=0`) homography for mapping.
2. Fixed camera intrinsics per experiment block.
3. Gaussian belief approximation.
4. Static costmap during run.

### 12.2 Assumptions that should be let go next (high priority)
1. Yaw leakage through pixel path:
   - current sim pipeline copies true yaw into pixel message orientation.
   - consequence: pixel mode still has near-truth heading.
   - where: `src/perception/perception/nodes/homography_sim_node.py`
2. Oracle shortcut baseline mixed with pixel-correction flags:
   - `state_source=oracle` with pixel-correction can create stale/invalid behavior.
3. Single-camera hard assumption:
   - no camera id, no asynchronous fusion, no per-camera covariance.
4. Ad-hoc metric covariance mapping:
   - finite-difference from `(u,v)` offsets gives local scalar sigmas but not full Jacobian covariance propagation.
5. Detector fallback not probabilistic:
   - current ArUco template fallback emits detections without calibrated confidence->covariance mapping.
   - consequence: state covariance does not reflect detector ambiguity in a principled way.

### 12.3 Assumptions to let go in medium term
1. Numerical derivatives (finite diff ET1/ET2) as primary path:
   - sensitive to `eps`, can be noisy near singular geometry.
2. Boundary penalty discontinuity:
   - hard infeasible jump can dominate objective and mask uncertainty terms.
3. Goal publication on system time:
   - should align with sim time for deterministic replay.
4. Manual CLI experiment orchestration:
   - should rely on named presets + frozen manifests (partially addressed).

### 12.4 Assumptions to let go for broader future scope
1. Homography sim as stand-in for real vision forever.
2. No explicit information-gain term but epistemic claims in text.
3. Planner-coupled filtering logic inside node callback without estimator abstraction.

## 13) Mathematical Consistency Notes (Critical)

1. EFE objective in code is currently:
   - weighted risk + weighted ambiguity + control + boundary
2. "MPC-like" in this repo should be defined as:
   - same stack with ambiguity disabled
3. ET1 vs ET2 claim is about approximation behavior under nonlinear projection; not automatically about explicit mutual information seeking.
4. `obs_mode=uv` and `obs_mode=uvt` are different measurement models and must not be mixed in one comparison batch.

## 14) Immediate Checks Before Claiming Results

1. Confirm same optimizer settings across methods.
2. Confirm same `obs_mode`, `state_source`, `boundary_weight`, `seed` policy.
3. Confirm no stale pixel warnings in logs for compared runs.
4. Confirm camera usable region constraints for all tasks.
5. Confirm all plotted metrics correspond to logged run ids/manifests.

## 15) Minimal Refactor Targets to Improve Mathematical Integrity

1. Extract planner-side measurement update into dedicated estimator module.
2. Replace yaw copy in homography sim with measurement-model-consistent alternative for pixel-only mode.
3. Add full Jacobian-based covariance mapping in state node (not only scalar sigma estimates).
4. Add camera id and multi-camera observation message contract.
5. Separate "objective definition" from "optimizer backend" in experiment manifests for cleaner ablation claims.
