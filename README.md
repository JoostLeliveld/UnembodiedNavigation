# UnembodiedNavigation

This repository is the active MSc thesis workspace for a controlled navigation study under a fixed external camera. The current milestone is not a general navigation framework. It is one research comparison:

1. `efe1`: expected free energy planning with a GP-based learned visibility / detection-success model
2. `visibility_unaware_baseline`: the same planning loop without the learned state-dependent visibility model

Both methods run under the same robot, simulator, external camera, state-estimation path, task definitions, and evaluation tooling.

## Research Question

Under one shared external-camera setup, does a planner that models state-dependent observation quality choose meaningfully different trajectories from a visibility-unaware baseline?

The repository is meant to answer that question at three levels:

- project story
- package responsibilities
- file-level execution flow

## Current Milestone

The current milestone is a controlled comparison, not the full thesis scope.

- **Shared across methods**
  - Gazebo world
  - TurtleBot3 robot model
  - fixed external camera
  - detector and pixel-to-BEV state estimator
  - goal/task definitions
  - planner horizon and control interface
  - run logging and summary scripts
- **Changed between methods**
  - whether the planner uses the learned GP visibility field inside the observation model

## Current Perception Status

The current perception story is intentionally narrow:

- a fixed external camera observes the robot
- the active runtime detector of interest is YOLO via `perception_backend:=yolo`
- the runtime detector publishes an image-space observation, not a full pose
- downstream state interpretation is:
  - `x,y` from detector pixel -> homography
  - `theta` from odometry fallback

The recommended perception workflow is:

1. try a local out-of-box YOLO segmentation model
2. if it is not good enough, fine-tune `yolo11n-seg` from simple offline pseudo-labels
3. use the resulting local `.pt` file at runtime

The active runtime path is:

`camera image -> YOLO -> mask-bottom or bbox-bottom pixel -> /perception/pixel_pose + diagnostics`

## Tutorial In One Page

The repository is easiest to understand through three figures:

![Empirical visibility artifact tutorial](docs/figures/visibility_capture_tutorial.png)

The learned artifact is a scalar field over the planner-facing state estimate. The current fitter supports two scalar targets over `/state/bev` x-y: a normalized blob-area baseline and a binary usable-detection label. For first-pass experiments, the script now defaults to the normalized blob-area target.

![Observation model tutorial](docs/figures/observation_model_tutorial.png)

Planner-side visibility enters through the observation model, not by modifying the robot dynamics. A simple first-pass target is

\[
\hat s_t = [\hat x_t,\hat y_t]^\top,\qquad
y_t = \mathrm{clip}\left(\frac{A_t}{A_{\mathrm{ref}}},0,1\right),
\]

to learn a scalar field

\[
p_{\mathrm{vis}}(\hat x,\hat y).
\]

Inside the planner, this becomes an effective planned observation covariance:

\[
p_{\mathrm{vis,eff}} = \mathrm{clip}(p_{\mathrm{vis}}^\gamma,\varepsilon,1-\varepsilon),
\qquad
R_{\mathrm{plan}} = p_{\mathrm{vis,eff}}R_{\mathrm{visible}} + (1-p_{\mathrm{vis,eff}})R_{\mathrm{miss}}.
\]

![Planner field story](docs/figures/planner_field_story.png)

This is the key method story: the GP does not command a path directly. It changes the expected quality of future observations, which changes risk and ambiguity, which changes route choice.

## What This Repository Does

- defines worlds, camera setup, and benchmark tasks
- fits and stores one empirical GP visibility artifact per supported world from noisy simulated pose sampling, with an optional retained driving-sweep mode
- launches the simulator, detector, state estimator, planner, goal node, and optional logger
- compares `efe1` and `visibility_unaware_baseline` under matched settings
- logs runs and generates milestone-grade summaries and qualitative plots

## What This Repository Does Not Yet Claim

- The baseline is **not** true dead reckoning. It is visibility-unaware planning under the same correction loop.
- The main image-based path is **not** fully camera-based pose estimation. In the primary runtime path, `x,y` come from camera projection while `theta` falls back to odometry.
- The GP is **not** a general occlusion model. It is a learned visibility / detection-success prior for the current simulated camera-detector stack.
- ET1 is the primary validated planning path for the current milestone. `efe2`, `efer`, and `mpc` remain as secondary modes on the same cleaned planner core, but they are not the main thesis claim.
- The current evaluator is useful for milestone inspection, but still thin for thesis-final claims.

## Repository Map

| Path | Role in the current milestone |
| --- | --- |
| [`src/experiments`](src/experiments/README.md) | Launches, world/task configs, logging, and experiment wiring |
| [`src/planning`](src/planning/README.md) | Planner node, planner core, visibility model loading, EFE objective path |
| [`src/perception`](src/perception/README.md) | External-camera detector and synthetic observation helpers |
| [`src/state`](src/state/README.md) | Pixel-to-BEV state conversion |
| [`src/sim`](src/sim/README.md) | Gazebo bringup, world files, robot description, startup gates |
| [`src/unav_common`](src/unav_common/README.md) | Shared camera and occlusion geometry helpers |
| [`scripts`](scripts/README.md) | Offline comparison capture, GP fitting, run summaries, and plotting |
| [`docs`](docs/README.md) | Canonical architecture, dataflow, comparison, planner, and limitation notes |

Generated directories such as `build/`, `install/`, `log/`, and `logs/` are not part of the conceptual repository story. Sibling root folders such as `core_tue4tm00_humble`, `ICRA2026-WUnEmbodied-main`, and `SPAICE2026-RxRover-main` are treated as archived/reference material, not as part of the active thesis repo.

## End-to-End Flow

```mermaid
flowchart LR
    A[world_profiles.yaml + tasks.yaml] --> B[warehouse_primary_comparison.launch.py]
    C[capture_visibility_samples.py] --> D[extract_perception_targets.py]
    D --> E[build_gp_targets.py]
    E --> F[fit_visibility_gps.py]
    F --> G[current_gp/*.npz]
    G --> H[planner core]
    B --> I[Gazebo + robot + external camera]
    I --> J[yolo_robot_detector_node]
    J --> K[pixel_to_bev_state_node]
    K --> H
    L[goal_mission_node] --> H
    H --> M[/cmd_vel]
    M --> I
    H --> N[experiment_logger]
    K --> N
    J --> N
    F --> O[plot_gp_and_ambiguity_maps.py]
    N --> P[plot_planned_paths.py]
    O --> Q[make_visibility_comparison_report.py]
    P --> Q
```

Caption: the offline comparison stage is separate from the planning runtime. The active workflow now uses teleport-sampled raw capture, shared scalar GP targets, planner-compatible GP artifacts, and consistent ambiguity/path plotting.

## State Estimator Reality Check

![State pipeline tutorial](docs/figures/state_pipeline_tutorial.png)

The current estimator used in the thesis-facing runtime is hybrid:

- `x,y` come from the external camera through the detector and homography
- `theta` comes from odometry fallback in the main image-detector path

That is good enough for the current milestone, but it should be stated honestly in every presentation.

## Quick Start

Build the active packages:

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
colcon build --packages-select unav_common perception state planning experiments sim
source install/setup.bash
```

Set a writable ROS log directory before launching:

```bash
export ROS_LOG_DIR=/tmp/roslog_thesis
mkdir -p "$ROS_LOG_DIR"
```

## Example Launches

### Offline visibility comparison backbone

Start the simulator:

```bash
ros2 launch sim bringup_sim.launch.py \
  world:=warehouse_occ_light.world.sdf \
  reset_world:=false \
  use_lidar:=false \
  bridge_scan:=false \
  spawn_x:=-1.55 \
  spawn_y:=0.45 \
  spawn_z:=0.05 \
  spawn_yaw:=0.0
```

In a second terminal, build the shared comparison artifacts:

```bash
source install/setup.bash
python3 scripts/visibility_comparison/capture_visibility_samples.py \
  --world warehouse_occ_light.world.sdf \
  --sample-nx 15 \
  --sample-ny 15

python3 scripts/visibility_comparison/extract_perception_targets.py
python3 scripts/visibility_comparison/build_gp_targets.py
python3 scripts/visibility_comparison/fit_visibility_gps.py
python3 scripts/visibility_comparison/plot_gp_and_ambiguity_maps.py
```

At the shared-backbone stage, `oracle_visibility` is the first fully populated method. Red and YOLO target columns are added in later method-specific passes without changing the capture, GP, planner, or report contracts.

For the full comparison design, see:

- [`docs/perception_to_visibility_comparison.md`](docs/perception_to_visibility_comparison.md)
- [`docs/perception_visibility_cleanup_plan.md`](docs/perception_visibility_cleanup_plan.md)

### Primary thesis comparison

Run the GP-aware EFE method on the main world:

```bash
ros2 launch experiments warehouse_primary_comparison.launch.py \
  planner:=efe1 \
  use_rviz:=false
```

Run the visibility-unaware baseline:

```bash
ros2 launch experiments warehouse_primary_comparison.launch.py \
  planner:=visibility_unaware_baseline \
  use_rviz:=false
```

Launch without the logger for a cleaner `rqt_graph`:

```bash
ros2 launch experiments warehouse_primary_comparison.launch.py \
  planner:=efe1 \
  enable_logging:=false \
  use_rviz:=false
```

Inspect the current launch surface:

```bash
ros2 launch experiments warehouse_primary_comparison.launch.py --show-args
```

### Change world or task

Run the main method on the secondary support world:

```bash
ros2 launch experiments warehouse_primary_comparison.launch.py \
  planner:=efe1 \
  world:=warehouse_open_shelves.world.sdf \
  use_rviz:=false
```

Run a specific named task instead of the profile default:

```bash
ros2 launch experiments warehouse_primary_comparison.launch.py \
  planner:=efe1 \
  task:=E5_shadow_tradeoff \
  use_rviz:=false
```

### Retained secondary planners

Use the broader retained-planner launch when you want `efe2`, `efer`, or `mpc`:

```bash
ros2 launch experiments warehouse_visibility_agent.launch.py \
  planner:=efe2 \
  use_rviz:=false
```

```bash
ros2 launch experiments warehouse_visibility_agent.launch.py \
  planner:=efer \
  use_rviz:=false
```

```bash
ros2 launch experiments warehouse_visibility_agent.launch.py \
  planner:=mpc \
  use_rviz:=false
```

Show the full retained-planner launch surface:

```bash
ros2 launch experiments warehouse_visibility_agent.launch.py --show-args
```

### Runtime graph

After launching with `enable_logging:=false`, inspect the online node/topic graph:

```bash
ros2 run rqt_graph rqt_graph
```

## Current Status

- **Primary comparison path**: `efe1` vs `visibility_unaware_baseline`
- **Primary world**: `warehouse_occ_light.world.sdf`
- **Secondary support world**: `warehouse_open_shelves.world.sdf`
- **Primary runtime estimator**: camera-derived `x,y` plus odometry-backed `theta`
- **Primary planner path**: ET1
- **Retained but secondary planners**: `efe2`, `efer`, `mpc`

## How To Read This Repository

### Pass 1: Project story

1. Read this README.
2. Read [`docs/architecture_overview.md`](docs/architecture_overview.md).
3. Read [`docs/runtime_dataflow.md`](docs/runtime_dataflow.md).

### Pass 2: Package responsibilities

1. Read [`src/experiments/README.md`](src/experiments/README.md).
2. Read [`src/planning/README.md`](src/planning/README.md).
3. Read [`src/perception/README.md`](src/perception/README.md) and [`src/state/README.md`](src/state/README.md).

### Pass 3: File-level execution flow

1. [`src/experiments/launch/warehouse_primary_comparison.launch.py`](src/experiments/launch/warehouse_primary_comparison.launch.py)
2. [`src/experiments/config/world_profiles.yaml`](src/experiments/config/world_profiles.yaml)
3. [`src/experiments/config/tasks.yaml`](src/experiments/config/tasks.yaml)
4. [`src/perception/perception/nodes/yolo_robot_detector_node.py`](src/perception/perception/nodes/yolo_robot_detector_node.py)
5. [`src/state/state/nodes/pixel_to_bev_state_node.py`](src/state/state/nodes/pixel_to_bev_state_node.py)
6. [`src/planning/planning/nodes/unicycle_planner_node.py`](src/planning/planning/nodes/unicycle_planner_node.py)
7. [`src/planning/planning/planners/base_planner.py`](src/planning/planning/planners/base_planner.py)
8. [`src/planning/planning/core/visibility_gp_map.py`](src/planning/planning/core/visibility_gp_map.py)
9. [`src/experiments/experiments/nodes/experiment_logger.py`](src/experiments/experiments/nodes/experiment_logger.py)

## Documentation Map

- [`docs/architecture_overview.md`](docs/architecture_overview.md): package architecture and minimum reading path
- [`docs/runtime_dataflow.md`](docs/runtime_dataflow.md): offline preparation and runtime ROS dataflow
- [`docs/planner_method.md`](docs/planner_method.md): planner-internal method and comparison logic
- [`docs/evaluation_and_plots.md`](docs/evaluation_and_plots.md): outputs, summaries, plots, and presentation figures
- [`docs/limitations.md`](docs/limitations.md): current caveats and claim guardrails
- [`docs/figures/README.md`](docs/figures/README.md): figure catalog and regeneration script
