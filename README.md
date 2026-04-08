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

## What This Repository Does

- defines worlds, camera setup, and benchmark tasks
- fits and stores one empirical GP visibility artifact per supported world
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
| [`scripts`](scripts/README.md) | Offline GP fitting, run summaries, and plotting |
| [`docs`](docs/README.md) | Canonical architecture, dataflow, planner, evaluation, and limitation notes |

Generated directories such as `build/`, `install/`, `log/`, and `logs/` are not part of the conceptual repository story. Sibling root folders such as `core_tue4tm00_humble`, `ICRA2026-WUnEmbodied-main`, and `SPAICE2026-RxRover-main` are treated as archived/reference material, not as part of the active thesis repo.

## End-to-End Flow

```mermaid
flowchart LR
    A[world_profiles.yaml + tasks.yaml] --> B[warehouse_primary_comparison.launch.py]
    C[fit_empirical_visibility_gp.py] --> D[empirical_visibility_gp.npz]
    D --> E[planner core]
    B --> F[Gazebo + robot + external camera]
    F --> G[image_marker_detector_node]
    G --> H[pixel_to_bev_state_node]
    H --> E
    I[goal_mission_node] --> E
    E --> J[/cmd_vel]
    J --> F
    E --> K[experiment_logger]
    H --> K
    G --> K
    K --> L[evaluate_occlusion_comparison.py]
    K --> M[plot_visibility_run.py]
```

Caption: the offline GP fitting stage is separate from the ROS runtime. Online, the planner receives a fixed visibility artifact rather than learning during execution.

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

Run the main method on the secondary exploratory world:

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
- **Secondary exploratory world**: `warehouse_open_shelves.world.sdf`
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
4. [`src/perception/perception/nodes/image_marker_detector_node.py`](src/perception/perception/nodes/image_marker_detector_node.py)
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
