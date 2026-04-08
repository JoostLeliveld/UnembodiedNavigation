# `sim`

This package provides the plant and simulator plumbing for the thesis experiments.

![Example top-down run in the simulated world](../../docs/figures/planner_field_story.png)

This package exists so that the planner and detector run inside one fixed, inspectable external-camera world.

## Why This Folder Exists

The repository is built around a fixed Gazebo external-camera setup. This package defines:

- the world files
- the robot description
- the external camera model
- startup helpers for the runtime

## Inputs And Outputs

- **Inputs**
  - selected world
  - spawn pose
  - launch parameters
- **Outputs**
  - Gazebo world
  - `/odom`
  - `/external_camera/image_raw`
  - robot spawn and startup readiness

## Central Files

| File | Role |
| --- | --- |
| [`launch/bringup_sim.launch.py`](launch/bringup_sim.launch.py) | main simulator bringup used by `experiments` |
| [`gazebo_worlds/worlds/warehouse_occ_light.world.sdf`](gazebo_worlds/worlds/warehouse_occ_light.world.sdf) | primary benchmark world |
| [`gazebo_worlds/worlds/warehouse_open_shelves.world.sdf`](gazebo_worlds/worlds/warehouse_open_shelves.world.sdf) | exploratory secondary world |
| [`sim/wait_for_odom.py`](sim/wait_for_odom.py) | startup gate used in launches |

## Support Files

| File | Role |
| --- | --- |
| `launch/gazebo.launch.py` | Gazebo support launch |
| `launch/robot_description.launch.py` | robot-description support launch |
| `models/external_camera/model.sdf` | external camera simulation model |
| `robot_description/urdf/*.xacro` | TurtleBot3 description |
| `sim/reset_world.py`, `sim/wait_for_clock.py` | support utilities |

## What To Read First

1. `launch/bringup_sim.launch.py`
2. `gazebo_worlds/worlds/warehouse_occ_light.world.sdf`
3. `models/external_camera/model.sdf`
4. `sim/wait_for_odom.py`

## Important Caveat

This package is infrastructure, not the main method. It should appear in documentation as the plant and sensing environment, not as the thesis contribution.
