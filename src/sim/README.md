# `sim`

This package provides the plant and simulator plumbing for the thesis experiments.

## Central Files

| File | Role |
| --- | --- |
| [`launch/bringup_sim.launch.py`](launch/bringup_sim.launch.py) | simulator bringup used by `experiments` |
| [`gazebo_worlds/worlds/warehouse_occ_light.world.sdf`](gazebo_worlds/worlds/warehouse_occ_light.world.sdf) | compact reported benchmark world |
| [`gazebo_worlds/worlds/warehouse_aws.world.sdf`](gazebo_worlds/worlds/warehouse_aws.world.sdf) | exploratory Experiment B AWS/JdeRobot-style warehouse visibility benchmark with B1/B2/B3 tasks |
| [`models/external_camera/model.sdf`](models/external_camera/model.sdf) | external camera model |
| [`sim/wait_for_odom.py`](sim/wait_for_odom.py) | startup gate used in launches |

## Outputs

- Gazebo world
- `/odom`
- `/external_camera/image_raw`
- robot spawn and startup readiness

## Caveat

This package is infrastructure. It should appear in the paper as the plant and sensing environment, not as the thesis contribution.
