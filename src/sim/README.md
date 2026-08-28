# `sim`

This package provides the plant and simulator plumbing for the thesis experiments.

It is the physical stage for the demo: Gazebo warehouse, external camera,
TurtleBot3 Burger description, startup gates, and optional command/encoder
noise.

## Central Files

| File | Role |
| --- | --- |
| [`launch/bringup_sim.launch.py`](launch/bringup_sim.launch.py) | simulator bringup used by `experiments` |
| [`gazebo_worlds/worlds/warehouse_aws.world.sdf`](gazebo_worlds/worlds/warehouse_aws.world.sdf) | locked paper-facing AWS-style warehouse visibility benchmark |
| [`models/external_camera/model.sdf`](models/external_camera/model.sdf) | external camera model |
| [`robot_description/urdf/turtlebot3_burger.urdf.xacro`](robot_description/urdf/turtlebot3_burger.urdf.xacro) | TurtleBot3 Burger robot description |
| [`sim/actuation_noise_node.py`](sim/actuation_noise_node.py) | optional executed-command perturbation |
| [`sim/encoder_noise_node.py`](sim/encoder_noise_node.py) | optional noisy odometry stream |
| [`sim/wait_for_odom.py`](sim/wait_for_odom.py) | startup gate used in launches |

## Outputs

- Gazebo world
- `/odom`
- `/external_camera/image_raw`
- robot spawn and startup readiness

## Demo Focus

The current public storyline should use `warehouse_aws.world.sdf`. Older compact
worlds and route probes are historical material unless a current registry entry
names them as evidence.

## Caveat

This package is infrastructure. It should appear in the paper as the plant and sensing environment, not as the thesis contribution.
