# Dynamic Gazebo Demo

Generate and launch the dynamic AWS warehouse world:

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
python3 scripts/sim/generate_dynamic_warehouse_world.py
source install/setup.bash
ros2 launch sim bringup_sim.launch.py \
  world:=warehouse_aws_dynamic.world.sdf reset_world:=false \
  bridge_contacts:=false use_lidar:=false bridge_scan:=false
```

The derived world preserves `warehouse_aws.world.sdf` and adds four native
Gazebo Sim `TrajectoryFollower` models: a west-lane worker, apron forklift,
north worker, and cross-aisle pallet jack. Their names appear under
`/world/warehouse_aws_dynamic/dynamic_pose/info` while moving.

The actors are intended to demonstrate the D1 dynamic-occlusion fusion input;
they are not part of the static benchmark or a live performance result.
