# Dynamic Occlusion Extension (D1)

The linked `belal-ibrahim/dynamic_logistics_warehouse` project demonstrates a
warehouse with waypoint-following actors. It targets Gazebo Classic/ROS 1 and is
GPL-2.0; this repository uses Gazebo Sim/ROS 2. Therefore D1 does not copy or
vendor that world or its assets. It independently adds the compatible fusion
mechanism: operational actor tracks affect the current fixed-camera update.

## Runtime contract

For each fusion frame, an actor tracker supplies `DynamicActorState` records:
an ID, ground-plane centre, footprint radius, and tracking confidence. D1
projects each actor centre onto the calibrated camera-to-observation segment.
Near-segment actors are combined into a conservative probability that at least
one actor blocks the sight line. The filter then:

1. multiplies camera availability by `1 - p_occ`;
2. inflates the full 2D observation covariance by a bounded odds response;
3. records the actor IDs, probability, and inflation factor in the replay step.

No actor track is inferred from map ground truth, and evaluation pose is still
only supplied as `EvaluationFrame` after replay.

## Results boundary

`tools/run_dynamic_occlusion_regression.py` creates a deterministic,
synthetic moving-occluder regression. It is a code-level result ensuring the
new response contains a gate-evading association displacement while an actor
crosses the line of sight. It is not a live Gazebo or physical performance
claim. A live study should first record timestamped actor tracks alongside the
existing camera observations, then run the same D1 replay configuration.
