# F56 - Robust Local Tracker Ablation

Status: diagnostic / exploratory, not paper evidence.

## Artifact paths

- Config: `/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/aws_f56_robust_local_tracker_config.yaml`
- Log root: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f56_robust_local_tracker_v2`
- Dashboard PNG: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F56/F56_dashboard.png`
- Dashboard PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F56/F56_dashboard.pdf`

`f56_robust_local_tracker_v1` is invalid and should not be used. It was run inside the sandbox and failed due Gazebo / RTPS / Ignition runtime restrictions before producing real runs.

## Change Relative To F55

Changed exactly one meaningful runtime component:

- `use_simple_local_controller: false -> true`

Held fixed:

- world: `warehouse_aws.world.sdf`
- task: `F31_b1_apron_a3_mid`
- seeds: `[0, 1, 2]`
- detector: `aws_yolo_simseg_v2`
- GP: `aws_gp_v5/yolo_score_raw_gp.npz`
- command noise and encoder noise enabled
- global multistart enabled
- local multistart disabled
- no mission waypoints / no route-forcing script

The `optimizer_initial_routes_json` entries are condition-neutral global optimizer initializations. They are not mission waypoints and are used identically for C1 and C2.

## Outcomes

| Condition | Seed | Outcome | Goal? | Path [m] | Min Goal [m] | Mean Truth Error [m] | Min Obstacle Distance [m] |
|---|---:|---|---|---:|---:|---:|---:|
| C1 | 0 | collision | no | 8.67 | 1.008 | 0.412 | -0.024 |
| C1 | 1 | collision | no | 3.01 | 1.420 | 0.249 | -0.004 |
| C1 | 2 | collision | no | 5.26 | 1.794 | 0.268 | -0.068 |
| C2 | 0 | goal_reached | yes | 9.16 | 0.038 | 0.214 | 0.141 |
| C2 | 1 | collision | no | 2.33 | 3.367 | 0.116 | -0.004 |
| C2 | 2 | collision | no | 2.20 | 3.405 | 0.111 | -0.001 |

Global solve times from ROS logs:

- C1: 50.3 s, 75.3 s, 64.5 s
- C2: 89.7 s, 83.5 s, 75.3 s

## Comparison To F55

F55 logger/outcome-classification run:

- C1: 2 stuck, 1 collision, 0/3 goals
- C2: 3/3 goal reached
- mean C2 truth-state error: 0.178 m

F56 robust local tracker:

- C1: 3 collisions, 0/3 goals
- C2: 1/3 goal reached, 2 early collisions
- mean C2 truth-state error: 0.147 m

The simple tracker did reduce average localization error and removed stuck labels, but it made the C2 execution less reliable. The failed C2 seeds collided early with large remaining goal distance, so this is not an acceptable replacement execution layer yet.

## Diagnosis

The F56 result separates two issues:

1. The global route-choice layer still runs and hands off routes, but global solve latency remains very high: roughly 50-90 s.
2. The simple local tracker is not robust enough around the AWS route geometry under command + encoder noise. It can complete one C2 seed, but two C2 seeds collide before making meaningful goal progress.

This means the problem is not only local EFE solver basin behavior. The local execution layer also needs reliable timing / handoff / clearance handling. F56 does not justify replacing local EFE tracking with the simple tracker as the default method.

## Keep / Reject Decision

Reject F56 as the preferred AWS execution layer.

Keep the F55 logging semantics fix, but do not keep `use_simple_local_controller: true` as the locked runtime method. The next step should be F57-style timing and handoff diagnostics around the F55-style local EFE tracker, not IMU or keypoint heading yet.

F57 should focus on:

- command age and active control index,
- global-to-local handoff latency,
- local replan timing,
- waypoint progression / oscillation,
- whether long global solve time is starving the runtime loop.

