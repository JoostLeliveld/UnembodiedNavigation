# Runtime Dataflow

This document describes how the offline observability artifact connects to the online ROS/Gazebo runtime. For paper-level protocol and naming, see [`paper_alignment.md`](paper_alignment.md).

## Offline Preparation

```text
world_profiles.yaml
-> capture_visibility_samples.py
-> raw detector samples
-> extract_perception_targets.py
-> build_gp_targets.py
-> fit_visibility_gps.py
-> logs/visibility_comparison/current_gp/*.npz
-> warehouse_primary_comparison.launch.py
```

The paper-facing GP artifact is fitted before navigation trials and then held fixed. The current compact benchmark uses:

- world: `warehouse_occ_light.world.sdf`
- artifact: `logs/visibility_comparison/current_gp/yolo_score_raw_gp.npz`
- planner-facing field: `P_conservative_plan_map`

Visibility-aware planner runs must pass `visibility_artifact_path` explicitly. The primary launch now fails instead of silently falling back to profile defaults.

## Online Runtime Path

```text
Gazebo external camera
-> /external_camera/image_raw
-> yolo_robot_detector_node
-> /perception/pixel_pose + /perception/detection_diagnostics
-> pixel_to_bev_state_node
-> /state/bev
-> efe_agent
-> /cmd_vel_raw
-> optional actuation_noise_node
-> /cmd_vel
-> Gazebo
-> experiment_logger
-> experiment.csv + run_manifest.json + run_summary.json
```

The method intervention is inside the planner prediction model:

```text
predicted state
-> GP reliability query
-> state-dependent detector-observation covariance
-> EFE risk and ambiguity terms
-> receding-horizon control
```

Observability is not a direct reward in the paper path.

## State Sources

The stable part of the estimator is:

- `x,y`: YOLO-selected pixel projected to the ground plane by camera geometry.
- GP input: planar `x,y` only.

The heading source is run-config dependent and is recorded in `run_manifest.json`:

| Run setting | Manifest value | Meaning |
| --- | --- | --- |
| `use_displacement_heading:=false`, `use_odom_heading_correction:=true` | `odometry_heading` | heading anchored by odometry |
| `use_displacement_heading:=true`, `use_odom_heading_correction:=false` | `pixel_displacement_heading` | diagnostic heading from consecutive camera-derived position updates |
| `keypoint_marker_world_z > 0` | `keypoint_bev_heading_with_odom_fallback` | visual keypoint heading with odometry fallback |

The current paper-facing configs and Task A figure manifests use odometry heading.

## Main Entry Points

| Purpose | File |
| --- | --- |
| Primary launch | `src/experiments/launch/warehouse_primary_comparison.launch.py` |
| Compact benchmark campaign | `scripts/visibility_comparison/paper_campaign_config.yaml` |
| Compact benchmark runner | `scripts/visibility_comparison/run_visibility_campaign.py` |
| Paper metrics | `scripts/visibility_comparison/compute_paper_metrics.py` |
| Paper figures | `scripts/visibility_comparison/thesis_plots/make_thesis_figures.py` |

The primary planner names are `constant_R_efe`, `visibility_aware_efe`, and optional `risk_only_ablation`.

## Failure-Oriented Extension

Experiment B has one active world:

- `warehouse_aws.world.sdf`, configured by
  `scripts/visibility_comparison/aws_campaign_config.yaml` only after an
  AWS-specific detector and GP are fitted for the final geometry. Use
  `scripts/visibility_comparison/aws_smoke_config.yaml` first to validate the
  AWS-style storage racks, green driveable boundaries, loading-apron props,
  B1/B2/B3 task starts/goals, wall-mounted camera, R4/A4 occluder, and detector
  behavior.

It should be treated as exploratory until it has fitted artifacts, completed
campaign logs, and figures/tables in the paper. Do not use mission waypoints to
force the route.

## Main Topics

| Topic | Produced by | Consumed by | Role |
| --- | --- | --- | --- |
| `/external_camera/image_raw` | simulator/bridge | detector | external camera image |
| `/perception/pixel_pose` | detector | state node, planner, logger | image-space robot observation |
| `/perception/detection_diagnostics` | detector | state node, planner, logger | observation diagnostics |
| `/state/bev` | state node | planner, logger | camera-derived BEV position state |
| `/goal_bev` | goal node | planner, logger | experiment goal |
| `/cmd_vel_raw` | planner | command-noise node or simulator | commanded control before optional noise |
| `/cmd_vel` | command-noise node or planner | simulator | executed control |
| `/planner_belief`, `/efe/metrics`, `/planner/diagnostics` | planner | logger | planner introspection |

## What This Omits

This page intentionally omits TF, robot-state publishing details, and plotting internals. Those are infrastructure, not the main method story.
