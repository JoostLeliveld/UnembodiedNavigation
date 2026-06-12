# Runtime Dataflow

This document describes how the offline observability artifact connects to the online ROS/Gazebo runtime. For paper-level protocol and naming, see [`paper_runtime_contract.yaml`](paper_runtime_contract.yaml). For how odometry/encoder/heading/process/measurement uncertainty is propagated (the three noise families, the analytical process-noise covariance `Q_d`, and how covariance couples to the obstacle cost and global plan), see [`uncertainty_propagation.md`](uncertainty_propagation.md).

## Offline Preparation

```text
world_profiles.yaml
-> capture_visibility_samples.py
-> raw detector samples
-> extract_perception_targets.py
-> build_gp_targets.py
-> fit_visibility_gps.py
-> logs/visibility_comparison/<world_gp>/*.npz
-> warehouse_primary_comparison.launch.py
```

The paper-facing GP artifact is fitted before navigation trials and then held fixed. The current AWS paper-facing line uses:

- world: `warehouse_aws.world.sdf` (external camera locked at z=4.8, y=-5.5)
- artifact: `paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz`
- planner-facing field: `P_conservative_plan_map`

(The former compact-benchmark line — `warehouse_occ_light` + `current_gp` — is retired/archived.)

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
- selected pixel source in the active AWS runtime: `bbox_bottom` (the most stable ground-plane
  proxy). `yolo_use_masks: false` in the locked runtime; masks are training and
  diagnostic artifacts only, not the selected localization pixel.
- GP input: planar `x,y` only.

The current paper-facing AWS configs use **`heading_update_mode: camera_xy_only`**:
odometry provides the dead-reckoning prediction, and pixel `(x,y)` updates influence
heading only indirectly through the unicycle prediction cross-covariance — there is no
separate odom-yaw anchor and no keypoint/visual heading. (The legacy
`odom_measurement` / `visual_heading` / `keypoint_bev_heading` modes are retired.)
If `/state/bev` becomes stale, planner code should use a predicted belief or refuse
planning; stale latest camera states must not be interpreted as fresh localization.

## Metrics

Runtime means for localization, yaw, control, and solve timing start after the
first non-trivial command. Pre-command rows contain launch waiting, global
solve time, and estimator warm-up. Use `truth_belief_error_m` for planner
localization quality, and interpret `truth_state_error_m` as the raw
camera-state pathway. Fresh `/state/bev` error and stale latest-state error
should be reported separately when diagnosing perception.

For delayed pixel corrections, inspect
`pixel_corr_motion_replay_source` in `experiment.csv`. Paper-facing runs should
normally report `odom_noisy`; `command_log` is an allowed fallback only when no
odom samples exist for the measurement interval, and `single_fallback` should be
rare enough to treat as a timing diagnostic. The older
`pixel_corr_cmd_replay_*` columns are retained for backward-compatible plotting;
interpret them as selected motion-replay sample counts/duration together with
`pixel_corr_motion_replay_source`, not necessarily as command-only replay.

## Main Entry Points

| Purpose | File |
| --- | --- |
| Primary launch | `src/experiments/launch/warehouse_primary_comparison.launch.py` |
| Locked robustness campaign config | `scripts/visibility_comparison/aws_f31b1_final_config.yaml` |
| Campaign runner | `scripts/visibility_comparison/run_visibility_campaign.py` |
| Paper metrics | `scripts/visibility_comparison/compute_paper_metrics.py` |
| Paper figures | `scripts/paper_figures/*.py` and selected `scripts/visibility_comparison/*plot*.py` diagnostics |

The primary planner names are `constant_R_efe`, `visibility_aware_efe`, and optional `risk_only_ablation`.

## Paper-Facing Campaign

The paper-facing AWS campaign uses `warehouse_aws.world.sdf`, detector
`aws_yolo_simseg_v2`, GP `aws_gp_v7b`, and
`scripts/visibility_comparison/aws_f31b1_final_config.yaml`. The config contains
four tasks (three discriminators plus one control) and five seeds per condition.
Do not use mission waypoints to force the route; route families in the optimizer
are condition-neutral multistart basins from the known driveable floor.

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
