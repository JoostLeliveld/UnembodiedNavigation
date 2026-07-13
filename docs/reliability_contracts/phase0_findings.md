# Phase 0 Findings

## Repository Inventory

- Run artifacts are written below `logs/experiments/<experiment_*>/` with
  `experiment.csv`, `perception.csv`, `plan_samples.csv`, `run_manifest.json`, and
  `run_summary.json`. Hierarchical runs also write `global_plan.csv`,
  `global_waypoints.csv`, and `global_plan_meta.json`.
- Campaign state lives below `logs/visibility_comparison/.../campaign_log.json`.
  Offline GP workflows use `current_capture`, `current_targets`, and `current_gp`.
- `yolo_robot_detector_node` publishes `/perception/pixel_pose` and
  `/perception/detection_diagnostics`. Active selected-pixel localization is
  `bbox_bottom` because `yolo_use_masks: false`.
- `/state/bev` is derived from selected-pixel homography plus BEV calibration.
  Heading is odometry-driven under `camera_xy_only`.
- Planner belief is exposed on `/planner_belief`; prediction uses `/odom_noisy` by
  default.
- Current GP planning visibility queries load
  `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`.
- Current planning covariance diagnostics expose `p_vis`, `p_vis_eff`, `R_plan`, and
  per-axis standard deviations.
- The active runtime config is
  `scripts/visibility_comparison/warehouse_visibility_campaign.yaml`, launched
  through `warehouse_primary_comparison.launch.py` and
  `visibility_launch_common.py`.

## Ground-Truth Audit

- `/ground_truth_tf` is bridged from Gazebo for logger evaluation metrics,
  collision/clearance/goal/path auditing, and figures/metrics.
- YOLO dataset generation may use Gazebo semantic labels when clearly labeled as
  data generation.
- Visibility capture may teleport the robot and record oracle geometry fields for
  diagnostics/provenance.
- `use_truth_localization` is an oracle planner input and must remain disabled in
  normal reliability/campaign configs.

## Phase 0 Changes

- Added `src/reliability`, a ROS-independent package with dataclass contracts for
  operational samples, evaluation-only samples, reliability predictions, update
  covariance, and planning covariance.
- Added `src/reliability/config/leakage_firewall.yaml`, listing forbidden
  evaluation columns, forbidden topics/paths, allowed oracle contexts, forbidden
  planner-facing imports, and normal-runtime config constraints.
- Added schema docs and example JSON records under `docs/reliability_contracts/`.
- Added pytest coverage for serialization, unknown-field rejection, covariance
  shape/unit/SPD checks, feature/source leakage checks, planner-facing import
  checks, and the active runtime config GT-free regression.

## Deliberately Unchanged

- No planner, controller, EKF, no-go geometry, GP artifact, detector, or active
  campaign configuration behavior was changed.
- Existing legacy CSV files can still co-locate operational and evaluation columns.
  Phase 0 creates the split contracts and leakage tests; exporters that write split
  records belong to later phases.
- Existing dirty or untracked GP/geometry-visibility work is user-owned and was left
  outside the Phase 0 scope.

## Follow-On Work

- Add dataset exporters that write one operational record and one evaluation-only
  record per sample.
- Add training loaders that call the leakage firewall before fitting models.
- Add learned reliability adapters that output `ReliabilityPrediction`,
  `UpdateCovariance`, and `PlanningCovariance`.
