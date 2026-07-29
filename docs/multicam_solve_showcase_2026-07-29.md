# Multicamera solve showcase — 2026-07-29

## Outcome

The waypoint-free `rob_easy` task now succeeds in the real
`warehouse_full_4cam.world.sdf` Gazebo world for all three seeds (0–2). This is
a **showcase**, not a full robustness-campaign claim.

The global optimizer was not the failed component: the earlier R-sweep runs
already recorded `rollout_valid: true`. The blocking defect was in the new
per-camera correction path:

1. disabling single-camera re-anchor accidentally selected paper-1's 0.5 m
   **pixel** jump limiter for a **metric** observation; and
2. `camera_xy_only` combined the posterior XY block with prior XY-heading cross
   terms, creating an indefinite belief covariance. The live signature was
   negative NIS, which is impossible for a valid innovation covariance.

The fix keeps paper-1's shared gate chain, treats each fused/per-camera result as
another measurement source, separates measurement space from re-anchor policy,
and makes the odometry-anchored heading independent of camera XY in the
committed covariance.

## Evidence chain

- World: `warehouse_full_4cam.world.sdf`
- Detector:
  `logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt`,
  four-camera batched YOLO, confidence threshold 0.05
- GP:
  `logs/visibility_comparison/spawn_grid_20260727/gp/camera_{A,B,C,D}/det_hit_expected_kernel_gp.npz`
- Projection calibration:
  `logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2/projection_calibration.json`
- Config: `scripts/visibility_comparison/_rob_rob_easy.yaml`
- Before-fix log:
  `logs/visibility_comparison/multicam_solve_showcase_real/rob_easy/C2/seed0/`
- Fixed logs:
  `logs/visibility_comparison/multicam_solve_showcase_fixed/rob_easy/C2/seed{0,1,2}/`
- Figure:
  `logs/studies/multicam_nav_demo/figures/fig33_multicam_solve_showcase.png`
- Figure generator:
  `scripts/visibility_comparison/render_multicam_solve_showcase.py`

## Measured showcase result

| quantity | before fix, seed 0 | fixed, seeds 0–2 |
|---|---:|---:|
| Goal success | 0/1 (stuck) | 3/3 |
| Collision | no | no (all seeds) |
| Deduplicated accepted corrections | 72/271 (26.6%) | 526/526 (100%) |
| Belief error p95 | 1.155 m | 0.174 m pooled |
| Negative NIS | present | 0 |
| Path length | 16.56 m | 15.71–15.82 m |
| Closest goal distance | 0.679 m | 0.007–0.065 m |

All three fixed runs used one start and one goal. The global planner produced
its own route; the task contains no mission waypoints.

## Claim boundary / paper wording

Safe wording:

> In a three-seed four-camera showcase, the consolidated correction path
> completed the waypoint-free central-corridor task in all runs while retaining
> a positive-semidefinite belief covariance and bounded localization error.

Do not generalize this to the difficult aisle missions yet. Those older tasks
contain mission waypoints and their prior failure attributions are provisional.
The next honest test is the waypoint-free difficult start–goal Gazebo run. Its
offline prerequisite now passes on both hard start/final-goal pairs; see
`docs/multicam_offline_hard_route_showcase_2026-07-29.md`.
