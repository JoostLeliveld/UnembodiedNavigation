# 03 — One camera: actual route, uncertainty, and GP update

## Show, in this order

1. [`figures/01_camera_c_actual_route_and_observations.png`](figures/01_camera_c_actual_route_and_observations.png)
2. [`figures/02_camera_c_recorded_pose_uncertainty.png`](figures/02_camera_c_recorded_pose_uncertainty.png)
3. [`figures/03_camera_c_gp_before_after_update.png`](figures/03_camera_c_gp_before_after_update.png)

## Say

“We start with just Camera C. The robot drove three separate 3.4 m warehouse
aisles—west of rack E1, between E1/E2, and between E2/E3. That is 10.2 m of
actual robot motion spread across the full Camera C collection region, not
three redundant passes in one aisle.”

“The recorder produced 64 Camera C GP samples: 51 detections and 13 misses.
Every sample is matched to the nearest independent noisy-odometry belief and
carries its propagated covariance. The median supplied 1σ positional
uncertainty is 0.034 m.”

“The first GP map is the day-zero Camera C prior: geometry before this drive.
The second is the expected-kernel GP fitted to these 64 recorded outcomes. A
Beta(1,1) prior smooths only the binary fitting target, so a single hit or miss
does not become a mathematically certain 0 or 1 update. It lowers reliability
in the west aisle, where Camera C missed repeatedly, and retains higher
reliability along the middle/east aisles, where it detected the robot. The
final panel is the GP latent uncertainty; the system should remain cautious
outside the sampled aisles.”

## Boundary

- All route and detector data come from
  `logs/studies/multicamera_commissioning_bigwarehouse/single_camera_c_multi_aisle_20260716`.
- The fitted artifact is
  `logs/visibility_comparison/single_camera_c_multi_aisle_20260716/camera_C_expected_kernel_beta11/det_hit_expected_kernel_gp.npz`.
- The three raw route logs are `01_west`, `02_middle`, and `03_east`; the
  fitting manifest records the Beta target pseudocount of 2.0.
- This is a collection-and-update demonstration, not held-out validation or a
  multi-camera claim. It contains no ground-truth input.
