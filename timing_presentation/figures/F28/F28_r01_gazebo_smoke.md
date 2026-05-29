# F28 - R01 Gazebo Smoke Diagnostic

Config: `scripts/visibility_comparison/aws_f28_r01_gazebo_smoke_config.yaml`
Changes vs F27: `local_nogo_safe_distance: 0.13` (Fix A), `run_timeout_after_first_cmd_s: 160→200`.

## Results

| condition | outcome | path m | min goal m | min obs m | min wall m | mean solve ms | rollout_valid | truth_state_err m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | collision:obstacle_penetration | 5.25 | 1.47 | -0.058 | 2.095 | 941 | 1.00 | 0.549 |
| C2 | infra: Gazebo died overnight | — | — | — | — | — | — | — |

## What Fix A resolved

- `rollout_valid = 1.00` throughout — local planner is no longer frozen. Fix A works.
- `min_wall_distance_m = 2.095` — north wall crash from F27 is fully gone.
- C1 travelled 5.25 m and reached within 1.47 m of goal before crash (vs 3.57 m in F25).

## New primary failure: systematic belief y-lag

The crash is `geometry:obstacle_penetration` (rack/crate, 5.8 cm). Root cause is NOT the
planner choosing a dangerous path — it is a **systematic southward bias in the belief**:

| stamp | truth_y | belief_y | y-bias |
|---|---|---|---|
| 9 s (first cmd) | -1.96 | -1.86 | 0.10 m |
| 10 s | -0.98 | -1.28 | 0.30 m |
| 12 s | +0.56 | +0.11 | 0.45 m |
| 14 s | +0.99 | +0.40 | 0.59 m |
| 15.6 s (crash) | +2.25 | +1.53 | **0.72 m** |

The belief consistently places the robot 0.3–0.72 m south of truth. The bias
**grows monotonically** as the robot moves north. x-error is small (< 0.1 m).

This is NOT random homography blow-ups (|belief_y| > 4 outliers = 0 rows).
This is a systematic perspective projection error: the homography is calibrated to
z=0 (ground plane), but the camera projects the robot to a southward-biased position
that worsens as the robot moves further north from the camera.

Evidence for perspective origin: bias grows by ~0.13–0.20 m per metre of northward
travel, consistent with the camera at z=4.5m introducing a per-height perspective shift.

The planner plans "from y=1.53" but the real robot is at y=2.25 — 0.72 m closer to
the rack ahead. The plan looked clear; the actual robot was not.

## Preconditions analysis (before proposing F29 fix)

Precondition check per failure:
- Belief y-bias root cause: `keypoint_marker_world_z = 0.0` in config. The homography
  back-projects detected pixels assuming the robot contact point is at z=0. If the
  effective YOLO centroid height is h > 0, the projection is offset by
  ~h × (robot_y − camera_y) / camera_z southward.
  **Check needed**: measure bias vs distance to verify the perspective formula matches.
  Estimated effective centroid height from bias slope: ~0.55–0.65 m — suspiciously high
  for TurtleBot3 (~0.2 m body). Camera calibration error may also contribute.
- `rollout_valid = 1.00`: local planner working. Do NOT change local_nogo_safe_distance.
- `mean_solve_time_ms = 941 ms` (still over 500 ms budget). With maxiter=25 this is
  higher than expected. Check `optimizer_nit` to see if the optimizer converged in < 25
  iterations — if so, maxiter is not the bottleneck (nogo cost computation is).

## Proposed F29 fixes (in priority order)

FIX 1 (systematic belief y-bias — blocking everything):
  Empirically test `keypoint_marker_world_z: 0.15` (robot body centre height for TurtleBot3).
  Expected bias reduction: ~0.15 × (robot_y - (-4.9)) / 4.5 per metre of travel.
  Safe to apply: affects both C1 and C2 identically (localization calibration fix).
  Scientific justification: z=0 projection is wrong if YOLO centroid is above floor.
  VERIFY: check if URDF puts the robot base link at z=0 and the body centre at z≈0.15.

FIX 2 (C2 infrastructure — Gazebo process died overnight):
  Ensure C2 campaign run does not get killed by OS process limits or timeout.
  The workflow's bash timeout (550 s) is sufficient for C2 (46s solve + 200s run).
  Check if the system killed the process during overnight inactivity.

Figure: `timing_presentation/figures/F28/F28_dashboard.png`
PDF:    `timing_presentation/figures/F28/F28_dashboard.pdf`
