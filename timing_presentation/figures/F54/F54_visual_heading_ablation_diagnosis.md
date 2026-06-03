# F54 - Visual Heading Ablation Diagnosis

Figure files:

- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F54/F54_dashboard.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F54/F54_dashboard.pdf`

Log root:

- `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f54_b1_visual_heading_ablation_v1`

## Purpose

F54 tests whether replacing local tracking yaw from noisy odometry with
visual-displacement heading makes the AWS B1 visibility-aware route more stable.

This is an exploratory ablation. It is not a paper-locked runtime setting.

## Configuration Change Relative To F50

Only C2 was run, for seeds 0, 1, and 2.

Important changes:

- `use_displacement_heading: true`
- `heading_min_displacement_m: 0.18`
- `local_tracking_use_odom_yaw: false`
- `global_horizon: 80`
- `local_horizon: 12`
- `local_optimizer_multistart: false`
- `local_use_ambiguity: false`
- command noise and encoder noise remain enabled

The intent was to test heading handling, not to change the route-choice
objective.

## Outcomes

| Condition | Seed | Outcome | Min goal distance | Notes |
|---|---:|---|---:|---|
| C2 | 0 | collision | 0.389 m | Got fairly close, then collided |
| C2 | 1 | infra invalid | n/a | Wall-clock timeout |
| C2 | 2 | infra invalid | n/a | Wall-clock timeout |

Campaign summary:

- `0/3` reached the goal.
- `1/3` completed as a collision.
- `2/3` were wall-clock timeouts.

## Important Diagnostics

The planner-facing visibility was not the limiting factor in these runs:

| Seed | Mean `p_vis_plan` | Mean truth-state error | Mean rollout valid | Mean solve time |
|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.241 m | 0.918 | 484 ms |
| 1 | 1.00 | 0.261 m | 1.000 | 428 ms |
| 2 | 1.00 | 0.118 m | 1.000 | 435 ms |

This is the key negative result: the failures happened while the local planner
believed it was in highly visible, valid regions. That points away from GP
route selection and toward execution/tracking/runtime termination.

The visual-displacement heading signal was also not actually a clean heading
source:

| Seed | Heading source counts |
|---:|---|
| 0 | `held_previous_heading`: 631, `odom_heading_fallback`: 46, `unknown`: 39 |
| 1 | `held_previous_heading`: 881, `odom_heading_fallback`: 4, `unknown`: 48 |
| 2 | `held_previous_heading`: 898, `odom_heading_fallback`: 2, `unknown`: 48 |

Most frames used a held previous heading. Very few used odometry fallback, and
none show a robust visual heading stream. In other words, enabling displacement
heading mostly replaced the local tracker heading with a stale/held heading
policy.

## Conclusion

F54 rejects the current visual-displacement-heading ablation as a stabilization
fix. It does not make C2 more predictable, and it introduces long no-progress
timeouts.

The root issue is more likely:

1. local tracking and command generation after the global plan,
2. stale or held heading during local tracking,
3. missing runtime classification for no-progress / safe-stop behavior,
4. expensive global solves making these failures costly to diagnose.

Follow-up code check: the campaign wrapper previously mapped any completed
non-goal/non-timeout outcome to `collision`, unless the run summary explicitly
reported `crashed=True`. That would have mislabeled future `stuck` outcomes as
collisions. This has been corrected so `stuck` remains a distinct outcome and
`goal_reached_stable` counts as goal reached.

For the paper-facing method, the safer position is:

- keep heading as odometry-derived in the current implementation,
- state clearly that the camera update affects `(x,y)` only,
- treat visual or detector-based heading as future work unless separately
  validated.

## Next Method Fix

Do not keep tuning visibility weights based on F54. The next useful fix is a
runtime-method fix:

- classify sustained no-progress / repeated zero-command / stale local tracking
  as a logged `stuck` outcome instead of waiting for a wall-clock timeout;
- keep collision as a hard terminal failure;
- keep the planner objective route-choice mechanism unchanged while debugging
  local tracking.

After that, rerun the F49/F50-style setting with the paper-safe heading setup:

- C1 and C2,
- command and encoder noise enabled,
- odometry heading for local tracking,
- global multistart allowed,
- local multistart disabled unless explicitly reported as a neutral solver
  robustness ablation.
