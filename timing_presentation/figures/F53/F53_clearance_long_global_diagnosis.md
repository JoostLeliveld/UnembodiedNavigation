# F53 - Clearance + Long Global Horizon Diagnosis

Generated from:

- Config: `scripts/visibility_comparison/aws_f53_b1_clearance_long_global_config.yaml`
- Logs: `logs/visibility_comparison/f53_b1_clearance_long_global_v1`
- Dashboard: `timing_presentation/figures/F53/F53_dashboard.png`

## What Changed

F53 kept the F49/F50 method but changed two targeted settings:

- `global_horizon: 80 -> 120`
- `local_nogo_safe_distance: 0.13 -> 0.20`

The hypothesis was:

1. longer global lookahead may catch the later low-visibility tail seen in F50 seed2;
2. local clearance matching the global keep-in layer may avoid the early C2 seed0 corner clip.

## Result

F53 did not improve robustness.

Campaign summary:

- 6 attempted runs.
- 0 reached the goal.
- 4 completed as failure.
- 2 were infrastructure invalid / wall-clock timeout.

Representative results:

| condition | seed | outcome | min goal | path | detector | max pixel age | mean p_vis | note |
|---|---:|---|---:|---:|---:|---:|---:|---|
| C1 | 0 | timeout / invalid | n/a | n/a | 0.33 | 76.0 s | 1.00 | baseline loses visual updates and wastes runtime |
| C1 | 1 | collision | 1.17 m | 3.78 m | 0.87 | 5.3 s | 1.00 | baseline still collides |
| C2 | 0 | collision | 3.49 m | 1.42 m | 1.00 | 1.0 s | 1.00 | local tracking / geometry clip |
| C2 | 1 | stuck | 0.35 m | 6.51 m | 1.00 | 1.0 s | 1.00 | safer route, but too conservative near goal |
| C2 | 2 | collision | 3.45 m | 1.37 m | 1.00 | 1.1 s | 1.00 | local tracking / geometry clip |

Global solve times increased substantially:

- C1: 46.1 s, 49.7 s, 76.5 s
- C2: 66.2 s, 70.2 s, 112.1 s

That makes `global_horizon=120` expensive for Gazebo campaigns.

## Interpretation

F53 rules out a simple explanation.

The C2 failures are not caused by missing live detections or a too-weak visibility term. In the completed C2 failures, YOLO detections are essentially always available, pixel corrections are fresh, and `p_vis_plan` remains high. The route is visible enough.

The problem is local execution.

The dashboard shows that C2 can follow the visible route toward the goal, but yaw/heading error remains large enough that the local tracker drifts toward the no-go boundary. Increasing `local_nogo_safe_distance` to 0.20 m makes seed1 safer but also causes it to stop at about 0.35 m from the goal instead of reaching the 0.25 m success radius. Seeds 0 and 2 still collide early, so clearance alone is not sufficient.

## Most Likely Root Cause

The local tracker uses yaw from the planner odometry topic. In these configs:

```yaml
use_encoder_noise: true
odom_topic: /odom_noisy
local_tracking_use_odom_yaw: true
```

Therefore the local EFE tracker steers using noisy wheel-odometry heading. The camera/GP pathway improves `(x,y)` localization, but it does not directly correct heading. This is paper-faithful, but it explains the unstable path shapes:

> The robot can be position-corrected by the camera while still steering with a biased/noisy heading estimate.

That makes the next runtime question a heading/local-control question, not a GP-strength question.

## Decision

Do not keep F53 as the locked method.

- `global_horizon=120` is too slow for routine Gazebo campaigns.
- `local_nogo_safe_distance=0.20` alone does not solve local collisions and can make the tracker stop just outside the goal.
- The next diagnostic should isolate heading/control, not increase ambiguity weight or global horizon.

## Recommended F54 Direction

Use F49/F50 as the base again, but test heading/local-control variants explicitly:

1. Keep the paper-faithful baseline:
   - camera updates `(x,y)`;
   - heading from noisy odometry;
   - command and encoder noise enabled.

2. Add a controlled heading ablation:
   - either enable displacement-based visual heading only when visual motion is large enough;
   - or add a separate future-work `visual heading / pose detector` variant.

3. Keep global horizon moderate:
   - return to `global_horizon=80` for runtime sanity;
   - avoid treating H120 as the default method.

4. Treat repeated zero-control no-progress as a valid early failure:
   - C1 should not spend minutes timing out after it has lost visibility and all local candidates violate the keep-in layer.

Paper-safe conclusion:

> In the AWS exploratory task, the learned visibility model can choose a more observable route, but robust closed-loop execution still depends on heading estimation and local tracking. Since the current paper-facing system only uses the external camera for `(x,y)` updates, yaw drift from wheel odometry can dominate the final execution behavior.

