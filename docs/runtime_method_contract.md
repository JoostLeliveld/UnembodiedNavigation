# Runtime Method Contract

Last updated: 2026-06-03.

This document locks the runtime interpretation for the current visibility-aware
planning experiments. It is the reference for offline diagnostics, Gazebo smoke
tests, and any future paper-facing claim.

## Scientific Comparison

The comparison is not "shortest path" versus "visibility path". Both methods
are EFE planners with the same dynamics, same task, same driveable-region
constraints, same optimizer settings, and same candidate initializations.

The intended difference is only the planner-facing camera observation
covariance:

| Condition | Meaning | Uses GP? | EFE Risk | EFE Ambiguity |
| --- | --- | --- | --- | --- |
| C1 | constant-observability EFE | no | yes | yes |
| C2 | learned-observability EFE | yes | yes | yes |

C1 must not be implemented as risk-only. It still includes the ambiguity term;
it simply evaluates that term under a spatially constant observation covariance.

C2 uses the learned observation reliability map to blend the planner-facing
camera `(x, y)` covariance. The GP does not directly change heading, robot
dynamics, traversability, or mission goals.

## Multistart Contract

Multistart is allowed as condition-neutral optimizer basin handling. It is not a
mission planner and not route forcing.

Robotics motivation: the EFE planner is solved as a nonlinear receding-horizon
control problem, so local optima can dominate the behavior. Avoiding bad local
basins is part of a realistic robot-planning stack, not an optional cosmetic
trick. A warehouse robot would normally have access to the known floor/lane
layout; using that 2D traversability layer to seed plausible route basins is
scientifically faithful as long as the seeds are identical for C1 and C2 and do
not use learned visibility.

The AWS runtime uses a hierarchical solve:

- a longer first/global EFE solve chooses the route under the condition's own
  objective;
- the resulting states are converted into planner-derived waypoints;
- a shorter local controller tracks those waypoints with the same known
  driveable-region constraints.

These waypoints are allowed because they come from the optimizer, not from a
hand-authored mission script. They must be regenerated independently for C1 and
C2 under each condition's objective.

Allowed candidate sources:

- `cold_zero_init`: zero or previous-control initialization;
- `local_left_escape` / `local_right_escape`: short local maneuvers from the
  current robot heading;
- modest lane-graph route candidates generated from the known 2D
  traversability layer;
- shifted previous solution during closed-loop replanning.

Forbidden candidate sources:

- candidates generated from the GP visibility map;
- candidates only supplied to C2;
- labels or route logic that encode the expected conclusion, such as
  `visibility_route`, `A3_safe_route`, or mission waypoints;
- changing intermediate goals to force a route.

Lock-in rule:

```text
candidate generation may use known 2D driveability;
candidate generation may not use learned visibility;
candidate selection may use learned visibility only through the C2 objective.
```

For the locked Gazebo smoke/campaign path, keep the candidate set modest:
default/cold initialization, shifted previous solution, small condition-neutral
lateral seeds, and two lane-graph route seeds are acceptable. Do not use
visibility-derived route names or seeds as runtime evidence.

Locked AWS route-seed interpretation:

```text
mid_cross_lane:    route through the nearest mid cross-aisle
lower_sweep_lane:  route through the lower driveable corridor before returning north
```

These names describe known 2D floor topology only. They must not be described as
`visible`, `safe`, `A3`, or `A4` seeds in run-facing labels.

## Driveability / Forbidden-Zone Handling

The known 2D floor layer is a traversability constraint, not an observation
model. Non-driveable floor is not a visibility tradeoff.

Runtime planning should use:

- known driveable-region geometry from `world_profiles.yaml`;
- a 2-sigma belief-tube driveable-region log barrier;
- the same barrier parameters for C1 and C2;
- a hard validity gate that prevents invalid rollouts from beating valid ones;
- a safe-stop fallback only as execution hygiene.

The barrier clearance is:

```text
c_t = d_driveable(mean_xy_t) - r_clearance - 2 sigma_max(S_xy,t)
```

where `d_driveable` is positive inside the known driveable floor. For
planner-facing no-go cost, `S_xy,t` is the expected posterior covariance after
the predicted camera update. This is important: visible regions can keep the
belief tube narrower, while camera-poor regions increase uncertainty near
forbidden floor.

Paper wording should use `known driveable / forbidden-zone layer`, not
`obstacle map`, unless the text is specifically about physical collision
geometry.

The barrier should be hard enough that leaving the known driveable floor is not
an attractive tradeoff. If the robot physically collides or penetrates a
forbidden region during execution, the run terminates as a tracked failure.

## Active AWS Candidate Task

The active no-waypoint AWS paper-facing candidate task is:

```text
world: warehouse_aws.world.sdf
task:  F31_b1_apron_a3_mid
start: (3.30, -1.00, yaw=0.0)
goal:  (1.00, 1.75)
```

Interpretation: the robot starts in the right-side route-choice region and must
move to the upper target through the known driveable floor.

This remains candidate evidence until the full artifact chain is complete:
accepted world geometry, detector validation, GP capture/fit, complete C1/C2/C3
seeded logs, figures, and paper text alignment.

## Parameters To Freeze Before Gazebo

Do not tune these during a Gazebo campaign:

- `horizon`, `dt`, and first-solve horizon policy;
- hierarchical/global-local settings and waypoint extraction spacing;
- `v_max`, `w_max`, and control cost;
- goal-prior schedule and final covariance;
- ambiguity weight and risk scale;
- driveable barrier type, weight, clearance, and `nogo_belief_kappa`;
- process noise and initial belief covariance;
- `R_visible`, `R_miss`, and GP artifact path;
- multistart candidate set.
- command/encoder noise settings.

Tune offline first. Gazebo should test execution/replanning of a locked method,
not be used as the design loop.

## Runtime Finalization Gates

Before AWS route-choice runs are interpreted as evidence, the runtime must make
the state-observation-command chain reconstructable from logs. A run is only
paper-eligible when these diagnostics are available:

- image, YOLO receive/start/finish/publish, and frame-age timestamps;
- pixel-correction target/apply stamps, innovation, NIS, accept/reject flag, and
  reject reason;
- belief input stamp, odometry/command replay count/duration, and whether
  replay fell back to a single prediction;
- raw command, command after noise, command age, active control index, and active
  plan age;
- `/odom` and `/odom_noisy` pose/twist records for dead-reckoning comparison;
- raw and calibrated homography projection errors, so plots distinguish detector
  projection bias from the active state estimator.

The configured `odom_topic` must be explicit in the run manifest and shared by
the planner and BEV state node. When encoder noise is enabled this should be
`/odom_noisy`; otherwise state yaw fallback can silently use a cleaner heading
source than planner dead reckoning, making C1/C2 behavior hard to interpret.

Pixel corrections must be treated as delayed measurements:

```text
image-time belief -> correction -> replay/predict to now -> planning belief
```

Large corrections are diagnostic failures unless explicitly accepted by the
gate. Extreme updates should be rejected by configured jump/NIS thresholds, and
detection gaps should grow covariance rather than yanking the belief.

Belief prediction must replay the configured odometry topic over real
timestamped intervals. For paper-facing runs this is `/odom_noisy`, so
dead-reckoning follows the noisy encoder-style estimate rather than ideal
requested commands. Command logs are a fallback only: a 10 Hz command stream
must not be replayed as if every command lasted the planner `dt`, and a command
published after an image stamp must not be applied before that image existed.

Local execution is route tracking, not route choice. The long/global EFE solve
selects the route; the local controller tracks planner-derived waypoints with no
local visibility reward. Local solver outputs may replace the active control
tape only when they are finite, rollout-valid, have nonnegative predicted
driveable clearance, are not stale, and reduce the current waypoint distance.
Otherwise the execution layer safe-stops and the run is logged as stuck/safe-stop
if progress does not recover.

F66-F72 gate sequence:

- F66: static BEV calibration grid; compare raw homography, calibrated
  homography, `/state/bev`, and planner correction-implied BEV.
- F67: open-loop estimator replay; compare raw command replay, applied/noisy
  command, `/odom_noisy`, camera-corrected state, and planner belief.
- F68: pixel-correction gate test; inspect innovation/NIS/rejection and
  covariance behavior near racks and on open floor.
- F69: open-floor local tracking; no visibility tradeoff, just timing, waypoint
  progress, convergence, and safe command publication.
- F70: uniform-visibility C1/C2 sanity; both methods should behave almost the
  same when GP should not matter.
- F71: AWS B1 single-seed smoke only after F66-F70 pass.
- F72: AWS B1 three-seed smoke only after F71 is clean.

## Locked AWS Gazebo Candidate

The current candidate for AWS smoke testing is:

```text
use_hierarchical: true
global_horizon: 80
global_dt: 0.25
use_simple_local_controller: true
local_horizon: 6
local_plan_rate: 5.0
global_use_ambiguity: true
local_use_ambiguity: false
global_optimizer_multistart: true
local_optimizer_multistart: false
local_use_visibility_model: false
optimizer_multistart_include_direct: true
optimizer_multistart_lateral_offsets: -1.0,1.0
optimizer_initial_routes_json: mid_cross_lane + lower_sweep_lane, generated
  from known driveable-floor geometry only; the direct warm start remains
  enabled so lane candidates cannot silently exclude the short route
nogo_weight: 200.0
nogo_safe_distance: 0.30
use_belief_nogo_cost: true
nogo_belief_kappa: 2.0
r_visible_uv: 2.5
r_miss_uv: 40.0
ambiguity_weight: 8.0
use_command_noise: true
use_encoder_noise: true
```

Interpretation: the long global solve can see the route-level tradeoff; the
shared local tracker keeps execution smooth. Noise is part of the realism claim
and must be on for interpretable AWS Gazebo results.

## Current Candidate Evidence Status

The current candidate log root is:

`logs/visibility_comparison/paper_final_v1`

Current inspected summaries show C1 reaching by a shorter route with much larger
localization error, and C2 reaching by a longer route with lower localization
error. This supports a localization-safety/stability claim, not yet a
deterministic C1-fails/C2-succeeds claim.

Before using this as paper evidence:

- complete C1/C2/C3 with matched seeds;
- run the visible-route sanity task;
- generate cost-decomposition and perception/belief validation figures;
- clean stale inherited comments from the final config.
