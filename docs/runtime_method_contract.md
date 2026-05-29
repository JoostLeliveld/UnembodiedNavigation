# Runtime Method Contract

Last updated: 2026-05-28.

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

## Active AWS Diagnostic Task

The active no-waypoint AWS diagnostic task is:

```text
world: warehouse_aws.world.sdf
task:  B1_apron_a4_to_uppermid_a3
start: (3.20, -1.00, yaw=0.0)
goal:  (1.00, 1.75)
```

Interpretation: the robot is shelf-facing after servicing the right-side shelf
and must move to the upper target through the known driveable floor.

This remains diagnostic/exploratory until the full artifact chain is complete:
accepted world geometry, detector validation, GP capture/fit, smoke runs,
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

## Locked AWS Gazebo Candidate

The current candidate for AWS smoke testing is:

```text
use_hierarchical: true
global_horizon: 80
local_horizon: 20
local_plan_rate: 4.0
global_use_ambiguity: true
local_use_ambiguity: false
global_optimizer_multistart: true
local_optimizer_multistart: true
optimizer_multistart_include_direct: false
optimizer_multistart_lateral_offsets: -1.0,1.0
optimizer_initial_routes_json: mid_cross_lane + lower_sweep_lane, generated
  from known driveable-floor geometry only
use_command_noise: true
use_encoder_noise: true
```

Interpretation: the long solve can see the route-level tradeoff; the local
controller keeps execution smooth. Noise is part of the realism claim and must
be on for interpretable AWS Gazebo results.

## Current Diagnostic Status

F22 is the current fair initial-planning diagnostic:

- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F22_realistic_multistart_choice.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F22_realistic_multistart_choice.csv`

It shows the corrected condition definitions: C1 now has nonzero ambiguity under
constant covariance. It also shows that the AWS task is not yet final paper
evidence: C1 and C2 select different basins, but the qualitative contrast still
needs to be made cleaner before Gazebo testing.
