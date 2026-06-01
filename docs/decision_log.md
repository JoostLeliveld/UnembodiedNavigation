# Decision Log

Short, dated decisions that prevent the project from re-litigating the same
scientific choices.

## 2026-05-20

- `warehouse_aws.world.sdf` is now the paper benchmark. `warehouse_occ_light` was the original candidate but superseded before seeded Gazebo validation. It is the simplest
  validated setting for showing state-dependent observation uncertainty.
- Keep `warehouse_aws.world.sdf` exploratory. It requires final geometry,
  detector retraining/validation, visibility capture, GP fitting, smoke tests,
  seeded logs, and figures before it can support a claim.
- Remove mission waypoint support. Route choice must emerge from the planner
  objective, not from a mission script that changes the goal sequence.
- Reject the AWS visible-goal route-choice probe as paper evidence. The baseline
  already took the detour-like route, while the learned condition stalled when
  ambiguity was weighted aggressively.
- Reject the AWS dark-final-goal route-choice probe as paper evidence. It mixed
  route visibility with a camera-poor final goal, making the result hard to
  interpret.
- Treat sparse planning as future work. A scientifically fair version may score
  coarse route candidates with the same objective terms, but it should not
  inject route-forcing waypoints into the local controller.

## 2026-05-27

- Move AI/research authority to `/home/joostleliveld/Thesis/CLAUDE.md`. The
  `UnembodiedNavigation` and `thesis-report` guidance files are supplements only.
- Keep broad Claude permissions for speed, but encode stronger behavioral rules:
  no destructive cleanup without explicit delete lists, no YOLO/GP recapture
  before accepted geometry, no Gazebo campaigns before offline sanity checks, and
  no paper claim without the full artifact chain.
- Retire repo-local agent prompts in favor of root agents:
  `experiment-designer`, `rollout-runner`, `planner-diagnostician`,
  `figure-analyst`, and `paper-rigor-writer`.
- Preserve multistart. It is allowed as condition-neutral optimizer basin
  handling and must be reported. It is not a mission waypoint mechanism.
- Treat long-horizon/multistart timing results as a useful diagnostic:
  they can show that the visibility-aware solution exists in the objective, but
  current solve times are a scalability limitation.
- Prefer general planner mechanisms such as goal-prior scheduling/annealing and
  normalized costs over simply increasing ambiguity weight.

## 2026-05-28

- Lock the runtime method contract in `docs/runtime_method_contract.md`.
- Define C1 as constant-observability EFE with both risk and ambiguity active.
  C1 differs from C2 by not querying the GP and by using spatially constant
  observation covariance, not by removing ambiguity.
- Define C2 as learned-observability EFE with both risk and ambiguity active.
  The GP affects planner-facing camera `(x, y)` covariance only.
- Use condition-neutral multistart as optimizer basin handling. Candidate
  generation may use the known 2D driveable floor and local maneuvers, but not
  learned visibility or the condition label.
- Use a shared 2-sigma belief-tube driveable-region log barrier for AWS
  diagnostics. Non-driveable floor is a forbidden-zone/traversability layer, not
  an observation-reliability tradeoff.
- Lock AWS Gazebo diagnostics to a robotics-faithful hierarchical runtime:
  longer global EFE route solve, short local tracker, command/encoder noise on,
  and crash/contact as terminal tracked failures.
- Permit modest lane-graph optimizer seeds generated from the known 2D
  traversability layer. This addresses supervisor feedback about local optima
  without scripting the desired visibility-aware route. The seeds must be shared
  by C1/C2 and must not use GP visibility.
- Treat any older diagnostic in which `constant_R_efe` has `ambiguity_cost=0`
  as stale for C1/C2 interpretation.

## Stable Wording Decisions

- Use `known driveable / forbidden-zone layer` for 2D planner constraints.
- Use `learned observation reliability` for the GP-derived reliability map.
- Use `3D occlusion affecting camera observations` for shelves, boxes, distance,
  perspective, and calibration effects.
- State that the GP affects camera `(x, y)` observation covariance only; heading
  is odometry-backed in the paper-facing runs.
