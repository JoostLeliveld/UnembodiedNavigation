# Decision Log

Short, dated decisions that prevent the project from re-litigating the same
scientific choices.

## 2026-05-20

- Keep `warehouse_occ_light.world.sdf` as the paper core. It is the simplest
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

## Stable Wording Decisions

- Use `known driveable / forbidden-zone layer` for 2D planner constraints.
- Use `learned observation reliability` for the GP-derived reliability map.
- Use `3D occlusion affecting camera observations` for shelves, boxes, distance,
  perspective, and calibration effects.
- State that the GP affects camera `(x, y)` observation covariance only; heading
  is odometry-backed in the paper-facing runs.
