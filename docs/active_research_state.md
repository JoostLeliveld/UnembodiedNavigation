# Active Research State

Last updated: 2026-05-20.

## Current Paper Position

The compact `warehouse_occ_light.world.sdf` benchmark remains the current paper
core. It is the cleanest evidence for the mechanism: a known driveable /
forbidden-zone layer is shared across conditions, while the learned observation
reliability changes the planner-facing camera `(x, y)` covariance.

The AWS-style warehouse is an exploratory extension. It may be useful for a
future Experiment B, but it is not paper evidence until the final geometry,
AWS-specific detector, AWS-specific GP, smoke validation, seeded logs, and
figures are complete.

## Active Hypothesis

The addition of learned observation reliability should make the planner prefer
routes that are longer but more observable when that tradeoff is meaningful. It
may also stop or become cautious when the planner-facing covariance makes a
state estimate too unreliable. The baseline can still reach easy goals, but is
expected to be more prone to poor localization and collision in camera-poor
regions.

## Current Validity

Valid paper-facing line:

- compact benchmark world;
- explicit detector and GP artifacts;
- C1 constant-covariance baseline versus C2 learned-observability EFE;
- paper metrics focused on compact tasks.

Exploratory line:

- AWS world with richer shelves, loading apron, and a high wall-mounted camera;
- latest smoke-style routes can test navigability, but route-choice evidence is
  not yet validated;
- R4 stack placement, route geometry, and GP recapture must be settled before
  interpreting C2 behavior.

Rejected AWS lessons:

- A visible-goal route-choice probe was not faithful evidence because the
  baseline already selected the detour-like route and the learned condition
  stalled at high ambiguity weight.
- A dark-final-goal probe confounded the experiment because both the route and
  the final goal were camera-poor.

## Next Decision

Decide whether Experiment B is worth completing now. If yes, first fix the AWS
geometry and recapture the AWS detector/GP chain. If no, keep AWS as future work
and strengthen the compact benchmark figures: traversability map, learned
reliability/coverage, ambiguity field, and total EFE cost decomposition.

Sparse route candidates are future work only. They should be framed as fair
coarse route scoring, not mission waypoints.

## Current Coarse-Planning Diagnostic

`scripts/visibility_comparison/coarse_route_evaluator.py` is an offline
diagnostic for testing whether the known driveable map plus learned observation
reliability naturally prefers a longer visible route. It does not publish
waypoints and is not an online planner.

Initial AWS B1 checks show that the current AWS GP/geometry still favors the A4
direct family even with high visibility weights. The explicit offline A3
candidate is longer and not lower-cost under the current artifact, so the
present AWS setup does not yet demonstrate a natural A3 detour incentive.
