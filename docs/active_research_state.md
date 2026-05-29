# Active Research State

Last updated: 2026-05-28.

## Current Paper Position

The compact `warehouse_occ_light.world.sdf` benchmark remains the current paper
core. It is the cleanest validated evidence for the mechanism: a known driveable
/ forbidden-zone layer is shared across conditions, while learned observation
reliability changes the planner-facing camera `(x, y)` covariance.

The new warehouse/AWS-style line is exploratory. It may become Experiment B, but
it is not paper evidence until final geometry, camera view, detector, GP, smoke
validation, seeded logs, figures, and paper wording are complete.

## Active Hypothesis

The addition of learned observation reliability should make the planner prefer
routes that are longer but more observable when that tradeoff is meaningful. It
may also stop or become cautious when planner-facing covariance makes the state
estimate too unreliable. The baseline can still reach easy goals, but is
expected to be more prone to poor localization and collision in camera-poor
regions.

This should not be demonstrated by simply using an oversized ambiguity weight.
The preferred direction is a general planner setup: adequate horizon coverage,
condition-neutral multistart for local minima, and goal-prior scheduling or
annealing so the planner can choose visible progress before tightening to the
goal.

## Current Validity

Valid paper-facing line:

- compact benchmark world;
- explicit detector and GP artifacts;
- C1 constant-covariance baseline versus C2 learned-observability EFE;
- paper metrics focused on compact tasks.

Exploratory line:

- AWS world with richer shelves, loading apron, and a high wall-mounted camera;
- newer warehouse layouts and timing diagnostics can test route-choice
  mechanisms, but route-choice evidence is not yet validated;
- AWS route geometry, camera view, GP capture, and solver timing must be
  settled before interpreting C2 behavior.

Rejected AWS lessons:

- A visible-goal route-choice probe was not faithful evidence because the
  baseline already selected the detour-like route and the learned condition
  stalled at high ambiguity weight.
- A dark-final-goal probe confounded the experiment because both the route and
  the final goal were camera-poor.

## Current Timing / Optimizer Diagnostic

Initial rollout diagnostics show that longer horizons and multistart can reveal a
visibility-aware solution basin that shorter horizons miss. In particular, H120
/ H200 multistart diagnostics reached near the intended goal while H40 / H80 did
not. This is useful evidence about local minima and horizon coverage, but the
solve times are too high to treat as a solved closed-loop method without further
work.

Interpretation: multistart may stay, but only as condition-neutral optimizer
basin handling. It must not become route-specific waypoint scripting.

Update from the latest AWS cleanup/check:

- `efe_offline_lab.py` now passes the same `nogo_mode=keep_in` and serialized
  driveable-region geometry used by Gazebo. Earlier offline route-choice
  diagnostics were therefore too optimistic when they omitted the true
  driveable-layer no-go.
- The active AWS task registry has been reduced to
  `B1_apron_a4_to_uppermid_a3` plus `visible_aisle_sanity_aws` to avoid
  selecting stale B1 variants. The active route-choice diagnostic starts at
  `(3.20, -1.00)` with yaw `0.0` (facing east toward R5 after a hypothetical
  pick) and goals at `(1.00, 1.75)`.
- For this active task, F18 shows that C2 H80 can prefer the A3-detour basin
  when seeded there, but the original/cold initialization does not find it. At
  H120 the direct-to-goal basin wins. This supports a local-optimum /
  first-long-solve investigation, but not yet a clean closed-loop paper result.
- The C2 Gazebo smoke run on the cross task failed by leaving the known
  driveable layer after only three replans. The main failure was solve-time /
  execution-cadence mismatch at `H=120`, not an interpretable visibility
  tradeoff.
- Important next-method option: the first solve can use a higher horizon than
  later replans. A long initial solve can expose the route-scale visibility
  tradeoff, while shorter subsequent replans keep closed-loop timing realistic.
  This should be treated as condition-neutral optimizer basin handling /
  warm-starting, not as mission waypoint forcing.
- Locked AWS runtime direction: hierarchical global-local planning is now the
  intended Gazebo path. The first/global solve uses a longer horizon to avoid
  local route-choice basins; the local tracker follows planner-derived
  waypoints. This is a robotics/MPC practicality, not a hand-authored mission.
- Realistic, modest lane-graph seeds are allowed because local optima strongly
  influence this nonlinear EFE controller. They may use the known 2D
  traversability/lane layout, but not the GP visibility field or condition
  labels.
- Runtime method contract is now locked in
  `docs/runtime_method_contract.md`. The corrected comparison is:
  C1 = constant-observability EFE with risk and ambiguity active; C2 =
  learned-observability EFE with risk and ambiguity active. The only intended
  condition difference is planner-facing camera observation covariance.
- Earlier offline/runtime wiring incorrectly disabled ambiguity for
  `constant_R_efe`; this has been corrected. Any older figure or run where C1
  has `ambiguity_cost=0` should be treated as stale for C1/C2 interpretation.
- AWS diagnostics now use a shared 2-sigma belief-tube driveable-region log
  barrier. Non-driveable floor is not a negotiable visibility tradeoff.
- F22 is the current fair initial-planning diagnostic. It uses the same neutral
  multistart candidates for C1 and C2; candidates may use known 2D driveability
  but not GP visibility.

Latest diagnostic figure:

- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F17_aws_route_choice_diagnosis.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F18_pick_east_plan_alternatives.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F22_realistic_multistart_choice.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F23/F23_hier_locked_smoke.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F24/F24_visible_route_battery.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F25/F25_r01_gazebo_smoke.png`

Latest AWS route-choice status:

- F23 shows that the hierarchical long-global / short-local method can make C2
  select the lower visible detour in offline initial planning when the goal prior
  and missing-observation covariance are set less aggressively.
- F24 shows a broader visible-to-visible offline route battery, but it is an
  initial-planning diagnostic only. Command and encoder noise do not act in that
  figure.
- F25 tested R01 in Gazebo. Both C1 and C2 crashed before reaching the goal.
  During local tracking, `p_vis_plan` was approximately 1 for both conditions,
  so the failure is not yet evidence about the visibility tradeoff. It is a
  closed-loop tracking / driveable-margin problem: the local tracker cuts close
  to the forbidden staging region under noise and belief error.

## Next Decision

Decide whether to invest in making the new warehouse line a validated Experiment
B. If yes, the next blocker is not another full Gazebo campaign; it is local
closed-loop tracking that preserves the global route while maintaining margin to
the known driveable/forbidden-zone layer. If no, keep AWS as future work and
strengthen compact benchmark figures: traversability map, learned
reliability/coverage, ambiguity field, total EFE cost decomposition, and solver
limitations.

Sparse route candidates are future work only. They should be framed as fair
coarse route scoring, not mission waypoints.

## Historical Coarse-Planning Diagnostic

`scripts/visibility_comparison/coarse_route_evaluator.py` is an offline
diagnostic for testing whether the known driveable map plus learned observation
reliability naturally prefers a longer visible route. It does not publish
waypoints and is not an online planner.

Initial AWS B1 checks showed that earlier non-hierarchical/cold-start settings
did not cleanly demonstrate the intended route-choice incentive. The current
direction is to test a robotics-faithful hierarchical global/local runtime with
shared lane-graph seeds before making any AWS claim.
