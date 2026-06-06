# Active Research State

Last updated: 2026-06-03.

## Current Paper Position

The compact benchmark remains the cleanest validated mechanism evidence already
available in the repository. The AWS-style warehouse line is now the active
paper-facing candidate, but it should be treated as final paper evidence only
after the complete artifact chain is present:

- fixed world and camera pose;
- validated YOLO detector for that world;
- fitted AWS GP artifact;
- locked config;
- complete C1/C2/C3 seeded logs;
- final figures/tables generated from those logs;
- paper wording aligned with the actual runtime method.

The current AWS claim should be framed as localization-safety and route
observability, not as deterministic goal-reaching superiority.

## Active Hypothesis

Learned observation reliability should make the planner prefer a longer route
through better-observed floor space when that route keeps the belief more
localized near the known driveable / forbidden-zone layer. The constant-R
baseline may still reach the goal, but it is expected to do so with larger
localization error, larger p95 error, worse yaw consistency, and lower safety
margin in camera-poor regions.

The effect must emerge from planner-facing covariance, shared driveability
handling, and condition-neutral optimizer basin handling. It must not be
created by mission waypoints, GP-dependent route seeds, or simply overwhelming
the objective with an oversized ambiguity weight.

## Locked Runtime Interpretation

The current AWS runtime is hierarchical:

- a global EFE solve performs route-level planning;
- neutral multistart seeds may use the known 2D driveable/lane layout;
- the selected global plan is converted into planner-derived waypoints;
- a shared local tracker follows those waypoints;
- local tracking is execution plumbing and is not the GP contribution.

The C1/C2 comparison is:

| Condition | Paper name | GP? | Risk | Ambiguity | Intended difference |
| --- | --- | --- | --- | --- | --- |
| C1 | constant-observability EFE | no | active | active | constant camera covariance |
| C2 | learned-observability EFE | yes | active | active | GP-derived camera covariance |
| C3 | GP-risk-only ablation | yes | active | off | tests whether ambiguity is needed |

The only intended C1/C2 difference is planner-facing camera observation
covariance. The task, detector, GP availability except for C1 queries, known
driveable barrier, tracker, noise, seeds, horizons, optimizer budgets, and
candidate route seeds must be shared.

Runtime metrics must start after the first non-trivial command. Camera-derived
`/state` must also be interpreted with freshness: `state_available` only means a
latest state message exists, not that the current frame produced a fresh YOLO
update. Fresh camera-state error, stale latest-state error, and planner
truth-belief error should be reported separately when diagnosing perception.

Runtime cleanup note (2026-06-05): planner fallback paths must not silently use
stale `/state/bev`. When pixel correction is disabled or a camera update is
stale, planning should use the predicted belief from the last valid state or
refuse to plan until a fresh state is available. Camera-off/correction-off
ablations created before this rule are diagnostic only.

Local execution note (2026-06-05): the simple proportional local tracker is
shared execution plumbing, not the GP contribution. It tracks planner-derived
waypoints with odometry yaw and a configurable yaw gate, and it has a lightweight
predicted mean-clearance gate. Do not describe it as a local EFE/belief-tube
optimizer unless `use_simple_local_controller:false` and the local EFE path is
actually used.

## Current AWS Candidate

Primary candidate config:

`scripts/visibility_comparison/aws_paper_final_config.yaml`

Primary candidate log root:

`logs/visibility_comparison/paper_final_v1`

Primary task:

```text
world: warehouse_aws.world.sdf
task:  F31_b1_apron_a3_mid
start: (3.30, -1.00, yaw=0.0)
goal:  (1.00, 1.75)
```

Current inspected summaries:

Runtime localization metrics are interpreted as after-first-command quantities.
Pre-command launch, global-solve, and estimator warm-up rows must not be mixed
into these means.

| Condition | Completed seeds | Outcome | Mean path | Mean truth-state error after first cmd | Mean p95 truth-state error after first cmd |
| --- | ---: | --- | ---: | ---: | ---: |
| C1 constant-R | 5 | 5/5 goal reached | 4.64 m | 0.366 m | 1.66 m |
| C2 learned-R | 5 | 5/5 goal reached | 6.82 m | 0.195 m | 0.43 m |
| C3 risk-only | 5 | 5/5 goal reached | 6.89 m | 0.196 m | 0.45 m |

Current interpretation:

> The constant-observability baseline takes a shorter route and can reach the
> goal, but with much larger localization error. The learned-observability
> planner takes a longer route and reaches with lower localization error and a
> more stable belief.

C3 currently behaves similarly to C2, which suggests the dominant AWS mechanism
is GP-conditioned belief-risk / driveable-margin behavior rather than an
ambiguity-only route flip. The paper should present this honestly.

Do not claim yet:

- C1 fails while C2 succeeds;
- AWS is final paper evidence;
- the GP directly models physical occlusion geometry;
- YOLO provides heading;
- local waypoint tracking is itself the visibility-aware contribution.

## Runtime Details To Keep Paper-Accurate

- Current YOLO-selected localization pixel is bounding-box bottom center. Mask
  diagnostics exist, but mask-bottom is not the selected pixel source.
- Camera updates provide `(x, y)` through ground-plane projection.
- Heading comes from odometry.
- Projection/calibration uncertainty is not separately propagated into EKF or
  EFE covariance.
- The known 2D driveable / forbidden-zone layer is shared across conditions.
- The 2-sigma belief-tube driveable barrier is a feasibility/safety mechanism,
  not learned occlusion.
- Command and encoder noise are part of the AWS realism claim and should remain
  on for final AWS campaigns.
- Crash/contact ends a run and must be tracked separately from stuck, timeout,
  and goal reached.

## Immediate Next Decisions

1. Run the uniform-visible sanity task with the same locked runtime.
2. Generate final dashboards, summary tables, and cost-decomposition figures
   from the completed logs.
3. Clean stale inherited comments from `aws_paper_final_config.yaml`.
4. Once the artifact chain is complete, decide whether the thesis/paper presents
   AWS as the main result or as an Experiment B extension alongside the compact
   benchmark.

For the detailed final experiment suite, use:

`docs/final_experiment_definition.md`
