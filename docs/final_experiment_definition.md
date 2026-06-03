# Final Experiment Definition And Runtime Audit

Last updated: 2026-06-03.

This note defines the paper-facing experiment direction after the latest AWS
runtime finalization work. It is intentionally cautious: a result is paper
evidence only when the world, detector, GP, config, logs, figures, metrics, and
paper wording all agree.

## Current Runtime Audit

The final AWS runtime logic is mostly coherent, with one important framing
change. The strongest current claim is not that the baseline always fails. The
latest runs show that the baseline can often reach by a shorter route, but with
substantially worse localization, yaw consistency, and clearance. The learned
condition takes a longer, more observable route and reaches with lower state
error and more stable belief.

### Condition Logic

- `constant_R_efe` is a valid baseline: it uses constant planner-facing camera
  covariance, with EFE risk and ambiguity still active.
- `visibility_aware_efe` is the method: it uses the GP-derived
  state-dependent camera covariance.
- `risk_only_ablation` is the key ablation: GP covariance remains active, but
  ambiguity is disabled.
- The intended C1/C2 difference is only planner-facing observation covariance.
  The world, detector, task, driveable barrier, route seeds, tracker, noise, and
  optimizer budgets must be shared.

### Perception And Projection

- The runtime uses YOLO segmentation outputs, but the selected localization
  pixel is currently the bounding-box bottom center.
- Mask diagnostics are logged, but mask-bottom is not the selected state update
  source in the current code.
- The paper should therefore describe the current projection semantic as
  bbox-bottom ground-plane projection unless the implementation is changed.
- The camera supplies `(x, y)` position updates only. Heading is from odometry,
  not YOLO heading or keypoints.
- Projection and calibration uncertainty are still not propagated as a separate
  uncertainty source; they are handled only through empirical calibration,
  gating, and the planner-facing covariance model.

### Belief And Yaw

- The current heading story is paper-safe: visual detections update position,
  while odometry supplies yaw.
- The previous stale/held heading issue was real; fresh odometry-heading
  fallback should remain locked and logged.
- Delayed visual corrections are now more defensible because command-log
  replay and pixel correction diagnostics exist.
- Final runs should report truth-state error, belief error, yaw error, pixel
  correction accept/reject statistics, and detection availability.

### Driveability And Obstacle Cost

- Non-driveable floor must be treated as a known 2D traversability/forbidden
  layer, not as visibility or occlusion.
- The 2-sigma belief-tube driveable barrier is the right paper-facing mechanism:
  predicted uncertainty near forbidden zones should be penalized, and leaving
  the driveable floor should never be an acceptable visibility tradeoff.
- Contact/collision still ends a run and is tracked as an execution outcome.
- This should be described as execution safety and known traversability, not as
  learned occlusion.

### Local Execution

- The AWS runtime is now hierarchical:
  global EFE performs the route-level choice, then a shared local tracker follows
  planner-derived waypoints.
- This is scientifically acceptable if it is reported honestly. The local
  tracker is execution plumbing, not the GP contribution.
- Global multistart is allowed only as condition-neutral optimizer basin
  handling using known 2D lanes; it must not use GP visibility, condition labels,
  or route-forcing mission waypoints.
- The simple tracker yaw gate is currently hard-coded in the node. If it remains
  part of the final runtime, the value should be logged or surfaced as an
  explicit config parameter.

## Latest Evidence Snapshot

Current log root inspected:

`logs/visibility_comparison/paper_final_v1`

Current config inspected:

`scripts/visibility_comparison/aws_paper_final_config.yaml`

The latest completed summaries show:

| Condition | Completed seeds | Outcome | Mean path | Mean min-goal | Mean truth-state error | Mean p95 truth-state error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| C1 constant-R | 5 | 5/5 goal reached | 4.64 m | 0.060 m | 0.366 m | 1.66 m |
| C2 learned-R | 5 | 5/5 goal reached | 6.82 m | 0.118 m | 0.195 m | 0.43 m |
| C3 GP risk-only | 5 | 5/5 goal reached | 6.89 m | 0.091 m | 0.196 m | 0.45 m |

This supports the following paper-safe interpretation:

> The constant-observability baseline takes the shorter route and can still
> reach the goal, but it accumulates much larger localization error in the
> camera-poor part of the warehouse. The learned-observability planner chooses a
> longer route with better observation reliability and reaches with lower
> localization error and more stable belief.

It does not yet support the stronger statement:

> C1 fails and C2 succeeds.

That stronger claim would require a completed stress task or multi-seed result
where failure rates separate cleanly without changing the method unfairly.

Current blockers before treating this as final paper evidence:

- The final config header contains inherited diagnostic notes from many older
  F-runs and should be cleaned before publication or agent reuse.
- Repo docs currently disagree about whether AWS is paper core or exploratory.
  Resolve this only after the final artifact chain is complete.
- The uniform-visible sanity campaign still needs to show that C1 and C2 behave
  similarly when learned visibility should not matter.
- Final cost-decomposition and perception/belief validation figures still need
  to be generated from the completed campaign.

## Final Experiment Suite

### E0 Artifact And Runtime Hygiene Gate

Purpose: make sure the final runs are interpretable.

Required checks:

- world: `warehouse_aws.world.sdf`;
- detector: `logs/perception_models/aws_yolo_simseg_v2/model.pt`;
- GP: `logs/visibility_comparison/aws_gp_v5/yolo_score_raw_gp.npz`;
- config: cleaned final AWS paper config;
- no extra `encoder_noise_node`, `actuation_noise_node`, Gazebo, or planner
  stragglers before each campaign;
- every run has `run_manifest`, `experiment.csv`, `plan_samples.csv`,
  `run_summary.json`, and a dashboard.

### E1 Main AWS Route-Choice Campaign

Purpose: primary paper experiment for the new warehouse line.

Task:

- `F31_b1_apron_a3_mid`
- start: `(3.30, -1.00, yaw=0.0)`
- goal: `(1.00, 1.75)`

Conditions:

- C1: `constant_R_efe`
- C2: `visibility_aware_efe`
- C3: `risk_only_ablation`

Minimum seeds:

- 5 complete seeds per condition.

Preferred seeds:

- 10 complete seeds per condition if runtime allows.

Primary measurements:

- goal reached, collision, stuck, timeout;
- route class and path length;
- elapsed time after first command;
- mean and p95 truth-state error;
- mean and p95 belief error;
- mean and p95 yaw error;
- min driveable clearance / obstacle distance;
- detection availability and rejected correction rate;
- global solve time and local tracker timing.

Expected evidence:

- C1 should favor the shorter route.
- C2 should favor the longer observable route.
- C2 should reduce localization error and improve safety margin.
- C3 decides whether the main mechanism is mostly GP-conditioned belief-risk /
  driveability or whether ambiguity is necessary for the observed route choice.

### E2 Uniform-Visibility Sanity Campaign

Purpose: prove that the GP does not arbitrarily change behavior when visibility
is not a meaningful route variable.

Task:

- `visible_aisle_sanity_aws`, or an equivalent visible-to-visible route from
  the same final task registry.

Conditions:

- C1 and C2 required.
- C3 optional.

Acceptance:

- C1 and C2 should choose similar routes;
- path length, state error, and clearance should be comparable;
- any strong route difference must be explained by the cost terms, not assumed.

### E3 Mechanism Figure: Initial Plan And Cost Decomposition

Purpose: show what the optimizer is using, not only the GP field.

Figure requirements:

- same neutral route seeds for C1/C2/C3;
- initial global plans;
- route labels from known 2D layout, for example `mid_cross_lane` and
  `lower_sweep_lane`, not condition-specific labels like "safe route";
- decomposition of risk, ambiguity, driveable/belief-nogo, and total cost;
- overlay of learned reliability as context, not as the only evidence.

Acceptance:

- C1 has ambiguity active under constant covariance;
- C2 and C3 differ only by ambiguity;
- driveable cost is shared and cannot be traded away against visibility.

### E4 Perception And Belief Validation Figure

Purpose: answer supervisor concerns about projection, calibration, and
dead-reckoning.

Figure requirements:

- bbox-bottom pixel to ground-plane `(x, y)` projection;
- static or synchronized residuals versus truth;
- accepted and rejected pixel corrections;
- detection availability over the route;
- yaw source and yaw error over time;
- statement that heading comes from odometry.

Acceptance:

- perception failures explain uncertainty growth rather than hidden control
  changes;
- detection gaps increase uncertainty or reduce correction quality;
- large YOLO jitter is gated instead of silently yanking belief.

### E5 Secondary Stress / Generalization Task

Purpose: optional stronger claim that C1 is more likely to fail or stop while C2
remains safer.

Rules:

- same world, detector, GP, runtime, and task registry;
- no hand-authored route-forcing waypoints;
- start and goal must remain physically valid and visible enough to avoid a
  dark-final-goal confound;
- only the route segment should expose meaningful observation unreliability.

Acceptance:

- if C1 failure rate separates from C2, report it as secondary evidence;
- if both reach, report localization/safety margin instead;
- if the task requires special tuning, keep it exploratory.

## Paper Claim To Use

Use this as the main claim if the final campaign completes as currently
indicated:

> In a warehouse route-choice task, the learned-observability planner selected a
> longer route through better-observed floor space. Compared with the
> constant-observability baseline, it reached the same target with substantially
> lower localization error and a larger safety margin, at the cost of a longer
> path.

Avoid claiming:

- that C2 is always faster;
- that C1 cannot reach the goal;
- that the GP directly models physical occlusion geometry;
- that local tracker behavior is evidence of the GP contribution;
- that YOLO provides heading in the current final runtime.

## Immediate Next Actions

1. Generate final dashboards and summary tables from `paper_final_v1`.
2. Run the uniform-visibility sanity campaign with the same locked runtime.
3. Generate the cost-decomposition and perception/belief validation figures.
4. Clean the final config header so it no longer carries stale F-run notes.
5. Reconcile `active_research_state.md` and `paper_runtime_contract.yaml` once
   the artifact chain is complete.
