# Active Research State

Last updated: 2026-06-12.

This is the current paper-facing truth. The machine-readable runtime contract is
`docs/paper_runtime_contract.yaml`; evidence-chain status is
`docs/experiment_registry.md`; dated history is `docs/decision_log.md`.

## Current Paper Position

The AWS-style warehouse line is now the paper-facing experiment line. The claim
axis is localization safety and route observability under a fixed external
camera, not faster travel time or deterministic goal-reaching superiority.

Paper-facing claims must be tied to the full artifact chain: locked world,
detector, GP artifact, campaign config, seeded logs, generated figures/metrics,
and matching paper wording.

## Locked Setup

- World: `warehouse_aws.world.sdf`; external camera at `(x=0.0, y=-5.5, z=4.8)`.
- Detector: `logs/perception_models/aws_yolo_simseg_v2/model.pt`.
- GP: `logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz`
  (`P_conservative_plan_map`, length scale `0.90`, noise variance `0.05`,
  beta `0.5`).
- Runtime config:
  `scripts/visibility_comparison/aws_f31b1_final_config.yaml`.
- Runtime planner: global long-horizon EFE (`global_horizon=120`, `dt=0.25`) for
  route choice plus a shared simple proportional local tracker
  (`use_simple_local_controller=true`).
- Heading mode: `camera_xy_only`. Odometry is used for dead-reckoning
  prediction; pixel `(x,y)` updates influence heading only through
  position-heading cross-covariance. There is no visual/keypoint heading update.
- Runtime detector point: bottom centre of the selected YOLO bounding box.
  `yolo_use_masks=false` in the locked runtime; masks are training/diagnostic
  artifacts only.
- No direct visibility reward: `visibility_weight=0.0`. The GP affects the
  planner-facing camera `(x,y)` observation covariance.
- Feasibility layer: shared known driveable/forbidden-zone keep-in no-go cost
  with `warning_band`, `nogo_weight=2000`, `nogo_safe_distance=0.25`,
  `nogo_warning_band=0.05`.
- Belief-tube no-go is active in the locked campaign:
  `use_belief_nogo_cost=true`, `nogo_belief_kappa=1.0`.

## Current Campaign Evidence

The robustness campaign uses four tasks with five seeds per condition:

- `F31_b1_apron_a3_mid`
- `b5_a4_apron_to_a2_mid`
- `b2_a0_west_to_a1_upper`
- `b6_a0_west_to_a1_low_control`

Main comparison:

- C1: `constant_R_efe` with spatially uniform detector-observation covariance.
- C2: `visibility_aware_efe` with GP-derived state-dependent
  detector-observation covariance.

Aggregate outcome:

- C2 reaches `18/20` runs cleanly and has `2/20` collisions.
- C1 reaches `12/20` runs cleanly, has one near-success, and has `7/20`
  collisions.

Per-task outcome:

| Task | C1 clean/collision | C2 clean/collision | Note |
| --- | ---: | ---: | --- |
| F31_b1 | 4/1 | 5/0 | discriminator |
| b5 | 3/2 | 5/0 | discriminator |
| b2 | 1/4 | 3/2 | hard west-side discriminator |
| b6 | 4/0 + 1 near | 5/0 | control |

Mechanism summary: C2 stays more observable in the discriminating tasks. The
reported observability aggregates show lower shadow exposure and higher
detection fraction for C2, while the constant-R baseline more often takes the
visually poor route and collides.

## Honest Caveats

- b2 remains hard: C2 improves the result (`3/5` clean vs `1/5`) but still
  collides in `2/5` seeds.
- Continuous localization metrics are pooled over clean successes only. C1's
  localization error can look deceptively low because failed runs are excluded.
- The route mechanism should be described as learned covariance plus belief-tube
  feasibility under shared driveability constraints, not as a direct visibility
  reward.
- The local tracker is shared execution plumbing; the scientific route choice is
  the global EFE solve.
- Projection/calibration uncertainty is not separately propagated into EKF/EFE
  covariance.

## Do Not Claim

- YOLO/camera provides heading.
- GP reliability is a physical traversability or obstacle map.
- The GP directly models all physical occlusion geometry.
- C2 always reaches, always has lower Euclidean localization error, or is faster.
- Route choice is forced by mission waypoints or condition-specific route seeds.
- Continuous localization metrics alone prove robustness.

## Runtime Details To Keep Paper-Accurate

- Camera updates provide `(x,y)` via ground-plane projection; heading is
  odometry-backed dead reckoning with indirect correction through covariance.
- Command and encoder noise stay on for final AWS campaigns.
- Crash/contact ends a run and is tracked separately from stuck/timeout/near
  goal.
- Runtime means start after the first non-trivial command; launch/global-solve
  time and estimator warm-up are not mixed into localization/control means.

## Immediate Codebase Cleanup Priorities

1. Keep `paper_runtime_contract.yaml`, `experiment_registry.md`,
   `paper_alignment.md`, and `PLANNER_HYPERPARAMETERS.md` aligned to the locked
   campaign values above.
2. Keep generated paper figures reproducible from `scripts/paper_figures` and
   `logs/paper_figures`.
3. Remove stale compact-benchmark wording from public-facing docs unless it is
   explicitly labeled archived.
4. Keep all paper wording in `thesis-report` inside the Experiments section; the
   Results section remains an empty compatibility stub.
