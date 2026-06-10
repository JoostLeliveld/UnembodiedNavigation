# Active Research State

Last updated: 2026-06-10.

This is the honest current truth. The machine-readable runtime contract is
`docs/paper_runtime_contract.yaml` (v0.5); evidence-chain status is
`docs/experiment_registry.md`; dated history is `docs/decision_log.md`.

## Current Paper Position

The AWS-style warehouse line is the paper-facing candidate. It is final paper
evidence only when the complete artifact chain is present (fixed world+camera,
validated detector, fitted GP, locked config, complete seeded logs, figures from
those logs, wording matching the runtime). The claim axis is localization
safety / route observability, NOT deterministic goal-reaching superiority.

## Locked Setup (2026-06-10)

- World `warehouse_aws.world.sdf`; external camera locked at **z=4.8, y=-5.5**.
- GP **`aws_gp_v7`** (length_scale 0.90, beta 0.5, noise_var 0.05); detector
  **`aws_yolo_simseg_v2`**. v5/v6/v6b GPs and earlier cameras are superseded.
- Runtime: global long-horizon EFE (horizon 120, dt 0.25) for route choice +
  a shared **simple proportional local tracker** (`use_simple_local_controller:true`).
  Heading = **`camera_xy_only`** (odom dead-reckoning predict; pixel (x,y) updates
  couple to heading only via the prediction cross-covariance — no odom-yaw anchor,
  no keypoint/visual heading). No-go = hinged-log **`warning_band`** keep_in,
  weight 2000, band 0.05 (the old log_barrier / belief-nogo mechanism is retired).
- Conditions: **C1** constant_R_efe (no GP), **C2** visibility_aware_efe (GP modulates
  camera (x,y) covariance only — not heading). Optional **C3** GP-risk-only ablation.
- Main task **`F31_b1_apron_a3_mid`** (start (3.30,-1.00), goal (1.00,1.75)).
  `a0_west_to_a1_upper_blocked_mid` is the **saved secondary** line
  (config `aws_f86a_camera_xy_config.yaml`) for a future multi-task run.
- Active config: `scripts/visibility_comparison/aws_f31b1_final_config.yaml`.

## Active Hypothesis

Learned observation reliability should make C2 prefer a more-observable route (or
delay commitment / stop safely) where that keeps the belief localized, while the
constant-R baseline C1 may take a shorter visually-unreliable route and suffer
larger localization error or collision risk. The effect must emerge from
planner-facing covariance, shared driveability handling, horizon, the
condition-neutral goal-prior schedule, and optimizer basin handling — NOT from
mission waypoints, GP-dependent route seeds, or an oversized ambiguity weight.

## OPEN blocker (F88, unresolved)

On F31_b1 under this runtime, **the intended route-split does not yet emerge**:
across the goal-prior sweep both C1 and C2 optimize to the lower-sweep (visible)
route. Root cause from offline diagnosis: with `control_weight=0` and
`goal_progress_cost=0` the objective has **no path-length/effort term**, so C1 has
no incentive to prefer the shorter occluded mid/connector route — when both routes
reach, C1 falls back to the marginally-lower-no-go lower-sweep, same as C2. The
connector seam artifact (false keep-in violation at edge-touching driveable prisms)
was fixed by overlapping the connector prisms. The split question is therefore an
objective-design question (whether to add a condition-neutral effort term), not a
seam/visibility/optimizer bug. No closed-loop F31_b1 campaign is currently valid as
route-split evidence.

## Do NOT claim yet

- a closed-loop F31_b1 route-split (it is not present in any current log);
- C1 fails while C2 succeeds on F31_b1;
- AWS is final paper evidence;
- the GP directly models physical occlusion geometry;
- YOLO/camera provides heading;
- the simple local tracker is itself the visibility-aware contribution.

## Runtime details to keep paper-accurate

- Camera updates provide `(x, y)` via ground-plane projection; heading is
  odometry-backed dead-reckoning (camera_xy_only), not camera-derived.
- Projection/calibration uncertainty is not separately propagated into EKF/EFE covariance.
- The known 2D driveable / forbidden-zone layer (keep_in warning_band) is shared across conditions.
- Command and encoder noise stay ON for final AWS campaigns.
- Crash/contact ends a run and is tracked separately from stuck/timeout/goal-reached.
- Runtime means start after the first non-trivial command; launch/global-solve/warm-up are reported separately.

## Immediate next decisions

1. Resolve the F88 objective-design question (whether a condition-neutral
   path-length/effort term, or a task/geometry choice, yields the route-split honestly).
2. Only then run the seeded F31_b1 closed-loop campaign and regenerate figures/tables from those logs.
3. Decide whether AWS is the main result or an Experiment-B extension once the chain is complete.
