# Canonical thesis method lock

Status: active and authoritative from 2026-09-20.

This document is the single human-readable source of truth for the thesis method. It
supersedes earlier contribution locks, candidate protocols, availability studies,
Stage 07/08/09 manifests, commissioning plans, and method-scope notes wherever they
conflict with it. Those files may document development history, but they do not reopen a
method choice recorded here.

The thesis studies **camera-network modelling for belief-space robot navigation**. A
known-pose survey of an installed camera network is used to learn the measurement errors
and the information that the network supplies across the warehouse. The resulting model
is used consistently in runtime localization and in belief prediction during planning.
Detector-specific bias correction is necessary supporting calibration, not a separate
headline contribution.

## Two work tracks that must not be conflated

### Track A: finish the complete thesis draft now

The present development captures, fitted models, residuals, covariance results, planner
runs, and navigation runs may be used to complete the structure of the thesis and obtain
feedback on the full argument. The manuscript describes the locked method in completed-
study form so that supervisors can review the intended final paper rather than an outline.

Current numerical results and derived figures are draft-population artifacts. They are
used to expose missing analyses, weak explanations, inconsistent tables, and problems in
the experiment design. They must remain reproducible and must not be silently mixed with
later final-campaign artifacts. The Results chapter contains the single internal
`TODOJOOST` notice identifying the replacement boundary; the rest of the paper uses normal
paper wording.

Track A does **not** authorize method drift. A convenient old artifact must not reintroduce
a rejected correction, covariance, availability model, fusion rule, map, planner objective,
or navigation condition.

### Track B: replace the evidence with the improved final campaign

After the new reference-position capture is complete, the final pipeline is rerun from the
beginning: validate the capture, freeze its partitions, retrain the correction model,
produce out-of-sample residuals, refit all three covariance models, export each covariance
model's matched planning precision, rerun runtime fusion and
navigation evaluation, and regenerate every
paper number, table, and figure that depends on those artifacts.

The final campaign changes the **data and fitted parameters**, not the scientific method.
If the detector checkpoint is retrained or replaced, it must be frozen before the
reference-position pipeline is run and its identity must be recorded in the capture and
artifact manifests. Final-audit data may evaluate frozen choices but may not select or tune
them.

Only final Track-B artifacts may support the final submitted numerical claims. Track-A
artifacts remain development evidence and are replaced rather than pooled with the final
campaign.

## Locked end-to-end method

### 1. Environment and reference-position data collection

- Freeze the warehouse geometry, site boundary, camera poses and calibration, robot
  geometry, detector checkpoint, and fixed sensor gate before fitting the final
  models.
- Collect repeated camera opportunities at accurately known static ground-plane positions
  and declared headings. Retain an entry for every expected camera, including detector
  misses and gate refusals.
- Store camera identity, capture position and heading, timestamps, raw detector output,
  bounding-box features, admission outcome, projection, reference position, and the
  declared visibility representation.
- Partition by complete reference position. All repetitions, headings, and cameras for one
  physical position remain in one partition. Individual frames from a position may not be
  split across fitting, development, and final-audit partitions.
- Repetitions must not make one surveyed position dominate fitting; fitting and reporting
  use position-balanced weighting or aggregation where required.
- Ground truth is offline fitting and evaluation data only. It never enters admission,
  runtime correction, covariance querying, fusion, the estimator, planning, or stopping.

### 2. Detector, admission, and raw observation

- The deployed detector is the frozen YOLO11n blue-robot detector.
- Admission is deterministic and belief-independent. It may inspect only declared detector
  and image/box validity fields; it may not inspect ground truth, the state estimate,
  innovation, NIS, localization error, or route preference.
- The raw metric observation is the ground-plane homography projection of the raw bounding-
  box bottom centre.
- Visual hulls, hull-equivalent positions, masks used as replacement observations,
  belief-projected boxes, CAD silhouettes, and ground-truth-derived runtime features are
  prohibited.

### 3. Bias correction

- Deploy one correction network shared by all cameras so that the main weights are shared.
- Inputs comprise the locked structured runtime box/camera features, camera identity, and
  the declared 16 by 16 visibility-residual representation.
- The network predicts the metric correction along and across the camera-to-observation
  ray. The corrected observation is the raw projection plus this correction.
- The correction target is the reference position minus the raw projected observation.
- The visibility branch is part of the deployed correction model; raw and structured-only
  variants are diagnostic ablations, not alternative final methods.
- Freeze the correction before fitting covariance. Covariance fitting receives only
  corrected residuals from positions not used to fit the corresponding correction
  prediction, using disjoint position partitions or position-level cross-fitting.

The final implementation record must copy the exact feature vector, normalization,
visibility construction, architecture, activations, loss, optimizer, training schedule,
seed handling, and checkpoint-selection rule into the methodology or appendix.

### 4. Runtime residual covariance

The covariance evaluation and fixed-event localization replay additionally include one
external uncertainty-aware baseline: `Rproj`. It uses one pooled scalar
`sigma_px > 0`, fitted with equal aggregate weight per physical fitting position, and
propagates `sigma_px^2 I2` through the numerical Jacobian of the camera-specific
pixel-to-ground mapping at the detected box bottom centre. It uses the same correction,
admission decisions and independent information-fusion rule as `R0`--`R2`. It is not a
fourth navigation model and does not enter the intact/removal factorial.

The active comparison contains exactly three full two-dimensional covariance models, all
fitted to the same out-of-sample residual population from the frozen correction:

- `R0`: one global full covariance;
- `R1`: one full covariance for each camera;
- `R2`: one spatial full covariance field for each camera, locally estimated with a
  broad covariance prior where admitted-residual support is weak.

All three covariance models are fitted as full matrices in the signed
camera-to-query ray frame and are made positive definite using the locked
regularization/eigenvalue floor. `R0` is one constant ray-frame matrix and `R1_i` is one
constant ray-frame matrix per camera. `R2_i(p)` is a spatial ray-frame field per camera.
At every runtime or planning query, the selected matrix is rotated into the common
`map_bev` frame by the along-ray/across-ray basis at the query position. Thus fusion and
the EKF receive world-frame covariances without discarding the physically meaningful
radial/lateral error structure.

The correction basis is formed from the camera to the raw projection. The covariance
basis is formed from the camera to the corrected observation at runtime, and from the
camera to the predicted position during planning. Covariance-fit residuals use this same
camera-to-corrected-observation basis. Hyperparameters of `R2` are chosen only on
fitting/development positions and then frozen. Observations at one physical position are
first reduced to a position-balanced ray-frame second moment. `R2` selects neighbouring
physical positions, not neighbouring frames.

`R2` represents the residual conditional on an admitted measurement. Its posterior mean is
the sum of the kernel-weighted local second-moment scatter and the prior scatter, divided
by the effective kernel support plus the prior strength. The prior covariance is
`100 I m2`, which corresponds to a 10 m marginal standard deviation, and its strength is
`2.5e-6`. The selected spatial model uses 16 neighbouring physical positions and a
0.4 m Gaussian length scale. As local support vanishes, the runtime covariance approaches
the broad prior instead of the camera-average `R1` covariance.

`R0`, `R1`, and `R2` are compared for likelihood, calibration, coverage, sharpness, and
downstream fusion behaviour. Each is also carried into navigation together with its
corresponding planner-information representation. A planner may never use `R0` at runtime
while receiving spatial information learned from `R2`, or otherwise mix model levels.

### 5. Runtime multi-camera fusion and belief update

For each sufficiently synchronized admitted camera batch and model level `m`:

1. project and correct each camera observation independently;
2. query the corresponding runtime covariance `R_m` and express observation and covariance in the
   common world frame;
3. apply the declared synchronization and gross-disagreement rules;
4. fuse accepted camera observations once using independent information-form fusion;
5. apply the estimator NIS gate to the fused update; and
6. perform one EKF update of the robot belief.

The fusion assumption is conditional independence after correction. Simultaneous
cross-camera residual correlation is measured and reported as a diagnostic and limitation;
it does not change the primary fusion rule. A camera measurement is consumed once. Cascaded
filters, posterior reuse as a new measurement, generalized least-squares fusion, and an
augmented persistent-bias state are outside the active method.

### 6. Planner-facing covariance

No separate availability or planner-information model is fitted. For camera `i`, covariance
level `m` and planning position `p`, define

```text
Lambda_m,i(p) = inverse(R_m,i^run(p)).
```

The ray-frame covariance is queried and rotated into `map_bev` at the same camera and
position used by the planner. The resulting precision is exported on the planner grid.
Detector opportunities, misses, gate refusals and NIS outcomes do not enter this export.
They affect runtime when an observation is processed. Weak residual support already reduces
the R2 precision because the covariance approaches its broad prior. No second spatial
support penalty is applied.

At a predicted future position, planning uses the representation belonging to that arm and
sums only active cameras:

```text
M0: Lambda_network(p) = sum_{i in C_active} inverse(R_0,i^run(p))
M1: Lambda_network(p) = sum_{i in C_active} inverse(R_1,i^run(p))
M2: Lambda_network(p) = sum_{i in C_active} inverse(R_2,i^run(p))

inverse(P_plus) = inverse(P_minus) + H' Lambda_network(p) H.
```

Planning does not invent or fuse future measurement values. Runtime fusion combines realized
observations using the matching `R_m`. Planning applies the precision of that same `R_m`.

For `M2`, the spatial field is averaged over the predicted XY belief using the locked
positive-weight five-point sigma rule before it enters the information update. A mean-only
field lookup is not the canonical method.

Under removal, `M0` can only reduce information uniformly through the active-camera count.
`M1` can remove the failed camera's constant contribution but cannot localize where the loss
occurs. `M2` removes that camera's spatial field and can therefore represent which warehouse
regions lost information. Giving `M0` or `M1` the `M2` spatial field invalidates the
comparison.

### 7. Motion prediction and global belief-space planning

- Use the existing IWAI expected-free-energy planner with the locked unicycle prediction,
  process-noise model, fixed information update, one-sided risk term, anchored
  ambiguity term, and uncertainty-aware no-go cost.
- For ambiguity only, convert the summed active-camera information into the equivalent
  covariance `R_amb = (Lambda_network + epsilon_amb I)^-1`, with the fixed shared
  regularizer `epsilon_amb = 1.0 m^-2`. This regularizer is independent of `Q`, gives
  `R_amb = 1.0 m^2 I` at zero camera information, and does not enter the belief update.
  The final analysis must report a sensitivity check showing whether plausible changes to
  `epsilon_amb` alter candidate-route rankings.
- Compare variable-duration global routes by their active-discount-normalized
  IWAI stage cost: divide the sum of running risk, ambiguity, control and soft
  no-go terms by the common active discounted weight. Use running risk (not
  terminal-only risk), effective risk multiplier 1.0, ambiguity multiplier 1.0,
  discount 0.995 and goal-annealing power 0.9. The preferred position standard
  deviation ends at 0.10 m; the terminal arrival tolerance remains 0.35 m. The
  discount uses the same numerical value as the IWAI setting. Because the present
  rollout uses 1 s rather than 0.4 s steps, this is weaker discounting per unit time.
- The process covariance is identical in the estimator and planner and fixed before the
  final navigation campaign. Estimating `Q` is not a contribution of this thesis.
- Goal-prior annealing, terminal arrival constraint, optimization horizon, control blocking,
  route initializations, solver budget, and failure policy are shared by all camera-network
  conditions.
- Geometric feasibility is independent of camera availability or information. Camera
  information may change route cost and predicted belief, never whether the same geometric
  candidate exists.
- The global route is optimized once before execution for the relevant task, active planner
  camera set, initial belief, and frozen configuration. It is then handed to the local
  follower. The expensive global solve does not run concurrently in a way that blocks
  estimator or command processing.

### 8. Non-traversable regions and safety

- The authoritative free space is the site boundary minus the declared warehouse collision
  geometry, including the locked geometric safety margin. It is not reconstructed as the
  union of old lane polygons.
- The robot is an oriented rectangle of 0.80 by 0.55 metres.
- A hard swept-footprint check rejects geometrically invalid route segments and unsafe
  commands.
- A separate soft, uncertainty-aware no-go cost discourages routes whose predicted belief
  places the robot footprint near non-traversable space.
- Hard geometric rejection and the soft belief-dependent planning cost must be implemented,
  reported, and tested separately.

### 9. Local route execution

- Resample the selected global route into the declared local waypoint spacing.
- Use the frozen `ff_fb` current-belief segment-feedback waypoint follower. It combines
  route-tangent feedforward with bounded heading and cross-track feedback, continuously
  reduces speed under lateral departure, previews retained corners for braking, and applies
  arrival braking on the final route segment. Collinear densification points are not treated
  as control targets, and crossed intermediate segments are advanced rather than chased.
- The `ff_fb` follower queries neither camera quality nor belief covariance, and it performs
  no local EFE optimization. It is not described as local MPC or as a learned controller.
- Final and draft campaign configurations must set `local_controller_type: ff_fb` explicitly;
  they may not inherit the stale `turn_then_go` launch default.
- Declare waypoint acceptance, final-goal acceptance, yaw gate, velocity laws, command rate,
  timeout, and recovery/failure behaviour in the final campaign configuration.
- Validate the swept rectangular footprint for every command immediately before publication.
  The local follower may not bypass the global safety representation.

### 10. Camera-removal navigation experiment

The primary experiment crosses three matched camera-network models with two physical
network states:

| model | covariance used by runtime fusion | corresponding planning model |
|---|---|---|
| `M0_global` | `R0`, global full covariance | inverse of the rotated `R0` covariance |
| `M1_per_camera` | `R1_i`, full covariance per camera | inverse of the rotated camera covariance |
| `M2_spatial` | `R2_i(p)`, spatial full covariance per camera | inverse of the rotated spatial covariance |

The two network states are:

- `intact`: all five cameras are active in runtime fusion and in the corresponding planner
  representation;
- `removal`: the task-relevant camera is absent from runtime fusion and removed from the
  corresponding planner representation. The blind-corridor task removes camera C; the
  lane-08-to-lane-12 task removes camera B.

This produces six arms. The scientific comparison is whether increasingly expressive,
consistently used camera-network models let the planner respond more appropriately to the
same physical camera removal. `M0` has no spatial knowledge of the affected region, `M1`
knows which camera contribution disappeared but not where it mattered, and `M2` represents
the spatial loss directly.

The detector, correction, EKF structure, dynamics, process noise, EFE terms, candidate
generation, feasibility, optimizer, local follower, tasks, and matched seeds are identical
across arms. Only the matched runtime covariance, its corresponding planner-information
representation, and the active camera set change. With two focused tasks and five matched
seeds, the primary matrix contains 3 models by 2 network states by 2 tasks by 5 seeds, for
60 retained outcomes. The two tasks were retained because they provide distinct feasible
routes and expose a task-relevant loss of camera information; superseded tasks are not part
of the canonical campaign.

### 11. Evaluation and leakage control

- Correction: metric residual error and bias, with raw and structured-only ablations.
- Covariance: held-out log likelihood, containment/coverage, calibration, ellipse area or
  sharpness, and per-camera/spatial diagnostics.
- Fusion and filtering: fused error, NIS/consistency, update availability, and belief error
  at matched timestamps.
- Planning: predicted information and covariance along routes, route changes, and removal-
  affected regions.
- Navigation: success, terminal error, belief error, uncertainty, path length, duration,
  minimum clearance, safety interventions, and failure category.
- Use matched task/seed comparisons and predeclare confidence intervals and invalid-run
  handling before the final campaign.
- Final-audit positions and navigation outcomes fit or select nothing. Every final artifact
  records its input identities, split manifest, configuration, seed, and hashes.

### Reporting rules

Moved here unchanged from the former `docs/localization_metrics.md` (2026-09-24); its
registry is replaced by the campaign manifest and `pipeline/dataset_lock.json`.

#### Keep the three layers separate

1. A camera reading is scored against the reference at its capture timestamp.
2. A fused camera correction is scored against the reference at its fusion timestamp.
3. The robot belief is scored against the reference at the belief timestamp.

Never compare statistics across these layers or call them all “localization error.”

For every reported value, state:

- the layer and quantity;
- the statistic and units;
- the reference and timestamp convention;
- the exact manifest-bound run set;
- the number of complete drives and, where relevant, observations;
- the aggregation order.

#### Required statistics

- Position error is Euclidean distance, stored in metres and reported in centimetres.
- Name median, mean, RMSE, and 95th percentile explicitly; they are not interchangeable.
- Report uncertainty consistency beside accuracy. For planar NEES, the Gaussian reference
  mean is 2, the median is `2 ln 2`, and the nominal 95% ellipse uses 5.991.
- Report covariance sharpness or ellipse area beside coverage.
- Report heading error separately.
- Report the fraction of camera corrections dropped or rejected and the longest interval
  without an accepted correction.

#### Event identity and aggregation

- A physical camera frame may contribute at most once.
- Detector batches, individual camera frames, fused corrections, estimator decisions, and
  beliefs retain their source identities and integer timestamps.
- Every published correction has exactly one terminal estimator classification.
- Valid terminal classifications are `accepted`, `accepted_bootstrap`, `rejected`, and
  `dropped`; any unaccounted, duplicate, or contradictory event invalidates the run.
- A reasoned refusal is an outcome, not automatic run invalidation.
- Frames within one drive are correlated. Compute each drive's statistic first, then compare
  matched drives or seeds.

#### Ground-truth firewall

Ground truth may construct offline correction targets and score completed results. It may not
enter the sensor gate, runtime correction, covariance query, camera fusion, estimator,
planner, controller, goal decision, collision decision, or stopping rule.

## Controlled implementation closures

The method above is frozen. The following are not permission to choose another method; they
are exact implementation details that must be resolved, recorded, and then copied into the
paper and final configuration before Track B is executed:

1. the runtime coordinate used to query spatial `R2` (it must be runtime-available and may
   not be ground truth);
2. the exact static-position capture manifest, repetitions, headings, partitions, and world/
   calibration hashes;
3. the final `R2` kernel, support, prior, centring, and PSD-floor constants;
4. the exact planner-grid resolution, covariance query, ray-frame rotation, matrix inversion,
   and PSD enforcement;
5. whether one predeclared scalar covariance calibration is fitted on development data;
6. the verified numerical process-noise values;
7. synchronization, disagreement, fusion-batch, and EKF update rates;
8. final planner and `ff_fb` numerical parameters; and
9. confidence-interval, infrastructure-invalid-run, and failure-classification rules.

No future chat may resolve one of these from an old config merely because it already runs.
It must compare current code, paper wording, and development evidence, record the explicit
   decision in the final campaign configuration, and keep all six experimental arms
   otherwise matched.

## Explicitly superseded alternatives

The following are development history and may not return to the central method without an
explicit scope amendment from the author:

- visual-hull or hull/ridge correction;
- raw-only, per-camera, RGB crop, or joint Gaussian correction as the deployed model;
- R2C width-tertile, range-stratified, image-conditioned, neural-Cholesky, GP, hierarchical,
  or joint multi-camera covariance as an active covariance condition;
- separate Q0/Q1/Q2 availability models or any planner term of the form `q_i(p) R_i^-1`;
- uncertainty inflation of runtime `R` solely because general survey support is weak;
- correlation-aware GLS fusion, cascade fusion, or an augmented bias state;
- legacy pixel charts, availability-based route eligibility, or different candidate routes
  and feasibility rules between camera conditions;
- lane-union/keep-in geometry as the warehouse authority;
- circular-body safety checks in place of the swept rectangular model;
- an online global solve that blocks belief or command processing; and
- `turn_then_go`, `turn_then_go_recovery`, hysteresis/damping, or pure pursuit as the final
  campaign follower, and describing `ff_fb` as MPC or as a learned controller.

## Authority and change control

For method questions, read sources in this order:

1. this canonical lock;
2. the final campaign configuration and its signed/hashed manifest, once created;
3. the implementation actually referenced by that manifest;
4. the manuscript;
5. older contracts, protocols, configs, results, and archived experiments only as history.

If a lower-ranked source conflicts with a higher-ranked source, it is wrong for the current
thesis. Existing code is not evidence that an obsolete choice remains active.

Changing a locked method requires an explicit author decision and a dated amendment stating
the scientific reason, affected artifacts, and required reruns. Completing a controlled
implementation closure does not require reopening the method, but the decision and evidence
must be recorded before the final campaign.

## Amendment 2026-09-23/24: final dataset, gate, fits, execution and campaign

Author decisions of 2026-09-23 and 2026-09-24. Where this amendment and an earlier section
differ, this amendment holds. It amends §1 (data), §2 (gate), §4 (R2 constants), §9
(execution and safety), §10 (campaign) and §11 (collision outcome). Every number below is
read from the named evidence file; the paper quotes the regenerated artifacts, not this text.

### A. Reference dataset (v8)

- **Composition.** All v5 reference positions, plus 12 camera-C supplement positions and a
  uniform top-up of every 1 m cell of the operating domain below the target density
  K / (pi (2 l)^2) = 8 positions per m^2 (K = 16, l = 0.4 m, scaled by the fraction of the cell
  where the 0.80 x 0.55 m footprint fits at every heading). Each added position takes the
  role of its nearest v5 position; none borders final_audit, so the audit set is the v5 one.
  Rule: `pipeline/capture/plan_topup.py`.
- **Repair.** In v5 capture session 71daa8ab the robot vanished from the simulator at pose
  4214; the 4,182 later poses were empty frames marked ok. They were re-captured in two
  passes (`captures/v8/repair`, `repair2`); the loader drops the broken rows and refuses any
  robot-absent run.
- **Positions inside objects.** The top-up planner and the capture pose check read the
  collision scene without the world's five included loose objects (forklift, two pallets,
  bin, pallet jack), so 46 poses at 12 top-up positions put the robot inside an object.
  Those 12 positions are dropped (author decision; not re-captured). The loader derives the
  drop from the world geometry and the frozen footprint, and the dataset audit fails if any
  pose is inside an object. Collision scenes are built only through
  `unav_common.occlusion_geometry.profile_collision_scene`.
- **Result.** 2,619 positions (D_mu 1,328, D_R 704, D_dev 437, final_audit 150) and 52,380
  camera opportunities, read only through `pipeline/dataset.py`, locked in
  `pipeline/dataset_lock.json` (every capture pass hashed) and audited by
  `pipeline/audit_dataset.py` (evidence: `logs/thesis/evidence/dataset_audit.json`).
- **Density weighting in reporting.** Dense cells are not thinned. Metrics are reported
  position-balanced and with each cell capped at 8 positions per m^2 of weight, whatever
  their direction.

### B. Detector and sensor gate

- The frozen YOLO11n checkpoint stays (`logs/thesis/detector`); it is not retrained.
- **Gate (author decision).** `config/sensor_gate.yaml` admits a box when its confidence is
  at least 0.25 and its bottom centre projects to the ground; there is no edge or size
  check. Four gates were refitted on identical inference. The rule pre-declared for that
  test preferred the gate with edge and size checks, because the extra boxes it refuses
  (bottom-cut or small) carry a heavier error tail (corrected p95 17 cm against 12 cm).
  The author chose coverage: poses seen by at least two cameras rise from 64.5% to 70.9%,
  and corrected D_dev RMSE on the commonly admitted boxes falls from 5.33 to 5.14 cm. Both
  facts are reported (evidence: `logs/thesis/evidence/gate_variants/`, measured on
  2026-09-23 before the 12-position drop). Runtime navigation uses the same gate.

### C. Correction and covariance (amends §4)

- **Deterministic fits.** The visibility-residual correction trains on CPU with deterministic
  algorithms; seeded GPU training was not reproducible and changed downstream selections.
  The network has one definition for training and runtime
  (`reliability.visibility_residual_net`).
- **One covariance family.** R0/R1/R2 are the inverse-Wishart posterior-mean covariances of
  §4, fitted on D_R by `pipeline/fit_covariance.py`. No second family is fitted; D_dev scores
  describe exactly the models the runtime fuses with and the planner inverts.
- **R2 constants are fixed, not selected:** 16 neighbouring positions and a 0.4 m Gaussian
  length scale. This replaces the sentence of §4 that R2 hyperparameters are chosen on
  fitting/development positions. The candidate grid is reported on D_dev as a sensitivity
  table only.
- `R_proj` is evaluated on final_audit for covariance and fusion. It is not a navigation arm.

### D. Simulation, collision and stopping (amends §9 and §11)

- **World.** `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf` is the world; the former
  generator no longer reproduced it and is retired. `world/warehouse_v2.py` describes its
  zones and cameras for planning and figures, and a test requires every collision box to lie
  inside a declared zone.
- **No contact sensing.** The world's static contact sensors and the robot's contact sensor
  are removed. Nothing that renders changes: images are identical to the capture world
  (capture world sha256 25fc40d3eb57...).
- **Collision** is the robot footprint (0.80 x 0.55 m) at its true pose leaving the driveable
  region: outside the world profile's site boundary, or overlapping any collision object, at
  zero margin. It is scored offline by `pipeline/score_collisions.py` from the per-run
  `ground_truth_pose.csv` (every true-pose sample; consecutive samples are checked with a
  certified sweep). Ground truth never stops a run: a run ends at the goal or at its
  simulated-time limit.
- **Stopping and success** (author's decision of 2026-09-22 for any rerun). A run stops when
  the belief holds within 0.10 m of the goal for 2 s (or, fallback, holds still within
  0.20 m); success is scored offline on ground truth: final goal distance below 0.30 m, no
  collision, evidence complete, and the run not ended as `stuck`, `goal_loiter_timeout` or
  timeout. The two radii are never equal, so a terminal belief error cannot turn a stop into
  a failure. A belief that stays within 0.20 m for 15 s without either hold ends the run as
  `goal_loiter_timeout` (stuck detection is off there). The time limit is 90 simulated
  seconds after the first command. These replace the arrival tolerance in §9.
- **Local execution** follows an admitted global route without re-vetoing its geometry from
  the drifting belief (the last bullet of §9 no longer holds): a re-veto turned localisation
  error into zero-command deadlocks, and the experiment must expose a collision rather than
  hide it.

### E. Execution (amends §9)

- **Lockstep.** Campaign and check runs step Gazebo in fixed blocks: 1 ms physics, 100
  iterations per 0.1 s control step, a camera frame every second control step (5 Hz), and
  every frame consumed before the next step. Real-time runs are not used.
- **Follower corner speed.** `ff_fb` limits speed while removing a heading error e to
  v <= w_max * 0.10 m / (1 - cos e), so the lateral swing stays within the 0.10 m geometric
  safety margin. Without it the follower swung about 0.5 m wide at 90-degree corners.
- **Executable routes only.** Every declared route candidate and every solved route is
  replayed through `ff_fb` offline with the campaign settings and must keep the footprint
  inside the driveable region before the campaign. Two candidates no follower can take
  (a 90-degree turn 0.35 m from a rack face) were removed from `pipeline/tasks.yaml`.

### F. Campaign (replaces the task paragraph of §10)

Five tasks, each with one declared dropped camera: A, B, C, E and E
(`pipeline/tasks.yaml`); the three covariance models, intact and removal; three matched seeds
(91500, 91501, 91502). That is 90 runs, executed seed by seed, each with exactly one outcome.
The design is locked by test.

### G. Rejected alternatives

- Thinning dense cells and retraining the detector on the thinned set: it raised D_dev
  correction error on identical positions (evidence:
  `logs/thesis/evidence/archive_rejected_20260923/README.md`).
- The pre-declared gate rule's choice (edge and size checks): see B.
- Contact-based collision outcomes and real-time campaign execution: see D and E.

### H. Reruns required

Detector inference, gate, correction, R0/R1/R2, planning precision, final audit, route
solving, the campaign, and every paper number, table and figure that depends on them.

## Amendment 2026-09-24 (afternoon): spatially balanced split, dataset v9

Decided by the author after the first campaign, when the v8 data map was reviewed.

- **Why.** In v8 the southern cross-aisle (y < -5.5 m, a quarter of the site's depth) held
  47 % of all positions and 59 % of the final audit, and camera C had 14 audit positions
  against 90 to 120 for the others. Every held-out number therefore described mostly one
  region. The density cap of amendment A does not change this: no evaluation cell exceeds
  it (the audit and D_dev reach at most 6 positions per m^2), so it reweights nothing.
- **Rule** (`pipeline/capture/plan_rebalance.py`, seeded, declared before capture, nothing
  chosen from errors). Strata are 2 x 2 m cells over the robot-valid area (0.1 m grid
  points where the footprint clears every object by the capture body clearance at every
  heading); cells under 0.5 m^2 join their nearest stratum (84 strata, 167.5 m^2). The
  final audit (150), D_dev (437) and D_R (704) are split over strata in proportion to area
  (largest remainder), the same totals as v8. Inside a stratum the captured positions are
  ordered by sha256(seed, key): D_dev first, then D_R, the rest D_mu. No captured position
  is dropped (thinning was rejected, amendment G); every stratum already held its D_dev and
  D_R quota, so no fill capture was needed.
- **Fresh audit.** The v8 audit was opened on 2026-09-24, so no existing image can be a
  sealed audit again. The v9 audit is 150 new positions drawn uniformly inside each
  stratum, at least 0.15 m from every captured and detector-training position and from
  each other (0.15 m is the capture grid spacing and the median distance of the v8 audit to
  its nearest training position). The 150 v8 audit positions join the pool
  (84 D_mu, 41 D_R, 25 D_dev).
- **Unchanged.** Detector, gate, correction and covariance methods, fixed R2 constants,
  planner, follower, tasks, seeds, goal rule and collision definition. The capture world is
  the current world file; its images are checked pixel-identical to the v8 capture world on
  re-captured v8 poses before the dataset audit accepts it
  (`captures/v9/world_equivalence/`).
- **Reruns required.** The whole chain of amendment H, from detector inference to the
  campaign, analysis, figures and drop-ins. The v8 results are superseded, not reported
  as final; their analysis outputs are kept as evidence of the change.

## Amendment 2026-09-24 (evening): calibrated odometry, Q evidence, belief bookkeeping

Decided by the author after the offline replay of the v8 campaign.

- **Encoder calibration (simulator, not estimator).** The simulated encoder no longer has a
  mean scale error: `encoder_noise_linear_slip_mean` 0.02 -> 0.0, set explicitly in both
  campaign templates. Real wheel odometry is scale-calibrated; no textbook odometry filter
  models an uncalibrated bias. The random, AR(1)-correlated encoder slip and additive noise
  stay, and so does the command (actuation) noise including its mean, which belongs to the
  plant and which the encoders measure. Why: under white process noise the 2 % bias made the
  along-track error exceed the predicted covariance after long camera gaps, and the NIS gate
  then refused correct camera updates for several seconds (lockouts).
- **Q unchanged, declared and conservative.** `process_noise_xy` 0.02 and
  `process_noise_theta` 0.08 stay; Q is neither derived from the simulated noise nor
  estimated. Evidence that it is conservative, reported with the results: starting from the
  true pose with zero covariance and predicting on odometry alone, the true error lies inside
  the 95 % ellipse in 0.98-0.99 of windows (position) and 1.00 (heading) at 1, 3 and 10 s,
  with calibrated odometry. A Q computed from the declared encoder parameters alone
  under-covers, so odometry error also arises outside the declared noise model; a "matched"
  Q would have to be estimated on separate drives, which is not part of this method. Camera
  covariance calibration is scored per observation against ground truth, where Q does not
  enter; at belief level a larger Q softens rejections equally in every arm.
- **Heading.** The heading follows odometry and camera updates act on position only
  (`heading_update_mode: camera_xy_only`, as run). A coupled update is not used: without a
  model of time-correlated camera errors it moves the heading on camera noise.
- **Unchanged.** Camera rate 5 Hz, gate, NIS threshold, fusion, R0/R1/R2, planner,
  follower, tasks, seeds, goal rule and collision definition.
- **Implementation closures (no method change).** (1) The belief anchor is committed a fixed
  lag (2 x the camera freshness limit) behind the clock, so the odometry replay no longer
  grows with a camera gap; before, the planner fell behind its own odometry in long gaps and
  its belief went invalid (7 of 90 v8 runs, all global or per-camera dropout failures).
  (2) Lockstep also waits until the planner has handled each step's odometry. The EKF
  equations are unchanged.
- **Reruns required.** The campaign and everything downstream of it (analysis, figures,
  drop-ins). Route solving uses no odometry noise and is unaffected.

## Amendment 2026-09-24 (night): targeted capture v10 (lanes and hard views)

Decided by the author, frozen before any v10 image is captured.

- **Why.** Runtime camera errors are larger than the static audit predicts because the robot
  drives lanes at aisle headings, a pose population the capture covers poorly (about a
  quarter of route pose cells had a training image within 0.25 m and 15 deg), and the
  correction's tail is concentrated in hard views. A focused learning curve on v9 (all easy
  D_mu kept, hard examples subsampled; known answer: 100 % reproduces the frozen v9
  correction) showed that edge-cut views keep improving with more data, while the
  bottom-occluded tail does not. No model, feature, gate or covariance method changes.
- **Rule** (`pipeline/capture/plan_hard_views.py`; campaign routes are never read). Views are
  classified geometrically (robot box 0.80 x 0.55 x 0.35 m; edge: box within 20 px of the
  image border; bottom hidden: ray to the footprint centre at 5 cm blocked; small: box under
  30 px). (1) Lane grid: 0.5 m grid over the driveable region at the four aisle headings,
  body clearance >= 0.05 m, minus poses already captured within 0.25 m and 15 deg.
  (2) Spaced hard-view top-up: 0.25 m grid, 8 headings, >= 0.15 m from every sealed-audit
  position; greedy by edge views toward +75 % edge views per camera, never duplicating an
  existing or lane pose, top-up positions >= 0.5 m apart, <= 4 headings per position. The
  spacing leaves the target partly unmet (edge views +44 to +54 % per camera): there are not
  enough distinct edge positions to go further without clustering. Result: 2260 lane poses
  + 1521 top-up poses at 1081 positions (`logs/thesis/captures/v10/capture_poses_v10.json`,
  manifest `capture_plan_manifest.json`).
- **Validation first.** 40 poses (24 top-up, 16 lane, seeded) are captured separately and
  checked by `pipeline/capture/check_v10_validation.py` (completeness; planned vs measured
  view class: edge recall >= 0.90, bottom-hidden recall >= 0.60; known answer on the v9
  audit: 1.00 and 0.93). The full capture runs only if it passes. Validation images do not
  enter the dataset.
- **Roles.** The sealed v9 audit is unchanged and stays sealed. New positions are assigned to
  D_dev, D_R and D_mu inside the v9 2 x 2 m strata, keeping each stratum's working-set role
  proportions (largest remainder), ordered by sha256(seed, key) as in the v9 rule. No position
  is chosen or moved on errors.
- **Unchanged.** Detector (frozen, not retrained), gate, correction and covariance methods,
  fixed R2 constants, planner, follower, tasks, seeds, goal rule and collision definition.
- **Reruns required.** The whole refit chain (the correction is retrained by `refit.sh`),
  the sealed audit, routes, campaign, analysis and every downstream figure and drop-in.
- **Validation 1 result (2026-09-24, 23:35) and decision.** The 40-pose validation capture was
  complete (200/200 rows). The geometric view classes were not reliable enough to select the
  hard-view top-up: of 46 views planned as edge only 11 were edge views in the images, 26
  showed no robot at all (in frame geometrically but fully hidden), and bottom-hidden recall
  was 0.58 (n = 12). A revised rule (a view counts only if a point of the robot box is in
  frame and in line of sight) reached edge precision 0.50 but lowered bottom-hidden recall to
  0.30, so it was not adopted. Decision: capture the **lane grid only** now (2260 poses;
  chosen from driveable space and existing coverage alone, independent of the view classes).
  The hard-view top-up is redesigned from **measured** view classes (segmentation masks of the
  existing captures) and frozen by a further dated note before it is captured.
- **Fusion median gate removed (author, 2026-09-24 night).** `manager_fusion_disagreement_gate_m`
  is 0 in both campaign templates, which the camera manager reads as "no median gate": every
  admitted camera in a synchronous batch is fused in information form. The former 0.6 m gate
  was never derived and did not change a fusion decision in the v8 campaign. Outliers remain
  subject to the estimator's NIS gate on the fused measurement.
- **Declared initial prior instead of a camera bootstrap (author, 2026-09-24 night).** The
  belief starts at the task's declared start pose, never ground truth:
  m0 = [x_start, y_start, theta_start], P0 = diag(0.10^2, 0.10^2, (15 deg)^2), committed at
  the first accepted odometry stamp (`initial_belief_from_task_start: true` in both campaign
  templates; `initial_belief_sigma_xy_m` 0.10, `initial_belief_sigma_theta_rad` 15 deg). The
  robot therefore starts without any camera. The first camera batch is an ordinary NIS-gated
  update; there is no `accepted_bootstrap` event and no two-camera quorum
  (`manager_use_task_start_as_bootstrap_prior` and `manager_bootstrap_prior_counts_as_support`
  are false; the camera manager treats the belief as anchored from the start). The mission's
  `initial_belief_max_sigma_m` (0.3 m) is met by the prior, so planning starts at once. With the
  flag off the former camera bootstrap is unchanged.

## Amendment 2026-09-25: task-visibility rule, task C goal moved

- **Rule (author, 2026-09-25).** In the removal condition the task's start and goal must each be
  seen by at least one active camera in at least 90% of the captured static views within 0.75 m
  (a view counts as seen when the robot has semantic pixels in it; dataset v10 via
  `pipeline/dataset.py`). A task that fails the rule tests a blind finish, not the camera model.
- **Check.** All starts and goals pass except the task C goal (-3.05, 8.625), seen under removal
  of C only by camera D, in about half the views.
- **Change.** The task C goal moves to the first point on its own lane (x = -3.05, scanned
  south in 0.1 m steps) that passes: (-3.05, 5.6). The immediate-east and lower-connector seeds
  keep their shape; the northern crossing now uses the y = 6.35 lane, so it is no longer equal in
  length to the other two. The contrast the task tests is kept: the northern route drives
  through a region that only C covers, the other two do not.
- **Evidence kept.** The superseded task C configs and routes are in
  `logs/thesis/superseded_taskC_blind_goal_20260925/`, and the old task C runs are moved there
  as evidence of the blind-finish failure mechanism. Task C alone is rerun (6 conditions x 3
  seeds); the other four tasks' runs are unchanged.

## Amendment 2026-09-25 (afternoon): one dataset, even density (v11)

Decided by the author after the v10 audit was opened.

- **Why.** In v10 the roles came from different captures and different densities: the whole
  final audit was one separate capture, and the correction fit held 12 positions per m^2 in
  the southern apron against 3.5 to 5 in the aisles. On D_dev the corrected error and the R2
  calibration are worst in the narrow aisles between the racks, where the correction fit is
  thinnest. A development set and an audit drawn differently from the fit sets do not
  describe the same data.
- **Design** (`pipeline/capture/plan_fill_v11.py`, `pipeline/capture/partition_v11.py`,
  seed 20260925). One dataset: every role is an even sample of the same valid area and only
  its density differs. Covariance fit 8 / m^2, derived from R2 (K / (pi (2 l_R)^2), K = 16,
  l_R = 0.4 m); correction fit 8 / m^2 (the same resolution, a choice); development 2 / m^2;
  test (final audit) 1 / m^2; 19 / m^2 in total over 167.5 m^2 of valid area.
- **Valid positions.** Four captured headings, and the 0.80 x 0.55 m footprint clears the
  collision scene by the capture body clearance at every heading inside the site (the v8
  top-up rule). 223 captured positions fail this (109 with two headings) and are `excluded`.
- **Fill capture.** Every v9 2 x 2 m stratum below ceil(19 x valid area) is filled with new
  positions at valid 0.1 m sub-grid points farthest from every existing position, never
  closer than 0.15 m (473 positions, four headings each). 25 strata stay one to four short
  because no point is 0.15 m clear; they fall short in D_mu.
- **Roles.** Inside a stratum the valid positions are ordered by sha256(seed, key),
  regardless of capture, and take area-proportional quotas in the order test, D_dev, D_R,
  D_mu. Positions beyond the quotas, mostly in the dense apron, are `unused`: they enter no fit
  and no evaluation. Thinning by density was rejected in amendment G to keep data; it is
  accepted here because an even sample is the purpose of this amendment.
- **Not a sealed audit.** The v10 audit has been opened and some of its images are now
  working data, so the v11 test set is a held-out split of one dataset, not a fresh capture.
  It is still used once and never for selection.
- **Unchanged.** Detector (its training positions are outside the pool), gate, correction and
  covariance methods, fixed R2 constants, planner, tasks, seeds, goal rule and collision.
- **Reruns required.** The whole chain from detector inference to the campaign, analysis,
  figures and drop-ins. The v10 fits and campaign are superseded and kept as evidence.

## Amendment 2026-09-26: B/C dropout swap, goals moved by the task-visibility rule

- **Swap (author).** Task B removes camera C and task C removes camera B. Run in lockstep, like
  every other campaign run (a first rerun without lockstep is kept in
  `logs/thesis/revisions/bc_dropout_swap/superseded_nonlockstep_20260926/`).
- **Rule re-checked after the swap** with `pipeline/check_task_visibility.py` (the rule of the
  2026-09-25 amendment, now a script; `pipeline/campaign_configs.py` refuses a config whose
  start or goal fails it). Under the swap both goals failed: task B (0.975, 8.625) seen by E in
  51% of views, task C (-3.05, 5.6) by C in 52%. The first swap campaign therefore tested blind
  finishes; it is kept as evidence in `.../superseded_blindgoal_20260926/`.
- **Change.** Each goal moves to the first point on its own lane that passes, scanned from the
  old goal in 0.1 m steps: task B to (0.975, 7.25) (D, 93%), task C to (-3.05, 6.35) (C, 90%).
  Route seeds keep their shape and end at the new goal. Tasks B and C are rerun in all six
  conditions x three seeds into `logs/thesis/revisions/bc_dropout_swap/`;
  `pipeline/analyze_campaign.py` takes every B and C entry from there. Tasks A and both E
  tasks are unchanged.

## Amendment 2026-09-27: task C redesigned by search

- **Why.** With camera B removed, the old task C (and every start tried along its bottom lane)
  was decided by the risk term for every model, not by camera information: the planner's
  objective is the arrival-gated mean of the step costs, so a covered detour that spends many
  steps near the goal wins on risk. No model changed route. (The thesis equation now states this
  mean; the old seed `northern_crossing` also crossed a rack and was fixed first.)
- **Search** (`pipeline/search_dropout_tasks.py`, result
  `logs/thesis/revisions/bc_dropout_swap/task_search_camera_B.json`). Lane-grid start/goal pairs
  that pass the task-visibility rule without B; routes up to 1.4x the shortest; the shortest
  blind (spatial field < 10 m^-2) for >= 3 m and an alternative with no blind stretch; each scored
  as a fixed rollout with the planner's own objective. Kept: every model and condition picks the
  same route except spatial after the dropout. 9 of 702 pairs; the two best confirmed with the
  full solver.
- **Choice.** Start (-3.05, 8.625) facing south, goal (0.975, -5.75), seeds `middle_aisle` and
  `north_east_lanes`, equal length (18.4 m), start and goal seen by C / A in 100% of views.
  Solved routes: middle_aisle for five conditions, north_east_lanes for spatial_removal.

## Amendment 2026-09-27: one heading variance, encoder from true motion, Q set from its noise

- **Finding.** Split by time since the last accepted camera update, the belief was far too
  wide in camera-free stretches while pooled coverage looked calibrated
  (`pipeline/process_noise/replay_heading_model.py` reproduces the logged belief).
- **Estimator bug (code).** In camera_xy_only the committed record kept the initial heading
  prior plus accumulated Q while the read-out reported only the odometry-heading variance;
  prediction turned the record value into cross-track uncertainty. The record now holds the
  same odometry-heading variance (`_odometry_heading_variance_or`). Test:
  `test_camera_xy_only_commit_uses_the_odometry_heading_variance`.
- **Simulated odometry.** The DiffDrive odometry already carried Gazebo's own wheel-floor slip
  (measured, undescribed by any parameter). The encoder now starts from true body velocity and
  adds only the declared noise, plus systematic wheel errors from UMBmark (TRC LabMate:
  D_R/D_L 1.00121, wheelbase 337.2/340 mm). The realised drift of the previous setting was
  measured at 0.98 % of distance and 0.76 deg per 90 deg turned (median).
- **Q.** Set, not fitted: the white-noise equivalent of that encoder noise in the IWAI
  continuous-time derivation, with input-dependent PSDs (PROCESS_NOISE.md). Planner and
  estimator use the same Q; the planner predicts the estimator's belief, whose growth is the
  encoder noise.
- **Heading.** Heading from odometry; camera updates correct position. Coupled heading was
  tested offline on the campaign runs: better inside camera coverage, worse after it ends,
  because the camera error drifts with the view as the robot nears a coverage edge (all three
  covariance models). A look-ahead variant (heading corrected only if >= 4 of the next 5
  frames follow) removed most of that damage but stayed behind odometry heading. Re-evaluate
  on runs with the new wheel errors.
  Replayed again with the encoder-model Q (`--encoder-q`): coupling improves heading while
  cameras see the robot but still roughly doubles the median cross-track error after coverage
  ends; tails are close for the look-ahead variant. Absolute coverage there is not meaningful
  (old-world odometry against new-world Q).
- **Gate.** The tighter belief rejects more camera batches, almost all right after an update
  and mostly measurements far outside their own R (`gate_analysis.py`), largest near camera C.
- **Reruns required.** Routes re-solved, full campaign. Thesis text: appendix noise table and
  the "ideal odometry" sentence, the Q appendix line, the initial heading prior and the
  heading statement in the estimator section.
