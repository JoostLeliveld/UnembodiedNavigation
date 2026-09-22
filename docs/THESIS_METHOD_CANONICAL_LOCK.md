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
