> Current execution status and measured results: [ICRA_STATUS.md](ICRA_STATUS.md).
> This earlier protocol is retained for its design rationale; manuscript work is paused.

# ICRA execution plan: commissioned external-camera information

Current as of 2026-09-04. This is the executable decision plan for the active paper. It
implements the dependency order in `PLAN.md`; it does not replace the metric contract or
promote historical fusion drives into current evidence.

The sequential identification and data-ownership contract is now specified in
[`SEQUENTIAL_CAMERA_PROTOCOL.md`](SEQUENTIAL_CAMERA_PROTOCOL.md), incorporating the latest
"Cascaded Kalman Filtering" discussion. Robot `Q` stays fixed; per-frame `R_NN`, optional
perception-filter output `R_cam`, and planner-facing `R_plan` have separate meanings.

## Primary hypothesis

> A fixed-camera model commissioned from actual localization outcomes can predict the
> information delivered by individual and overlapping cameras well enough for the existing
> belief-aware planner to rank routes by their realized localization value.

The contribution is the measured interface

```text
camera opportunity -> {usable or unavailable; z_i, R_hit,i if usable}
                   -> multi-camera information
                   -> future belief prediction
                   -> route choice
```

YOLO, the crop network, NIW estimation, Gaussian fusion, a GP, and the existing planner are
components. None is claimed as the paper contribution by itself.

## Rules that apply to every stage

1. Separate systematic conditional mean `b_i(s)`, conditional hit covariance `R_hit,i(s)`,
   availability `q_i(s)`, cross-camera covariance `C_ij`, and robot belief covariance `P`.
2. Report unconditional opportunity coverage beside conditional-on-use accuracy.
3. Split and resample by physical place, transect, trajectory, or seed as appropriate;
   camera frames are not independent replicates.
4. A runtime admission rule may use only current image, detected box, camera calibration,
   camera identity, and robot belief fields explicitly declared in advance. Ground-truth
   error and evaluation-only occlusion ratios may not enter it.
5. A successful development diagnostic freezes a candidate. It does not consume or replace
   the later independent test capture.
6. Distinguish the covariance attached to a current image-conditioned measurement from the
   quality the planner predicts before seeing that image:

   ```text
   R_meas,t = Cov(x_GT - z_t | current runtime image/box features)
   R_plan(s) = expected future quality before those image conditions are known
   ```

   `R_plan` is a planner-facing prediction and must not be described as literal per-frame EKF
   noise.
7. Because the simulated camera pipeline is deterministic at an exact state, define the
   randomness behind `q_i(s)` explicitly. If usability is deterministic given exact state,
   learn a support map and integrate it over future state/execution uncertainty instead of
   inventing Bernoulli randomness at a point.

## Gate 1: operational measurement mean

Candidate: the frozen mean-only 96 px paired-crop correction from experiment 25. It predicts
`(du,dv)` and then uses the unchanged calibrated ground projection. It has no sigma head and
does not use a temporal filter or robot prior.

### 1A — existing dense-line transfer test

Apply the retained validation-selected checkpoint to
`warehouse_v2_dense_lines_20260902`. This capture was not used to fit the correction, but it
has been inspected for other questions, so classify it as independent development evidence,
not a final test.

Report, for all target-defined detections and for partial views separately:

- raw, scalar-only, and paired-crop pixel error;
- ground-plane median, RMS, p90, p95, and fraction above 1 m;
- the same metrics by camera, physical line, range, and frame-edge status;
- opportunity coverage and retained coverage for every admission rule;
- paired differences with whole-line bootstrap intervals.

Evaluate two support conditions without tuning either on this capture:

1. exact image-boundary clipping refusal;
2. the previously proposed raw-backprojection range at or below 16 m, explicitly labelled a
   development candidate.

Advance the crop candidate only if it improves the line-clustered partial-view ground-plane
median and p90 without increasing the greater-than-1-m failure rate. Advance a support rule
only if the crop also improves RMS under that rule and retains at least half of partial-view
opportunities. Otherwise stop crop-model work and use the dense spatial/refusal branch.

### 1B — fresh bias/support capture

If 1A passes, freeze checkpoint hash, crop construction, scalar normalization, projection,
and admission rule. Then run the P0/P1 capture in
`experiments/camera_observation_characterization/NEXT_CAPTURE_PROTOCOL.md` on geometry-chosen
transects. The final held-out capture remains untouched.

Gate 1 passes only when the correction reduces heading-marginal camera-place mean error on
unseen places and controls the physical tail with a runtime-observable rule.

### Execution record — 2026-09-04

Gate 1 has passed as a **development freeze**, not as final-paper validation.

- The older independent dense-line transfer exposed two adjacent, repeatable camera-A
  extrapolation failures in the paired-crop output. The unrestricted paired model improved
  partial-view median/RMS/p90 but increased the greater-than-1-m count from one to two, so it
  did not pass the predeclared 1A tail rule.
- A post-hoc development refusal candidate was then defined without reading transfer errors:
  admit the paired model only when its predicted correction magnitude is at or below the
  original validation-set q99, `23.1778307 px`. On the old dense lines it retained 547/577
  partial detections and removed both paired-model greater-than-1-m failures. This result only
  justified a fresh test; it did not repair 1A retroactively.
- P0 was rerun through the final five-camera path. All 45/45 opportunities were captured in
  nine complete batches, maximum within-batch timestamp span was `0.0 s`, all hashes and
  source-batch identifiers were verified, and exact-state image and box repetitions were
  identical. Therefore `R_repeat = 0` in this static simulator condition.
- P1 was selected by a script that reads only declared driveable regions, frozen occluder
  geometry, and camera geometry. It used seven 1.2 m transects, 0.15 m spacing, all eight
  headings, and all five cameras: 504 synchronized batches and 2,520 camera opportunities.
  Capture integrity passed every timing, row-accounting, image, world, code, weight, and plan
  hash check. The detector produced 1,238 hits and 1,282 explicit misses.
- On 187 fresh partial-view detections, raw versus paired-crop ground error was respectively
  `35.61 -> 18.67 cm` median, `40.72 -> 27.65 cm` RMS, and `58.43 -> 42.25 cm` p90, with zero
  greater-than-1-m paired failures. The frozen q99 refusal retained 177/187 and gave
  `17.87 cm` median, `25.45 cm` RMS, and `39.74 cm` p90.
- The primary heading-marginal unit also passed. On 45 output-supported camera/places with at
  least one partial heading, raw versus paired-crop mean error was `24.29 -> 12.26 cm` median,
  `33.55 -> 22.09 cm` RMS, and `53.86 -> 34.73 cm` p90. Whole-transect bootstrap 95% intervals
  for raw-minus-corrected improvement were `[6.39, 20.97] cm`, `[7.68, 16.80] cm`, and
  `[10.80, 29.24] cm`, respectively.

Decision: freeze the paired-RGB checkpoint, crop/scalar construction, calibrated projection,
and q99 output-support rule. Drop the scalar-only correction from the operational candidate;
its fresh partial-view tail was materially worse. Keep raw projection as the required baseline.

P1 did **not** identify final field spacing. After the frozen correction, the deterministic
residual field at 0.15 m remained strongly correlated for all detections (correlation proxy
about 0.85--0.99 by camera), and four cameras did not reach the `exp(-1)` scale within the
1.2 m lines. The partial-view strata were too sparse for cameras B, D, and E. This is evidence
for longer and redundantly targeted P1b transects; it is not a covariance estimate and may not
be used as `R_meas`.

Next dependency-preserving run:

1. P1b: geometry-only 2.4--3.6 m transects with at least three predicted transition transects
   per camera; keep 0.15 m spacing and eight headings, then freeze spatial sampling distance.
2. P2: independently executed trajectories/passes at the frozen strata; estimate
   mean-corrected `R_meas`, temporal decorrelation/effective update rate, and simultaneous
   camera-pair covariance.
3. Only after P2, test fusion gain, then availability/`R_plan`, and finally route ranking.

The P1b plan has been generated at the conservative 3.6 m span and passed the same collision
filter used by capture: eight transects, 25 positions per transect, eight headings, all five
cameras, 1,600 synchronized batches / 8,000 camera opportunities. Its multicover selector
requires three distinct predicted sightline-transition transects per camera and three overlap
transects; it still consumes no detector residual or learned-model output. The plan is
`experiments/camera_observation_characterization/icra_p1b_geometry_poses.json`.

## Gate 2: conditional hit covariance

Do not enter this gate until Gate 1 passes or partial views have been declared unavailable.
Collect independently executed passes at predeclared states/strata, with all five cameras
attempting every `source_batch_id`. Static repeated renders may diagnose determinism but may
not set deployed covariance.

Fit the simplest adequate rung and stop:

```text
R0 global isotropic
R1 per-camera isotropic
R2 per-camera full 2x2
R3 camera + range/view-angle dependence
R4 spatial/heading dependence
```

Judge held-out likelihood/proper score, 50/90/95% containment, sharpness, and tail rate. NIW
is retained only when finite commissioning data materially improves held-out calibration; it
is not a novelty claim.

Also estimate residual correlation against elapsed time and travelled distance. Runtime and
planning may not count every camera frame as independent information when the mean-corrected
residual remains correlated. Start with an effective decorrelation time/distance or thinned
update rate; add temporal filtering only if it beats that simpler treatment.

## Gate 3: camera dependence and fusion

Estimate `C_ij` from simultaneous, pairwise-complete, mean-corrected residuals, resampling
whole independent passes. Compare:

- best single camera;
- independent block-diagonal fusion;
- a constant camera-pair correlation model;
- an oracle covariance using evaluation references, as an upper bound only.

The selected fusion model must predict both the realized fused error ellipse and the marginal
gain from adding each overlapping camera. If the independent model is as calibrated and sharp
as the correlated model, use it. Otherwise retain the smallest correlated model that earns
the improvement.

## Gate 4: availability and future information

Define one binary runtime-computable `usable` label for every camera opportunity. Fit
`q_i(x,y,theta)` and compare constant probability, geometry visibility, the prior confidence
proxy, and the commissioned model using held-out Brier score and calibration.

Construct the planner-facing information model using exactly the covariance/dependence model
selected in Gate 3. Do not let planning assume independent cameras if runtime fusion requires
correlation.

Compare deterministic expected-information prediction against Monte Carlo hit/miss (or
support-crossing) rollouts. Retain the deterministic approximation only if it preserves the
route ranking and relevant tail risk.

## Gate 5: navigation

Freeze the existing planner, dynamics, safety constraints, horizon, action costs, goal, and
belief propagation. Compare repeated paired seeds on routes with a real length-versus-sensing
trade-off:

1. shortest path/no camera-quality model;
2. prior IWAI detector-confidence model;
3. commissioned camera-information model;
4. optional simulation oracle.

Primary result: predicted route ranking versus realized belief/localization ranking. Secondary
results: task success, collision/contact, path length, dropped-correction fraction, longest
camera gap, belief median/p95 position error, NEES, and 95% coverage. Score camera readings,
fused corrections, and planner belief at their own timestamps and never combine those layers.

## Explicitly deferred

- temporal perception KF, unless a mean-corrected dense/trajectory test beats a commissioned
  static covariance in both accuracy and calibration;
- neural variance head;
- keypoint detector;
- spatial GP for every cross-covariance;
- GP epistemic uncertainty propagated through the planner;
- online covariance adaptation beyond the simplest truth-free baseline.

If a perception KF is later tested, its posterior may not be injected into the robot EKF as
a fresh independent measurement every frame. The experiment must pass new innovation only,
use a correlation-aware track-fusion rule, or otherwise demonstrate that information from
old frames is not counted repeatedly.

## Pivot conditions

- If paired crops cannot control the physical tail with a defensible support rule, partial
  views become unavailable or receive a dense commissioned mean model; do not tune a larger
  CNN without a diagnosed representation failure.
- If honest refusal removes so much coverage that overlap provides little usable information,
  pivot the paper toward characterizing limits of fixed-camera commissioning rather than
  claiming navigation improvement.
- If a commissioned overlap model cannot predict realized fusion gain, do not proceed to a
  planner claim based on that gain.
- If predicted route ranking does not transfer to repeated execution, the full navigation
  hypothesis is rejected even if individual camera calibration is good.
