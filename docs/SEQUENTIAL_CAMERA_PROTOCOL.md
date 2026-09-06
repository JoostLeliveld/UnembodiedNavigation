> Current execution status and measured results: [ICRA_STATUS.md](ICRA_STATUS.md).
> This earlier protocol is retained for its design rationale; manuscript work is paused.

# Sequential camera identification protocol

Current as of 2026-09-04. This supplements `ICRA_EXECUTION_PLAN.md` with the latest
"Cascaded Kalman Filtering" discussion. It declares dependencies, not new experimental
results. The localization metrics contract and exact manifest selection remain mandatory.

## Freeze the quantities and their provenance

- `Q_robot`: the user's existing process-noise estimate. Record the exact artifact, parameter
  values, units, time scaling, dynamics code hash, and applicable motion regime. Validate it;
  do not estimate a replacement from the camera-calibration data.
- `R_NN`: conditional uncertainty of the current-frame corrected observation, before any
  temporal filter. Declare whether coordinates are pixels or metres.
- `Q_p`: process uncertainty of an optional perception tracker. This is a different state
  and dynamics model from the robot's, so it is not automatically `Q_robot`.
- `R_cam`: uncertainty and temporal dependence of the complete frozen camera output supplied
  to robot localization. If no temporal tracker is used, this can coincide with the projected
  current-frame uncertainty; otherwise it requires its own validation and fusion semantics.
- `R_plan`: future sensor quality before the future image is observed. It remains distinct
  from current-image uncertainty and must use the same dependence assumptions as execution.

The existing `experiments/gate0_process_noise/validate_q.py` is **exploratory only**. It
selects drives by glob, pools overlapping windows across drives, adds a heading term informed
by observed residuals, and sums step covariance without the full `F P F^T` propagation.
Its output cannot establish the fixed-Q gate. Do not copy its alternative parameters into
runtime or use its numbers to size camera R. A replacement must use a frozen run manifest,
the production prediction recursion, and aggregation within each independent drive first.

## Data ownership

Assign complete executions and held-out physical regions before inspecting their results.
These roles may share a collection campaign, but never an execution or adjacent-frame split.

| Role | Permitted use | Must remain frozen while scoring |
|---|---|---|
| S: existing static captures, P0/P1/P1b | Mean correction, projection/support development, deterministic spatial structure | No sequential R or temporal-independence claim |
| D0: camera-free prediction validation | Validate the supplied Q and initial covariance | Q, dynamics, timing/alignment convention |
| D1: sequential development | Residual-mean probe, provisional R_NN, support decisions, optional Q_p and tracker decision | Record every development choice |
| D2: camera covariance fit | Fit positive-definite R_NN or final R_cam by innovation likelihood | Mean model, support rule, pipeline, Q_robot |
| D3: camera covariance validation | Innovation mean, coverage, proper score, tails, temporal correlation; GT cross-check | All D2 parameters |
| D4: fusion fit | Estimate simultaneous camera dependence, if necessary | Per-camera pipeline and covariance |
| D5: fusion validation | Compare single camera, independent fusion and justified correlated fusion | All D4 parameters |
| M: spatial commissioning | Fit availability/support and R_plan; validate on held-out regions | Complete observation/fusion interface |
| F: final navigation test | Paired independent executions and realized route ranking | Entire method, controller and planner |

If D3 or D5 informs a change, mark the inspected data as development and reserve fresh
validation executions. F remains untouched until the full pipeline is frozen. Exact run IDs,
seeds, region assignments, expected artifacts and hashes belong in the selection manifest;
directory names and modification times never select evidence.

## Identification with fixed Q

Start with NN-only measurements. Use short, predeclared odometry-only prediction windows,
with a stated initial mean, covariance and coordinate frame. No camera update enters a
window before the scored observation. A GT initialization is permitted only as an explicitly
labelled offline oracle diagnostic, not as evidence that the method operates without truth.
An operational identification claim needs an available initialization with quantified error.

At capture time, predict with the production recursion:

```text
P_minus = F P_previous F^T + Q_d
innovation = z - h(x_minus)
S = H P_minus H^T + R
loss = logdet(S) + innovation^T solve(S, innovation)
```

Parameterize R as positive definite and fit the likelihood on D2. Q alone does not determine
P_minus: the initialization, elapsed time, dynamics Jacobians and any earlier information
also matter. Do not fit R by subtracting a fixed odometry covariance from pooled residuals.

The additive formula for S requires negligible cross-covariance between prediction error
and camera measurement error. Check this assumption, timestamps, conditional mean and
linearization before interpreting a likelihood optimum as camera uncertainty. Normal causal
filtering may use previous camera observations when its correlations are correctly modeled;
the short camera-free windows are a clean identification baseline, not a claim that every
camera-informed prior is intrinsically invalid.

For multiple cameras, innovations share the same uncertain robot prior. Consequently,
cross-covariance of camera innovations is not by itself camera-noise cross-covariance.
Separate `H_i P_minus H_j^T` from it, or use synchronized GT-aligned camera residuals for
the explicitly offline dependence diagnostic. Do not count common prior error twice.

## Gates before adding complexity

1. Validate the frozen Q on D0 with full covariance propagation and a declared initial-state
   convention. If it fails, stop innovation-based R identification and diagnose the model;
   do not let R absorb process-model error.
2. On D1, test whether signed residuals remain predictable from runtime-available geometry
   and camera/image features on held-out places. Persistent predictable mean calls for
   correction or refusal, not larger covariance. Global cancellation is insufficient.
3. Freeze geometric support, including projection conditioning. GT visibility may stratify
   offline evaluation but cannot choose a runtime covariance or admission decision unless
   a separately validated runtime classifier supplies that label.
4. Fit the simplest adequate R: global, per-camera, then runtime-observable conditioning.
   Report conditional mean, shape/tails, proper score, sharpness and 50/90/95% containment.
5. Preserve original dense driving sequences for temporal tests. Measure elapsed-time and
   distance dependence; select any effective update rate on development data and verify it
   on held-out executions. Static spatial correlation is not temporal whiteness.
6. Test a perception KF only after R_NN exists. Compare NN-only and NN+KF on whole held-out
   sequences, including lag, first acquisition under partial occlusion, clear/partial
   transitions and recovery after misses. Keep it only for demonstrated benefit.
7. A retained tracker needs an explicit correlated-information interface. Marginal R_cam
   calibration alone does not license repeated independent assimilation of its posteriors.
   Passing a current-frame likelihood while using the tracker for auxiliary decisions is
   a possible baseline; any such decisions must also be included in validation.
8. Validate simultaneous fusion before commissioning its planner-facing information field.

Each gate needs its acceptance criteria and minimum independent-run support written into
the campaign manifest before scoring. This document deliberately assigns no numerical pass
thresholds or claims about the uncollected D0--F datasets.

## Driving capture requirements

Include clean views, persistent partial views, visibility transitions, camera overlap and
handover, projection-sensitive geometry, and camera gaps. Repeat passes with predeclared
realistic variations in start pose, heading, lateral offset and supported execution seeds.
Do not add image noise solely to manufacture covariance.

Save raw RGB, exact camera stamps and batch identities, all attempted opportunities including
misses, detector outputs, corrected pixels, projections and Jacobians/support flags, GT and
odometry with their own stamps, controls, prior/posterior states and covariances, innovations,
and assimilated camera identities. Preserve raw ordering for replay. Fit/evaluate through
the repository alignment contract; do not reuse logger-held messages as new detections.

## Interrupted P1b capture

The exact capture is `logs/perception_datasets/warehouse_v2_icra_p1b_geometry_20260904`.
The interrupted index contains 1,203 complete batches / 6,015 opportunities out of 8,000;
397 pose batches remain. A lossless PNG-to-WebP converter started in the earlier thread
is still working as of this handoff. It verifies decoded hashes and updates index paths
only at completion. Do not start a second converter or resume capture against that index
while conversion is active.

After the converter exits, run the read-only preflight:

```bash
python3 experiments/camera_observation_characterization/audit_capture_resume.py \
  --capture logs/perception_datasets/warehouse_v2_icra_p1b_geometry_20260904 \
  --output /tmp/icra_p1b_resume_preflight.json
```

It checks complete, contiguous camera batches, actual capture-stamp coherence, frozen input
and capture-code hashes, every unique decoded image, and space for the remaining PNG writes.
Space estimates assume no future image deduplication. The estimate is conservative relative
to observed sizes but cannot guarantee a hard bound on future image complexity. The original
capture writes PNG for new images even when retained ones have been converted to WebP.

Only after a passing preflight should the original isolated transport and unchanged capture
command resume with `--resume`. Then run the frozen detector, interpretation and integrity
audit before estimating spatial spacing. P1b remains a spatial development diagnostic; it
does not identify deployed sequential covariance.
