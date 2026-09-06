> **Current approved execution plan (2026-09-06):**
> [ICRA_STATUS.md — work map, thesis scope and ICRA direction](docs/ICRA_STATUS.md).
> The user has authorized the commissioning-to-fusion-to-navigation study, with the existing
> metric-reference NN retained. The 2026-09-06 user clarification sets a 12-page,
> two-column AIES thesis. The scope is IWAI extended to a camera network with a compact
> commissioning audit. Official AIES criteria are verified: appendices are not assessed.
> The exhaustive six-run camera-subset pilot is complete and remains diagnostic;
> `../papers/master_thesis/thesis.pdf` is a long-form source bank, not the required format.
> A separate ICRA paper remains conditional on evidence. The staged 2026-08-31 plan below
> is preserved as historical rationale; its "only active stage" statement is superseded.
> The IWAI network planner adapter, three fitted fields and short optimization probe are
> implemented. See the [code plan](experiments/icra_commissioning/planner_implementation_plan.md)
> and [paper map](../papers/master_thesis/planning/paper_map.md). Complete-route feasibility
> and live camera-model equivalence pass their scoped checks. Two integration pilots expose
> tracking/filter failures; the separately frozen corrected runtime is under live test.
> See [runtime_integrity_audit.md](docs/runtime_integrity_audit.md). Fusion/forecast
> equivalence and independent matched navigation effects remain required evidence.

# The paper: plan of record

Current as of 2026-08-31. This replaces the earlier split between a fusion paper, an
availability-planning paper, and a learned-correction paper.

## The paper in one sentence

We characterize how each fixed warehouse camera turns YOLO output into a robot-position
observation, separate conditional measurement noise from the probability of obtaining a
usable observation, combine the resulting per-camera information, and give that observation
model to an otherwise unchanged belief-aware planner.

The short conceptual sentence is:

> **Fusion constructs the observation model; the planner evaluates its consequence.**

The paper is not “a different objective function,” “a Gaussian process for `R`,” or
“precision-weighted fusion.” The contribution is the complete, measurable interface from
camera evidence to expected belief evolution:

```text
YOLO observation interpretation
        -> per-camera bias and conditional noise
        -> probability of a usable future observation
        -> per-camera expected information
        -> camera selection or fusion
        -> the existing planner's observation-dependent belief prediction
```

## What is active now

**Sensor characterization is the only active experimental stage.** We do not choose a bias
correction, an operational estimator for `R`, a fusion rule, or a planning treatment until
the warehouse-wide residual structure is visible.

All earlier fusion drives remain diagnostic or historical, and the previous generated deck
plots have been retired. There is still no frozen paper-facing fusion selection under
`docs/localization_metrics_registry.json`.

## The contribution, if the evidence supports it

1. **Per-camera observation characterization.** Show where each runtime-plausible YOLO
   observation method is biased, noisy, anisotropic, non-Gaussian, or unavailable instead of
   assuming uniform measurement quality.
2. **Observation interface and camera aggregation.** Separate the covariance of an
   observation that actually arrived from the probability that one will arrive, then turn
   the per-camera quantities into an effective observation-information model through camera
   selection or fusion.
3. **Planning evidence.** Substitute only that observation model into the existing IWAI
   belief-aware planner and test whether navigation outcomes change, while freezing the
   dynamics, horizon, safety constraints, action costs, goal objective, belief propagation,
   and tuning.

Fusion is part of contribution 2, but ordinary Gaussian precision addition is not claimed as
novel by itself. Learned bias correction is not presumed to be a contribution; it enters only
if the characterization stage shows a material, structured bias and a realistic calibration
method removes it on held-out data.

## Quantities that must never be collapsed into one `R`

For camera `i`, observation method `m`, state `s = (x,y)`, and heading `theta`, write

```text
z_i,m = h_i,m(x) + b_i,m(s, theta) + v_i,m,
v_i,m ~ N(0, R_hit,i,m(s, theta)).
```

The components have different meanings:

| quantity | meaning | where it may be used |
|---|---|---|
| `b_i,m(s,theta)` | systematic reading error | offline characterization; later realistic calibration only if justified |
| `R_hit,i,m` | random covariance conditional on a usable observation arriving | filter update for an actual observation |
| `q_i,m(s,theta)` | probability of a usable observation arriving | future-observation model for planning |
| `P` | robot belief covariance | filter and planner state; never measurement noise |
| `R_plan` / expected information | information the planner expects before knowing whether a future observation will arrive | planner prediction only |

A missed detection creates no filter update. It is not a fictitious measurement with a very
large covariance.

## Stage 1 — characterize every camera and observation method

### 1.1 Freeze one sensor output and vary only its interpretation

The observation source is fixed: **one frozen YOLO detector producing a bounding box**. The
capture and YOLO result are shared. We do not compare segmentation, keypoint, residual-network,
or shape-network detectors in this study.

What varies is how the same box is interpreted downstream:

| ID | handling of the same YOLO bounding box |
|---|---|
| `M0_raw_box_ipm` | take the bounding-box bottom-centre and back-project it to the floor |
| `M1_fixed_offset` | apply the existing fixed camera-ray offset to the raw result |
| `M2_analytic_hull` | compare the detected box with the projected robot hull used by the current manager |

Each map must therefore use the same captured image and the same YOLO box for all three
panels. Differences between maps are consequences of interpretation, not different sensor
inputs. Evaluation-only semantic masks and commanded ground-truth poses may define references
and explain occlusion, but never become method inputs.

### 1.2 Capture design

Use a controlled factorial design in two deliberately separate captures:

```text
field capture: drivable warehouse grid x five cameras x eight headings x one RGB image
repeat panel: predeclared state strata x the repetitions justified by a variance/power pilot
```

The field capture answers where errors and detector misses occur; the repeat panel answers
how readings vary when state is held fixed. Random poses may extend coverage but may not
replace the grid or repeated headings. The manifests must record the world, camera
calibration hashes, detector artifact, method registry, exact position IDs, heading IDs,
repetition IDs, and every failed detection. Split by position for validation; never
interleave neighbouring samples from the same grid cell across fit and evaluation sets.

The field capture is frozen at
`logs/perception_datasets/warehouse_v2_bbox_characterization_20260831`: 386 drivable floor
positions, eight headings, five cameras, 3,088 robot poses, and 15,440 attempted camera views,
with zero failed capture batches. The frozen YOLO artifact returns 6,412 boxes before any
post-detection admission gate. Every `M0`–`M2` interpretation is derived from the same selected
box and image hash. This is the primary source for the first conceptual figures.

This capture supports **observed and heading-marginalized error fields, box-return maps, and
pooled histograms**. It cannot identify a local conditional mean and covariance separately:
at a specific camera-position-heading cell, one residual is a sample, not a mean bias vector.
Therefore these figures are field characterization, followed by a separately frozen repeat
panel for conditional covariance.

Use a two-tier recapture rather than repeating every warehouse pose dozens of times:

1. a whole-warehouse grid with eight headings and a small fixed number of repeats, which
   establishes the field and misses;
2. a predeclared stratified repeat panel spanning cameras, range, viewing angle, occlusion,
   and image position, with enough repeats to estimate conditional covariance and tails.

Freeze both repeat counts after a pilot variance/power calculation, before looking at method
rankings. This keeps the design concrete without committing to an arbitrary image count now.

### 1.3 Residuals and units

Keep the image-space and ground-plane views together, but do not mix their covariances.

```text
e_uv = [u_YOLO - u_expected, v_YOLO - v_expected]       pixels
e_xy = p_method - p_GT                                   metres
```

Image-space residuals diagnose the detector and are the natural units for a pixel covariance.
Ground-plane residuals show the physical consequence and form the warehouse vector field.
Every plotted residual must name its method, camera, reference, admission rule, sample count,
and whether it is unconditional or conditional on a usable detection.

### 1.4 Required figures

For every camera-method pair, produce the same diagnostics from one manifest-bound table:

1. **Warehouse error vector field:** for the one-sample field capture, show observed arrows or
   a clearly labelled heading-median arrow; call it a mean bias vector only after repetitions
   exist. Background = support or usable-detection probability; camera pose and aim shown.
2. **Magnitude and spread map:** median or RMSE and a local covariance ellipse, kept distinct
   from the mean arrow.
3. **Residual histograms:** `e_u`, `e_v`, along-ray, and cross-ray residuals, with global and
   location-conditioned views visibly separated.
4. **Two-dimensional residual scatter and ellipse:** reveals correlation, anisotropy,
   multimodality, and outliers hidden by marginal histograms.
5. **Heading small multiples:** the vector field or signed residual for each of the eight
   headings; a pooled heading plot alone is insufficient.
6. **Mechanism plots:** residual versus range, viewing angle, image position, box truncation,
   and detector confidence.
7. **Detection/miss map:** every attempted capture counts, so detector reliability is not
   confused with accuracy conditional on a hit.
8. **Q-Q plot or equivalent tail diagnostic:** globally and within sufficiently populated
   conditional cells.

A pooled warehouse histogram is a mixture over position and heading. It may be heavy-tailed
even when the residual is approximately Gaussian conditional on state. Therefore the paper
must show both `p(e)` and `p(e | x,y,theta,camera,method)` and must not reject or accept a
Gaussian model from the pooled histogram alone.

### 1.5 The first decision gate: bias before covariance

For each camera-method pair estimate

```text
b_i,m(s,theta) = E[e_i,m | s,theta].
```

Then make one of four decisions:

| observed structure | consequence |
|---|---|
| smooth systematic field explained by calibration/projection | test a realistic camera, homography, affine, or surveyed-point recalibration |
| method- and heading-dependent field | repair or reject the observation interpretation; do not hide it in `R` |
| mean small relative to local spread, but spread changes with state | keep the method and move to conditional covariance modelling |
| multimodal or heavy-tailed even after conditioning | revise admission/outlier modelling; a single Gaussian `R` is not adequate |

A dense map built directly from Gazebo truth may characterize the available structure and
serve as an oracle bound. It may not become a hidden operational correction or reliability
input. Any correction advanced toward the main method must be something a real warehouse
could commission and must improve a spatially held-out evaluation.

## Stage 2 — choose the simplest conditional measurement covariance

Only after the bias gate, fit the covariance of measurements that actually arrive and pass
the frozen usability rule. Start at the simplest rung and stop when the next one does not
earn itself on held-out data:

```text
R0 = sigma^2 I
R1 = one isotropic covariance per camera
R2 = one full 2x2 covariance per camera
R3 = camera + range/viewing-angle dependence
R4 = spatial/heading-dependent covariance
```

Compare predicted and observed residual ellipses, held-out likelihood or another proper
score, 95% containment, and sharpness. Always report unconditional detection coverage beside
conditional accuracy so admission cannot make a method look good by dropping difficult
cases.

The offline ground-truth residual covariance is the evaluation reference, not an online
input.

## Stage 3 — test a truth-free operational estimate of `R_hit`

For a delivered measurement, log the pre-fit innovation and post-fit residual:

```text
nu_k = z_k - h(x_k^-)
mu_k = z_k - h(x_k^+).
```

Maintain histories per camera and method. Start with covariance matching or a sliding-window
pre/post-residual estimate; do not begin with a full variational Bayesian filter. The
innovation covariance also contains projected belief uncertainty, so `Var(nu)` may not be
called `R` without removing that term.

Validate the truth-free estimate against the Stage 2 empirical covariance on held-out runs.
Ground truth scores whether the estimator recovered a sensible covariance, but the estimator
itself may not read ground truth. A Bayesian inverse-Wishart treatment is an optional later
extension only if uncertainty in the covariance estimate matters to the main conclusion.

## Stage 4 — learn whether a usable future observation will arrive

For each camera separately learn

```text
q_i(s,theta) = P(usable observation from camera i | s,theta).
```

This is a probability-of-use map, not an error map and not `R`. Its labels must be computable
from operational detector, gate, and filter evidence without ground truth. Compare it on
held-out positions or routes against:

- a constant probability;
- geometry-only visibility;
- the prior confidence-based method;
- the proposed commissioned model.

Use a proper probability score such as Brier score plus calibration plots. The fitting tool
(GP or otherwise) is implementation detail; the claim is the calibrated spatial probability
of receiving a usable observation.

## Stage 5 — selection, fusion, and the planner-facing model

When an observation actually arrives, the filter uses that camera's `R_hit,i`. For planning,
the robot does not yet know which future observations will arrive. An initial expected-
information approximation is

```text
J_i(s) = q_i(s) H_i(s)^T R_hit,i(s)^-1 H_i(s).
```

The camera aggregation interface is explicit:

```text
{q_i, R_hit,i, H_i} per camera
        -> J_i
        -> A(J_1, ..., J_N)        selection or fusion
        -> effective expected observation information
        -> predicted posterior belief
        -> existing IWAI objective
```

For selection, `A` keeps the most informative valid camera. For conditionally independent
fusion, `A` sums information. Independence is a hypothesis to test, not an entitlement:
shared timing, heading, shape, calibration, and detector errors must be measured before
precision addition is treated as calibrated.

`R_plan = R_hit/q` is only an information-equivalent shorthand in the scalar/common-`H`
case. The state-information expression above is the general definition and avoids comparing
pixel covariances from different viewpoints as though they were in the same coordinates.

## Stage 6 — keep the planner fixed

The baseline and proposed conditions use the same planner and objective structure. Only the
future observation model supplied to belief propagation changes:

```text
baseline: J_IWAI(constant observation model)
method:   J_IWAI(state-dependent effective camera information)
```

Do not add a direct `-lambda * q(s)` visibility reward, a new route heuristic, camera-switching
costs, obstacle costs, a new uncertainty objective, or different tuning in the proposed arm.
If several of those change together, the camera model's planning consequence is no longer
identifiable.

The planned ablation ladder is:

| condition | observation assumption | question |
|---|---|---|
| `C0_constant` | one constant observation model | original uniform assumption |
| `C1_camera_constant` | calibrated constant `R_hit` per camera | is camera identity enough? |
| `C2_spatial_expected` | per-camera `q(s)` and conditional `R_hit` | does spatial reliability matter? |
| `C3_selection` | choose the most informative expected camera | is selection sufficient? |
| `C4_fusion` | combine justified per-camera information | does multi-camera information add value honestly? |

Do not run this ladder until Stages 1–5 have frozen the observation interface and the fusion
independence assumptions.

## Paper order

The paper follows the evidence in the order a reader needs it:

1. Problem: external cameras are neither uniformly available nor uniformly informative.
2. Observation methods: what a YOLO output is being treated as physically.
3. Characterization: warehouse vector fields, histograms, heading/range structure, and misses.
4. Bias decision: correction, rejection, or zero-mean approximation.
5. Conditional measurement model: `R_hit` for an observation that arrives.
6. Operational estimation: whether pre/post residuals recover `R_hit` without truth.
7. Availability model: per-camera `q(s,theta)` for a future observation.
8. Camera aggregation: selection/fusion in state-information space.
9. Planner integration: unchanged IWAI machinery, changed observation input only.
10. Experiments: characterization, covariance validation, fusion calibration, and planning
    outcomes, with limitations stated at the layer they affect.

## Evidence and reporting rules

The repository contracts remain in force:

- Studies before 2026-08-25 are superseded.
- A camera reading is scored at `obs_stamp`, a fused correction at `fused_stamp`, and a belief
  at `planner_belief_stamp`.
- Never compare reading error, fused-correction error, and belief error as though they were
  the same quantity.
- Count a physical detection once, aggregate within a drive first, and compare replicated
  drives or held-out positions—not logger ticks.
- Use exact, frozen manifests; never pool `latest` directories or arbitrary globs.
- `gt_*` is an offline reference only. It may characterize and score; it may not enter the
  online estimator, admission rule, reliability learner, planner, goal decision, or stuck
  decision.
- Report dropped-correction fraction and longest correction gap beside belief accuracy.
- Report calibration and sharpness together; a covariance can obtain coverage merely by
  becoming uninformatively large.

The executable run-alignment contract remains
`experiments/fusion_on_fixed_routes/aligned.py`; see `docs/localization_metrics.md` and
`docs/localization_metrics_registry.json` before reporting a run result.

## Immediate deliverable

The next meeting is not a fusion-results meeting. It is a conceptual measurement-model
meeting. The required deck and the decisions it must obtain are specified in
[`docs/NEXT_MEETING.md`](docs/NEXT_MEETING.md).
