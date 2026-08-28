# The paper: plan of record

Locked 2026-08-25. This is what the work is for. Anything not serving one of the five
sentences below belongs in a study README or an appendix, not in the paper and not in a
status report.

---

## The five sentences we are trying to earn

1. **The fixed infrastructure-camera detector provides usable robot observations reliably
   enough to support localization experiments.**
2. **A commissioned spatial availability model predicts usable camera observations on
   held-out states better than uniform and geometry-only assumptions.**
3. **Detector confidence [does / does not] provide additional predictive information about
   localization measurement quality once observation geometry is accounted for.**
4. **Calibrated external-camera observations improve localization over onboard estimation
   alone without producing overconfident fusion.**
5. **Planning with predicted camera support reduces localization degradation and long
   observation outages relative to uniform, geometry-based and previous
   confidence-based planning, at a modest navigation-cost increase.**

Sentence 3 is deliberately written with both outcomes available. Either is a result.

---

## The idea in plain terms

A robot's wheels drift, so it cannot trust its own sense of position for long. The
warehouse has fixed cameras that can see it and tell it where it is — but not everywhere,
and not equally well. Racks block views, distance blurs the measurement, and some corners
no camera reaches.

So the robot should choose its route knowing **where the cameras can help it**, instead of
assuming they always can.

> **Availability determines whether a correction is likely to arrive.
> The measurement covariance determines how much to trust it once it has.**

```
RUNTIME    an observation is received  ->  position + its covariance  ->  fusion

PLANNING   a place we have not reached ->  how likely a sighting is,
                                           and how good it would be
                                       ->  expected future uncertainty
```

Two separate things have to be commissioned for each camera:

- **How often will this camera give me a usable sighting here?** — a probability, known
  *before* driving there, which is what makes route planning possible.
- **How accurate is a sighting when I do get one?** — a covariance, used when a
  measurement actually arrives.

Keeping those apart is the conceptual core. The planner asks the first question about
places it has not been yet. The filter asks the second about measurements it already has.

---

## Contribution versus engineering

Be explicit about this, in the paper and in every conversation about it.

**The contribution:** modelling where in a warehouse the camera network can actually
support localization, and planning routes using that model.

**Supporting engineering, not contributions:** the camera-and-detector sensor, the bias
correction, the covariance calibration, standard fusion, and whatever interpolates the
availability map.

Practical consequence: **do not spend ten minutes defending the bias model.** It is one
sentence and a backup figure. The same goes for the covariance ladder, the gate thresholds
and the choice of interpolator. If a supporting module needs a long defence, that is a
signal it has been over-built, not that it deserves more slides.

## Status

| stage | state | where |
|---|---|---|
| Train the detector | **done, frozen** | `logs/perception_models/..._halfopen_20260825_r1`, sha `efff1949…` |
| Commission the sensor | **done** | `experiments/measurement_commissioning/` |
| Freeze what "usable sighting" means | **mostly** — false positives unmeasured | `observation.py: GATE` |
| Remove the systematic offset | **done** — 0.48 cm → 0.19 cm | `logs/studies/measurement_commissioning/calibration.json` |
| Estimate measurement covariance | **partial** — one number works; ladder not re-run on the frozen pipeline | — |
| Test whether confidence predicts quality | **not started** — data is already on disk | — |
| Learn the availability map | **not started** — needs its own truth-free dataset, not the commissioning capture | — |
| Validate availability on held-out routes | **needs driving** | — |
| Validate fusion on one fixed route | **done, one drive per arm** — six arms all reached the goal; the fusion rule moves honesty, not accuracy | `logs/studies/fusion_on_fixed_routes/RESULTS.md` |
| Run the planner comparison | **needs everything above** | — |

---

## Two datasets, and why they can never be one

The two things commissioning produces have opposite requirements. Keeping them apart is a
design decision, not bookkeeping.

| | **bias correction** | **availability map** |
|---|---|---|
| needs ground truth? | **yes, unavoidably** — the offset *is* reading minus truth | **no** — "did a usable sighting arrive?" is answerable at runtime |
| how much data | ~70 surveyed spots | 80–250+ positions, and still improving at 250 |
| how it is collected | once, deliberately, robot placed on marked spots | continuously, during ordinary operation |
| can it be updated later? | only by repeating the survey | yes, it grows as the robot works |

So the expensive dataset is the one that needs no truth, and the one that needs truth is
small. That is a good deployment story — but it only holds if the two are never merged.

**And they are not interchangeable, measured.** The admission check compares a detection
against a box predicted from the robot's pose. During commissioning that pose is exact; at
runtime it is the robot's own estimate. Re-judging the same detections with a deliberately
wrong pose (`availability_robustness.py`):

| pose error | usable rate | wrongly kept | wrongly dropped |
|---|---|---|---|
| none | 0.699 | — | — |
| 10 cm / 2° | 0.561 | 0.9 % | 14.6 % |
| 25 cm / 5° | 0.420 | 1.3 % | 29.2 % |

**A truth-free availability map measures lower availability than a truth-based one** — 0.56
against 0.70 at a realistic belief quality. Commissioning availability with ground truth and
then deploying would leave the planner a map that is optimistic by a fifth.

Two consequences worth stating in the paper:

- **The gate fails safe.** Wrongly-kept stays near 1 % while wrongly-dropped climbs to 29 %.
  Under pose error it discards good sightings rather than admitting bad ones — a bad sighting
  corrupts the estimate, a lost one only costs availability.
- **Availability is not purely a property of the building.** It depends on how well localized
  the robot already is: better belief → the predicted box matches better → more sightings
  admitted → better belief. So the map must be learned at the belief quality the robot will
  actually have, which is another reason it comes from operational data rather than a
  truth-based survey.

## The data, and the rule that keeps it honest

**Detector training data ≠ commissioning data ≠ final planning data.** Nothing may serve
two of those roles.

| dataset | role | state |
|---|---|---|
| `warehouse_v2_yolo_20260821` | trained the detector | exists |
| `warehouse_v2_yolo_shared_20260822` | **bias and covariance only** — needs ground truth, ~70 spots would do | exists, 11 585 trials |
| driven operational logs | **availability only** — no ground truth, labels computed the way the runtime computes them | **missing** |
| pictures of the empty warehouse | false-positive check | **missing** |
| driven routes | operational validation and fusion | **missing** |
| planning episodes | final evaluation | **missing** |

Ground truth is used to form residuals and to score results. It never becomes an input the
online planner or filter can see.

---

## The stages, and what each has to show

### 1. The sensor works (sentence 1)

Frozen detector, characterized and reported rather than tuned. **Done:** finds the robot
93 % of the time, 98 % close in, 85 % beyond 20 m, 56 % when more than 70 % of it is
hidden. More than one box in 0.33 % of frames.

**Open:** false positives. The commissioning capture kept one empty frame per camera, so
"does it report a robot where there is none" is unknown — and that is half the definition
of a usable sighting. A handful of empty-warehouse pictures closes it. Do this before the
availability model, because it changes what the availability model is counting.

**Do not quote the training mAP.** Its held-out places sit a median 0.70 m from training
places against a 0.67 m grid, so it measures memorisation of the grid. The 93 % figure,
measured on a separate capture, is the honest one.

### 2. "Usable sighting" is frozen before anything is learned

A sighting counts when the detector fires above a fixed confidence threshold, the
projection to a floor position succeeds, and it passes checks that compare the detection
against the prediction — height, width, bottom edge, frame edge. **All four are available
at runtime**: none needs ground truth or the segmentation mask.

Partial occlusion needs no special handling. A hidden robot is missed more often, which
lowers the availability probability. A sighting that arrives but is noisier is handled by
the covariance. Something grossly wrong is caught by the admission checks.

No ground-truth error threshold may ever define usability.

### 3. The observation model, and the leftover offset (supports sentence 4)

**Two different things, and only the second is a bias.**

**The observation model** is the box-versus-centre problem. The bottom-centre of a
detector's box lands **24–35 cm** from the robot's true centre, swinging **11 cm** as the
robot turns, because a 0.80 x 0.55 m footprint presents 27.5 to 48.5 cm of half-extent along
the viewing ray. It is **not fitted and not corrected** — it is removed by predicting the
box the same way the detector measures it, so that both sides are the same physical
quantity. That is the largest error in the whole chain, and because the prediction needs a
heading, **heading error feeds straight into any RMSE measured against the true robot
centre** at roughly 0.23 cm per degree. Worse, it barely moves the predicted *pixel*, so the
admission check cannot catch it.

**The leftover offset** is what remains once that is right: half a centimetre. Measure it
during commissioning, subtract it, move on.

**Done.** The leftover offset was 0.53 cm; after correction 0.29 cm, against 2.2 cm of
random scatter. One sentence in the paper, one figure in the appendix. The observation model
gets a figure in the main deck, because it is worth 30 cm rather than 0.5.

Worth carrying forward: the offset comes from mildly occluded sightings the gate admits,
and those get commoner with distance — so it is a property of *this stock arrangement*,
not of the cameras, and must be re-measured if the warehouse is restocked.

### 4. Measurement covariance, simplest first

The ladder, in order, stopping as soon as the next rung fails to earn itself:

1. one covariance per camera
2. plus dependence on distance
3. plus dependence on viewing angle

**Already indicated, to be confirmed on the frozen pipeline:** a single noise number pushed
through the imaging geometry beat a per-camera constant decisively, and also beat a fitted
spatial map with a thousand times as many parameters. Geometry already knows the shape of
the error, so there may be little for data to learn. Confirm, then stop.

### 5. Confidence gets a fair trial (sentence 3)

Confidence was central to the previous method, so it is tested rather than dropped. It has
three possible roles and they are not the same:

- **Admission** — a fixed threshold decides whether a detection counts. Standard. Keep.
- **Baseline** — the previous confidence-based method stays intact as a comparison, so
  nobody can say an informative signal was quietly discarded.
- **Covariance predictor** — does confidence say anything about how accurate a sighting
  turned out to be, *after* accounting for camera and viewing geometry?

The experiment compares four covariance models on held-out sightings: camera only;
confidence; distance and angle; distance, angle and confidence. Scored by how well each
predicts the errors actually observed, plus whether its stated ellipses contain the truth
as often as claimed.

**This can run today** — the confidence value is already recorded for every sighting in
`logs/studies/measurement_commissioning/sightings.csv`.

**Why confidence probably cannot go in the planner regardless of the outcome:** it only
exists *after* an image has been processed. When planning a route through a place the robot
has not reached, there is no confidence value to consult. What commissioning can supply for
a future place is the probability of getting a sighting at all, and the expected quality if
one arrives. That is a principled reason for confidence to appear in fusion but not in
planning — much stronger than a preference.

### 6. The availability map (sentence 2)

For every camera and every commissioned place: did a usable sighting happen? Fit the
probability of that across the floor.

The claim is not "we used a Gaussian process". It is: **we empirically commissioned the
spatial probability that each camera will provide a usable sighting.** The fitting method
is an implementation detail and must be compared against simpler ideas:

| model | what it assumes |
|---|---|
| constant | everywhere equally reliable |
| geometry | field of view, range, viewing angle — no data |
| previous method | the old confidence-based reliability map |
| **proposed** | measured probability of a usable sighting |

Headline metric: **Brier score on held-out places**, because the output is a probability and
what matters for planning is that it is *calibrated*, not merely well-ranked. Supporting:
calibration curve, predicted-versus-actual detection rate. Report ranking measures only in
the appendix.

Evaluate on held-out places or held-out routes — never on samples interleaved with the
fitted ones at grid spacing, which tests interpolation rather than prediction.

One headline number and one map. Then stop.

### 7. Fusion on one fixed route (sentence 4)

Before asking a planner to seek camera coverage, show the coverage is worth seeking. Full
design: [`experiments/fusion_on_fixed_routes/README.md`](experiments/fusion_on_fixed_routes/README.md).

**Two names, kept apart.** `R_pix = sigma_px^2 I` is the detector's noise, in pixels, one
commissioned number. `Sigma_c = J_c^-1 R_pix J_c^-T` is what that camera then knows about the
robot's position, in metres. Distance and viewing angle are already inside `Sigma_c`, because
they are what changes `J_c` — so **nothing in the proposed method penalises range or angle by
hand.** They appear once, as the heuristic baseline.

**One route, six arms.** Task `fusion_network_traverse`: the west dock door, 3.4 m from
camera A, to the east cross aisle under camera D — 30.62 m, all five cameras contributing,
and the number of cameras watching at once spanning 0 to 4 (2.4 / 9.6 / 9.2 / 5.2 / 4.4 m).
The polyline is frozen by sha256, so every arm executes the same coordinates.
Every arm drives it identically, so the fusion rule is the only difference. Figure:
`logs/studies/deck_figures/fusion/01_the_route.png`.

| run | observation model | fusion rule |
|---|---|---|
| 1 | hull | single best camera, smallest `tr(Sigma_c)` |
| 2 | hull | distance-and-angle weights — the untuned heuristic |
| 3 | hull | independent Gaussian fusion, `Sigma^-1 = sum Sigma_c^-1` |
| **4** | **hull** | **network pooling, `Sigma^-1 = (1/N) sum Sigma_c^-1`** |
| 5 | raw box bottom-centre | network pooling |
| 6 | fixed box-to-centre offset | network pooling |

Runs 1–4 answer *which fusion rule*; runs 4–6 answer *what a detector's box means*. Run 4
serves both, so it is six runs and not eight.

`Sigma_c` carries camera quality and the `1/N` exponent carries conservative pooling — two
jobs, two mechanisms, and the exponent is never also set from `Sigma_c`.

Headline: **position error in centimetres**, median and 95th. Beside it, always: whether the
stated uncertainty is honest. **The plot the experiment exists for:** error and *claimed*
uncertainty against the number of cameras contributing at that moment. If independent fusion
grows overconfident from two cameras on while network pooling stays honest, that is the
evidence for treating the network as one sensor; if both stay honest, `1/N` was unnecessary
and that is also a result.

**Six live closed-loop drives, one per arm, each with its own folder and its own storyline**
(`experiments/fusion_on_fixed_routes/OUTLINE.md`). The arms do not see the same detections,
because fusion feeds the belief, the belief predicts the box and the box decides admission.
That coupling is the method: an arm that only wins on another arm's admitted detections has not
won anything a robot could use. Nothing is paired or averaged across arms yet, so each arm's
path length and deviation from the commanded route are reported beside its error, and repeats
are a decision to take after one drive each. The predicted-bounding-box hull method gets its
own storyline ahead of the arms, since it is what every arm's measurement means.

Availability is **not** used to downweight a measurement that has already arrived. It
predicts the future; it does not judge the present. Worth saying explicitly in the paper.

**Driven, 2026-08-26. All six arms reached the goal.** Full results:
`logs/studies/fusion_on_fixed_routes/RESULTS.md`.

- **THE HEADLINE CHANGED. Fix the covariance model and the fusion rule stops mattering.**
  Three modelling failures were found by asking why a reading was a metre wrong: a correction
  applied 400 ms stale, an admission check with no runtime caller, and a residual bias declared
  nowhere. Fixing all three lifts every arm's calibration by 15-26 points and **collapses the
  advantage of dividing by N from +28.7 points to +3.6**:

  | arm | both defects live | + timing | + check | + floor |
  |---|---|---|---|---|
  | F1 single best | 43.4% | 52.3% | 51.2% | **66.0%** |
  | F2 distance and angle | 45.9% | 48.5% | 53.5% | **68.2%** |
  | F3 precisions add | 26.0% | 39.2% | 43.0% | **69.4%** |
  | F4 network / N | 54.8% | 50.1% | 54.9% | **73.0%** |

  Most of what read as a pooling result was a conservative claim absorbing those failures. The
  defensible contribution is the measurement model, not the rule. Still not honest: 73% against
  95%. `experiments/anomaly_investigation/`.
- **The fusion rule barely moves accuracy and clearly moves honesty.** Confirmed on **four
  routes, 24 drives**, including a pair that share start, goal and length and differ only in
  coverage (92% vs 45% two-camera) and a 63 m out-and-back. Median error stays inside
  2.5–4.5 cm for all four fusion arms with no consistent ordering; **F4 is the most honest on
  every route** (55/60/50/46%) against F3's 26/40/39/38%. No arm is honest — 95% is the target.
- **Coverage buys the tail, not the median.** On the well-covered corridor the 95th percentile
  falls from 49–57 cm to 19–28 cm while medians barely move.
- **Two of 24 drives ended in contact, both on the 63 m route**, after drifting 0.62 m and
  0.79 m off a corridor whose physical clearance never drops below 0.495 m. The belief
  excursions that look merely ugly at 30 m become collisions at 63 m. One drive per arm, so
  this ranks nothing — it is an argument for fixing the covariance before the planner.
- **Fusing can be worse than the best camera already on the table** — in 10–22% of corrections,
  most often under precisions-add. A camera claiming a tiny ellipse and sitting a metre off
  drags a precision-weighted mean with it, even when a camera 6 cm from the truth is in the
  same fusion. F3 and F4 share that rate to within a point, because dividing by N changes the
  claim and not the estimate.
- **Every rule over-claims as cameras are added.** From one camera to four, the published
  correction claims 4.15x more precision under precisions-add while actually improving 1.10x;
  dividing by N halves the over-claim to 2.07x. Neither reaches honest, so the paper's claim is
  "conservative pooling narrows the over-claim", not "it fixes it".
- **The observation model is worth 7.4x the median error**: hull 3.12 cm, fixed offset
  17.46 cm, raw box 23.08 cm — same rule, same route.
- **TWO runtime defects invalidated the absolute numbers of the fusion study, and both are
  fixed.** (a) The admission check — `plausibility_reasons()`, same thresholds as commissioning,
  unit-tested — **had no caller in the runtime**, so 30% of detections that should have been
  refused became corrections; their median error is 24 cm and their worst 122 cm. (b) A
  correction was applied ~400 ms stale. With both fixed a published correction is **1.79 cm
  median, 9.1 cm p95**, against commissioning's 1.44 / 9.2 — the operational sensor now matches
  the commissioned one. `experiments/anomaly_investigation/`.
  - The gate is not free: it converts bad measurements into ABSENT ones. Corrections drop 7%
    and the worst outage grows from 5 s to 13 s, which is an argument for the availability work.
  - **A measurement bug of my own**: `state_error_gt_m` re-scores a held correction while the
    robot moves, reporting the robot's travel as sensor error. Corrections must be scored once,
    when published. Every correction-level number measured before this is inflated.
- **The 8 cm correction error is explained, and it is not the covariance model.** The
  correction is **400 ms old** by its own timestamp and is applied as if current: at 0.22 m/s
  that is 8.8 cm of travel, the error sits 7.3 cm *behind* the robot along its heading, and
  scoring the same corrections against the pose 0.35 s earlier collapses the median from
  **8.17 cm to 2.48 cm**. Odometry, which has no camera pipeline, bottoms out at 0.1 s, so
  ground truth is not the late one. Heading is not detectable at a standstill and there is no
  belief-to-correction feedback: standing still, a good camera reads 0.9–3.3 cm with the belief
  a metre wrong. `logs/studies/fusion_on_fixed_routes/latency/README.md`.
  - **So sigma_px IS about right** for the measurement at its own timestamp, and the fix is to
    apply the correction at that timestamp (odometry forward-prediction, or a delayed-state
    update) rather than to inflate R. A bias is not a covariance.
  - **The honesty numbers must be re-earned after that fix.** Every arm shares the defect, so
    the ordering may survive, but the absolute calibration figures are not yet properties of
    the fusion rules.
  - **The tail is a different problem**: p95 is untouched by the lag. Camera C reads 17.3 cm
    median, 2–4x worse than any other camera at the same range; camera A reads 87.9 cm out
    below 6 m. Both need chasing, after the timestamp fix.

### 8. Planning (sentence 5)

For a candidate future place, the planner asks each camera how likely a sighting is and how
good it would be, then works out what the robot's uncertainty would become either way and
weights the two outcomes by that probability. **Probability that a sighting happens is kept
separate from how good it is if it does.** That separation is the conceptual centre of the
paper.

| planner | what it assumes about the cameras |
|---|---|
| P0 shortest path | ignores them |
| P1 uniform | same support everywhere |
| P2 geometry | field of view and range only |
| P3 previous method | the old confidence-based map |
| P4 proposed | measured availability plus calibrated covariance |

Plus the ablation that could simplify the whole method: **availability alone** versus
**availability plus conditioned covariance**. If they tie, say so — "availability is the
dominant quantity and elaborate state-dependent covariance is unnecessary" is a cleaner
contribution, not a failed experiment.

Obstacle and keep-out costs stay completely separate from all of this.

**Scenarios must contain a real choice.** Routes of roughly equal length with clearly
different camera support are the most convincing, because they isolate localization support
from path length. Also include camera-to-camera handovers, overlap zones, and long stretches
with poor support.

**Metrics.** Headline: localization error in centimetres. Mechanism: **the longest stretch
with no usable sighting**, in seconds — better than "time visible", because a robot may only
need occasional well-timed corrections. Cost: extra distance or time, as a percentage. The
result a reviewer can absorb in one line is *"X % lower localization error for Y % extra
travel."*

### 9. Why a network beats one camera

Compare the best single camera against all five. The mechanisms are redundancy,
complementary coverage, differing quality, and handover as the robot moves.

**Do not claim multiple cameras cancel bias.** Two cameras with offsets fuse to a weighted
combination of those offsets, which does not vanish. Measured here: a heading error is
shared by every camera, and fusing all of them removes 1 % of it.

---

## What still has to be collected

1. **Pictures of the empty warehouse** — enough to characterize false positives and close
   the definition of a usable sighting. Small.
2. **Driven routes**, recording the robot's own position estimate, the detector output, the
   sightings actually received, timestamps, and reference position kept separate for
   scoring only.
3. **Planning episodes**, entirely separate from anything used to fit the availability map
   or the covariance.

**Not needed:** another detector dataset. The detector is frozen and adequate. Retrain only
if the false-positive check fails.

---

## The heading architecture, and the one gate that validates it

**Locked.** The heading in the observation model comes from the robot's own operational
estimator — odometry-driven — never from ground truth. The camera update moves `x, y` and
nothing else. This is not a preference: the runtime raises a `RuntimeError` if
`heading_update_mode` is anything but `camera_xy_only`, in both
`unicycle_planner_node.py` and `visibility_launch_common.py`.

**The failure mode it creates.** A camera sees one residual containing position error and
heading error together. Forbidden to move the heading, the filter explains all of it by
moving `x, y` — so **a wrong heading is silently absorbed as a position correction**, and
the admission check cannot catch it, because the bottom-centre barely moves in pixels when
the heading changes.

**Both halves of that follow from the same small sensitivity.** Weak sensitivity means a few
degrees of heading error disturbs the prediction less than the detector's own noise. It also
means the bottom-centre carries almost no information with which to *correct* a bad heading.
So: good for `x, y`, weak for heading — and **the paper must not claim that bottom-centre
observations recover heading.**

**Half the gate is measured** (`heading_gate.py`, on the commissioning capture):

| heading error | prediction moves | vs detector noise | position error absorbed |
|---|---|---|---|
| 1° | 0.06 px | 0.08 | 0.23 cm |
| 3° | 0.19 px | 0.24 | 0.73 cm |
| 5° | 0.33 px | 0.44 | 1.24 cm |
| 10° | 0.64 px | 0.84 | 2.29 cm |
| 20° | 0.98 px | 1.29 | 3.22 cm |

**Break-even is about 14°** — that is where heading error starts disturbing the prediction
more than the detector's own noise already does.

**The missing half** is the operational heading error itself, which needs a driven
trajectory: log the estimator's own heading, record ground truth separately, and use it only
afterwards to score the drift. Then read the answer off the table.

- **Heading holds to a few degrees** → keep `camera_xy_only`. The paper says the operational
  estimator supplied heading while the cameras corrected translational drift. Simple, and
  defensible.
- **Heading drifts near 10–20°** → the filter is turning orientation error into position
  correction. Either heading enters the camera update, or a heading-sensitive feature such
  as the box width is added. Both are new modules with their own validation.

**Resolve this before complicating the covariance, the availability map or the planner.**

**And state the claim at the right strength.** This method corrects drift around an already
initialized estimate. It is not built for recovery from an arbitrary starting pose: a
sufficiently wrong estimate puts the predicted box in the wrong part of the image, the
admission check rejects it, and nothing is corrected. Say "external camera measurements
correct drift around an operational state estimate", not "the cameras relocalize the robot".

## Metrics: three different error numbers, and never confusing them

The word "error" covers three quantities in this work. They differ by more than a factor of
two and only one of them is the paper's headline.

| | what it is | measured? | value |
|---|---|---|---|
| **single-sighting measurement error** | how far one camera reading lands from the truth, if the prior were exactly right | **yes** | 1.49 cm median, 3.50 cm RMSE |
| **localization error** | how far the robot's own belief lands from the truth, over a run | **not yet** — needs driving | the headline |
| **belief consistency** | whether the stated uncertainty is honest | partly | see the ladder |

**Never quote the first as the second.** The filter fuses many sightings, so localization
error can be *lower*; it also carries odometry drift and heading error, so it can be
*higher*. They are not comparable and a reviewer will notice if they are conflated.

### What makes the comparison valid at all

The filter estimates the robot's centre and ground truth reports the robot's centre. That is
**verified, not assumed**: the visual hull's origin sits 0.5 mm from the centre of its own
bounding box, and the pose logged as `robot_x, robot_y` is the pose commanded to the
simulator. Same point on both sides. It is also *why* the commissioned residual is 0.5 cm
rather than tens of centimetres — a frame mismatch would have shown up immediately as a large
constant offset.

**Re-check this whenever the robot description changes.** A base frame moved to the wheel
axle would put a fixed offset into every reported RMSE and nothing else would look wrong.

### What the localization error will be made of

| contribution | size | visible during commissioning? |
|---|---|---|
| pixel noise, through the geometry | 0.9 cm near a camera, 4.0 cm far | yes |
| heading error, through the observation model | ~0.23 cm per degree | **no** |
| the leftover commissioned offset | 0.29 cm | yes |
| odometry drift between corrections | unmeasured | no |

**The second row is the one to watch.** Commissioning used the true pose, so heading error
contributes nothing to any number measured so far — not the residual, not the admission rate,
not the covariance. It appears for the first time in the localization error. If the reported
RMSE comes out worse than the single-sighting numbers suggest, heading is the first suspect,
and `heading_gate.py` converts a measured heading drift straight into the centimetres it
would explain.

### Requirements for reporting it

1. **Say which of the three numbers you mean**, every time.
2. **Same reference point on both sides** — re-verify if the robot description changes.
3. **Report consistency beside accuracy.** A filter that widens its ellipse passes any
   calibration test and is useless; a filter that is accurate but overconfident is dangerous.
   Neither number means anything alone.
4. **Ground truth scores the result and never enters the estimate.** It forms residuals and
   computes metrics; it is not an input to the filter or the planner.
5. **Report the heading contribution separately**, because it is invisible everywhere else.
6. **Percentiles as well as RMSE.** The single-sighting median is 1.49 cm and the RMSE is
   3.50 cm — a factor of 2.4, because the far-range tail dominates the square. Quoting only
   one of them misrepresents the sensor in opposite directions.

## The largest open risk

Everything commissioned so far was measured while the system was *told* exactly where the
robot was. In operation it must work from its own estimate — which changes the predicted
box, and therefore changes which sightings pass the admission checks. This is untested,
it is offline, and it uses data already on disk. It should be settled before any driving.
