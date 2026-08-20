# Preregistration — does the observation model survive a reconfigured warehouse?

Frozen 2026-08-19, **after** the appearance pilot and **before** any reconfigured-layout
detector outcome existed. What each section knew at freezing time is stated in that
section; nothing below is rewritten in the light of a result.

## 1. The question, and why it is not the previous paper's question

`EXP-AVAIL-SOURCE` established that a fixed camera network's usable-observation
probability can be estimated from the cameras' own images without a surveyed 3-D
model: monocular-depth raycast reaches held-out Brier 0.068 against the CAD
raycast's 0.062, difference +0.006 with 95 % interval [-0.002, +0.015]. Every number
in that study was measured in **one** warehouse configuration, with the fields fitted
and scored in that same configuration.

That leaves the deployment question untouched. A warehouse is restocked between
shifts, and a spatial reliability map fitted before the restock describes a building
that no longer exists. So:

> Which parts of a survey-free availability field survive a warehouse
> reconfiguration that the field was never shown, and does the component that makes
> the field sharp enough to plan with also make it stale?

The second clause is the part that is not obvious, and it comes out of the earlier
study rather than out of a hunch. Monocular depth alone **cannot** drive avoidance at
four cameras: its calibration link is half as steep as the CAD raycast's (0.293
against 0.628), so it never asserts "definitely not visible", and noisy-OR over four
cameras lifts its fused floor to 0.208 — erasing the blackspot entirely. The field
that does discriminate is the hybrid: a Gaussian-process residual fitted on detector
outcomes, on top of a monocular-depth prior mean (fused floor 0.0004). But the GP
residual is exactly the component fitted to the old building. If sharpness and
staleness come from the same term, a survey-free field cannot be both deployable and
adaptive, and that is a result worth reporting either way.

## 2. Environments

Four, differing only in the world file. Identical camera poses, identical intrinsics,
identical declared lanes, identical spawn pose, identical robot, identical detector
and identical detector thresholds.

| key | layout | lighting |
|---|---|---|
| `L0` | nominal | nominal |
| `L1` | 12 of 27 rack segments carry one extra 0.40 m layer of stock | nominal |
| `L0_lit` | nominal | two overhead lamps out, third dimmed to 0.35, low-angle directional light at 1.45 from the west |
| `L1_lit` | restocked | changed |

**Why restocking and not a pallet in an aisle.** Measured before choosing: three
obstacles placed in the driveable aisles, selected greedily to maximise newly blind
*reachable* floor, produce +46 newly blind cells out of 3397 covered — 1.4 %. Four
cameras on opposite corners cover each other's shadows, and an obstacle large enough
to darken an aisle also severs it, at which point every planner detours and the
comparison stops being about visibility. Restocking the racks costs the network 191
of 3397 covered cells (5.6 %) and changes 575 camera-cell visibility pairs, while
leaving the driveable network bit-identical: no aisle is touched, so obstacle
avoidance and observation modelling stay separate, which is the whole point of
keeping them apart. Per camera, coverage of eligible driveable ground falls
A 1234→1093, B 1134→987, C 1214→1066, D 1246→1107 — 11 to 13 % each.

The restocked segments were selected by `choose_layout.py` on *camera-cell
visibility pairs changed*, deliberately not on fused blackspot count: the headline
experiment scores per-camera fields, and fused blackspot count depends on how many
cameras the analysis keeps, so selecting on it would bake a camera count into the
world.

**Deviation from the two-world rule, declared.** `research/06_world_camera_design.md`
reserves the four-camera world for evaluating frozen methods and sends method
development to `warehouse_aws`. The reconfiguration holdout is evaluation of frozen
fields, which is what the four-camera world is for. The one genuinely new model here
is the conditional-detection term of §6; it is fitted on `L0` data only and never on
any changed environment, so it is a held-out evaluation in the four-camera world
rather than development in it. Any claim about that term names this deviation.

## 3. Capture and labels

Real Gazebo, one grid-teleport capture per environment, four cameras rendered per
teleport, detector scored offline on the saved frames. Grid geometry and detector
configuration are copied from the `L0` reference capture
(`logs/visibility_comparison/commissioning_grid_20260807`) rather than chosen:
46 × 36 positions over x ∈ [-11.7, 11.7], y ∈ [-9.0, 9.0], 0.45 m wall margin,
region-filtered to declared traversable lanes; YOLO `warehouse_yolo_detector_4cam_v3_960`
at `imgsz` 640, IoU 0.45, class `robot`, no masks.

New captures use 4 headings {0, π/2, π, 3π/2}; the `L0` reference used 8. The four
are a subset of the eight, so `L0` is subset to the same four headings for every
paired comparison. The reason is disk, and it is stated rather than hidden: one
capture is ≈4 GB of frames and the machine had 7.6 GB free.

`det_hit` is the label, not truth about the world. Ground truth — the commanded pose,
`oracle_visible`, the CAD prisms — is evaluation-only and never a model input.

## 4. Detector-threshold decision, taken from the pilot before the main captures

The `L0` reference capture was scored at confidence 0.01. A pilot of 24 positions ×
2 headings × 4 cameras in `L0` and `L0_lit` showed what that threshold costs:

| threshold | `L0` hit \| visible | `L0` hit \| not visible | `L0_lit` hit \| visible | `L0_lit` hit \| not visible |
|---|---:|---:|---:|---:|
| 0.01 | 1.000 | 0.125 | 0.982 | **0.603** |
| 0.05 | 1.000 | 0.125 | 0.982 | 0.132 |
| 0.25 | 1.000 | 0.118 | 0.982 | 0.118 |
| 0.50 | 0.982 | 0.081 | 0.964 | 0.088 |

At 0.01 the detector fires at 60 % of `L0_lit` poses that **no camera has a
sight-line to**. That is not a lighting effect on detection; it is the 0.01 threshold
admitting marginal boxes, and it vanishes by 0.05. Two consequences, both fixed here
before the main captures:

1. **The primary threshold is 0.25**, chosen as the middle of the plateau where both
   environments agree, not tuned to an outcome. Every headline number uses it.
2. The 0.01 numbers are reported as a **sensitivity row**, and the observation that
   the `L0` reference labels contain ~12 % detections at poses with no sight-line is
   reported as a property of that dataset.

## 5. What the appearance condition already showed, before the main captures

At any threshold from 0.05 up, `L0_lit` and `L0` agree on detection to within 0.02
absolute (hit | visible 0.982 against 1.000). **This lighting change does not move
this detector.** Frame statistics confirm the images really did change — mean grey
163 → 112, standard deviation 53 → 36 — so the null is detector robustness, not a
world that failed to change.

Registered consequence: the conditional-detection term of §6 has, on present
evidence, nothing to model in `L0_lit`, and the paper says so. Rather than escalate
the lighting until an effect appears — which would be selecting a world to fit a
claim — the study runs a **dose sweep** at pilot resolution over directional-light
intensity and lamp state, and reports the dose at which detection first departs from
nominal, or that no dose within a plausible warehouse range does. `L0_lit` and
`L1_lit` are captured at full resolution regardless, because the *availability
field* arms may degrade under appearance change even where detection does not, and
that is a separate question from detector robustness.

## 6. Arms

Every arm is fitted on `L0` only and then **frozen**. No arm sees any changed
environment at fit time.

| arm | operational inputs | recomputed per environment? | needs a survey |
|---|---|---|---|
| Constant (`L0` prevalence) | detector outcomes | no | no |
| Distance to camera | calibration | no | no |
| FOV / range | calibration, drivable map | no | no |
| CAD raycast, `L0` geometry | surveyed model of the *old* building | no | yes |
| CAD raycast, `L1` geometry | surveyed model of the *new* building | yes, by re-survey | yes |
| **Monocular depth raycast** | calibration, drivable map, camera RGB | **yes, from the camera's own frame** | no |
| GP on `L0` detector outcomes | calibration, detector outcomes | no | no |
| **Hybrid: GP residual on the monocular-depth prior** | calibration, drivable map, RGB, detector outcomes | prior yes, residual no | no |

The two CAD arms are the bracket the deployable arms are read against: `L0` geometry
is what a surveyed system has after the warehouse changes and nobody re-surveyed,
`L1` geometry is what it has if somebody did. Neither is called an oracle and neither
is a deployable result.

The monocular arm's floor-affine anchor is fitted on `L0` frames and **reused
unchanged** in every other environment, matching the earlier study's protocol. Only
the depth prediction itself is recomputed from the new frame. That is what makes
"adapts without recommissioning" a checkable statement.

## 7. Endpoints and metrics

All from `scripts/shared/metrics.py`; nothing hand-rolled.

**Primary.** Held-out Brier score of each arm against `det_hit`, per camera, in each
environment, with the two-parameter calibration link every arm gets fitted on `L0`
training folds only. The headline quantity is the *degradation* Brier(`L1`) −
Brier(`L0`) per arm.

**Secondary.** Log loss, AUROC, expected calibration error; and the false-visible
rate P(p̂ ≥ 0.5 | `det_hit` = 0), because asserting "the camera will see me" where it
will not is the error that hurts a planner most.

**Camera density.** Every metric is additionally reported over the 4-, 3-, 2-opposite,
2-same-wall and 1-camera subsets. This is analysis-only — one capture serves all
subsets — and it is registered in advance because the earlier study found the benefit
of availability modelling is largest at two cameras on opposite walls, not at four.

**Paired inference.** Camera × spatial-block units, six contiguous blocks, sign test
plus a 10 000-resample percentile bootstrap on paired differences, fixed seed.

## 8. Falsifiers, stated before the numbers exist

The claim "the monocular field adapts and the historical field goes stale" is
**refuted** if any of these hold:

- **F1** The GP arm's Brier degrades by less than the monocular arm's between `L0`
  and `L1`. Then history is not the stale component and the paper's premise is wrong.
- **F2** The monocular arm's `L1` Brier is not within the earlier study's +0.015
  bound of the `L1`-geometry CAD raycast. Then it did not track the change.
- **F3** The hybrid arm degrades no more than the monocular arm. Then the
  sharpness/staleness tension does not exist and §1's second clause is empty.
- **F4** No arm's Brier changes by more than 0.010 between `L0` and `L1`. Then the
  reconfiguration was too small to be a test, whatever the geometry said.

**F4 is the one this design is most exposed to** and it is checkable from the
geometry alone: 5.6 % of covered ground going dark may not move an average-case
score. If F4 fires, the honest report is that the four-camera network absorbs a
one-layer restock, and the camera-density axis of §7 becomes the primary analysis
rather than a secondary one.

## 9. What this study does not claim

No closed-loop navigation claim. `EXP-AVAIL-CL` stopped at 12 of 45 runs because all
12 persisted global plans were coordinate-identical, and an offline solve with the
runtime objective reproduced it: availability reaches the objective but lifts
ambiguity by ~17 units against a risk term of ~5400, so it never changes which route
wins. That is a property of the frozen planner objective and is not fixed here by
reweighting the visibility term, which stays frozen method. Route-level consequences
are reported only through the offline decision rule of `factorized_observation_successor`
— minimise the expected longest missed-update run subject to a 5 % length budget —
and are labelled offline route choice, never navigation performance.
