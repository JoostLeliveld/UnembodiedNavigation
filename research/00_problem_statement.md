# Problem statement

Companion to `01_questions.md` (the immutable SQ1–SQ4 spine). Methodology is in
`00b_methodology.md`.

Numbers re-derived from study logs on 2026-08-20; sources named inline. Nothing from the
retracted `e4` closed-loop campaign is used (see `project_efe_null_retracted_2026-08-20`).

---

## The statement

> A mobile robot that navigates on infrastructure sensing does not own its own perception. Its
> belief is assembled from cameras it cannot inspect, and what reaches it is not one camera's
> measurement but a **fused** quantity — combined across cameras, by a policy someone chose.
> To plan, the robot must know where along a route the network will actually deliver a usable
> update. To act on that, it needs a rule allowed to trade path length for observability.
>
> **The problem this thesis addresses is that observation quality is fused and consumed by
> planners in ways that do not preserve what the planner needs: per-camera prediction quality
> does not determine fused decision quality, the fusion architecture silently decides what the
> belief can know, and the field the planner reads goes out of date when the building changes.**

Three coupled failures, none of which shows up in a prediction score: **fusion** discards the
tail a planner acts on; **fusion transport** discards state the filter needs; and **staleness**
invalidates the field without announcing it.

---

## 1. Setting

Warehouses, terminals and production halls increasingly instrument the building rather than
the robot. Wall- and ceiling-mounted cameras localise a robot from viewpoints it does not
have and cut what it must carry. The robot is then navigating on borrowed sensing: much of
its state estimate arrives from sensors it does not own, mounted where it cannot check them,
looking down sight-lines that other people's forklifts change.

Concretely: fixed wall-mounted cameras with known calibration watch a differential-drive
robot in a Gazebo warehouse adapted from the AWS RoboMaker Small Warehouse World. A YOLO
detector fires or does not; when it fires, a floor-plane projection becomes a position
measurement. Networks of one to four cameras are studied, and the network is a variable, not
a constant.

## 2. The three problems

**(a) Planners read the wrong quantity.** It is common to fold a missing detection into the
measurement model as an observation with a very large covariance. That needs an arbitrary
miss endpoint and conflates two different things: whether an update arrives, and how sharp it
is given that it did. A planner needs both, separately, along a candidate route — and it
needs a rule that can act on them.

**(b) Fusion is where prediction quality and decision quality come apart.** What the planner
reads is fused across cameras. Fusion is not a neutral aggregation step: it decides the *tail*
of the field, and avoidance behaviour lives entirely in the tail. Two estimators can tie on
average-case per-camera score and behave completely differently once fused. Model selection
done on per-camera prediction accuracy can therefore select the wrong estimator.

**(c) Fusion architecture decides what the belief can know.** How corrections are transported
from the network to the robot — a per-camera image-space measurement versus a pre-fused
world-frame position — determines which state variables remain observable. This is an
architectural choice usually made for engineering convenience, and its effect on the filter
is not stated anywhere.

Behind all three sits the reason the field is hard to keep at all: it is normally fitted to
past detections, and warehouses are restocked between shifts.

## 3. Why each is real

Measured in real Gazebo against real detector outcomes: 15,072 camera-pose samples nominal
and 15,033 restocked, four cameras each, labelled at confidence 0.25.

**(a) Availability and accuracy are different fields, and following the wrong one costs.**
Geometry predicts *whether* a camera delivers a detection at Spearman **0.807–0.835** across
four cameras, but the *conditional position error* at only **0.098–0.229**. At **42.2 %** of
the 353 positions covered by two or more cameras the most-available camera is not the most
accurate one, and following availability there costs a median **1.3 cm**
(`e2_availability_vs_accuracy/`). One scalar trust score cannot carry both, so camera
selection and route selection are different decisions on different fields.

**(b) Fusion erases the blackspots that a planner needs, and recalibration cannot fix it.**
Fitting the same two-parameter link to each field:

| Field | link slope | per-camera floor | **fused floor, 4 cameras** |
|---|---:|---:|---:|
| Surveyed CAD ray-cast | 0.628 | 0.024 | 0.094 |
| FOV / range | 0.481 | 0.028 | 0.106 |
| **Monocular depth** | **0.293** | **0.057** | **0.208** |

Monocular depth's link is half as steep — well calibrated on average, but it never asserts
"definitely not visible". Noisy-OR over four cameras lifts its floor to
`1 − (1 − 0.057)⁴ ≈ 0.21`, **erasing the blackspot entirely**, so a planner reading it has
nothing to avoid. Replacing the logistic link with isotonic regression — a strictly more
flexible calibrator — moves the fused floor only 0.208 → 0.183 and leaves held-out Brier
unchanged (0.0680 → 0.0689). The compression is in the field, not the calibrator. Monocular
depth *ties* the surveyed ray-cast on held-out Brier while being useless for avoidance at four
cameras. **Aggregate calibration parity does not imply usable fused discrimination**, and
reporting Brier alone hides it.

**(c) The value of availability modelling depends on camera placement, not count.** Time
driven with no camera on the robot, and what modelling availability saves:

| Network | blind baseline | CAD saves | depth saves | detour |
|---|---:|---:|---:|---:|
| 4 cameras | 2.40 s | 1.70 s | 0.60 s | 0.04 m |
| 3 cameras | 4.70 s | **3.00 s** | 2.00 s | 0.04 m |
| 2, **opposite** walls | 7.50 s | **3.40 s** | **2.60 s** | 0.19 m |
| 2, **same** wall | 8.80 s | 1.50 s | 1.20 s | 0.00 m |
| 1 camera | 14.50 s | 3.20 s | 3.10 s | 0.00 m |

Two cameras on opposite walls give the largest saving of any configuration; two on the same
wall the smallest, despite the same count and a *worse* baseline. Where views are genuinely
complementary, knowing which camera will see you is worth a lot; where they overlap there is
little to choose. None of it is bought with path length — detours stay at or below 0.19 m.
This makes network design part of the problem, not a fixed backdrop.

**(d) The fusion transport silently removed heading observability.** In the single-camera
stack, corrections arrive as an image-space measurement and the filter's Jacobian couples
heading, so heading is observed (`cov_yaw` moves 0.007 → 0.074 → 0.004 within a run). The
four-camera stack replaces that with a pre-fused world-frame position: `use_pixel_correction`
is forced off whenever multicam fusion is enabled, and `camera_manager_node` stamps every
fused correction with a non-informative yaw variance — an honest statement that a metric xy
fix says nothing about heading. Nothing replaced the lost observability. The planner's
`cov_yaw` then sits at π² ≈ 9.87 from the first sample while its position variance is
corrected to 0.0008 over and over. Propagated through a 75-step horizon this inflates the
belief tube until the obstacle term reaches **1,017,055** against a risk term of 511,683 —
the objective is swamped by an artefact of the fusion architecture. **How you fuse decides
what the robot can know, and nobody checked.**

**(e) Where the navigation question has been answered, the answer was yes — at one camera.**
The single-camera campaign (`honest_campaign_v2`; 3 conditions × 4 tasks × 5 seeds = 60 runs)
is the only place a reliability model has been varied end-to-end. The availability-aware
planner reached the goal **20/20** with **0** collisions; geometry-only managed 15/20 with 5
physical crashes, constant-covariance 15/20 with 4. Minimum clearance rose monotonically with
model quality on every route (0.10 → 0.18 → 0.36 m; 0.13 → 0.19 → 0.39 m), and the
availability-aware planner committed to one route in all 20 runs where the blind planner
scattered across three. **Whether this survives at four cameras is open** — blocked on (d),
and on giving every arm a real route menu. There is plenty left to steer for: **41 of 89**
tasks have over a metre of removable blind distance at four cameras, up to 10.5 m.

**(f) A rule that can act on the field exists.** Minimise expected blind distance subject to
`length ≤ (1+ε)·shortest`. At ε = 20 %, against real detector outcomes: 2.62 → **1.18 m** at
four cameras (55 % less, 10.2 % detour), and the effect holds at every network size down to
one camera (18 %), all p < 1e-10 over 89 tasks. Routing on a survey of the *old* warehouse
costs **0.31 m** more blind distance than a fresh one (p = 0.002).

**(g) The field the planner reads goes stale, and its sharpest component is the stale one.**
Restocking twelve of twenty-seven rack rows changes 575 camera–cell sight-lines while leaving
the lane network **bit-identical**. The learned field is the only estimator to lose real
skill (−0.140 [0.080, 0.202], 14/16 units, p = 0.004) and the ranking inverts: monocular
depth 0.485 → **0.619** overtakes the learned field 0.726 → 0.587 and even a fresh re-survey
(0.604); a survey of the old building (0.413) is worse than no survey. Worse, the two
properties share a component — monocular depth recovers **95.2 %** of newly-dark cells from
the camera's own image, but adding the frozen learned residual drops that to **34.2 %**. And
ranking survives while calibration does not: the frozen field keeps the best AUROC (0.959)
while its false-visible rate **doubles**, 0.046 → 0.091. A planner cannot detect any of this
from the field itself.

## 4. The questions

- **SQ1 Representation.** What must the network expose to a filter and a planner, given that
  availability and conditional accuracy are separate fields? (→ C1)
- **SQ2 Estimation.** Which estimators can be recomputed from deployment inputs, and — the
  part that matters here — which survive **fusion** with their tail intact? (→ C2, C6)
- **SQ3 Planning.** Under what decision rule does observation quality change what the robot
  does, and what does it cost in path length? (→ C3, C4)
- **SQ4 Deployment trade-offs.** Network design, fusion architecture, staleness, cold start,
  failure modes, fallback — not prediction accuracy. (→ C5, C6)

## 5. Scope, and what is outside it

- **In scope:** static wall-mounted cameras, known calibration, planar pose, a 2-D lane map
  the planner already has, one detector, networks of one to four cameras, noisy-OR fusion of
  availability and recursive fusion of position, route-level decisions, and closed-loop
  navigation at one camera.
- **Out of scope, as stated assumptions:** dynamic obstacles and other traffic; extrinsic
  drift beyond the probe already run; multi-robot interaction; detector generality beyond
  YOLO (RQ08); optical and vendor diversity (RQ09).
- **Simulation only,** with every number from real Gazebo captures and real detector
  outcomes — never synthetic labels, never an oracle standing in for the detector.
- **One reconfiguration.** All prediction inference is conditional on a single fixed
  nominal→restocked layout pair. This is the largest threat to external validity; a second,
  externally randomised layout is preregistered.
- **No four-camera closed-loop claim** until the heading observability in §3d is fixed. That
  is a property of the current implementation, not of the method.

## 6. Two evaluation findings that belong here

1. **A plausible lighting change is a null for this detector.** Frame mean grey 163 → 112,
   std 53 → 36 — the images genuinely changed — yet hit-rate given visible is 1.000 vs 0.982
   and given not-visible 0.118 vs 0.118. An appearance-conditioned term has nothing to model.
   Report the dose, or the absence of an effect at that dose.
2. **The labelling threshold matters more than the environment change.** At 0.01 the detector
   fires at 60 % of poses with *no* sight-line. The widely reused reference grid
   (`commissioning_grid_20260807`, 30,144 samples) was scored at 0.01, so about 12 % of its
   detection labels are detections at poses no camera can see. Current work uses 0.25.

## 7. Status

**Established:** availability and accuracy are separate fields; fusion decides the
decision-relevant tail and per-camera parity does not transfer; the value of availability
modelling is set by placement rather than count; the budgeted routing rule and its cost; the
staleness decomposition; and the navigation effect at one camera, end-to-end.

**Open:** whether the navigation effect survives at four cameras (blocked on §3d and on a real
route menu); generalisation beyond one reconfiguration (RQ10); commissioning cost (RQ12); the
per-method failure taxonomy (RQ14).

**Withdrawn:** the claim that the deployed planning objective cannot act on availability.
