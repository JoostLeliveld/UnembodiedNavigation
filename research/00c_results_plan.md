# What the results report

The pipeline is: **initialise an availability map and an uncertainty map → plan on both →
fuse weighted by expected uncertainty → navigate.** The results follow that order, and each
stage reports one claim with one primary metric. Status is one of **DONE** (measured, in
hand), **BLOCKED** (precondition fails), **NOT RUN** (protocol exists, no data).

Numbers re-derived 2026-08-20 unless the source is named as a study record.

---

## R0. Both maps initialise from geometry, without ground truth — **DONE**

The claim that makes the pipeline deployable: neither map needs a survey or a detector
history to exist on day zero.

| Map | Initialised from | Held-out evidence |
|---|---|---|
| **Availability** `p_use(x,y)` | monocular metric depth, anchored on the calibrated floor plane, occlusion ray-cast | predicts real detector outcomes at Spearman **0.807–0.835** per camera; held-out skill **0.485** nominal, **0.619** after reconfiguration |
| **Uncertainty** `R_cond(x,y)` | `R = σ_px² · J Jᵀ`, `J` the Jacobian of the deployed back-projection | long axis lies along the camera's own sightline (**median 7°** from the bearing); anisotropy predicted **1.70–1.79** vs measured **1.49 / 1.74 / 1.80 / 1.49**; camera-frame aspect matches the closed form `slant / height` (1.42 at 4.8 m → 1.77 at 11 m) |

**Report the frame.** `R` is anisotropic **in the camera frame**. Pooled in the map frame over
a ±35° fan of ray directions it looks round (0.80 × 0.73 cm, aspect 1.10) — that is a pooling
artefact, not roundness. Every `R` number carries its frame.

## R1. The two maps are different objects — **DONE**

This is what justifies carrying both rather than one trust score.

- Geometry predicts *whether* a detection arrives at Spearman **0.807–0.835**, but the
  *conditional position error* at only **0.098–0.229**.
- Availability itself correlates with error at only **−0.18 to −0.39**.
- At **42.2 %** of the 353 positions covered by two or more cameras, the **most-available
  camera is not the most accurate one**, and following availability costs a median **1.3 cm**.

Primary metric: fraction of covered floor where `argmax p_use ≠ argmin tr(R_cond)`.
That fraction is the size of the decision the pipeline exists to make.

## R2. Planning on both maps — **DONE offline, one primary metric, one guardrail**

Decision rule: minimise expected blind distance subject to `len ≤ (1+ε)·len_shortest`.

| Network | blind, shortest route | with availability | reduction | detour |
|---|---:|---:|---:|---:|
| 4 cameras | 2.62 m | **1.18 m** | 55 % | 10.2 % |
| 3 | 4.87 m | 2.72 m | 44 % | 10.8 % |
| 2, opposite walls | 7.15 m | 5.60 m | 22 % | 7.7 % |
| 2, same wall | 7.60 m | 5.16 m | 32 % | 9.3 % |
| 1 | 11.45 m | 9.44 m | 18 % | 6.8 % |

All p < 1e-10, Wilcoxon paired over 89 tasks. Routing on a survey of the **old** warehouse
costs **0.31 m** more blind distance than a fresh one (p = 0.002).

**Guardrail — report this explicitly.** Terminal belief covariance is **not** an endpoint:
it moves **0.6 mm** across arms. Anything that looks like "the availability-aware planner
ends up more certain" is below noise. The planning endpoints are blind distance, longest
continuous unobserved stretch, minimum clearance, and goal attainment.

**Fusion caveat that belongs in the planning result.** A field's usefulness to the planner is
set by its *fused* floor, not its per-camera score:

| Field | per-camera floor | fused floor, 4 cameras |
|---|---:|---:|
| Surveyed CAD ray-cast | 0.024 | 0.094 |
| Monocular depth | 0.057 | **0.208** |

Noisy-OR lifts monocular depth's floor to `1 − (1−0.057)⁴ ≈ 0.21`, erasing the blackspots
entirely, even though it *ties* the surveyed ray-cast on held-out Brier. Isotonic regression
moves it only to 0.183, so it is the field and not the calibrator. Report the fused floor at
the deployed camera count alongside any aggregate score.

## R3. Fusion weighted by expected uncertainty — **DONE offline, BLOCKED in the deployed stack**

This is the pipeline's punchline and it has a hard precondition.

**The mechanism decomposes into two parts with opposite optima.** For two cameras at crossing
angle θ:

- *random* pixel noise: fusing two 1.75:1 ellipses is best at **90°** (×0.61), only `1/√2` at
  0° or 180°, because both cameras are then vague about the same direction;
- a *repeatable per-camera lean* along each sightline: what survives is **`b·cos(θ/2)`** —
  unchanged at 0°, ×0.71 at 90°, and **exactly cancelled at 180°**.

**The lean dominates, so head-on wins.** Predicted vs measured lean (mm) by crossing angle:
54° 46.6/44.9 · 97° 16.7/15.9 · 159° 8.0/9.3 · 177° 1.1/2.3.

**Weighting by the correlated `R` beats plain averaging exactly where it matters** — delivered
error, fused ÷ single camera:

| crossing angle | plain average | **R-weighted** |
|---|---:|---:|
| 45–75° | 0.98 | 0.82 |
| 75–105° | 0.93 | 0.91 |
| 105–135° | 0.60 | **0.41** |
| 135–165° | 0.50 | 0.42 |
| 165–180° | 0.42 | **0.38** |

Range-matched control at ~10 m, so only the angle moves: narrow ×1.03, head-on ×0.37.

**THE PRECONDITION, and it must be stated as a result.** On the observation function **as
deployed today** none of this exists: ratios ×1.02 / 1.03 / 0.95 / 0.92 / 0.87 — a second
camera is worth nothing at any angle. The error is then 40–48 mm of yaw-driven scatter
**shared between cameras**, and no weighting recovers information from a shared error. The
deployed runtime is `state_pipeline: homography_to_bev`, `observation_model: uv` — the
box-bottom reading — with no keypoint reading wired in.

So the reported finding is an **ordering**: uncertainty-weighted fusion is downstream of
fixing the observation function, not an alternative to it. The fix is measured and available
(marker keypoints cut the lean 7.78 → 1.51 cm median, bias along the ray −4.75 → −0.30 cm,
heading dependence gone) but is **not in the runtime**.

## R4. Placement, not count — **DONE, and it is the result that unifies the two maps**

Both halves of the pipeline independently select the same camera geometry.

- *Availability side*: two cameras on **opposite** walls give the largest saving of any
  configuration (3.40 s of blind driving) while two on the **same** wall give the smallest
  (1.50 s), despite equal count and a worse baseline (8.80 s vs 7.50 s).
- *Uncertainty side*: the lean cancels as `cos(θ/2)`, so **large crossing angles pay**;
  delivered error ×0.38 at 165–180° against ×0.91 at 75–105°.

And the layout constrains what is available: this warehouse offers only narrow crossings from
same-wall pairs (A–C median 68°, B–D 78°); the useful pairs are cross-wall (A–B 124°, C–D
122°) and diagonal (A–D 165°, B–C 167°). Independently corroborated by the identifiability
result — pairs >100° apart land within 2 mm, pairs <80° are confidently wrong-signed.

**Report as: the benefit of modelling observation quality is set by camera placement, not
camera count, and availability and uncertainty agree on which placement.** That is a network
design claim, and it is the strongest single message the pipeline produces.

## R5. Navigation, end to end — **DONE at one camera, BLOCKED at four**

Only place a reliability model has been varied end-to-end: `honest_campaign_v2`,
3 conditions × 4 tasks × 5 seeds = 60 runs, single camera.

| Condition | goal | physical collisions | min clearance (2 open routes) |
|---|---|---|---|
| geometry-only | 15/20 | **5** | 0.10 m, 0.13 m |
| constant covariance | 15/20 | **4** | 0.18 m, 0.19 m |
| **availability-aware** | **20/20** | **0** | **0.36 m, 0.39 m** |

Minimum clearance rises monotonically with model quality on every route, and the
availability-aware planner committed to one route in all 20 runs while the blind planner
scattered across three. All five crashes were on the camera-poor route.

**Four cameras is blocked, on two named preconditions** — a route menu whose members differ
by more than the robot's width, and a measured heading variance rather than the
non-informative prior the multicam fusion transport leaves it at.

---

## The one experiment the pipeline needs and does not have

Everything above is either an offline map property, an offline route decision, or a
single-camera navigation result. **The fusion *policy* comparison has never been run**:
selecting and weighting cameras by expected uncertainty versus by availability, on frozen
maps, scored on filter calibration and navigation outcome. `C5` is `READY` with "policy
comparison not run"; `EXP-CAM-MGMT` is `PLANNED`.

That is exactly the stage the pipeline is built around, and R1 says it is worth running: the
two policies disagree at **42.2 %** of multi-camera positions. Its precondition is R3's — the
observation function must be fixed first, or the comparison measures shared yaw scatter.

**Suggested reporting order for the paper:** R0 (both maps, survey-free) → R1 (they differ,
42.2 %) → R4 (placement, both maps agree) → R2 (planning, 55 % blind reduction) → R3 (fusion,
×0.38 head-on, with the ordering caveat) → R5 (navigation at one camera). R4 early, because
it is the claim a reviewer will remember.
