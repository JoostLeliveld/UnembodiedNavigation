# Preregistration — availability-aware external-camera navigation

Frozen analysis plan. E1–E3 ran before this document was finalised as described in
their sections. E0 subsequently passed on 2026-08-18. E4 started and was stopped
after 12 of 45 planned runs because all 12 persisted `mc_blind_L` global plans were
coordinate-identical across C1–C3; a thirteenth attempt ended before persisting a
plan. The frozen design below is preserved rather than rewritten after that result.
The incomplete campaign supports no comparative navigation-performance claim.

## 1. The contribution, in one sentence

> An external-camera network's usable-observation probability can be estimated from
> the cameras' own images, without a surveyed 3-D model of the building, well
> enough to plan with — and planning with it reduces the time a robot spends
> unobserved.

Two deliberate choices about where the novelty is claimed:

**The novelty is the deployable estimator, not the separation itself.** That an
observation either arrives or does not, and that this is different from a noisy
observation, is standard in partially-observable formulations. Claiming it as the
contribution invites the correct objection that it is textbook. What is not in the
literature is the chain from a fixed camera's RGB, through monocular metric depth,
to a calibrated per-camera detection probability, to an expected-free-energy
planner — measured end to end against real detector outcomes.

**The explicit Bernoulli planner is a formulation result, not the headline.** Its
measured benefit over folding availability into the covariance is small, and this
was known before this study: the runtime precision blend is linear in precision, so
it already carries the availability-weighted mean information, and the four-task
static replay produced identical routes and identical replay metrics for the two.
The registry's own falsifier for C3 is that the two agree within 10 %. What the
explicit model does earn is that its miss branch takes no update at all, which
removes the need for a miss endpoint and so dissolves the unreconciled 40 px versus
120 px `r_miss` constant. That is one paragraph of the paper.

## 2. Estimand and arms

`p_use,c(x, y)` is the probability that camera `c` returns a usable detection for a
robot at `(x, y)`, marginal over heading. `det_hit` in the frozen spawn-grid capture
is its label, not evaluation truth about the world.

| Arm | Operational inputs | Needs a surveyed 3-D model |
|---|---|---|
| Constant (train prevalence) | detector outcomes | no |
| Distance to camera only | calibration | no |
| FOV / range | calibration, drivable map | no |
| **Monocular depth raycast** | calibration, drivable map, camera RGB | **no** |
| CAD raycast | calibration, surveyed 3-D model | yes |
| GP, no geometric prior mean | calibration, detector outcomes | no |
| GP on a geometric prior mean | calibration, surveyed model, detector outcomes | yes |

The CAD raycast is a **reference, not an upper bound**. It is the most complete
geometric field available and it requires a survey; it is not deployable and, on
this evidence, not best. The word "oracle" is not used for it anywhere.

"Without a surveyed 3-D obstacle model" is the claim. Not "map-free": the monocular
path still needs camera calibration and the 2-D drivable map, the latter to fit the
floor depth anchor.

## 3. E1 — held-out availability calibration  *(run)*

Frozen before running: the six arms, the spatially blocked split, the two-parameter
link, Brier as primary, and the paired sign test against the CAD reference.

- **Split.** Leave-one-spatial-block-out over six contiguous blocks (three x-thirds
  by two y-halves). Verified to reproduce the six `block_x{i}_y{j}` groups in the
  frozen capture. Random k-fold is prohibited: on a dense pose grid it leaves
  held-out points centimetres from training points, so any smoother scores well and
  the comparison measures interpolation instead of spatial transfer.
- **Link.** `P_D = sigmoid(a · logit(v) + b)`, fitted on training folds only, applied
  identically to every arm so no arm is advantaged by its native scale.
- **Leakage control.** The cached GP fields are fitted on every event including the
  held-out block, so the GP arms are refitted per fold, and their link is fitted on
  inner out-of-sample predictions from models that saw neither the inner block nor
  the outer test block.
- **Primary endpoint.** Pooled held-out Brier, 24 camera-fold units per arm.
- **Prespecified reading.** A no-surveyed-model arm matching CAD refutes CAD as an
  upper bound. An arm that wins AUROC while losing Brier is a better ranker and a
  worse probability, and the planner consumes the probability.

**Decided after seeing data, and declared as such (1):** the distance-only arm was
added after the first run. It is a *baseline*, not a change of endpoint or of
analysis, and it makes the comparison harder rather than easier: it is the standard
"reliability decays with range" model and the obvious reviewer question. It has no
fitted parameters of its own, so it cannot be tuned to lose.

**Decided after seeing data, and declared as such (2):** the GP arm was split into
"with" and "without" a geometric prior mean. A single GP arm would have conflated
what the detector outcomes contribute with what the geometry contributes, and the
split is what shows the geometry is doing the work. This is a reporting refinement,
not a change of endpoint; both variants are reported.

## 4. E2 — availability is not accuracy  *(run, compressed)*

Supports the representation, not the navigation claim. **Reported as one figure
panel and two sentences**, not as a section: it argues for estimating two fields
rather than one, which the paper needs stated but does not need defended at length.
Full tables stay in the study output for the thesis.

## 5. E3 — route discrimination and network density  *(run)*

Frozen before running: the four declared tasks, the shared candidate set, decision
regret against a common reference, and the exposure metrics.

- **Endpoint.** Longest continuous unobserved stretch, plus exposure and unobserved
  fraction. **Terminal covariance is explicitly not an endpoint**: the belief
  re-converges after a blackspot, so a route can cross nine metres of unobserved
  aisle and still arrive sharp. Reporting terminal sigma would hide the effect.
- **Shared candidate set.** Every source's field generates routes; every source then
  chooses from the union. A source can only be penalised for choosing badly, never
  for having searched differently.
- **Decision regret.** Each source's chosen route is scored under the common
  reference field, so estimator error is priced where it matters.

**Corrected after a clearance pre-flight, and declared as such:** route search now
runs on the driveable mask **eroded by the 0.25 m keep-in contract**. The first
version searched the raw binary mask, which has no clearance term, so Dijkstra hugged
boundaries and selected routes with 0.062 m and 0.005 m minimum clearance — routes the
local controller rejects at step zero, and which E4 would have seeded its planner
from. This is a correction to an invalid apparatus, not a change of endpoint: the
erosion keeps 64.9 % of the floor, every task stays connected at 0.293-0.458 m
clearance, and route lengths move by under 0.9 m. All §5 numbers were regenerated
under the constraint.

**Added after the four-camera result, and declared as such:** the camera-density
sweep (four, three, two opposing, two same-side, one). The four-camera result was
nearly null, and the reason is mechanical rather than surprising — with noisy-OR
over four cameras the probability that at least one sees the robot is high almost
everywhere, so there is little to avoid. Density is therefore the axis the effect
lives on, and a sparse network is the more realistic deployment. The two-camera
pairs separate placement from count because opposing views and same-side views are
known to behave differently in this repository.

This is a scope condition discovered from data, and the paper must present it as
one: *availability modelling matters when coverage is incomplete, and we measure
where that boundary is*. It must not be presented as if the density sweep had been
planned from the start.

## 6. E0 — online-EFE readiness gate  *(passed 2026-08-18)*

Full criteria in [`e0_online_efe_readiness/gate.yaml`](e0_online_efe_readiness/gate.yaml).
Six criteria on a single availability-blind seed. G6 is the one the 2026-08-15
pilot failed before it even got stuck: the route must enter the region where the
arms differ. A run that arrives while staying in well-observed ground proves the
stack works and still cannot discriminate the models.

Prohibited responses to a gate failure: widening the clearance, relaxing the
terminal goal tolerance, or replaying pre-solved waypoints instead of solving
online. The last is the protocol that was explicitly not chosen for this paper.

## 7. E4 — matched closed-loop campaign  *(stopped after 12/45)*

Full configuration in [`e4_closed_loop/campaign.yaml`](e4_closed_loop/campaign.yaml).

- **Arms.** C1 availability-blind constant covariance; C2 availability-aware folded
  covariance; C3 availability-aware explicit Bernoulli. Identical in every other
  key, so a difference is attributable to the observation model.
- **Headline contrast.** C1 versus C2. C2 versus C3 is reported and expected small.
- **Tasks.** `mc_blind_L` as the discriminating task, chosen because E3 measured it
  to be so; `mc_m2_w2e_traverse` as a **prespecified negative control** where no
  candidate route escapes the blackspot; `route_tall_shadow_west_safe_start` for
  continuity with the pilot. If C2 or C3 beats C1 on the negative control, the
  effect is not the observation model and the campaign has found a confound.
- **Primary endpoint.** Longest continuous interval with no accepted camera
  correction. Secondary: belief RMSE against the ground-truth bridge, fraction of
  the run with truth outside the stated 95 % ellipse, accepted-correction rate,
  path length, time to goal, completion reason.
- **Analysis.** Paired by `(task, seed)`; sign test plus mean paired difference with
  a confidence interval. 3 tasks × 3 arms × 5 seeds = 45 runs, 15 matched pairs per
  contrast.
- **Stopping rule.** Run all 45. No early stop on a favourable interim; no seeds
  added to a task after seeing its result.
- **A null is a result.** If C2 does not beat C1 on the primary endpoint, that is
  reported as a bounded null with the density sweep as the stated scope condition,
  and the paper's claim retreats to the estimator result, which stands on its own.

## 8. What this evidence cannot say

- Nothing about hardware. All evidence is simulation.
- Nothing about detector generality: one frozen four-camera YOLO.
- Nothing about optical or vendor diversity: four identical simulated cameras.
- Nothing about lighting or real-image transfer.
- Dynamic occluders are a separate regime. The monocular fields used for the static
  comparison come from the clear frame; the pallet frame is a different regime and
  is reported separately, not pooled.
- E1–E3 support no navigation or safety claim of any kind. Only E4 may cross that
  line, and only for the arm actually varied.

## 9. Open governance item

`research/06_world_camera_design.md` §2 reserves `warehouse_full_4cam` for
frozen-method evaluation, and E1 compares six estimators there. This needs either a
registered exception or E1's geometric arms re-run in the single-camera world. Not
settled here; see the study README.
