# Claims

`C1`–`C6` are the only claim identifiers in this thesis. This document refines their prose,
gives each one a falsifiable endpoint, names the registered evidence, bounds the scope, and
states the interpretation that is prohibited. It creates no competing claim system, carries
no progress state (`registry.yaml` is the authority for that), and does not restate
`01_questions.md`, which is immutable.

## The four layers

Every claim belongs to exactly one layer. The layers exist so that later work cannot mix a
field with the method that estimates it, or an estimator with a policy that consumes it.

| Layer | Question | What it owns | Claims |
|---|---|---|---|
| **1 Representation** | SQ1 | The *fields*: `p_use`, `R_cond`, persistent bias / correlation floor, epistemic support, freshness and health | C1 |
| **2 Estimation** | SQ2 | The *sources* that produce those fields: constant, distance, FOV/range, depth/raycast, GP, hybrid, gated DL | C2, part of C6 |
| **3 Planning-time use** | SQ3 | The *expected hit/miss belief* the planner propagates from those fields | C3, C4 |
| **4 Camera management** | SQ4 | The *policies* that act on frozen fields: nearest, max `p_use`, achievable precision, hysteresis, conservative fusion | C5, part of C6 |

Boundary rules, all enforceable at review time:

1. A representation field is defined by its estimand, not by the method that produced it.
   Changing the estimator never changes what the field means.
2. A camera-selection or fusion policy is never an estimator arm, and never appears in a
   source comparison.
3. An estimator for a *future candidate pose* may use only information available at that
   pose. Instantaneous detector outcome and confidence, and "was I seen a moment ago", are
   management-layer signals — see the confidence null in the preserved-negatives list.
4. Layer 4 runs on **frozen** layer-1 fields. No estimator is refitted per policy.
5. Nothing measured offline at frame level may be reported as a navigation or safety
   result. That crossing is C4's job alone.

## Claim summary

| ID | Layer | Falsified if… | Evidence stage reached | Prohibited interpretation |
|---|---|---|---|---|
| C1 | 1 | availability-only models match richer representations on preregistered belief-honesty and decision tests | Offline mechanism evidence; the complete five-field contract has not been tested jointly | "the five fields are minimal or sufficient"; "per-camera `R_cond` beats pooled"; any safety reading |
| C2 | 2 | family labels do not predict legal inputs, commissioning requirements or prespecified failure modes | Taxonomy defined; common benchmark blocked pending evidence recovery | "the GP is the method"; any family ranking; detector- or hardware-generality |
| C3 | 3 | `E[P+]` and the folded-`R` shortcuts agree within 10 % over the operating envelope | Analytic, on the real runtime expressions | "route choice changes"; any navigation or safety reading |
| C4 | 3 | a prespecified campaign shows a practically meaningful effect in the opposite direction | **Not reached**; a no-difference result is a bounded null, not proof of no effect | any inference from offline gains; any claim beyond the arm actually varied |
| C5 | 4 | estimation and management cannot be varied independently without changing the estimand or leaking evaluation data | Composition of locked fields; policy comparison not run | "selection beats fusion in the loop"; "policy X is safe" |
| C6 | 2 + 4 | one family and one calibration policy win in every regime | Calibration-lifecycle regime only | regime coverage beyond drift and commissioning sample size; optical or vendor diversity |

---

## C1 — Availability alone is insufficient to characterize observation quality

**Claim.** A single availability field — or any single scalar trust score — cannot represent
what an external-camera network delivers to a filter and a planner. The minimum contract
carries five separately-estimated fields:

| Field | Symbol | Question it answers |
|---|---|---|
| Usable-observation probability | `p_use,c(x, y)` | Will a usable update arrive here? |
| Conditional localization covariance | `R_cond,c(x, y)` | How sharp is it, given one arrives? |
| Persistent bias / correlation floor | `b_c`, `floor_c` | How much of the error repeats and never averages down? |
| Epistemic support | detections per cell, in/out of fitted range | Is the field measured here, or extrapolated? |
| Freshness and health | correction age, change statistic | Is the commissioned field still valid? |

The historical precision blend remains only as a named legacy baseline.

**Falsifiable endpoint.** On preregistered held-out captures and decision maps, an
availability-only representation matches richer alternatives on belief honesty, sharpness
and camera/route decisions. Passing those tests would refute the claimed need for information
beyond availability. The current programme has **not** compared a joint five-field contract
against all reductions, so C1 makes no minimality or sufficiency claim.

**Evidence.** Five independent failures of the collapsed representation:

- *Availability is not accuracy.* `EXP-PRECISION`: on 15.7 % of the reachable floor the
  most-available camera is not the most informative one; camera C's territory falls from
  25 % (coverage) to 14.8 % (precision), and following coverage costs a median 3.6 cm where
  the two disagree.
- *A per-frame noise model cannot represent a lean.* `EXP-BELIEF`: the conventional filter
  reports median NEES 4.22 with 1.9 cm stated σ against 5.3 cm RMSE and 41.9 % of truth
  outside its stated 95 % ellipse. Sharpening the per-camera covariance makes it **worse**
  (NEES 5.11, 43.6 %); a hard innovation gate rejects 0.2 % of updates and changes nothing
  (4.13).
- *The floor is the missing field.* Adding the per-camera residual floor plus the
  leave-one-out cross-check gives NEES 0.46, 3.3 % outside, stated 5.1 cm against 5.0 cm
  RMSE — honest to 2 % at unchanged accuracy. Ablations locate the mechanism: floor without
  cross-check 6.9 %, one **pooled** floor 19.3 %, baseline 41.9 %. Leave-one-capture-out
  gives 1.1 / 3.5 / 11.4 % against a 36–46 % baseline, so it is not in-sample.
- *Persistent structure is what bounds the covariance.* `EXP-RCOND`: median NEES 8.5–10.8 at detection
  instants under the deployed calibration, worse at detections than over the whole track —
  an update contracts `P` toward a measurement that is 3–8 cm systematically off. A
  historical fitted correction moves one held-out capture 8.51 → 1.06, but E6 shows that its
  camera-bias interpretation is confounded with robot silhouette and route yaw.
- *Geometry is a term, not the explanation.* `EXP-PROJ-AMP`: the pixel-to-ground Jacobian
  spans 4.1× across one camera's footprint, so any `R_cond(x)` that ignores it is credited
  with structure it did not find — yet a one-parameter geometric variance model only wins on
  half the cameras, and every variance model on this data is limited by bias transfer, not
  by its variance form. `EXP-PIXEL-GROUND` closes the same loop from the pixel side: the
  covariance needs two terms (pixel and yaw-marginal body offset), and adding the second
  takes NEES 45.4 → 2.83, uniform across camera, range and yaw.

**Scope.** Gazebo only; simulated detector imagery; one robot, no association ambiguity;
2-D position with odometry-backed heading; three captures (1424–1426 detections, 125–530 per
camera) for the residual/belief evidence and 1849 commanded-pose samples for the pixel-ground
evidence; four nominally identical cameras. Camera A is the weakest case throughout (125
samples, 91.7 % of its footprint outside the fitted calibration range).

**Non-claim.** C1 does **not** claim the five fields are sufficient — only that one is not
enough. It does **not** claim per-camera `R_cond` beats a pooled constant: it does not
(`EXP-RCOND`, MNLL −0.397 against −1.618; a marginal +0.04 nats after the bias fix is a tie,
not a win). It does **not** claim the floor is a precisely-known per-camera constant: across
captures the bias is stable only for the camera whose bias is large (C, 1.4×; A, B, D move
3.7–6.3×), so the floor is a generous bound and must err high. It does **not** claim the
covariance can be sized without robot poses: GT-free sizing from inter-camera disagreement
returned 4× too little variance, and a CAD-only prediction of that shortfall was 171 % off.
No statement here is about closed-loop navigation, collisions, or safety.

---

## C2 — Reliability methods form three operational families

**Claim.** Sources for observation quality fall into three families — geometric, learned,
hybrid — that differ in *legal inputs*, *commissioning requirement* and *failure mode*, not
merely in accuracy. Family membership predicts how a method fails. On the evidence available
today the learned family does not dominate the geometric one.

**Falsifiable endpoint.** Under the frozen source-benchmark splits (`EXP-USABLE`), family
membership must predict legal inputs, commissioning requirements and at least one
prespecified failure mode. If those properties do not cluster by family, the taxonomy is not
useful and C2 is refuted. Whether one family outperforms another is a C6 regime question, not
a refutation of the taxonomy.

**Evidence.**

- *The narrated source ranking is not verified evidence.* The usable-observation README
  reports pooled/held-out Brier values for grid, distance, FOV/range and GP arms, but the five
  named result directories are absent. Those numbers are retained only as recovery targets,
  not as thesis evidence, and FOV/range is not declared the winner. `EXP-USABLE` must recover
  a hash-verifiable package or rebuild the comparison on frozen splits.
- *The geometric family's failure mode is also visible.* Historical fitted corrections do not
  transfer when the systematic is not resolvable against its own scatter: gating at
  `|b_cross|/σ_cross ≳ 1.2` earns +42.4 mm held-out on camera C and costs −26.9 mm on camera
  A, monotone in that ratio across all four cameras. E6 then shows that even the apparently
  resolvable C/D term can disappear after modelling silhouette geometry, so resolvability is
  necessary but not sufficient for causal attribution. A six-parameter world-affine model
  extrapolates to a 3.0 m held-out error on camera A. The shipped pipeline carries **two**
  fitted parameters where twelve were fitted before.
- *Commissioning cost is a family property, and it is measurable.* `EXP-NET-COMMISSION`: a
  large resolvable bias is decided correctly from 20 detections (100 %), while three cameras
  near the decision boundary never reach 60 % with all available data, and the small-sample
  failure direction is false CALIBRATE — the harmful one.

**Scope.** The missing `p_use` package is described as **single-camera** and nearly a pure
`p_det` benchmark, but its quoted component rates are also unverified until recovery. The
common distance/FOV/depth/GP/hybrid/DL benchmark has not been run.

**Non-claim.** Not a claim that GPs are useless: it is a null under this corpus, these
splits and this level of belief uncertainty, and it stays visible as such. Not a family
ranking — the seven-arm comparison is unrun. Not a claim that instantaneous detector
confidence carries quality information for planning: it is refuted below. No claim of
transfer beyond the frozen detector, the simulated image domain, or these worlds.

---

## C3 — Explicit observation-quality modelling changes the predicted belief

**Claim.** Availability is Bernoulli and conditional accuracy is Gaussian; folding the first
into the second misstates the predicted posterior. The canonical expected correction is

```text
E[P+] = p_use · P_hit + (1 - p_use) · P-
```

with `P_hit` in Joseph form and the miss branch taking **no update**. It is compared against
constant covariance, the legacy precision blend, and `R/p`. The discrepancy is largest in
the mid-availability, high-prior-uncertainty regime.

**Falsifiable endpoint.** Refuted if `E[P+]` and the folded-`R` shortcuts agree within 10 %
on posterior trace and log-determinant across the operating envelope. The separate *route*
half is refuted if a prespecified paired route library shows no ranking difference between
the mixture and the shortcuts.

**Evidence.** `EXP-HIT-MISS`: at the runtime endpoints and `p_use = 0.5` the precision blend
reports 24.6 px² against the mixture's 906.2 px² — a 36.8× understatement, peaking exactly at
`p_use = 0.5`, and up to ≈257× over the swept prior grid; both models agree at the endpoints,
as they must. `EXP-PLANNER-BRANCH`: at the representative operating point all three shortcuts
(`R/p`, blend at `r_miss` 40 px, blend at 120 px) agree with each other to <0.2 % and
understate the honest posterior by 89.8 %; worst grid cell −99.75 %; median absolute relative
error 0.63–0.67; only 23.6 % of the factorial grid falls within 10 %. The equivalent single
`R` spans 6.6–10.8× across priors at fixed `p_use` and fixed geometry, so **any interface
caching one `R_plan(s)` per position is structurally unable to be correct**, however well
`p_use(s)` and `R_cond(s)` are measured. Two side effects worth recording: the unreconciled
miss endpoint (40 vs 120 px) is irrelevant to the discrepancy, and on the mixture path no
`r_miss` constant is needed at all. The belief-side half of C3 is carried by `EXP-BELIEF`
(above).

**Scope.** Analytic posterior algebra evaluated through the real runtime expressions — no
captures, no ground truth, no fitted model. `σ_det` stands in for a measured `R_cond(x)`
field. The mixture is behind a flag that defaults off, and the flag-off path is bit-identical
to the deployed precision blend.

**Non-claim.** C3 does **not** claim that route choice changes; the earlier
"mixture changes route choice" headline is retired and route discrimination is an unrun gate.
It does **not** claim any navigation, success-rate, breach or belief-error outcome. It is
**not** evidence that the deployed planner is unsafe — only that its predicted posterior is
optimistic in the regime the contribution is about.

---

## C4 — Navigation consequence

**Claim.** Better observation models improve navigation only on routes where observation
quality changes the achievable belief. This is an **open hypothesis**.

**Decision endpoint.** A prespecified matched-seed campaign with one changed key, scored
on clean-goal completion, ground-truth no-go breaches and physics contacts, NEES/NIS at
detection instants, correction acceptance and age, and path/time. A practically meaningful
effect in the opposite direction would refute C4. No detected difference produces a bounded,
publishable null; it does not prove equivalence unless an equivalence margin and adequate
power are preregistered.

**Evidence.** None yet. `EXP-CL-CAL` is the registered vehicle and is in protocol resolution,
not execution.

**Scope and structural limit.** The currently specified arms vary **calibration only**. A
positive result therefore supports a *calibration-consequence* claim; it cannot establish
closed-loop benefit for the complete correlation-floor plus leave-one-out method, which is
not the variable. E6 shows that the historical C/D correction terms are not identifiable as
camera calibration on the existing logs. No closed-loop effect size or affected-camera
segment is authorized until the grouped WS05 study passes.

**Non-claim.** No offline result — held-out Brier, NEES, posterior trace, achievable
precision — may be reported as a navigation or safety improvement. No formal safety guarantee
is claimed under any outcome.

---

## C5 — Camera management is evaluated separately from estimation

**Claim.** Selection and fusion policies are a distinct decision layer. Reliability fields are
frozen first; then nearest camera, maximum availability, achievable precision, hysteretic
selection and conservative fusion are compared on those fixed fields.

**Falsifiable endpoint.** Refuted if estimation and management cannot be varied independently
without changing the estimand or introducing evaluation-only inputs. Agreement between
policies would make management practically unimportant in the tested regime, but would not
refute the need to evaluate it separately.

**Evidence.** `EXP-PRECISION`: 15.7 % of the reachable floor selects a different camera under
the two criteria; the achievable-precision map is 2.6 cm median against 3.5 cm for
coverage-following, and the gap is in the *typical* case, not the tail (p90 7.8 vs 7.9 cm).
`EXP-BIAS`: uniform fusion over 103 multi-camera clusters scores 0.052 m against 0.039 m for
the best single camera and beats it in only 12.6 % of clusters — you cannot fuse well without
per-camera conditional accuracy. `EXP-BELIEF`: a single pooled floor is 6× worse than a
per-camera one (19.3 % against 3.3 % unearned confidence), and knowing *which* camera is
currently informing the belief is a quantity that exists only in a network.

**Scope.** A steady-state analytic field over a network with 13 % overlap, built by composing
already-measured quantities at the recorded 3 Hz / 0.3 m/s operating point; no trajectory, no
handover transient. Cells below 2 % availability are excluded as unreachable. The policy
comparison itself (`EXP-CAM-MGMT`) has not been run.

**Non-claim.** No claim that selection beats fusion in closed loop, that any policy is safe,
or that conservative fusion is calibrated. No policy result may be obtained by refitting an
estimator for that policy. The disagreement fraction depends on the operating point (faster
detection or slower driving would *increase* it) and is not a constant of the method.

---

## C6 — Different operational regimes favour different methods

**Claim.** Regime, not average accuracy, decides which estimator family and which calibration
policy is correct. Relevant regimes: layout change, stale or rescanned geometry, frozen or
updated learning, calibration drift, dropout, and latency. A commissioning decision is a
decision *with an expiry*, and the statistic that makes it is not the statistic that monitors
it.

**Falsifiable endpoint.** Refuted if one family and one calibration policy win across every
regime in the sensitivity ladder, or if no regime flips the preferred method. The
calibration-lifecycle slice is refuted if a stale correction never becomes harmful within the
swept drift ladder, or if no GT-free statistic detects drift before harm.

**Evidence, calibration-lifecycle slice only.** `EXP-DRIFT`, on the capture held out of the
historical calibration fit: camera C's v3 correction halves its error at rest (0.043 vs 0.088 m) and
**inverts by 0.25° of yaw** (0.100 corrected vs 0.094 raw), so lifecycle risk is proportional
to correction size and falls entirely on the cameras the policy acts on. The commissioning
gate cannot double as the in-service monitor: it fires at rest on the cameras it correctly
left raw (ratios 10.2 and 5.0 against a 1.2 threshold) and it can be *masked* — camera B's
absolute ratio falls 5.02 → 0.31 under real drift as the induced bias cancels its resident
one, so absolute detection is non-monotone, not merely late. The **change** form of the same
residual is monotone for all four cameras and both fault types and detects at 0.1° yaw or
0.025–0.05 m translation, one rung before harm. `EXP-NET-COMMISSION` supplies the
sample-size regime for the historical statistic. It does not reopen the calibration-benefit
claim: a raw camera cannot be harmed by a stale correction, but an apparently resolvable term
must first survive the RQ15 identifiability gate.

**Scope.** Drift is injected geometrically at a single magnitude per capture (a step, not a
ramp), single-factor in yaw or translation, on one capture of 230 detections; detection
*latency in time* is not measured, and compound drift is not swept. The 1.2 threshold is
inherited from the commissioning gate, not re-tuned, and no false-alarm rate is quoted. The
change monitor needs a stored per-camera commissioning baseline and therefore inherits the
commissioning-duration requirement rather than escaping it. All other regimes — layout
change, stale geometry, dropout, latency — are planned, not evidenced.

**Non-claim.** No claim that this generalizes beyond the frozen detector (a standing
limitation). No claim of optical, resolution, frame-rate, vendor or hardware diversity. No
claim that these worlds are representative of warehouses in general. A change alarm means
"re-commission", not "degraded" — one camera's error *improved* under injected translation
drift, and the detector correctly flagged it anyway.

---

## Preserved null and negative results

These stay visible in every write-up. Each one closes a door a reviewer will otherwise ask
about, and several were expensive.

| Null | Where | Why it must survive |
|---|---|---|
| The usable-observation README narrates a GP/FOV null, but its evidence directories are absent | `EXP-USABLE` recovery gate | No ranking claim until recovery or deterministic rebuild |
| Per-camera `R_cond` does not beat one pooled constant | `EXP-RCOND` | Blocks the obvious "just measure per-camera noise" story |
| Sharper per-camera covariance makes belief honesty *worse* | `EXP-BELIEF` A2 | The missing field is bias, not noise resolution |
| A hard innovation gate rejects 0.2 % and restores nothing | `EXP-BELIEF` A1 | Classical robust filtering does not address a lean |
| Health-checking each camera against the full belief is backwards (NEES 23.2) | `EXP-BELIEF` step 3 | The biased camera captures the belief and the honest ones look faulty |
| Uniform fusion loses to the best single camera (12.6 % of clusters) | `EXP-BIAS` | No "more cameras is better" claim |
| YOLO confidence is *positively* associated with localization error (partial Spearman +0.59 after geometry controls) and adds nothing out-of-route | confidence critique | Confidence must never be used as inverse covariance |
| The deployed along-bearing correction is not demonstrably better than a trivial world constant, and no along-bearing correction generalizes across captures | `EXP-BIAS` exp1/exp3 | The incumbent pipeline is an arm, not a baseline of record |
| GT-free covariance sizing from inter-camera disagreement is 4× too small; the CAD prediction of that shortfall is 171 % off | `EXP-PIXEL-GROUND` e3/e4 | Sizing needs a pose-bearing commissioning measurement |
| Grid/lookup route-memorization numbers are narrative-only until their package is recovered | `EXP-USABLE` recovery gate | Pooled scores are never the promotion gate |

---

## Paper A — correlated camera error and belief honesty

Scope of record: `papers/correlated_error_icra.md`. Claims used: **C1 in full**, the
belief half of **C3**, **C4 as an explicitly open gate**, and the **calibration-lifecycle
slice of C6**.

| Subquestion | Answered by | Status of the answer |
|---|---|---|
| A-SQ1 What does the deployed projection pipeline actually get wrong? | `EXP-BIAS`, `EXP-PROJ-AMP`, `EXP-PIXEL-GROUND` | Answered: a persistent per-camera cross-bearing lean the deployed model cannot reach, plus a body-offset term seen through unobserved yaw |
| A-SQ2 Why does a conventional filter become confidently wrong on it? | `EXP-BELIEF`, `EXP-RCOND` | Answered: repeated looks from one camera are counted as independent evidence while the error floor does not shrink |
| A-SQ3 What restores honest uncertainty without losing sharpness? | `EXP-BELIEF` A4 with X1/X2 ablations | Answered, with the mechanism attributed: the per-camera floor does most of the work; the leave-one-out check is secondary |
| A-SQ4 Can the decision be made, and kept valid, without operational truth? | `EXP-NET-COMMISSION`, `EXP-DRIFT`, E6/RQ15 | Lifecycle monitoring is supported for a historical correction; whether the correction is an identifiable camera term is open |
| A-SQ5 Does any of it change navigation? | `EXP-CL-CAL` | **Open.** A documented null is an acceptable answer and bounds the paper to a belief-calibration result |

**Independence contract.** Paper A consumes no field produced by the source benchmark. Its
one availability input is the frozen four-camera coverage artifact inside `EXP-PRECISION`,
which is a composition of already-locked quantities, not a benchmark output. Paper A is
therefore writable to completion while `EXP-USABLE` is untouched, and no gate in Chapter B
blocks it.

**Explicitly outside Paper A:** C2 in full, C5 in full, and the regime slice of C6.

## Chapter B — reliability-source comparison

Scope of record: `papers/reliability_source_comparison.md`. Opens only after Paper A's
package is closed. Claims used: **C2 in full**, **C6 in full**, the **route half of C3**, and
**C5**.

| Subquestion | Gate | Notes |
|---|---|---|
| B-SQ1 Which source predicts held-out usable observations most accurately and honestly? | Held-out calibration under frozen splits | Must beat or explain FOV/range on held-out *routes*, not pooled |
| B-SQ2 Which failures are explained by occlusion, unsupported space, stale geometry or layout shift? | Failure audit | No arm is promoted without a documented failure case and fallback |
| B-SQ3 Which sources change expected belief and discriminate meaningful route alternatives? | Offline route discrimination | This is where the route half of C3 is decided |
| B-SQ4 What commissioning, runtime, transfer and update cost purchases those gains? | Deployment matrix | Cost is an outcome, not a footnote |
| B-SQ5 With fields frozen, which management policy uses them without becoming overconfident? | `EXP-CAM-MGMT` | Runs last, on frozen fields, and never refits an estimator |

Chapter B inherits the C1 representation contract as fixed and may not reopen Paper A's
locked evidence. It also inherits the corpus limitation: the existing usable-observation
study is single-camera and nearly a pure `p_det` benchmark, so a multi-camera `p_qual` result
requires new evidence, not reinterpretation.

## Camera-diversity scope, applied to every claim above

The four cameras are **optically nominally identical** — same intrinsics, height and pitch,
differing in yaw and position (the pixel-to-ground Jacobians agree to 0.5 % across all four,
which is the check). Supported diversity claims are limited to:

- **viewpoint geometry** — position, range distribution, footprint, amplification;
- **occlusion exposure** — what each camera's line of sight is blocked by;
- **overlap and handover role** — the network has 13 % overlap, so which cameras can
  cross-check which is a structural property;
- **measured installed-view residual structure** — historical per-camera fits differ, but
  E6 shows that camera-specific attribution is unresolved against route, region, yaw and
  robot silhouette (RQ15).

Not supported by any evidence in this thesis: optical archetype, resolution, frame-rate,
lens or vendor diversity; hardware transfer; generalization to warehouses beyond the
measured world properties. Four cameras is four points — every per-camera rule stated here
(the 1.2 resolvability gate above all) is a threshold with a mechanism, offered so the next
camera can falsify it, not a fitted law.
