# Observation-quality method taxonomy

This document classifies **four different kinds of thing** and keeps them apart: the fields
that represent observation quality (§1), the source arms that estimate those fields (§2), the
planner expression that consumes them (§3), and the camera-management policies that act on
them once frozen (§4). §5 is the gate ladder every arm walks in order.

Two rules make the separation enforceable:

- **Legality.** An estimator for a *future candidate pose* may use only information available
  at that pose. Instantaneous detector outcome and confidence, current detection validity and
  "was I seen a moment ago" are **management** signals (§4), never planning-time predictors,
  unless a causal forecast of them is explicitly defined and itself gated.
- **Frozen fields.** §4 policies run on fields fixed by §2. No estimator is refitted per
  policy, and no policy ever appears as an arm in a source comparison.

---

## 1. Representation layer — the fields (SQ1)

Five separately-estimated fields. A source arm is only ever an estimator *of one of these*;
changing the arm never changes what the field means.

| ID | Field | Estimand | Consumed by | Current evidence and open edge |
|---|---|---|---|---|
| R1 | `p_use,c(x, y)` | P(a usable localization update arrives from camera `c` at this pose) — factored `p_det · p_qual` | Planner mixture (§3), max-`p_use` and precision policies (§4) | The README narrates a single-camera, mostly-`p_det` result, but its evidence package is absent. Multi-camera `p_use` is unmeasured. |
| R2 | `R_cond,c(x, y)` | Covariance of the ground-point measurement **given** an update arrives | Planner `P_hit`, achievable precision, conservative fusion | Two terms are required, not one: pixel Jacobian **and** a yaw-marginal body-offset term (NEES 45.4 → 2.83 with both, uniform 2.54–3.18 across camera/range/yaw). Per-camera `R_cond` does **not** beat one pooled constant. Sizing needs a pose-bearing commissioning measurement — a GT-free estimate from inter-camera disagreement was 4× too small. |
| R3 | `b_c`, `floor_c` | The persistent, temporally-correlated part of the error: an installed-view residual or bias, and the covariance floor the belief may not claim to beat | Belief correction (correlation floor), achievable precision, calibration gate | Floor takes unearned confidence 41.9 % → 3.3 %; a pooled floor is 6× worse (19.3 %). Sensitivity is one-sided: too small fails honesty, too large costs sharpness. E6 shows that an apparent per-camera mean may actually be robot-silhouette × route-yaw structure, so attribution requires RQ15. |
| R4 | Epistemic support | Was this field measured here, or extrapolated? Detections per cell; inside/outside the fitted range | Gate decisions, fallback selection, planner conservatism | Not optional: 91.7 % of camera A's footprint lies outside its fitted calibration range. The GP's expected failure is confident extrapolation, but the common benchmark has not yet verified it. |
| R5 | Freshness and health | Correction age; the **change** statistic against the stored commissioning baseline | In-service monitor, camera management, safe-stop | The commissioning gate cannot serve as the monitor: absolute `\|b_cross\|/σ_cross` fires at rest on correctly-raw cameras (10.2, 5.0 against 1.2) and can be *masked* by cancellation (5.02 → 0.31 under real drift). The change form is monotone and detects at 0.1° yaw / 0.025–0.05 m — before harm at 0.25°. |

**Legacy.** The historical precision blend (availability folded into `R`) is retained only as
a named baseline arm. It is not a representation.

---

## 2. Estimation layer — the source arms (SQ2)

Seven arms in three families. Same estimand for every arm in a comparison (R1 unless stated),
same detector hash and threshold, same splits, paired seeds.

### 2a. Information contract

| ID | Family / arm | Estimand | Legal operational inputs | Cold start | Commissioning | Transfer assumption | Runtime |
|---|---|---|---|---|---|---|---|
| E0 | Geometric: constant (null) | R1 | Camera identity at most | Yes | None | None — the field is assumed spatially flat | Free |
| E1 | Geometric: distance | R1 | Candidate pose, camera position | Yes | Calibration only | Quality is a monotone function of range alone | Free |
| E2 | Geometric: FOV/range | R1, and the `J J^T` shape of R2 | Candidate pose, calibrated extrinsics and optics | Yes | Calibration only | Frustum and calibration remain valid; occluders are absent or elsewhere | Closed form |
| E3 | Geometric: depth/raycast | R1 | Candidate pose, camera model, depth or occupancy **with provenance** (static-mapped vs live-sensed) | Yes if a map exists | Map build or depth-sensor setup | The mapped scene still matches the operational scene | Per-candidate raycast |
| E4 | Learned: GP | R1 | Candidate pose plus the fitted operational field | No | A labelled commissioning route | Test poses lie in the support of the commissioning data | Kernel evaluation per candidate |
| E5 | Hybrid | R1 (+ R3 update) | Geometric prior plus operational post-run updates | Partial | Calibration **plus** samples | Evidence can override a wrong prior where support exists | Prior plus stored update |
| E6 | Learned: DL challenger | R1 | Only features computable at future candidate poses | No | Dataset plus calibration | Both the geometry and the image domain transfer | Network forward pass |

### 2b. Failure, fallback, and what the evidence currently says

| ID | Characteristic failure | Fallback on failure | Current evidence |
|---|---|---|---|
| E0 | Ignores all spatial structure; cannot express that being seen somewhere is worse than elsewhere | — (it *is* the conservative floor) | Baseline of record. Beaten by any spatial model on ranking (AUROC 0.50 vs ≈0.78), yet **beats the GP on logloss** under region-disjoint extrapolation. |
| E1 | Range is a proxy; misses occlusion and viewing geometry entirely | E0 constant | Mandatory baseline; common benchmark evidence missing. |
| E2 | Predicts "visible" straight through occluders | E0, or mark unavailable | Mandatory baseline; not yet selected as a winner. |
| E3 | Stale, missing, or misregistered geometry; silently confident through unknown cells | Fall back to E2, or an explicit conservative-unknown value (never "visible") | Infrastructure exists; **benchmark not run**. Provenance and the missing-depth fallback are pre-registered assumptions, not results. |
| E4 | Unsupported regions and distribution shift; memorizes routes; confident extrapolation | Revert to the geometric prior with the R4 flag raised | Challenger. The narrated held-out null is unverified because its package is absent; retain any reconstructed loss/tie as a permanent null. |
| E5 | A wrong prior that the evidence never overrides; update rate too low to matter | Revert to prior, R4 flag raised | **Planned.** Not yet built or scored. |
| E6 | OOD geometry and miscalibrated probability, with no support signal to warn you | Geometric prior | **Gated; not admitted.** Must pass feature legality, probability calibration, held-out prediction and route discrimination before it consumes any campaign time. |

### 2c. Rules that bind every arm

- **Parameter budget is part of the result.** Richer bias models are not safer ones at 8 %
  spatial coverage: a six-parameter world-affine fit extrapolates to a 3.0 m held-out error on
  the weakest camera, and the shipped projection pipeline carries **two** fitted parameters
  where twelve were fitted before.
- **Resolvability is necessary, not sufficient.** In the historical split,
  `|b_c|/σ_c ≳ 1.2` separated a helpful fitted term from a harmful marginal one. E6 shows
  that the C/D signal can nevertheless be explained by silhouette geometry. The default is
  **no correction** until a term is both statistically resolvable and identifiable across
  held-out yaw and region groups.
- **Small samples fail toward false CALIBRATE.** A large resolvable bias is decided from 20
  detections; cameras near the boundary never reach 60 % confidence with all available data,
  and `|mean|/σ` on few autocorrelated samples exceeds the gate by chance. Sample-efficiency
  curves are a required deliverable, not a footnote.
- **Split by route or region, never pooled.** The missing package narrates route memorization
  by a grid model. Regardless of recovery, held-out-route or leave-region-out is the gate;
  pooled score is not evidence of transfer.
- **Fitting at one yaw and testing at another is not a held-out test.** Two "independent"
  captures that are two straight lines at two fixed yaws cannot validate a spatial bias model
  — that is a design fault, not a noise problem.
- **Instantaneous detector confidence is not a quality signal.** It is *positively* associated
  with localization error (partial Spearman +0.59 after geometry controls, U-shaped) and adds
  nothing beyond geometry out-of-route. It must never be used as an inverse covariance. It
  remains a legal §4 management input.

---

## 3. Planning-time use (SQ3)

The planner propagates the **expected hit/miss belief**, not a folded covariance:

```text
P_hit = (I - KH) P- (I - KH)^T + K R_cond K^T      (Joseph form, PSD by construction)
P_miss = P-                                        (a miss is no update, not a wide update)
E[P+] = p_use · P_hit + (1 - p_use) · P_miss
```

R1 enters as `p_use`, R2 as `R_cond`, R3 as the floor the posterior may not claim to beat, R4
as the reason to stay conservative where the field is unsupported.

**Forbidden shortcuts, with their measured cost.**

| Shortcut | Why it is wrong | Measured error |
|---|---|---|
| Precision blend `1/var = p/r_vis² + (1-p)/r_miss²` | Averages precisions, so the sharp branch dominates: a coin-flip detection is treated as 1.4× worse than a certain one | 36.8× understatement at `p_use = 0.5`; ≈257× at the worst swept prior |
| `R/p` scaling | Same failure by a different route; a miss is not a Gaussian update with a large `R` | −89.8 % posterior trace at the representative operating point; −99.75 % worst grid cell |
| One cached `R_plan(s)` per position | The correct effective covariance is `R_eff(s, P⁻, H)` — it depends on the prior, not only the pose | Equivalent σ spans 6.6–10.8× across priors at fixed `p_use` and fixed geometry |

Two consequences worth carrying: the shortcuts are accurate exactly when the camera barely
matters (error <10 % once `σ_det/σ_prior ≳ 1`) and worst when it matters most; and on the
mixture path the miss branch takes no update, so **no `r_miss` constant is required at all**
— which dissolves the unreconciled 40-vs-120 px endpoint rather than measuring it.

**What the planning layer may not conclude.** Posterior algebra is not a route result and a
route result is not a navigation result. Route discrimination is an unrun offline gate;
navigation consequence is C4 and requires the closed loop.

---

## 4. Camera-management layer (SQ4)

Policies, not estimators. Each consumes **frozen** §1 fields and decides which camera(s) the
belief listens to. They are compared against each other with the fields held fixed.

| ID | Policy | Inputs it reads | Decision rule | Evidence today |
|---|---|---|---|---|
| M1 | Nearest camera | Pose, camera positions | Minimum range | Baseline. No spatial quality reasoning at all. |
| M2 | Maximum availability | R1 | `argmax_c p_use,c` | Selects a different camera from M3 on 15.7 % of the reachable floor, and over-trusts the leaning camera on ~10 % of it. |
| M3 | Achievable precision | R1, R2, R3, odometry growth rate | `argmin_c σ_c` where `σ_c² = floor_c² + q_rate/(f · p_use,c)` | Median 2.6 cm against 3.5 cm for M2; median 3.6 cm penalty where the two disagree, max 6.0 cm. Camera territory shifts 25 % → 14.8 % for the leaning camera. |
| M4 | Hysteretic selection | Any of the above plus switch history | Same criterion with a dwell/threshold band | Motivated by handover steps (corrected pairwise bias steps 0.040–0.089 m) but **not yet evaluated**. |
| M5 | Conservative fusion | R2, R3, R5 across cameras | Combine with inflation or per-camera rejection rather than independent-Gaussian product | Uniform fusion loses to the best single camera (0.052 vs 0.039 m; beats it in only 12.6 % of clusters). Independent fusion is prohibited outright — errors are correlated. |

**Legal management-only signals** (illegal at planning time): instantaneous detection
validity and confidence, recent-observation recency, per-camera innovation and its change
statistic, correction age, and node health.

**Structural findings this layer must respect.** Judging a camera against the *full* belief is
backwards — a biased camera captures the belief and then the honest cameras look faulty
(NEES 23.2); the comparison must be against a belief built **without** the camera under test.
That check is only possible because other cameras exist, and it is a real but secondary
contributor (≈2×) next to the per-camera floor. An anchored estimate that lets a camera into
its own reference understated one camera's error by 4.2×.

---

## 5. Gate ladder

Every arm and every policy is evaluated in this order. A method that fails an earlier gate
consumes no campaign time and no simulator hours.

1. **Feature legality** — does every input exist at a future candidate pose?
2. **Held-out prediction and calibration** — held-out *route* or *region*, never pooled.
3. **Failure audit** — at least one documented failure case and a named fallback.
4. **Offline route discrimination** — does it change expected belief and rank routes
   differently on a prespecified library?
5. **Closed-loop navigation** — prespecified matched campaign, or a documented null.
6. **Deployment decision matrix** — commissioning samples and time, runtime, transfer,
   adaptation cost, and the failure/fallback pair.

Camera-management policies (§4) enter at gate 2 with the fields already frozen, and never
re-enter gate 2 as source arms.
