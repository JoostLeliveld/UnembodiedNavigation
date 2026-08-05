# Operational residuals → `R_cond` — adapted implementation plan

Adaptation of the "Bayesian Reliability Learning and State-Dependent Observation
Covariance" spec (M0–M9) onto **this** repository. The incoming spec is a
greenfield design; roughly two thirds of it is already built, deployed, or has
already been *answered* here — in two cases with a documented null that the spec
would otherwise re-litigate. This document records what was cut and why, then
specifies the work that genuinely remains.

Date: 2026-08-04. Serves `research_story/01_operational_belief_and_logging`
(PARTIAL) and `research_story/04_factorised_observation_model` (PLANNED,
`implemented_now: []`).

---

## 0. The one-sentence delta

> The repo's only measurement residual is `eval_res_x/y = pred_world − ground_truth`,
> which is EVALUATION-ONLY and firewalled (`^eval_` in
> `src/reliability/config/leakage_firewall.yaml`). It can therefore never train a
> deployable `R_cond`. Referencing the residual to a **smoothed operational belief**
> instead of ground truth — `r_t = z_t − h(μ_t^s)` — makes `R_cond` operationally
> learnable for the first time, which is exactly the ICRA-2027 open blocker.

Everything below exists to serve that one sentence.

---

## 1. Structural adaptation (spec §4 is rejected)

The spec proposes a new top-level `bayesian_camera_model/` tree. `CLAUDE.md`
forbids it ("do not invent new top-level directories"; `modules/` are landing
pages that never own runtime code or data). Mapping instead:

| spec location | this repo |
|---|---|
| `src/ekf.py`, `src/rts_smoother.py`, `src/motion_model.py` | `src/state/state/core/trajectory_smoother.py` (state estimation package) |
| `src/conditional_noise.py` | `src/reliability/reliability/conditional_covariance.py` — **already exists**; extend |
| `src/detection_reliability.py` | `src/reliability/reliability/observation_gp.py` + `observation_baselines.py` — **already exist** |
| `src/r_plan_mapping.py`, `src/planner_adapter.py` | `src/reliability/reliability/covariance_mapping.py` — **already THE single source of truth, frozen** |
| `src/camera_model.py` | `unav_common.camera_model.ObliqueCameraModel` + `reliability.projection` |
| `src/psd_utils.py` | `reliability.fusion._validate_spd` / `planning.core.belief_correction.project_to_psd` |
| `scripts/0*_*.py` | `scripts/reliability/` |
| `tests/` | `tests/state/`, `tests/reliability/` |
| `configs/` | `config/` | 
| `schemas/` | `schemas/` |
| `artifacts/` | `logs/studies/operational_residual_rcond/<expN>/` |
| the study itself | `experiments/operational_residual_rcond/` |

---

## 2. What is cut, and the evidence for cutting it

### M0 (log schema + frame contract) — DONE, reduce to a timing audit

Built at Phase P1/P2 of the observability workstream:
`reliability/observation_opportunity.py` (`ObservationOpportunity` +
`FailureReason` enum, one record per *opportunity* including misses — the spec's
central §5 requirement), `observation_gates.py` (frozen ordered gate),
`schemas/observation_opportunity.schema.json`,
`docs/usable_observation/{audit,data_contract}.md`. The spec's
`test_no_ground_truth_fields` already exists as
`contracts.EVALUATION_ONLY_FIELD_NAMES` + `tests/reliability/test_leakage_firewall.py`,
and it is *stronger* than the spec's version (the spec's forbidden-set misses
`true_x/true_y`, which is what `perception.csv` actually calls GT).

**Retained from M0:** Plot M0.1 (sensor timing). Not as a gate — as a diagnosis
of the recorded ICRA blocker.

**Already measured, and it corrects the recorded blocker.** The blocker on file
is *"the detection↔odometry join yields no in-window pairs"*, with a spawn-grid
re-capture named as the unblocking step. Measured on the raw operational
recordings, camera `diag_stamp` → nearest `experiment.csv` odometry stamp:

| capture | odom rate | cam | n | detections | median gap | max gap |
|---|---|---|---|---|---|---|
| `gt_validation_smoke_20260716` | 50 Hz | B | 674 | 69 | 10 ms | 190 ms |
| | | C | 673 | 320 | 10 ms | 10 ms |
| | | D | 673 | 348 | 10 ms | 390 ms |
| `gt_validation_smoke2_20260716` | 50 Hz | A | 181 | 99 | 0 ms | 280 ms |
| | | B | 300 | 211 | 0 ms | 80 ms |
| | | C | 300 | 65 | 0 ms | 80 ms |
| | | D | 300 | 84 | 0 ms | 84 ms |

The join is **fine** — median offset is one odometry tick. The 0-events failure
is a gate inside the commissioning event builder, not a data or clock problem.
**Consequence: the spawn-grid re-capture is not required to unblock `R_cond`.**
This plan proceeds on already-recorded real data. (A re-capture remains desirable
for *coverage* — 1196 usable detections total is thin for a spatial field — but
it is no longer a precondition.)

### M1 (fixed-Q EKF) — DEPLOYED; collapses into M2

The online filter is live, its gate chain is shared code
(`planning/core/belief_correction.py`, 6365 locked corrections replay with zero
verdict changes), and the spec's entire M1 plot set is **already logged
per-step**: `pixel_corr_innov_u/v` (M1.3), `pixel_corr_nis` +
`pixel_corr_nis_threshold` (M1.2, χ²₂,₀.₉₉ = 9.21), `pixel_corr_accepted`,
`pixel_corr_reject_reason_code`, `planner_cov_*` (M1.1/M1.4). Test M1.6
(rejected detection must not move the belief) is already
`tests/planning/test_belief_correction.py`.

What M1 is *actually* needed for is the smoother's forward pass over an offline
log. So M1 survives only as "the forward pass of M2", not as a second filter.

**Cut outright:** the spec's suggestion to switch the state model. `CLAUDE.md`
and the existing formulation note (`belief_filter_note.tex`) fix this as an
odometry-driven filter; the spec itself says "do not redesign the state model
solely to support the new reliability module". Agreed — we don't.

### M4 (spatial detection reliability) — DONE, **with a null. Do not re-run it.**

Phases P3/P4 built the full baseline ladder (B0 constant, B1 distance-logistic,
B2 FOV/range-logistic, B3 Beta-smoothed grid) and the GP
(`observation_gp.py`, reusing canonical `fit_belief_aware_gp.py`), scored
leave-one-route-out. Outcome, on the held-out spatially-novel route
(`p_det` Brier): B0 0.135 · B3 grid 0.133 (memorises) · **GP best 0.058** ·
B2 0.056 · **B1 0.055**.

Per the pre-registered Gate-4 rule (*must beat the simplest competitive
baseline*), the **simpler model was selected: `p_use` source = B2 FOV/range**
(Brier tied with B1, best ECE 0.013, and transfers to any camera from
calibration alone). The GP is retained as a diagnostic only.

The spec's M4 would rebuild this and, worse, its Gate M4 ("improves over the
*global-rate* baseline") is the weak gate this repo already rejected — the GP
clears the global baseline easily and still loses to a two-parameter logistic.
**Re-running M4 under the weaker gate would manufacture a positive result.**

**Retained from M4:** only the uncertain-input treatment (spec Levels B/C), which
is genuinely unbuilt. It belongs to `research_story/03_uncertain_input_gp`, which
already has a *richer* ladder than the spec's three levels — U0 constant, U1
point GP, U2 long lengthscale, U3 Gaussian smoothing, U4 covariance-weighted,
U5 expected kernel, U6 GT-position (EVAL-ONLY). Spec Level A = U1, Level B = U4,
Level C ≈ U5. **Use the existing ladder and its naming**, and note the standing
ch.03 finding: the methods *tie at real σ*, so the discriminating regime is an
α-sweep on inflated σ, not the nominal operating point.

### M6 (trust → `R_plan`) — EXISTS, FROZEN, and the spec has it backwards

`reliability/covariance_mapping.py` is declared THE one source of truth and
closes a three-way divergence. It registers two forms:

* `bounded_interpolation` (**default / preferred**):
  `R_update = R_good + (1−τ)^γ (R_bad − R_good)`, where `R_good` **is the plan-06
  conditional covariance** — i.e. it is already wired to consume `R_cond`;
* `precision_blend` (documented ablation): `1/var = τ/r_vis² + (1−τ)/r_miss²`.

The spec labels its `R_plan = R_det / max(ρ_LCB, ρ_min)` the "recommended"
mapping and bounded interpolation the "alternative baseline". That is inverted
here, and the spec's recommended form is **superseded on both ends**:

1. `_blend_observation_covariance_ca` (the precision blend) was measured to
   understate the posterior trace by **37×** at `p_use = 0.5` (24.6 px² reported
   vs 906 px² correct). The fix already exists — `hit_miss_posterior_ca`
   (Joseph form, flag `use_hit_miss_mixture`, default off) — and it is a proper
   hit/miss mixture, not a covariance rescale. Adding a third scalar mapping form
   would move *backwards*.
2. Dividing `R_det` by `ρ` is exactly the "inflate R to stand in for availability"
   move the mixture replaces. It also collides with
   `feedback_no_visibility_term_tuning`: the R_plan/EFE visibility path is
   **frozen method**, not a knob.

**Retained from M6:** nothing new is built. `R_cond` is delivered *into the
existing* `R_good` endpoint, and its own tests (monotonicity, PSD, endpoints)
already exist in `tests/reliability/test_covariance_mapping.py`. One bonus: a
measured `R_cond` plus the mixture needs **no `r_miss` constant**, which
dissolves the `MissEndpointPolicy.require_reconciled()` blocker on that path.

### M8/M9 — out of scope for this study

M8's matched campaign is `research_story/09` + `planner_conditions_v1` (4 matched
artifacts through the identical frozen adapter; closed-loop P7 awaiting a
machine). M9's alternating EM is explicitly gated behind M0–M8 and stays there.
The self-confirming risk M9 warns about is real *and immediate* — see §4.

---

## 3. What genuinely remains (this study)

Four items. Two are missing code; two are missing *ideas* the spec does not have.

### R1 — A reusable filter+smoother library  *(missing code)*

An RTS smoother exists, but only as a one-off study script:
`experiments/optionA_commissioning/exp5_trajectory_smoothing.py::smooth_run`,
"KF + RTS over 2D position, driven by wheel-odometry increments and anchored by
the external camera's absolute BEV fixes", GT-free by construction. It is not
importable, is hard-wired to one campaign layout, one constant camera `R`, and
2-D position only.

Promote it to `src/state/state/core/trajectory_smoother.py` with the spec's M2
test battery. **Deliberately kept 2-D position** (matching exp5 and the
`camera_xy_only` runtime): heading is odometry-driven under the locked campaign
configuration, and the spec's own instruction is not to redesign the state model.
A 3-state variant is a later option, not a precondition.

### R2 — Operational, GT-free residuals  *(missing code — the point of the study)*

`r_t = z_t − h(μ_t^s)`, with `C_t = H_t P_t^s H_tᵀ + R_cond(s_t)` so that the
robot-state contribution is *subtracted* rather than absorbed. The spec is right
about this and it is the one part of §11 with no counterpart in the repo. It also
kills the naive estimator the repo would otherwise reach for: `R = rrᵀ` includes
`H P Hᵀ`, and at these belief sizes that term is not negligible.

### R3 — Leave-one-camera-out identifiability  *(missing idea — spec blind spot)*

The spec computes `r_t` against a smoothed trajectory that is **anchored by the
very camera whose noise it is estimating**. The residual is then driven toward
zero and `R_cond` is biased low. The spec raises self-confirmation only in M9,
for the EM loop, and never for the one-pass estimator — but the one-pass
estimator has the same defect.

This is not hypothetical: exp5 already observed the smoother *"swallows
moving-frame BEV anchor errors"* (its p95 error is worse than the online
belief's, precisely because it absorbed them).

**Fix:** when scoring camera `c`, exclude camera `c`'s measurements from the
smoother, keeping odometry and the other cameras. With 4 cameras plus odometry
this is well posed. `R_cond,c` is then estimated against a trajectory that is
*independent of `c`*. For the single-camera world (`warehouse_aws`) LOO is
degenerate — the camera is the only absolute anchor — so the single-camera
number must be reported as **odometry-only-anchored** and read as a bound, not a
measurement. Report both, and report the gap: the gap *is* the size of the
circularity the spec would have silently absorbed.

### R4 — Oracle-vs-operational validation  *(missing idea — free and decisive)*

The GT-referenced residual already exists (`eval_res_x/y`, eval-only,
`experiments/external_camera_bias_model` has it characterised per camera). So the
operational estimate can be scored against an **oracle `R_cond`** on the same
detections. Nobody has done this. It answers the only question that matters for
deployment — *how much accuracy is lost by giving up ground truth?* — and it
directly instantiates the spec's own optional "Oracle" condition, which the spec
leaves as an afterthought.

Known target to beat/reproduce (`external_camera_bias_model`, post-deployed-
correction, metres): σ_along/σ_cross = A 0.015/0.041 · B 0.013/0.012 ·
C 0.046/0.017 · D 0.030/0.025, and camera C's **+0.078 m uncorrected lateral
bias** — the largest residual systematic in the network. If the operational
estimator cannot see C's lateral bias, it is not useful.

---

## 4. Corrections to the spec's gates

Two gates are wrong for this repo and would misfire.

**Gate M2 is wrong as written.** It requires "the smoothed trajectory is not
worse than filtering on most evaluation routes". exp5 already measured the
answer: mean error is **not** improved (the online belief is already
camera-anchored, so an offline smoother cannot beat its mean) and p95 is
*somewhat worse*. What improved is **calibration: NEES 16.8 → 2.8**, essentially
calibrated. Under the spec's gate this study would stop at M2 having measured a
success.

Restated gate, matching both exp5 and ch.01's own gate ("covariance at least
directionally related to actual error"):

> **Gate R1.** The smoother's *covariance* is better calibrated than the online
> filter's (NEES toward 1, ellipse coverage toward nominal), GT eval-only. Point
> error is reported, not gated. Rationale: the smoothed belief is consumed as a
> *training distribution* — `Σ_{s,t}` is what R2/R3 use — so covariance honesty
> is the property that matters, and point accuracy is not.

**Gate M5 needs an identifiability clause.** Add:

> **Gate R3.** `R_cond,c` estimated with camera `c` held out of the smoother must
> be reported alongside the not-held-out estimate, and the ratio between them
> reported as the circularity factor. A `R_cond` fitted against a trajectory its
> own measurements anchored may not be quoted as a measurement.

The spec's other M5 gates are kept as-is (PSD, held-out NLL vs constant `R`,
Mahalanobis coverage, no bias absorbed into covariance) — they are good, and
`conditional_covariance.py` already implements MNLL / χ² coverage / sharpness,
including the "always report sharpness, because a huge `R` passes coverage
trivially" rule.

---

## 5. Frames and units — the spec is single-path, the repo is dual-path

The spec asserts `units == "pixel^2"` throughout (Test M6.1). The runtime has
**two** measurement paths (`planning/core/belief_correction.py:13-15`):

* **paper-1:** `z = (u,v)` px, `h(x)` nonlinear via the camera model, `R = R_plan` (px²)
* **multicam:** `z = (x,y)` m, `h(x) = [I₂ | 0]`, `R =` fused covariance (m²)

`conditional_covariance.py` already carries `_ALLOWED_FRAMES = ("uv", "xy")` for
exactly this reason, and the frame is recorded on every `CovarianceEstimate`.

**Decision: this study estimates `R_cond` in the `xy` frame (m²)**, because the
consuming path is the multicam ICRA-2027 line, and because the deployed
projection correction and the entire existing residual characterisation are in
metres. Plan 06's `R_xy = J R_uv Jᵀ + J_θ Σ_θ J_θᵀ` conversion (with calibration
uncertainty as an *explicit* second term, never silently folded in) remains the
route to the px² path and is not exercised here. Every artifact records its frame;
no number is quoted without one.

---

## 6. Milestones, revised

| id | milestone | status | gate |
|---|---|---|---|
| R0 | timing + coverage audit on raw operational recordings | **measured, §2** | join tolerance documented; per-camera detection counts published before any fit |
| R1 | `trajectory_smoother.py` + M2 test battery | to build | **Gate R1** (§4): covariance calibration, not point error |
| R2 | operational residual builder, `C_t = H P^s Hᵀ + R_cond` | to build | one record per accepted detection; GT firewall test passes; `rrᵀ` never used as `R` |
| R3 | leave-one-camera-out `R_cond,c` | to build | **Gate R3** (§4): circularity factor reported |
| R4 | operational vs oracle `R_cond` | to build | reproduces the per-camera σ table; sees camera C's lateral bias |
| R5 | deliver `R_cond` into the frozen `R_good` endpoint | wiring only | no new mapping form; existing `test_covariance_mapping.py` unchanged |

R5 is deliberately last and deliberately small. If R3 shows a large circularity
factor or R4 shows the operational estimate is blind to C's bias, **R5 does not
happen** and the honest result is "operational `R_cond` is not yet measurable at
this coverage" — which is a publishable negative and is consistent with the
existing single-camera nulls.

## 7. Reuse map

| need | reused from |
|---|---|
| repo root | `scripts/shared/paths.repo_root` |
| KF+RTS reference behaviour | `experiments/optionA_commissioning/exp5_trajectory_smoothing.smooth_run` |
| per-camera 2×2 + shrinkage + MNLL/χ²/sharpness | `reliability.conditional_covariance` |
| SPD validation, 2×2 algebra | `reliability.fusion._validate_spd`, `_quad_form_inverse_2x2` |
| pixel→world + deployed bias correction | `reliability.projection._project_pixel_to_world` |
| camera models from SDF | `reliability.projection.camera_model_from_world` |
| GT firewall | `reliability.contracts.EVALUATION_ONLY_FIELD_NAMES`, `config/leakage_firewall.yaml` |
| oracle residual target | `experiments/external_camera_bias_model/residual_audit.py` |
| Brier/NLL/ECE/Spearman | `scripts/shared/metrics.py` (never hand-rolled) |
| nearest-stamp truth join (eval only) | `attach_evaluation_truth._nearest` |
