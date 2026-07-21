# Plan 10 — Planning adapter: R_plan and expected-information updates (§13, RQ5/H5)

## Purpose
Convert PREDICTED reliability into future observation covariance for the
planner. Hard rule (already thesis architecture): current confidence is never
copied across the horizon — planning consumes only spatially predictable
quantities plus a slow health average:

```
r_plan_i(s) = ã_i(s) · q̃_i(s) · h̄_i          (LCB heads from plan 03, health EWMA from plan 08)
R_plan_i(s) = R_visible_i + (1 − r_plan_i(s))^γ (R_miss_i − R_visible_i)
```
using the SAME frozen mapping constants as plan 07 (one implementation rule).

## Two candidate belief-propagation forms (compare, keep better-calibrated)
1. Current interpolation: per-camera R_plan into the existing planner path.
2. Expected-information form (multi-camera principled):
   `Λ⁺ = (P⁻)⁻¹ + Σ_i r_plan_i(s) Hᵢᵀ (R_cond_i)⁻¹ Hᵢ ; P⁺ = (Λ⁺)⁻¹`
   — an update that arrives with probability r_plan.

Decision metric: calibration of predicted-vs-realized tr(P) on held-out
rollouts (not navigation wins).

## What exists / reuse
- `unicycle_planner_node.py` belief-propagation path + `PlanningCovariance`
  contract; `providers` multi-camera dispatch; day-zero prior artifact for the
  pre-data variant; `belief` cost terms (log-det / trace) already in the EFE
  planner — no new objective terms.

## New code
- `src/reliability/reliability/planning_covariance.py` + tests:
  `r_plan`, `R_plan`, `expected_information_update(cov_prior, cameras)` —
  pure functions, planner node imports them.
- Batch planner-query adapter: candidate states → per-state, per-camera
  {reliability, R_plan} (the §19 planner query), served by `providers`.

## Gates (§21 planning gate)
- Obstacle/no-go geometry identical across all E8 conditions; direct
  visibility reward stays ZERO in the principal comparison (locked-campaign
  precedent); route changes must correspond to predicted belief behaviour.
- No future-confidence leak: test asserts the planner-facing API has no
  confidence/health-snapshot-per-future-state argument beyond h̄.
- Unit tests: information form vs sequential expected updates on fixtures;
  r=0 ⇒ pure prediction; r=1 ⇒ full update; monotone in r; PSD.
