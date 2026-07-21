# DECISION — miss-endpoint constant `r_miss_uv` (plan 07, work item 3)

**Status: PENDING DATA.** No `r_miss_uv` number may be quoted in any R_plan /
R_update until this record is filled from measurement and
`MissEndpointPolicy(reconciled=True, chosen_r_miss_uv=...)` is set.

## What is being decided

The single miss-regime endpoint (per-axis pixel std, px) used by the one
reconciled trust→covariance mapping in
`src/reliability/reliability/covariance_mapping.py`. This is the **only** open
degree of freedom in that mapping.

## The two candidates

| candidate | value | origin | why it was that value |
|---|---|---|---|
| offline | **40 px** | `scripts/geometry_visibility/geometry_visibility.py` (`trust_to_r_plan`) and `single_camera_adapter.SingleCameraAdapterConfig` default | geometry-visibility offline default; chosen for the original single-camera warehouse figures |
| runtime | **120 px** | `unicycle_planner_node.py` planner default | planner conservative default (larger miss covariance ⇒ the filter trusts a missed detection even less) |

`r_visible_uv = 2.5 px` (visible-regime std) is **not** in dispute.

## Why they differ (and why it is ONLY the constant)

The FORMULA is already proven identical across the offline and runtime paths.
`tests/reliability/test_covariance_mapping.py::test_formula_equivalence_adapter_vs_geometry_visibility`
asserts, on a 101-point trust grid for **both** 40 px and 120 px, that
`single_camera_adapter.precision_blend_covariance` and
`geometry_visibility.trust_to_r_plan` return the same variance to ~1e-9. So the
historical divergence documented in `CLAUDE.md` ("`r_miss_uv` = 40 px offline vs
120 px runtime — reconcile before quoting R_plan numbers") is **purely an
endpoint-constant choice**, not a mathematical disagreement. This decision is
therefore a single number, not a re-derivation.

## Deciding measurement owed

**Miss-regime residual tails at the 4-camera operating ranges.** From the
big-warehouse commissioning captures, collect the projected-pixel residuals for
detections in the miss / near-miss regime (low availability, occlusion-boundary
and long-range/oblique cells) at the ranges the 4-cam world actually exercises,
and set `r_miss_uv` from the tail of that residual distribution (e.g. a robust
high-quantile per-axis std), NOT tuned to make navigation win (plan 07 §
"R_bad must be estimated/justified"). Pre-register the exact estimator (quantile,
robustification, per-axis vs pooled) here before reading the number.

## Blocking rule

`MissEndpointPolicy.require_reconciled()` raises `ContractValidationError` until
this record is filled and the policy is constructed with `reconciled=True` and
`chosen_r_miss_uv` set. Every entry point of the mapping calls
`require_reconciled()` before producing any covariance, so an unreconciled policy
cannot silently bless 40 or 120. The error names this file and restates the owed
empirical basis.

## To close this decision (checklist)

- [ ] Fix the estimator for the residual-tail quantile (pre-register above).
- [ ] Run it on the 4-cam commissioning residuals; record the value + provenance here.
- [ ] Set `chosen_r_miss_uv`, `reconciled=True`, `provenance=...` on the shared policy.
- [ ] Grep-kill the losing constant (40 or 120) from offline tooling and planner default so one config remains (plan 07 work item 2/3).
- [ ] Flip status to DECIDED and freeze (plan 07 gate: Fig 05B/05C + property tests).
