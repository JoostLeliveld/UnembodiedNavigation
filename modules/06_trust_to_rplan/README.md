# 06 — Trust → R_plan interface

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | Trust maps to planner-facing observation covariance through **one** frozen, bounded, monotone, positive-definite interface — not an ad-hoc per-call formula. |
| **Status** | Status report — architecturally CLOSED, empirically PENDING. |
| **Chapter** | [05 — trust to R_plan](../../research_story/05_trust_to_rplan/) (PARTIAL) |

## What it computes
The precision-blend / bounded-interpolation map from a trust scalar to a 2×2
`R_plan`. `covariance_mapping.py` is now the single source of truth: a grid test
proves the offline (`geometry_visibility.trust_to_r_plan`) and adapter formulas
are identical to ~1e-9 — the historical 40-vs-120 px divergence is only the
miss-endpoint constant, not the math.

## Where it lives
- Single source of truth: [`../../src/reliability/reliability/covariance_mapping.py`](../../src/reliability/reliability/covariance_mapping.py) (property tests: monotone / PSD / 2×2 / finite / factor-logged)
- Planning adapter: [`../../src/reliability/reliability/planning_covariance.py`](../../src/reliability/reliability/planning_covariance.py)
- Miss-endpoint decision: [`../../experiments/multicamera_fusion_extension/plans/DECISION_r_miss.md`](../../experiments/multicamera_fusion_extension/plans/DECISION_r_miss.md)

## Blocking
`MissEndpointPolicy.require_reconciled()` blocks any quote of 40 or 120 px until
the residual-tail measurement is made on real commissioning data.
