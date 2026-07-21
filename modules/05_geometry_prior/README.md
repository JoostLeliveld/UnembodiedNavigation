# 05 — Day-zero geometry / FOV prior

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | A geometry- and calibration-derived observability prior predicts where each camera can be believed *before any driving* — a transferable alternative to the learned GP on day zero. |
| **Status** | Partial — module + figures exist; S1 gate PASSED (geometry explains GP Spearman 0.73 / R² 0.51). |
| **Chapter** | [07 — weak priors and geometry](../../research_story/07_weak_priors_and_geometry/) (ACTIVE) |

## What it computes
Raycast visibility fans from each camera mount → a day-zero prior map, plus the
range/FOV term and occlusion prisms. Compared head-to-head against the learned
trust field on an identical colour scale.

## Where it lives
- Offline module: [`../../scripts/geometry_visibility/`](../../scripts/geometry_visibility/) (9/9 tests, `VALIDATION.md`)
- Deployment reasoning: [`../../docs/geometry_visibility_deployment.md`](../../docs/geometry_visibility_deployment.md)
- Outputs: `../../logs/studies/geometry_visibility_prior/`
- 4-cam day-zero artifact: `warehouse_full_4cam_dayzero_v1`

## Demo owed (D1)
Fans sweep and the day-zero map fills camera by camera; day-zero prior vs learned
map side by side; coverage/overlap (99.2% union / 42.2% overlap) drawn on the map.
