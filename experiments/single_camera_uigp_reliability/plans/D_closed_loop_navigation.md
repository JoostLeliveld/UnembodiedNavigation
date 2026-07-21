# Experiment D — closed-loop navigation (ch.06)

Links the map-level claim to the thesis result chain. NEW `warehouse_aws` nav
runs. Mirrors the frozen `honest_campaign_v1` structure but must NOT modify it.

## Conditions
| id | reliability model | planning covariance |
|---|---|---|
| P0 | none | constant |
| P1 | point-input GP (U1) | state-dependent |
| P2 | uncertain-input GP (U5) | state-dependent |
| P3 | U5 + conservative GP σ (LCB) | state-dependent |
| P4 | full update w/ confidence | future planning still GP-based (no future confidence) |

Conservative planning head (ch.03/§7.4, §13): `r_LCB(s) = clip(μ_r(s) − κ·σ_r(s), 0, 1)`,
then `R_plan(s) = R_visible + (1−r_LCB)^γ (R_miss − R_visible)` via
`reliability.covariance_mapping`. κ, γ pre-registered.

## Held-constant (only the planner-facing observation model differs)
No-go geometry, task, seed set, planner horizon, process noise Q, and
**visibility reward = 0** are identical across P0–P4. Current confidence is
NEVER copied across the planning horizon (planner uses only `ã·q̃·h̄`).

## r_miss endpoint discipline
`MissEndpointPolicy.require_reconciled()` blocks quoting the 40-vs-120 px miss
constant until the residual-tail is measured on THIS study's real data (see
`../../multicamera_fusion_extension/plans/DECISION_r_miss.md` + CLAUDE.md).
Measure it here from Experiment B/C residuals; do not hardcode either legacy value.

## Metrics (eval-only GT)
Clean-goal rate + Wilson interval; final goal distance; GT geometry breaches;
physics contacts; min clearance; path length; travel time; time above belief-σ
threshold; mean/max `tr(P_xy)`; localization-loss duration. Obstacle cost,
reliability, and GT geometry reported SEPARATELY.

## Gate (pre-registered, §17-style)
P2/P3 improve breach-free clean-goal OR reduce GT breaches vs P0/P1 at
≤ pre-declared path-length/time overhead and no increase in physics contacts.
Route changes must correspond to predicted belief behaviour. Paired run-level
comparison, grouped bootstrap.
Outputs → `logs/studies/single_camera_uigp_reliability/expD_navigation/RESULTS.md`.
