# Experiment C — localization effect (ch.05)

A well-calibrated reliability map can still have no practical effect. Test the
observation-model interface separately, on **replay of Experiment B's detections**
(identical detections + process model across conditions — kills detector
stochasticity as a confound).

## Conditions
| id | R_update source |
|---|---|
| L0 | constant covariance `R_const` |
| L1 | calibrated confidence only `M(g(c_t))` |
| L2 | point-input GP `M(r_U1(s_t))` |
| L3 | uncertain-input GP `M(r_U5(s_t))` |
| L4 | full stacked trust `M(τ_t)` (GP σ + calibrated confidence + health) |

`M` = `reliability.covariance_mapping` (THE single source of truth; do NOT
reimplement — reconciled to ~1e-9). `g` = `reliability.confidence_calibration`.
`τ` = `reliability.trust_stacker`. All already implemented.

## Frozen-Q discipline
Tune process noise ONCE on a validation set and FREEZE it across L0–L4. Do NOT
re-tune Q per method — that would let a differently-tuned Q hide a poor R
(recurring trap; see the fusion gate).

## Metrics (eval-only GT, via `scripts/shared/metrics.py`)
2D ATE RMSE, median, **p95**, max; mean NIS; empirical NIS coverage vs χ²₂;
NEES (eval-only); fraction of measurements rejected; longest localization
outage; belief-covariance sharpness `log|P|`; predicted 95% ellipse coverage.
Run-level units, paired across L-conditions on identical replays.

## Gate
L3/L4 lower NIS inconsistency AND ATE p95/max relative to L0/L2, without a Q
change. If L3 ≈ L2 the uncertain-input map adds calibration but not localization
value — report that honestly (feeds the ch.03 null discussion).
Outputs → `logs/studies/single_camera_uigp_reliability/expC_localization/RESULTS.md`.
