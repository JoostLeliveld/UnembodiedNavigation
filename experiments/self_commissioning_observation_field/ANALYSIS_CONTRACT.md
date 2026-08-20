# Frozen-input exploratory analysis contract

This analysis was introduced after the earlier range-only conditional-R audit
had returned a null result. It is therefore labelled exploratory, not a
retrospective preregistration or a confirmatory holdout.

## Inputs and response

- Availability is the frozen four-camera A3 field: monocular-depth geometry
  prior plus the already learned GP residual.
- Conditional measurement evidence is exactly `PG-IPM-CURRENT` (1,844 detected
  bottom-centre readings). No diagnostic keypoint story is merged into it.
- `clear` means per-camera A3 `p_use >= 0.80`; `marginal` means `< 0.80`.
- Ground truth is available only during offline commissioning/scoring.

## Bayesian candidates

The baseline uses one Normal-inverse-Wishart posterior per camera and
operational visibility stratum. The spatial candidate is a 30-centre,
3 m-length-scale RBF finite-rank GP approximation for both bias and log
conditional variance. It is not the earlier range-squared model.

Each of six outer spatial blocks is tested once. The next block cyclically is
used only to estimate a conservative covariance multiplier; the remaining four
fit the model. The spatial model is selected only when:

- pooled held-out NLL improves by at least 0.01 nat;
- absolute 95% coverage error is no more than 0.015 worse;
- no camera loses more than 0.05 nat NLL; and
- the spatial field is non-degenerate (at least 10% R span or 0.5 px bias span).

A spatial failure selects the per-mode constant Bayesian model. It does not
invalidate learning a constant conditional covariance.

## Planning ablation

All arms use the same frozen E3 solved-route library and a 5% length budget.
The `p_use` arm minimizes the exact expected longest missed-update run. The
complete arm minimizes mean exact belief trace among routes whose availability
risk is within `max(0.25 step, 5%)` of that optimum. This is a constrained rule,
not a tuned weighted sum.

All 16 independent four-camera hit/miss subsets are enumerated at every 0.4 s
step. The complete arm subtracts learned bias and adds posterior bias
uncertainty to conditional R. The `p_use` arm uses one pooled global bias
correction and calibrated pixel covariance. Process and initial sigmas are 2 cm per step and 25 cm,
respectively.

The route result is offline discrimination only. It cannot authorize a
closed-loop claim without a separately frozen execution campaign.
