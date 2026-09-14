# Localization metrics contract

This contract applies to every thesis result. No camera, fusion, belief, or navigation result
is currently selected. A result becomes current only when
`docs/localization_metrics_registry.json` names its exact manifest and the manifest verifies.

## Keep the three layers separate

1. A camera reading is scored against the reference at its capture timestamp.
2. A fused camera correction is scored against the reference at its fusion timestamp.
3. The robot belief is scored against the reference at the belief timestamp.

Never compare statistics across these layers or call them all “localization error.”

For every reported value, state:

- the layer and quantity;
- the statistic and units;
- the reference and timestamp convention;
- the exact manifest-bound run set;
- the number of complete drives and, where relevant, observations;
- the aggregation order.

## Required statistics

- Position error is Euclidean distance, stored in metres and reported in centimetres.
- Name median, mean, RMSE, and 95th percentile explicitly; they are not interchangeable.
- Report uncertainty consistency beside accuracy. For planar NEES, the Gaussian reference
  mean is 2, the median is `2 ln 2`, and the nominal 95% ellipse uses 5.991.
- Report covariance sharpness or ellipse area beside coverage.
- Report heading error separately.
- Report the fraction of camera corrections dropped or rejected and the longest interval
  without an accepted correction.

## Event identity and aggregation

- A physical camera frame may contribute at most once.
- Detector batches, individual camera frames, fused corrections, estimator decisions, and
  beliefs retain their source identities and integer timestamps.
- Every published correction has exactly one terminal estimator classification.
- Valid terminal classifications are `accepted`, `accepted_bootstrap`, `rejected`, and
  `dropped`; any unaccounted, duplicate, or contradictory event invalidates the run.
- A reasoned refusal is an outcome, not automatic run invalidation.
- Frames within one drive are correlated. Compute each drive's statistic first, then compare
  matched drives or seeds.

## Ground-truth firewall

Ground truth may construct offline correction targets and score completed results. It may not
enter the sensor gate, runtime correction, covariance query, camera fusion, estimator,
planner, controller, goal decision, collision decision, or stopping rule.

## Evidence boundary

Do not select data by directory timestamp, `latest`, glob, remembered value, or an arbitrary
`RESULTS.md`. The registry and its exact manifest are the only entry points. If the registry,
manifest, loader, or event ledger disagree, stop instead of reporting a number.

