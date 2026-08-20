# Repository instructions for localization evidence

Before answering any question involving camera accuracy, localization error, belief error,
NEES, coverage, RMSE, bias, or a comparison of runs, read
`docs/localization_metrics.md` and `docs/localization_metrics_registry.json`.

Hard rules:

1. Never say “camera error” or “localization error” without naming the metric object,
   statistic, reference, projection/runtime, dataset/run IDs, experimental unit, sample size,
   and current/historical status.
2. The only current A–D camera-accuracy comparison is `PG-IPM-CURRENT`. The historical
   Camera C 76.9/77/78 mm value is a signed cross-bearing bias from `MC-DRIVE-V2`; it is not
   total error, not current, and not evidence that Camera C hardware is worse.
3. Do not compare camera-measurement error with filter-belief RMSE or navigation-run error.
4. Do not compare mean Euclidean error, RMSE, p95, signed bias, NEES, or ellipse coverage as
   if they were interchangeable.
5. A camera never knows ground truth, its own localization error, NEES, or coverage. State
   which online inputs the method had and which fields the offline evaluator used.
6. Never pool arbitrary run directories. Use a frozen manifest with exact run IDs,
   conditions, seeds, inclusion rules, runtime/projection version, join rule, and experimental
   unit. Fail on missing or extra runs.
7. Use `gt_x/y` as the reference. Legacy `truth_x/y` is wheel odometry and is diagnostic only.
8. Historical results may remain as mechanism evidence, but every quotation must carry the
   retired-pipeline and sampling-confound caveat.

When the canonical registry and another prose document disagree, report the conflict and use
the canonical registry until the underlying analysis is rerun and the registry is updated.
