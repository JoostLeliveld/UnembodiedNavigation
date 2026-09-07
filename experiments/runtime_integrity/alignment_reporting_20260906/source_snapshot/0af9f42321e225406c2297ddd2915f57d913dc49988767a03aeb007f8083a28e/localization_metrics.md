# Localization metrics contract

Current as of 2026-08-29. This file defines which quantities may be compared and which
runtime evidence makes a drive scoreable.

## Evidence boundary

- Studies before 2026-08-25 are superseded.
- Fusion drives with `logging_schema_version < 4` are diagnostic only.
- There is currently no frozen paper-facing fusion selection. Scoring must stop until
  `logs/studies/fusion_on_fixed_routes/frozen_runs.json` explicitly names a complete,
  provenance-homogeneous schema-4 set.
- A run directory, `RESULTS.md`, modification time, or `latest` name is never evidence
  selection.

## Three quantities that must stay separate

1. A camera reading is scored against ground truth at `obs_stamp`.
2. A fused correction is scored against ground truth at `fused_stamp`.
3. A planner belief is scored against ground truth at `planner_belief_stamp`.

State the layer, statistic, reference, run set, seed count and sample count. Never compare a
camera-reading RMSE with a belief median or call either one simply "localization error".

## Event identity and weighting

- One physical detector invocation is identified by `source_batch_id`.
- Camera capture timestamps inside one batch must span less than 0.20 seconds; the repaired
  campaign enforces 0.05 seconds at both detector and manager boundaries.
- Every published fused correction must have exactly one row in
  `correction_assimilations.csv` with the same `source_batch_id`.
- Only `accepted`, `accepted_bootstrap`, or `reanchored` assimilation rows are belief-update
  events. NIS rejections and dropped corrections are not post-correction beliefs.
- A run is invalid when a correction is **unaccounted for**: a missing assimilation, a
  duplicate one, an extra one, an unclassifiable status, or a refusal with no recorded
  reason. The five valid statuses are `accepted`, `accepted_bootstrap`, `reanchored`,
  `rejected`, `dropped`.
- A **refusal that records its reason does not invalidate the run.** The filter declining a
  measurement it cannot causally bridge is a gate decision, the same class of event as a NIS
  rejection, which has never invalidated a run. The commonest cause is a camera outage longer
  than the replay cap — a property of the warehouse, not a fault in the drive. Measured on
  the 2026-08-29 drives, 90% of runs contain such an outage and the median longest one is
  13–17 s on three of the four routes; failing the run on it would discard those drives *by
  coverage*, keeping only the well-covered route and deleting the comparison.
- **Report `correction_dropped_fraction` and `longest_correction_gap_s` beside the accuracy.**
  How much of a drive the cameras actually carried, and how blind its worst stretch was, are
  results — not preconditions.
- Aggregate within each drive first, then compare the five paired seeds. Logger ticks and
  detector frames are not independent replicates.

## Accuracy and consistency

- Position errors are Euclidean errors in metres internally and reported in centimetres.
- Report median and 95th percentile position error; label means and RMSE explicitly when used.
- Report uncertainty consistency beside accuracy. For planar NEES, the mean target is 2,
  the median target is `2 ln 2`, and 95% ellipse coverage uses chi-square threshold 5.991.
- Heading error is reported separately in degrees or radians and scored at its own stamp.

## Ground-truth firewall

`gt_*` fields are offline references. They may score estimates and physical outcomes but
may not enter the online estimator, admission gate, planner, goal decision or stuck decision.
Wheel odometry is diagnostic input and must never be named ground truth.

The executable implementation of alignment and event selection is
`experiments/fusion_on_fixed_routes/aligned.py`. If this prose and that loader disagree,
stop and repair the contract before reporting a number.
